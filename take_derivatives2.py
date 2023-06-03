import sys
sys.path.append('..')
import h5py
import numpy as np
from numpy import einsum
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import configparser
from pathlib import Path

import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.losses import MeanSquaredError
mse = MeanSquaredError()
from MD_AE_tools.models import model_evaluation
from tensorflow.keras.optimizers import Adam

import MD_AE_tools.models.models_no_bias as mdl_nobias
import MD_AE_tools.mode_decomposition as md
from project_specific_utils.data_and_train import TrainNN_from_config, train_autoencder



folder = '2mode-regularised-1-764454'
saveas = 'regularisation1.npy'

config = configparser.ConfigParser()
config.read('./_system.ini')
results_dir = config['system_info']['alternate_location']


parent_folder = Path(results_dir,'experiment_standard_nonlinear')
folder_path = Path(parent_folder,folder)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
        # tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]) 
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


filename = Path(folder_path,'training_param.ini')
mdl_config = TrainNN_from_config(filename)

filename = Path(folder_path,'results.h5')
with h5py.File(filename,'r') as hf:
    u_train = np.array(hf.get('u_train'))
    u_all = np.array(hf.get('u_all'))
    y_train = np.array(hf.get('y_train'))
    u_test = np.array(hf.get('u_test'))
    modes_train = np.array(hf.get('modes_train'))
    modes_test = np.array(hf.get('modes_test'))
    latent_train = np.array(hf.get('latent_train'))
    latent_test = np.array(hf.get('latent_test'))

    u_test_mean = np.array(hf.get('u_avg_test'))

print('MSE of traning is: ', mse(u_train,y_train).numpy())

x = einsum('t y z u -> y z t u',np.squeeze(u_all))
X = np.vstack((x[:,:,:,0],x[:,:,:,1]))
pod = md.POD(X)
Q_POD,lam_data = pod.get_modes
Q_mean = pod.Q_mean
X_reconstructed = pod.reconstruct(2,shape=X.shape)
print('MSE reconstructed with 2 modes is: ', mse(X,X_reconstructed).numpy())

mdl = mdl_nobias.Autoencoder(Nx=mdl_config.Nx,Nu=mdl_config.Nu,
                            features_layers=mdl_config.features_layers,
                            latent_dim=mdl_config.latent_dim,
                            filter_window=mdl_config.filter_window,
                            act_fct=mdl_config.act_fct,
                            batch_norm=mdl_config.BATCH_NORM,
                            drop_rate=mdl_config.drop_rate,
                            lmb=mdl_config.lmb,
                            resize_meth=mdl_config.resize_meth)
mdl.compile(optimizer=Adam(learning_rate=mdl_config.learning_rate),loss='mse')
mdl.evaluate(np.squeeze(u_train),np.squeeze(u_train))

# ==============================================
filename = Path(folder_path,'md_ae_model.h5')
mdl.load_weights(filename)
mdl.evaluate(np.squeeze(u_train),np.squeeze(u_train))

# ===============================================
decoder = mdl.decoder

gridx,gridy = (1001,1001)
z1 = np.linspace(-0.8,0.8,gridx).astype('float32')
z2 = np.linspace(-0.8,0.8,gridy).astype('float32')
idx_z2_0 = np.squeeze(np.argwhere(np.abs(z2-0)<1.19209e-07))
print(idx_z2_0)
zx,zy = np.meshgrid(z1,z2)

### compute derivatives in batches
batch_size = 60
dy_dz = []

# hold z1 constant, change z2
for i in range(len(z1)):
    dy_dz_i = []
    t = 0
    # for j in range(len(z2)):
    while (t+batch_size) < len(z2):

        # print(f'in this batch t={t} to {t+batch_size}')
        # print(tf.stack([zx[t:t+batch_size,i],zy[t:t+batch_size,0]],axis=1)[:3,...])

        gradients = model_evaluation.get_gradient_m_z(
            tf.stack([zx[t:t+batch_size,i],zy[t:t+batch_size,0]],axis=1),
            decoder
        ) # [batch,....,z]

        dy_dz_i.append(gradients)

        t = t + batch_size
    
    # print(f'in this batch t={t} to {len(z2)}')
    gradients = model_evaluation.get_gradient_m_z(tf.stack([zx[t:,0],zy[t:,0]],axis=1),decoder) 

    dy_dz_i.append(gradients)

    dy_dz.append(np.vstack(dy_dz_i))

dy_dz = np.array(dy_dz) # (no.of z1, z2, ny, nz, nu, latent_dim)
np.save(saveas,dy_dz)