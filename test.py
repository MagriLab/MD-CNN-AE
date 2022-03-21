import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization, Conv2DTranspose, Flatten, PReLU, Reshape, Dropout, AveragePooling2D, Add, Lambda, Layer

from MD_AE_model import *

import h5py
import numpy as np

# data
data_file = './PIV4_downsampled_by8.h5'
Ntrain = 1500 # snapshots for training
Nval = 632 # sanpshots for validation
Ntest = 600

# Boolean 
LATENT_STATE = True # save latent state
SHUFFLE = True # shuffle before splitting into sets, test set is extracted before shuffling
REMOVE_MEAN = False # train on fluctuating velocity

## ae configuration
lmb = 0.0 #1e-05 #regulariser
drop_rate = 0.2
features_layers = [32, 64, 128]
# latent_dim = 10
batch_size = Ntrain
act_fct = 'tanh'
resize_meth = 'bilinear'
filter_window= (3,3)
batch_norm = False

Nz = 24 # grid size
Ny = 21
Nu = 2
Nt = 2732 # number of snapshots available
D = 196.5 # mm diameter of bluff body
U_inf = 15 # m/s freestream velocity
f_piv = 720.0 # Hz PIV sampling frequency  
dt = 1.0/f_piv 

print('Reading dataset from :' + data_file)
hf = h5py.File(data_file,'r')

z = np.array(hf.get('z'))
y = np.array(hf.get('y'))
u_all = np.zeros((Nt,Nz,Ny,Nu))
u_all[:,:,:,0] = np.array(hf.get('vy'))
if Nu==2:
    u_all[:,:,:,1] = np.array(hf.get('vz'))
u_all = np.transpose(u_all,[0,2,1,3]) # shape of u_all = (Nt,Ny,Nz,Nu)
hf.close()

u_all = u_all[:,:,:,:].astype('float32')

# remove mean for modes
if REMOVE_MEAN:
    u_mean_all = np.mean(u_all,axis=0) # time averaged, (Ny,Nz,Nu)
    u_all = u_all - u_mean_all


if SHUFFLE:
    # temp_list = list(u_all)
    # np.random.shuffle(temp_list) # this shuffles the first axis
    # u_all = np.array(temp_list)

    idx_test = np.random.randint(0,Nt-Ntest)
    u_test = u_all[idx_test:idx_test+Ntest,:,:,:].astype('float32') # test set needs to be in order and has continuous snapshots
    u_all = np.delete(u_all,np.s_[idx_test:idx_test+Ntest],0) # remove the test set from available samples
    idx_shuffle = np.arange(Nt-Ntest) # use idx_shuffle to shuffle the rest of samples before taking a validation set
    np.random.shuffle(idx_shuffle)
    idx_unshuffle = np.argsort(idx_shuffle) # use idx_unshuffle to unshuffle the data
    u_all = u_all[idx_shuffle,:,:,:]
    u_train = u_all[0:Ntrain,:,:,:].astype('float32')
    u_val = u_all[Ntrain:Ntrain+Nval,:,:,:].astype('float32')
    u_all = np.vstack((u_train,u_val,u_test))
else:
    u_train = u_all[0:Ntrain,:,:,:].astype('float32')
    u_val = u_all[Ntrain:Ntrain+Nval,:,:,:].astype('float32')
    u_test = u_all[Ntrain+Nval:Ntrain+Nval+Ntest,:,:,:].astype('float32')
    u_all = u_all[0:Ntrain+Nval+Ntest,:,:,:].astype('float32') # u_all has shape (Ntrain+Nval+Ntest,Ny,Nz,Nu)


# u_train = u_all[0:Ntrain,:,:,:].astype('float32')
# u_val = u_all[Ntrain:Ntrain+Nval,:,:,:].astype('float32')
# u_test = u_all[Ntrain+Nval:Ntrain+Nval+Ntest,:,:,:].astype('float32')
# u_all = u_all[0:Ntrain+Nval+Ntest,:,:,:].astype('float32') # u_all has shape (Ntrain+Nval+Ntest,Ny,Nz,Nu)

u_all = np.reshape(u_all,(1,Ntrain+Nval+Ntest,Ny,Nz,Nu)) # new shape (1,Nval+Ntrain+Ntest,Ny,Nz,Nu)
u_train = np.reshape(u_train,(1,Ntrain,Ny,Nz,Nu))
u_val = np.reshape(u_val,(1,Nval,Ny,Nz,Nu))
u_test = np.reshape(u_test,(1,Ntest,Ny,Nz,Nu))
Nx = [Ny, Nz]

#======================================= CREATE AUTOENCODER =======================================
ae_subnet = hierarchicalAE_sub(Nx=Nx,Nu=Nu,previous_dim=[1],features_layers=features_layers,latent_dim=1)
print(ae_subnet.summary())