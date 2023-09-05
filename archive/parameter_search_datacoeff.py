import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam

import numpy as np
import wandb
import h5py

import MD_AE_tools.mode_decomposition as md
import MD_AE_tools.models.models_no_bias as my_models
from project_specific_utils.data_and_train import create_update_fn, set_gpu

set_gpu(2,2048)


#============================== these values do not change ======================

# What is the input?
source = 'data' # data or coeff
cut_off = None
if not (source == 'coeff'): cut_off = None
if (source!='coeff') and (source!='data'): 
    raise ValueError('Wrong input. Is this typo?')

# Data
data_file = './data/ufluc_shuffle_1632.h5'

REMOVE_MEAN = True
latent_dim = 2
nb_epoch = 20000


with h5py.File(data_file,'r') as hf:
    u_train = np.squeeze(np.array(hf.get('u_train'))).astype('float32')
    u_val = np.squeeze(np.array(hf.get('u_val'))).astype('float32')
    u_test = np.squeeze(np.array(hf.get('u_test'))).astype('float32')

[ntrain,ny,nz,nu] = u_train.shape
nval = u_val.shape[0]
ntest = u_test.shape[0]

if source == 'coeff':
    print('Using POD time coefficients as input.')

    u_all = np.vstack((u_train,u_val,u_test))
    vy = u_all[:,:,:,0]
    vy = np.transpose(vy,[1,2,0])
    vz = u_all[:,:,:,1]
    vz = np.transpose(vz,[1,2,0])
    X = np.vstack((vz,vy))

    pod = md.POD(X,method='classic')
    time_coeff = pod.get_time_coefficient.astype('float32') # shape (time, number of points)
    
    if cut_off:
        time_coeff = time_coeff[:,:cut_off] 
    
    data_train = tf.data.Dataset.from_tensor_slices((time_coeff[:ntrain,:]))
    data_val = tf.data.Dataset.from_tensor_slices((time_coeff[ntrain:ntrain+nval,:])).batch(nval)
    data_test = tf.data.Dataset.from_tensor_slices((time_coeff[ntrain+nval:,:])).batch(ntest)

elif source == 'data':
    data_train = tf.data.Dataset.from_tensor_slices((u_train.reshape((ntrain,-1))))
    data_val = tf.data.Dataset.from_tensor_slices((u_val.reshape((nval,-1)))).batch(nval)
    data_test = tf.data.Dataset.from_tensor_slices((u_test.reshape((ntest,-1)))).batch(ntest)


# ========================== set up sweep ==============


default_config = {
    'cut_off':cut_off,
    'latent_dim':latent_dim,
    'REMOVE_MEAN':REMOVE_MEAN,
    'batch_size':500,
    'learning_rate':0.0003,
    'regularisation':0.00001,
    'num_layers':3,
    'layer1':500,
    'layer2':300,
    'layer3':100,
    'activation':'tanh',
    'dropout':0.0
}

run = wandb.init(config=default_config,project='POD_and_AE',group=f'FF_AE_trained_on_{source}')
layers = [wandb.config.layer1, wandb.config.layer2, wandb.config.layer3][:wandb.config.num_layers]
wandb.config.update({'layer':layers})
params = wandb.config

# ========================= Model ==================================

input_shape = cut_off if cut_off else ny*nz*nu
loss_fn = tf.keras.losses.MeanSquaredError()

data_train = data_train.batch(params.batch_size)

optimiser = Adam(learning_rate=params.learning_rate)


mdl = my_models.Autoencoder_ff(
    input_shape=input_shape,
    latent_dim=latent_dim,
    layer_sizes=params.layer,
    regularisation=params.regularisation,
    act_fct=params.activation,
    drop_rate=params.dropout
)

mdl.build((None,input_shape))
# print(mdl.summary())


# ======================== training ========================


update = create_update_fn(mdl,loss_fn,optimiser)

for epoch in range(1,nb_epoch+1):
    loss_epoch = []
    for xx in data_train:
        loss_batch = update(xx,xx).numpy()
        loss_epoch.append(loss_batch)
    loss = np.mean(loss_epoch)

    # validation
    for xx_val in data_val:
        y_val = mdl(xx_val,training=False)
    loss_val = loss_fn(xx_val,y_val).numpy()
    
    run.log({'loss':loss, 'loss_val':loss_val})
    

for xx_test in data_test:
    y_test = mdl(xx_test,training=False)    
    loss_test = loss_fn(xx_test,y_test).numpy()

if source == 'coeff':
    if not cut_off:
        m = ny*nz*nu
    else:
        m = cut_off
    y_reconstructed = pod.reconstruct(m,A=y_test.numpy()).astype('float32')#y_test.numpy()
    loss_image = loss_fn(X[...,(ntrain+nval):],y_reconstructed).numpy()
else:
    loss_image = loss_test

run.log({'loss_test(mse)':loss_test, 'loss_image':loss_image})
run.save('parameter_search.py')
run.finish()