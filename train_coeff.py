import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam

import numpy as np
import configparser
import wandb
import datetime
import h5py
from pathlib import Path

import MD_AE_tools.mode_decomposition as md
import MD_AE_tools.models.models_no_bias as my_models
from project_specific_utils.data_and_train import create_update_fn, set_gpu


# get system information
config = configparser.ConfigParser()
config.read('_system.ini')
system_info = config['system_info']

set_gpu(2,2048)



#============================== CHANGE THESE VALUES ======================

# What is the input?
source = 'coeff' # data or coeff
cut_off = None
if not (source == 'coeff'): cut_off = None
if (source!='coeff') and (source!='data'): 
    raise ValueError('Wrong input. Is this typo?')

# Data
data_file = './data/ufluc_shuffle_1632.h5'

# Boolean 
REMOVE_MEAN = True
LOG_WANDB = False # record loss with weights and biases

## ae configuration
lmb = 0.00001 #1e-05 #regulariser
layers = [500,100,10]
latent_dim = 2
act_fct = 'tanh'

## training
nb_epoch = 5000
batch_size = 500
learning_rate = 0.0001

loss_fn = tf.keras.losses.MeanSquaredError()
optimiser = Adam(learning_rate=learning_rate)
config_wandb = {
    'cut_off':cut_off,
    'layer':layers,
    'latent_dim':latent_dim,
    'batch_size':batch_size,
    'learning_rate':learning_rate,
    'activation':act_fct,
    'regularisation':lmb,
    'REMOVE_MEAN':REMOVE_MEAN
}



# ============================ Read data ==========================

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
    
    data_train = tf.data.Dataset.from_tensor_slices((time_coeff[:ntrain,:])).batch(batch_size)
    data_val = tf.data.Dataset.from_tensor_slices((time_coeff[ntrain:ntrain+nval,:])).batch(nval)
    data_test = tf.data.Dataset.from_tensor_slices((time_coeff[ntrain+nval:,:])).batch(ntest)

elif source == 'data':
    data_train = tf.data.Dataset.from_tensor_slices((u_train.reshape((ntrain,-1)))).batch(batch_size)
    data_val = tf.data.Dataset.from_tensor_slices((u_val.reshape((nval,-1)))).batch(nval)
    data_test = tf.data.Dataset.from_tensor_slices((u_test.reshape((ntest,-1)))).batch(ntest)


# ========================= Model ==================================

input_shape = cut_off if cut_off else ny*nz*nu


mdl = my_models.Autoencoder_ff(
    input_shape=input_shape,
    latent_dim=latent_dim,
    layer_sizes=layers,
    regularisation=lmb,
    act_fct=act_fct
)

mdl.build((None,input_shape))
print(mdl.summary())



# ======================== Training ==========================

# initalise weights&biases
if LOG_WANDB:
    run_name = f'{source}{datetime.datetime.now().strftime("%H:%M:%S")}'
    run = wandb.init(config=config_wandb,project="POD_and_AE",group=f'FF_AE_trained_on_{source}',name=run_name)

update = create_update_fn(mdl,loss_fn,optimiser)

hist_loss = [np.inf]
hist_val = [np.inf]
for epoch in range(1,nb_epoch+1):
    loss_epoch = []
    for xx in data_train:
        # print(xx.shape)
        loss_batch = update(xx,xx)
        loss_epoch.append(loss_batch.numpy())

    loss = np.mean(loss_epoch)
    hist_loss.append(loss)

    # validation
    for xx_val in data_val:
        y_val = mdl(xx_val,training=False)
    loss_val = loss_fn(xx_val,y_val).numpy()
    if loss_val < hist_val[-1]:
        mdl.save_weights('temp_weights.h5')
    hist_val.append(loss_val)


    if LOG_WANDB:
        run.log({'loss':loss, 'loss_val':loss_val})
    
    if epoch % 1 == 0:
        print(f'Epoch {epoch}: loss {loss}, validation loss {loss_val}')
mdl.load_weights('temp_weights.h5')

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

print(f'Final loss from testing: {loss_image}')

if LOG_WANDB:
    run.log({'loss_test(mse)':loss_test, 'loss_image':loss_image})
    run.finish()