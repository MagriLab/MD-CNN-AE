import tensorflow as tf
tf.keras.backend.set_floatx('float32')

import numpy as np
import wandb
import h5py

from pathlib import Path
from tensorflow.keras.optimizers import Adam

import MD_AE_tools.models.models_ff as my_models
from project_specific_utils.data_and_train import create_update_fn, set_gpu



# ==================== Change these ====================


latent_dim = 3
encoder_layers = [128,256,256,128,64]
decoder_layers = [64,128,256,256,128]

act_fct = 'tanh'
nb_epoch = 3000


# =======================================================
## these parameters doesn't matter
lmb = 0.0000 #1e-05 #regulariser
drop_rate = 0.0
batch_norm = True
learning_rate = 0.004
batch_size = 200000


# ================== system & data =====================

with h5py.File('./data/raw_pressure_long.h5','r') as hf:
    print(hf.keys())
    fs = np.squeeze(hf.get('fs'))
    static_p = np.squeeze(hf.get('static_p'))
    esp_allt = np.array(hf.get('esp')).T
    r = np.array(hf.get('r')).T
    theta = np.array(hf.get('theta')).T
x=(np.cos(theta*np.pi/180).T)*r
y=(np.sin(theta*np.pi/180).T)*r 
x = x.flatten()
y = y.flatten()

pmean = np.mean(esp_allt,axis=1).reshape(8,8)
prms = np.std(esp_allt,axis=1)

[n,nt] = esp_allt.shape
input_shape = n

p_train = esp_allt - pmean.flatten()[:,np.newaxis]
p_train = p_train.T
dataset = tf.data.Dataset.from_tensor_slices((p_train))


# =================== set up sweep ====================

default_config = {
    'latent_dim' : latent_dim,
    'encoder_layers': encoder_layers,
    'decoder_layers': decoder_layers,
    'activation': act_fct,
    'regularisation': lmb,
    'dropout': drop_rate,
    'batch_norm': batch_norm,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'REMOVE_MEAN' : True
}

run = wandb.init(
    config=default_config,
    project='POD_and_AE'
)
params = wandb.config

# =================== Model ============================

loss_fn = tf.keras.losses.MeanSquaredError()
# data_train = dataset.batch(params.batch_size)
data_train = dataset.shuffle(params.batch_size+100).batch(params.batch_size)

lrschedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    params.learning_rate,
    100,
    t_mul=50,
    m_mul=0.9
)
optimiser = Adam(learning_rate=lrschedule)

ae = my_models.Autoencoder(
    input_shape = input_shape,
    encoder_layers = encoder_layers,
    decoder_layers = decoder_layers,
    latent_dim = latent_dim,
    act_fct = params.activation,
    batch_norm = params.batch_norm,
    drop_rate = params.dropout,
    lmb = params.regularisation
)

ae.build((None,input_shape))
print(ae.summary())
print(params)

update = create_update_fn(ae,loss_fn,optimiser)


# =================== Training =========================

current_best_loss = np.inf

for i in range(nb_epoch):
    loss_epoch = []
    for batch in data_train:
        loss_batch = update(batch,batch).numpy()
        loss_epoch.append(loss_batch)
    loss = np.mean(loss_epoch)

    if loss_batch < current_best_loss:
        current_best_loss = loss_batch
    
    run.log({'loss':loss, 'current_best_loss':current_best_loss})

    if i%10 == 0:
        print(f'Epoch {i}, loss {loss}')


run.save('parameter_search.py')
run.finish()    

