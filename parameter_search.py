import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow import keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import backend as K

from MD_AE_model import *

from random import choice
import h5py
import numpy as np
import wandb

# use gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# return u_train, u_val, u_test, Nx
def get_data(data_file,Nt,Ntrain,Nval,Ntest,SHUFFLE=True,REMOVE_MEAN=True):
    Nz = 24 # grid size
    Ny = 21
    Nu = 2
    D = 196.5 # mm diameter of bluff body
    U_inf = 15 # m/s freestream velocity
    f_piv = 720.0 # Hz PIV sampling frequency  
    dt = 1.0/f_piv 

    hf = h5py.File(data_file,'r')
    z = np.array(hf.get('z'))
    y = np.array(hf.get('y'))
    u_all = np.zeros((Nt,Nz,Ny,Nu))
    u_all[:,:,:,0] = np.array(hf.get('vy'))
    if Nu==2:
        u_all[:,:,:,1] = np.array(hf.get('vz'))
    u_all = np.transpose(u_all,[0,2,1,3]) # shape of u_all = (Nt,Ny,Nz,Nu)
    hf.close()

    ## training set
    u_all = u_all[:,:,:,:].astype('float32')

    # remove mean for modes
    if REMOVE_MEAN:
        u_mean_all = np.mean(u_all,axis=0) # time averaged, (Ny,Nz,Nu)
        u_all = u_all - u_mean_all


    if SHUFFLE:
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
    
    u_all = np.reshape(u_all,(1,Ntrain+Nval+Ntest,Ny,Nz,Nu)) # new shape (1,Nval+Ntrain+Ntest,Ny,Nz,Nu)
    u_train = np.reshape(u_train,(1,Ntrain,Ny,Nz,Nu))
    u_val = np.reshape(u_val,(1,Nval,Ny,Nz,Nu))
    u_test = np.reshape(u_test,(1,Ntest,Ny,Nz,Nu))
    Nx = [Ny, Nz]

    return u_train, u_val, u_test, Nx

u_train,u_val,u_test,Nx = get_data('./PIV4_downsampled_by8.h5',2732,1500,632,600,True,True)
nb_epochs=500

def create_and_train(hyperparameter_defaults = None):
    # set up defualt hyperparameters
    hyperparameter_defaults = {'features_layers':[32,64,128],'latent_dim':10,'filter_window':(3,3),'batch_size':100, "learning_rate":0.01, "dropout":0.0, "activation":'tanh', "regularisation":0.0, "batch_norm":True, 'REMOVE_MEAN':True}
    run = wandb.init(config=hyperparameter_defaults,project='my_test',entity='yaxinm',group='parameter_search')
    config = wandb.config


    md_ae = MD_Autoencoder(Nx=Nx,Nu=2,features_layers=config.features_layers,latent_dim=config.latent_dim,filter_window=config.filter_window,act_fct=config.activation,batch_norm=config.batch_norm,drop_rate=config.dropout,lmb=config.regularisation)
    md_ae.compile(optimizer=Adam(learning_rate=config.learning_rate),loss='mse')

    hist_train = []
    hist_val = []
    tempfn = './temp_md_autoencoder.h5'
    model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
    early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
    cb = [model_cb, early_cb]
    
    learning_rate_list = [] # learning rate list
    learning_rate_list.extend([config.learning_rate])
    learning_rate_list.extend([config.learning_rate/10])
    learning_rate_list.extend([config.learning_rate/100])

    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        K.set_value(md_ae.optimizer.lr,learning_rate)
        hist0 = md_ae.fit(u_train[0,:,:,:,:], u_train[0,:,:,:,:],
                        epochs=nb_epochs,
                        batch_size=config.batch_size,
                        shuffle=True,
                        validation_data=(u_val[0,:,:,:,:], u_val[0,:,:,:,:]),
                        callbacks=cb,
                        verbose=0)  
        md_ae.load_weights(tempfn)
        hist_train.extend(hist0.history['loss'])
        hist_val.extend(hist0.history['val_loss'])
    print('finished one training')
    
    loss_test= md_ae.evaluate(u_test[0,:,:,:,:],u_test[0,:,:,:,:],verbose=0)
    print(loss_test)

    # log loss to weights and bias
    with run:
        for epoch in range(len(hist_train)):
            wandb.log({"loss_train":hist_train[epoch], "loss_val":hist_val[epoch], "loss_test":loss_test})

sweep_config = {
  'method': 'random', 
  'metric': {
      'name': 'loss_val',
      'goal': 'minimize'
  },
  'parameters': {
      'activation': {
          'values': ['tanh','relu']
      },
      'batch_norm':{
          'values': [True,False]
      },
      'dropout':{
          'values': [0.0,0.1,0.2,0.3,0.5]
      },
      'regularisation':{
          'values': [0.0,0.001,0.0001,0.00001]
      },
      'learning_rate':{
          'values': [0.01, 0.005, 0.001]
      }
  }
}

sweep_id = wandb.sweep(sweep_config, project="MD-CNN-AE")
wandb.agent(sweep_id, function=create_and_train, count=20)