import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import backend as K

from MD_AE_tools.models.models import *

import h5py
import numpy as np
import wandb
import configparser

# get system information
config = configparser.ConfigParser()
config.read('__system.ini')
system_info = config['system_info']

# use gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

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


def create_and_train(u_train,u_val,u_test,Nx,Nu,features_layers,latent_dim,filter_window,act_fct,batch_norm,drop_rate,lmb,batch_size,learning_rate_list,loss,nb_epoch,id):
    md_ae = MD_Autoencoder(Nx=Nx,Nu=Nu,features_layers=features_layers,latent_dim=latent_dim,filter_window=filter_window,act_fct=act_fct,batch_norm=batch_norm,drop_rate=drop_rate,lmb=lmb)
    md_ae.compile(optimizer=Adam(learning_rate=learning_rate_list[0]),loss=loss)

    hist_train = []
    hist_val = []
    tempfn = './temp_md_autoencoder.h5'
    model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
    early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
    cb = [model_cb, early_cb]

    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        K.set_value(md_ae.optimizer.lr,learning_rate)
        hist0 = md_ae.fit(u_train[0,:,:,:,:], u_train[0,:,:,:,:],
                        epochs=nb_epoch,
                        batch_size=batch_size,
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

    return hist_train, hist_val, loss_test



# data parameters
Nt = 2732
Ntrain = 1500
Nval = 632
Ntest = 600
SHUFFLE = True
REMOVE_MEAN = True
Nu = 2

# network parameters
latent_dim = 10
act_fct = 'tanh'
batch_norm = True
drop_rate = 0.2
learning_rate = 0.001
learning_rate_list = [learning_rate,learning_rate/10,learning_rate/100]
lmb = 0.001
features_layers = [32, 64, 128]
filter_window = (3,3)
batch_size = 100
loss = 'mse'
epochs = 500

for i in range(8):
    u_train,u_val,u_test,Nx = get_data('./data/PIV4_downsampled_by8.h5',Nt,Ntrain,Nval,Ntest,SHUFFLE,REMOVE_MEAN)

    config_wandb = {'features_layers':features_layers,'latent_dim':latent_dim,'filter_window':filter_window,'batch_size':batch_size, "learning_rate":learning_rate, "dropout":drop_rate, "activation":act_fct, "regularisation":lmb, "batch_norm":batch_norm, 'REMOVE_MEAN':REMOVE_MEAN}
    run_name = str(latent_dim)+"-mode"
    run = wandb.init(reinit=True,config=config_wandb,project="MD-CNN-AE",entity="yaxinm",group="MD-CNN-AE",name=run_name)

    hist_train,hist_val,loss_test = create_and_train(u_train,u_val,u_test,Nx,Nu,features_layers,latent_dim,filter_window,act_fct,batch_norm,drop_rate,lmb,batch_size,learning_rate_list,loss,epochs,i)

    # log loss to weights and bias
    with run:
        for epoch in range(len(hist_train)):
            run.log({"loss_train":hist_train[epoch], "loss_val":hist_val[epoch], "loss_test":loss_test})
