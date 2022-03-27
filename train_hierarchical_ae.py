import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import backend as K

from MD_AE_model import *

import h5py
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import time
import os
import datetime

start_time = datetime.datetime.now().strftime("%H:%M")

#============================== CHANGE THESE VALUES ======================
# data
data_file = './PIV4_downsampled_by8.h5'
Ntrain = 1500 # snapshots for training
Nval = 632 # sanpshots for validation
Ntest = 600

# Boolean 
LATENT_STATE = True # save latent state
SHUFFLE = True # shuffle before splitting into sets, test set is extracted before shuffling
REMOVE_MEAN = True # train on fluctuating velocity

## ae configuration
lmb = 0.0 #1e-05 #regulariser
drop_rate = 0.2
features_layers = [32, 64, 128]
latent_dim = 1
no_of_modes = 2
batch_size = Ntrain
act_fct = 'tanh'
resize_meth = 'bilinear'
filter_window= (3,3)
batch_norm = False

## training
nb_epoch = 500
batch_size = 100
learning_rate = 0.001
learning_rate_list = [0.001,0.0005,0.0001]

#================================= IMPORT DATA ==========================================================
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

#=================================== DEFINE TRAINING ==========================================
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

#======================================= CREATE AUTOENCODER =======================================
subnet1 = hierarchicalAE_sub(Nx=Nx,Nu=Nu,previous_dim=[],features_layers=features_layers,latent_dim=latent_dim,filter_window=filter_window,act_fct=act_fct,drop_rate=drop_rate,lmb=lmb)
subnet2 = hierarchicalAE_sub(Nx=Nx,Nu=Nu,previous_dim=[1],features_layers=features_layers,latent_dim=1,filter_window=filter_window,act_fct=act_fct,drop_rate=drop_rate,lmb=lmb)

#================================================ TRAINING ==========================================
subnet1.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse') # or use tf.keras.losses.MeanAbsoluteError()
subnet2.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse') 
pat = 100 # patience for EarlyStopping

hist_train_full = []
hist_val_full = []

# subnet1
hist_train = []
hist_val = []
tempfn = './temp_hierarchical_autoencoder.h5'
model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
cb = [model_cb, early_cb]

print('Training subnet1...')
for i in range(len(learning_rate_list)):
    learning_rate = learning_rate_list[i]
    K.set_value(subnet1.optimizer.lr,learning_rate)
    hist0 = subnet1.fit([u_train[0,:,:,:,:]], u_train[0,:,:,:,:],
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=([u_val[0,:,:,:,:]], u_val[0,:,:,:,:]),
                    callbacks=cb)  
    subnet1.load_weights(tempfn)
    hist_train.extend(hist0.history['loss'])
    hist_val.extend(hist0.history['val_loss'])
hist_train_full.append(hist_train)
hist_val_full.append(hist_val)
z_train_1 = subnet1.encoder.predict(u_train[0,:,:,:,:])
z_val_1 = subnet1.encoder.predict(u_val[0,:,:,:,:])

# subnet2
hist_train = []
hist_val = []
tempfn = './temp_hierarchical_autoencoder.h5'
model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
cb = [model_cb, early_cb]

print('Training subnet2...')
for i in range(len(learning_rate_list)):
    learning_rate = learning_rate_list[i]
    K.set_value(subnet2.optimizer.lr,learning_rate)
    hist0 = subnet2.fit([u_train[0,:,:,:,:],z_train_1], u_train[0,:,:,:,:],
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=([u_val[0,:,:,:,:],z_val_1], u_val[0,:,:,:,:]),
                    callbacks=cb)  
    subnet2.load_weights(tempfn)
    hist_train.extend(hist0.history['loss'])
    hist_val.extend(hist0.history['val_loss'])
hist_train_full.append(hist_train)
hist_val_full.append(hist_val)
z_train_2 = subnet2.encoder.predict(u_train[0,:,:,:,:])
z_val_2 = subnet2.encoder.predict(u_val[0,:,:,:,:])
print('Finished training')

# ============================================= Saving =============================#
print('Saving results')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
# Create a new folder for the results
folder = 'C:/Users/tracy/OneDrive - Imperial College London/PhD/Code_md-ae/Hierarchical_' + str(no_of_modes) +'_' + str(latent_dim) + '__' + st + '/'
os.mkdir(folder)

# summary of structure
filename = folder + 'Autoencoder_summary.txt'
with open(filename,'w') as f:
    with redirect_stdout(f):
        print('Subnet1')
        print(subnet1.summary())
        print('\nSubnet2')
        print(subnet2.summary())

# model parameters
filename = folder + 'Model_param.h5'
hf = h5py.File(filename,'w')
hf.create_dataset('Ny',data=Ny)
hf.create_dataset('Nz',data=Nz)
hf.create_dataset('Nu',data=Nu)
hf.create_dataset('features_layers',data=features_layers)
hf.create_dataset('no_of_modes',data=no_of_modes)
hf.create_dataset('latent_dim',data=latent_dim)
hf.create_dataset('resize_meth',data=np.string_(resize_meth),dtype="S10")
hf.create_dataset('filter_window',data=filter_window)
hf.create_dataset('act_fct',data=np.string_(act_fct),dtype="S10")
hf.create_dataset('batch_norm',data=bool(batch_norm))
hf.create_dataset('drop_rate',data=drop_rate)
hf.create_dataset('lmb',data=lmb)
hf.create_dataset('LATENT_STATE',data=LATENT_STATE)
hf.create_dataset('SHUFFLE',data=SHUFFLE)
hf.create_dataset('REMOVE_MEAN',data=REMOVE_MEAN)
if SHUFFLE:
    hf.create_dataset('idx_unshuffle',data=idx_unshuffle) # fpr un-shuffling u_all[0:Ntrain+Nval,:,:,:]
hf.close()

# save models
filename = folder + 'subnet1'
subnet1.save(filename)
filename = folder + 'subnet2'
subnet2.save(filename)

# save results
filename = folder + 'results.h5'
hf = h5py.File(filename,'w')
hf.create_dataset('u_all',data=u_all[0,:,:,:,:])
hf.create_dataset('hist_train',data=np.array(hist_train_full))
hf.create_dataset('hist_val',data=hist_val_full)
hf.create_dataset('u_train',data=u_train[0,:,:,:,:])
hf.create_dataset('u_val',data=u_val[0,:,:,:,:])
hf.create_dataset('u_test',data=u_test[0,:,:,:,:])
# hf.create_dataset('y_test',data=y_test)
# hf.create_dataset('y_train',data=y_train)
if REMOVE_MEAN:
    hf.create_dataset('u_avg',data=u_mean_all)
# if LATENT_STATE:
#     hf.create_dataset('latent_dim',data=latent_dim)
#     hf.create_dataset('latent_train',data=coded_train)
#     hf.create_dataset('latent_test',data=coded_test)
#     hf.create_dataset('modes_train',data=mode_train)
#     hf.create_dataset('modes_test',data=mode_test) # has shape (modes,snapshots,Nx,Ny,Nu)
# hf.close()