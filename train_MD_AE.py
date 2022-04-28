# the mode decomposing autoencoder is introduced by
# > T. Murata, K. Fukami & K. Fukagata,
# > "Nonlinear mode decomposition with convolutional neural networks for fluid dynamics,"
# > J. Fluid Mech. Vol. 882, A13 (2020).
# > https://doi.org/10.1017/jfm.2019.822

from re import S
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import backend as K

# from MD_AE_model import *
from temp_model_no_bias import *

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
latent_dim = 10
batch_size = Ntrain
act_fct = 'linear'
resize_meth = 'bilinear'
filter_window= (3,3)
batch_norm = False

## training
nb_epoch = 500
batch_size = 100
learning_rate = 0.0001
learning_rate_list = [0.001, 0.001, 0.0005, 0.0001, 0.00005] #[0.001,0.0001,0.00001]

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
print('Creating a mode-decomposing autoencoder with latent dimension', latent_dim)
md_ae = MD_Autoencoder(Nx=Nx,Nu=Nu,features_layers=features_layers,latent_dim=latent_dim,filter_window=filter_window,act_fct=act_fct,drop_rate=drop_rate,lmb=lmb)
print(md_ae.summary())

#================================================ TRAINING ==========================================
md_ae.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse') # or use tf.keras.losses.MeanAbsoluteError()
pat = 100 # patience for EarlyStopping

# Early stopping
hist_train = []
hist_val = []
tempfn = './temp_md_autoencoder.h5'
model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
cb = [model_cb, early_cb]

# Training
print('Training...')
for i in range(len(learning_rate_list)):
    learning_rate = learning_rate_list[i]
    K.set_value(md_ae.optimizer.lr,learning_rate)
    hist0 = md_ae.fit(u_train[0,:,:,:,:], u_train[0,:,:,:,:],
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(u_val[0,:,:,:,:], u_val[0,:,:,:,:]),
                    callbacks=cb)  
    md_ae.load_weights(tempfn)
    hist_train.extend(hist0.history['loss'])
    hist_val.extend(hist0.history['val_loss'])
print('Finished training')

# ============================================= Testing =============================
encoder = md_ae.encoder
decoders = []
for name in md_ae.name_decoder:
    decoders.append(md_ae.get_layer(name))

print('Testing...')
if LATENT_STATE:
    coded_train = encoder.predict(u_train[0,:,:,:,:])
    mode_train = []
    for i in range(0,latent_dim):
        z = coded_train[:,0]
        z = np.reshape(z,(-1,1))
        mode_train.append(decoders[i].predict(z))
    y_train = np.sum(mode_train,axis=0)
    coded_test = encoder.predict(u_test[0,:,:,:,:])
    mode_test = []
    for i in range(0,latent_dim):
        z = coded_test[:,0]
        z = np.reshape(z,(-1,1))
        mode_test.append(decoders[i].predict(z))
    y_test = np.sum(mode_test,axis=0)
else:
    y_train = md_ae.predict(u_train[0,:,:,:,:])
    y_test = md_ae.predict(u_test[0,:,:,:,:])
print('Finished testing')

#========================================== Saving Results ==============================
print('Saving results')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
# Create a new folder for the results
folder = '/home/ym917/OneDrive/PhD/Code_md-ae/MD_' + str(latent_dim) + '__' + st + '/'
os.mkdir(folder)

filename = folder + 'test_summary.txt'
with open(filename,'w') as f:
    with redirect_stdout(f):
        print("Shuffle ",SHUFFLE)
        print("Remove mean ",REMOVE_MEAN)
        print("Learning rate ",learning_rate_list)
        print("Drop out ",drop_rate)
        print('Activation function',act_fct)

# summary of structure
filename = folder + 'Autoencoder_summary.txt'
with open(filename,'w') as f:
    with redirect_stdout(f):
        print('Autoencoder')
        print(md_ae.summary(),)
        print('\nEncoder')
        print(md_ae.encoder.summary())
        print('\nDecoder')
        print(md_ae.decoder.summary())

# model parameters
filename = folder + 'Model_param.h5'
hf = h5py.File(filename,'w')
hf.create_dataset('Ny',data=Ny)
hf.create_dataset('Nz',data=Nz)
hf.create_dataset('Nu',data=Nu)
hf.create_dataset('features_layers',data=features_layers)
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

# model weights
filename = folder + 'md_ae_model.h5'
md_ae.save_weights(filename)

# save results
filename = folder + 'results.h5'
hf = h5py.File(filename,'w')
hf.create_dataset('u_all',data=u_all[0,:,:,:,:])
hf.create_dataset('hist_train',data=np.array(hist_train))
hf.create_dataset('hist_val',data=hist_val)
hf.create_dataset('u_train',data=u_train[0,:,:,:,:]) #u_train_fluc before
hf.create_dataset('u_val',data=u_val[0,:,:,:,:])
hf.create_dataset('u_test',data=u_test[0,:,:,:,:])
hf.create_dataset('y_test',data=y_test)
hf.create_dataset('y_train',data=y_train)
if REMOVE_MEAN:
    hf.create_dataset('u_avg',data=u_mean_all)
if LATENT_STATE:
    hf.create_dataset('latent_dim',data=latent_dim)
    hf.create_dataset('latent_train',data=coded_train)
    hf.create_dataset('latent_test',data=coded_test)
    hf.create_dataset('modes_train',data=mode_train)
    hf.create_dataset('modes_test',data=mode_test) # has shape (modes,snapshots,Nx,Ny,Nu)
hf.close()

#=================================== PLOT ==============================================
fig_count = 0

# training history
path = folder + 'training_history.png'
fig_count = fig_count + 1
plt.figure(fig_count)
plt.plot(hist_train,label="training")
plt.plot(hist_val,label="validation")
plt.title("Training autoencoder")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(path)

# find mean absolute error
if REMOVE_MEAN:
    y_test = y_test + u_mean_all # add fluctuation and average velocities
    u_test = u_test[0,:,:,:,:] + u_mean_all
else:
    u_test = u_test[0,:,:,:,:]

y_mean = np.mean(y_test,0)
u_mean = np.mean(u_test[:,:,:,:],0)
e = np.abs(y_test-u_test)
e_mean = np.mean(e,0)

# plot comparison
# find common colourbar
umin = min(np.amin(u_mean[:,:,0]),np.amin(y_mean[:,:,0]))
umax = max(np.amax(u_mean[:,:,0]),np.amax(y_mean[:,:,0]))

vmin = min(np.amin(u_mean[:,:,1]),np.amin(y_mean[:,:,1]))
vmax = max(np.amax(u_mean[:,:,1]),np.amax(y_mean[:,:,1]))

fig_count = fig_count + 1
path = folder + 'autoencoder_results.png'
plt.figure(fig_count)

ax1 = plt.subplot(2,3,1,title="True",xticks=[],yticks=[],ylabel='v')
ax1 = plt.imshow(u_mean[:,:,0],'jet',vmin=umin,vmax=umax)
plt.colorbar()

ax2 = plt.subplot(2,3,2,title="Predicted",xticks=[],yticks=[])
ax2 = plt.imshow(y_mean[:,:,0],'jet',vmin=umin,vmax=umax)
plt.colorbar()

ax3 = plt.subplot(2,3,3,title="Absolute error",xticks=[],yticks=[]) # u error
ax3 = plt.imshow(e_mean[:,:,0],'jet')
plt.colorbar()

ax4 = plt.subplot(2,3,4,xticks=[],yticks=[],ylabel='w')
ax4 = plt.imshow(u_mean[:,:,1],'jet',vmin=vmin,vmax=vmax)
plt.colorbar()

ax5 = plt.subplot(2,3,5,xticks=[],yticks=[])
ax5 = plt.imshow(y_mean[:,:,1],'jet',vmin=vmin,vmax=vmax)
plt.colorbar()

ax6 = plt.subplot(2,3,6,xticks=[],yticks=[]) 
ax6 = plt.imshow(e_mean[:,:,1],'jet')
plt.colorbar()
plt.savefig(path)

# plot modes
if LATENT_STATE:
    fig_count = fig_count + 1
    path = folder + 'averaged_decomposed_field.png'
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

    mode1_1 = ax1.imshow(np.mean(mode_test,axis=1)[0,:,:,0],'jet')
    ax1.set_title("Decomposed field 1, v")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(mode1_1,ax=ax1)

    mode1_2 = ax2.imshow(np.mean(mode_test,axis=1)[0,:,:,1],'jet')
    ax2.set_title("Decomposed field 1, w")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(mode1_2,ax=ax2)

    mode2_1 = ax3.imshow(np.mean(mode_test,axis=1)[1,:,:,0],'jet')
    ax3.set_title("Decomposed field 2, v")
    ax3.set_xticks([])
    ax3.set_yticks([])
    fig.colorbar(mode2_1,ax=ax3)

    mode2_2 = ax4.imshow(np.mean(mode_test,axis=1)[1,:,:,1],'jet')
    ax4.set_title("Decomposed field 2, w")
    ax4.set_xticks([])
    ax4.set_yticks([])
    fig.colorbar(mode2_2,ax=ax4)
    plt.suptitle("u and v autoencoder modes")
    plt.savefig(path)

# plot latent variables
fig_count = fig_count + 1
path = folder + 'latent_variables.png'
plt.figure(fig_count)
plt.plot(coded_test[:,0],label='1')
plt.plot(coded_test[:,1],label='2')
plt.legend()
plt.ylabel('value of latent variable')
plt.title("Testing autoencoder")
plt.savefig(path)


now = datetime.datetime.now().strftime("%H:%M")
print("Started at ", start_time)
print("Finished at ",now)
