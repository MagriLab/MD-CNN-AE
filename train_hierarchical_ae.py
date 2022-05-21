import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import backend as K

from MD_AE_model import *
# from model_no_bias import *

import h5py
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import wandb

import time
import os
import configparser
import datetime

# get system information
config = configparser.ConfigParser()
config.read('__system.ini')
system_info = config['system_info']

# use gpu
if system_info.getboolean('GPU'):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[1], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

start_time = datetime.datetime.now().strftime("%H:%M")

#============================== CHANGE THESE VALUES ======================
# data
data_file = './PIV4_downsampled_by8.h5'
Ntrain = 1500 # snapshots for training
Nval = 632 # sanpshots for validation
Ntest = 600

# Boolean 
SHUFFLE = True # shuffle before splitting into sets, test set is extracted before shuffling
REMOVE_MEAN = True # train on fluctuating velocity

## ae configuration
lmb = 0.001 #1e-05 #regulariser
drop_rate = 0.2
features_layers = [32, 64, 128]
latent_dim = 1
no_of_modes = 10
act_fct = 'tanh'
resize_meth = 'bilinear'
filter_window= (3,3)
batch_norm = True

## training
nb_epoch = 500
batch_size = 100
learning_rate = 0.001
learning_rate_list = [learning_rate,learning_rate/10,learning_rate/100]
save_network = 'all' # the subnets (e.g. [0,1,3]) whose wights will be saved to a .h5 file. Use 'all' if saving all network (be careful if there are many subnets.)
LOG_WANDB = True

# initalise weights&biases
if LOG_WANDB:
    config_wandb = {'features_layers':features_layers,'latent_dim':no_of_modes,'filter_window':filter_window,'batch_size':batch_size, "learning_rate":learning_rate, "dropout":drop_rate, "activation":act_fct, "regularisation":lmb, "batch_norm":batch_norm, 'REMOVE_MEAN':REMOVE_MEAN}
    run_name = "hierarchical-"+str(no_of_modes)+"-mode"
    run = wandb.init(config=config_wandb,project="MD-CNN-AE",entity="yaxinm",group="hierarchical",name=run_name)

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

# remove mean for training
if REMOVE_MEAN:
    u_mean_all = np.mean(u_all,axis=0)
    u_all = u_all - u_mean_all
    u_mean_train = np.mean(u_train,axis=0)
    u_train = u_train-u_mean_train
    u_mean_val = np.mean(u_val,axis=0)
    u_val = u_val - u_mean_val
    u_mean_test = np.mean(u_test,axis=0)
    u_test = u_test-u_mean_test

u_all = np.reshape(u_all,(1,Ntrain+Nval+Ntest,Ny,Nz,Nu)) # new shape (1,Nval+Ntrain+Ntest,Ny,Nz,Nu)
u_train = np.reshape(u_train,(1,Ntrain,Ny,Nz,Nu))
u_val = np.reshape(u_val,(1,Nval,Ny,Nz,Nu))
u_test = np.reshape(u_test,(1,Ntest,Ny,Nz,Nu))
Nx = [Ny, Nz]

#======================================= CREATE AUTOENCODER =======================================
previous_dim = []
subnets = []
for _ in range(no_of_modes):
    subnets.extend([HierarchicalAE_sub(Nx=Nx,Nu=Nu,previous_dim=previous_dim,features_layers=features_layers,latent_dim=latent_dim,filter_window=filter_window,act_fct=act_fct,batch_norm=batch_norm,drop_rate=drop_rate,lmb=lmb)])
    previous_dim.extend([latent_dim])

#================================================ TRAINING ==========================================
for i in range(no_of_modes):
    subnets[i].compile(optimizer=Adam(learning_rate=learning_rate),loss='mse') # or use tf.keras.losses.MeanAbsoluteError()
pat = 100 # patience for EarlyStopping

hist_train_full = []
hist_val_full = []


# tempfn = './temp_hierarchical_autoencoder.h5'
# model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
# early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
# cb = [model_cb, early_cb]

input_train = [u_train[0,:,:,:,:]]
input_val = [u_val[0,:,:,:,:]]

print('Starting training')
# train each subnet in a loop
for j in range(no_of_modes):
    # if os.path.exists(tempfn):
    #     os.remove(tempfn)
    tempfn = './temp_hierarchical_autoencoder' + str(j) + '.h5'
    model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
    early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
    cb = [model_cb, early_cb]
    print('training subnet ', str(j+1))
    hist_train = []
    hist_val = []
    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        K.set_value(subnets[j].optimizer.lr,learning_rate)
        hist0 = subnets[j].fit(input_train, u_train[0,:,:,:,:],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(input_val, u_val[0,:,:,:,:]),
                        callbacks=cb)  
        subnets[j].load_weights(tempfn)
        hist_train.extend(hist0.history['loss'])
        hist_val.extend(hist0.history['val_loss'])
    hist_train_full.extend(hist_train)
    hist_val_full.extend(hist_val)
    hist_train_full.extend([-1]) # separate each subnet. 
    hist_val_full.extend([-1]) #use np.where(hist_train_full==-1) to return the idx 

    # get latnet variable for the next subnet
    z_train = subnets[j].encoder.predict(u_train[0,:,:,:,:])
    z_val = subnets[j].encoder.predict(u_val[0,:,:,:,:])
    input_train.append(z_train)
    input_val.append(z_val)
    os.remove(tempfn)
print('Finished training')

# ======================================== Testing =================================#
y_test = []
input_test = [u_test[0,:,:,:,:]]

for i in range(no_of_modes):
    y_test.append(subnets[i].predict(input_test))
    input_test.append(subnets[i].encoder.predict(u_test[0,:,:,:,:]))

# loss_test = subnets[-1].evaluate(input_test[:-1],input_test[:-1])
loss_test= tf.keras.losses.MeanSquaredError(u_test[0,:,:,:,:],subnets[-1].predict(input_test[:-1]))

finish_time = datetime.datetime.now().strftime("%H:%M")

# ============================================= Saving =============================#
print('Saving results')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
# Create a new folder for the results
new_folder = '/Hierarchical_' + str(no_of_modes) +'_' + str(latent_dim) + '__' + st + '/'
folder = system_info['save_location'] + new_folder
os.mkdir(folder)


# save to wandb
with run:
    for i in range(len(hist_train_full)):
        if hist_train_full[i] != -1:
            run.log({"loss_train":hist_train_full[i], "loss_val":hist_val_full[i],"loss_test":loss_test})


# summary of structure
filename = folder + 'Autoencoder_summary.txt'
with open(filename,'w') as f:
    with redirect_stdout(f):
        for i in range(no_of_modes):
            print('Subnet ',str(i+1))
            print(subnets[i].summary())
            print(' ')

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
hf.create_dataset('SHUFFLE',data=SHUFFLE)
hf.create_dataset('REMOVE_MEAN',data=REMOVE_MEAN)
if SHUFFLE:
    hf.create_dataset('idx_unshuffle',data=idx_unshuffle) # for un-shuffling u_all[0:Ntrain+Nval,:,:,:]
hf.close()

# save models
if save_network == 'all':
    for network in range(no_of_modes):
        filename = folder + 'subnet' + str(network+1) + '.h5'
        subnets[network].save_weights(filename)
else:
    for network in save_network:
        filename = folder + 'subnet' + str(network+1) + '.h5'
        subnets[network].save_weights(filename)

# save results
filename = folder + 'results.h5'
hf = h5py.File(filename,'w')
hf.create_dataset('u_all',data=u_all[0,:,:,:,:])
hf.create_dataset('hist_train',data=np.array(hist_train_full))
hf.create_dataset('hist_val',data=hist_val_full)
hf.create_dataset('u_train',data=u_train[0,:,:,:,:])
hf.create_dataset('u_val',data=u_val[0,:,:,:,:])
hf.create_dataset('u_test',data=u_test[0,:,:,:,:])
hf.create_dataset('latent_train',data=input_train[1:])# latent space [z1,z2,...]
hf.create_dataset('latent_val',data=input_val[1:])
hf.create_dataset('latent_test',data=input_test[1:])
hf.create_dataset('y_test',data=y_test) # results from [subnet1, subnet2 ...]
if REMOVE_MEAN:
    hf.create_dataset('u_avg',data=u_mean_all)
    hf.create_dataset('u_avg_train',data=u_mean_train)
    hf.create_dataset('u_avg_val',data=u_mean_val)
    hf.create_dataset('u_avg_test',data=u_mean_test)
hf.close()

# summary of test
filename = folder + 'test_summary.txt'
with open(filename,'w') as f:
    with redirect_stdout(f):
        print('Shuffle: ', SHUFFLE)
        print('Remove_mean: ', REMOVE_MEAN)
        print('Latent dimension of each subnet: ', latent_dim)
        print('Number of subnets: ', no_of_modes)
        print('Learning rate: ', learning_rate_list)
        print('Activation: ', act_fct)
        print('Dropout rate', drop_rate)
        print("Batch normalisation ",batch_norm)
        print('Start and finish time: ', start_time, finish_time)

print('Time taken for training and testing')
print('Started at ', start_time)
print('Finished at ', finish_time)