'''
Functions used in the training of the network.
'''

from git import Remote
import h5py
import numpy as np
import einops as ei

# utils 
import pathlib
import typing
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.losses import MeanSquaredError as mse

path = typing.Union[str,pathlib.Path]
dtype = typing.Union[str,np.dtype]



def data_partition(data_file:path, 
                    data_size:list, 
                    partition:list, 
                    SHUFFLE=True, 
                    REMOVE_MEAN=True, 
                    data_type:dtype=np.float32) -> np.ndarray:

    '''Split the experimental data into sets. 

    Arguments:
        data_file: location of the data file, must be .h5
        data_size: a list of integers describing the size of the data matrix [Nt,Nz,Ny,Nu]
        partition: a list giving the number of snapshots for each set [training, validation, testing]
        SHUFFLE: if the data is shuffled before being split into sets. The testing data will never not be shuffled (all snapshots are in sequence).
        REMOVE_MEAN: if True, return fluctuating velocity.
    Return:
        u_all, (u_train, u_val, u_test), (idx_test, idx_unshuffle), (u_mean_all, u_mean_train, u_mean_val, u_mean_test)

    '''
    Nt = data_size[0]
    Nz = data_size[1]
    Ny = data_size[2]
    Nu = data_size[3]

    Ntrain = partition[0]
    Nval = partition[1]
    Ntest = partition[2]

    if (Ntrain+Nval+Ntest) > Nt:
        raise ValueError('Not enough snapshots to be split into the expected training sets.')

    # get data
    hf = h5py.File(data_file,'r')
    z = np.array(hf.get('z'))
    y = np.array(hf.get('y'))
    u_all = np.zeros((Nt,Nz,Ny,Nu))
    u_all[:,:,:,0] = np.array(hf.get('vy'))
    if Nu == 2:
        u_all[:,:,:,1] = np.array(hf.get('vz'))
    elif Nu > 2:
        sys.exit('Function not for handling Nu > 2.')
    hf.close()

    u_all = ei.rearrange(u_all,'nt nz ny nu -> nt ny nz nu') #shape of u_all = (Nt,Ny,Nz,Nu)

    if SHUFFLE:
        idx_test = np.random.randint(0,Nt-Ntest)
        u_test = u_all[idx_test:idx_test+Ntest,:,:,:] # test set needs to be in order and has continuous snapshots
        u_all = np.delete(u_all,np.s_[idx_test:idx_test+Ntest],0) # remove the test set from available samples
        idx_shuffle = np.arange(Nt-Ntest) # use idx_shuffle to shuffle the rest of samples before taking a validation set
        np.random.shuffle(idx_shuffle)
        idx_unshuffle = np.argsort(idx_shuffle) # use idx_unshuffle to unshuffle the data
        u_all = u_all[idx_shuffle,:,:,:]
        u_train = u_all[0:Ntrain,:,:,:]
        u_val = u_all[Ntrain:Ntrain+Nval,:,:,:]
        u_all = np.vstack((u_train,u_val,u_test))
    else:
        u_train = u_all[0:Ntrain,:,:,:]
        u_val = u_all[Ntrain:Ntrain+Nval,:,:,:]
        u_test = u_all[Ntrain+Nval:Ntrain+Nval+Ntest,:,:,:]
        u_all = u_all[0:Ntrain+Nval+Ntest,:,:,:] # u_all has shape (Ntrain+Nval+Ntest,Ny,Nz,Nu)

    if REMOVE_MEAN:
        u_mean_all = np.mean(u_all,axis=0).astype(data_type)
        u_all = u_all - u_mean_all
        u_mean_train = np.mean(u_train,axis=0).astype(data_type)
        u_train = u_train-u_mean_train
        u_mean_val = np.mean(u_val,axis=0).astype(data_type)
        u_val = u_val - u_mean_val
        u_mean_test = np.mean(u_test,axis=0).astype(data_type)
        u_test = u_test-u_mean_test

    # add a new axis to all matrix
    u_all = np.reshape(u_all,(1,Ntrain+Nval+Ntest,Ny,Nz,Nu)).astype(data_type) # new shape (1,Nval+Ntrain+Ntest,Ny,Nz,Nu)
    u_train = np.reshape(u_train,(1,Ntrain,Ny,Nz,Nu)).astype(data_type)
    u_val = np.reshape(u_val,(1,Nval,Ny,Nz,Nu)).astype(data_type)
    u_test = np.reshape(u_test,(1,Ntest,Ny,Nz,Nu)).astype(data_type)

    if not SHUFFLE:
        idx_test = None
        idx_unshuffle = None
    
    if not REMOVE_MEAN:
        u_mean_all = None
        u_mean_test = None
        u_mean_train = None
        u_mean_val = None
    
    return u_all, (u_train, u_val, u_test), (idx_test, idx_unshuffle), (u_mean_all, u_mean_train, u_mean_val, u_mean_test)


def train_autoencder(mdl:Model, data:tuple, batch_size:int, epochs:int, 
            early_stopping_patience:int=100, 
            save_model_to:path='./temp_md_autoencoder.h5', 
            history:tuple=None):

    '''Train a tensorflow autoencoder 

    Using the defined batch size for the defined number of epochs. 

    Arguments:
        mdl: a compiled autoencoder model
        data: (u_train, u_val, u_test)
        batch_size: size of each batch
        epochs: number of epochs
        early_stopping_patience: pat for early stopping
        save_model_to: '.h5' file to save the weights
        history: tupe of lists (hist_train, hist_val). Use empty lists if not given
    Returns:
        hist_train: a list of training losses, length equals to the numbe of epochs
        hist_val: a list of validation losses
        mse_test: mean squared error of u_test and predicted u_test
    '''

    u_train = data[0] # unpack data
    u_val = data[1]
    u_test = data[2]

    if history: # start fresh if history is not provided
        hist_train = history[0]
        hist_val = history[1]
    else:
        hist_train = []
        hist_val = []
        

    model_cb=ModelCheckpoint(save_model_to, monitor='val_loss',save_best_only=True,verbose=1,save_weights_only=True)
    early_cb=EarlyStopping(monitor='val_loss', patience=early_stopping_patience,verbose=1)
    cb = [model_cb, early_cb]

    hist0 = mdl.fit(np.squeeze(u_train,axis=0), np.squeeze(u_train,axis=0),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(np.squeeze(u_val,axis=0), np.squeeze(u_val,axis=0)),
                    callbacks=cb,
                    verbose=0)
    mdl.load_weights(save_model_to)
    hist_train.extend(hist0.history['loss'])
    hist_val.extend(hist0.history['val_loss'])
    mse_test= mse()(np.squeeze(u_test,axis=0), mdl.predict(np.squeeze(u_test,axis=0))).numpy()

    return hist_train, hist_val, mse_test

