'''
Functions used in the training of the network.
'''

import h5py
import numpy as np
import einops as ei
import wandb

# utils 
import pathlib
import typing
import sys
import datetime
import os
import configparser
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.losses import MeanSquaredError as mse

from .helpers import read_config_ini 



StrOrPath = typing.Union[str,pathlib.Path]
dtype = typing.Union[str,np.dtype]
DataTuple = tuple[np.ndarray]
idx = typing.Union[int,list[int]]



def data_partition(data_file:StrOrPath, 
                    data_size:list, 
                    partition:list, 
                    SHUFFLE=True, 
                    REMOVE_MEAN=True, 
                    data_type:dtype=np.float32,
                    rng:typing.Optional[np.random.Generator]=None):

    '''Split the experimental data into sets. 

    Arguments:
        data_file: location of the data file, must be .h5
        data_size: a list of integers describing the size of the data matrix [Nt,Nz,Ny,Nu]
        partition: a list giving the number of snapshots for each set [training, validation, testing]
        SHUFFLE: if the data is shuffled before being split into sets. The testing data will never not be shuffled (all snapshots are in sequence).
        REMOVE_MEAN: if True, return fluctuating velocity.
        data_type: dtype to cast the data into
        rng: numpy random number generator
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
        if rng is not None:
            idx_test = rng.integers(0,Nt-Ntest)
        else:
            idx_test = np.random.randint(0,Nt-Ntest)
        u_test = u_all[idx_test:idx_test+Ntest,:,:,:] # test set needs to be in order and has continuous snapshots
        u_all = np.delete(u_all,np.s_[idx_test:idx_test+Ntest],0) # remove the test set from available samples
        idx_shuffle = np.arange(Nt-Ntest) # use idx_shuffle to shuffle the rest of samples before taking a validation set
        if rng is not None:
            rng.shuffle(idx_shuffle)
        else:
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


def read_data(data_file:StrOrPath):
    '''Read a already partitioned data file
    
    Argument:
        data_file: path to a .h5 file containing a set of already partitioned data

    Return:
        np.ndarray or tuples of np.ndarray
        u_all, (u_train, u_val, u_test), (idx_test, idx_unshuffle), (u_mean_all, u_mean_train, u_mean_val, u_mean_test)
        data_info: dictionary containing basic information of the dataset
    '''
    data_info ={}

    hf = h5py.File(data_file,'r')

    data_info["SHUFFLE"] = hf.get('SHUFFLE')[()]
    data_info["REMOVE_MEAN"] = hf.get('REMOVE_MEAN')[()]
    data_info["Ntrain"] = hf.get('Ntrain')[()]
    data_info["Nval"] = hf.get('Nval')[()]
    data_info["Ntest"] = hf.get('Ntest')[()]
    data_info["Nz"] = hf.get('Nz')[()]
    data_info["Ny"] = hf.get('Ny')[()]
    data_info["Nu"] = hf.get('Nu')[()]
    data_info["Nt"] = hf.get('Nt')[()]

    u_all = np.array(hf.get('u_all'))
    u_train = np.array(hf.get('u_train'))
    u_val = np.array(hf.get('u_val'))
    u_test = np.array(hf.get('u_test'))
    if data_info["SHUFFLE"]:
        idx_test = np.array(hf.get('idx_test')).tolist()
        idx_unshuffle = np.array(hf.get('idx_unshuffle')).tolist()
    else:
        idx_test = None
        idx_unshuffle = None
    if data_info["REMOVE_MEAN"]:
        u_mean_all = np.array(hf.get('u_mean_all'))
        u_mean_train = np.array(hf.get('u_mean_train'))
        u_mean_val = np.array(hf.get('u_mean_val'))
        u_mean_test = np.array(hf.get('u_mean_test'))
    else:
        u_mean_all = None
        u_mean_train = None
        u_mean_val = None
        u_mean_test = None
    hf.close()
    return u_all, (u_train, u_val, u_test), (idx_test, idx_unshuffle), (u_mean_all, u_mean_train, u_mean_val, u_mean_test), data_info


def train_autoencder(mdl:Model, 
                    data:DataTuple, 
                    batch_size:int, 
                    epochs:int, 
                    callback:list=[],
                    save_model_to:StrOrPath='./temp_md_autoencoder.h5', 
                    history:DataTuple=None,):

    '''Train a tensorflow autoencoder 

    Using the defined batch size for the defined number of epochs. 

    Arguments:
        mdl: a compiled autoencoder model
        data: (u_train, u_val, u_test)
        batch_size: size of each batch
        epochs: number of epochs
        callbacks: a list of keras callback
        save_model_to: '.h5' file to save the weights
        history: tupe of lists (hist_train, hist_val). Use empty lists if not given
    Returns:
        hist_train: a list of training losses, length equals to the numbe of epochs
        hist_val: a list of validation losses
        mse_test: mean squared error of u_test and predicted u_test
    '''
    start_time = datetime.datetime.now().strftime("%H:%M")

    u_train = data[0] # unpack data
    u_val = data[1]
    u_test = data[2]

    if history: # start fresh if history is not provided
        hist_train = history[0]
        hist_val = history[1]
    else:
        hist_train = []
        hist_val = []
        

    hist0 = mdl.fit(np.squeeze(u_train,axis=0), np.squeeze(u_train,axis=0),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(np.squeeze(u_val,axis=0), np.squeeze(u_val,axis=0)),
                    callbacks=callback,
                    verbose=0)
    mdl.load_weights(save_model_to)
    hist_train.extend(hist0.history['loss'])
    hist_val.extend(hist0.history['val_loss'])
    mse_test= mse()(np.squeeze(u_test,axis=0), mdl.predict(np.squeeze(u_test,axis=0))).numpy()

    finish_time = datetime.datetime.now().strftime("%H:%M")
    time_info = [start_time,finish_time]

    return hist_train, hist_val, mse_test, time_info


def set_gpu(gpu_id, memory_limit:typing.Optional[int]=None, MEMORY_GROWTH=False) -> None:
        '''Initialise gpu
        
        Set create a device for tensorflow on the assigned gpu with the required\n amount of memory. This function does nothing if tensorflow cannot find any\n available gpu on the computer.
        
        Arguments:
            gpu_id: int, which gpu. Use [] for cpu. 
            memory_limit: int or None, how much memory to allocate in MB. \nIf None tensorflow will map all memory. This argument is ignored if \nMEMORY_GROWTH is True.
            MEMORY_GROWTH: if True, allow memory growth.
        '''

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
                if MEMORY_GROWTH:
                    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                elif isinstance(memory_limit,int):
                    tf.config.set_logical_device_configuration(gpus[gpu_id], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]) 
                # set hard memory limit
                print('this process will run on gpu %i'%gpu_id)
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)



# base class for training a network from a config file
class TrainNN_from_config:
    '''Train a network from the .ini config file. 
    
    To complete training, the child class need methods; \n
        1. make_model: make and compile a tensorflow model \n
        2. train_model: train the tensorflow model \n
        3. log_wandb: log information with wandb \n
    '''
    def __init__(self, training_file:StrOrPath, gpu_id:typing.Optional[int]=None) -> None:
        # import parameters as class attribute
        training_param = self.read_training_config(training_file)
        self.import_config(training_param)


    @staticmethod
    def read_training_config(f:str) -> configparser.ConfigParser:
        '''Read 'train_mp_MD_AE.ini' and import relavant modules.'''
        config_training = read_config_ini(f)
        return config_training

    def import_config(self,training_param:configparser.ConfigParser) -> None:
        self.LATENT_STATE = training_param['training'].getboolean('LATENT_STATE')
        self.SHUFFLE = training_param['training'].getboolean('SHUFFLE')
        self.REMOVE_MEAN = training_param['training'].getboolean('REMOVE_MEAN')
        self.nb_epoch = training_param['training'].getint('nb_epoch')
        self.batch_size = training_param['training'].getint('batch_size')
        self.learning_rate = training_param['training'].getfloat('learning_rate')
        self.loss = training_param['training']['loss']

        self.latent_dim = training_param['ae_config'].getint('latent_dim')
        self.lmb = json.loads(training_param['ae_config']['lmb'])
        self.drop_rate = training_param['ae_config'].getfloat('drop_rate')
        self.features_layers = json.loads(training_param['ae_config']['features_layers'])
        self.act_fct = training_param['ae_config']['act_fct']
        self.resize_meth = training_param['ae_config']['resize_meth']
        self.filter_window = tuple(json.loads(training_param['ae_config']['filter_window']))
        self.BATCH_NORM = training_param['ae_config'].getboolean('BATCH_NORM')
        self.NO_BIAS = training_param['ae_config'].getboolean('NO_BIAS')

        self.read_data_file = training_param['data']['read_data_file']
        self.RAW_DATA = training_param['data'].getboolean('RAW_DATA')
        self.Nz = training_param['data'].getint('Nz')
        self.Ny = training_param['data'].getint('Ny')
        self.Nu = training_param['data'].getint('Nu')
        self.Nt = training_param['data'].getint('Nt')
        self.Ntrain = training_param['data'].getint('Ntrain')
        self.Nval = training_param['data'].getint('Nval')
        self.Ntest = training_param['data'].getint('Ntest')
        self.data_type = training_param['data']['data_type']

        self.LOG_WANDB = training_param['wandb'].getboolean('LOG_WANDB')
        self.project_name = training_param['wandb']['project_name']
        self.group_name = training_param['wandb']['group_name']

        self.save_to = training_param['system']['save_to']
        self.folder_prefix = training_param['system']['folder_prefix']

        self.Nx = [self.Ny,self.Nz]


    @staticmethod
    def set_gpu(gpu_id:int, memory_limit:int) -> None:
        '''Allocate the required memory to task on gpu'''
        set_gpu(gpu_id=gpu_id,memory_limit=memory_limit)
    
    def set_save_location(self,system_save_path:StrOrPath, folder_suffix:str) -> pathlib.Path:
        '''Make a folder to save results
        
        Folder_path is made by combining the system_save_path, folder_prefix and folder_suffix.
        Return folder_path.
        '''
        folder_name = self.folder_prefix + folder_suffix
        folder_path = os.path.join(system_save_path,self.save_to,folder_name)
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=False)
        return folder_path

    def set_wandb(self, wandb_config:dict, run_name:str) -> None:
        '''Initialise Weights and Biases
        
        initialise a wandb run with the provided wandb_config. This method does nothing if self.LOG_WANDB is False.

        Arguments:
            wandb_config: dict, config to pass to wandb.ini
            run_name: the name of the run, wandb.init(name=run_name)
        '''
        if self.LOG_WANDB:
            self.run = wandb.init(config=wandb_config, project=self.project_name, group=self.group_name, name=run_name)
    
    
    def get_data(self) -> np.ndarray:
        '''Read data from file

        If the data file given is already partitioned, the class attribute will be \n modified by the data file. 
        
        Return:
            u_all, (u_train, u_val, u_test), (idx_test, idx_unshuffle), (u_mean_all, u_mean_train, u_mean_val, u_mean_test). 

        Data array will be NoneType if they do not exits with the chosen options.
        '''
        if self.RAW_DATA:
            u_all, data, data_shuffle, data_mean = data_partition(
                                self.read_data_file,
                                [self.Nt,self.Nz,self.Ny,self.Nu],
                                [self.Ntrain,self.Nval,self.Ntest],
                                SHUFFLE=self.SHUFFLE,
                                REMOVE_MEAN=self.REMOVE_MEAN,
                                data_type=self.data_type)
        else:
            u_all, data, data_shuffle, data_mean, data_info = read_data(self.read_data_file)
            self.SHUFFLE = data_info["SHUFFLE"]
            self.REMOVE_MEAN = data_info["REMOVE_MEAN"]
            self.Ntrain = data_info["Ntrain"]
            self.Nval = data_info["Nval"]
            self.Ntest = data_info["Ntest"]
            self.Nz = data_info["Nz"]
            self.Ny = data_info["Ny"]
            self.Nu = data_info["Nu"]
            self.Nt = data_info["Nt"]
            self.Nx = [self.Ny,self.Nz]

        return u_all, data, data_shuffle, data_mean