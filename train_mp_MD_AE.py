import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from contextlib import redirect_stdout, redirect_stderr
import multiprocessing as mp
import datetime
import os
import configparser
import json
import typing
import pathlib
import wandb
from shutil import copyfile
import h5py

from project_specific_utils import training
from MD_AE_tools.models import models_no_bias
from MD_AE_tools.models import models

strorpath = typing.Union[str, pathlib.Path]
data_tuple = tuple[np.ndarray]


def read_system_config(f):
    '''Read _system.ini'''

    config_system = configparser.ConfigParser()
    config_system.read(f)
    system_info = config_system['system_info']
    return system_info


def read_mp_config(f):
    '''Read configuration for job queue'''
    conifg_mp = configparser.ConfigParser()
    conifg_mp.read(f)
    return conifg_mp


class Train_MD_AE:
    ''' Train an MD-AE with multiprocessing

    The most methods need to be called explicitly, the process should be 
    set_folder_path -> set_gpu -> get_data -> make_model -> set_wandb -> train -> log_wandb
    '''
    def __init__(self, training_file:strorpath, gpu_id:int) -> None:
        
        # import parameters as class attribute
        training_param = self.read_training_config(training_file)
        self.import_config(training_param)

        self.Nx = [self.Ny,self.Nz]
    

    @staticmethod
    def read_training_config(f:str):
        '''Read 'train_mp_MD_AE.ini' and import relavant modules.'''
        config_training = configparser.ConfigParser()
        config_training.read(f)
        return config_training
    
    def import_config(self,training_param:configparser.ConfigParser) -> None:
        self.LATENT_STATE = training_param['training'].getboolean('LATENT_STATE')
        self.SHUFFLE = training_param['training'].getboolean('SHUFFLE')
        self.REMOVE_MEAN = training_param['training'].getboolean('TRUE')
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

    @staticmethod
    def set_gpu(gpu_id:int,memory_limit:int) -> None:
        '''Allocate the required memory to task on gpu'''
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
                tf.config.set_logical_device_configuration(gpus[gpu_id],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]) 
                    # set hard memory limit
                print('this process will run on gpu %i'%gpu_id)
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
    
    def set_save_location(self,system_save_path:strorpath, ident:int=mp.current_process().ident) -> strorpath:
        folder_name = self.folder_prefix + str(ident)
        parent_folder = os.path.join(system_save_path,self.save_to)
        folder_path = os.path.join(system_save_path,self.save_to,folder_name)
        if not os.path.exists(os.path.join(system_save_path,self.save_to)):
            os.mkdir(parent_folder)
        os.mkdir(folder_path)
        return folder_path

    def make_model(self) -> None:
        if self.NO_BIAS:
            print('The model has no bias.')
            MD_Autoencoder = models_no_bias.MD_Autoencoder
        else:
            MD_Autoencoder = models.MD_Autoencoder

        self.md_ae = MD_Autoencoder(Nx=self.Nx,Nu=self.Nu,
                            features_layers=self.features_layers,
                            latent_dim=self.latent_dim,
                            filter_window=self.filter_window,
                            act_fct=self.act_fct,
                            batch_norm=self.BATCH_NORM,
                            drop_rate=self.drop_rate,
                            lmb=self.lmb,
                            resize_meth=self.resize_meth)
        self.md_ae.compile(optimizer=Adam(learning_rate=self.learning_rate),loss=self.loss)

    def set_wandb(self,ident:int=mp.current_process().ident) -> None:
        if self.LOG_WANDB:
            config_wandb = {'features_layers':self.features_layers,
                    'latent_dim':self.latent_dim,
                    'filter_window':self.filter_window,
                    'batch_size':self.batch_size, 
                    "learning_rate":self.learning_rate, 
                    "dropout":self.drop_rate, 
                    "activation":self.act_fct, 
                    "regularisation":self.lmb, 
                    "batch_norm":self.BATCH_NORM, 
                    'REMOVE_MEAN':self.REMOVE_MEAN}
            run_name = str(self.latent_dim)+"-mode-"+str(ident)
            self.run = wandb.init(config=config_wandb,project=self.project_name,group=self.group_name,name=run_name)

    def log_wandb(self, hist_train:np.ndarray, hist_val:np.ndarray, mse_test:np.ndarray) -> None:
        if self.LOG_WANDB:
            with self.run:
                for epoch in range(len(hist_train)):
                            self.run.log({"loss_train":hist_train[epoch], "loss_val":hist_val[epoch],"loss_test(mse)":mse_test})


    def get_data(self) -> np.ndarray:
        if self.RAW_DATA:
            u_all, data, data_shuffle, data_mean = training.data_partition(
                                self.read_data_file,
                                [self.Nt,self.Nz,self.Ny,self.Nu],
                                [self.Ntrain,self.Nval,self.Ntest],
                                SHUFFLE=self.SHUFFLE,
                                REMOVE_MEAN=self.REMOVE_MEAN,
                                data_type=self.data_type)
        else:
            u_all, data, data_shuffle, data_mean = training.read_data(self.read_data_file)
        return u_all, data, data_shuffle, data_mean
        

    def train(self,temp_file:strorpath,data:np.ndarray) -> np.ndarray:
        hist_train, hist_val, mse_test, time_info = training.train_autoencder(self.md_ae, data, self.batch_size, self.nb_epoch, save_model_to=temp_file)
        return hist_train, hist_val, mse_test, time_info

    def test_network(self, u_train:np.ndarray, u_test:np.ndarray) -> np.ndarray:
        encoder = self.md_ae.encoder
        decoders = self.md_ae.get_decoders()
        if self.LATENT_STATE:
            coded_train = encoder.predict(np.squeeze(u_train,axis=0))#(time,mode)
            mode_train = []
            for i in range(0,self.latent_dim):
                z = coded_train[:,i]
                z = np.reshape(z,(-1,1))
                mode_train.append(decoders[i].predict(z))
            y_train = np.sum(mode_train,axis=0)
            coded_test = encoder.predict(np.squeeze(u_test,axis=0))
            mode_test = []
            for i in range(0,self.latent_dim):
                z = coded_test[:,i]
                z = np.reshape(z,(-1,1))
                mode_test.append(decoders[i].predict(z))
            y_test = np.sum(mode_test,axis=0)
            # test if results are the same
            y_test_one = self.md_ae.predict(np.squeeze(u_test,axis=0))
            the_same = np.array_equal(np.array(y_test),np.array(y_test_one))
            print('Are results calculated the two ways the same. ', the_same)
            mode_train = np.array(mode_train)
            mode_test = np.array(mode_test)
        else:
            y_train = self.md_ae.predict(np.squeeze(u_train,axis=0))
            y_test = self.md_ae.predict(np.squeeze(u_test,axis=0))
            coded_test = None
            coded_train = None
            mode_train = None
            mode_test = None

        return (y_train, y_test), (coded_train, coded_test), (mode_train, mode_test)
    


    def post_processing(self, folder_path:strorpath, data:data_tuple, data_mean:data_tuple, coded:data_tuple, mode_packed:data_tuple, u_all:np.ndarray, data_shuffle:data_tuple, y:data_tuple, hist:data_tuple) -> None:
        '''Post processing results
        
        Save results and plots to the given folder.
        '''

        # unpack
        u_train, u_val, u_test = data
        u_mean_all, u_mean_train, u_mean_val, u_mean_test = data_mean
        coded_train, coded_test = coded
        mode_train, mode_test = mode_packed
        idx_test, idx_unshuffle = data_shuffle
        y_train, y_test = y
        hist_train, hist_val = hist


        # summary of structure
        filename = os.path.join(folder_path,'Autoencoder_summary.txt')
        with open(filename,'w') as f:
            with redirect_stdout(f):
                print('Autoencoder')
                print(self.md_ae.summary(),)
                print('\nEncoder')
                print(self.md_ae.encoder.summary())
                print('\nDecoder')
                print(self.md_ae.get_decoders()[0].summary())
        
        # model parameters
        filename = os.path.join(folder_path,'Model_param.h5')
        hf = h5py.File(filename,'w')
        hf.create_dataset('Ny',data=self.Ny)
        hf.create_dataset('Nz',data=self.Nz)
        hf.create_dataset('Nu',data=self.Nu)
        hf.create_dataset('features_layers',data=self.features_layers)
        hf.create_dataset('latent_dim',data=self.latent_dim)
        hf.create_dataset('resize_meth',data=np.string_(self.resize_meth),dtype="S10")
        hf.create_dataset('filter_window',data=self.filter_window)
        hf.create_dataset('act_fct',data=np.string_(self.act_fct),dtype="S10")
        hf.create_dataset('batch_norm',data=bool(self.BATCH_NORM))
        hf.create_dataset('drop_rate',data=self.drop_rate)
        hf.create_dataset('lmb',data=self.lmb)
        hf.create_dataset('LATENT_STATE',data=bool(self.LATENT_STATE))
        hf.create_dataset('SHUFFLE',data=bool(self.SHUFFLE))
        hf.create_dataset('REMOVE_MEAN',data=bool(self.REMOVE_MEAN))
        if self.SHUFFLE:
            hf.create_dataset('idx_unshuffle',data=idx_unshuffle) # fpr un-shuffling u_all[0:Ntrain+Nval,:,:,:]
        hf.close()

        # model weights
        filename = os.path.join(folder_path,'md_ae_model.h5')
        self.md_ae.save_weights(filename)

        # save results
        filename = os.path.join(folder_path,'results.h5')
        hf = h5py.File(filename,'w')
        hf.create_dataset('u_all',data=u_all[0,:,:,:,:])
        hf.create_dataset('hist_train',data=np.array(hist_train))
        hf.create_dataset('hist_val',data=hist_val)
        hf.create_dataset('u_train',data=u_train[0,:,:,:,:]) #u_train_fluc before
        hf.create_dataset('u_val',data=u_val[0,:,:,:,:])
        hf.create_dataset('u_test',data=u_test[0,:,:,:,:])
        hf.create_dataset('y_test',data=y_test)
        hf.create_dataset('y_train',data=y_train)
        if self.REMOVE_MEAN:
            hf.create_dataset('u_avg',data=u_mean_all)
            hf.create_dataset('u_avg_train',data=u_mean_train)
            hf.create_dataset('u_avg_val',data=u_mean_val)
            hf.create_dataset('u_avg_test',data=u_mean_test)
        if self.LATENT_STATE:
            hf.create_dataset('latent_dim',data=self.latent_dim)
            hf.create_dataset('latent_train',data=coded_train)
            hf.create_dataset('latent_test',data=coded_test)
            hf.create_dataset('modes_train',data=mode_train)
            hf.create_dataset('modes_test',data=mode_test) # has shape (modes,snapshots,Nx,Ny,Nu)
        hf.close()

        fig_count = 0

        # training history
        path = os.path.join(folder_path,'training_history.png')
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
        if self.REMOVE_MEAN:
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
        path = os.path.join(folder_path,'autoencoder_results.png')
        fig_count = fig_count + 1
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
        if self.LATENT_STATE:
            fig_count = fig_count + 1
            t = 0
            figname = 'autoencoder mode at time ' + str(t) + '.png'
            path = os.path.join(folder_path,figname)
            fig, ax = plt.subplots(2,self.latent_dim,sharey='all')
            fig.suptitle('autoencoder modes')
            for u in range(2):
                for i in range(self.latent_dim):
                    im = ax[u,i].imshow(mode_train[i,t,:,:,u],'jet')
                    div = make_axes_locatable(ax[u,i])
                    cax = div.append_axes('right',size='5%',pad='2%')
                    plt.colorbar(im,cax=cax)
                    ax[u,i].set_xticks([])
                    ax[u,i].set_yticks([])
            for i in range(self.latent_dim):
                ax[0,i].set_title(str(i+1))
            ax[0,0].set_ylabel('v')
            ax[1,0].set_ylabel('w')
            plt.savefig(path)
        
        # plot latent variables
        fig_count = fig_count + 1
        path = os.path.join(folder_path,'latent_variables.png')
        plt.figure(fig_count)
        plt.plot(coded_test[:,0],label='1')
        plt.plot(coded_test[:,1],label='2')
        plt.legend()
        plt.ylabel('value of latent variable')
        plt.title("Testing autoencoder")
        plt.savefig(path)




def run_job(train_id):
    '''train one autoencoder'''
    gpu_id = queue.get() # get the next available gpu

    ident = mp.current_process().ident

    try:
        training_file = jobs[train_id]
        tempfn = './temp_md_autoencoder'+str(train_id)+'.h5'
        print('Running job No.%i, PID %i, from file %s, on gpu %i.'%(
                    train_id,ident,training_file,gpu_id))

        # initiate job
        job = Train_MD_AE(training_file, gpu_id)

        # save everything to this folder
        folder_path = job.set_save_location(system_info['alternate_location'],ident) 
        # save the configuration
        copyfile(training_file,os.path.join(folder_path,'training_param.ini'))

        # redirect output to files
        log_path = os.path.join(folder_path,'log')
        error_path = os.path.join(folder_path,'err')
        log_out = open(log_path,'w')
        err_out = open(error_path,'w')

        with redirect_stdout(log_out), redirect_stderr(err_out):
            job.set_gpu(gpu_id, memory_limit)
            u_all, data, data_shuffle, data_mean = job.get_data()

            # start training
            job.make_model()
            job.set_wandb(ident)
            hist_train, hist_val, mse_test, time_info = job.train(temp_file=tempfn,data=data)
            job.log_wandb(hist_train,hist_val,mse_test)
            print('\nTraining started at ', time_info[0], ', finished at ',time_info[1],'.')

            # unpack data
            u_train, _, u_test = data

            # test network
            y, coded, mode_packed = job.test_network(u_train,u_test) 
        
        log_out.close()
        err_out.close()

        job.post_processing(folder_path,data,data_mean,coded,mode_packed,u_all,data_shuffle,y,(hist_train,hist_val))

    finally:
        queue.put(gpu_id)




if __name__ == "__main__":

    multiprocessing_experiment_config = '_train_mp.ini'
    # ====================== read from config file ========================
    system_info = read_system_config('_system.ini')
    config_mp = read_mp_config(multiprocessing_experiment_config)

    which_gpus = json.loads(config_mp['system']['which_gpus'])
    n_task_gpu = config_mp['system'].getint('n_task_gpu')
    memory_limit = config_mp['system'].getint('memory_limit')
    OFFLINE_WANDB = config_mp['system'].getboolean('OFFLINE_WANDB')


    # check if gpu_id is valid
    if tf.config.list_physical_devices('GPU') and (len(which_gpus) > len(tf.config.list_physical_devices('GPU'))):
        raise ValueError('Assigned gpu_id out of range.')
    elif not tf.config.list_physical_devices('GPU'):
        print('No gpu found, run on cpu.')

    # ======================== set up multiprocessing ========================

    # wandb
    if OFFLINE_WANDB:
        os.environ["WANDB_API_KEY"] = config_mp['system']['wandb_api_key']
        os.environ["WANDB_MODE"] = "offline"
    wandb.require("service") # set up wandb for multiprocessing
    wandb.setup()

    # a list of jobs
    jobs_dict = dict(config_mp['jobs'])
    jobs = []
    for key in jobs_dict:
        for _ in range(int(jobs_dict[key])):
            jobs.append(key)

    # set up a queue of gpu
    queue = mp.Queue()
    for gpu_id in which_gpus: 
        for _ in range(n_task_gpu):
            queue.put(gpu_id)
    
    pool = mp.Pool(n_task_gpu) # how many task at the same time

    start_time = datetime.datetime.now().strftime("%H:%M")

    pool.map(run_job,range(len(jobs)))# send jobs
    # tensorflow problem? calling pool.close() and pool.join() fixed the AttributeError:'NoneType' object has no attribute 'dump'
    pool.close() 
    pool.join()
    
    finish_time = datetime.datetime.now().strftime("%H:%M")
    print(f"Program started at {start_time}, finished at {start_time}")