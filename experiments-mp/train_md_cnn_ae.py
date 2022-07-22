import sys
sys.path.append('..')

import typing
import pathlib
import argparse
import os
from shutil import copyfile
from contextlib import redirect_stderr, redirect_stdout
import h5py

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import project_specific_utils.data_and_train as training
from MD_AE_tools.models import models, models_no_bias
from project_specific_utils import md_cnn_ae as post_processing


StrOrPath = typing.Union[str,pathlib.Path]
dtype = typing.Union[str,np.dtype]
DataTuple = tuple[np.ndarray]
idx = typing.Union[int,list[int]]


class Train_md_cnn_ae(training.TrainNN_from_config):
    '''Tran a MD-CNN-AE
    
    use get_mdl to get model.
    '''
    def __init__(self, training_file: StrOrPath, gpu_id: typing.Optional[int] = None) -> None:
        super().__init__(training_file, gpu_id)

    def make_model(self):
        '''Make and compile a MD-CNN-AE model'''
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

    def train_model(self, temp_file:StrOrPath, data:DataTuple, callback:list):
        hist_train, hist_val, mse_test, time_info = training.train_autoencder(self.md_ae, data, self.batch_size, self.nb_epoch, callback, save_model_to=temp_file)
        return hist_train, hist_val, mse_test, time_info

    def log_wandb(self, hist_train:typing.Iterable, hist_val:typing.Iterable, mse_test:typing.Iterable) -> None:
        '''Log information to weights and biases
        
        This function does nothing if self.LOG_WANDB is false.
        '''
        if self.LOG_WANDB:
            with self.run:
                for epoch in range(len(hist_train)):
                            self.run.log({"loss_train":hist_train[epoch], "loss_val":hist_val[epoch],"loss_test(mse)":mse_test})

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

    
    @property
    def get_mdl(self):
        return self.md_ae

    def get_wandb_config(self):
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
        return config_wandb




def save_results(training_class:Train_md_cnn_ae, 
                    folder_path:StrOrPath, 
                    u_all:np.ndarray, 
                    data:DataTuple, 
                    data_mean:DataTuple, 
                    coded:DataTuple, 
                    mode_packed:DataTuple, 
                    data_shuffle:DataTuple, 
                    y:DataTuple, 
                    hist:DataTuple):
    '''Write results to .h5 files'''
    # unpack
    u_train, u_val, u_test = data
    u_mean_all, u_mean_train, u_mean_val, u_mean_test = data_mean
    coded_train, coded_test = coded
    mode_train, mode_test = mode_packed
    _, idx_unshuffle = data_shuffle
    y_train, y_test = y
    hist_train, hist_val = hist

    # model parameters
    filename = os.path.join(folder_path,'Model_param.h5')
    with h5py.File(filename,'w') as hf:
        hf.create_dataset('Ny',data=training_class.Ny)
        hf.create_dataset('Nz',data=training_class.Nz)
        hf.create_dataset('Nu',data=training_class.Nu)
        hf.create_dataset('features_layers',data=training_class.features_layers)
        hf.create_dataset('latent_dim',data=training_class.latent_dim)
        hf.create_dataset('resize_meth',data=np.string_(training_class.resize_meth),dtype="S10")
        hf.create_dataset('filter_window',data=training_class.filter_window)
        hf.create_dataset('act_fct',data=np.string_(training_class.act_fct),dtype="S10")
        hf.create_dataset('batch_norm',data=bool(training_class.BATCH_NORM))
        hf.create_dataset('drop_rate',data=training_class.drop_rate)
        hf.create_dataset('lmb',data=training_class.lmb)
        hf.create_dataset('LATENT_STATE',data=bool(training_class.LATENT_STATE))
        hf.create_dataset('SHUFFLE',data=bool(training_class.SHUFFLE))
        hf.create_dataset('REMOVE_MEAN',data=bool(training_class.REMOVE_MEAN))
        if training_class.SHUFFLE:
            hf.create_dataset('idx_unshuffle',data=idx_unshuffle) # fpr un-shuffling u_all[0:Ntrain+Nval,:,:,:]
    

    # model weights
    filename = os.path.join(folder_path,'md_ae_model.h5')
    mdl = training_class.get_mdl
    mdl.save_weights(filename)

    # summary of structure
    filename = os.path.join(folder_path,'Autoencoder_summary.txt')
    with open(filename,'w') as f:
        with redirect_stdout(f):
            print('Autoencoder')
            print(mdl.summary(),)
            print('\nEncoder')
            print(mdl.encoder.summary())
            print('\nDecoder')
            print(mdl.get_decoders()[0].summary())

     # save results
    filename = os.path.join(folder_path,'results.h5')
    with h5py.File(filename,'w') as hf:
        hf.create_dataset('u_all',data=u_all)
        hf.create_dataset('hist_train',data=np.array(hist_train))
        hf.create_dataset('hist_val',data=hist_val)
        hf.create_dataset('u_train',data=u_train) 
        hf.create_dataset('u_val',data=u_val)
        hf.create_dataset('u_test',data=u_test)
        hf.create_dataset('y_test',data=y_test)
        hf.create_dataset('y_train',data=y_train)
        if training_class.REMOVE_MEAN:
            hf.create_dataset('u_avg',data=u_mean_all)
            hf.create_dataset('u_avg_train',data=u_mean_train)
            hf.create_dataset('u_avg_val',data=u_mean_val)
            hf.create_dataset('u_avg_test',data=u_mean_test)
        if training_class.LATENT_STATE:
            hf.create_dataset('latent_train',data=coded_train)
            hf.create_dataset('latent_test',data=coded_test)
            hf.create_dataset('modes_train',data=mode_train)
            hf.create_dataset('modes_test',data=mode_test) # has shape (modes,snapshots,Nx,Ny,Nu)


def plot_results(folder_path:StrOrPath, 
                    data:DataTuple, 
                    data_mean:DataTuple, 
                    coded:DataTuple, 
                    mode_packed:DataTuple, 
                    y:DataTuple, 
                    hist:DataTuple,
                    decoders:list, 
                    latent_dim:int,
                    act_fct:int):
    '''Plot results
    
    folder_path: path to the result folder
    data: (u_train, u_val, u_test)
    data_mean: (u_mean_all, u_mean_train, u_mean_val, u_mean_test)
    coded: the latent variables, (train, test)
    y: predicted velocity, (y_train, y_test)
    hist: training history, (hist_train, hist_val)
    '''

     # unpack
    u_train, u_val, u_test = data
    u_mean_all, u_mean_train, u_mean_val, u_mean_test = data_mean
    coded_train, coded_test = coded
    mode_train, mode_test = mode_packed
    y_train, y_test = y
    hist_train, hist_val = hist
    
    # training history
    post_processing.plot_training_history(hist_train,hist_val,savefig=True,folder_path=folder_path)

    # autoencoder results
    post_processing.plot_ae_results(np.squeeze(u_train),np.squeeze(y_train),u_mean_train,error='mae',savefig=True,folder_path=folder_path)

    # modes
    post_processing.plot_autoencoder_modes(latent_dim,mode_train,t=0,savefig=True,folder_path=folder_path)

    # latent variables
    post_processing.plot_latent_variable(coded_test,savefig=True,folder_path=folder_path,figtitle='latent variables of the testing set')

    # decoder weights
    post_processing.plot_decoder_weights(act_fct, decoders, savefig=True, folder_path=folder_path)
    


    

def main(train_id:int, training_config:str, gpu_id:int, memory_limit:int, save_to:str, CPU:bool):
    pid = os.getpid()
    suffix = str(train_id) + '-' + str(pid)
    tempfn = './temp_md_autoencoder'+ str(pid) +'.h5'
    if CPU:
        message = 'cpu'
    else:
        message = 'gpu ' + str(gpu_id)
    print('Running job No.%i, PID %i, from file %s, on %s.'
            %(train_id,pid,training_config,message))

    # initiate job
    job = Train_md_cnn_ae(training_config, gpu_id)
    job.set_gpu

    # set up callbacks
    model_cb=ModelCheckpoint(tempfn, monitor='loss',save_best_only=True,verbose=1,save_weights_only=True)#'val_loss or loss
    early_cb=EarlyStopping(monitor='loss', patience=200,verbose=1)
    cb = [model_cb, early_cb]

    # save everything to this folder
    folder_path = job.set_save_location(save_to,suffix) 
    # save the configuration
    copyfile(training_config,os.path.join(folder_path,'training_param.ini'))

    # redirect output to files
    log_path = os.path.join(folder_path,'log')
    error_path = os.path.join(folder_path,'err')
    log_out = open(log_path,'w')
    err_out = open(error_path,'w')
    with redirect_stdout(log_out), redirect_stderr(err_out):
        job.set_gpu(gpu_id,memory_limit)
        u_all, data, data_shuffle, data_mean = job.get_data()
        job.make_model()
        wandb_config = job.get_wandb_config()
        run_name = str(job.latent_dim)+"-mode-"+ suffix
        job.set_wandb(wandb_config, run_name)
        hist_train, hist_val, mse_test, time_info = job.train_model(temp_file=tempfn,data=data, callback=cb)
        job.log_wandb(hist_train,hist_val,mse_test)
        print('\nTraining started at ', time_info[0], ', finished at ',time_info[1],'.')
        
        # unpack data
        u_train, _, u_test = data
        y, coded, mode_packed = job.test_network(u_train,u_test) 
    log_out.close()
    err_out.close()

    save_results(job, 
                folder_path, 
                u_all, 
                data, 
                data_mean, 
                coded, 
                mode_packed, 
                data_shuffle, 
                y, 
                (hist_train,hist_val))


    plot_results(folder_path, 
                data,data_mean, 
                coded, 
                mode_packed, 
                y, 
                (hist_train,hist_val), 
                job.get_mdl.get_decoders(), 
                job.latent_dim, 
                job.act_fct)    

    os.remove(tempfn)



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train and post process a md-cnn-ae from a config file.')
    parser.add_argument('id', type=int, help='a number of identify this training')
    parser.add_argument('training_config', help='path to the config file to use')
    parser.add_argument('gpu_id', type=int, help='which GPu to put this process on')
    parser.add_argument('memory_limit', type=int, help='How much GPU memory to allocate to this process, in MB.')
    parser.add_argument('save_to', help='path to the result folder')
    parser.add_argument('--cpu', action='store_true', help='number of available gpu on this computer')
    args = parser.parse_args()
    main(args.id, args.training_config, args.gpu_id, args.memory_limit, args.save_to, args.cpu)
