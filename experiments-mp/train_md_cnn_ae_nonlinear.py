import sys
sys.path.append('..')

import typing
import pathlib
import argparse
import os
from shutil import copyfile
from contextlib import redirect_stderr, redirect_stdout
import h5py
import time
import datetime
import matplotlib
matplotlib.use('agg')

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from project_specific_utils import md_cnn_ae as post_processing
from project_specific_utils.md_cnn_ae import Train_md_cnn_ae


StrOrPath = typing.Union[str,pathlib.Path]
dtype = typing.Union[str,np.dtype]
DataTuple = tuple[np.ndarray]
idx = typing.Union[int,list[int]]



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
                    latent_dim:int):
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
    post_processing.plot_latent_variable(coded_test,savefig=True,folder_path=folder_path,figtitle='latent_variables_of_the testing_set')

    


    

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

        hist_train = []
        hist_val = []
        start_time = datetime.datetime.now().strftime("%H:%M")
        for _ in range(4):
            hist_train1, hist_val1, mse_test, _ = job.train_model(temp_file=tempfn,data=data, callback=cb)
            hist_train.extend(hist_train1)
            hist_val.extend(hist_val1)
        finish_time = datetime.datetime.now().strftime("%H:%M")

        job.log_wandb(hist_train,hist_val,mse_test)
        print('\nTraining started at ', start_time, ', finished at ',finish_time,'.')
        
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
                job.latent_dim)    

    time.sleep(30)

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
