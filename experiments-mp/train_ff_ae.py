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

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

from project_specific_utils import data_and_train
from project_specific_utils.ff_ae import Train_ff_ae



def main(train_id, training_config, gpu_id, memory_limit, save_to, CPU):
    pid = os.getpid()
    suffix = str(train_id) + '-' + str(pid)
    tempfn = './temp_ff_ae'+ str(pid) +'.h5'
    if CPU:
        message = 'cpu'
    else:
        message = 'gpu ' + str(gpu_id)
    print('Running job No.%i, PID %i, from file %s, on %s.'
            %(train_id,pid,training_config,message))
    
    job = Train_ff_ae(training_config)
    data_and_train.set_gpu(gpu_id,memory_limit)
    data = job.read_data()
    folder_path = job.set_save_location()
    job.make_model()
    run = job.set_wandb()
    job.train_model()
    loss_test, loss_image = job.get_reconstructed_loss()
    

    # save file to wandb
    # if loss_val improve    
    run.log({'loss_test(mse)':loss_test, 'loss_image':loss_image})
    run.save(tempfn)
    run.finish()


    time.sleep(30)
    os.remove(tempfn)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a feedforward autoencoder with either image or time coefficients.')
    parser.add_argument('id', type=int, help='a number of identify this training')
    parser.add_argument('training_config', help='path to the config file to use')
    parser.add_argument('gpu_id', type=int, help='which GPu to put this process on')
    parser.add_argument('memory_limit', type=int, help='How much GPU memory to allocate to this process, in MB.')
    parser.add_argument('save_to', help='path to the result folder')
    parser.add_argument('--cpu', action='store_true', help='number of available gpu on this computer')
    args = parser.parse_args()
    main(args.id, args.training_config, args.gpu_id, args.memory_limit, args.save_to, args.cpu)
