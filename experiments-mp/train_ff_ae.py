import sys
sys.path.append('..')

import typing
from pathlib import Path
import argparse
import os
# from shutil import copyfile
import h5py
import time

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

from project_specific_utils import data_and_train
from project_specific_utils.ff_ae import Train_ff_ae


def save_results(folder_path, job, loss_test, loss_image):

    hist = job.get_loss_history
    with h5py.File(Path(folder_path,'params.h5'),'w') as hf:
        if job.RAW_DATA:
            hf.create_dataset('random_seed',data=job.random_seed)
            if job.SHUFFLE:
                hf.create_dataset('idx_unshuffle',data=job.idx_unshuffle)
        hf.create_dataset('loss_train',data=hist['train'])
        hf.create_dataset('loss_val',data=hist['val'])
        hf.create_dataset('loss_test', data=loss_test)
        hf.create_dataset('loss_image', data=loss_image)
        
    job.save_weights(Path(folder_path,'weights.h5'))




def main(train_id, training_config, gpu_id, memory_limit, save_to, CPU):
    pid = os.getpid()
    suffix = str(train_id) + '-' + str(pid)
    tempfn = './temp_ff_ae'+ str(pid) +'.h5'
    if CPU:
        message = 'cpu'
    else:
        message = 'gpu ' + str(gpu_id)
        data_and_train.set_gpu(gpu_id,memory_limit)
    print('Running job No.%i, PID %i, from file %s, on %s.'
            %(train_id,pid,training_config,message))
    
    job = Train_ff_ae()
    job.import_config(training_file=training_config)

    data_train, data_val, data_test = job.read_data(random_seed=pid)
    folder_path = job.set_save_location(save_to,suffix)
    job.make_model()

    if job.LOG_WANDB:
        run = job.set_wandb()
    else:
        run = None
    job.train_model(
        data_train,
        data_val,
        checkpoint=True,
        temp_file=Path(tempfn),
        wandb_run=run,
        print_freq=100
    )

    loss_test, loss_image = job.get_reconstructed_loss(data_test)
    print(f'Testing loss is {loss_test}. Reconstructed loss is {loss_image}')
    

    # save file to wandb
    if job.LOG_WANDB:
        run.log({'loss_test(mse)':loss_test, 'loss_image':loss_image})
        run.save(training_config)
        run.tags = run.tags + ("repeat",)
        run.finish()
    save_results(folder_path,job,loss_test,loss_image)


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
