import sys
sys.path.append('..')
import tensorflow as tf

import matplotlib
matplotlib.use('agg')

import multiprocessing as mp
import datetime
import os
import json
import wandb
import subprocess

sys.path.append('..')
from project_specific_utils.helpers import read_config_ini

def run_job(one_job):
    '''train one autoencoder by spawning a subprocess'''
    train_id, config_file, python_file = one_job
    # train_id: int, config_file: str, python_file: str

    gpu_id = queue.get() # get the next available gpu

    try:
        subprocess.run(["python", python_file, str(train_id), config_file, str(gpu_id), str(memory_limit), save_to])

    finally:
        # put gpu back in the queue
        queue.put(gpu_id) 





if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%H:%M")

    multiprocessing_experiment_config = '_train_mp.ini'
    # ====================== read from config file ========================
    system_info = read_config_ini('../_system.ini')['system_info']
    save_to = system_info['alternate_location']
    config_mp = read_config_ini(multiprocessing_experiment_config)

    which_gpus = json.loads(config_mp['system']['which_gpus'])
    n_task_gpu = config_mp['system'].getint('n_task_gpu')
    memory_limit = config_mp['system'].getint('memory_limit')
    OFFLINE_WANDB = config_mp['system'].getboolean('OFFLINE_WANDB')

    CPU = False
    # check if gpu_id is valid
    if tf.config.list_physical_devices('GPU') and (len(which_gpus) > len(tf.config.list_physical_devices('GPU'))):
        raise ValueError('Assigned gpu_id out of range.')
    elif not tf.config.list_physical_devices('GPU'):
        CPU = True
        print('No gpu found, run on cpu.')

    # ======================== set up multiprocessing ========================

    # wandb
    if OFFLINE_WANDB:
        os.environ["WANDB_API_KEY"] = config_mp['system']['wandb_api_key']
        os.environ["WANDB_MODE"] = "offline"
    wandb.require("service") # set up wandb for multiprocessing
    wandb.setup()

    # a list of jobs
    # each job is (unique id, config file to use, python file to run)
    jobs_dict = dict(config_mp['jobs'])
    jobs = []
    train_id = 0
    for config_file in jobs_dict:
        values = jobs_dict[config_file]
        repeat, python_file = values.split(",",1)
        for _ in range(int(repeat)):
            train_id = train_id+1
            jobs.append((train_id,config_file,python_file))

    # set up a queue of gpu
    queue = mp.Queue()
    for gpu_id in which_gpus: 
        for _ in range(n_task_gpu):
            queue.put(gpu_id)
    
     # set up jobs for cpu or gpu
    if CPU:
        # if only using cpu, the n_task_gpu is the total number of tasks to be run on the cpu at the same time. 
        n_parallel_jobs = n_task_gpu
    else:
        n_parallel_jobs = n_task_gpu*len(which_gpus)

    pool = mp.Pool(n_parallel_jobs) # how many task at the same time

    # send jobs
    pool.map(run_job,jobs)
    # tensorflow problem? calling pool.close() and pool.join() fixed the AttributeError:'NoneType' object has no attribute 'dump'
    pool.close() 
    pool.join()
    
    finish_time = datetime.datetime.now().strftime("%H:%M")
    print(f"Program started at {start_time}, finished at {finish_time}")