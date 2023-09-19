import argparse
import h5py
import wandb
import os
import time
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
mse = tf.keras.losses.MeanSquaredError()
import MD_AE_tools.models.models_ff as modelff

from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from project_specific_utils.data_and_train import set_gpu



def train_and_save(config, p_train, folder, lrschedule):

    ae = modelff.Autoencoder(
        input_shape = p_train.shape[1],
        encoder_layers = config['encoder_layers'],
        decoder_layers = [64,128,256,256,128],
        latent_dim = config['latent_dim'],
        act_fct = config['act_fct'],
        batch_norm = config['batch_norm'],
        drop_rate = config['dropout'],
        lmb = config['regularisation']
    )


    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrschedule),loss='mse')
    _ = ae.evaluate(p_train,p_train,batch_size=config['batch_size'])
    ae.load_weights(Path('_results/training/Nz2-1348382-repeat1/weights.h5'))
    print('Previous weights loaded.')

    z = ae.encoder.predict(p_train, batch_size=config['batch_size'])


    
    time.sleep(10)


    del ae


    f = Path(folder,'weights.h5')

    model_cb = ModelCheckpoint(
        f,
        monitor='loss',
        save_best_only=True,
        verbose=0,
        save_weights_only=True
    )
    cb = [model_cb]

    decoder = modelff.Decoder(
        output_shape=p_train.shape[1],
        layer_sizes=config['decoder_layers'],
        latent_dim=config['latent_dim'],
        act_fct=config['act_fct'],
        batch_norm=config['batch_norm'],
        drop_rate=config['dropout'],
        lmb=config['regularisation']
    )
    decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrschedule),loss='mse')
    hist = decoder.fit(
        z,
        p_train,
        epochs=config['nb_epoch'],
        batch_size=config['batch_size'],
        shuffle=True,
        callbacks=cb,
        verbose=2
    )

    loss_list = hist.history['loss']
    best_loss_list = np.zeros_like(loss_list)
    current_best_loss = np.inf
    for i in range(config['nb_epoch']):
        if loss_list[i] < current_best_loss:
            current_best_loss = loss_list[i]
        best_loss_list[i] = current_best_loss

    with h5py.File(Path(folder,'results.h5'),'x') as hf:
        hf.create_dataset('latent_dim', data=config['latent_dim'])
        hf.create_dataset('loss_best', data=best_loss_list)
        hf.create_dataset('loss_history', data=loss_list)

    return loss_list, best_loss_list



def main(args):

    if args.gpu_id is not None:
        set_gpu(args.gpu_id,args.memory_limit)
    if args.wandb_mode == 'offline':
        os.environ["WANDB_MODE"] = "offline"

    save_to = Path(args.save_to)    
    if not save_to.exists():
        print('Making result directory.')
        save_to.mkdir()

    repeats = args.repeats
    pid = os.getpid()
    
    # ======== hyperparameters=============
    
    nb_epoch = args.nb_epoch
    latent_dim = args.latent_dim
    batch_norm = True
    batch_size = 300000
    dropout = 0.014
    learning_rate = 0.0022
    regularisation = 0.00003
    act_fct = 'tanh'
    encoder_layers = [128,256,256,128,64]
    decoder_layers = [64,256,256,256,256,256]

    lrschedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        learning_rate,
        100,
        t_mul=50,
        m_mul=0.9
    )

    config = {
        'nb_epoch': nb_epoch,
        'latent_dim': latent_dim,
        'batch_norm': batch_norm,
        'batch_size': batch_size,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'regularisation': regularisation,
        'act_fct': act_fct,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers
    }

    # ========== data =====================

    with h5py.File('./data/raw_pressure_long.h5','r') as hf:
        # fs = np.squeeze(hf.get('fs'))
        # static_p = np.squeeze(hf.get('static_p'))
        esp_allt = np.array(hf.get('esp')).T
        r = np.array(hf.get('r')).T
        theta = np.array(hf.get('theta')).T
    x=(np.cos(theta*np.pi/180).T)*r
    y=(np.sin(theta*np.pi/180).T)*r 
    x = x.flatten()
    y = y.flatten()

    pmean = np.mean(esp_allt,axis=1).reshape(8,8)
    # prms = np.std(esp_allt,axis=1)


    p_train = esp_allt - pmean.flatten()[:,np.newaxis]
    p_train = p_train.T


    # =============== Train ================
    for i in range(repeats):

        name = f'decoderonly-Nz{latent_dim}-{pid}-repeat{i}'
        folder = Path(save_to,name)
        print(f'Running repeat {i+1}, saving to {str(folder)}.')
        folder.mkdir(exist_ok=False)

        run = wandb.init(config=config, project='POD_and_AE',group='decompose_pressure', reinit=True, name=name, tags=('decoder_only',))

        loss_list, best_loss_list = train_and_save(config, p_train, folder, lrschedule)

        for i in range(nb_epoch):
            run.log({'loss':loss_list[i], 'current_best_loss':best_loss_list[i]})

        run.finish()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an MLP network to decompose experimental pressure data.')
    parser.add_argument('--nb_epoch', type=int, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, help='Number of latent variables to use.')
    parser.add_argument('--gpu_id', type=int, help='Which gpu')
    parser.add_argument('--memory_limit', type=int, help='How much GPU memory to allocate in MB.')
    parser.add_argument('--save_to', help='Path to the result folder.')
    parser.add_argument('--repeats', type=int, default=1, help='Number of repeats to do.')
    parser.add_argument('--wandb_mode', type=str, help="'offline' for using wandb offline." )

    args = parser.parse_args()
    print(args)

    main(args)