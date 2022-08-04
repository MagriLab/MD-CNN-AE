''' Collection of functions for training a standard autoencoder

Include a class for multiprocessing training and functions for post processing.'''

import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from MD_AE_tools.models import models, models_no_bias

from . import data_and_train as training

import typing
from typing import Optional
import pathlib
StrOrPath = typing.Union[str,pathlib.Path]
DataTuple = tuple[np.ndarray]
DataList = typing.Union[np.ndarray,list]


class TrainAE(training.TrainNN_from_config):
    '''train a standard autoencoder with multiprocessing
    
    use .get_mdl to get model.
    '''
    def __init__(self, training_file: StrOrPath, gpu_id: typing.Optional[int] = None) -> None:
        super().__init__(training_file, gpu_id)

    def make_model(self):
        '''Make and compile a standard autoencoder.
        
        Creates a model self.ae.
        '''
        if self.NO_BIAS:
            print('This model has no bias')
            Autoencoder = models_no_bias.Autoencoder
        elif self.NO_BIAS is False:
            Autoencoder = models.Autoencoder
        else:
            raise ValueError('Please set the NO_BIAS flag to true or false.')
        
        self.ae = Autoencoder(Nx=self.Nx,
                                Nu=self.Nu,
                                features_layers=self.features_layers,
                                latent_dim=self.latent_dim,
                                filter_window=self.filter_window,
                                act_fct=self.act_fct,
                                batch_norm=self.BATCH_NORM,
                                drop_rate=self.drop_rate,
                                lmb=self.lmb,
                                resize_meth=self.resize_meth)
        self.ae.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)

    def train_model(self, temp_file:StrOrPath, data:DataTuple, callback:list):
        '''Train the model.
        
        Arguments:
            temp_file: temp .h5 file to store weights
            data: (u_train, u_val, u_test), the data array should be a numpy.ndarray
        '''
        hist_train, hist_val, mse_test, time_info = training.train_autoencder(self.ae, data, self.batch_size, self.nb_epoch, callback, save_model_to=temp_file)
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
        encoder = self.ae.encoder
        decoder = self.ae.decoder
        if self.LATENT_STATE:
            coded_train = encoder.predict(np.squeeze(u_train,axis=0)) #(time,mode)
            mode_train = []
            for i in range(self.latent_dim):
                temp_z = np.zeros_like(coded_train)
                temp_z[:,[i]] = coded_train[:,[i]]
                mode_train.append(decoder.predict(temp_z))
            y_train = decoder.predict(coded_train)

            coded_test = encoder.predict(np.squeeze(u_test,axis=0)) #(time,mode)
            mode_test = []
            for i in range(self.latent_dim):
                temp_z = np.zeros_like(coded_test)
                temp_z[:,[i]] = coded_test[:,[i]]
                mode_test.append(decoder.predict(temp_z))
            y_test = decoder.predict(coded_test)
            mode_train = np.array(mode_train)
            mode_test = np.array(mode_test)
        else:
            y_train = self.ae.predict(np.squeeze(u_train,axis=0))
            y_test = self.ae.predict(np.squeeze(u_test,axis=0))
            coded_test = None
            coded_train = None
            mode_train = None
            mode_test = None

        return (y_train, y_test), (coded_train, coded_test), (mode_train, mode_test)

    @property
    def get_mdl(self):
        return self.ae

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




# ========================== Plots =================================

def plot_training_history(hist_train:DataList, 
                            hist_val:DataList,
                            savefig:bool=False, 
                            folder_path:Optional[StrOrPath]=None):
    '''Plot training history (training & validation).
    
    Arguments: \n
        hist_train: List or array, training loss at every epoch.\n
        hist_val: list or array, validation loss at every epoch.\n
        savefig: if true, figure will be saved and not showed.\n
        folder_path: path to the folder where the figure will be saved.
    '''
    fig = plt.figure(clear=True)
    plt.plot(hist_train,label="training")
    plt.plot(hist_val,label="validation")
    plt.title("Training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if savefig:
        path = pathlib.Path(folder_path,'training_history.png')
        fig.savefig(path)
    else:
        fig.show()


def plot_ae_results(u:np.ndarray,
                    y:np.ndarray,
                    u_avg:typing.Optional[np.ndarray]=None,
                    error:str='mae',
                    savefig:bool=False,
                    folder_path:Optional[StrOrPath]=None):
    '''

    Arguments:
        u: input to the autoencoder, shape [time,Ny,Nz,u]\n
        y: output from ae.predict(u)\n
        u_avg: the time-averaged component if the network was trained on fluctuating velocity\n
        error: the error function to plot. 'mse' or 'mae' '''

    if u.ndim != 4:
        raise ValueError("Input shape must be [nt,ny,nz,nu].")
    
    if u_avg is not None: #always plot the full flow field
        u_mean = u + u_avg
        y_mean = y + u_avg
        y_mean = np.mean(y_mean,0)
        u_mean = np.mean(u_mean,0)
    else:
        y_mean = np.mean(y,0)
        u_mean = np.mean(u,0)
    
    if error == 'mae':
        e = np.abs(y-u)
        e_mean = np.mean(e,0)
        e_title = 'Absolute error'
    elif error == 'mse':
        e = (y-u)**2
        e_mean = np.mean(e,0)
        e_title = 'Mean square error'
    else:
        raise ValueError('Please choose from mse or mae, or define your own error function.')

    # set colorbar properties
    umin = min(np.amin(u_mean[:,:,0]),np.amin(y_mean[:,:,0]))
    umax = max(np.amax(u_mean[:,:,0]),np.amax(y_mean[:,:,0]))

    vmin = min(np.amin(u_mean[:,:,1]),np.amin(y_mean[:,:,1]))
    vmax = max(np.amax(u_mean[:,:,1]),np.amax(y_mean[:,:,1]))

    fig = plt.figure(clear=True)

    fig.add_subplot(2,3,1,title="True",xticks=[],yticks=[],ylabel='v')
    plt.imshow(u_mean[:,:,0],'jet',vmin=umin,vmax=umax)
    plt.colorbar()

    fig.add_subplot(2,3,2,title="Predicted",xticks=[],yticks=[])
    plt.imshow(y_mean[:,:,0],'jet',vmin=umin,vmax=umax)
    plt.colorbar()

    fig.add_subplot(2,3,3,title=e_title,xticks=[],yticks=[]) # u error
    plt.imshow(e_mean[:,:,0],'jet')
    plt.colorbar()

    fig.add_subplot(2,3,4,xticks=[],yticks=[],ylabel='w')
    plt.imshow(u_mean[:,:,1],'jet',vmin=vmin,vmax=vmax)
    plt.colorbar()

    fig.add_subplot(2,3,5,xticks=[],yticks=[])
    plt.imshow(y_mean[:,:,1],'jet',vmin=vmin,vmax=vmax)
    plt.colorbar()

    fig.add_subplot(2,3,6,xticks=[],yticks=[]) 
    plt.imshow(e_mean[:,:,1],'jet')
    plt.colorbar()
    if savefig:
        path = pathlib.Path(folder_path,'autoencoder_result.png')
        fig.savefig(path)
    else:
        fig.show()


def plot_autoencoder_modes(latent_dim:int,
                            modes:np.ndarray,
                            t:int=0,  
                            savefig:bool=False, 
                            folder_path:Optional[StrOrPath]=None):
    '''Plot instantaneous autoencoder modes for the first two decoders
    
    mode_train: shape [mode, nt, ny, nz, nu]
    t: int, which snapshot to plot
    savefig: bool
    folder_path: folder to save the figure to if savefig is true
    '''
    if modes.ndim != 5:
        raise ValueError("Input shape must be [mode,nt,ny,nz,nu].")

    figname = 'autoencoder_mode_at_time_' + str(t) + '.png'
    fig, ax = plt.subplots(2,latent_dim,sharey='all',clear=True)
    fig.suptitle(figname)
    for u in range(2):
        for i in range(latent_dim):
            im = ax[u,i].imshow(modes[i,t,:,:,u],'jet')
            div = make_axes_locatable(ax[u,i])
            cax = div.append_axes('right',size='5%',pad='2%')
            plt.colorbar(im,cax=cax)
            ax[u,i].set_xticks([])
            ax[u,i].set_yticks([])
    for i in range(latent_dim):
        ax[0,i].set_title(str(i+1))
    ax[0,0].set_ylabel('v')
    ax[1,0].set_ylabel('w')
    if savefig:
        path = pathlib.Path(folder_path,figname)
        fig.savefig(path)
    else:
        fig.show()


def plot_decoder_weights_bias(decoder:Model, latent_dim:int, savefig=False, folder_path:StrOrPath=None):
    '''Plot the first two decoder weights and biases for a linear autoencoder.

    activation_function: activation function passed to the autoencoder
    decoders: a list of deocders (class Decoder)
    savefig: save figure to folder_path if true, show figure otherwise
    folder_path: path to the result folder
    
    '''
    if decoder.act_fct != 'linear':
        raise ValueError('Calculations only apply to linear autoencoders.')
    

    z_zero = np.zeros((1,latent_dim))
    b = decoder.predict(z_zero)

    wb1 = decoder.predict(np.reshape([1,0],(1,2)))
    w1 = wb1-b
    wb2 = decoder.predict(np.reshape([0,1],(1,2)))
    w2 = wb2-b

    fig, ax = plt.subplots(2,2,sharey='all',clear=True)
    fig.suptitle('decoder weights')
    im1v = ax[0,0].imshow(w1[0,:,:,0],'jet')
    div = make_axes_locatable(ax[0,0])
    cax = div.append_axes('right',size='5%',pad='2%')
    plt.colorbar(im1v,cax=cax)
    im1w = ax[1,0].imshow(w1[0,:,:,1],'jet')
    div = make_axes_locatable(ax[1,0])
    cax = div.append_axes('right',size='5%',pad='2%')
    plt.colorbar(im1w,cax=cax)
    im2v = ax[0,1].imshow(w2[0,:,:,0],'jet')
    div = make_axes_locatable(ax[0,1])
    cax = div.append_axes('right',size='5%',pad='2%')
    plt.colorbar(im2v,cax=cax)
    im2w = ax[1,1].imshow(w2[0,:,:,1],'jet')
    div = make_axes_locatable(ax[1,1])
    cax = div.append_axes('right',size='5%',pad='2%')
    plt.colorbar(im2w,cax=cax)
    ax[0,0].set_title('decoder 1')
    ax[0,1].set_title('decoder 2')
    ax[0,0].set_ylabel('v')
    ax[1,0].set_ylabel('w')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])

    if savefig:
        path = pathlib.Path(folder_path,'decoder_weights.png')
        fig.savefig(path)
    else:
        fig.show()

    fig, ax = plt.subplots(1,2,sharey=True,sharex=True,clear=True)
    fig.suptitle('decoder bias')
    imv = ax[0].imshow(b[0,:,:,0],'jet')
    div = make_axes_locatable(ax[0])
    cax = div.append_axes('right',size='5%',pad='2%')
    plt.colorbar(imv,cax=cax)
    imw = ax[1].imshow(b[0,:,:,1],'jet')
    div = make_axes_locatable(ax[1])
    cax = div.append_axes('right',size='5%',pad='2%')
    plt.colorbar(imw,cax=cax)
    ax[0].set_title('v')
    ax[1].set_title('w')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    if savefig:
        path = pathlib.Path(folder_path,'decoder_bias.png')
        fig.savefig(path)
    else:
        fig.show()