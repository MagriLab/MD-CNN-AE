import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import typing
from typing import Optional
import pathlib
StrOrPath = typing.Union[str,pathlib.Path]
DataTuple = tuple[np.ndarray]
DataList = typing.Union[np.ndarray,list]


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
    fig = plt.figure()
    plt.plot(hist_train,label="training")
    plt.plot(hist_val,label="validation")
    plt.title("Training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if savefig:
        path = pathlib.Path(folder_path,'training_history.png')
        fig.savefig(path)
        fig.clear()
    else:
        fig.show()
        fig.clear()


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

    fig = plt.figure()

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
        fig.clear()
    else:
        fig.show()
        fig.clear()


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

    figname = 'autoencoder mode at time ' + str(t) + '.png'
    fig, ax = plt.subplots(2,latent_dim,sharey='all')
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
        fig.clear()
    else:
        fig.show()
        fig.clear()


def plot_latent_variable(coded_test:DataList, savefig:bool=False, folder_path:Optional[StrOrPath]=None, figtitle:Optional[str]=None):
    if np.array(coded_test).ndim != 2:
        raise ValueError("Input shape must be [nt modes].")

    fig = plt.figure(clear=True)
    for i in range(np.array(coded_test).shape[1]):
        plt.plot(coded_test[:,i],label=str(i+1))
    plt.legend()
    plt.ylabel('value of latent variable')
    if figtitle:
        plt.title(figtitle)
    if savefig:
        path = pathlib.Path(folder_path,'latent_variables.png')
        fig.savefig(path)
        fig.clear()
    else:
        fig.show()
        fig.clear()


def plot_decoder_weights(activation_function:str, decoders:list, savefig=False, folder_path:StrOrPath=None):
    '''Plot the first two decoder weights for a linear autoencoder.

    activation_function: activation function passed to the autoencoder
    decoders: a list of deocders (class Decoder)
    savefig: save figure to folder_path if true, show figure otherwise
    folder_path: path to the result folder
    
    '''
    if activation_function != 'linear':
        raise ValueError('Calculations only apply to linear autoencoders.')
    
    b1 = decoders[0].predict(np.reshape(0,(1,1)))
    wb1 = decoders[0].predict(np.reshape(1,(1,1)))
    w1 = wb1-b1
    b2 = decoders[1].predict(np.reshape(0,(1,1)))
    wb2 = decoders[1].predict(np.reshape(1,(1,1)))
    w2 = wb2-b2

    fig, ax = plt.subplots(2,2,sharey='all')
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
    ax[0,0].set_title('1')
    ax[0,1].set_title('2')
    ax[0,0].set_ylabel('v')
    ax[1,0].set_ylabel('w')

    if savefig:
        path = pathlib.Path(folder_path,'decoder weights.png')
        fig.savefig(path)
        fig.clear()
    else:
        fig.show()
        fig.clear()
