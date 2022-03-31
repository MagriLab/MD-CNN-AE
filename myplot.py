import numpy as np
from matplotlib import pyplot as plt
import sys


# Plot a figure for each autoencoder mode
# left: time-averaged velocity v, right time-averaged velocity w
def plot_ae_modes(modes,modes_to_plot,savefig=False,path=None):
    # Inputs:
    # modes: the autoencoder modes from training, has shape [latent_dim,nt,ny,nz,nu]
    # modes_to_plot: range or array, if plotting first 5 modes then range(5)
    # savefig: if True fgure is saved to the folder defined in path. Default False
    # path: path to target folder
    if modes.ndim != 5:
        sys.exit("Input shape must be [latent_dim,nt,ny,nz,nu].") 
    
    for WhichDecoder in modes_to_plot:
        vy = modes[WhichDecoder,:,:,:,0].astype('float64')
        vz = modes[WhichDecoder,:,:,:,1].astype('float64')
        vy = np.transpose(vy,[1,2,0])
        vz = np.transpose(vz,[1,2,0]) #(ny,nz,nt)
        X = np.vstack((vz,vy)) # new shape [2*ny,nz,nt]

        fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        title = 'Autoencoder mode '+str(WhichDecoder+1)
        fig.suptitle(title)
        fig1 = ax1.imshow(np.mean(vy,-1),'jet')
        fig2 = ax2.imshow(np.mean(vz,-1),'jet')
        fig.colorbar(fig1,ax=ax1)
        fig.colorbar(fig2,ax=ax2)
        ax1.set_title('v')
        ax2.set_title('w')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        if savefig:
            figpath = path + title
            plt.savefig(figpath)
        else:
            plt.show(block=False)


# plot autoencoder results
def plot_ae_results(u,y,u_avg=None,error='mae',savefig=False,path=None):
    # u: input to the autoencoder, shape [time,Ny,Nz,u]
    # y: output from ae.predict(u)
    # u_avg: the time-averaged component if the network was trained on fluctuating velocity
    # error: the error function to plot. 'mse' or 'mae'
    if u.ndim != 4:
        sys.exit("Input shape must be [nt,ny,nz,nu].") 
    
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
    elif error == 'mse':
        e = (y-u)**2
        e_mean = np.mean(e,0)
    else:
        sys.exit('Please choose from mse or mae, or define your own error function.')

    # set colorbar properties
    umin = min(np.amin(u_mean[:,:,0]),np.amin(y_mean[:,:,0]))
    umax = max(np.amax(u_mean[:,:,0]),np.amax(y_mean[:,:,0]))

    vmin = min(np.amin(u_mean[:,:,1]),np.amin(y_mean[:,:,1]))
    vmax = max(np.amax(u_mean[:,:,1]),np.amax(y_mean[:,:,1]))

    plt.figure()

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
    if savefig:
        plt.savefig(path)
    else:
        plt.show()
