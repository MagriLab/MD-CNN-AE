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
# def plot_ae_results(u,y,u_avg=None,error='mae'):
#     # u: input to the autoencoder, shape [time,Nx,Ny,u]
#     # y: output from ae.predict(u)