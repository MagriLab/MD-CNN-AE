import numpy as np
from numpy import einsum
import warnings


# Kneer, S., Sayadi, T., Sipp, D., Schmid, P. & Rigas, G. (2021) Symmetry-Aware Autoencoders: s-PCA and s-nlPCA. ArXiv. 10.13140/RG.2.2.11634.22728. 
def equivalent_pca_energy(autoencoder_modes:np.ndarray, pod_modes:np.ndarray) -> np.ndarray:
    '''Calculate the equivalent PCA energy of the autoencoder modes.
    
    Arguments:
        autoencoder_modes: array of autoencoder modes, with shape [latent_dim, nt, ny, nz, nu].\n
        pod_modes: array of POD modes, must be a square matrix with side ny*nz, each column is one POD mode.
    '''
    if isinstance(autoencoder_modes, np.ndarray):
        pass
    elif isinstance(autoencoder_modes, list):
        autoencoder_modes = np.array(autoencoder_modes)
        warnings.warn('Input is given as a list...converting to numpy array.')
    else:
        raise ValueError('Please provide autoencoder modes as an numpy array.')

    if (autoencoder_modes.ndim != 5) or (pod_modes.ndim != 2):
        raise ValueError('Input autoencoder_modes must have shape [latent_dim, nt, ny, nz, nu], pod_modes must be a square matrix of (ny*nz).')
    
    nt = autoencoder_modes.shape[1]

    x_train = einsum('k t y z u -> k y z t u', autoencoder_modes)
    lam_modes = [] # how much of a POD is in an AE mode
    for i in range(autoencoder_modes.shape[0]):
        if x_train.shape[-1] > 1:
            X_train = np.vstack((x_train[i,:,:,:,0], x_train[i,:,:,:,1]))
            X_train = np.reshape(X_train,(-1, nt))
        else:
            X_train = np.reshape(x_train[i,...],(-1, nt))
        A = einsum('x t, x m -> t m', X_train, pod_modes) # project
        if nt == 1:
            warnings.warn('The autoencoder modes contain only one snapshots, variance cannot be calculated. Continuing the calculatin without diving by nt-1.')
            lam_1 = einsum('t m -> m',A**2) 
        else: 
            lam_1 = einsum('t m -> m',A**2) / (nt - 1)
        lam_modes.append(lam_1)
    lam_modes = np.array(lam_modes) # [latent_dim, ny*nz]

    return lam_modes