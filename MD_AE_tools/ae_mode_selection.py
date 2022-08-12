'''Ranking methods of autoencoder modes.'''

import numpy as np
import sys
import typing

scalar = typing.Union[int, float, np.number]

def _mse(x:typing.Union[np.ndarray,list], y:typing.Union[np.ndarray,list]) -> scalar:
    if np.array(x).shape != np.array(y).shape:
        raise ValueError('Cannot get mean squared error, shapes of input do not match.')
    mse = np.mean((x - y)**2)
    return mse


# rms amplitude and mean
def rms_amplitude(signal):
    ''' Calculate the rms amplitude of a signal
    
    Calculate the rms amplitude of centred signal. The mean is substracted before the calculation.

    Arguments:
        signal: matrix of dimension 2, each column is a time series. 

    Return:
        amp_rms: an array rms amplitude of signal, length of the array = number of columns in the input.
        signal_mean: np.mean(signal, axis=0). Time averaged signal, length = number of columns in the input.
        rank: ranking of the siganls from the largest to the smallest rms amplitude.
    '''
    signal = np.array(signal)
    if signal.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).")
    amp_rms = np.mean((signal-np.mean(signal,0))**2,0)**0.5 
    signal_mean = np.mean(signal,0)

    rank = rank_array(amp_rms)
    return amp_rms, signal_mean, rank


# signal energy in frequency domain
def energy_freq(signal:typing.Union[np.ndarray,list],
                remove_mean:bool=False):
    '''Calculate signal energy
    
    Calculate signal energy, equivalent to the area under the curve. Calculation is done in the frequency domain using Parseval's theorem.

    Argument:
        signal: latent vector, has shape (time, latent_dim), the array must NOT be shuffled
        remove_mean: if True, signal is centred to 0
    
    return: 
        signal_energy: an array to total energy of each mode
        rank: ranking of signals from largest to smallest 
    '''
    if signal.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).") 
    if remove_mean:
        signal = signal - np.mean(signal,0)
    signal_energy = []
    for n in range(0,signal.shape[1]):
        signal_in = signal[:,n]
        signal_fft = np.fft.fft(signal_in)
        signal_energy.append(np.sum(np.abs(signal_fft)**2)/signal_fft.size)
    
    rank = rank_array(signal_energy)
    return signal_energy, rank


# percentage contribution to the output of the autoencoder
def percent_output(modes,y):
    # modes: the autoencoder modes (sum of modes = output). Has shape (modes,snapshots,Nx,Ny,Nu)
    # y: the autoencoder output, with shape (snapshots,Nx,Ny,Nu)
    # return:
    # per_mode: an array of percentage contribution to y
    if modes.ndim != 5:
        sys.exit("Input shape must be (modes,snapshots,Nx,Ny,Nu).") 
    per_mode = np.mean(np.sum(modes,axis=(2,3,4))/np.sum(y,axis=(1,2,3)),axis=1) # contribution per mode, adds up to 1
    return per_mode


# Normalised Entropy Difference (Fan, 2019)
# Fan, Y. J. (2019) Autoencoder node saliency: Selecting relevant latent representations. Pattern Recognition. 88 643-653. 10.1016/j.patcog.2018.12.015
def NED(a,k,act_fct):
    # a: latent vector with shape (snapshots,latent_dim)
    # k: number of histogram bins
    # act_fct: activation function -- 'tanh','sigmoid'. This information is for normalisation, if you want to normalise with data range give it a non-existing function name
    # return:
    # NED: an array of NED, length = number of modes
    if a.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).") 
    n = a.shape[0] # total number of snapshots
    if act_fct == 'tanh':
        data_range = [-1, 1]
    elif act_fct == 'sigmoid':
        data_range = [0, 1]
    else:
        data_range = None
    a = normalise(a,axis=0,data_range=data_range) # scale values to be between 0 and 1
    NED = [] 
    for x in range(0,a.shape[1]):
        i,temp = np.histogram(a[:,x],np.linspace(0,1,k+1))
        # print(i)
        p = i/float(n) # probability of latent variable in a histogram bin
        p = p[p!=0] # consider only occupied bins
        k_occupied = p.size
        if k_occupied == 1: # log2(1) is 0, cannot divide by 0
            NED.append(1)
        else:
            log2p = np.log2(p)
            E = -np.sum(p*log2p)
            NED.append(1-(E/np.log2(k_occupied)))
    return NED


# normalise vector to between 0 and 1
# a constant variable will return a list of 0
def normalise(a:np.ndarray, 
                axis:typing.Optional[int], 
                data_range:typing.Optional[list]=None) -> np.ndarray:
    '''Normalise a vector to between 0 and 1.

    A vector with the same number everywere will return a list of 0.

    Argument:
        a: a 2D array to be normalised.
        axis: the axis to perform normalisation.
        data_range: the range of data given in [min, max]. If None (default) min and max of the vector to be normalised will be used.

    Return:
        norm_a: a normalised along the chosen axis. If axis is not None, then norm_a has the same shape as a.    
    '''
    if not data_range:
        width = np.ptp(a.astype('float64'),axis=axis,keepdims=True)
        minimum = np.amin(a.astype('float64'),axis=axis,keepdims=True)
        # make sure the next step do not produce nan
        # in norm_a the column/row where a is constant will be all 0
        idx_const = (width == 0.).nonzero()
        if not idx_const:
            print('Peak-to-peak value ', idx_const, ' is 0., this variable is constant')
        width[idx_const] = width[idx_const] + 0.0001
        norm_a = (a - minimum)/width
    elif len(data_range) != 0:
        width = data_range[1]-data_range[0]
        norm_a = (a - data_range[0])/width
    return norm_a



def kinetic_energy_ae_modes(modes):
    '''Rank autoencoder modes by total kinetic energy.
    
    The total kinetic energy is the sum of kinetic energy of all cells in all snapshots.

    Argument: 
        modes: numpy array of autoencoder modes, has shape (modes,snapshots,Nx,Ny,Nu).

    Return:
        ke: list of total kinetic energy, length = number of autoencoder modes
        rank: ranking of total kinetic energy, largest to smallest.
    '''
    if modes.ndim != 5:
        raise ValueError("Input shape must be (modes,snapshots,Nx,Ny,Nu).") 
    # ke_total =\sum_t,n(1/2 * sqrt(u^2 + v^2 + w^2)), where t is the number of snapshots and n is the number of cells.
    ke = 0.5 * np.sqrt(np.einsum('z t x y u -> z', modes**2))
    rank = rank_array(ke)
    return ke, rank


def variance_of_latent(latent_space:np.ndarray):
    '''Rank autoencoder modes by covariance of the latent space.
    
    Calculate the variance-covariance matrix of the latent space and rank the modes by their variance.
    
    Argument:
        latent_space: numpy array. Output of the encoder, must have shape (time, latent dimension)
    
    Return:
        cov: variance-covariance matrix of the latent variables.
        rank: ranking of the latent variables by their variance, largest to smallest.
    '''
    if latent_space.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).") 
    cov = np.cov(latent_space.T)
    rank = rank_array(np.diag(cov))
    return cov, rank
    

def rank_array(array_like:typing.Union[np.ndarray,list], 
                ascending=False, 
                index_of_input:typing.Optional[typing.Union[np.ndarray,list]]=None) -> np.ndarray:
    '''Rank the array_like input.
    
    Returns a list of index of the sorted elements of the array. Default descending order. If ascending is True, then sort by ascending order.

    Input: 
        array_like: array to sort, must have only one dimension.
        ascending: default False (descending order). 
        index_of_input: if None (default), the index number of the input 'array_like' will be assigned to range(0, len_of_input). 
                        When provided by user, it must have the same shape as array_like. 
    '''
    array_like = np.squeeze(np.array(array_like))
    if array_like.ndim != 1:
        raise ValueError('Incorrect array dimension, must be 1D array.')
    
    sort_idx = np.argsort(array_like)

    if index_of_input is None:
        index_of_input = np.arange(0, len(array_like))
    else:
        index_of_input = np.squeeze(np.array(index_of_input))
        if array_like.shape != index_of_input.shape:
            raise ValueError('Incorrect user index for the input. Must be the same shape.')

    if ascending:
        rank = index_of_input[sort_idx]
    else:
        rank = index_of_input[np.flip(sort_idx)]

    return rank


def best_mse_combined(modes:np.ndarray, data:np.ndarray) -> np.ndarray:
    '''Rank autoencoder modes cumulatively by mean squared error.
    
    Find combinations of autoencoder modes that lead to the best reconstruction error by iteration. For example, a ranking of [2,1,3] means that the mse(data, mode_2) has the lowest error. Then try the combination mode_2 + mode_1 and mode_2 + mode_1 and mse(data, mode_2+mode_1) has lower error. 

    Arguments:
        modes: autoencoder modes with shape (modes,snapshots,Nx,Ny,Nu).
        data: original data (input to the autoencoder), has shape (snapshots,Nx,Ny,Nu)
    
    Return:
    rank: ranking of autoencoder modes, lowest mean square error first.
    '''
    if (modes.ndim != 5) or (data.ndim != 4):
        raise ValueError("Modes must have shape (modes,snapshots,Nx,Ny,Nu), data must have shape (snapshots, Nx, Ny, Nu).") 
    
    latent_dim = modes.shape[0]
    modes_E = np.copy(modes)
    rank = np.zeros(latent_dim,dtype='int')-1
    for j in range(latent_dim):
        E_ref = 0
        for i in range(latent_dim):
            if i not in rank:
                E = 1/_mse(data,modes_E[i,:,:,:,:])
                if E > E_ref:
                    E_ref = E
                    rank[j] = i
        modes_E = modes + modes_E[rank[j]]
    
    return rank


def best_mse_individual(modes:np.ndarray, data:np.ndarray) -> np.ndarray:
    '''Rank autoencoder modes by their individual mean squared error.
    
    Rank autoencoder modes by comparing the mse(a mode, data).

    Arguments:
        modes: autoencoder modes with shape (modes,snapshots,Nx,Ny,Nu).
        data: original data (input to the autoencoder), has shape (snapshots,Nx,Ny,Nu)
    
    Return:
    rank: ranking of autoencoder modes, lowest mean square error first.
    '''
    if (modes.ndim != 5) or (data.ndim != 4):
        raise ValueError("Modes must have shape (modes,snapshots,Nx,Ny,Nu), data must have shape (snapshots, Nx, Ny, Nu).")
        
    latent_dim = modes.shape[0]
    i_mse = []
    for i in range(latent_dim):
        i_mse.append(_mse(data,modes[i,:,:,:,:]))
    rank = rank_array(i_mse,ascending=True)
    return rank