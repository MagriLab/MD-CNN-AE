# Calculate an 'energy' that can be used to rank autoencoder modes.
# 4 methods available

import numpy as np
from matplotlib import pyplot as plt
import sys

# rms amplitude and mean
def rms_amplitude(signal):
    # signal: latent vector, has shape (time, latent_dim), the array must NOT be shuffled
    # return: 
    # amp_rms: an array of rms amplitude, has length = latent_dim
    # signal_mean: np.mean(siganl,axis=0). Time averaged signal mean
    if signal.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).")
    amp_rms = np.mean((signal-np.mean(signal,0))**2,0)**0.5 
    signal_mean = np.mean(signal,0)
    return amp_rms, signal_mean


# signal energy in frequency domain
def energy_freq(signal):
    # signal: latent vector, has shape (time, latent_dim), the array must NOT be shuffled
    # return: 
    # signal_energy: an array to total energy of each mode (parseval's theorem)
    if signal.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).") 
    signal_energy = []
    for n in range(0,signal.shape[1]):
        signal_in = signal[:,n]
        signal_fft = np.fft.fft(signal_in)
        signal_energy.append(np.sum(np.abs(signal_fft)**2)/signal_fft.size)
    return signal_energy


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
def NED(a,k):
    # a: latent vector with shape (snapshots,latent_dim)
    # k: number of histogram bins
    # return:
    # NED: an array of NED, length = number of modes
    if a.ndim != 2:
        sys.exit("Input shape must be (time, latent_dim).") 
    n = a.shape[0] # total number of snapshots
    a = normalise(a,axis=0) # scale values to be between 0 and 1
    NED = [] 
    for x in range(0,a.shape[1]):
        i,temp = np.histogram(a[:,x],k)
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
def normalise(a,axis):
    # a: is a 2D array
    # axis: the axis to perform normalisation  
    # return:
    # norm_a: a normalised along the chosen axis. If axis is not None, then norm_a has the same shape as a
    width = np.ptp(a.astype('float64'),axis=axis,keepdims=True)
    minimum = np.amin(a.astype('float64'),axis=axis,keepdims=True)
    
    # make sure the next step do not produce nan
    # in norm_a the column/row where a is constant will be all 0
    idx_const = (width == 0.).nonzero()
    print('Peak-to-peak value ', idx_const, ' is 0., this variable is constant')
    width[idx_const] = width[idx_const] + 0.0001

    norm_a = (a - minimum)/width
    return norm_a