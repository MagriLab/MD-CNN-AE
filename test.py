import numpy as np
from matplotlib import pyplot as plt

signal_span = np.r_[0: 10: 0.1]
signal_in = np.sin(signal_span*np.pi)
# plt.figure()
# plt.plot(signal_span,signal_in)
# plt.show()

fs = 1/0.1
signal_freq = np.r_[0:signal_span.size:1]
signal_freq = signal_freq*fs/signal_freq.size

signal_fft = np.fft.fft(signal_in)
signal_fft_power = np.abs(signal_fft)
# signal_freq = np.fft.fftfreq(signal_span.size)
# print(signal_freq/(np.pi/4/2/np.pi))
# signal_freq = signal_freq/(0.1)

# plt.figure()
# plt.plot(signal_freq,signal_fft_power,'o')
# plt.show()

# # print(signal_freq/(np.pi/4/2/np.pi))


import h5py
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import time
import os
import datetime

#============================== CCHANGE THESE VALUES ======================
# data
data_file = '../Data/PIV/PIV4_downsampled_by8.h5'
Ntrain = 1800 # snapshots for training
Ntest = 900 # sanpshots for testing

# path
folder = './PIV/Autoencoder/MD_2__2022_02_02__16_51_24/'

## ae configuration
filename = folder + 'Model_param.h5'
file = h5py.File(filename,'r')
lmb = file.get('lmb')[()]#1e-05 #regulariser
drop_rate = file.get('drop_rate')[()]
features_layers = np.array(file.get('features_layers')).tolist()
latent_dim = file.get('latent_dim')[()]
act_fct = file.get('act_fct')[()].decode()
resize_meth = file.get('resize_meth')[()].decode()
filter_window= np.array(file.get('filter_window')).tolist()
batch_norm = file.get('batch_norm')[()]
file.close()

#================================= IMPORT DATA ==========================================================
Nz = 24 # grid size
Ny = 21
Nu = 2
Nt = 2732 # number of snapshots available
D = 196.5 # mm diameter of bluff body
U_inf = 15 # m/s freestream velocity
f_piv = 720.0 # Hz PIV sampling frequency  
dt = 1.0/f_piv 
Nx = [Ny,Nz]

filename = folder + 'results.h5'
file = h5py.File(filename,'r')
u_train = np.array(file.get('u_train'))
y_train = np.array(file.get('y_train'))
u_test = np.array(file.get('u_test'))
y_test = np.array(file.get('y_test'))
# u_avg = np.array(file.get('u_avg'))
latent_train = np.array(file.get('latent_train'))
latent_test = np.array(file.get('latent_test'))
modes_train = np.array(file.get('modes_train'))
modes_test = np.array(file.get('modes_test')) #(modes,snapshots,Nx,Ny,Nu)
file.close()

signal_in = latent_test[:,0]
signal_fft = np.fft.fft(signal_in)
signal_fft_power = np.abs(signal_fft)/signal_in.size
signal_freq = np.fft.fftfreq(signal_in.size)
signal_freq = signal_freq/dt
print(signal_freq.size/2)
plt.figure()
# plt.bar(signal_freq,signal_fft_power,width=360/600)
plt.plot(signal_freq[:int(signal_freq.size/2)],signal_fft_power[:int(signal_freq.size/2)])
plt.ylim(bottom=0)
plt.yticks([0])
plt.xlim(left=0)
loc,label = plt.xticks()
loc = np.r_[0:max(loc):25]
plt.xticks(loc)
plt.show()