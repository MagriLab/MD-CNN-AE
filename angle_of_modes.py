import h5py
import numpy as np
from numpy import einsum
import configparser
import matplotlib.pyplot as plt
from pathlib import Path


config = configparser.ConfigParser()
config.read('_system.ini')
results_dir = config['system_info']['alternate_location']

parent_folder = Path(results_dir,'experiment_nonlinear_regularisation')
ls0 = [*parent_folder.glob('rmb00-*')]
ls01 = [*parent_folder.glob('rmb01-*')]
ls001 = [*parent_folder.glob('rmb001-*')]

angles0 = []
angles01 = []
angles001 = []

for folder in ls0:
    folder_path = Path(parent_folder,folder)
    filename = Path(folder_path,'results.h5')
    with h5py.File(filename,'r') as hf:
        modes_train = np.array(hf.get('modes_train'))
    Ntrain = modes_train.shape[1]

    # calculate orthogonality
    mode1 = einsum('t y z u-> y z t u',np.squeeze(modes_train[0,...]))
    mode2 = einsum('t y z u-> y z t u',np.squeeze(modes_train[1,...]))
    mode1 = np.vstack((mode1[:,:,:,0],mode1[:,:,:,1]))
    mode2 = np.vstack((mode2[:,:,:,0],mode2[:,:,:,1]))
    mode1 = np.reshape(mode1,(-1,Ntrain))
    mode2 = np.reshape(mode2,(-1,Ntrain))

    magnitude1 = np.sqrt(einsum('x t -> t', mode1**2))
    magnitude2 = np.sqrt(einsum('x t -> t', mode2**2))
    magnitude = magnitude1*magnitude2
    
    modes_dot_product = np.diag(einsum('ij,ik',mode1,mode2))
    angles = modes_dot_product / magnitude
    angles0.extend(angles)
angles0 = np.array(angles0)

for folder in ls01:
    folder_path = Path(parent_folder,folder)
    filename = Path(folder_path,'results.h5')
    with h5py.File(filename,'r') as hf:
        modes_train = np.array(hf.get('modes_train'))
    Ntrain = modes_train.shape[1]

    # calculate orthogonality
    mode1 = einsum('t y z u-> y z t u',np.squeeze(modes_train[0,...]))
    mode2 = einsum('t y z u-> y z t u',np.squeeze(modes_train[1,...]))
    mode1 = np.vstack((mode1[:,:,:,0],mode1[:,:,:,1]))
    mode2 = np.vstack((mode2[:,:,:,0],mode2[:,:,:,1]))
    mode1 = np.reshape(mode1,(-1,Ntrain))
    mode2 = np.reshape(mode2,(-1,Ntrain))

    magnitude1 = np.sqrt(einsum('x t -> t', mode1**2))
    magnitude2 = np.sqrt(einsum('x t -> t', mode2**2))
    magnitude = magnitude1*magnitude2
    
    modes_dot_product = np.diag(einsum('ij,ik',mode1,mode2))
    angles = modes_dot_product / magnitude
    angles01.extend(angles)
angles01 = np.array(angles01)

for folder in ls001:
    folder_path = Path(parent_folder,folder)
    filename = Path(folder_path,'results.h5')
    with h5py.File(filename,'r') as hf:
        modes_train = np.array(hf.get('modes_train'))
    Ntrain = modes_train.shape[1]

    # calculate orthogonality
    mode1 = einsum('t y z u-> y z t u',np.squeeze(modes_train[0,...]))
    mode2 = einsum('t y z u-> y z t u',np.squeeze(modes_train[1,...]))
    mode1 = np.vstack((mode1[:,:,:,0],mode1[:,:,:,1]))
    mode2 = np.vstack((mode2[:,:,:,0],mode2[:,:,:,1]))
    mode1 = np.reshape(mode1,(-1,Ntrain))
    mode2 = np.reshape(mode2,(-1,Ntrain))

    magnitude1 = np.sqrt(einsum('x t -> t', mode1**2))
    magnitude2 = np.sqrt(einsum('x t -> t', mode2**2))
    magnitude = magnitude1*magnitude2
    
    modes_dot_product = np.diag(einsum('ij,ik',mode1,mode2))
    angles = modes_dot_product / magnitude
    angles001.extend(angles)
angles001 = np.array(angles001)

plt.figure(figsize=(7,5))
plt.boxplot([angles0,angles001,angles01],sym='.')
plt.xticks([1,2,3],['$\gamma = 0.0$','$\gamma = 0.01$','$\gamma = 0.1$'],fontsize='large')
plt.ylabel('$\\alpha$',fontsize='large')
plt.ylim([-0.7,0.8])

plt.text(0.87,0.72,f'$\sigma$={np.std(angles0):.3f}',fontsize='large')
plt.text(0.87,0.65,f'$\mu$={np.mean(angles0):.3f}',fontsize='large')

plt.text(1.83,0.72,f'$\sigma$={np.std(angles01):.3f}',fontsize='large')
plt.text(1.83,0.65,f'$\mu$={np.mean(angles01):.3f}',fontsize='large')

plt.text(2.87,0.72,f'$\sigma$={np.std(angles001):.3f}',fontsize='large')
plt.text(2.87,0.65,f'$\mu$={np.mean(angles001):.3f}',fontsize='large')


plt.savefig('regularisation-orthogonality.pdf')