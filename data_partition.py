import h5py
import numpy as np
from project_specific_utils.training import data_partition

# =========================== Change these values ============================
Ntrain = 1500 # snapshots for training
Nval = 632 # sanpshots for validation
Ntest = 600

SHUFFLE = False # shuffle before splitting into sets, test set is extracted before shuffling
REMOVE_MEAN = True # train on fluctuating velocity
data_type = np.float32

save_as = 'training_data_unshuffle_1.h5'

read_data_file = './data/PIV4_downsampled_by8.h5'
Nz = 24 # grid size
Ny = 21
Nu = 2
Nt = 2732 # number of snapshots available
D = 196.5 # mm diameter of bluff body
U_inf = 15 # m/s freestream velocity
f_piv = 720.0 # Hz PIV sampling frequency  
dt = 1.0/f_piv 

# =============================================================
u_all, (u_train, u_val, u_test), (idx_test, idx_unshuffle), (u_mean_all, 
    u_mean_train, u_mean_val, u_mean_test) = data_partition(read_data_file, 
                                    [Nt,Nz,Ny,Nu], 
                                    [Ntrain,Nval,Ntest], 
                                    SHUFFLE=SHUFFLE, 
                                    REMOVE_MEAN=REMOVE_MEAN, 
                                    data_type=data_type)
# save to file
save_to = './data/' + save_as

hf = h5py.File(save_to,'w-')
hf.create_dataset('u_all',data=u_all)
hf.create_dataset('u_train',data=u_train)
hf.create_dataset('u_val',data=u_val)
hf.create_dataset('u_test',data=u_test)
if SHUFFLE:
    hf.create_dataset('idx_test',data=idx_test)
    hf.create_dataset('idx_unshuffle',data=idx_unshuffle)
if REMOVE_MEAN:
    hf.create_dataset('u_mean_all',data=u_mean_all)
    hf.create_dataset('u_mean_train',data=u_mean_train)
    hf.create_dataset('u_mean_val',data=u_mean_val)
    hf.create_dataset('u_mean_test',data=u_mean_test)
hf.create_dataset('SHUFFLE',data=SHUFFLE)
hf.create_dataset('REMOVE_MEAN',data=REMOVE_MEAN)
hf.close()