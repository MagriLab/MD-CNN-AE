import h5py
import numpy as np

# =========================== Change these values ============================
Ntrain = 1500 # snapshots for training
Nval = 632 # sanpshots for validation
Ntest = 600

SHUFFLE = True # shuffle before splitting into sets, test set is extracted before shuffling
REMOVE_MEAN = True # train on fluctuating velocity
data_type = 'float32'

save_as = 'training_data_1.h5'

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
# read data
hf = h5py.File(read_data_file,'r')
z = np.array(hf.get('z'))
y = np.array(hf.get('y'))
u_all = np.zeros((Nt,Nz,Ny,Nu))
u_all[:,:,:,0] = np.array(hf.get('vy'))
if Nu==2:
    u_all[:,:,:,1] = np.array(hf.get('vz'))
u_all = np.transpose(u_all,[0,2,1,3]) # shape of u_all = (Nt,Ny,Nz,Nu)
hf.close()

u_all = u_all[:,:,:,:].astype(data_type)

if SHUFFLE:
    idx_test = np.random.randint(0,Nt-Ntest)
    u_test = u_all[idx_test:idx_test+Ntest,:,:,:].astype(data_type) # test set needs to be in order and has continuous snapshots
    u_all = np.delete(u_all,np.s_[idx_test:idx_test+Ntest],0) # remove the test set from available samples
    idx_shuffle = np.arange(Nt-Ntest) # use idx_shuffle to shuffle the rest of samples before taking a validation set
    np.random.shuffle(idx_shuffle)
    idx_unshuffle = np.argsort(idx_shuffle) # use idx_unshuffle to unshuffle the data
    u_all = u_all[idx_shuffle,:,:,:]
    u_train = u_all[0:Ntrain,:,:,:].astype(data_type)
    u_val = u_all[Ntrain:Ntrain+Nval,:,:,:].astype(data_type)
    u_all = np.vstack((u_train,u_val,u_test))
else:
    u_train = u_all[0:Ntrain,:,:,:].astype(data_type)
    u_val = u_all[Ntrain:Ntrain+Nval,:,:,:].astype(data_type)
    u_test = u_all[Ntrain+Nval:Ntrain+Nval+Ntest,:,:,:].astype(data_type)
    u_all = u_all[0:Ntrain+Nval+Ntest,:,:,:].astype(data_type) # u_all has shape (Ntrain+Nval+Ntest,Ny,Nz,Nu)

# remove mean for training
if REMOVE_MEAN:
    u_mean_all = np.mean(u_all,axis=0)
    u_all = u_all - u_mean_all
    u_mean_train = np.mean(u_train,axis=0)
    u_train = u_train-u_mean_train
    u_mean_val = np.mean(u_val,axis=0)
    u_val = u_val - u_mean_val
    u_mean_test = np.mean(u_test,axis=0)
    u_test = u_test-u_mean_test

u_all = np.reshape(u_all,(1,Ntrain+Nval+Ntest,Ny,Nz,Nu)) # new shape (1,Nval+Ntrain+Ntest,Ny,Nz,Nu)
u_train = np.reshape(u_train,(1,Ntrain,Ny,Nz,Nu))
u_val = np.reshape(u_val,(1,Nval,Ny,Nz,Nu))
u_test = np.reshape(u_test,(1,Ntest,Ny,Nz,Nu))


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