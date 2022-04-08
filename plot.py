import h5py
import numpy as np
import matplotlib.pyplot as plt
import mode_decomposition as md

time = 0 # snapshot to plot
folder = '/home/ym917/OneDrive/PhD/Code_md-ae/MD_10__2022_02_19__14_04_07'
important_ae_modes = [10,6,8,7] # [1byenergy,2byenergy,1by%,2by%]

## Read parameters and results
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
REMOVE_MEAN = file.get('REMOVE_MEAN')[()]
file.close()

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
u_train = np.array(file.get('u_train')) # fluctuating velocity if REMOVE_MEAN is true
y_train = np.array(file.get('y_train'))
u_test = np.array(file.get('u_test')) # fluctuating velocity if REMOVE_MEAN is true
y_test = np.array(file.get('y_test'))
u_avg = np.array(file.get('u_avg'))
latent_train = np.array(file.get('latent_train'))
latent_test = np.array(file.get('latent_test'))
modes_train = np.array(file.get('modes_train'))
modes_test = np.array(file.get('modes_test')) #(modes,snapshots,Nx,Ny,Nu)
file.close()

POD_modes = []
POD_mean = []
## calculate POD
for i in important_ae_modes:
    vy = modes_test[i,:,:,:,0].astype('float64')
    vy = np.transpose(vy,[1,2,0])
    vz = modes_test[i,:,:,:,1].astype('float64')
    vz = np.transpose(vz,[1,2,0]) #(ny,nz,nt)
    X = np.vstack((vz,vy)) # new shape [2*ny,nz,nt]
    pod = md.POD(X)
    Q_POD,lam = pod.get_modes()
    POD_modes.append(Q_POD)
    POD_mean.append(pod.Q_mean)


count_figure = 0

count_figure += 1
plt.figure('testing_latent_space')
for z in range(latent_dim):
    label = 'autoencoder mode '+str(z+1)
    plt.plot(latent_test[:,z],label = label)
plt.xlim((0,latent_test.shape[0]))
plt.xlabel('snapshot')
plt.title('testing latent space')
plt.legend()

count_figure += 1
plt.figure('ae_mode_with_POD_modes')
fig,ax = plt.subplots(nrows=1,ncols=3)
for i in important_ae_modes:
    


plt.show()