import numpy as np
import matplotlib.pyplot as plt
import h5py
import mode_decomposition as md
import autoencoder_modes_selection as ranking

folder = '/home/ym917/OneDrive/PhD/Code_md-ae/MD_10__2022_02_21__10_01_23/'

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

a = ranking.NED(latent_test,2,act_fct=act_fct)
plt.figure()
plt.bar(np.arange(10)+1,a)

# plt.figure()
# plt.plot(latent_test[:,4])
# plt.plot(latent_test[:,5])
# plt.plot(latent_test[:,6])


b = ranking.normalise(latent_test,axis=0,data_range=[-1,1])
# plt.figure()
# plt.plot(b[:,4])
# plt.plot(b[:,5])
# plt.plot(b[:,6])

for x in range(4,7):
    i,temp = np.histogram(b[:,x],np.linspace(0,1,5+1))
    print(i)
    print(temp)
    p = i/float(600) # probability of latent variable in a histogram bin
    p = p[p!=0] # consider only occupied bins
    k_occupied = p.size
    print(k_occupied)

plt.show()
