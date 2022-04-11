from ast import Lambda
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mode_decomposition as md
import autoencoder_modes_selection as ranking



time = 0 # snapshot to plot

## MD-CNN-AE
folder = '/home/ym917/OneDrive/PhD/Code_md-ae/MD_10__2022_02_19__14_04_07/'
important_ae_modes = [9,5,7,6] # start from 0 [1byenergy,2byenergy,1by%,2by%]

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
POD_lam = []
Vy = []
Vz =[]
DMD_modes = []
DMD_lam = []
## calculate POD
for i in important_ae_modes:
    vy = modes_test[i,:,:,:,0].astype('float64')
    vy = np.transpose(vy,[1,2,0])
    vz = modes_test[i,:,:,:,1].astype('float64')
    vz = np.transpose(vz,[1,2,0]) #(ny,nz,nt)
    Vy.append(vy)
    Vz.append(vz)
    X = np.vstack((vz,vy)) # new shape [2*ny,nz,nt]
    pod = md.POD(X)
    Q_POD,lam = pod.get_modes()
    POD_modes.append(Q_POD)
    POD_mean.append(pod.Q_mean)
    POD_lam.append(lam)

    dmd = md.DMD(X[:,:,:-1],X[:,:,1:],r=50,keep_shape=True)
    Phi,Lambda,b = dmd.get_modes()
    DMD_modes.append(Phi.real)
    DMD_lam.append(np.diag(Lambda))


## HIERARCHICAL
folder = '/home/ym917/OneDrive/PhD/Code_md-ae/Hierarchical_10_1__2022_04_05__00_20_54/'

filename = folder + 'results.h5'
hf = h5py.File(filename,'r')
u_test_h = np.array(hf.get('u_test'))
u_avg = np.array(hf.get('u_avg'))
latent_test_h = np.array(hf.get('latent_test')) # shape [latent_variable, test_snapshots, 1]
y_test_h = np.array(hf.get('y_test')) # [modes,nt,ny,nz,nu]
hf.close()

filename = folder + 'Model_param.h5'
hf = h5py.File(filename,'r')
no_of_modes = int(hf.get('no_of_modes')[()])
hf.close()

modes_test_h = np.copy(y_test_h)
for z in range(1,no_of_modes):
    modes_test_h[z,:,:,:,:] = modes_test_h[z,:,:,:,:] - modes_test_h[z-1,:,:,:,:] # [modes,snapshots,ny,nz,u]

POD_modes_h =[]
POD_mean_h = []
POD_lam_h = []
Vy_h = []
Vz_h = []
for i in range(2):
    vy_h = modes_test_h[i,:,:,:,0].astype('float64')
    vy_h = np.transpose(vy_h,[1,2,0])
    vz_h = modes_test_h[i,:,:,:,1].astype('float64')
    vz_h = np.transpose(vz_h,[1,2,0])
    Vy_h.append(vy_h)
    Vz_h.append(vz_h)
    X = np.vstack((vz_h,vy_h)) # new shape [2*ny,nz,nt]
    print(X.shape)
    pod_h = md.POD(X)
    Q_POD,lam = pod_h.get_modes()
    POD_modes_h.append(Q_POD)
    POD_mean_h.append(pod_h.Q_mean)
    POD_lam_h.append(lam)


## DATA
hf = h5py.File("./PIV4_downsampled_by8.h5",'r')
vy = np.array(hf.get('vy'))
vz = np.array(hf.get('vz'))
hf.close()
# [nt,nz,ny] = vz.shape
vy = np.transpose(vy,[2,1,0])
vz = np.transpose(vz,[2,1,0]) #(ny,nz,nt)
X = np.vstack((vz,vy)) # new shape [2*ny,nz,nt]
pod_data = md.POD(X)
POD_modes_data,lam_data = pod_data.get_modes()
POD_mean_data = pod_data.Q_mean

dmd_data = md.DMD(X[:,:,:-1],X[:,:,1:],r=50,keep_shape=True)
Phi, DMD_lam_data, DMD_b_data = dmd.get_modes()
DMD_modes_data = Phi.real
DMD_lam_data = np.diag(DMD_lam_data)


count_figure = 0

# ======================= latent space ======================
count_figure += 1
fig1 = plt.figure('testing_latent_space')
for z in range(latent_dim):
    label = str(z+1)
    plt.plot(latent_test[:,z],label = label)
plt.xlim((0,latent_test.shape[0]))
plt.xlabel('snapshot')
plt.title('testing latent space')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=5)

    
# ========================= ae mode energy ranking ============================
# look up matplotlib.gridspec
count_figure += 1
fig,ax = plt.subplots(nrows=4,ncols=5,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('ae_mode_energyranking')
ax[3,0].set_xticks([])
ax[3,0].set_yticks([])

# first column instantaneous
v_max = np.amax(np.array(Vy)[:2,:,:,time])
v_min = np.amin(np.array(Vy)[:2,:,:,time])
w_max = np.amax(np.array(Vz)[:2,:,:,time])
w_min = np.amin(np.array(Vz)[:2,:,:,time])
idx = 0
for i in range(2): 
    ax[idx,0].imshow(Vy[i][:,:,time],'jet',vmin=v_min,vmax=v_max)
    ax[idx+1,0].imshow(Vz[i][:,:,time],'jet',vmin=w_min,vmax=w_max)
    idx += 2

# second column mean
v_max = np.amax(np.mean(np.array(Vy)[:2,:,:,:],-1))
v_min = np.amin(np.mean(np.array(Vy)[:2,:,:,:],-1))
w_max = np.amax(np.mean(np.array(Vz)[:2,:,:,:],-1))
w_min = np.amin(np.mean(np.array(Vz)[:2,:,:,:],-1))
idx = 0
for i in range(2):
    ax[idx,1].imshow(np.mean(Vy[i],-1),'jet',vmin=v_min,vmax=v_max)
    ax[idx+1,1].imshow(np.mean(Vz[i],-1),'jet',vmin=w_min,vmax=w_max)
    idx += 2

# column 3-5 POD modes of ae modes
idx = 0
for i in range(2):
    for iphi in range(3): # plot v
        pltV = POD_modes[i][:,iphi]
        pltV = np.reshape(pltV,[2*Ny,Nz])
        pltV = pltV[Ny:,:]
        ax[idx,iphi+2].imshow(pltV,'jet')
    for iphi in range(3): # plot w
        pltV = POD_modes[i][:,iphi]
        pltV = np.reshape(pltV,[2*Ny,Nz])
        pltV = pltV[0:Ny,:]
        ax[idx+1,iphi+2].imshow(pltV,'jet')
    idx += 2

ax[0,0].text(-1,-1.5,'instantaneous')
ax[0,1].text(6,-1.5,'mean')
ax[0,3].text(-27,-1.5,'POD modes of the autoencoder modes')
label = 'ae mode ' + str(important_ae_modes[0]+1)
ax[0,0].text(-6,32,label,rotation='vertical')
label = 'ae mode ' + str(important_ae_modes[1]+1)
ax[2,0].text(-6,32,label,rotation='vertical')


# ========================= ae mode contribution ranking ============================
# look up matplotlib.gridspec
count_figure += 1
fig,ax = plt.subplots(nrows=4,ncols=5,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('ae_mode_contributionranking')
ax[3,0].set_xticks([])
ax[3,0].set_yticks([])

# first column instantaneous
v_max = np.amax(np.array(Vy)[2:,:,:,time])
v_min = np.amin(np.array(Vy)[2:,:,:,time])
w_max = np.amax(np.array(Vz)[2:,:,:,time])
w_min = np.amin(np.array(Vz)[2:,:,:,time])
idx = 0
for i in range(2,4): 
    ax[idx,0].imshow(Vy[i][:,:,time],'jet',vmin=v_min,vmax=v_max)
    ax[idx+1,0].imshow(Vz[i][:,:,time],'jet',vmin=w_min,vmax=w_max)
    idx += 2

# second column mean
v_max = np.amax(np.mean(np.array(Vy)[2:,:,:,:],-1))
v_min = np.amin(np.mean(np.array(Vy)[2:,:,:,:],-1))
w_max = np.amax(np.mean(np.array(Vz)[2:,:,:,:],-1))
w_min = np.amin(np.mean(np.array(Vz)[2:,:,:,:],-1))
idx = 0
for i in range(2,4):
    ax[idx,1].imshow(np.mean(Vy[i],-1),'jet',vmin=v_min,vmax=v_max)
    ax[idx+1,1].imshow(np.mean(Vz[i],-1),'jet',vmin=w_min,vmax=w_max)
    idx += 2

# column 3-5 POD modes of ae modes
idx = 0
for i in range(2,4):
    for iphi in range(3): # plot v
        pltV = POD_modes[i][:,iphi]
        pltV = np.reshape(pltV,[2*Ny,Nz])
        pltV = pltV[Ny:,:]
        ax[idx,iphi+2].imshow(pltV,'jet')
    for iphi in range(3): # plot w
        pltV = POD_modes[i][:,iphi]
        pltV = np.reshape(pltV,[2*Ny,Nz])
        pltV = pltV[0:Ny,:]
        ax[idx+1,iphi+2].imshow(pltV,'jet')
    idx += 2

ax[0,0].text(-1,-1.5,'instantaneous')
ax[0,1].text(6,-1.5,'mean')
ax[0,3].text(-27,-1.5,'POD modes of the autoencoder modes')
label = 'ae mode ' + str(important_ae_modes[2]+1)
ax[0,0].text(-6,32,label,rotation='vertical')
label = 'ae mode ' + str(important_ae_modes[3]+1)
ax[2,0].text(-6,32,label,rotation='vertical')


# ============================= Data modes =============================
count_figure += 1

fig,ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('data_POD')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

# first column instantaneous
ax[0,0].imshow(vy[:,:,time],'jet')
ax[1,0].imshow(vz[:,:,time],'jet')
# second column mean 
ax[0,1].imshow(np.mean(vy,-1),'jet')
ax[1,1].imshow(np.mean(vz,-1),'jet')

#column 3-5 POD modes of data
for iphi in range(3):
    pltV = POD_modes_data[:,iphi]
    pltV = np.reshape(pltV,[2*Ny,Nz])
    ax[0,iphi+2].imshow(pltV[Ny:,:],'jet') # v
    ax[1,iphi+2].imshow(pltV[0:Ny,:],'jet') # w

ax[0,0].text(-1,-1.5,'instantaneous')
ax[0,1].text(6,-1.5,'mean')
ax[0,3].text(0,-1.5,'POD modes')


# ============================ POD spectrum ======================
count_figure += 1
plt.figure('POD_spectrum')
plt.plot(np.cumsum(lam_data/np.sum(lam_data)),label='data')
for i in range(4):
    label = 'ae mode ' + str(important_ae_modes[i]+1)
    plt.plot(np.cumsum(POD_lam[i]/np.sum(POD_lam[i])),label=label)
plt.legend()

# ======================= hierarchical with POD ===================
count_figure += 1
fig,ax = plt.subplots(nrows=4,ncols=5,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('hierarchical_modes')
ax[3,0].set_xticks([])
ax[3,0].set_yticks([])

# first column instantaneous
v_max = np.amax(np.array(Vy_h)[:2,:,:,time])
v_min = np.amin(np.array(Vy_h)[:2,:,:,time])
w_max = np.amax(np.array(Vz_h)[:2,:,:,time])
w_min = np.amin(np.array(Vz_h)[:2,:,:,time])
idx = 0
for i in range(2): 
    # ax[idx,0].imshow(Vy_h[i][:,:,time],'jet',vmin=v_min,vmax=v_max)
    # ax[idx+1,0].imshow(Vz_h[i][:,:,time],'jet',vmin=w_min,vmax=w_max)
    ax[idx,0].imshow(Vy_h[i][:,:,time],'jet')
    ax[idx+1,0].imshow(Vz_h[i][:,:,time],'jet')
    idx += 2

# second column mean
v_max = np.amax(np.mean(np.array(Vy_h)[:2,:,:,:],-1))
v_min = np.amin(np.mean(np.array(Vy_h)[:2,:,:,:],-1))
w_max = np.amax(np.mean(np.array(Vz_h)[:2,:,:,:],-1))
w_min = np.amin(np.mean(np.array(Vz_h)[:2,:,:,:],-1))
idx = 0
for i in range(2):
    # ax[idx,1].imshow(np.mean(Vy_h[i],-1),'jet',vmin=v_min,vmax=v_max)
    # ax[idx+1,1].imshow(np.mean(Vz_h[i],-1),'jet',vmin=w_min,vmax=w_max)
    ax[idx,1].imshow(np.mean(Vy_h[i],-1),'jet')
    ax[idx+1,1].imshow(np.mean(Vz_h[i],-1),'jet')
    idx += 2

# column 3-5 POD modes of ae modes
idx = 0
for i in range(2):
    for iphi in range(3): # plot v
        pltV = POD_modes_h[i][:,iphi]
        pltV = np.reshape(pltV,[2*Ny,Nz])
        pltV = pltV[Ny:,:]
        ax[idx,iphi+2].imshow(pltV,'jet')
    for iphi in range(3): # plot w
        pltV = POD_modes_h[i][:,iphi]
        pltV = np.reshape(pltV,[2*Ny,Nz])
        pltV = pltV[0:Ny,:]
        ax[idx+1,iphi+2].imshow(pltV,'jet')
    idx += 2

ax[0,0].text(-1,-1.5,'instantaneous')
ax[0,1].text(6,-1.5,'mean')
ax[0,3].text(-32,-1.5,'POD modes of the hierarchical ae modes')
ax[0,0].text(-6,45,'hierarchical ae mode 1',rotation='vertical')
label = 'ae mode ' + str(important_ae_modes[1]+1)
ax[2,0].text(-6,45,'hierarchical ae mode 2',rotation='vertical')

# ======================== MD-CNN-AE dmd modes by energy======================
count_figure += 1
fig,ax = plt.subplots(nrows=4,ncols=3,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('ae_dmd_mode_energyranking')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

idx = 0
for i in range(2):
    lam_r = np.abs(DMD_lam[i])
    lam_f = np.angle(DMD_lam[i]/(2*np.pi)/dt)
    for iphi in range(3): 
        # plot v
        ax[idx,iphi].imshow(DMD_modes[i][Ny:,:,iphi],'jet')
        # plot w
        ax[idx+1,iphi].imshow(DMD_modes[i][:Ny,:,iphi],'jet')
        title = str(iphi+1)+', a='+str(np.around(lam_r[iphi],decimals=2))+" f="+str(np.around(lam_f[iphi],1))
        ax[idx,iphi].set_title(title)
    idx += 2

fig.suptitle('DMD modes of the autoencoder modes')
label = 'ae mode ' + str(important_ae_modes[0]+1)
ax[0,0].text(-6,32,label,rotation='vertical')
label = 'ae mode ' + str(important_ae_modes[1]+1)
ax[2,0].text(-6,32,label,rotation='vertical')


# ======================== MD-CNN-AE dmd modes by contribution ======================
count_figure += 1
fig,ax = plt.subplots(nrows=4,ncols=3,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('ae_dmd_mode_contributionranking')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

idx = 0
for i in range(2,4):
    lam_r = np.abs(DMD_lam[i])
    lam_f = np.angle(DMD_lam[i]/(2*np.pi)/dt)
    for iphi in range(3): 
        # plot v
        ax[idx,iphi].imshow(DMD_modes[i][Ny:,:,iphi],'jet')
        # plot w
        ax[idx+1,iphi].imshow(DMD_modes[i][:Ny,:,iphi],'jet')
        title = str(iphi+1)+', a='+str(np.around(lam_r[iphi],decimals=2))+" f="+str(np.around(lam_f[iphi],1))
        ax[idx,iphi].set_title(title)
    idx += 2

fig.suptitle('DMD modes of the autoencoder modes')
label = 'ae mode ' + str(important_ae_modes[2]+1)
ax[0,0].text(-6,32,label,rotation='vertical')
label = 'ae mode ' + str(important_ae_modes[3]+1)
ax[2,0].text(-6,32,label,rotation='vertical')


# ========================= Data dmd modes ========================
count_figure += 1
lam_r = np.abs(DMD_lam_data)
lam_f = np.angle(DMD_lam_data/(2*np.pi)/dt)
fig,ax = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True)
fig.canvas.manager.set_window_title('data_DMD')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
for iphi in range(3):
    ax[0,iphi].imshow(DMD_modes_data[Ny:,:,iphi],'jet')
    ax[1,iphi].imshow(DMD_modes_data[:Ny,:,iphi],'jet')
    title = str(iphi+1)+', a='+str(np.around(lam_r[iphi],decimals=2))+" f="+str(np.around(lam_f[iphi],1))
    ax[0,iphi].set_title(title)

fig.suptitle('DMD modes')


plt.show()