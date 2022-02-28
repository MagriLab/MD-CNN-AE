import numpy as np
from matplotlib import pyplot as plt
from mode_decomposition import *
import h5py

# parameters
D       = 196.5;    # Model diameter in mm
Uinf    = 15;       # Nominal free stream velocity
fPIV    = 720;      # PIV sampling frequency
dt      = 1/fPIV;   # Delta t between image pairs
#=================== downsampled data ======================
filename = "./PIV4_downsampled_by8.h5" # .h5 file of data
hf = h5py.File(filename,'r')
z = np.array(hf.get('z'))
y = np.array(hf.get('y'))
vy = np.array(hf.get('vy'))
vz = np.array(hf.get('vz'))
hf.close()
print("Finished loading data.")
# print(vy.shape)
[nt,nz,ny] = vz.shape

vy = np.transpose(vy,[2,1,0])
vz = np.transpose(vz,[2,1,0]) #(ny,nz,nt)
#==================== prepare data =========================
# build matrix
Q = np.vstack((vz,vy)) # new shape [2*ny,nz,nt]
# Q = np.reshape(Q,(2*nz*ny,nt)) # [2*ny*nz,nt]
v_true = np.copy(Q)
print(Q.shape)


a = POD(Q)

PlotWhichVelocity = 'v' 

fig = plt.figure(1)
# title = "Mode in decoder " + str(WhichDecoder+1)
title = "POD Modes "+PlotWhichVelocity
plt.suptitle(title)
for iphi in range(9):
    ax = plt.subplot(3,3,iphi+1,title=str(iphi+1),xticks=[],yticks=[])
    pltV = a.modes[:,iphi];
    pltV = np.reshape(pltV,[2*ny,nz])
    if PlotWhichVelocity == 'w': # Q was built [vz,vy]
        pltV = pltV[0:ny,:]
    elif PlotWhichVelocity == 'v':
        pltV = pltV[ny:,:]
    elif PlotWhichVelocity == 'V':
        pltV = (pltV[0:ny,:]**2 + pltV[ny:,:]**2)**0.5
    
    ax = plt.imshow(pltV,'jet')
    plt.colorbar()
plt.show()