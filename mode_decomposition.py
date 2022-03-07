# contains POD and DMD 
from crypt import methods
from re import S
from time import time
import numpy as np
import sys

# Performs DMD
# reference: Brunton, S. L. & Kutz, J. N. (2019) Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control . 1st edition. Cambridge, UK, Cambridge Univeristy Press. Chapter 7.
class DMD:
    # Inputs:
    # X: [x0, x1 ... xn-1], X can have any shape, but the last dimension must be time
    # Xprime: [x1, x2, ... xn], has the same number of snapshots as X
    # r: how many modes to keep. r <= rank(Sigma), r<=n
    # keep_shape: if true, the returned matrix 'self.modes' will have the same spatial dimension as the data matrix X
    #       example X has shape (2,3,2,10), then self.modes will have shape (2,3,2,modes)

    # Attributes
    # self.r: how many modes to keep
    # self.keep_shape: keep_shape
    # self.grid_shape: the shape of the grid
    # self.modes: DMD modes, the last dimension is time
    # self.lam: eigenvalues
    # self.b: amplitude

    # Methods
    # get_modes(): returns modes, lambda and amplitude
    # get_frequency(dt=1): dt is the interval between snapshots. Returns the rate of growth/decay and the corresponding physical frequency. 
    # get_amplitude(): returns |b|, which is the contribution of each mode
    def __init__(self,X,Xprime,r,keep_shape=False):
        self.r = r
        self.keep_shape = keep_shape

        # prepare data
        nx, nt, self.grid_shape, Q, Qprime = self.prepare_data(X,Xprime)
        self.modes, self.lam, self.b = self.dmd(Q,Qprime,self.r)

        if keep_shape:
            self.restore_shape()

    def prepare_data(self,X,Xprime):
        Q = np.copy(X)
        Qprime = np.copy(Xprime)
        grid_shape = list(Q.shape[:-1])
        nt = Q.shape[-1]
        Q = np.reshape(Q,(-1,nt))
        Qprime = np.reshape(Qprime,(-1,nt))
        nx = Q.shape[0]
        return nx,nt,grid_shape,Q,Qprime
    
    def dmd(self,Q,Qprime,r):
        U,Sigma,VT = np.linalg.svd(Q,full_matrices=0) # Step 1
        Ur = U[:,:r]
        Sigmar = np.diag(Sigma[:r])
        VTr = VT[:r,:]
        Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Qprime @ VTr.T).T).T # Step 2
        Lambda, W = np.linalg.eig(Atilde) # Step 3
        Lambda = np.diag(Lambda)
        
        Phi = Qprime @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4
        alpha1 = Sigmar @ VTr[:,0]
        b = np.linalg.solve(W @ Lambda,alpha1)

        # sort modes based on their contribution ('amplitude')
        idx = np.argsort(np.abs(b))
        idx = np.flip(idx)
        b = b[idx]
        Lambda = np.diag(np.diag(Lambda)[idx])
        Phi = Phi[:,idx]

        return Phi, Lambda, b
    
    def restore_shape(self):
        original_shape = self.grid_shape.copy()
        original_shape.extend([-1])
        self.modes = np.reshape(self.modes,original_shape)

    def get_modes(self):
        return self.modes, self.lam, self.b

    def get_frequency(self,dt=1.):
        lam = np.diag(self.lam)
        lam_r = np.abs(lam) # rate of decay/growth
        lam_f = np.angle(lam)/(2*np.pi)/dt # physical frequency
        return lam_r, lam_f

    def get_amplitude(self):
        return np.abs(self.b)




# calculate POD modes, contains a reconstruction function
# Originally a matlab script
class POD:
    # Input:
    # X: data array, the last dimension must be time
    # method: 'auto' will select the method base on the size of the data matrix. 
    #       'classic' is suitable for when Nx < Nt
    #       'snapshot' is suitable for when Nx > Nt
    # weight: weight applied to each snapshot
    # keep_shape: if true, the returned matrix 'self.modes' will have the same spatial dimension as the data matrix X
    #       example X has shape (2,3,2,10), then self.modes will have shape (2,3,2,modes)
    
    # Attributes:
    # self.nt: number of snapshots
    # self.Q: reshaped X'
    # self.Q_mean: time averaged X
    # self.typePOD: which method to use
    # self.Phi: eigenvectors Phi
    # self.modes: matrix of POD modes
    # self.lam: eigenvalues
    # self.nt: number of snapshots
    # self.nx: total number of grid points
    # self.w: weights
    # self.grid_shape: the shape of the grid

    # Methods:
    # reconstruct(Q_mean,Q,Phi,number_of_modes,t,shape='self'): use this to reconstruct a snapshot at time t with chosen number of modes. 
    # get_modes(): returns modes and eigenvalues.
    
    def __init__(self,X,method='auto',weight='ones',keep_shape=False):
        self.method = method
        self.weight = weight
        self.keep_shape = keep_shape

        # Prepare data matrix Q
        self.nx,self.nt,self.grid_shape,self.Q,self.Q_mean = self.prepare_data(X)
        self.w = self.set_weight()


        # choose POD method
        print("Calculating POD ...")
        if self.method == 'auto':
            if self.nt >= self.nx:
                self.typePOD = 'classic'
                print("Use classic POD, Nx <= Nt.")
                self.modes, self.lam, self.Phi = self.classic_POD(self.Q)
            else:
                self.typePOD = 'snapshot'
                print("Use snapshot POD, Nx > Nt.")
                self.modes, self.lam, self.Phi = self.snapshot_POD(self.Q)
        else:
            self.typePOD = self.method
            print("User has selected " + self.typePOD + " POD")
            if self.typePOD == 'classic':
                self.modes, self.lam, self.Phi = self.classic_POD(self.Q)
            elif self.typePOD == 'snapshot':
                self.modes, self.lam, self.Phi = self.snapshot_POD(self.Q)
            else:
                sys.exit("Method does not exist. Please choose between 'auto', 'classic' or 'snapshot'.")
        print("POD done.")

        if self.keep_shape:
            self.restore_shape()


    def prepare_data(self,X):
        Q = np.copy(X)
        grid_shape = list(Q.shape[:-1])
        nt = Q.shape[-1]
        Q = np.reshape(Q,(-1,nt))
        nx = Q.shape[0]

        # remove mean
        Q_mean = np.mean(Q,axis=1)
        for ti in range(0,nt):
            Q[:,ti] = Q[:,ti] - Q_mean;  

        return nx,nt,grid_shape,Q,Q_mean


    def set_weight(self):
        if self.weight == 'ones':
            weights = np.ones((self.nx,1))
        return weights

    
    def classic_POD(self,Q):
        C = Q @ ((Q.T)*(self.w.T)) # 2-point spatial correlation tesnsor: Q*Q'
        # print('C is Hermitian?',np.allclose(C,np.conj(C.T)))
        lam,Phi = np.linalg.eigh(C) # right eigenvectors and eigenvalues
        idx = np.argsort(lam) # sort
        idx = np.flip(idx)
        Q_POD = Phi[:,idx]
        lam = lam[idx]
        Phi = np.copy(Q_POD)
        # normalise energy in the weighted inner product
        normQ = (Q_POD.T @ Q_POD*self.w).real**0.5
        Q_POD = Q_POD@np.diag(1/np.diag(normQ))
        return Q_POD, lam, Phi
    
    
    def snapshot_POD(self,Q):
        C = (Q.T) @ (Q*self.w) # 2-point temporal correlation tesnsor: Q'*Q 
        lam,Phi = np.linalg.eigh(C)
        idx = np.argsort(np.abs(lam)) # sort
        idx = np.flip(idx)
        Phi = Phi[:,idx]
        lam = lam[idx]
        # get spatial POD modes: PSI = Q*Phi
        Q_POD = (Q@Phi)*(1/(lam**0.5).T)
        return Q_POD, lam, Phi


    def restore_shape(self):
        original_shape = self.grid_shape.copy()
        original_shape.extend([-1])
        self.modes = np.reshape(self.modes,original_shape)

    def reconstruct(self,t,number_of_modes,Q_mean='self',Q='self',Phi='self',shape='self'):
        # Input: 
        # shape: reshape the reconstructed vector into this shape, if not given the result is reshaped to self.grid_shape
        # If any of 'Q_mean', 'Q', or 'Phi' is not given, the method uses the values stored in self
        if Q_mean == 'self' or Q == 'self' or Phi == 'self':
            Q_mean = self.Q_mean
            Q = self.Q
            Phi = self.Phi
        if self.typePOD == 'classic':
            # temporal coefficient A
            self.A = Q.T @ Phi #(nt,nx)
            Q_add = Phi[:,0:number_of_modes] @ self.A[:,0:number_of_modes].T
            Q_add = Q_add[:,t]
        elif self.typePOD == 'snapshot':
            # spatial coefficient A
            self.A = Q @ Phi #(nx,nt)
            Q_add = self.A[:,0:number_of_modes] @ Phi[:,0:number_of_modes].T
            Q_add = Q_add[:,t]
        rebuildv = Q_mean + Q_add
        if shape == 'self':
            rebuildv = np.reshape(rebuildv,self.grid_shape)
        else:
            rebuildv = np.reshape(rebuildv,shape)
        return rebuildv
    

    def get_modes(self):
        return self.modes,self.lam