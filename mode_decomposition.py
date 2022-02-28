# contains POD and DMD 
import numpy as np
import sys


# Performs POD on a data matrix
# Originally a matlab script
def POD(X,method='auto',weight='ones',keep_shape=False):
    # Input:
    # X: data array, the last dimension must be time
    # method: 'auto' will select the method base on the size of the data matrix. 
    #       'classic' is suitable for when Nx < Nt
    #       'snapshot' is suitable for when Nx > Nt
    # weight: weight applied to each snapshot
    # keep_shape: if true, the returned matrix 'Q_POD' will have the same spatial dimension as the data matrix X
    #       example X has shape (2,3,2,10), then Q_POD will have shape (2,3,2,modes)
    # Return:
    # Q_POD: matrix, each column is a mode, sorted in descending order
    # lam: a vector of eigenvalues, sorted in descending order

    # Prepare data matrix Q
    Q = np.copy(X)
    grid_shape = list(Q.shape[:-1])
    nt = Q.shape[-1]
    Q = np.reshape(Q,(-1,nt))
    nx = Q.shape[0]
    
    # remove mean
    x_mean = np.mean(Q,axis=1)
    for ti in range(0,nt):
        Q[:,ti] = Q[:,ti] - x_mean;  
    
    # set weights
    if weight == 'ones':
        weight = np.ones((nx,1))

    # choose POD method
    if method == 'auto':
        if nt >= nx:
            typePOD = 'classic'
            print("Use classic POD, Nx <= Nt.")
        else:
            typePOD = 'snapshot'
            print("Use snapshot POD, Nx > Nt.")
    else:
        typePOD = method
        print("User has selected " + typePOD + " POD")
    
    # calculate POD
    print("Calculating POD ...")
    if typePOD == 'classic':
        C = Q @ ((Q.T)*(weight.T)) # 2-point spatial correlation tesnsor: Q*Q'
        lam,Phi = np.linalg.eigh(C) # right eigenvectors and eigenvalues
        idx = np.argsort(lam) # sort
        idx = np.flip(idx)
        Q_POD = Phi[:,idx]
        lam = lam[idx]
        # normalise energy in the weighted inner product
        normQ = (Q_POD.T @ Q_POD*weight).real**0.5
        Q_POD = Q_POD@np.diag(1/np.diag(normQ))
    elif typePOD == 'snapshot':
        C = (Q.T) @ (Q*weight) # 2-point temporal correlation tesnsor: Q'*Q 
        lam,Phi = np.linalg.eigh(C)
        idx = np.argsort(np.abs(lam)) # sort
        idx = np.flip(idx)
        Phi = Phi[:,idx]
        lam = lam[idx]
        # get spatial POD modes: PSI = Q*Phi
        Q_POD = (Q@Phi)*(1/(lam**0.5).T)
    else:
        sys.exit("Method does not exist. Please choose between 'auto', 'classic' or 'snapshot'.")
    print("POD done.")

    if keep_shape:
        grid_shape.extend([-1])
        Q_POD = np.reshape(Q_POD,grid_shape)

    return Q_POD, lam


# Performs DMD
# reference: Brunton, S. L. & Kutz, J. N. (2019) Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control . 1st edition. Cambridge, UK, Cambridge Univeristy Press. Chapter 7.
def DMD(X,Xprime,r,keep_shape=False):
    # X: [x0, x1 ... xn-1], X can have any shape, but the last dimension must be time
    # Xprime: [x1, x2, ... xn], has the same number of snapshots as X
    # r: how many modes to keep. r <= rank(Sigma), r<=n
    # return:
    # Phi: square matrix, the column is the modes. Modes are sorted in descending order using b
    # Lambda: square matrix of eigenvalues. 
    # b: a vector of amplitude

    Q = np.copy(X)
    Qprime = np.copy(Xprime)
    grid_shape = list(Q.shape[:-1])
    nt = Q.shape[-1]
    Q = np.reshape(Q,(-1,nt))
    Qprime = np.reshape(Qprime,(-1,nt))
    nx = Q.shape[0]


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

    if keep_shape:
        grid_shape.extend([-1])
        Phi = np.reshape(Phi,grid_shape)
    
    return Phi, Lambda, b