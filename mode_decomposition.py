# contains POD and DMD 
import numpy as np
import sys


# Performs POD on a data matrix
# Originally a matlab script
def POD(X,method='auto',weight='ones'):
    # Input:
    # X: data array, the last dimension must be time
    # method: 'auto' will select the method base on the size of the data matrix. 
    #       'classic' is suitable for when Nx < Nt
    #       'snapshot' is suitable for when Nx > Nt
    # Return:
    # Q_POD: matrix, each column is a mode, sorted in descending order
    # lam: a vector of eigenvalues, sorted in descending order

    # Prepare data matrix Q
    Q = np.copy(X)
    original_shape = Q.shape
    nt = Q.shape[-1]
    Q = np.reshape(Q,(-1,nt))
    nx = Q.shape[0]
    
    # remove mean
    x_mean = np.mean(Q,axis=1)
    for ti in range(0,nt):
        Q[:,ti] = Q[:,ti] - x_mean;  
    
    # set weights
    if weight == 'ones':
        weight = np.ones((nx,nt))

    # choose POD method
    if method == 'auto':
        if nt > nx:
            typePOD = 'classic'
            print("Use classic POD, Nx < Nt.")
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
        normQ = (np.matmul(Q_POD.T,Q_POD*weight).real)**0.5
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
        print("Method does not exist. Please choose between 'auto', 'classic' or 'snapshot'.")
        sys.exit()
    print("POD done.")

    Q_POD = np.reshape(Q_POD,original_shape)

    return Q_POD, lam