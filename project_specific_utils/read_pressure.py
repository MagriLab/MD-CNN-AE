import numpy as np
from scipy import interpolate, ndimage
from enum import IntEnum

# functions are from PIV matlab script


_r = np.array([[0.05597964,0.11195929,0.16793893,0.22391857,0.27989823,0.33587787,0.3918575 ,0.44783714]],dtype='float32')
_theta = np.array([[270.,315.,0.,45.,90.,135.,180.,225.]],dtype='float32')


class PIVdata(IntEnum):
    PIV3 = 0
    PIV4 = 1
    PIV5 = 2
    PIV7 = 3
    PIV9 = 4
    PIV10 = 5
    PIV12 = 6
    PIV13 = 7
    PIV14 = 8
    PIV15 = 9
    PIV16 = 10 
    PIV17 = 11
    PIV18 = 12




def interp(p:np.ndarray, 
                theta:np.ndarray = _theta, 
                r:np.ndarray = _r,
                nint:int = 15,
                method:str = 'cubic',
                filter:bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''Interpolate the raw pressure data on a polar grid.\n
    
    p: numpy array of raw pressure data, with shape [theta,r] or flattened\n
    theta, r: arrays of  angles and radius\n
    nint: int, (nint-1) number of points are added between each measurements\n
    method: will be passed to the 'kind' argument scipy.interpolate.interp2d.\n
    filter: default True, whether to filter the interpolated values with gaussian filter. \n

    Return:\n
        x1_interp,y1_interp: arrays of x,y coordinates of the interpolated points.\n
        p1_interp: pressure at the coordinates
    '''


    # prepare data / reshaping 
    p1 = p.reshape((8,8)) # has shape [theta, r]
    x1=np.cos(np.concatenate([theta,[[theta[0,0]]]],axis=1)*np.pi/180).T @ np.concatenate([[[0]], r],axis=1); 
    y1=np.sin(np.concatenate([theta,[[theta[0,0]]]],axis=1)*np.pi/180).T @ np.concatenate([[[0]], r],axis=1);

    # Periodic
    c1 = np.concatenate([p1,p1[[0],:]],axis=0)# p1 has shape [theta, r]

    # Add center point r=0
    _center = np.array([np.mean(p1[:,0])]*9).reshape((9,1))
    c1 = np.concatenate([_center,c1],axis=1)


    # Interpolation
    x1_interp = interpolate.interp2d(np.arange(9), np.arange(9), x1, kind=method)(np.arange(0, 8.01, 1/nint), np.arange(0, 8.01, 1/nint))
    y1_interp = interpolate.interp2d(np.arange(9), np.arange(9), y1, kind=method)(np.arange(0, 8.01, 1/nint), np.arange(0, 8.01, 1/nint))
    p1_interp = interpolate.interp2d(np.arange(9), np.arange(9), c1, kind=method)(np.arange(0, 8.01, 1/nint), np.arange(0, 8.01, 1/nint))

    # Filtering
    if filter:
        myfilter = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16.0  # Gaussian filter approximation

        p1_interp = ndimage.convolve(p1_interp, myfilter, mode='nearest')

    return x1_interp, y1_interp, p1_interp



def cop(p:np.ndarray ,r:np.ndarray = _r, theta:np.ndarray = _theta):
    '''Centre of pressure for a pressure snapshot.\n
    
    p: numpy array of raw pressure data, with shape [theta,r] or flattened\n
    theta, r: arrays of  angles and radius\n

    Return:\n
        rx,ry: the x,y coordinates of the centre of pressure
    '''

    ptmp = p.reshape(8,8)
    dAi = np.zeros((1,8))
    ri = np.zeros((1,8))
    dr = r[0,1]-r[0,0]
    dth = 2*np.pi/8
    for i in range(1,8):
        ri[0,i] = 0.5*(r[0,i]+r[0,i-1])
        dAi[0,i] = ri[0,i]*dr*dth
    ptrapz = 0.5*(ptmp[:,:-1]+ptmp[:,1:])
    dAi = dAi[[0],1:]
    ri = ri[[0],1:]
    xi = np.cos(theta*np.pi/180).T * ri##
    yi = np.sin(theta*np.pi/180).T * ri
    dA = np.tile(dAi,(8,1))
    sum_pa = np.sum(ptrapz * dA)
    rx = (1/sum_pa) * (np.sum(ptrapz*xi*dA))
    ry = (1/sum_pa) * (np.sum(ptrapz*yi*dA))

    return rx,ry