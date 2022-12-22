import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model



@tf.function
def get_gradient_m_z(z:tf.Tensor,decoder:Model) -> tf.Tensor:
    '''Combute the gradients of dm/dz, where m is the decomposed field and z is the latent variable.\n
    
    Arguments:\n
        z: latent variable, given in tf.tensor. First dimension must be batch size.\n
        decoder: a decoder.\n
    
    Returns:\n
        dm_dz: [batch_size, ..., number of latent variables]
    '''
    with tf.GradientTape() as tape:
        tape.watch(z)
        pred = decoder(z,training=False)
    dm_dz = tape.batch_jacobian(pred,z)
    return dm_dz




def one_step_integrate(t, dx_dt, t0, x0, dt):
    '''Numerical integration from derivatives. Compute x from t and dx/dt, given x=x0 when t=t0.\n
    
    Arguments (everthing should be at float32):\n
        t: array\n
        dx_dt: derivative of x with respect to t at provided t.\n
        t0: initial condition.\n
        x0: initial condition.\n
        dt: interval between two consecutive t.\n
    
    Returns:
        x: array of x at provided t.
    '''

    x = [0]
    for i in range(0,len(t)-1):
        x.append(x[i] + dt*dx_dt[i])
    x = np.array(x)
    idx_0 = np.squeeze(np.argwhere(np.abs(t-t0)<1.19209e-07)) # machine percision for float32
    # idx_0 = np.squeeze(np.argwhere(t==t0))
    x = x - x[idx_0] + x0
    return x 