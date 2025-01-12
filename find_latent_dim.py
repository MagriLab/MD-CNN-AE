import h5py
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
mse = tf.keras.losses.MeanSquaredError()
import MD_AE_tools.models.models_ff as modelff

from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[2], 'GPU')# use [] for cpu only, gpus[i] for the ith gpu
        tf.config.set_logical_device_configuration(gpus[2],[tf.config.LogicalDeviceConfiguration(memory_limit=25000)]) # set hard memory limit
        # tf.config.experimental.set_memory_growth(gpus[2], True) # allow memory growth
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)



# =================== Parameters ============================
# fname = Path('./_results/find_latent_dim_2to10.h5')
# tempfn_ae = './temp_weights_2to10.h5'
# fname = Path('./_results/find_latent_dim_16to24.h5')
# tempfn_ae = './temp_weights_16to24.h5'
# fname = Path('./_results/find_latent_dim_6and9.h5')
# tempfn_ae = './temp_weights_6and9.h5'
fname = Path('./_results/find_latent_dim_4.h5')
tempfn_ae = './temp_weights_4.h5'
if fname.exists():
    raise ValueError('File for writing results already exists.')


nfft_psd = 1024
overlap = nfft_psd/2
sampling_freq = 720
D = 196.5 #mm
Uinf = 15 #m/s

## ae configuration
lmb = 0.0000 #1e-05 #regulariser
drop_rate = 0.0
act_fct = 'tanh'
batch_norm = True

## feedforward ae configuration
encoder_layers = [128,256,256,128,64]
decoder_layers = [64,128,256,256,128]


## training
nb_epoch = 4000
batch_size = 200000
learning_rate = 0.004

lrschedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    learning_rate,
    100,
    t_mul=50,
    m_mul=0.9
)


# latent_dim_list = [2,3,5,10]
# latent_dim_list = [16,19,20,24]
# latent_dim_list = [6,9]
latent_dim_list = [4]
num_repeats = 3


# ================ Data ================

with h5py.File('./data/raw_pressure_long.h5','r') as hf:
    print(hf.keys())
    fs = np.squeeze(hf.get('fs'))
    static_p = np.squeeze(hf.get('static_p'))
    esp_allt = np.array(hf.get('esp')).T
    r = np.array(hf.get('r')).T
    theta = np.array(hf.get('theta')).T
x=(np.cos(theta*np.pi/180).T)*r
y=(np.sin(theta*np.pi/180).T)*r 
x = x.flatten()
y = y.flatten()

pmean = np.mean(esp_allt,axis=1).reshape(8,8)
prms = np.std(esp_allt,axis=1)

[n,nt] = esp_allt.shape
input_shape = n

p_train = esp_allt - pmean.flatten()[:,np.newaxis]
p_train = p_train.T



# ============ repeats =================
loss_history = np.zeros((len(latent_dim_list),num_repeats,nb_epoch))
loss_best = np.zeros((len(latent_dim_list),num_repeats))



for i in range(len(latent_dim_list)):
    for j in range(num_repeats):

        print(f'Starting test {i} with {latent_dim_list[i]} latent variables, repeat no. {j}')
        
        ae = modelff.Autoencoder(
            input_shape = input_shape,
            encoder_layers = encoder_layers,
            decoder_layers = decoder_layers,
            latent_dim = latent_dim_list[i],
            act_fct = act_fct,
            batch_norm = batch_norm,
            drop_rate = drop_rate,
            lmb = lmb
        )

        model_cb = ModelCheckpoint(
            tempfn_ae,
            monitor='loss',
            save_best_only=True,
            verbose=0,
            save_weights_only=True
        )
        cb = [model_cb]

        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrschedule),loss='mse')

        hist = ae.fit(
            p_train,
            p_train,
            epochs=nb_epoch,
            batch_size=batch_size,
            shuffle=True,
            callbacks=cb,
            verbose=2
        )
        loss_history[i,j,:] = hist.history['loss']
        
        ae.load_weights(tempfn_ae)
        l2error = ae.evaluate(p_train,p_train,batch_size=batch_size)
        
        loss_best[i,j] = l2error


# =============== results =====================
print('Writing results')
f = h5py.File(fname,'x')
f.create_dataset('latent_dim_list', data=latent_dim_list)
f.create_dataset('encoder_layers',data=encoder_layers)
f.create_dataset('decoder_layers',data=decoder_layers)
f.create_dataset('act_fct',data=act_fct)
f.create_dataset('batch_norm',data=batch_norm)
f.create_dataset('drop_rate',data=drop_rate)
f.create_dataset('lmb',data=lmb)
f.create_dataset('learning_rate',data=learning_rate)


f.create_dataset('loss_best',data=loss_best)
f.create_dataset('loss_history',data=loss_history)
f.close()
