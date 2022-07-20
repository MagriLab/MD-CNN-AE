import project_specific_utils.training as training
from MD_AE_tools.models.models import MD_Autoencoder
from tensorflow.keras.optimizers import Adam


# Ntrain = 1500 # snapshots for training
# Nval = 632 # sanpshots for validation
# Ntest = 600

# SHUFFLE = False # shuffle before splitting into sets, test set is extracted before shuffling
# REMOVE_MEAN = True # train on fluctuating velocity
# data_type = 'float32'


# read_data_file = './data/PIV4_downsampled_by8.h5'
# Nz = 24 # grid size
# Ny = 21
# Nu = 2
# Nt = 2732 # number of snapshots available
# Nx = [Ny, Nz]


# latent_dim = 10
# act_fct = 'tanh'
# batch_norm = True
# drop_rate = 0.2
# learning_rate = 0.001
# learning_rate_list = [learning_rate,learning_rate/10,learning_rate/100]
# lmb = 0.001
# features_layers = [32, 64, 128]
# filter_window = (3,3)
# batch_size = 100
# loss = 'mse'
# epochs = 1


def test_var(a):
    print('variable type: ', type(a), ', value: ', a)