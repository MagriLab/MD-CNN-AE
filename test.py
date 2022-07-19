import project_specific_utils.training as training
from MD_AE_tools.models.models import MD_Autoencoder
from tensorflow.keras.optimizers import Adam


Ntrain = 1500 # snapshots for training
Nval = 632 # sanpshots for validation
Ntest = 600

SHUFFLE = False # shuffle before splitting into sets, test set is extracted before shuffling
REMOVE_MEAN = True # train on fluctuating velocity
data_type = 'float32'


read_data_file = './data/PIV4_downsampled_by8.h5'
Nz = 24 # grid size
Ny = 21
Nu = 2
Nt = 2732 # number of snapshots available
Nx = [Ny, Nz]


latent_dim = 10
act_fct = 'tanh'
batch_norm = True
drop_rate = 0.2
learning_rate = 0.001
learning_rate_list = [learning_rate,learning_rate/10,learning_rate/100]
lmb = 0.001
features_layers = [32, 64, 128]
filter_window = (3,3)
batch_size = 100
loss = 'mse'
epochs = 1

u_all, data, data_shuffle, data_mean = training.data_partition(read_data_file,[Nt,Nz,Ny,Nu],[Ntrain,Nval,Ntest],SHUFFLE=False,REMOVE_MEAN=True)


md_ae = MD_Autoencoder(Nx=Nx,Nu=Nu,features_layers=features_layers,latent_dim=latent_dim,filter_window=filter_window,act_fct=act_fct,batch_norm=batch_norm,drop_rate=drop_rate,lmb=lmb)
md_ae.compile(optimizer=Adam(learning_rate=learning_rate_list[0]),loss=loss)

hist_train, hist_val, mse_test = training.train(md_ae,data,batch_size,epochs)
print(hist_train,hist_val,mse_test)

# # import configparser
# # a = configparser.ConfigParser()
# # a.read('_test_config.ini')


# # class GetAttr(object):
# #     def __init__(self, _dict):
# #         self.__dict__.update(_dict)


# # # a = GetAttr(a)
# # # print(a.a1)

# # def nested(a): # a is two layer nested dict
# #     dict1 = {}
# #     a = dict(a)
# #     for key in a:
# #         b = dict(a[key])
# #         b = GetAttr(b)
# #         dict1.update({key:b})
# #     dict2 = GetAttr(dict1)
# #     return dict2

# # dict_nested = nested(a)

# # print(dict_nested.group2.b3)


