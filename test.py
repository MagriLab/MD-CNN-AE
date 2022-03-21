import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Conv2D, Concatenate, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd

num_of_ts=10000 # number of time slice
x_num=384; y_num=192; # no need to change
act = 'linear'
input_img = Input(shape=(y_num,x_num,1))
input_enc_1 = Input(shape=(1,))
input_enc_2 = Input(shape=(1,))

x1 = Conv2D(16, (3,3),activation=act, padding='same')(input_img)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Reshape([3*6*4])(x1)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(32,activation=act)(x1)
x1 = Dense(16,activation=act)(x1)

x1 = Dense(1,activation=act)(x1)
# x1 = Concatenate()([x1, input_enc_1,input_enc_2])

# x1 = Dense(16,activation=act)(x1)
# x1 = Dense(32,activation=act)(x1)
# x1 = Dense(64,activation=act)(x1)
# x1 = Dense(72,activation=act)(x1)
# x1 = Reshape([3,6,4])(x1)
# x1 = UpSampling2D((2,2))(x1)
# x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
# x1 = UpSampling2D((2,2))(x1)
# x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
# x1 = UpSampling2D((2,2))(x1)
# x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
# x1 = UpSampling2D((2,2))(x1)
# x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
# x1 = UpSampling2D((2,2))(x1)
# x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
# x1 = UpSampling2D((2,2))(x1)
# x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)

# x_final = Conv2D(1, (3,3),padding='same')(x1)
# autoencoder = Model([input_img,input_enc_1,input_enc_2], x_final)
# autoencoder.compile(optimizer='adam', loss='mse')


a = [x1]
a.extend([])
x1 = Concatenate()(a)

autoencoder = Model([input_img,input_enc_1,input_enc_2], x1)
# print(autoencoder.summary())

s = autoencoder.layers[17].output_shape[1]
# print(s)
# print(input_enc_1)
# inn = Input(shape=s)
# print(inn)

from MD_AE_model import Autoencoder as AE

ae = AE(Nx=[10,10],Nu=2,features_layers=[10,30],latent_dim=5,)
# print(ae.summary())
print(ae.layers[0].output_shape)

print(np.sum([1,1,1]))