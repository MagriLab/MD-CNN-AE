
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float32')

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization, Conv2DTranspose, Flatten, PReLU, Reshape, Dropout, AveragePooling2D, Add, Lambda, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

class Autoencoder_ff(Model):
    def __init__(
        self,
        input_shape:int,
        latent_dim:int,
        layer_sizes:list = [700,300,100],
        regularisation:float = 1e-5,
        act_fct:str = 'tanh',
        drop_rate = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.inn = Input(shape=(input_shape))


        # encoder
        self.encoder_layers = []
        for l in layer_sizes:
            self.encoder_layers.append(
                Dense(l,act_fct,use_bias=False,kernel_regularizer=l2(regularisation))
            )
            if drop_rate > 0.0:
                self.encoder_layers.append(Dropout(drop_rate))
        
        self.encoder_layers.append(
            Dense(latent_dim,act_fct,use_bias=False,kernel_regularizer=l2(regularisation))
        )


        # decoder
        self.decoder_layers = []
        for l in layer_sizes[::-1]:
            self.decoder_layers.append(
                Dense(l,act_fct,use_bias=False,kernel_regularizer=l2(regularisation))
            ) 
            if drop_rate > 0.0:
                self.encoder_layers.append(Dropout(drop_rate))
        # last layer
        self.decoder_layers.append(
            Dense(input_shape,'linear',use_bias=False,kernel_regularizer=l2(regularisation))
        )
        self.out = self.call(self.inn)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x