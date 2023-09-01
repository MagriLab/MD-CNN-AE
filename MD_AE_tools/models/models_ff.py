from os import name
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float32')
from typing import Union

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization, Conv2DTranspose, Flatten, PReLU, Reshape, Dropout, AveragePooling2D, Add, Lambda, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


import numpy as np

class Encoder(Model):
    def __init__(self,
                 input_shape:int,
                 layer_sizes:list = [200,200,100],
                 latent_dim:int = 16,
                 act_fct:str = 'tanh',
                 batch_norm:bool = False,
                 drop_rate:float = 0.0, 
                 lmb:Union[float,list] = 0.0, 
                 keras_mdl_kwargs:dict = {}, 
                 **kwargs):
        super().__init__(**keras_mdl_kwargs)

        self.act_fct = act_fct
        self.layer_sizes = layer_sizes
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate

        if 'last_act' in kwargs:
            self.last_act = kwargs['last_act']
            print('setting custum last layer activation for the encoder.')
        else:
            self.last_act = act_fct

        if isinstance(lmb,float):
            self.lmb = [lmb, lmb]



        self.input_img = Input(shape=(input_shape,))

        self.layers = []

        for l in layer_sizes:

            self.layers.append(
                Dense(l, activation=self.act_fct, use_bias=False, kernel_initializer=l2(self.lmb[0]))
            )

            if batch_norm:
                self.layers.append(
                    BatchNormalization()
                )
            
            self.layers.append(
                Dropout(drop_rate)
            )

        # last layer
        self.layers.append(
            Dense(latent_dim, activation=self.last_act, use_bias=False, kernel_regularizer=l2(self.lmb[0]))
        )

        self.out = self.call(self.input_img)
        
    def call(self,inputs,training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x

    def summary(self): # override the .summary() so it would print layer shape
        encoder = Model(self.input_img,self.out)
        return encoder.summary()





class Decoder(Model):
    def __init__(self,
                 output_shape:int,
                 layer_sizes:list = [100,200,200],
                 latent_dim:int = 16,
                 act_fct:str = 'tanh',
                 batch_norm:bool = False,
                 drop_rate:float = 0.0, 
                 lmb:Union[float,list] = 0.0, 
                 keras_mdl_kwargs:dict = {}, 
                 **kwargs):
        super().__init__(**keras_mdl_kwargs)

        


        if isinstance(lmb,float):
            self.lmb = [lmb, lmb]
        
        if 'first_act' in kwargs:
            self.first_act = kwargs['first_act']
            print('setting custom first layer activation for the decoder.')
        else:
            self.first_act = self.act_fct

        self.input_img = Input(shape=(latent_dim,))
        
        self.layers = []
        
        ## first layer  
        self.layers.append(
            Dense(layer_sizes[0], activation=self.first_act, use_bias=False, kernel_regularizer=l2(self.lmb[0]))
        )
        if batch_norm:
            self.layers.append(BatchNormalization())    
        self.layers.append(Dropout(drop_rate))

        for l in layer_sizes[1:]:

            self.layers.append(
                Dense(l, activation=act_fct, use_bias=False, kernel_regularizer=l2(self.lmb[0]))
            )

            if batch_norm:
                self.layers.append(
                    BatchNormalization()
                )    
            
            self.layers.append(
                Dropout(drop_rate)
            )

        ## last layer
        self.layers.append(
            Dense(output_shape, activation='linear', use_bias=False, kernel_regularizer=l2(self.lmb[0]))
        )

        self.out = self.call(self.input_img)
        

    def call(self,inputs,training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x

    def summary(self): # override the .summary() so it would print layer shape
        decoder = Model(self.input_img,self.out)
        return decoder.summary()



class MD_Autoencoder(Model):
    def __init__(self,
                 input_shape:int,
                 encoder_layers:list = [200,200,100],
                 decoder_layers:list = [100,200,200],
                 latent_dim:int = 2,
                 act_fct:str = 'tanh',
                 batch_norm:bool = False,
                 drop_rate:float = 0.0, 
                 lmb:Union[float,list] = 0.0,
                 encoder_kwargs={}, 
                 decoder_kwargs={}, 
                 *args, 
                 **kwargs):
        # see Murata, Fukami and Fukagata. 2020. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics. J. Fluid Mech.

        self.latent_dim = latent_dim
        self.act_fct = act_fct

        if latent_dim == 1:
            sys.exit("Latent dimension is 1, use standard autoencoder instead")

        self.input_img = Input(shape = (input_shape,))

        super(MD_Autoencoder,self).__init__(*args, **kwargs)
        
        # Define layers here, not in call

        # DEFINE ENCODER
        self.encoder = Encoder(
            input_shape=input_shape,
            layer_sizes=encoder_layers,
            latent_dim=latent_dim,
            act_fct=act_fct,
            batch_norm=batch_norm,
            drop_rate=drop_rate,
            lmb=lmb,
            **encoder_kwargs
        )
        
        # Define layers
        self.name_lambda = self.name_layer(prefix='z')
        self.name_decoder = self.name_layer(prefix='decoder')
        self.layer_decoder_group = []
        for i in range(0,self.latent_dim):
            self.layer_decoder_group.append((
                Lambda(
                    lambda x,i: x[:,i:i+1],
                    arguments={'i':i},
                    name=self.name_lambda[i]
                ),
                Decoder(
                    output_shape = input_shape,
                    layer_sizes = decoder_layers,
                    latent_dim = 1,
                    act_fct = act_fct,
                    batch_norm = batch_norm,
                    drop_rate = drop_rate, 
                    lmb = lmb,
                    keras_mdl_kwargs={'name':self.name_decoder[i]}, 
                    **decoder_kwargs)
                ))
        
        self.layer_add = Add()

        self.out = self.call(self.input_img)
        
        # # re-initialise
        # super(MD_Autoencoder, self).__init__(inputs=self.input_img,outputs=self.out,**kwargs)

    def call(self,inputs,training=None):
        encoded = self.encoder(inputs)
        modes = []
        for lam,decoder in self.layer_decoder_group:
            x = lam(encoded)
            x = decoder(x)
            modes.append(x)
            del x
        out = self.layer_add(modes)
        return out

    def summary(self):
        mdl = Model(self.input_img,self.out)
        return mdl.summary()

    # def build(self):
    #     # Initialize the graph
    #     self._is_graph_network = True
    #     self._init_graph_network(inputs=self.input_img,outputs=self.out)

    def name_layer(self, prefix='MyLayer'):
        # returns a list of string ['MyLayer_0', 'MyLayer_1', ...]
        names = []
        for i in range(0,self.latent_dim):
            name = prefix + '_' + str(i)
            names.append(name)
        return names
    
    def get_encoder(self):
        return self.encoder
    
    # returns a list with all decoders
    def get_decoders(self):
        decoders = []
        for name in self.name_decoder:
            decoders.append(self.get_layer(name))
        return decoders


class Autoencoder(Model):
    def __init__(self,
                 input_shape:int,
                 encoder_layers:list = [200,200,100],
                 decoder_layers:list = [100,200,200],
                 latent_dim:int = 2,
                 act_fct:str = 'tanh',
                 batch_norm:bool = False,
                 drop_rate:float = 0.0, 
                 lmb:Union[float,list] = 0.0,
                 encoder_kwargs={}, 
                 decoder_kwargs={}, 
                 *args, 
                 **kwargs):
        super(Autoencoder,self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.act_fct = act_fct

        # ENCODER
        self.encoder = Encoder(
            input_shape=input_shape,
            layer_sizes=encoder_layers,
            latent_dim=latent_dim,
            act_fct=act_fct,
            batch_norm=batch_norm,
            drop_rate=drop_rate,
            lmb=lmb,
            **encoder_kwargs
        )
        
        # DECODER
        self.decoder = Decoder(
            output_shape = input_shape,
            layer_sizes = decoder_layers,
            latent_dim = latent_dim,
            act_fct = act_fct,
            batch_norm = batch_norm,
            drop_rate = drop_rate, 
            lmb = lmb, 
            **decoder_kwargs
        )


        self.input_img = Input(shape = (input_shape,))
        self.out = self.call(self.input_img)
    
        # re-initialise, with the build() function
        # super(Autoencoder, self).__init__(inputs=self.input_img,outputs=self.out,**kwargs)

    def call(self,inputs,training=False): # input as a layer
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    # def build(self):
    #     # Initialize the graph
    #     self._is_graph_network = True
    #     self._init_graph_network(inputs=self.input_img,outputs=self.out)

    def summary(self):
        encoder = Model(self.input_img,self.out)
        return encoder.summary()    
