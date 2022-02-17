from os import name
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, concatenate, BatchNormalization, Conv2DTranspose, Flatten, PReLU, Reshape, Dropout, AveragePooling2D, Add, Lambda, Layer, TimeDistributed, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import numpy as np

class Encoder(Model):
    def __init__(self,Nx,Nu,features_layers=[1],latent_dim=16,
        filter_window=(3,3),act_fct='tanh',batch_norm=False,drop_rate=0.0, lmb=0.0, **kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.Nx = Nx
        self.Nu = Nu
        self.features_layers = features_layers
        self.latent_dim = latent_dim
        self.filter_window = filter_window
        self.act_fct = act_fct
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate
        self.lmb =lmb
        input_shape = (self.Nx[0],self.Nx[1],self.Nu)
        self.input_img = Input(shape = input_shape)

        # define layers
        self.add_layers = []
        for i in range(len(self.features_layers)):
            self.add_layers.append(Conv2D(self.features_layers[i],self.filter_window, padding='same',kernel_regularizer=l2(self.lmb), bias_regularizer=l2(self.lmb), activation=self.act_fct))
            
            if self.batch_norm:
                self.add_layers.append(BatchNormalization())
            
            self.add_layers.append(MaxPool2D(pool_size=(2,2), padding='same'))
            self.add_layers.append(Dropout(self.drop_rate))

        self.dense = Dense(self.latent_dim,kernel_regularizer=l2(self.lmb), bias_regularizer=l2(self.lmb),activation=self.act_fct)
        self.last_dropout = Dropout(self.drop_rate)
        
        # define model
        self.out = self.call(self.input_img)

    def call(self,inputs,training=False):
        x = inputs
        for layer in self.add_layers:
            x = layer(x)

        x = Flatten()(x)
        x = self.dense(x)
        return self.last_dropout(x)

    def summary(self): # override the .summary() so it would print layer shape
        encoder = Model(self.input_img,self.out)
        return encoder.summary()

    def get_layer_shape(self): # obtain the layer shape for use in decoder
        layer_size = []
        x = self.input_img
        for _ in range(len(self.features_layers)):
            x = MaxPool2D(pool_size=(2,2), padding='same')(x)
            layer_size.append(x.get_shape().as_list()[1:3])
        return layer_size




class Decoder(Model):
    def __init__(self,Nx,Nu,layer_size,features_layers=[1],latent_dim=1,
        filter_window=(3,3),act_fct='tanh',batch_norm=False,drop_rate=0.0, lmb=0.0,resize_meth='bilinear',**kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.Nx = Nx
        self.Nu = Nu
        self.layer_size = layer_size # from Encoder.get_layer_shape
        self.features_layers = features_layers
        self.latent_dim = latent_dim
        self.filter_window = filter_window
        self.act_fct = act_fct
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate
        self.lmb = lmb
        self.resize_meth = resize_meth

        prelatent_size = np.prod(self.layer_size[-1],initial=self.features_layers[-1])
        # prelatent_size = np.prod(self.layer_size[-1])

        self.add_layers = [] # store layers

        self.add_layers.append(Dense(prelatent_size,kernel_regularizer=l2(lmb), bias_regularizer=l2(lmb), activation=act_fct)) # restore the number of elements in the layer before the latent space
        if self.batch_norm:
            self.add_layers.append(BatchNormalization())
        self.add_layers.append(Dropout(self.drop_rate))
        self.add_layers.append(Reshape((self.layer_size[-1][0],self.layer_size[-1][1],self.features_layers[-1])))
        for i in range(len(self.features_layers)-1):
            self.add_layers.append(ResizeImages((self.layer_size[-i-2]),self.resize_meth))
            self.add_layers.append(Conv2D(self.features_layers[-i-2],self.filter_window, padding='same',kernel_regularizer=l2(self.lmb), bias_regularizer=l2(self.lmb), activation=self.act_fct))
            if self.batch_norm:
                self.add_layers.append(BatchNormalization())

            self.add_layers.append(Dropout(self.drop_rate))

        self.add_layers.append(ResizeImages((self.Nx[0],self.Nx[1]),self.resize_meth)) 
        self.add_layers.append(Conv2D(self.Nu,self.filter_window, padding='same',kernel_regularizer=l2(self.lmb), bias_regularizer=l2(self.lmb), activation='linear')) # last layer

        # define input shape
        input_shape = (self.latent_dim,) 
        self.input_img = Input(shape = input_shape)
        # define output
        self.out = self.call(self.input_img)

    def call(self,inputs,training=False):
        x = inputs
        for layer in self.add_layers:
            x = layer(x)

        return x

    def summary(self): # override the .summary() so it would print layer shape
        decoder = Model(self.input_img,self.out)
        return decoder.summary()



class Autoencoder(Model):
    def __init__(self,Nx,Nu,features_layers=[1],latent_dim=1,
        filter_window=(3,3),act_fct='tanh',batch_norm=False,
        drop_rate=0.0, lmb=0.0,resize_meth='bilinear',**kwargs):
        super(Autoencoder,self).__init__(**kwargs)

        self.Nx = Nx # [num1,num2]
        self.Nu = Nu
        self.features_layers = features_layers
        self.latent_dim = latent_dim
        self.filter_window = filter_window
        self.act_fct = act_fct
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate
        self.lmb =lmb
        self.resize_meth = resize_meth

        # ENCODER
        self.encoder = Encoder(Nx=self.Nx,Nu=self.Nu,features_layers=self.features_layers,latent_dim=self.latent_dim,filter_window=self.filter_window,act_fct=self.act_fct,batch_norm=self.batch_norm,drop_rate=self.drop_rate,lmb=self.lmb)
        # DECODER
        layer_size = self.encoder.get_layer_shape()
        self.decoder = Decoder(Nx=self.Nx,Nu=self.Nu,layer_size=layer_size,features_layers=self.features_layers,latent_dim=self.latent_dim,filter_window=self.filter_window,act_fct=self.act_fct,batch_norm=self.batch_norm,drop_rate=self.drop_rate,lmb=self.lmb)


        input_shape = (self.Nx[0],self.Nx[1],self.Nu)
        self.input_img = Input(shape = input_shape)
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



class MD_Autoencoder(Model):
    def __init__(self,Nx,Nu,features_layers=[1],latent_dim=2,
        filter_window=(3,3),act_fct='tanh',batch_norm=False,
        drop_rate=0.0, lmb=0.0,resize_meth='bilinear', *args, **kwargs):
        # see Murata, Fukami and Fukagata. 2020. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics. J. Fluid Mech.
        self.Nx = Nx
        self.Nu = Nu
        self.features_layers = features_layers
        self.latent_dim = latent_dim
        self.filter_window = filter_window
        self.act_fct = act_fct
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate
        self.lmb = lmb
        self.resize_meth = resize_meth

        if latent_dim == 1:
            print("Latent dimension is 1, use standard autoencoder instead")

        input_shape = (Nx[0],Nx[1],Nu)
        self.input_img = Input(shape = input_shape)

        super(MD_Autoencoder,self).__init__(*args, **kwargs)
        
        # Define layers here, not in call

        # DEFINE ENCODER
        self.encoder = Encoder(Nx=Nx,Nu=Nu,features_layers=features_layers,latent_dim=latent_dim,filter_window=filter_window,act_fct=act_fct,batch_norm=batch_norm,drop_rate=drop_rate,lmb=lmb)
        layer_size = self.encoder.get_layer_shape()
        # DEFINE DECODER
        # one decoder only decodes a scalar
        # this decoder is just for setting up the structure, not used in call()
        self.decoder = Decoder(Nx=Nx,Nu=Nu,layer_size=layer_size,features_layers=features_layers,latent_dim=1,filter_window=filter_window,act_fct=act_fct,batch_norm=batch_norm,drop_rate=drop_rate,lmb=lmb,resize_meth=resize_meth,name='decoder')

        # Define layers
        self.name_lambda = self.name_layer(prefix='z')
        self.name_decoder = self.name_layer(prefix='decoder')
        self.layer_decoder_group = []
        for i in range(0,self.latent_dim):
            self.layer_decoder_group.append((Lambda(lambda x: x[:,i:i+1],name=self.name_lambda[i]),Decoder(Nx=Nx,Nu=Nu,layer_size=layer_size,features_layers=features_layers,latent_dim=1,filter_window=filter_window,act_fct=act_fct,batch_norm=batch_norm,drop_rate=drop_rate,lmb=lmb,resize_meth=resize_meth,name=self.name_decoder[i])))
        
        # print(self.layer_decoder_group)
        # for lam,decoder in self.layer_decoder_group:
        #     print(lam)
        #     print(decoder)

        self.layer_add = Add()

        self.out = self.call(self.input_img)
        
        # re-initialise
        super(MD_Autoencoder, self).__init__(inputs=self.input_img,outputs=self.out,**kwargs)

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

    # def summary(self):
    #     mdl = Model(self.input_img,self.out)
    #     return mdl.summary()

    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(inputs=self.input_img,outputs=self.out)

    def name_layer(self, prefix='MyLayer'):
        # returns a list of string ['MyLayer_0', 'MyLayer_1', ...]
        names = []
        for i in range(0,self.latent_dim):
            name = prefix + '_' + str(i)
            names.append(name)
        return names



class ResizeImages(Layer):
    """Resize Images to a specified size

    # Arguments
        output_size: Size of output layer width and height
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """
    def __init__(self, output_dim=(1, 1),resize_meth='bilinear', **kwargs):
        self.output_dim = output_dim
        super(ResizeImages, self).__init__(**kwargs)
        #data_format = conv_utils.normalize_data_format(data_format)
        #self.output_dim = conv_utils.normalize_tuple(output_dim, 2, 'output_dim')
        #self.input_spec = InputSpec(ndim=4)
        self.resize_meth = resize_meth
        if resize_meth=='bicubic':
            self.resize_method = tf.image.ResizeMethod.BICUBIC
        else:
            self.resize_method = tf.image.ResizeMethod.BILINEAR

    def build(self, input_shape):
        super(ResizeImages, self).build(input_shape)
        #self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
#        if self.data_format == 'channels_first':
#            return (input_shape[0], input_shape[1], self.output_dim[0], self.output_dim[1])
#        elif self.data_format == 'channels_last':
         return (input_shape[0], self.output_dim[0], self.output_dim[1], input_shape[3])

#    def _resize_fun(self, inputs, data_format):
#        try:
#            assert keras.backend.backend() == 'tensorflow'
#            assert self.data_format == 'channels_last'
#        except AssertionError:
#            print( "Only tensorflow backend is supported for the resize layer and accordingly 'channels_last' ordering")
#        output = tf.resize_images(inputs, self.output_dim,method=self.resize_method)
#        return output

    def call(self, inputs):
#        output = self._resize_fun(inputs=inputs, data_format=self.data_format)
        output = tf.image.resize(inputs, self.output_dim,method=self.resize_method)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                'resize_meth':self.resize_meth}
        base_config = super(ResizeImages, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

