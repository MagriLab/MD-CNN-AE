import sys
sys.path.append('..')

import json
import typing
from typing import Tuple
import wandb
from pathlib import Path
import tensorflow as tf
import numpy as np
import h5py

StrOrPath = typing.Union[Path,str]



from .helpers import read_config_ini
from .data_and_train import data_partition, create_update_fn
from MD_AE_tools.mode_decomposition import POD
from MD_AE_tools.models.models_no_bias import Autoencoder_ff



class Train_ff_ae():
    def __init__(self) -> None:
        pass


    def import_config(self,training_file) -> None:
        params = read_config_ini(training_file)

        self.nb_epoch = params['training'].getint('nb_epoch')
        self.batch_size = params['training'].getint('batch_size')
        self.learning_rate = params['training'].getfloat('learning_rate')
        self.lmb = params['training'].getfloat('lmb')
        self.source = params['training'].getint('source')
        if params['training']['cut_off'] == 'None':
            self.cut_off = None
        else:
            self.cut_off = float(params['training']['cut_off'])

        self.latent_dim = params['ae_config'].getint('latent_dim')
        self.drop_rate = params['ae_config'].getfloat('drop_rate')
        self.layers = json.loads(params['ae_config']['layers'])
        self.act_fct = params['ae_config']['act_fct']
        self.loss = params['ae_config']['loss']
        if self.loss == 'mse':
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        _optimiser = params['ae_config']['optimiser']
        if _optimiser == 'adam':
            self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.read_data_file = Path(params['data']['read_data_file'])
        self.RAW_DATA = params['data'].getboolean('RAW_DATA')
        self.SHUFFLE = params['data'].getboolean('SHUFFLE')
        self.REMOVE_MEAN = params['data'].getboolean('REMOVE_MEAN')
        self.Nz = params['data'].getint('Nz')
        self.Ny = params['data'].getint('Ny')
        self.Nu = params['data'].getint('Nu')
        self.Nt = params['data'].getint('Nt')
        self.Ntrain = params['data'].getint('Ntrain')
        self.Nval = params['data'].getint('Nval')
        self.Ntest = params['data'].getint('Ntest')
        self.data_type = params['data']['data_type']

        self.LOG_WANDB = params['wandb'].getboolean('LOG_WANDB')
        self.project_name = params['wandb']['project_name']
        self.group_name = params['wandb']['group_name']

        self.save_to = params['system']['save_to']
        self.folder_prefix = params['system']['folder_prefix']

    def set_save_location(self,system_save_path:StrOrPath, folder_suffix:str) -> Path:
        '''Make a folder to save results
        
        Folder_path is made by combining the system_save_path, folder_prefix and folder_suffix.
        Return folder_path.
        '''
        folder_name = self.folder_prefix + folder_suffix
        folder_path = Path(system_save_path,self.save_to,folder_name)
        folder_path.mkdir(parents=True, exist_ok=False)
        return folder_path

    def read_data(self,random_seed:typing.Optional[int]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:


        if self.RAW_DATA:
            if random_seed:
                self.random_seed = random_seed
                rng = np.random.default_rng(self.random_seed)
            else: 
                self.random_seed = np.random.randint(1,1000)
                rng = np.random.default_rng(self.random_seed)
            (u_train, u_val, u_test), _, self.idx_unshuffle = self.read_raw_data(rng=rng)
            u_train = np.squeeze(u_train).astype(self.data_type)
            u_val = np.squeeze(u_val).astype(self.data_type)
            u_test = np.squeeze(u_test).astype(self.data_type)
        else:
            with h5py.File(self.read_data_file,'r') as hf:
                u_train = np.squeeze(np.array(hf.get('u_train'))).astype(self.data_type)
                u_val = np.squeeze(np.array(hf.get('u_val'))).astype(self.data_type)
                u_test = np.squeeze(np.array(hf.get('u_test'))).astype(self.data_type)

        if self.source == 1:
            data_train, data_val, data_test, self._pod_with_data = self.prepare_coeff_data(u_train,u_val,u_test,self.batch_size,self.cut_off)
        elif self.source == 2:
            data_train, data_val, data_test = self.prepare_image_data(u_train,u_val,u_test,self.batch_size)
        else:
            raise ValueError('Unsupported input type.')
        return data_train, data_val, data_test


    def read_raw_data(self,rng):
        _, (u_train, u_val, u_test), (_,idx_unshuffle), (_, u_mean_train, u_mean_val,u_mean_test) = data_partition(
                self.read_data_file,
                [self.Nt,self.Nz,self.Ny,self.Nu],
                [self.Ntrain, self.Nval, self.Ntest],
                SHUFFLE=self.SHUFFLE,
                REMOVE_MEAN=self.REMOVE_MEAN,
                data_type=self.data_type,
                rng=rng
            )
        return (u_train, u_val, u_test), (u_mean_train, u_mean_val, u_mean_test), idx_unshuffle



    @staticmethod
    def prepare_image_data(u_train:np.ndarray,
                            u_val:np.ndarray,
                            u_test:np.ndarray,
                            batch_size:int,
                        ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        ntrain = u_train.shape[0]
        nval = u_val.shape[0]
        ntest = u_test.shape[0]
        
        data_train = tf.data.Dataset.from_tensor_slices((u_train.reshape((ntrain,-1)))).batch(batch_size)
        data_val = tf.data.Dataset.from_tensor_slices((u_val.reshape((nval,-1)))).batch(nval)
        data_test = tf.data.Dataset.from_tensor_slices((u_test.reshape((ntest,-1)))).batch(ntest)

        return data_train, data_val, data_test

    @staticmethod
    def prepare_coeff_data(u_train:np.ndarray,
                            u_val:np.ndarray,
                            u_test:np.ndarray,
                            batch_size:int,
                            cut_off:int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        ntrain = u_train.shape[0]
        nval = u_val.shape[0]
        ntest = u_test.shape[0]

        u_all = np.vstack((u_train,u_val,u_test))
        vy = u_all[:,:,:,0]
        vy = np.transpose(vy,[1,2,0])
        vz = u_all[:,:,:,1]
        vz = np.transpose(vz,[1,2,0])
        X = np.vstack((vz,vy))

        pod = POD(X,method='classic')
        time_coeff = pod.get_time_coefficient.astype('float32') # shape (time, number of points)
        
        if cut_off:
            time_coeff = time_coeff[:,:cut_off] 
        
        data_train = tf.data.Dataset.from_tensor_slices((time_coeff[:ntrain,:])).batch(batch_size)
        data_val = tf.data.Dataset.from_tensor_slices((time_coeff[ntrain:ntrain+nval,:])).batch(nval)
        data_test = tf.data.Dataset.from_tensor_slices((time_coeff[ntrain+nval:,:])).batch(ntest)
        
        return data_train, data_val, data_test, (pod,X)


    def make_model(self):
        
        input_shape = self.cut_off if self.cut_off else self.Ny*self.Nz*self.Nu
        self._mdl = Autoencoder_ff(
            input_shape=input_shape,
            latent_dim=self.latent_dim,
            layer_sizes=self.layers,
            regularisation=self.lmb,
            act_fct=self.act_fct,
            drop_rate=self.drop_rate
        )

        self._mdl.build((None,input_shape))



    def train_model(
        self, 
        data_train:tf.data.Dataset, 
        data_val:tf.data.Dataset, 
        checkpoint:bool=False, 
        temp_file:typing.Optional[StrOrPath]=None, 
        wandb_run=None,
        print_freq:int=1):
        '''Train the model.\n

        Argument:\n
            data_train: batched tf.data.Dataset for training.\n
            data_val: single-batch tf.data.Dataset for validation.\n
            checkpoint: if true, save model weights when validation loss improve.\n
            temp_file: the name of the file to save checkpoint to.\n
            wandb_run: an wandb run object.\n
            print_freq: how frequent to print loss (number of epochs).\n 
        '''

        if checkpoint:
            if not temp_file:
                raise ValueError('Cannot save checkpoint to an un-named file.')
        
        temp_file = Path(temp_file)

        update = create_update_fn(self._mdl,self.loss_fn,self.optimiser)
        
        hist_loss = []
        hist_val = [np.inf]

        for epoch in range(1,self.nb_epoch+1):
            loss_epoch = []
            for xx in data_train:
                # print(xx.shape)
                loss_batch = update(xx,xx)
                loss_epoch.append(loss_batch.numpy())

            loss = np.mean(loss_epoch)
            hist_loss.append(loss)

            # validation
            for xx_val in data_val:
                y_val = self._mdl(xx_val,training=False)
            loss_val = self.loss_fn(xx_val,y_val).numpy()
            if checkpoint and (loss_val < hist_val[-1]): # only save weights if validation improve
                self._mdl.save_weights(temp_file)
            hist_val.append(loss_val)

            if wandb_run:
                wandb_run.log({'loss':loss, 'loss_val':loss_val})
            
            if epoch % print_freq == 0:
                print(f'Epoch {epoch}: loss {loss}, validation loss {loss_val}')

        hist_val.pop(0)
        self._mdl.load_weights(temp_file)

        self._loss_dict = {'train':hist_loss, 'val':hist_val}

    
    @property
    def get_loss_history(self) -> dict:
        if not hasattr(self,'_loss_dict'):
            raise ValueError('Loss history does not exist yet, please train the model first.')
        
        return self._loss_dict
        

    def get_reconstructed_loss(self, data_test:tf.data.Dataset):

        pod,X = self._pod_with_data

        for xx_test in data_test:
            y_test = self._mdl(xx_test,training=False)    
            loss_test = self.loss_fn(xx_test,y_test).numpy()

        if self.source == 'coeff':
            if not self.cut_off:
                m = self.Ny*self.Nz*self.Nu
            else:
                m = self.cut_off
            y_reconstructed = pod.reconstruct(m,A=y_test.numpy()).astype(self.data_type)
            loss_image = self.loss_fn(X[...,(self.Ntrain+self.Nval):],y_reconstructed).numpy()
        else:
            loss_image = loss_test
        
        return loss_test, loss_image


    @property
    def get_model(self):
        return self._mdl

    def set_wandb(self):
        config = {
            'cut_off':self.cut_off,
            'latent_dim':self.latent_dim,
            'REMOVE_MEAN':self.REMOVE_MEAN,
            'batch_size':self.batch_size,
            'learning_rate':self.learning_rate,
            'regularisation':self.lmb,
            'num_layers':len(self.layers),
            'layer':self.layers,
            'activation':self.act_fct,
            'dropout':self.drop_rate
        }
        run = wandb.init(config=config,project=self.project_name,group=self.group_name)
        return run

    def save_weights(self,file_path):
        self._mdl.save_weights(file_path)