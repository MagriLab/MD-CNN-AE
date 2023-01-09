import json
import pathlib
import typing
import wandb

StrOrPath = typing.Union[pathlib.Path,str]



from .helpers import read_config_ini



class Train_ff_ae():
    def __init__(self, training_file) -> None:
        pass

    def import_config(self,training_file) -> None:
        params = read_config_ini(training_file)

        # self.LATENT_STATE = params['training'].getboolean('LATENT_STATE')
        # self.SHUFFLE = params['training'].getboolean('SHUFFLE')
        # self.REMOVE_MEAN = params['training'].getboolean('REMOVE_MEAN')
        # self.nb_epoch = params['training'].getint('nb_epoch')
        # self.batch_size = params['training'].getint('batch_size')
        # self.learning_rate = params['training'].getfloat('learning_rate')
        # self.loss = params['training']['loss']

        # self.latent_dim = params['ae_config'].getint('latent_dim')
        # self.lmb = json.loads(params['ae_config']['lmb'])
        # self.drop_rate = params['ae_config'].getfloat('drop_rate')
        # self.features_layers = json.loads(params['ae_config']['features_layers'])
        # self.act_fct = params['ae_config']['act_fct']
        # self.resize_meth = params['ae_config']['resize_meth']
        # self.filter_window = tuple(json.loads(params['ae_config']['filter_window']))
        # self.BATCH_NORM = params['ae_config'].getboolean('BATCH_NORM')
        # self.NO_BIAS = params['ae_config'].getboolean('NO_BIAS')

        self.read_data_file = params['data']['read_data_file']
        self.RAW_DATA = params['data'].getboolean('RAW_DATA')
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

    def set_save_location(self,system_save_path:StrOrPath, folder_suffix:str) -> pathlib.Path:
        '''Make a folder to save results
        
        Folder_path is made by combining the system_save_path, folder_prefix and folder_suffix.
        Return folder_path.
        '''
        folder_name = self.folder_prefix + folder_suffix
        folder_path = pathlib.Path(system_save_path,self.save_to,folder_name)
        folder_path.mkdir(parents=True, exist_ok=False)
        return folder_path

    def read_data(self):

        if self.source == 'coeff':
            self.prepare_coeff_data()
        if self.source == 'data':
            self.prepare_image_data()
        else:
            raise ValueError('Unsupported input type.')
        pass


    def prepare_image_data():
        pass

    def prepare_coeff_data():
        pass


    def make_model(self):
        pass

    def train_model(self, temp_file, train_data, val_data, checkpoint=False, wandb_run=None):
        pass


    def get_reconstructed_loss():
        pass

    @property
    def get_model(self):
        return self._mdl

    def set_wandb(self):
        config = {}
        run = wandb.init()
        return run