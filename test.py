from MD_AE_tools.models.models_no_bias import Autoencoder_ff
from pathlib import Path 

w = Path('../../Results/MD-CNN-AE/experiment_ff_ae_data/no_cutoff_10-3792699/weights.h5')
# print(w.exists())

mdl=Autoencoder_ff(1008,2,[450,150,50],act_fct='relu',drop_rate=0.12)

mdl.build((None,1008))
mdl.summary()
mdl.load_weights(w)

