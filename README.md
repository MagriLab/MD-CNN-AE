# MD-CNN-AE

Mode decomposition with autoencoders. 

This repo is linked to the paper under review [Decoder Decomposition for the Analysis of the Latent Space of Nonlinear Autoencoders With Wind-Tunnel Experimental Data](https://arxiv.org/abs/2404.19660).
For lecture material please go to branch **lecture-material-may2024**.

## Content:

    1. Models for mode decomposing autoencoder (MD-CNN-AE) (Murata, Fukami & Fukagata, 2020), 
       hierarchical autoencoder (Fukami, Nakamura & fukagata, 2020), 
       and standard autoencoder. All as keras model subclass.
    2. Methods for POD and DMD (Brunton and Kutz, 2019).
    3. Ranking methods for MD-CNN-AE -- cross entropy, signal energy and contribution.
    4. Two data files from the same wake experiment, downsampled to different sizes. Original data from George Rigas.

## Python versions:

    - python 3.9.7 
    - tensorflow 2.7.0 
    - numpy 1.20.3 
    - matplotlib 3.4.3 
    - jax 0.4.9

## Setting up:
Create a file **_system.ini** that contains results saving location. Example config files are available in *examples/config_files/*

    [system_info]
    save_location=/home/results_location # required, training results will be saved at this location
    alternate_location=/home/experiment_results # optional, this is needed for running repeated experiments with multiprocessing, can be the same as the previous

### For running experiments with multiprocessing script

Create a config file using the template **examples/config_files/_train_md_ae.ini**. 
Create a file **_train_mp.ini** and save it in *experiments-mp*. See **examples/config_files/_train_mp.ini**. Specify the config file and python file to run as shown in the example. 
Run **start_mp_training.py**

## References:

Brunton, S. L. & Kutz, J. N. (2019) Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. 1st edition. <br>
            &nbsp;&nbsp;&nbsp;&nbsp;Cambridge, UK, Cambridge Univeristy Press. Chapter 7.<br>
Murata T.,Fukami K. & Fukagata K., "Nonlinear mode decomposition with convolutional neural networks for fluid dynamics," J. Fluid Mech. <br>
            &nbsp;&nbsp;&nbsp;&nbsp;Vol. 882, A13 (2020). https://doi.org/10.1017/jfm.2019.822<br>
Fukami, K., Nakamura, T. & Fukagata, K. <br>
            &nbsp;&nbsp;&nbsp;&nbsp;(2020) Convolutional neural network based hierarchical autoencoder for nonlinear mode decomposition of fluid field data. Physics of Fluids. <br>
            &nbsp;&nbsp;&nbsp;&nbsp;32 (9), 095110. 10.1063/5.0020721. <br>
