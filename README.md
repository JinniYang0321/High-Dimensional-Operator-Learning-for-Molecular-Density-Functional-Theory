# High-Dimensional Operator Learning for Molecular Density Functional Theory

This repository contains code, data sets and models corresponding to the following publication:

**High-Dimensional Operator Learning in Molecular Density Functional Theory**  
*Jinni Yang, Runtong Pan, Jikai Sun, and Jianzhong Wu, [arXiv:2411.03698](https://arxiv.org/abs/2411.03698).*


### Setup

There are three packages that need to be used in the code, namely 'numpy', 'scipy', and 'torch'. The required packages can be installed with `pip install -r requirements.txt`.


### Instructions

In the `Data` directory, the raw simulation data named `original_data.npz` can be found, which was obtained with the program for GCMC simulation of CO2, provided in `CO2_GCMC.py`. The program named `Training_data_generator.py` can be used to generate the dataset for training.

In the `Model` directory, the trained model named `model.pth` can be found and the high dimensional operator learning is implemented in the program named `High_Dimensional_Operator.py`. The spherical harmonic expansion is implemented in the program named `Spherical_Harmonic_Expansion`.
