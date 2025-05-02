# High-Dimensional Operator Learning for Molecular Density Functional Theory

This repository contains code, data sets and models corresponding to the following publication:

**High-Dimensional Operator Learning in Molecular Density Functional Theory**  
*Jinni Yang, Runtong Pan, Jikai Sun, and Jianzhong Wu, [arXiv:2411.03698](https://arxiv.org/abs/2411.03698).*


### Setup

There are three packages that need to be used in the code, namely 'numpy', 'scipy', and 'torch'. The required packages can be installed with `pip install -r requirements.txt`.


### Instructions

The raw simulation data named 'original_data.npz' can be found in `Data` and the program named 'Training_data_generator.py' can be used to generate the dataset for training.
The trained model is located in `Model`.
The program for GCMC simulation of CO2 is provided in 'CO2_GCMC.py' and the high dimensional operator learning is implemented in `High_dimensional_operator.py`.
The program for spherical harmonic expansion can be found in 'Spherical_harmonic_expansion' and there is a demo data for illustration named 'Demo_data.txt'
