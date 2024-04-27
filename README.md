# Using DeepONets as surrogates in complex simulations

Pytorch implementation of *DeepONet* and its multiple-output variant which will be referred to as *MultiONet*. The purpose of this implementation is to use the models as surrogates to speed up simulations. The datasets and applications used here mainly come from astrophyics, e.g. modelling chemical reaction networks (coupled ODEs) or learning the time-evolutions of spectra. However, DeepONet as a neural operator can be applied in any setting where there is a mapping from functions to functions that is sufficiently well-behaved. It also seems effective in learning mappings between state vectors rather than functions.

Currently the repo is still WIP. The goal is to provide a guide to the whole process of obtaining a usable surrogate model: 
1. Choosing the architecture best suited to the goal - *DeepONet* or *MultiONet*.
1. Formatting the dataset.
1. Initial (short) training and evaluation of the model to ensure that data and architecture are working.
1. Hyperparameter optimization using OPTUNA.
1. Full training of the model with the optimal hyperparameters.
1. If applicable: Optimizing the model for efficiency.
1. If applicable: Deploying the model for use in a pipeline.


