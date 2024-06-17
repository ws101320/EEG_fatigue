CWSTR-Net: A Channel-Weighted Spatial-Temporal Residual Network
Based on NonSmooth Nonnegative Matrix Factorization for Fatigue
Detection Using EEG Signals

# Introduce
Offline unsupervised processing for EEG data channel weights.

While the main implementation of CWSTR-Net is in Python, the nsNMF algorithm that underpins the channel weighting functionality is based on the NMFLibrary-master, a pure-Matlab library that houses various algorithms for non-negative matrix factorization.

The nsNMF Channel Weighting Algorithm：algorithmrank1.m 
Adjusting the rank yields different weights.

Mapping weights to channels for each sample：algorithm.m 


# Installation:

* Python 3.10
* Pytorch 2.0.0+cu118
* Numpy  1.23.5
* Scikit-learn  1.2.1
* scipy  1.9.3


