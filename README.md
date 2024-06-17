# CWSTR-Net: A Channel-Weighted Spatial-Temporal Residual Network Based on NonSmooth Nonnegative Matrix Factorization for Fatigue Detection Using EEG Signals

## Introduction

CWSTR-Net employs offline unsupervised processing for determining EEG data channel weights. While the primary implementation of CWSTR-Net is in Python, the **nsNMF** algorithm, which underpins the channel weighting functionality, is derived from **NMFLibrary-master**, a pure-Matlab library containing various algorithms for non-negative matrix factorization.

### nsNMF Channel Weighting Algorithm

- **algorithmrank1.m**: Adjusting the rank yields different weights.
- **algorithm.m**: Mapping weights to channels for each sample.

## Installation

Ensure you have the following dependencies installed:

- Python 3.10
- Pytorch 2.0.0+cu118
- Numpy 1.23.5
- Scikit-learn 1.2.1
- Scipy 1.9.3
