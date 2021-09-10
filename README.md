# IIC
 TensorFlow Implementation of https://arxiv.org/abs/1807.06653 for bats calls clustering


## Repository Overview
* `data_batsnet.py` contains code to construct a TensorFlow data pipeline. Input directory for data is also defined in the start of this file.
* `models_iic_batsnet.py` contains the `ClusterIIC` class, which implements unsupervised clustering. Batch size, Learning rate, Number of epochs are defined at the end of this file

