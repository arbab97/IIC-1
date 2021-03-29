# IIC
 TensorFlow Implementation of https://arxiv.org/abs/1807.06653 for bats calls clustering


## Requirements

Tested on Python 3.6.8 and TensorFlow 1.14 with GPU acceleration.
I always recommend making a virtual environment.
To install required packages on a GPU system use:
```
pip install -r requirements.txt
```
For CPU systems replace `tensorflow-gpu==1.14.0` with `tensorflow==1.14.0` in `requirements.txt` before using pip.
Warning: I have not tried this.

## Repository Overview
* `data.py` contains code to construct a TensorFlow data pipeline where input perturbations are handled by the CPU.
* `graphs.py` contains code to construct various computational graphs, whose output connects to an IIC head.
* `models_iic.py` contains the `ClusterIIC` class, which implements unsupervised clustering.
* `utils.py` contains some utility functions.

## Running the Code
```
python models_iic.py
```
