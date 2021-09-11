# IIC
 TensorFlow Implementation of https://arxiv.org/abs/1807.06653 for bats calls clustering


### Setup and Model Train
* `data_batsnet.py` contains code to construct a TensorFlow data pipeline. Input directory for data is also defined in the start of this file.
* `models_iic_batsnet.py` contains the `ClusterIIC` class, which implements unsupervised clustering. Batch size, Learning rate, Number of epochs are defined at the end of this file
* After the setup, run model using the command: `source activate [environment_name] && python models_iic_batsnet.py`

### Running via Notebook on Google Colab (`IIC_tensorflow_astirn.ipynb`)
A notebook has already been setup which will fetch the required data, install all the required packages, fetch the code, run the algorithm and output results in a `csv` format. The details are given below:
* Install miniconda to setup the environmen later
* Clone the code from github repository. Or the code folder (provided) could also be uploaded manually in the notebook directory. 
* Connect to Google Drive and copy all the required data which needs to be clustered. You can also upload the data manually. Another way include using `gdown` with shared link of data. 
* Setup environment and install packages using `pip install -r 'requirements.txt'`. Then install additional packages `ipykernel` and `pandas`. Alternatively, you can also use the provided `requirements_iic.yml` file to setup the new environment using : `conda env create -f 'requirements_[algo].yml'`
* Finally, run the model using the command given above. In notebook it is:  `source activate iic_astirn_env && python models_iic_batsnet.py`
* The output is provided in `csv` format in the file: `results_iic.csv`

### Citation
This Algorithms is taken from:
```bibtex
@inproceedings{ji2019invariant,
  title={Invariant information clustering for unsupervised image classification and segmentation},
  author={Ji, Xu and Henriques, Joao F and Vedaldi, Andrea},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9865--9874},
  year={2019}
}
```
