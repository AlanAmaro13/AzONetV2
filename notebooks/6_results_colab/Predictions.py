#!/usr/bin/env python
# coding: utf-8

# ## Used libraries

# In[1]:


_name = 'MSE'


# In[2]:


import h5py
import numpy as np
from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import glob
import keras
import time
import pandas as pd


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# ## Required Functions

# In[4]:


def normalization_by_sample(x, p = False) -> np.ndarray:
    '''Normalize the array by taking a max and min value over a sample.

    Args:
        x: A NumPy array of shape (n, m), where n is the number of samples and m is the number of features.

    Returns:
        A NumPy array of the same shape (n, m), normalized by sample.
    '''

    mi = np.min(x, axis = 1, keepdims=True)
    ma = np.max(x, axis = 1, keepdims=True)
    if p:
        print(ma)
        print(mi)

    return (x - mi) / (ma - mi + 1e-6)


# In[5]:


def load_data_normalization_sample_General(folder: str,
                                  size: list = None,
                                  names: list = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Description:
        Loads train, test, and validation datasets from .h5 files in the given folder.
        Applies per-sample normalization to x datasets.

    Args:
        folder (str): Path to the folder containing the datasets in .h5 format.
        size (list): Number of samples to take from each dataset: [train, test, val].
        names (list): Dataset key names for x and y inside the .h5 files.

    Returns:
        x_train, y_train, x_test, y_test, x_val, y_val (np.ndarray)
    '''

    if size is None:
        size = [0, 0, 0]
    if names is None:
        names = ['x_total', 'y_total']

    _datapath_list = glob.glob(folder + '/*.h5')

    f1 = f2 = f3 = None

    for path in _datapath_list:
        if 'train' in path:
            f1 = h5py.File(path, 'r')
        elif 'test' in path:
            f2 = h5py.File(path, 'r')
        elif 'val' in path:
            f3 = h5py.File(path, 'r')

    if f1 is None:
        raise FileNotFoundError(f"No train .h5 file found in {folder}")
    if f2 is None:
        raise FileNotFoundError(f"No test .h5 file found in {folder}")
    if f3 is None:
        raise FileNotFoundError(f"No val .h5 file found in {folder}")

    if size[0] > 0:
        _xtrain = normalization_by_sample(f1[names[0]][:size[0]])
        _ytrain = f1[names[1]][:size[0]]
        _xtest, _ytest = normalization_by_sample(f2[names[0]][:size[1]]), f2[names[1]][:size[1]]
        _xval, _yval = normalization_by_sample(f3[names[0]][:size[2]]), f3[names[1]][:size[2]]
        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval)
    else:
        _xtrain = normalization_by_sample(f1[names[0]][:])
        _ytrain = f1[names[1]][:]
        _xtest, _ytest = normalization_by_sample(f2[names[0]][:]), f2[names[1]][:]
        _xval, _yval = normalization_by_sample(f3[names[0]][:]), f3[names[1]][:]
        return (_xtrain, _ytrain, _xtest, _ytest, _xval, _yval)


# ## Main Data - Train/Test/Val

# In[6]:


data = load_data_normalization_sample_General(
    folder = '/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/Sets',
)


# In[7]:


x_train, y_train, x_test, y_test, x_val, y_val = data
del data


# ## GPU Allocation

# In[8]:


get_ipython().system(' nvidia-smi')


# In[9]:


def get_gpu(gpu_number: int = 0, p: bool = False) -> None:
    '''
    Description:
        This function gets, if available, a GPU.

    Args:
        gpu_number (int): Referring to the GPU to take.
        p (bool): Whether to print the total number of GPUs.
    '''
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        if p:
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")


# In[10]:


get_gpu(0)


# ## Used model

# In[11]:


modelo_UNET = keras.models.load_model('/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/models/Copia de model.keras')


# In[12]:


# Integrity check
score = modelo_UNET.evaluate(x = x_test, y = y_test)
score


# In[13]:


y_train.shape, y_val.shape, y_test.shape


# ## Predictions

# In[14]:


preds_train = modelo_UNET.predict(x_train)


# In[15]:


preds_test = modelo_UNET.predict(x_test)


# In[16]:


preds_val = modelo_UNET.predict(x_val)


# In[17]:


preds_train.shape, preds_test.shape, preds_val.shape


# ## Save the data

# In[18]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/preds_train.npy'.format(_name),
    preds_train
    )


# In[19]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/preds_test.npy'.format(_name),
    preds_test
    )


# In[20]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/preds_val.npy'.format(_name),
    preds_val
    )


# In[21]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/y_train.npy'.format(_name),
    y_train
    )


# In[22]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/y_test.npy'.format(_name),
    y_test
    )


# In[23]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/y_val.npy'.format(_name),
    y_val
    )


# ## Experimental Data

# In[24]:


data = pd.read_pickle('/content/drive/MyDrive/AzONet_New_145/results/dataframe_spectrum_thickness_145_final.pkl')
data.head()


# In[25]:


x_145 = np.array([ x[0] for x in data['Espectro'] ])
x_145.shape


# In[26]:


x_145_norm = normalization_by_sample(x_145)
x_145_norm.shape


# In[27]:


preds_145 = modelo_UNET.predict(x_145_norm)
preds_145.shape


# In[28]:


param = np.load('/content/drive/MyDrive/AzONet_New_145/results/SciPy_IIM/differential_evolution_NonLinear_145F.npy')
param = param[param[:, 0].argsort()] # organize by element


# In[29]:


y_145 = np.expand_dims(param[:, 1], axis = -1 )
y_145.shape


# In[30]:


y_145[0]


# In[31]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/y_145.npy'.format(_name),
    y_145
    )


# In[32]:


np.save(
    '/content/drive/MyDrive/AzONet_New_145/notebooks/6_results_colab/{}_data/preds_145.npy'.format(_name),
    preds_145
    )

