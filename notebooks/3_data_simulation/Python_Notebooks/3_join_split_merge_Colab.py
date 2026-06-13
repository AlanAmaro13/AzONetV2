#!/usr/bin/env python
# coding: utf-8

# ## Merge and Split

# In the previous notebook we have simulated 400k samples for each bin, the purpose of this notebook is to merge all the samples, shuffle them and split them into Train, Test and Val datasets.

# ## Used functions

# In[1]:


import pandas as pd
import numpy as np
import h5py
import os


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# ## Load each dataset

# In[3]:


d1 = pd.read_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/bin_0.parquet')


# In[4]:


d2 = pd.read_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/bin_1.parquet')


# In[5]:


d3 = pd.read_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/bin_2.parquet')


# In[6]:


d4 = pd.read_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/bin_3.parquet')


# In[7]:


d5 = pd.read_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/bin_4.parquet')


# In[8]:


d5.head()


# In[9]:


df_final = pd.concat([d1, d2, d3, d4, d5], ignore_index=True)


# In[10]:


df_final.shape


# In[11]:


df_final = df_final.sample(frac=1).reset_index(drop=True)


# In[12]:


df_final.head(15)


# In[13]:


df_final.to_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/Final_Dataset.parquet')


# ## Convert the Final DF into Train Test Val in h5

# In[14]:


df_final = pd.read_parquet('/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/Final_Dataset.parquet')


# In[15]:


x = df_final['Espectro']
n, m = x.shape[0], len(x[0])
x = np.concatenate(x)
x = x.reshape(n, m, 1) # Organizamos la informacion de manera conveniente
x.shape


# In[16]:


y = np.array(df_final['Espesor'], dtype = np.float32)
y = y.reshape(-1, 1) # Los datos deben ser de la forma (n, 1)
y.shape


# In[17]:


def data_to_TrainTestVal(data:list, folder_path:str, p_train: float = 0.95, p_test: float = 0.04,  )-> None:
    '''
    Description:
        This functions converts a unique h5 file containing all the data into multiples set: train, test y val
        according to the proportions given in the args. It save all the files in a given folder.

    Args:
        file_path (str): Path to the h5 data
        folder_path (str): Folder path where the data will be saved
        p_train (float): Data porcentage to be used for train
        p_test (float): Data porcentage to be used for test

    Returns:
        None
    '''

    x_data = data[0]
    y_data = data[1]

    x_train, y_train = x_data[ : int( x_data.shape[0] * p_train) ], y_data[ : int( y_data.shape[0] * p_train) ]

    x_test, y_test = x_data[int( x_data.shape[0] * p_train) :int( x_data.shape[0] * (p_train + p_test) ) ], y_data[int( y_data.shape[0] * p_train) :int( y_data.shape[0] * (p_train + p_test) ) ]

    x_val, y_val =  x_data[int( x_data.shape[0] * (p_train + p_test) ) : ], y_data[int( y_data.shape[0] * (p_train + p_test) ) : ]

    to_h5(x = x_train, y = y_train, file_name = folder_path + '/train.h5')
    to_h5(x = x_test, y = y_test, file_name = folder_path + '/test.h5')
    to_h5(x = x_val, y = y_val, file_name = folder_path + '/val.h5')

    print('Done!')


# In[18]:


def to_h5(x, y, file_name):

    if not os.path.exists(file_name): # Verifica que la ruta no
        print(file_name)
        f = h5py.File(file_name, 'w') # Crea la ruta
        f.create_dataset('x_total', data = x, chunks=True, maxshape=(None, x.shape[1], x.shape[2]) )
        f.create_dataset('y_total', data = y, chunks=True, maxshape=(None, y.shape[1]) )
        print(f.keys())
        f.close()


# In[19]:


data_to_TrainTestVal([x, y],
                     folder_path = '/content/drive/MyDrive/AzONet_New_145/notebooks/3_data_simulation/NewDataAzONetV2/Sets', #'/content/drive/MyDrive/AzONet_New_161/notebooks/3_data_simulation/NewDataAzONetV2/Sets',
                     p_train = 0.80,
                     p_test = 0.1)

