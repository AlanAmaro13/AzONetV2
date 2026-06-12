#!/usr/bin/env python
# coding: utf-8

# ## Introducing U-NET

# In this section, we perform a hyper-parameters search over the F1-UNET Model. This is a UNET model with the following features: 
# 
# 
# *U-NET*:
# * Base Filters: 21
# * Kernel Size: [150, 50, 5]
#   
# *DNN*:
# * Nodes: [470, 340, 330]
# 
# The purpose of this notebook is to implement L1L2 in the convolutional section with L1L2 and Dropout over the DNN section. 
# 

# ## Used libraries

# In[1]:


from AmaroXI.AmaroX.ai_functions import * 
from AmaroXI.AmaroX.UNET import *
from AmaroXI.AmaroX.DNN import *
from AmaroXI.AmaroX.data_manipulation import *
from AmaroXI.AmaroX.utilities import *

import keras_tuner
import pandas as pd


# ## 1/10 Samples

# In[2]:


n = 10
size = [
    1080000//n,
    108000//n, 
    12000//n
]


# In[3]:


data = load_data_normalization_sample_General(
    '/home/bokhimi/Transmittance_Thesis/data_simulation/F1', 
    size = size
)


# In[4]:


show_dimensions(data)


# In[5]:


plot_xy(data)


# In[6]:


x_train, y_train, x_test, y_test, x_val, y_val = data
del data 


# ## GPU Allocation

# In[7]:


#get_ipython().system(' nvidia-smi')


# In[8]:


get_gpu(2)


# ## Paths

# In[9]:


name = 'DNN_UNET_F1_R'
folder_path = './models'
final_path = os.path.join(folder_path, name)


# ## Callbacks

# In[10]:


callbacks = standard_callbacks(folder_name= name, 
                               folder_path= folder_path, 
                               patiences= [50, 1000])


# ## UNET Model

# In[11]:


G_UNET


# In[12]:


G_Dense


# In[13]:


def _UNET(l1c:float, l2c:float, DP:int, L1:float, L2:float):
    
    inputs = keras.layers.Input((911,1))

    __UNET = G_UNET(
        inputs = inputs, 
        layers = 21,
        unet_kernel= [150, 50, 5], 
        WIC = 'he_normal', 
        WRC= keras.regularizers.L1L2(l1 = l1c, l2=l2c),
        pad_type='valid', 
        unet_act_func='leaky_relu', 
        pool = 4, 
        stride = 4, 
        final_func_act= 'relu', 
        stride_conv=1,
        pool_op= 'AP', 
    )

    _flatten = keras.layers.Flatten()(__UNET)

    _DNN = G_Dense(
        inputs = _flatten, 
        nodes = [470, 340, 330],
        DP = DP, 
        n_final = 1, 
        act_func = 'leaky_relu', 
        final_act_func= 'relu',
        WI = 'he_normal', 
        L1 = L1, 
        L2 = L2, 
        use_bias= True
    )
    

    return keras.models.Model(inputs = inputs, outputs = _DNN)


# In[14]:


def compile_model(l1c:float, l2c:float, DP:int, L1:float, L2:float, optimizer, modelo):

  model = modelo(l1c = l1c, l2c=l2c, DP=DP, L1=L1, L2=L2)

  model.compile(optimizer = optimizer,
                loss = 'mae',
                metrics = ['mape'])

  return model


# In[15]:


def build_model(hp):


  DP = hp.Int('Dropout', min_value = 0, max_value = 50, step = 5)

  L1 = hp.Choice('L1', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  L2 = hp.Choice('L2', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  l1c = hp.Choice('L1C', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  l2c = hp.Choice('L2C', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  optimizer = hp.Choice('optimizer', ['adam'])

  if optimizer == 'adam': opt = keras.optimizers.Adam(
        learning_rate = 0.001
    )

  elif optimizer == 'sgd': opt = keras.optimizers.SGD(
        learning_rate = 0.001
    )

  elif optimizer == 'adagrad': opt = keras.optimizers.Adagrad(
        learning_rate = 0.001
    )


  model_f = compile_model(
      L1 = L1, 
      L2 = L2, 
      l1c = l1c, 
      l2c = l2c,
      DP = DP,
      optimizer = optimizer, 
      modelo = _UNET)

  return model_f


# In[16]:


build_model(keras_tuner.HyperParameters())


# In[17]:


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective= keras_tuner.Objective('val_mape', 'min') ,
    max_trials= 50, # Set to 3
    executions_per_trial = 2,
    overwrite=True,
    directory= final_path,
    project_name="UNET-F1-R-KT",
)


# In[18]:


tuner.search_space_summary()


# In[19]:


tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=512)


# In[ ]:


file_path = os.path.join(final_path, 'best_models.txt')

with open(file_path, "w") as file:
    # Save the original stdout
    original_stdout = sys.stdout
    try:
        sys.stdout = file  # Redirect stdout to the file
        tuner.results_summary()  # Call your function
    finally:
        sys.stdout = original_stdout


