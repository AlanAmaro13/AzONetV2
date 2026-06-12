#!/usr/bin/env python
# coding: utf-8

# ## Introducing Deep Neural Networks 

# Just 5 hidden layer. 

# ## Used libraries

# In[1]:


from AmaroX.AmaroX.ai_functions import * 
from AmaroX.AmaroX.ai_models import * 
from AmaroX.AmaroX.data_manipulation import *
from AmaroX.AmaroX.utilities import *

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
    '../../../../data_simulation/F2', 
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


#! nvidia-smi


# In[8]:


get_gpu(0)


# ## Paths

# In[9]:


name = 'DNN_HP_5_Reg'
folder_path = './models'
final_path = os.path.join(folder_path, name)


# ## Callbacks

# In[10]:


callbacks = standard_callbacks(folder_name= name, 
                               folder_path= folder_path, 
                               patiences= [50, 1000])


# ## DNN Model

# In[11]:


def _DNN(nodes:list, DP:int, L1: float, L2: float):

    inputs = keras.layers.Input((911,))

    __DNN = G_Dense(
        inputs = inputs, 
        nodes = nodes,
        DP = DP,
        n_final = 1,
        act_func = 'leaky_relu', 
        final_act_func= 'relu', 
        WI= 'he_normal', 
        L1 = L1, 
        L2 = L2, 
        use_bias= True
    )

    return keras.models.Model(inputs = inputs, outputs = __DNN)


# In[12]:


def compile_model(nodes: list, DP:int, L1:float, L2:float, optimizer, modelo):

  model = modelo(nodes = nodes, DP=DP, L1=L1, L2=L2)

  model.compile(optimizer = optimizer,
                loss = 'mae',
                metrics = ['mape'])

  return model


# In[13]:


def build_model(hp):

  nodes = [
      400, 
      50, 
      300, 
      100, 
      250 
  ]

  DP = hp.Int('Dropout', min_value = 0, max_value = 50, step = 5)

  L1 = hp.Choice('L1', [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

  L2 = hp.Choice('L2', [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

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


  model_f = compile_model(nodes = nodes, DP = DP, L1=L1, L2=L2, optimizer = optimizer, modelo = _DNN)

  return model_f


# In[14]:


build_model(keras_tuner.HyperParameters())


# In[15]:


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective= keras_tuner.Objective('val_mape', 'min') ,
    max_trials= 50, # Set to 3
    executions_per_trial = 2,
    overwrite=True,
    directory= final_path,
    project_name="DNN-MI-KT",
)


# In[16]:


tuner.search_space_summary()


# In[17]:


tuner.search(x_train[:,:,0], y_train, epochs=25, validation_data=(x_test[:,:,0], y_test), batch_size=512)


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


