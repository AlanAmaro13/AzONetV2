#!/usr/bin/env python
# coding: utf-8

# # Fitting Experimental Transmittance Spectrum using Direct with Bounds and Constraints. 

# In previous works, I've compared different optimization methods for the fitting of Experimental Transmittance Spectrum. The best optimizer's Differential Evolution (based on Genetic Algorithm). 

# ## About Direct:

# Introduced in 1993, the DIRECT global optimization algorithm provided a fresh approach to minimizing a black-box function subject to lower and upper bounds on the variables. In contrast to the plethora of nature-inspired heuristics, DIRECT was deterministic and had only one hyperparameter (the desired accuracy). Moreover, the algorithm was simple, easy to implement, and usually performed well on low-dimensional problems (up to six variables). Most importantly, DIRECT balanced local and global search (exploitation vs. exploration) in a unique way: in each iteration, several points were sampled, some for global and some for local search. This approach eliminated the need for “tuning parameters” that set the balance between local and global search. However, the very same features that made DIRECT simple and conceptually attractive also created weaknesses. For example, it was commonly observed that, while DIRECT is often fast to find the basin of the global optimum, it can be slow to fine-tune the solution to high accuracy. In this paper, we identify several such weaknesses and survey the work of various researchers to extend DIRECT so that it performs better. All of the extensions show substantial improvement over DIRECT on various test functions. An outstanding challenge is to improve performance robustly across problems of different degrees of difficulty, ranging from simple (unimodal, few variables) to very hard (multimodal, sharply peaked, many variables). Opportunities for further improvement may lie in combining the best features of the different extensions.

# ## Used libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import time 
import numpy as np
import random


# ## Transmittance Model

# This is the theoretical Transmittance model proposed by Alonso et al.

# In[2]:


# Archivo para obtener los valores de transmitancia del vidrio

#print(t_v[:,0][0])
def modelo_transmitancia(x, 
                         d, 
                         rugo_1, rugo_2, 
                         A, B, C, D, E, 
                         alpha, beta, gamma, 
                         ne):
    """
    x = longitud de onda
    d = espesor de la película
    t_vidrio = transmitancia del vidrio, valor experimental
    rugo_1 = rugosidad_1
    rugo_2 = rugosidad_2
    sellmeier = es el arreglo donde van a estar los coeficientes de la ecuacion de Sellmeier
    absorcion =  es el arreglo donde se van a guardar los coeficientes de la ecuacion de la absorcion
    ne = concentracion de electrones

    """
    # Constantes utilizadas

    c = 3e8 # Velocidad de la luz

    mu = 3.90e-4 # Movilidad

    df = pd.read_csv('../../experimental_samples/Background_data/TexpglassO.txt', sep = "\t", header = 0)
    t_v = df.values

    sellmeier = [A, B, C, D, E]

    absorcion = [alpha, beta, gamma]

    #ome_pla = np.sqrt(3182.61/0.417)*ne #Frecuencia de Plasma

    # Funciones

    def frecuencia(x):

        omega = 2 * np.pi * c * 1e9 / x

        return omega

    def gamma_f(x):

        gama = 2.8e11*x

        return gama

    def e1f_f(omega,gama,ne):

        return -(3182.61*ne)/(omega**2 + gama**2)

    def e2f_f(omega,gama,ne):

        return (3182.61 * ne * gama)/(omega * (omega**2 + gama**2))

    def e1b_f(x,sellmeier):
        # A = 2.0065
        # B = 1.574e6
        # C = 1e7
        # D = 1.5868
        # E = 260.63
        return sellmeier[0] + (sellmeier[1] * x**2)/(x**2 - sellmeier[2]**2 + 1e-6) + (sellmeier[3] * x**2)/(x**2 - sellmeier[4]**2 + 1e-6)

    def e2b_f(x):

        return 0.0

    def e1_f(e1f,e1b):

        return e1f + e1b

    def e2_f(e2f,e2b):

        return e2f + e2b

    def ng_f(t_v):
        """
        t_v transmitancia del vidrio
        """
        return (1/t_v) + np.sqrt(1/(t_v**2) - 1)

    def n_f(e1,e2):

        return (1/np.sqrt(2)) * np.sqrt(e1 + np.sqrt(e1**2 + e2**2)) + 1e-6

    def kapa_f(e1,e2):

        return (1/np.sqrt(2)) * np.sqrt(-e1 + np.sqrt(e1**2 + e2**2))

    def neff1_f(n):

        return np.sqrt((1/2) * (n**2 + 1))

    def T1_f(neff1,x,n,rugo_1):

        return np.exp(-0.5 * (2 * np.pi * rugo_1 * (neff1 - 1))**2 / x**2) * (n/1) * (4/(n + 1)**2)

    def T2_f(neff1,x,ng,n,rugo_2):

        return np.exp(-0.5 * (1/x**2) * (2 * np.pi * rugo_2 * (neff1 - 1))**2) * (ng/n) * 4 * (n**2) * (1/(n + ng)**2)

    def T3_f(ng):

        return (1/ng) * 4 * ng**2 * (1/(1 + ng)**2)

    def R1_f(n,x,rugo_1):

        return np.exp(-2 * (2 * np.pi * rugo_1 * n)**2 * (1/x)**2) * (n - 1)**2 * (1/(n + 1))**2

    def R2_f(n,x,ng,rugo_2):

        return np.exp(-2 * (2 * np.pi * rugo_2 * n)**2 * (1/x)**2) * (n - ng)**2 * (1/(n + ng))**2

    def R21_f(n,ng):

        return (n - ng)**2 * (1/(n + ng)**2)

    def R3_f(ng):

        return (ng - 1)**2 * (1/(ng + 1)**2)

    def phi_f(n,d,x):

        return 4 * np.pi * n * d * (1/x)

    def alfa_f(kapa,x,absorcion):
        #alpha_0 = 2.5e-3
        #beta = 9.8
        #lambdag = 363

        return kapa * 4 * np.pi * (1/x) + absorcion[0] * np.exp(1240 * absorcion[1] * ((1/x) - (1/absorcion[2])))

    def Tf_f(T1,T2,alfa,d,phi,R1,R2):

        exp1 = np.exp(np.clip(-alfa * d, -700, 700))
        exp2 = np.exp(np.clip(-2 * alfa * d, -700, 700))
        
        denominator = 1 - 2 * np.sqrt(R1 * R2) * np.cos(phi) * exp1 + R1 * R2 * exp2
        result = (T1 * T2) * exp1 / denominator

        return result
        #(T1 * T2) * np.exp(-alfa * d) / (1 - 2 * np.sqrt(R1 * R2) * np.cos(phi) * np.exp(-alfa * d) + R1 * R2 * np.exp(-2 * alfa * d))

    def T_f(T3,R21,R3,Tf):

        return 1.0 * ((T3 / (1 - R21 * R3)) * Tf * 100)

    def N_f(T1,R1):

        return T1 + k

    # Realizamos los calculos en cadena para obtener la transmitancia del modelo.

    omega = frecuencia(x) # Check 

    gama = gamma_f(x) # Check 

    e1f = e1f_f(omega,gama,ne)

    e2f = e2f_f(omega,gama,ne)

    e1b = e1b_f(x,sellmeier)

    e2b = e2b_f(x)

    e1 = e1_f(e1f,e1b)

    e2 = e2_f(e2f,e2b)

    ng = ng_f(t_v[:,1]) # Al ser un df, se transforma en un arreglo donde tomo solo la segunda columna

    #nswanep = nswanep(e1b[0])

    n = n_f(e1,e2)

    kapa = kapa_f(e1,e2)

    neff1 = neff1_f(n)

    T1 = T1_f(neff1,x,n,rugo_1)

    T2 = T2_f(neff1,x,ng,n,rugo_2)

    T3 = T3_f(ng)

    R1 = R1_f(n,x,rugo_1)

    R2 = R2_f(n,x,ng,rugo_2)

    R21 = R21_f(n,ng)

    R3 = R3_f(ng)

    phi = phi_f(n,d,x)

    alfa = alfa_f(kapa,x,absorcion)

    Tf = Tf_f(T1,T2,alfa,d,phi,R1,R2)

    T = T_f(T3,R21,R3,Tf)

    # Input   x, t_vidrio, d, rugo_1, rugo_2, sellmeier, absorcion, ne / x longitud de onda 
    # Output  T, d, rugo_1, rugo_2, sellmeier, absorcion, ne / T transmitancia 
    if np.isnan(T).any():
        print('NAN: T')
    
    return np.nan_to_num(T, nan=1e-6) #, d, rugo_1, rugo_2, sellmeier, absorcion, ne


# ## Experimental Samples

# In[4]:


data = pd.read_pickle('../../results/dataframe_spectrum_thickness_145_final.pkl')
data.head()


# In[5]:


data.shape


# ## Algorithm

# In[6]:


_name = 'direct_NonLinear'


# In[7]:


from scipy.optimize import direct


# The function (MSE) to minimize: 

# In[8]:


def error(params, x, y):
    y_pred = modelo_transmitancia(x, *params)
    _ = np.mean( (y_pred - y)**2 )
    if np.isnan(_).any(): print('NAN in ERROR')
    return np.nan_to_num(_, nan=1e-6)


# The constraint in the system corresponds to the reffractive index, this index must have a value in the range [1.8, 2.1]

# In[9]:


from scipy.optimize import NonlinearConstraint


# In[10]:


def refractive_index(x):
    A, B, C, D, E = x[3], x[4], x[5], x[6], x[7]
    _x = np.linspace(190e-9, 1101e-9, 911)
    n = np.sqrt( A + (B*_x**2)/(_x**2 - C**2 + 1e-6) + (D*_x**2)/(_x**2 - E**2 + 1e-6) )  # Get the n
    #print(n.shape)
    #print(np.mean(n))
    if np.isnan(n).any():
        print('NAN: refractive Index')
    
    return np.mean(np.nan_to_num(n, nan=1e-6))


# In[11]:


nlc = NonlinearConstraint(refractive_index, 1.8, 2.1)


# In[12]:


def new_error(params, x, y): 
    _error = error(params, x, y)
    if (refractive_index(params) > 2.1 ) or (refractive_index(params) < 1.8): 
        _error += _error + refractive_index(params)

    return _error


# ## Deactivate Warnings

# In[13]:


import warnings

warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")
warnings.filterwarnings("ignore", message=".*differential_evolution: the 'workers' keyword.*")
warnings.filterwarnings("ignore", message = 'invalid value encountered in multiply')
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ## Main

# In[14]:


maes = [] # Used to save the fitting parameters


# In[15]:


_tour = [i for i in range(data.shape[0])] # iteration list 
#_tour = [0]
len(_tour)


# In[16]:


data.head()


# In[17]:


pl = 0.5


# In[18]:


pli = (1 - pl)
pls = (1 + pl)


# In[19]:


t1 = time.time()


# In[20]:


print('Ejecutando Direct en {} muestras...\n'.format(len(_tour)))

while len(_tour) != 0:
    
    for i in _tour:  # For all the samples
        x = np.arange(190, 1101, 1) # Set the x array
        y_data = data['Espectro'][i][0].squeeze() # Get the spectrum
        d = data['Espesor'][i] # Get the thickness
        d_error = data['Error'][i]
    
        # Set an initial values, the mean get by EGP-GA.
        # Not used 
        initial_guess = [d, 
                     30.6, 26.5,
                     2.0065, 1.574e6, 1e7, 1.5868, 260.63, 
                     5.0e-3, 1.0e1, 3.6e2, 
                     2.1e26 ]
        
        # Set the conditions (given by IIM), we allow 20% error in the thickness measure
        bounds = [
            (d-d_error, d+d_error), 
            (pli*10, 150*pls), (pli*10, 150*pls),
            (pli*2, 4*pls), (pli*0.1e6, 1.6e6*pls), (pli*1e6, 10e6*pls), (pli*1, 3*pls), (pli*270, 280*pls), 
            (pli*0.5e-3, 10.0e-3*pls), (pli*5, 15*pls), (pli*350, 410*pls), (pli*1.0e26, 5.0e26*pls)
        ]

        minimizer_kwargs = {
            "method": "SLSQP",  
            "args": (x, y_data),
            "bounds": bounds,
            "constraints": nlc
        }
    
        try:
            print('Intentando ajustar muestra {}...'.format(i))
            _result_dv = direct(new_error, bounds, args=(x, y_data))
            
            maes.append( [int(i)] + _result_dv.x.tolist() + [float(_result_dv.fun)] ) # save the parameters
            _tour.remove(i)

            print('--> Muestra {} terminada!'.format(i)) 
            print('-> Restantes: {}'.format(len(_tour)))
        
        except Exception as e: 
            # Depending in the parameters, we may get Error values
            print('Posponiendo muestra: {}\n'.format(i))
            print(e)
    


# In[21]:


t2 = time.time()


# In[22]:


maes = np.array(maes) # Save as array 


# In[23]:


plt.figure(figsize = (5,5))
plt.plot(x, y_data)
plt.plot(x, modelo_transmitancia(x, *maes[-1][1:-1]))
plt.grid()
plt.xlabel(r'Wavelenght $\lambda$ [nm]')
plt.ylabel('% Transmittace')
plt.title('MSE: {:.2f}'.format(maes[-1][-1]))
plt.show()


# The array contains: Index, Thickness, R1, R2, Sellmeier Coefficients (5), Absorption (3), ne and MSE value. 

# In[24]:


maes[0]


# In[25]:


refractive_index(maes[0][1:])


# In[26]:


maes.shape


# In[27]:


np.save('../../results/SciPy_IIM/{}_145F'.format(_name), maes)


# In[28]:


print('Tiempo: ', t2-t1)

