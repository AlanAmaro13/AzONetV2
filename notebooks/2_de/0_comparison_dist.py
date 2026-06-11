#!/usr/bin/env python
# coding: utf-8

# # How good is our fitting using Differential Evolution?

# In this notebook we analyze the fitting results provided by Differential Evolution and we analyze the distribution of each parameter $\Theta.$

# ## Used libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import time 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt


# ## Data 

# In[2]:


param = np.load('../../results/SciPy_IIM/differential_evolution_NonLinear_145F.npy')


# In[3]:


param = param[param[:, 0].argsort()] # organize by element 


# In[4]:


param[:3]


# In[5]:


param.shape # Index, Thickness, R1, R2, Sellemier Coefficients (5), absorption coefficients (3), ne and MSE.


# In[6]:


param = param[:, 1:]


# In[7]:


data_exp = pd.read_pickle('../../results/dataframe_spectrum_thickness_145_final.pkl')
data_exp = data_exp.reset_index(drop = True)
data_exp.head()


# In[8]:


data_exp['Espectro'][5][0].shape


# ## Transmittance Model

# In[9]:


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


# ## Seaborn 

# In[10]:


# Set style first
sns.set_style("whitegrid")

# Then customize individual elements through rcParams
plt.rcParams.update({
    "figure.dpi": 300,           # Default DPI for new figures
    "savefig.dpi": 300,          # DPI when saving figures

    # Font settings
    "font.family": "sans-serif",     # Universal font family
    "font.size": 11,                   # Base font size

    # Title and label sizes
    "axes.titlesize": 16,             # Axis title size
    "axes.titleweight": "bold",       # Axis title weight
    "axes.labelsize": 14,             # Axis label size
    "axes.labelweight": "semibold",   # Axis label weight

    # -------------------------------------------------

    # Tick LABEL sizes (text next to ticks)
    "xtick.labelsize": 11,       # Size of x-axis tick labels (e.g., "0", "1", "2")
    "ytick.labelsize": 11,       # Size of y-axis tick labels

    # Tick MARK sizes (physical marks on axes)
    "xtick.major.size": 6,       # Length of MAJOR tick marks on x-axis
    "ytick.major.size": 6,       # Length of MAJOR tick marks on y-axis

    # Additional tick parameters you might want to use:
    "xtick.minor.size": 3,       # Length of MINOR tick marks on x-axis
    "ytick.minor.size": 3,       # Length of MINOR tick marks on y-axis

    # Tick WIDTH (thickness)
    "xtick.major.width": 1,      # Width/Thickness of major ticks
    "ytick.major.width": 1,      # Width/Thickness of major ticks

    # Tick PADDING (distance from label to tick)
    "xtick.major.pad": 3.5,      # Padding between x-tick and label
    "ytick.major.pad": 3.5,      # Padding between y-tick and label

    # Tick DIRECTIONS
    "xtick.direction": "out",    # "in", "out", or "inout"
    "ytick.direction": "out",    # Points outward from axis

    # Tick COLORS
    "xtick.color": "black",      # Color of x-axis ticks and labels
    "ytick.color": "black",      # Color of y-axis ticks and labels

    # MINOR ticks (for more granular scales)
    "xtick.minor.visible": False,  # Show minor x-ticks
    "ytick.minor.visible": False,  # Show minor y-ticks

    # BOTTOM/TOP/LEFT/RIGHT ticks (which sides get ticks)
    "xtick.top": False,          # Show ticks on top of plot
    "xtick.bottom": True,        # Show ticks on bottom (default: True)
    "ytick.left": True,          # Show ticks on left (default: True)
    "ytick.right": False,        # Show ticks on right

    # ---------------------------------

    # Legend
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
    "legend.framealpha":0.9,

    # Figure title (for suptitle)
    "figure.titlesize": 18,
    "figure.titleweight": "bold",

    # ----------------------------------

    # Figure border
    #"figure.edgecolor": "black",
    #"figure.frameon": True,
    #"figure.linewidth": 2.0,

    # Axes borders (spines)
    "axes.linewidth": 1.0,           # Width of axis lines
    "axes.edgecolor": "black",       # Color of axis lines

    # Individual spine control via rcParams
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": True,

    # ----------------------------
    # Patch properties (for histograms, bars, kde fills, etc.)
    "patch.linewidth": 1.5,           # Default linewidth for patches
    "patch.edgecolor": "black",       # Default edge color
    "patch.facecolor": "blue",        # Default fill color (careful with this!)
    "patch.force_edgecolor": True,    # Always show edges
    "axes.grid": False,           # Turn off grid completely

})


# ## Comparison

# In[11]:


x = np.arange(190, 1101, 1) # Wavelength used to model the spectrum


# In[12]:


n = 0
for sample in param:
    spectrum = modelo_transmitancia( 
        x, *sample[:-1]
    )

    plt.figure(figsize = (5,5))
    
    plt.plot(np.arange(190, 1101, 1),
            spectrum, label = 'Ajuste')
    plt.plot(np.arange(190, 1101, 1), 
            data_exp['Espectro'][n][0], label = 'Experimental')
    plt.xlabel(r'Longitud de onda $\lambda$ [nm]')
    plt.ylabel('% Transmitancia')
    plt.title('Espectro #{:} | ECM: {:.2f}'.format(n, sample[-1]))
    plt.legend(loc = 'lower right')
    plt.savefig('./data_results/comparison/{:}.png'.format(n))
    plt.show()
    #break
    n +=1


# In[13]:


print(np.max(param[:, 12])) # Max MSE


# In[14]:


print(np.mean(param[:, 12])) # Mean MSE


# In[15]:


print(np.min(param[:, 12])) # Min MSE


# In[16]:


import os
from PIL import Image

# Parameters
input_folder = "./data_results/comparison"  # Replace with your folder
output_folder = "./data_results/comparison_grids"
grid_size = (3, 3)  # 3x3 = 9 images per output
image_size = (500, 500)  # Resize all images to this (optional)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Collect and sort image filenames
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

# Blank (white) filler image for padding incomplete final grid
blank = Image.new('RGB', image_size, color='white')

# Group images into batches of 9
total_images = len(image_files)
print(f"Total images found: {total_images}")

for i in range(0, total_images, 9):
    batch = image_files[i:i + 9]
    
    # Create blank canvas
    canvas = Image.new('RGB', (grid_size[1]*image_size[0], grid_size[0]*image_size[1]))

    for idx, file in enumerate(batch):
        img = Image.open(os.path.join(input_folder, file)).resize(image_size)
        row, col = divmod(idx, grid_size[1])
        canvas.paste(img, (col*image_size[0], row*image_size[1]))

    # If last batch is incomplete, pad with blank images
    if len(batch) < 9:
        for pad_idx in range(len(batch), 9):
            row, col = divmod(pad_idx, grid_size[1])
            canvas.paste(blank, (col*image_size[0], row*image_size[1]))
        print(f"  Grid {i//9}: padded {9 - len(batch)} blank slot(s)")

    # Save final image
    out_path = os.path.join(output_folder, f"grid_{i//9}.png")
    canvas.save(out_path)

print("All image grids generated.")
print(f"Saved {output_folder}/grid_0.png through grid_{(total_images - 1)//9}.png")


# ## Distribuciones

# This is the MSE Histogram over the DE estimation.

# In[17]:


plt.figure(figsize = (5.5,5))
bins = np.arange(0, int(np.max(param[:, -1])), 1)
plt.hist(param[:, -1], bins = bins,  edgecolor = 'black')
plt.xlabel('Error Cuadrado Medio')
plt.ylabel('# Muestras')
plt.savefig('./data_results/dist/ECM_DE.png')
plt.show()


# In[18]:


plt.figure(figsize = (5,5))
sns.kdeplot(param[:, -1], fill = True, color= 'blue', edgecolor = 'black', alpha = 0.6, clip = [0, np.inf])
plt.xlabel('Error Cuadrado Medio')
plt.ylabel('Densidad')
text = r'$\mu$: {:.1f} | $\delta$: {:.1f}'.format(
    np.mean(param[:, -1]), 
    np.std(param[:, -1])
)

# Add text box in top-right corner
plt.text(0.95, 0.95, text, transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))


#plt.xlim(0)
plt.savefig('./data_results/dist/error_kde.png')
plt.show()


# Here, we select the approximations with a MSE lower than 5%

# ## Thickness

# In[19]:


np.min(param[:, 0]), np.max(param[:, 0])


# In[20]:


_ = [x for x in range(param[:,0].shape[0])]


# In[21]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500
bins = np.arange(min_thickness, max_thickness + 100, 450)

plt.figure(figsize=(5, 5))

# Create histogram
ax = sns.histplot(param[:, 0], bins=bins)

# Labels and title
plt.xlabel('Espesor [nm]')
plt.ylabel('Frecuencia')

# Add statistical information as text box
stats_text = f'Media: {np.mean(param[:, 0]):.1f} nm\nSTD: {np.std(param[:, 0]):.1f} nm\nN: {len(param[:, 0])}'
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig('./data_results/dist/thickness_450.png')
plt.show()


# In[22]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500
bins = np.arange(min_thickness, max_thickness + 100, 200)

plt.figure(figsize=(5, 5))

# Create histogram
ax = sns.histplot(param[:, 0], bins=bins)

# Labels and title
plt.xlabel('Espesor [nm]')
plt.ylabel('Frecuencia')

# Add statistical information as text box
stats_text = f'Media: {np.mean(param[:, 0]):.1f} nm\nSTD: {np.std(param[:, 0]):.1f} nm\nN: {len(param[:, 0])}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig('./data_results/dist/thickness_200.png')
plt.show()


# In[23]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500
bins = np.arange(min_thickness, max_thickness + 100, 100)

plt.figure(figsize=(5, 5))

# Create histogram
ax = sns.histplot(param[:, 0], bins=bins)

# Labels and title
plt.xlabel('Espesor [nm]')
plt.ylabel('Frecuencia')

# Add statistical information as text box
stats_text = f'Media: {np.mean(param[:, 0]):.1f} nm\nSTD: {np.std(param[:, 0]):.1f} nm\nN: {len(param[:, 0])}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig('./data_results/dist/thickness_100.png')
plt.show()


# In[24]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500
bins = np.arange(min_thickness, max_thickness + 100, 50)

plt.figure(figsize=(5, 5))

# Create histogram
ax = sns.histplot(param[:, 0], bins=bins)

# Labels and title
plt.xlabel('Espesor [nm]')
plt.ylabel('Frecuencia')

# Add statistical information as text box
stats_text = f'Media: {np.mean(param[:, 0]):.1f} nm\nDesv. std: {np.std(param[:, 0]):.1f} nm\nN: {len(param[:, 0])}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig('./data_results/dist/thickness_50.png')
plt.show()


# In[25]:


plt.figure(figsize=(5, 5))

# Create KDE plot with enhanced styling
ax = sns.kdeplot(param[:, 0], fill = True, color= 'blue', edgecolor = 'black', alpha = 0.2, 
                clip = [100, 1500])

# Labels and title
plt.xlabel('Espesor [nm]')
plt.ylabel('Densidad')

# Add mean and std lines
mean_val = np.mean(param[:, 0])
std_val = np.std(param[:, 0])
plt.axvline(mean_val, color='red', linestyle='--', linewidth=1, 
            alpha=0.8, label=f'Media: {mean_val:.1e} nm')
plt.axvline(mean_val + std_val, color='purple', linestyle='-', linewidth=1, 
            alpha=0.8, label=f'+1σ: {mean_val + std_val:.1e} nm')
plt.axvline(mean_val - std_val, color='purple', linestyle='-', linewidth=1, 
            alpha=0.8, label=f'-1σ: {mean_val - std_val:.1e} nm')

# Add legend
plt.legend()

# Clean spines
sns.despine()

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax = plt.gca()

plt.tight_layout()
plt.savefig('./data_results/dist/thickness_kde.png')
plt.show()


# In[26]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500
bins = np.arange(min_thickness, max_thickness + 100, 50)

plt.figure(figsize=(5, 5))

# Create histogram with seaborn
ax = sns.histplot(param[:, 0], bins=bins)


# Add vertical lines with labels
vline_positions = [100, 250, 600, 950, 1100, 1500]
vline_labels = ['100 nm', '250 nm', '600 nm', '950 nm', '1100 nm', '1500 nm']

for i, label in zip(vline_positions, vline_labels):
    plt.axvline(i, color='crimson', linestyle='--', linewidth=1.2, 
                alpha=0.8)
    # Add text labels rotated vertically
    plt.text(i, plt.ylim()[1] * 0.98, label, rotation=90, 
             verticalalignment='top', horizontalalignment='right',
             fontsize=9, color='crimson', alpha=0.9)

# Labels
plt.xlabel('Espesor [nm]')
plt.ylabel('Frecuencia')

# Clean spines
sns.despine()

plt.tight_layout()
plt.savefig('./data_results/dist/thickness_50_sep.png')
plt.show()


# In[27]:


thickness = param[:, 0]
thickness.shape


# In[28]:


intervals = [100, 250, 600, 950, 1100, 1500]


# In[29]:


region_labels = [
    'R1: [100, 250] nm',
    'R2: [250, 600] nm',
    'R3: [600, 950] nm',
    'R4: [950, 1100] nm',
    'R5: [1100, 1500] nm'
]

os.makedirs('./data_results', exist_ok=True)

with open('./data_results/samples_per_region.txt', 'w') as f:
    f.write("Sample Count per Thickness Region\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total samples: {len(thickness)}\n\n")
    total_counted = 0
    for i in range(len(intervals) - 1):
        count = np.where((thickness >= intervals[i]) & (thickness <= intervals[i+1]))[0].shape[0]
        f.write(f"{region_labels[i]}: {count} samples\n")
        total_counted += count
    f.write(f"\n{'─' * 50}\n")
    f.write(f"Sum: {total_counted}\n")

print("Sample count per region saved to ./data_results/samples_per_region.txt")


# ## Usando Gaussian Mixture Model para modelar la distribucion

# In[30]:


#! pip install scikit-learn


# In[31]:


from sklearn.mixture import GaussianMixture


# In[32]:


thickness = param[:, 0].reshape(-1, 1)
thickness.shape


# In[33]:


bic_values = []
n_components_range = range(1, 13) # Hasta 12 Gaussianas se consideran


# In[34]:


for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=1360)
    gmm.fit(thickness)
    bic_values.append(gmm.bic(thickness))


# In[35]:


plt.plot(n_components_range, bic_values, label='BIC', marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC')
plt.title('BIC for different numbers of GMM components')
plt.grid(True)
plt.show()


# In[36]:


# Get the optimal number of components
optimal_n_components = n_components_range[np.argmin(bic_values)]
print(f"Optimal number of components: {optimal_n_components}")


# In[37]:


# Fit the GMM with the optimal number of components
gmm_optimal = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=0)
gmm_optimal.fit(thickness)


# In[38]:


# Generate range for plotting the density
_x = np.linspace(thickness.min() -1, thickness.max() + 1, 1000).reshape(-1, 1)
logprob = gmm_optimal.score_samples(_x)
pdf = np.exp(logprob)


# In[39]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500

# Create an array of bin edges with a step of 100 nm
bins = np.arange(min_thickness, max_thickness + 100, 100)

plt.hist(thickness, bins=bins, density=True, label='Data histogram')
plt.plot(_x, pdf, label=f'GMM fit (k={optimal_n_components})', linewidth=2)
plt.title(f'Gaussian Mixture Model with {optimal_n_components} components')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.show()


# In[40]:


# Access the model parameters
means = gmm_optimal.means_
covariances = gmm_optimal.covariances_
weights = gmm_optimal.weights_

# Print the parameters
print("Means (centroids):")
print(means)
print("\nCovariances (diagonal of covariance matrix for each component):")
print(covariances)
print("\nWeights (mixing coefficients):")
print(weights)

# If you want to see if the model converged
print("\nConverged status:", gmm_optimal.converged_)


# In[41]:


std_devs = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
std_devs


# In[42]:


# Determine the range of the data
min_thickness = 100
max_thickness = 1500

# Create an array of bin edges with a step of 100 nm
bins = np.arange(min_thickness, max_thickness + 100, 100)

plt.vlines( 650, 0, 0.0020 )
plt.vlines( 990, 0, 0.0020 )

plt.hist(thickness, bins=bins, density=True, label='Data histogram')
plt.plot(_x, pdf, label=f'GMM fit (k={optimal_n_components})', linewidth=2)
plt.title(f'Gaussian Mixture Model with {optimal_n_components} components')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.show()


# ## Statistics Function

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis, norm, shapiro, kstest, anderson, jarque_bera
import pandas as pd
import seaborn as sns
import os

def analyze_distribution(_hist, name="distribution", save_dir="./data_results/distribution_analysis"):
    """
    Analyze a distribution represented by a histogram array.
    
    Parameters:
    -----------
    _hist : array-like
        Array representing the distribution data (not histogram counts, but actual data values)
    name : str
        Name of the parameter (used for filenames and plot titles)
    save_dir : str
        Directory to save the report and plots. If None, no files are saved.
    """
    
    # Convert to numpy array for consistency
    data = np.asarray(_hist)
    
    # Create output directory
    param_dir = os.path.join(save_dir, name)
    os.makedirs(param_dir, exist_ok=True)
    
    # Open the report file
    report_path = os.path.join(param_dir, f"analysis_{name}.txt")
    f = open(report_path, 'w')
    
    def _print(*args, **kwargs):
        """Helper to print to both stdout and file."""
        print(*args, **kwargs)
        print(*args, file=f)
    
    _print("=" * 60)
    _print("DISTRIBUTION ANALYSIS REPORT")
    _print(f"Parameter: {name}")
    _print("=" * 60)
    
    # Basic statistics
    _print("\n1. BASIC STATISTICS")
    _print("-" * 40)
    _print(f"Sample size: {len(data):,}")
    _print(f"Mean: {np.mean(data):.4f}")
    _print(f"Median: {np.median(data):.4f}")
    _print(f"Std deviation: {np.std(data, ddof=1):.4f}")
    _print(f"Variance: {np.var(data, ddof=1):.4f}")
    _print(f"Min: {np.min(data):.4f}")
    _print(f"Max: {np.max(data):.4f}")
    _print(f"Range: {np.ptp(data):.4f}")
    _print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.4f}")
    
    # Shape statistics
    _print("\n2. DISTRIBUTION SHAPE")
    _print("-" * 40)
    
    # Skewness (Fisher-Pearson coefficient)
    skewness = skew(data, bias=False)
    _print(f"Skewness: {skewness:.4f}")
    
    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_interpret = "Approximately symmetric"
    elif abs(skewness) < 1:
        skew_interpret = "Moderately skewed"
    else:
        skew_interpret = "Highly skewed"
    
    if skewness > 0:
        skew_direction = "right (positive)"
    elif skewness < 0:
        skew_direction = "left (negative)"
    else:
        skew_direction = "no"
    
    _print(f"  → {skew_interpret}, {skew_direction} skew")
    
    # Kurtosis (Fisher's definition, excess kurtosis)
    kurt = kurtosis(data, bias=False, fisher=True)
    _print(f"Kurtosis (excess): {kurt:.4f}")
    
    # Interpret kurtosis
    if abs(kurt) < 0.5:
        kurt_interpret = "Mesokurtic (similar to normal)"
    elif kurt > 0.5:
        kurt_interpret = "Leptokurtic (heavy-tailed)"
    else:
        kurt_interpret = "Platykurtic (light-tailed)"
    _print(f"  → {kurt_interpret}")
    
    # Compare with normal distribution
    _print("\n3. COMPARISON WITH NORMAL DISTRIBUTION")
    _print("-" * 40)
    
    # Generate matching normal distribution
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    normal_dist = norm(loc=mean, scale=std)
    
    # Statistical tests for normality
    _print("\nNormality tests:")
    
    # Shapiro-Wilk test (good for n < 5000)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = shapiro(data)
        _print(f"Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_p:.6f}")
        if shapiro_p > 0.05:
            _print("  → Cannot reject normality (p > 0.05)")
        else:
            _print("  → Distribution appears non-normal (p ≤ 0.05)")
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(data, 'norm', args=(mean, std))
    _print(f"Kolmogorov-Smirnov test: D = {ks_stat:.4f}, p = {ks_p:.6f}")
    
    # Anderson-Darling test
    anderson_result = anderson(data, dist='norm')
    _print(f"Anderson-Darling test: A² = {anderson_result.statistic:.4f}")
    
    # Critical values comparison
    crit_value = anderson_result.critical_values[2]  # 5% significance level
    if anderson_result.statistic < crit_value:
        _print(f"  → Cannot reject normality at 5% level (A² < {crit_value:.4f})")
    else:
        _print(f"  → Reject normality at 5% level (A² ≥ {crit_value:.4f})")
    
    # Jarque-Bera test
    jb_stat, jb_p = jarque_bera(data)
    _print(f"Jarque-Bera test: JB = {jb_stat:.4f}, p = {jb_p:.6f}")
    
    # Quantile comparison
    _print("\nQuantile comparison (theoretical vs empirical):")
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    _print(f"{'Quantile':<10} {'Empirical':<12} {'Normal':<12} {'Difference':<12}")
    _print("-" * 46)
    for q in quantiles:
        emp_q = np.percentile(data, q * 100)
        norm_q = normal_dist.ppf(q)
        diff = emp_q - norm_q
        _print(f"{q:<10} {emp_q:<12.4f} {norm_q:<12.4f} {diff:<12.4f}")
    
    # Visualization
    _print("\n4. VISUAL ANALYSIS (plots saved to disk)")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Distribution Analysis — {name}', fontsize=16, fontweight='bold')
    
    # Histogram with KDE
    ax1 = axes[0, 0]
    sns.histplot(data, kde=True, ax=ax1, stat='density', color='skyblue', edgecolor='black')
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    ax1.plot(x, normal_dist.pdf(x), 'r-', alpha=0.7, linewidth=2, label='Normal PDF')
    ax1.set_title('Histogram with Normal PDF')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Q-Q Plot
    ax2 = axes[0, 1]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (vs Normal)')
    
    # Box plot
    ax3 = axes[0, 2]
    ax3.boxplot(data, vert=True, patch_artist=True)
    ax3.set_title('Box Plot')
    ax3.set_ylabel('Value')
    
    # ECDF vs Normal CDF
    ax4 = axes[1, 0]
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax4.plot(sorted_data, ecdf, 'b-', linewidth=2, label='ECDF')
    ax4.plot(sorted_data, normal_dist.cdf(sorted_data), 'r--', alpha=0.7, label='Normal CDF')
    ax4.set_title('ECDF vs Normal CDF')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Distribution moments comparison
    ax5 = axes[1, 1]
    moments_data = [skewness, kurt]
    moments_normal = [0, 0]  # Normal distribution has skewness=0, kurtosis=0
    labels = ['Skewness', 'Excess Kurtosis']
    x_pos = np.arange(len(labels))
    width = 0.35
    
    ax5.bar(x_pos - width/2, moments_data, width, label='Data', color='skyblue', edgecolor='black')
    ax5.bar(x_pos + width/2, moments_normal, width, label='Normal', color='lightcoral', edgecolor='black')
    ax5.set_ylabel('Value')
    ax5.set_title('Distribution Moments Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels)
    ax5.legend()
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    shapiro_p_str = f'{shapiro_p:.4f}' if shapiro_p is not None else 'N/A'
    summary_data = [
        ['Statistic', 'Value'],
        ['N', f'{len(data):,}'],
        ['Mean', f'{mean:.4f}'],
        ['Std Dev', f'{std:.4f}'],
        ['Skewness', f'{skewness:.4f}'],
        ['Kurtosis', f'{kurt:.4f}'],
        ['Shapiro p', shapiro_p_str],
        ['KS p', f'{ks_p:.4f}']
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(param_dir, f"analysis_{name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close instead of show
    
    _print(f"\n  → Plot saved to: {fig_path}")
    
    # Additional insights
    _print("\n5. ADDITIONAL INSIGHTS")
    _print("-" * 40)
    
    # Check for outliers using IQR method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    _print(f"Potential outliers (IQR method): {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
    
    # Entropy
    hist_counts, _ = np.histogram(data, bins='auto')
    hist_counts = hist_counts[hist_counts > 0]
    if len(hist_counts) > 0:
        probabilities = hist_counts / hist_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        entropy = 0.0
    _print(f"Entropy: {entropy:.4f} bits")
    
    # Coefficient of variation
    cv = std / mean if mean != 0 else np.inf
    _print(f"Coefficient of variation: {cv:.4f}")
    
    _print("\n" + "=" * 60)
    _print("ANALYSIS COMPLETE")
    _print(f"Report saved to: {report_path}")
    _print("=" * 60)
    
    f.close()
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'normality_tests': {
            'shapiro': (shapiro_stat, shapiro_p) if len(data) <= 5000 else None,
            'ks': (ks_stat, ks_p),
            'anderson': anderson_result.statistic,
            'jarque_bera': (jb_stat, jb_p)
        },
        'outliers_count': len(outliers),
        'entropy': entropy,
        'cv': cv,
        'report_path': report_path,
        'fig_path': fig_path
    }


# ## r, R1, R2

# In[44]:


# Create figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot r distribution
_hist = param[:, 0] # data
sns.kdeplot(_hist, ax=axes[0], fill = True, color= 'gray', edgecolor = 'black', alpha = 0.6, clip = [np.min(_hist), np.max(_hist)] )
text = r'$\mu$: {:.1f} | $\delta$: {:.1f}'.format(
        np.mean(_hist), np.std(_hist)
)

#axes[0].set_xlim(np.min(_hist), np.max(_hist))  # Only show x from 10 to 20
axes[0].set_ylabel('Densidad')
axes[0].set_xlabel(r'$r$ [nm]')
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

axes[0].text(0.5, 0.1, text, transform=axes[0].transAxes,
             fontsize=18, verticalalignment='bottom', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))


# Plot R1 distribution
_hist = param[:, 1] # data
sns.kdeplot(_hist, ax=axes[1], fill = True, color= 'blue', edgecolor = 'black', alpha = 0.6, clip = [np.min(_hist), np.max(_hist)] )
text = r'$\mu$: {:.1f} | $\delta$: {:.1f}'.format(
        np.mean(_hist), np.std(_hist)
)

#axes[0].set_xlim(np.min(_hist), np.max(_hist))  # Only show x from 10 to 20
axes[1].set_ylabel('Densidad')
axes[1].set_xlabel(r'$\sigma_1$ [nm]')
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

axes[1].text(0.95, 0.95, text, transform=axes[1].transAxes,
             fontsize=18, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# Plot R2 distribution
_hist = param[:,2]
sns.kdeplot(_hist, ax=axes[2], fill = True, color= 'red', edgecolor = 'black', alpha = 0.6, clip = [np.min(_hist), np.max(_hist)] )

text_1 = r'$\mu$: {:.1f} | $\delta$: {:.1f}'.format(
        np.mean(_hist), np.std(_hist)
)

#axes[1].set_xlim(np.min(_hist), np.max(_hist))  # Only show x from 10 to 20
axes[2].set_ylabel('Densidad')
axes[2].set_xlabel(r'$\sigma_2$ [nm]')
axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[2].text(0.95, 0.95, text_1, transform=axes[2].transAxes,
             fontsize=16, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))


# Adjust layout
plt.tight_layout()
plt.savefig('./data_results/dist_f/R1R2_kde.png')
plt.show()


# ### Statistics r, R1, R2

# In[45]:


results = analyze_distribution(param[:, 0], name="thickness_r") # r


# In[46]:


results = analyze_distribution(param[:, 1], name="roughness_R1") # R1


# In[47]:


results = analyze_distribution(param[:, 2], name="roughness_R2") # R2


# ## Sellmeier

# In[48]:


_x = np.linspace(190e-9, 1101e-9, 911)


# In[49]:


def refractive_index(x, A, B, C, D, E):
    
    n = A + (B*x**2)/(x**2 - C**2) + (D*x**2)/(x**2 - E**2)  # Get the n
    #print(n.shape)

    return np.mean(n)


# In[50]:


# Create figure and axes
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Define a function for the common formatting
def format_subplot(ax, data, title, color, edgecolor='black', alpha=0.6, params = [True, 0.95, 0.95, 'top', 'right'] ):
    sns.kdeplot(x=data, ax=ax, fill=True, color=color, edgecolor=edgecolor, alpha=alpha, clip = [np.min(data), np.max(data)] )
    
    # Create text with statistics
    if params[0]: 
        text = f'$\\mu$: {np.mean(data):.1f} | $\\delta$: {np.std(data):.1f}'
    else: 
        text = f'$\\mu$: {np.mean(data):.1e} | $\\delta$: {np.std(data):.1e}'
    
    # Add text box in top-right corner
    ax.text(params[1], params[2], text, transform=ax.transAxes,
            fontsize=18, verticalalignment=params[3], horizontalalignment=params[4],
            bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray'))
    
    # Set limits
    #ax.set_xlim(np.min(data), np.max(data))
    ax.set_xlabel(title)
    # Format y-axis for scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    return ax

# A
_hist = param[:, 3]
ax = format_subplot(axes[0, 0], _hist, 'A', 'blue', params = [True, 0.95, 0.95, 'top', 'right'])
axes[0, 0].set_ylabel('Densidad')

# B
_hist = param[:, 4]
ax = format_subplot(axes[0, 1], _hist, 'B', 'red', params = [False, 0.05, 0.05, 'bottom', 'left'])
axes[0, 1].set_ylabel('Densidad')

# C
_hist = param[:, 5]
ax = format_subplot(axes[0, 2], _hist, 'C [nm]', 'green', params = [False, 0.95, 0.95, 'top', 'right'])
axes[0, 2].set_ylabel('Densidad')

# D
_hist = param[:, 6]
ax = format_subplot(axes[1, 0], _hist, 'D', 'orange', params = [True, 0.95, 0.95, 'top', 'right'])
axes[1, 0].set_ylabel('Densidad')

# E
_hist = param[:, 7]
ax = format_subplot(axes[1, 1], _hist, 'E [nm]', 'black', params = [True, 0.05, 0.95, 'top', 'left'])
axes[1, 1].set_ylabel('Densidad')

# n (refractive index)
# Assuming _x is defined somewhere in your code
ni_list = np.array([
    np.sqrt(refractive_index(_x, A, B, C, D, E)) for A, B, C, D, E in param[:, 3:8]
])
_hist = ni_list
ax = format_subplot(axes[1, 2], _hist, r'$n(\lambda)$', 'magenta')
axes[1, 2].set_ylabel('Densidad')

# Adjust layout
plt.tight_layout()
plt.savefig('./data_results/dist_f/sellmeier_kde.png')
plt.show()


# ### Statistics Sellmeier

# In[51]:


results = analyze_distribution(param[:, 3], name="sellmeier_A") # A


# In[52]:


results = analyze_distribution(param[:, 4]/1e6, name="sellmeier_B") # B


# In[53]:


results = analyze_distribution(param[:, 5]/1e6, name="sellmeier_C") # C


# In[54]:


results = analyze_distribution(param[:, 6], name="sellmeier_D") # D


# In[55]:


results = analyze_distribution(param[:, 7], name="sellmeier_E") # E


# In[56]:


results = analyze_distribution(ni_list, name="refractive_index_n" ) # n(lambda)


# ## Absorption Coefficients and Ne

# In[57]:


# Create figure and axes (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Define a function for the common formatting
def format_subplot(ax, data, title, color, edgecolor='black', alpha=0.6, 
                   sci_notation=False, pos_x=0.95, pos_y=0.95, 
                   valign='top', halign='right', x_sci_format=False, scale_factor=None):
    """
    Parameters:
    - scale_factor: if not None, data will be divided by this for display in text
    """
    
    # Plot KDE
    sns.kdeplot(x=data, ax=ax, fill=True, color=color, edgecolor=edgecolor, alpha=alpha, clip = [np.min(data), np.max(data)])
    
    # Prepare data for text
    if scale_factor is not None:
        scaled_data = data / scale_factor
        text_data = scaled_data
    else:
        text_data = data
    
    # Create text with statistics
    if sci_notation: 
        text = f'$\\mu$: {np.mean(text_data):.1e} | $\\sigma$: {np.std(text_data):.1e}'
    else: 
        if scale_factor is not None:
            # For scaled data (like n_e/1e26), we want to show the "e26" in the text
            text = f'$\\mu$: {np.mean(text_data):.1f}e26 | $\\delta$: {np.std(text_data):.1f}e26'
        else:
            text = f'$\\mu$: {np.mean(text_data):.1f} | $\\delta$: {np.std(text_data):.1f}'
    
    # Add text box in specified position
    ax.text(pos_x, pos_y, text, transform=ax.transAxes,
            fontsize=16, verticalalignment=valign, horizontalalignment=halign,
            bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray'))
    
    # Set limits
    #ax.set_xlim(np.min(data), np.max(data))
    
    # Format y-axis for scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Format x-axis if needed
    if x_sci_format:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    return ax

# α (top-left) - using scientific notation
_hist = param[:, 8]
ax = format_subplot(axes[0, 0], _hist, r'$\alpha_0$', 'blue',
                   sci_notation=True, pos_x=0.05, pos_y=0.05, 
                   valign='bottom', halign='left', x_sci_format=True)
axes[0, 0].set_ylabel('Densidad', fontsize=12)
axes[0, 0].set_xlabel(r'$\alpha_0$ [nm$^{⁻1}$]')


# β (top-right) - using float notation
_hist = param[:, 9]
ax = format_subplot(axes[0, 1], _hist, r'$\beta$', 'red',
                   sci_notation=False, pos_x=0.95, pos_y=0.95,
                   valign='top', halign='right')
axes[0, 1].set_ylabel('Densidad', fontsize=12)
axes[0, 1].set_xlabel(r'$\beta$ [eV$^{⁻1}]$')


# λ (bottom-left) - using float notation
_hist = param[:, 10]
ax = format_subplot(axes[1, 0], _hist, r'$\lambda_g$', 'green',
                   sci_notation=False, pos_x=0.95, pos_y=0.95,
                   valign='top', halign='right')
axes[1, 0].set_ylabel('Densidad', fontsize=12)
axes[1, 0].set_xlabel(r'$\lambda_g$ [nm]')


# n_e (bottom-right) - scaled by 1e26
_hist = param[:, 11]
ax = format_subplot(axes[1, 1], _hist, r'$n_e$', 'purple',
                   sci_notation=False, pos_x=0.95, pos_y=0.95,
                   valign='top', halign='right', scale_factor=1e26)
axes[1, 1].set_ylabel('Densidad', fontsize=12)
axes[1, 1].set_xlabel(r'$n_e$ [nm$^{⁻3}$]')


# Adjust layout
plt.tight_layout()
plt.savefig('./data_results/dist_f/combined_absorption_ne.png')
plt.show()


# ### Statistics Sellemeier & Ne

# In[58]:


results = analyze_distribution(param[:, 8]/1e-3, name="absorption_alpha") # alpha


# In[59]:


results = analyze_distribution(param[:, 9], name="absorption_beta") # beta


# In[60]:


results = analyze_distribution(param[:, 10], name="absorption_lambda") # lambda


# In[61]:


results = analyze_distribution(param[:, 11]/1e26, name="free_carriers_ne") # ne


# ## Split by interval

# In[62]:


bin_indices = []


# In[63]:


for i in range(len(intervals) - 1):
    bin_indices.append( np.where((thickness >= intervals[i]) & (thickness <= intervals[i+1]))[0] )


# In[64]:


bin_indices[4]


# In[65]:


k = 0
for i in bin_indices:
    np.save('./bins/bin_index_{}'.format(k), i)
    k+=1


# ## Generating indexes

# This contains the min, max, average and std value for each dominan region. 

# In[66]:


k = 0
for _bin in bin_indices:
    _arr = param[_bin][:, :-1]
    _l = np.array([ 
        np.min(_arr, axis = 0), 
        np.max(_arr, axis = 0), 
        np.mean(_arr, axis = 0), 
        np.std(_arr, axis = 0) ])
    print(_l.T.shape)
    np.save('./bins/min_max_mean_std_{}'.format(k), _l.T)
    k+=1


# This contains the samples for each dominan region. 

# In[67]:


k = 0
for _bin in bin_indices:
    _arr = param[_bin][:, :-1]
    print(_arr.shape)
    np.save('./bins/region_{}'.format(k), _arr)
    k+=1


# ## Ridger Plots

# In this section I present the distribution of each parameters across the given 5 intervals. 

# In[68]:


values = []
for _bin in bin_indices:
    _arr = param[_bin][:, :-1]
    print(_arr.shape)
    values.append(_arr)
    
    #np.save('./bins/region_{}'.format(k), _arr)


# In[69]:


_sellmeier = param[:, 3:8]
_sellmeier.shape


# In[70]:


_x = np.linspace(190e-9, 1101e-9, 911)


# In[71]:


def refractive_index(x, A, B, C, D, E):
    
    n = A + (B*x**2)/(x**2 - C**2) + (D*x**2)/(x**2 - E**2)  # Get the n
    #print(n.shape)

    return np.mean(n)


# In[72]:


_indice_de_refraccion = np.sqrt( np.array([ refractive_index(_x, *x) for x in _sellmeier ]) )
_indice_de_refraccion.shape


# In[73]:


n_region = []
for _bin in bin_indices:
    _arr = _indice_de_refraccion[_bin]
    print(_arr.shape)
    n_region.append(_arr)


# In[74]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

names = [
    'Espesor [nm]', 
    r'$\sigma_1$ [nm]', 
    r'$\sigma_2$ [nm]',
    'A', 
    'B', 
    'C [nm]', 
    'D', 
    'E [nm]', 
    r'$\alpha_0$ [nm]$^{⁻1}$', 
    r'$\beta$ [eV]$^{-1}$', 
    r'$\lambda_g$ [nm]', 
    r'$n_e$ [nm]$^{-3}$'
]

names_f = [
    'thickness', 
    'sigma1', 
    'sigma2', 
    'A', 
    'B',
    'C', 
    'D', 
    'E',
    'alpha', 
    'beta', 
    'lambda', 
    'ne',
]

for _i in range(len(names)): 
    # Put your arrays in a list
    _element = _i
    arrays = [ values[i_][:, _element] for i_ in range(0,5) ] 
    '''
    labels = [
        'R1: [100, 250] n', 
        'R2: [250, 600] nm', 
        'R3: [600, 950] nm', 
        'R4: [950, 1100] nm', 
        'R5: [1100, 1500] nm' 
    ]  # Custom labels
    '''
    labels = [
        r'$R_1$', 
        r'$R_2$', 
        r'$R_3$', 
        r'$R_4$', 
        r'$R_5$' 
    ]
    
    plt.figure(figsize=(4, 4))
    
    # Create subplots
    fig, axes = plt.subplots(len(arrays), 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.1)  # Adjust spacing to create overlap effect
    
    # Plot each distribution
    for i, (ax, array, label) in enumerate(zip(axes, arrays, labels)):
        # Create KDE plot
        sns.kdeplot(data=array, ax=ax, fill=True, alpha=0.8, 
                    linewidth=1, color=plt.cm.viridis(i/len(arrays)), clip = [np.min(array), np.max(array)] )
        
        # Customize each subplot
        ax.set_ylabel(label, rotation=0, ha='right', fontsize = 28)
        ax.set_yticks([])
        #ax.set_xticks(labelsize=12)
        ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #ax.set_xlim( 
        #    np.min(param[:, _i]), 
        #    np.max(param[:, _i])
        #)
    
    # Set common x-axis label
    axes[-1].set_xlabel(names[_i], fontsize = 28)
    axes[-1].tick_params(axis='x', labelsize=18)

    offset_text = axes[-1].xaxis.get_offset_text()
    offset_text.set_fontsize(18)  # Larger than tick labels
    
    plt.savefig('./data_results/ridge_plot/{}.png'.format(names_f[_i]))
    plt.show()


# In[75]:


arrays = n_region

labels = [
    r'$R_1$', 
    r'$R_2$', 
    r'$R_3$', 
    r'$R_4$', 
    r'$R_5$' 
]

plt.figure(figsize=(4, 4))

# Create subplots
fig, axes = plt.subplots(len(arrays), 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.1)  # Adjust spacing to create overlap effect

# Plot each distribution
for i, (ax, array, label) in enumerate(zip(axes, arrays, labels)):
    # Create KDE plot
    sns.kdeplot(data=array, ax=ax, fill=True, alpha=0.8, 
                linewidth=1, color=plt.cm.viridis(i/len(arrays)), clip = [np.min(array), np.max(array)])
    
    # Customize each subplot
    ax.set_ylabel(label, rotation=0, ha='right', fontsize = 28)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.set_xlim( 
    #    np.min(_indice_de_refraccion), 
    #    np.max(_indice_de_refraccion)
    #)

# Set common x-axis label
axes[-1].set_xlabel(r'$n(\lambda)$', fontsize = 28)
axes[-1].tick_params(axis='x', labelsize=18)
offset_text = axes[-1].xaxis.get_offset_text()
offset_text.set_fontsize(18)  # Larger than tick labels

plt.savefig('./data_results/ridge_plot/indice.png' )
plt.show()


# ## Join Images

# In[76]:


from PIL import Image
import os
import glob

def create_simple_grid(image_folder, rows, cols, output_path="grid_output.png", 
                      spacing=10, scale_factor=None, target_resolution=None):
    """
    Enhanced function to create image grid with resolution control
    
    Args:
        image_folder: Folder containing PNG images
        rows: Number of rows
        cols: Number of columns
        output_path: Output file path
        spacing: Space between images
        scale_factor: Reduce size by this factor (e.g., 0.5 for half size)
        target_resolution: Target max dimension in pixels (e.g., 1000 for 1000px max side)
    """
    # Get all PNG files
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    image_files.sort()  # Sort alphabetically
    
    if not image_files:
        print("No PNG files found!")
        return
    
    # Load images
    images = []
    for img_path in image_files[:rows*cols]:
        img = Image.open(img_path)
        
        # Apply resolution reduction if requested
        if scale_factor or target_resolution:
            img = reduce_resolution(img, scale_factor, target_resolution)
        
        images.append(img)
    
    # Get image dimensions (assuming all are same size after processing)
    img_width, img_height = images[0].size
    
    # Calculate grid size
    grid_width = cols * img_width + (cols - 1) * spacing
    grid_height = rows * img_height + (rows - 1) * spacing
    
    # Create blank image
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * (img_width + spacing)
        y = row * (img_height + spacing)
        
        grid.paste(img, (x, y))
    
    # Apply output quality reduction (for JPEG)
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        grid.save(output_path, 'JPEG', quality=85, optimize=True)
    else:
        # For PNG, use optimize flag and optionally reduce palette
        grid.save(output_path, optimize=True)
    
    print(f"Grid created with {len(images)} images")
    print(f"Grid size: {grid_width}x{grid_height} pixels")
    print(f"Saved to: {output_path}")
    
    return grid

def reduce_resolution(image, scale_factor=None, target_resolution=None):
    """
    Reduce image resolution using different methods
    """
    original_width, original_height = image.size
    
    if scale_factor and 0 < scale_factor < 1:
        # Method 1: Scale by factor
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        print(f"Scaling from {original_width}x{original_height} to {new_width}x{new_height}")
    
    elif target_resolution:
        # Method 2: Scale to target max dimension
        max_dimension = max(original_width, original_height)
        if max_dimension > target_resolution:
            scale = target_resolution / max_dimension
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            print(f"Reducing to max {target_resolution}px: {original_width}x{original_height} → {new_width}x{new_height}")
        else:
            new_width, new_height = original_width, original_height
    
    else:
        # No reduction
        return image
    
    # Resize with high-quality downsampling
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def batch_reduce_resolution(image_folder, output_folder, scale_factor=None, 
                           target_resolution=None, quality=85):
    """
    Pre-process all images in a folder to reduce resolution before creating grid
    """
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    for img_path in image_files:
        img = Image.open(img_path)
        img_reduced = reduce_resolution(img, scale_factor, target_resolution)
        
        # Save with quality settings
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            img_reduced.save(output_path, 'JPEG', quality=quality, optimize=True)
        else:
            img_reduced.save(output_path, optimize=True)
    
    print(f"Reduced {len(image_files)} images saved to {output_folder}")


# In[77]:


create_simple_grid(
    image_folder="./data_results/ridge_plot/rugosidades/",
    rows=1,  # 3 rows
    cols=3,  # 4 columns
    output_path="./data_results/ridge_plot_comparison/combined_rugosidades.png",
    spacing=0,  # 5 pixels spacing
    scale_factor=0.5  # Reduce each image to 50% before creating grid
)


# In[78]:


create_simple_grid(
    image_folder="./data_results/ridge_plot/Absorcion/",
    rows=2,  # 3 rows
    cols=2,  # 4 columns
    output_path="./data_results/ridge_plot_comparison/combined_absorcion.png",
    spacing=0,  # 5 pixels spacing
    scale_factor=0.5  # Reduce each image to 50% before creating grid
)


# In[79]:


create_simple_grid(
    image_folder="./data_results/ridge_plot/Sellmeier/",
    rows=2,  # 3 rows
    cols=3,  # 4 columns
    output_path="./data_results/ridge_plot_comparison/combined_sellmeier.png",
    spacing=0,  # 5 pixels spacing
    scale_factor=0.5  # Reduce each image to 50% before creating grid
)


# ## Table for each bin

# In[80]:


param_new = np.column_stack(
    (param[:, :-1], _indice_de_refraccion)
)
param_new.shape


# In[81]:


import numpy as np
from scipy import stats

k = 0
for _bin in bin_indices:
    _arr = param[_bin]
    
    # Calcular todas las métricas
    metrics = {
        'min': np.min(_arr, axis=0),
        'max': np.max(_arr, axis=0),
        'mean': np.mean(_arr, axis=0),
        'std': np.std(_arr, axis=0),
        'median': np.median(_arr, axis=0),
        'q25': np.percentile(_arr, 25, axis=0),
        'q75': np.percentile(_arr, 75, axis=0),
    }
    
    # Calcular IQR
    metrics['iqr'] = metrics['q75'] - metrics['q25']
    
    # Calcular skewness y kurtosis
    metrics['skewness'] = stats.skew(_arr, axis=0, nan_policy='omit')
    metrics['kurtosis'] = stats.kurtosis(_arr, axis=0, nan_policy='omit')
    
    # Calcular outliers
    lower_bound = metrics['q25'] - 1.5 * metrics['iqr']
    upper_bound = metrics['q75'] + 1.5 * metrics['iqr']
    metrics['outliers_count'] = np.sum((_arr < lower_bound) | (_arr > upper_bound), axis=0)
    
    # Crear array ordenado de métricas
    metric_names = ['min', 'max', 'mean', 'std', 'median', 'q25', 'q75', 
                    'iqr', 'skewness', 'kurtosis', 'outliers_count']
    
    _l = np.array([metrics[name] for name in metric_names])
    
    print(f"Bin {k}: Shape de métricas: {_l.T.shape}")
    
    # Guardar tanto las métricas como los nombres
    np.save('./bins/all_metrics_{}'.format(k), _l.T)
    
    k += 1


# In[82]:


#! pip install prettytable
from prettytable import PrettyTable


# In[83]:


names = ['r [nm]', 
         r'$\sigma_{1} [nm]$', r'$\sigma_{2}$ [nm]',
         'A', 'B', 'C [nm]', 'D', 'E [nm]', 
         r'$\alpha_{0}$ [nm]$^{-1}$', r'$\beta$ [eV]$^{-1}$', r'$\lambda_{g}$ [nm]', 
         r'$n_{e}$ [nm]$^{-3}$', 
         r'$n(\lambda)$'
        ]


# ## Bin: 1

# In[84]:


k = 0
bin0 = np.load('./bins/all_metrics_0.npy')
table = PrettyTable()

table.field_names = ["", 'Min', 'Max', r'$\bar{x}$', '$\delta(x)$', 'Mediana', 
                    'IQR', 'Asimetría', 'Curtosis', '\# At.']

j = 0
for i in bin0:
    table.add_row([names[j],
                    "{:.3e}".format(i[0]),
                   "{:.3e}".format(i[1]), 
                   "{:.3e}".format(i[2]), 
                   "{:.3e}".format(i[3]),
                   "{:.3e}".format(i[4]),
                   #"{:.3e}".format(i[5]),
                   #"{:.3e}".format(i[6]),
                   "{:.3e}".format(i[7]),
                   "{:.3e}".format(i[8]), 
                   "{:.3e}".format(i[9]),
                   "{:.0f}".format(i[10]),
                  ]
                  )
    j +=1
print(table)


# In[85]:


latex_code = table.get_latex_string()
latex_dir = './data_results/latex_tables'
os.makedirs(latex_dir, exist_ok=True)
with open(f'{latex_dir}/latex_bin_{k}.txt', 'w', encoding='utf-8') as _f:
    _f.write(latex_code)
print(f"\n--- LaTeX Code (Bin {k}) saved to {latex_dir}/latex_bin_{k}.txt ---")
print(latex_code)


# ## Bin: 2

# In[86]:


k = 1
bin0 = np.load('./bins/all_metrics_1.npy')
table = PrettyTable()

table.field_names = ["", 'Min', 'Max', r'$\bar{x}$', '$\delta(x)$', 'Mediana', 
                    'IQR', 'Asimetría', 'Curtosis', '\# At.']

j = 0
for i in bin0:
    table.add_row([names[j],
                    "{:.3e}".format(i[0]),
                   "{:.3e}".format(i[1]), 
                   "{:.3e}".format(i[2]), 
                   "{:.3e}".format(i[3]),
                   "{:.3e}".format(i[4]),
                   #"{:.3e}".format(i[5]),
                   #"{:.3e}".format(i[6]),
                   "{:.3e}".format(i[7]),
                   "{:.3e}".format(i[8]), 
                   "{:.3e}".format(i[9]),
                   "{:.0f}".format(i[10]),
                  ]
                  )
    j +=1
print(table)


# In[87]:


latex_code = table.get_latex_string()
latex_dir = './data_results/latex_tables'
os.makedirs(latex_dir, exist_ok=True)
with open(f'{latex_dir}/latex_bin_{k}.txt', 'w', encoding='utf-8') as _f:
    _f.write(latex_code)
print(f"\n--- LaTeX Code (Bin {k}) saved to {latex_dir}/latex_bin_{k}.txt ---")
print(latex_code)


# ## Bin: 3

# In[88]:


k = 2
bin0 = np.load('./bins/all_metrics_2.npy')
table = PrettyTable()

table.field_names = ["", 'Min', 'Max', r'$\bar{x}$', '$\delta(x)$', 'Mediana', 
                    'IQR', 'Asimetría', 'Curtosis', '\# At.']

j = 0
for i in bin0:
    table.add_row([names[j],
                    "{:.3e}".format(i[0]),
                   "{:.3e}".format(i[1]), 
                   "{:.3e}".format(i[2]), 
                   "{:.3e}".format(i[3]),
                   "{:.3e}".format(i[4]),
                   #"{:.3e}".format(i[5]),
                   #"{:.3e}".format(i[6]),
                   "{:.3e}".format(i[7]),
                   "{:.3e}".format(i[8]), 
                   "{:.3e}".format(i[9]),
                   "{:.0f}".format(i[10]),
                  ]
                  )
    j +=1
print(table)


# In[89]:


latex_code = table.get_latex_string()
latex_dir = './data_results/latex_tables'
os.makedirs(latex_dir, exist_ok=True)
with open(f'{latex_dir}/latex_bin_{k}.txt', 'w', encoding='utf-8') as _f:
    _f.write(latex_code)
print(f"\n--- LaTeX Code (Bin {k}) saved to {latex_dir}/latex_bin_{k}.txt ---")
print(latex_code)


# ## Bin: 4

# In[90]:


k = 3
bin0 = np.load('./bins/all_metrics_3.npy')
table = PrettyTable()

table.field_names = ["", 'Min', 'Max', r'$\bar{x}$', '$\delta(x)$', 'Mediana', 
                    'IQR', 'Asimetría', 'Curtosis', '\# At.']

j = 0
for i in bin0:
    table.add_row([names[j],
                    "{:.3e}".format(i[0]),
                   "{:.3e}".format(i[1]), 
                   "{:.3e}".format(i[2]), 
                   "{:.3e}".format(i[3]),
                   "{:.3e}".format(i[4]),
                   #"{:.3e}".format(i[5]),
                   #"{:.3e}".format(i[6]),
                   "{:.3e}".format(i[7]),
                   "{:.3e}".format(i[8]), 
                   "{:.3e}".format(i[9]),
                   "{:.0f}".format(i[10]),
                  ]
                  )
    j +=1
print(table)


# In[91]:


latex_code = table.get_latex_string()
latex_dir = './data_results/latex_tables'
os.makedirs(latex_dir, exist_ok=True)
with open(f'{latex_dir}/latex_bin_{k}.txt', 'w', encoding='utf-8') as _f:
    _f.write(latex_code)
print(f"\n--- LaTeX Code (Bin {k}) saved to {latex_dir}/latex_bin_{k}.txt ---")
print(latex_code)


# ## Bin: 5

# In[92]:


k = 4
bin0 = np.load('./bins/all_metrics_4.npy')
table = PrettyTable()

table.field_names = ["", 'Min', 'Max', r'$\bar{x}$', '$\delta(x)$', 'Mediana', 
                    'IQR', 'Asimetría', 'Curtosis', '\# At.']

j = 0
for i in bin0:
    table.add_row([names[j],
                    "{:.3e}".format(i[0]),
                   "{:.3e}".format(i[1]), 
                   "{:.3e}".format(i[2]), 
                   "{:.3e}".format(i[3]),
                   "{:.3e}".format(i[4]),
                   #"{:.3e}".format(i[5]),
                   #"{:.3e}".format(i[6]),
                   "{:.3e}".format(i[7]),
                   "{:.3e}".format(i[8]), 
                   "{:.3e}".format(i[9]),
                   "{:.0f}".format(i[10]),
                  ]
                  )
    j +=1
print(table)


# In[93]:


latex_code = table.get_latex_string()
latex_dir = './data_results/latex_tables'
os.makedirs(latex_dir, exist_ok=True)
with open(f'{latex_dir}/latex_bin_{k}.txt', 'w', encoding='utf-8') as _f:
    _f.write(latex_code)
print(f"\n--- LaTeX Code (Bin {k}) saved to {latex_dir}/latex_bin_{k}.txt ---")
print(latex_code)

