#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:55:06 2025

@author: mabedi
This code loaded the validation data and Pre-trained models and plots:
    1. Velocity mode, 2. Real part of the Finite-difference simulation, 3. Real part of the Pretrained Model's prediction
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
# import scipy.io
from My_utilities_Gabor import load_validation_data,sin_activation
from My_CustomLayers import CustomLayer3D,GaborFunctionLayer

dtype="float32"
tf.keras.backend.set_floatx(dtype)



# User-defined input parameters, select among the existing models in the PreTrained_Models folder
frequency =10        # Frequency in Hz 
epoch=100000
model_type='Gabor'#'PINN','Gabor'
velocity_model='simple'#'simple','overthrust','marmousi'
use_PML=True

PML = "NOPML" if use_PML else "NOPML"
epoch = 99999 if epoch == 100000 else epoch
    
model_path = f"PreTrained_Models/{velocity_model}_{model_type}_{frequency}_{PML}/u_model_epoch_{epoch}.keras"


# Define the number of collocation points per epoch
if velocity_model=='overthrust':
    fig_siz=[15,4]
    if frequency==10:
        npts_x_val = 500 # Number of points along x-axis 501,201
        npts_z_val = 160  # Number of points along z-axis 161, 201
elif velocity_model=='simple':
    fig_siz=[10,4]
    if frequency==4:
        npts_x_val = 100 # Number of points along x-axis 71 in Gabor paper
        npts_z_val = 100  # Number of points along z-axis 71
    if frequency==10:
        npts_x_val = 200 # Number of points along x-axis 51 in Gabor paper
        npts_z_val = 200  # Number of points along z-axis 51
    elif frequency==20:
        npts_x_val = 200 # Number of points along x-axis  301
        npts_z_val = 200  # Number of points along z-axis 
elif velocity_model=='marmousi':
    fig_siz=[12,4]
    if frequency==10:
        npts_x_val = 150 # Number of points along x-axis = 151
        npts_z_val = 100  # Number of points along z-axis 101


#%% Load the validation data
data = load_validation_data(
    frequency=frequency,
    velocity_model=velocity_model,
    dtype=dtype,
    use_PML=use_PML
)

# Unpack with original variable names

dU_2d         = data['dU_2d']
xz_val        = data['xz_val']
s_xz          = data['s_xz']
factor        = data['factor']
v0            = data['v0']
v_val         = data['v_val']
domain_bounds = data['domain_bounds'] 
a_x,b_x,a_z,b_z=domain_bounds
domain_bounds_valid=domain_bounds

omega = np.float32(frequency*2*np.pi)  # Angular frequency

if use_PML:
    L_PML=0.5
    omega0=omega
    a0=1.
    print(a0)
    xz_PML=a_x,b_x,a_z,b_z

activation_penultima = 'sigmoid'
if model_type=='PINN':
    activation_penultima=sin_activation
 #%% plotting functions   
plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for text rendering
    "font.family": "serif",        # Set font family to serif
    "font.serif": ["Times"],       # Use Times as the serif font
    "font.size": 14,               # Set the default font size
    "axes.titlesize": 19,          # Title font size
    "axes.labelsize": 16,          # Label font size
    "xtick.labelsize": 14,         # x-tick label font size
    "ytick.labelsize": 14,         # y-tick label font size
    "legend.fontsize": 14,         # Legend font size
    "text.latex.preamble": r"\usepackage{amsmath}"  # Use amsmath for better LaTeX rendering
})

# # # ###### OLD!!! COMMENT if trained using the not using Gabor_enhanced_PINN.py and NOT the models in the PreTrained_Models folder
class EmbedderLayer(tf.keras.layers.Layer):#The old embedder
    def __init__(self, domain_bounds, **kwargs):
        super(EmbedderLayer, self).__init__(**kwargs)
        self.domain_bounds = domain_bounds  # Store domain bounds for normalization

    @tf.function()
    def call(self, inputs):
        input1 =  (inputs)
        input2 = tf.math.multiply(input1 , 2.0)
        input4 = tf.math.multiply(input1 , 4.0)
        input8 = tf.math.multiply(input1 , 8.0)
        
        input_all = tf.concat([input1, input2, input4, input8], axis=1)
        # Apply sine and cosine functions
        sin_embed = tf.sin(input_all)
        cos_embed = tf.cos(input_all)

        # Concatenate original input, sine, and cosine embeddings
        output = tf.concat([inputs, sin_embed, cos_embed], axis=1)
        return output
    def get_config(self):
        config = super(EmbedderLayer, self).get_config()
        config.update({"domain_bounds": self.domain_bounds})
        return config

def model_prediction(model_path,x_in):
    u_model =keras.models.load_model(model_path,
        custom_objects={
            'GaborFunctionLayer': GaborFunctionLayer,
            'CustomLayer3D': CustomLayer3D,
            'EmbedderLayer': EmbedderLayer,
            'sin_activation': sin_activation}, compile=False)

    prediction = u_model(x_in)
    
    # Extract real and imaginary parts
    u_real = prediction[:, 0]  # Real part
    # Reshape wavefields into 2D grids
    u_real_grid = tf.reshape(u_real, (npts_z_val, npts_x_val)).numpy()
    u_imag = prediction[:, 0]  # imag part
    # Reshape wavefields into 2D grids
    u_imag_grid = tf.reshape(u_imag, (npts_z_val, npts_x_val)).numpy()
    return u_real_grid,u_imag_grid


#%% plottings
c_lims=[-np.max(np.abs(dU_2d)),np.max(np.abs(dU_2d))]
u_real_grid = tf.reshape(dU_2d[:,0], (npts_z_val, npts_x_val)).numpy()
u_imag_grid = tf.reshape(dU_2d[:,1], (npts_z_val, npts_x_val)).numpy()

#Velocity model
plt.figure(figsize=fig_siz)
plt.imshow(tf.reshape(v_val, (npts_z_val, npts_x_val)), extent=[a_x, b_x, b_z, a_z], origin="upper", cmap="viridis", aspect="auto")
plt.title("Velocity model")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
cbar = plt.colorbar(label='$v$ (km/s)', orientation='vertical')
cbar.ax.invert_yaxis()
plt.tight_layout()  
plt.show()

#Loaded Finite_difference simulation
plt.figure(figsize=fig_siz)

# Add figure-wide title
plt.suptitle("FD simulation", fontsize=21, y=.98)

plt.subplot(1, 2, 1)
plt.imshow(u_real_grid, extent=[a_x, b_x, b_z, a_z], origin="upper",
           cmap="seismic", aspect="auto", interpolation='none')
plt.clim(c_lims)
plt.title("Real part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(u_imag_grid, extent=[a_x, b_x, b_z, a_z], origin="upper",
           cmap="seismic", aspect="auto", interpolation='none')
plt.clim(c_lims)
plt.title("Imaginary part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()

plt.tight_layout()
plt.show()


#Model prediction:
u_real_prediction,u_imag_prediction=model_prediction(model_path,xz_val)

#Loaded Finite_difference simulation
plt.figure(figsize=fig_siz)

# Add figure-wide title
plt.suptitle(f"{model_type} Prediction", fontsize=21, y=.98)

plt.subplot(1, 2, 1)
plt.imshow(u_real_prediction, extent=[a_x, b_x, b_z, a_z], origin="upper",
           cmap="seismic", aspect="auto", interpolation='none')
plt.clim(c_lims)
plt.title("Real part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(u_imag_prediction, extent=[a_x, b_x, b_z, a_z], origin="upper",
           cmap="seismic", aspect="auto", interpolation='none')
plt.clim(c_lims)
plt.title("Imaginary part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()

plt.tight_layout()
plt.show()
