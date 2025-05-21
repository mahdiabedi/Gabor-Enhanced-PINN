#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:23:37 2025

@author: mabedi

This code produces the results in the paper titled 
"Gabor-Enhanced Physics-Informed Neural Networks for Fast Simulations of Acoustic Wavefields"
By M.M. Abedi, D. Pardo, T. Alkhalifah

This code includes the main training loop
"""

import tensorflow as tf
import numpy as np
import time as time
import keras
import os
from My_utilities_Gabor import load_training_and_validation_data,compute_U0,save_model_and_history,sin_activation,create_xz_reg
from My_CustomLayers import CustomLayer3D,EmbedderLayer,GaborFunctionLayer

dtype="float32"
tf.keras.backend.set_floatx(dtype)



# User-defined input parameters 
frequency =10        # Frequency in Hz 
neurons = 64 # Number of neurons in the hidden layers
neurons_final = 64  # Number of neurons in penultimate layer 
activation = sin_activation
learning_rate=0.001
num_epochs=100000

model_type='Gabor'#'PINN','Gabor'
velocity_model='marmousi'#'simple','overthrust','marmousi'

use_source_reg=True
use_lr_decay=True
use_PML=False

beta=200# Regularization parameter for soft constraint
seed=1234


# Define the number of collocation points per epoch
if velocity_model=='overthrust':
    if frequency==10:
        npts_x = 501 # Number of points along x-axis 501,201
        npts_z = 161  # Number of points along z-axis 161, 201
elif velocity_model=='simple':
    if frequency==4:
        npts_x = 71 # Number of points along x-axis 71 in Gabor paper
        npts_z = 71  # Number of points along z-axis 71
    if frequency==10:
        npts_x = 51 # Number of points along x-axis 51 in Gabor paper
        npts_z = 51  # Number of points along z-axis 51
    elif frequency==20:
        npts_x = 201 # Number of points along x-axis  301
        npts_z = 201  # Number of points along z-axis 
elif velocity_model=='marmousi':
    if frequency==10:
        npts_x = 151 # Number of points along x-axis = 151
        npts_z = 101  # Number of points along z-axis 101
        
#Validation collocation points:
npts_x_val=npts_x-1#500
npts_z_val=npts_z-1#160
if velocity_model=='simple' and frequency==10:
    npts_x_val=200
    npts_z_val=200
if velocity_model=='simple' and frequency==4:
    npts_x_val=100
    npts_z_val=100
    
#Create a folder to save the models and history:
os.makedirs('Results/Models', exist_ok=True)

#%% Load the validation data and training collocation points 
data = load_training_and_validation_data(
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
v_all         = data['v_all']
xz_all        = data['xz_all']
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
    npts_x=int(npts_x*((b_x-a_x)+2*L_PML)/(b_x-a_x))
    npts_z=int(npts_z*((b_z-a_z)+2*L_PML)/(b_z-a_z))

npts=npts_z*npts_x#number of colocation points in each epoch

n_all=xz_all.shape[0]

random_indices = np.random.choice(n_all, npts, replace=False)
xz_train = tf.gather(xz_all, random_indices)  # Gather corresponding indices
v_train = tf.gather(v_all, random_indices)  # Gather corresponding indices

activation_penultima = 'sigmoid'
if model_type=='PINN':
    activation_penultima=sin_activation
#%%Model buildng
def make_u_model(neurons, activation=tf.math.sin, activation_penultima=tf.math.sin, neurons_final=None, dtype=tf.float32, trainableLastLayer=False,v0=1., omega=1. ,model_type='PINN',seed=1234):
    # Xavier (Glorot) initialization is commonly used for PINNs
    kernel_regularizer =keras.regularizers.L2(l2=0)
    # Use GlorotNormal (Xavier) initializer for the kernel
    b_init =keras.initializers.Zeros()  # Use zero bias initialization

    if neurons_final is None:
        neurons_final = neurons

    # Input layer
    l0 =keras.layers.Input(shape=(2,), name="x_input", dtype=dtype)
    
    # Apply the embedding layer
    l1 = EmbedderLayer(name="embedder",K = 4)(l0)
    
    # First dense layer 
    l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype,
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed),  # Different seed per layer
                            bias_initializer=b_init, name="layer_1")(l1)
    
    # Second dense layer 
    l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_2",
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.GlorotNormal(seed=seed+1),  # Different seed
                            bias_initializer=b_init)(l1)
    
    # # Third dense layer 
    # l1 =keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_3",
    #                         kernel_regularizer=kernel_regularizer,
    #                         kernel_initializer=keras.initializers.GlorotNormal(seed=seed+2),  # Different seed
    #                         bias_initializer=b_init)(l1)
    
    if model_type=='PINN':
        # #NO Gabor:
        # # # Penultimate layer
        l1 =keras.layers.Dense(neurons_final, activation=activation_penultima, dtype=dtype, name="penultimate_layer",
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=keras.initializers.GlorotNormal(seed=seed+3),  # Different seed
                                bias_initializer=b_init)(l1)

        output =keras.layers.Dense(2, use_bias=False, trainable=trainableLastLayer, dtype=dtype, name='Output_layer',
                                kernel_initializer=keras.initializers.GlorotNormal(seed=seed+4))(l1)  # Different seed
    
    elif model_type=='Gabor':
        # #GABOR:
        # Penultimate layer 
        l1 =keras.layers.Dense(neurons_final, activation=activation_penultima, dtype=dtype, name="penultimate_layer",
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=keras.initializers.GlorotNormal(seed=seed+3),  # Different seed
                                bias_initializer=b_init)(l1)
        
        # # # 1. Add Gabor with trainbale delta and theta before the output layer
        l1 = GaborFunctionLayer(neurons_final//2, v=v0, omega=omega, dtype=dtype,name="Gabor_layer")(l1)
        output = CustomLayer3D(neurons_final//2, dtype=dtype, trainable=trainableLastLayer,seed=seed+4)(l1)

    
    # # Define models
    u_bases =keras.Model(inputs=l0, outputs=l1)
    u_model =keras.Model(inputs=l0, outputs=output)
    
    return u_model, u_bases


# Define the loss function for the 2D Helmholtz equation (separating real and imaginary parts)
@tf.function()
def make_loss(u_model, U0, xz, v, v0, omega, use_source_reg,source_reg,cal_error=False,dU_2d_val=0.,use_Vin=False):
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # x and z each have shape [batch_size, 1]
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, z])  # Watch both x and z
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, z])
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]
            
            u = u_model(xz)  # Scattered wavefield, 2D outputs: real and imaginary

            # Split u into real and imaginary parts
            u_real = u[:, 0:1]  # Real part
            u_imag = u[:, 1:2]  # Imaginary part

        # Compute the first derivatives w.r.t both x and z for real and imaginary parts
        u_x_real,u_z_real = tape2.gradient(u_real, [x,z])  # First derivative w.r.t x (shape: [batch_size, 1])
        u_x_imag,u_z_imag = tape2.gradient(u_imag, [x,z])  # First derivative w.r.t x (imaginary)

    # Compute the second derivatives (Laplacian components) for real and imaginary parts
    u_xx_real = tape1.gradient(u_x_real, x)  # Second derivative w.r.t x
    u_zz_real = tape1.gradient(u_z_real, z)  # Second derivative w.r.t z
    u_xx_imag = tape1.gradient(u_x_imag, x)  # Second derivative w.r.t x (imaginary)
    u_zz_imag = tape1.gradient(u_z_imag, z)  # Second derivative w.r.t z (imaginary)

    # Compute the 2D Laplacian (sum of second derivatives w.r.t x and z) for both real and imaginary parts
    laplacian_u_real = u_xx_real + u_zz_real  # Real part of Laplacian
    laplacian_u_imag = u_xx_imag + u_zz_imag  # Imaginary part of Laplacian
    # Clean up tapes
    del tape1, tape2

    # Split real and imaginary parts of U0 
    U0_real = U0[:,0:1]  # Shape: [batch_size, 1]
    U0_imag = U0[:,1:2]  # Shape: [batch_size, 1]
    # Helmholtz equation residual for real and imaginary parts in 2D
    helmholtz_residual_real = omega**2 * (1 / v**2) * u_real + laplacian_u_real + omega**2 * (1 / v**2 - 1 / v0**2) * U0_real
    helmholtz_residual_imag = omega**2 * (1 / v**2) * u_imag + laplacian_u_imag + omega**2 * (1 / v**2 - 1 / v0**2) * U0_imag

    # Loss term from the Helmholtz equation residual for real and imaginary parts
    pde_loss_real =  tf.reduce_mean(tf.square(helmholtz_residual_real))
    pde_loss_imag =  tf.reduce_mean(tf.square(helmholtz_residual_imag))

    # Total loss is the sum of both real and imaginary losses
    total_loss = pde_loss_real + pde_loss_imag
    
    if use_source_reg:# Regularization loss term: Penalize the scattered wavefield near the source, scaled by `factor_d`
        # Compute scattered wavefield at regularization points
        xz_reg, factor_d,_=source_reg
        u_reg = u_model(xz_reg)
        
        # Regularization loss (only consider valid points using the mask)
        reg_loss = tf.reduce_mean( factor_d * tf.square(u_reg))
        total_loss = total_loss+reg_loss
    else:
        reg_loss=0
    if cal_error:    
        error=tf.reduce_mean(tf.abs(dU_2d_val-u))
    else:
        error=0.
    return total_loss,reg_loss,error

# PML %%%%%%%%%%%%%%%%%%%%%%%
@tf.function
def make_loss_PML(u_model, U0, xz, v, v0, omega,  a0, omega0, L_PML,xz_PML, use_source_reg, source_reg, cal_error=False, dU_2d_val=0.0):
    # epsilon = 1e-8
    xb1,xb2,zb1,zb2=xz_PML#boundaries
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # Split x and z coordinates
    c = ( a0 * omega0) / (omega * L_PML**2)  # Coupling constant
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, z])
        lx = tf.nn.relu(xb1 - x) +tf.nn.relu(x - xb2)  # Distance to PML boundary 
        lz = tf.nn.relu(zb1 - z) +tf.nn.relu(z - zb2 )
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, z])
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]
            u = u_model(xz)  # Model predictions: delta u (real and imaginary)
            u_real, u_imag = tf.split(u, num_or_size_splits=2, axis=-1)

        # First derivatives
        u_x_real, u_z_real = tape2.gradient(u_real, [x, z])
        u_x_imag, u_z_imag = tape2.gradient(u_imag, [x, z])
        
    # Compute the second derivatives for real and imaginary parts
    u_xx_real = tape1.gradient(u_x_real, x)  # Second derivative w.r.t x
    u_zz_real = tape1.gradient(u_z_real, z)  # Second derivative w.r.t z
    u_xx_imag = tape1.gradient(u_x_imag, x)  # Second derivative w.r.t x (imaginary)
    u_zz_imag = tape1.gradient(u_z_imag, z)  # Second derivative w.r.t z (imaginary)
    lx_x=tape1.gradient(lx, x)  
    lz_z=tape1.gradient(lz, z) 

    # Definitions of parameters
    Cxr = (1 + c**2 * lx**2 * lz**2) / (1 + c**2 * lx**4)
    Cxi = (c * (lx**2 - lz**2)) / (1 + c**2 * lx**4)
    Czr = (1 + c**2 * lx**2 * lz**2) / (1 + c**2 * lz**4)
    Czi = (c * (-lx**2 + lz**2)) / (1 + c**2 * lz**4)

    # Definitions of derivatives of above parameters
    dCxr = (2 * c**2 * lx * (-2 * lx**2 + lz**2 - c**2 * lx**4 * lz**2) * lx_x) / (1 + c**2 * lx**4)**2
    dCxi = (2 * c * lx * (1 - c**2 * lx**2 * (lx**2 - 2 * lz**2)) * lx_x) / (1 + c**2 * lx**4)**2
    dCzr = -(2 * c**2 * (2 * lz**3 + lx**2 * lz * (-1 + c**2 * lz**4)) * lz_z) / (1 + c**2 * lz**4)**2
    dCzi = (2 * c * lz * (1 - c**2 * lz**2 * (-2 * lx**2 + lz**2)) * lz_z) / (1 + c**2 * lz**4)**2

    Fr_xx=dCxr*u_x_real + Cxr*u_xx_real - dCxi*u_x_imag - Cxi*u_xx_imag
    Fr_zz=dCzr*u_z_real + Czr*u_zz_real - dCzi*u_z_imag - Czi*u_zz_imag
    Fi_xx=dCxr*u_x_imag + Cxr*u_xx_imag + dCxi*u_x_real + Cxi*u_xx_real
    Fi_zz=dCzr*u_z_imag + Czr*u_zz_imag + dCzi*u_z_real + Czi*u_zz_real
   
    del tape1, tape2

    # Extract U0 components
    U0_real, U0_imag = tf.split(U0, num_or_size_splits=2, axis=-1)
    U0_decay_factor = tf.math.exp(-omega/v0 * c * (tf.sqrt(lx**2+lz**2)**3) / 3)
    # Define auxiliary terms for Fr and Fi
    omega2 = omega**2
    m=v**-2
    m0=v0**-2
    term1_real = (1 - c**2 * lx**2 * lz**2) * omega2 * (m * u_real + (m - m0) * U0_real*U0_decay_factor)#*(1-lx)**2*(1-lz)**2
    term2_real = c * (lx**2 + lz**2)        * omega2 * (m * u_imag + (m - m0) * U0_imag*U0_decay_factor)
    
    term1_imag = (1 - c**2 * lx**2 * lz**2) * omega2 * (m * u_imag + (m - m0) * U0_imag*U0_decay_factor)
    term2_imag = -c * (lx**2 + lz**2)       * omega2 * (m * u_real + (m - m0) * U0_real*U0_decay_factor)

    # Fr and Fi terms
    Fr = (Fr_xx + Fr_zz + term1_real + term2_real)
    Fi = (Fi_xx + Fi_zz + term1_imag + term2_imag)

    # Compute losses
    loss_Fr = tf.reduce_mean(tf.square(Fr))
    loss_Fi = tf.reduce_mean(tf.square(Fi))

    # Total loss
    total_loss =  (loss_Fr + loss_Fi)

    if use_source_reg:# Regularization loss term: Penalize the scattered wavefield near the source, scaled by `factor_d`
        # Compute scattered wavefield at regularization points
        xz_reg, factor_d,_=source_reg
        u_reg = u_model(xz_reg)

        reg_loss = tf.reduce_mean( factor_d * tf.square(u_reg))
        total_loss = total_loss+reg_loss
    else:
        reg_loss=0
    if cal_error:    
        error=tf.reduce_mean(tf.abs(dU_2d_val-u))
    else:
        error=0.
    return total_loss,reg_loss,error

# Training step
@tf.function()
def train_step(u_model, U0,U0s, xz,s_xz, v, v0, omega, U0_val, xz_val, v_val,compute_validation=True,
               use_source_reg=False,source_reg=None,source_reg_val=None,cal_error=False,dU_2d_val=0.):
    reg_loss=0.
    with tf.GradientTape() as tape:
        Loss,reg_loss,_ = make_loss  (u_model, U0, xz, v, v0, omega, use_source_reg,source_reg,cal_error=False)

    if compute_validation:
        Loss_valid,_,error = make_loss (u_model, U0_val, xz_val, v_val, v0, omega, use_source_reg,source_reg_val,cal_error,dU_2d_val)
        
    else:
        Loss_valid =0.
        error=0.
    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    
    return Loss,Loss_valid,reg_loss,error

# Training step
@tf.function()
def train_step_PML(u_model, U0, xz,s_xz, v, v0, omega,  a0, omega0, L_PML,xz_PML,U0_val, xz_val, v_val,compute_validation=True,
               use_source_reg=False,source_reg=None,source_reg_val=None,cal_error=False,dU_2d_val=0.):
    reg_loss=0.
    with tf.GradientTape() as tape:
        Loss,reg_loss,_ = make_loss_PML  (u_model, U0, xz, v, v0, omega,  a0, omega0, L_PML,xz_PML, use_source_reg, source_reg, cal_error=False)

    if compute_validation:
        Loss_valid,_,error = make_loss_PML(u_model, U0_val, xz_val, v_val, v0, omega,  a0, omega0, L_PML,xz_PML, use_source_reg,source_reg_val,cal_error,dU_2d_val)
    else:
        Loss_valid =0.
        error=0.
    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    
    return Loss,Loss_valid,reg_loss,error

# Define the optimizer
if use_lr_decay:
    initial_learning_rate=learning_rate
    decay_steps=10000
    decay_rate=0.9
    final_learning_rate = initial_learning_rate * (decay_rate ** (num_epochs / decay_steps))
    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate)
    
if not use_PML:
    a0=0
if not use_source_reg:
    beta=0
#define parameters in history
parameters = {
    "model_type": model_type,
    "velocity_model":velocity_model,
    "use_source_reg":use_source_reg,
    "use_PML":use_PML,
    "a0":a0,
    "omega": float(omega),
    "v0": v0,
    "neurons": neurons,
    "neurons_final": neurons_final,
    "activation": activation,
    "activation_penultima": activation_penultima,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "domain_bounds":domain_bounds,
    "Source_xz": list(s_xz.numpy()),
    "npts_x": npts_x,
    "npts_z": npts_z,
    "seed":seed,
    "beta":beta}   

optimizer =keras.optimizers.Adam(learning_rate=learning_rate)

u_model, u_bases = make_u_model(neurons,neurons_final=neurons_final,activation=activation,activation_penultima=activation_penultima,
                            trainableLastLayer=True,v0=v0,omega=omega,model_type=model_type,seed=seed)

u_model.compile(optimizer=optimizer,loss = make_loss)
u_model.summary()

#%% TRAINING LOOP <<<<<<<<<<<<<<<<<<<
#!!!!
# Start the timer
start_time = time.time()
epoch_time=start_time

Loss= []
Loss_val = []
Error_val = []

U0s=compute_U0(s_xz, s_xz, v0, omega,factor) 

U0_all = compute_U0(xz_all, s_xz, v0, omega,factor)#calculate when chaning the colocation points
U0_val = compute_U0(xz_val, s_xz, v0, omega,factor)

source_reg_val=create_xz_reg(xz_val, s_xz,omega, v0,beta)
_,_,indices_around_source=create_xz_reg(xz_all,s_xz,omega, v0,beta,num_reg_points=npts//100)#adding 1 percent of the all collocation points around the source for stability

#training loop:
for epoch in range(num_epochs):
    cal_error=False
    compute_validation=False
    if epoch%100==0:
        cal_error=True
        compute_validation=True
    rng = np.random.default_rng(seed=epoch)  # Set the seed here
    random_indices = np.sort(rng.choice(n_all, npts, replace=False))
    random_indices=np.concatenate((random_indices,indices_around_source))
    xz_train = tf.gather(xz_all, random_indices)  # Gather corresponding indices
    v_train = tf.gather(v_all, random_indices)  # Gather corresponding v
    U0_train = tf.gather(U0_all, random_indices)  # Gather corresponding U0 
    if use_source_reg:
        source_reg=create_xz_reg(xz_train, s_xz,omega, v0,beta)
    else:
        source_reg=[]
                     
    if use_PML:
        loss_train,loss_val,reg_loss,error_val =  train_step_PML(u_model, U0_train, xz_train,s_xz, v_train, v0, omega,  a0, omega0, L_PML,xz_PML,U0_val, xz_val, v_val,compute_validation=compute_validation,
                                                 use_source_reg=use_source_reg,source_reg=source_reg,source_reg_val=source_reg_val,cal_error=cal_error,dU_2d_val=dU_2d)
    else:
        loss_train,loss_val,reg_loss,error_val =  train_step    (u_model, U0_train,U0s, xz_train,s_xz, v_train, v0, omega,                       U0_val, xz_val, v_val,compute_validation=compute_validation,
                                             use_source_reg=use_source_reg,source_reg=source_reg,source_reg_val=source_reg_val,cal_error=cal_error,dU_2d_val=dU_2d)

    Loss.append(loss_train)
    if cal_error:
        Error_val.append(error_val)
    if compute_validation:
        Loss_val.append(loss_val)
        
    # Check if loss_train is NaN
    if tf.math.is_nan(loss_train):
        print(f"Stopping training due to NaN loss at epoch {epoch}")
        break

    if epoch%100==0:
        print(" Epoch %d of %d" % (int(epoch), int(num_epochs)), end='\n')
        print(" Loss: %.4f, Validation: %.4f, Reg: %.4f, Val Error: %.4f,  Time taken: %.1fs" 
              % (float(loss_train),float(loss_val),float(reg_loss),float(error_val),time.time()-epoch_time),end='\n')
        epoch_time=time.time()
    if (epoch == 0 or epoch == 100 or epoch == 500 or (epoch % 5000 == 0)):
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        formatted_time = f'{int(minutes)} min {seconds:.0f} sec'
        #saving during training
        save_model_and_history(epoch, formatted_time, u_model, Loss, Loss_val,Error_val,parameters)

# End the timer
end_time = time.time()
elapsed_time = end_time - start_time
# Convert elapsed time to minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)
formatted_time = f'{int(minutes)} min {seconds:.0f} sec'
print(f'Training time: {formatted_time}')

#% final saving<<<<<<<<<<<<<<<
save_model_and_history(epoch, formatted_time, u_model, Loss, Loss_val,Error_val,parameters)

