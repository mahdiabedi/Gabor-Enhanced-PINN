#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:34:03 2024

@author: mabedi
"""
import tensorflow as tf
import numpy as np
from scipy.special import hankel1,hankel2  # Import Bessel and Hankel functions of the first and second kind (order 0)
import matplotlib.pyplot as plt
import scipy.interpolate

import scipy.io

def interpolator(v, domain_bounds, xz, dtype=tf.float32):
    """
    Interpolate 2D array v over given xz locations using linear interpolation.

    Parameters:
    - v: 2D numpy array to interpolate
    - domain_bounds: tuple (a_x, b_x, a_z, b_z)
    - xz: query points of shape [N, 2]
    - dtype: TensorFlow data type

    Returns:
    - v_interpolated: interpolated values as tf.Tensor of shape [N, 1]
    """
    a_x,b_x,a_z,b_z=domain_bounds
    nz,nx=np.shape(v)
    # Create a grid of the original coordinates for v
    x_orig = np.linspace(a_x, b_x, nx)
    z_orig = np.linspace(a_z, b_z, nz)
    X_orig, Z_orig = np.meshgrid(x_orig, z_orig)
    
    # Flatten the grid and v
    points_orig = np.column_stack([X_orig.ravel(), Z_orig.ravel()])
    v_flat = v.ravel()
    
    # Interpolate using scipy's griddata
    v_interpolated = scipy.interpolate.griddata(points_orig, v_flat, xz, method='linear')

    # Convert to TensorFlow tensor 
    v_interpolated = tf.convert_to_tensor(v_interpolated, dtype=dtype)
    v_interpolated = tf.reshape(v_interpolated, (-1, 1))
    return v_interpolated




def load_training_and_validation_data(frequency, velocity_model, dtype=tf.float32, use_PML=False):
    """
    Load validation and training data for Helmholtz or PINN-based simulation.

    Parameters:
    - frequency (int): Frequency in Hz
    - velocity_model (str): One of 'overthrust', 'simple', or 'marmousi'
    - dtype (tf.DType): TensorFlow data type (default: tf.float32)
    - use_PML (bool): Whether to use PML data for training

    Returns:
    - Dictionary with validation and training tensors
    """
    # Load validation data
    if velocity_model == 'overthrust':
        mat_data = scipy.io.loadmat(f'data/FD_results_{frequency}Hz_val_OVE_velocity_v0corrected.mat')
        a_x, b_x = 0.0, 12.5  # Bounds for x
        a_z, b_z = 0.0, 4 # Bounds for z
    elif velocity_model == 'simple':
        mat_data = scipy.io.loadmat(f'data/FD_results_{frequency}Hz_val_velocity.mat')
        a_x, b_x = 0., 2.5  # Bounds for x
        a_z, b_z = 0., 2.5 # Bounds for z
    elif velocity_model == 'marmousi':
        mat_data = scipy.io.loadmat(f'data/FD_results_{frequency}Hz_val_velocity_marmousi.mat')
        a_x, b_x = 0., 3.  # Bounds for x
        a_z, b_z = 0., 2. # Bounds for z
    else:
        raise ValueError(f"Unsupported velocity model: {velocity_model}")

    domain_bounds=a_x,b_x,a_z,b_z

    U0_analytic = tf.concat([
        tf.reshape(tf.math.real(mat_data['U0_analytic']), (-1, 1)),
        tf.reshape(tf.math.imag(mat_data['U0_analytic']), (-1, 1))
    ], axis=1)

    dU_2d_r = interpolator(np.real(mat_data['dU_2d']), domain_bounds, mat_data['xz_val'], dtype)
    dU_2d_i = interpolator(np.imag(mat_data['dU_2d']), domain_bounds, mat_data['xz_val'], dtype)
    dU_2d = tf.concat([dU_2d_r, dU_2d_i], axis=1)

    xz_val = tf.reshape(mat_data['xz_val'], (-1, 2))
    s_x = np.float32(np.squeeze(mat_data['s_x']))
    s_z = np.float32(np.squeeze(mat_data['s_z']))
    s_xz = tf.cast(tf.stack([s_x, s_z], axis=0), dtype=dtype)
    factor = np.float32(np.squeeze(mat_data['factor']))
    v0 = np.float32(np.squeeze(mat_data['v0']))

    v_val = mat_data['v_val']
    npts_z_val, npts_x_val = v_val.shape
    v_val = tf.reshape(v_val, (-1, 1))

    # Load training data
    if velocity_model == 'simple':
        filename = 'data/Simple_random_training_PML.mat' if use_PML else 'data/Simple_random_training.mat'
    elif velocity_model == 'overthrust':
        filename = 'data/Smooth_OVE_random_training_PML.mat' if use_PML else 'data/Smooth_OVE_random_training.mat'
    elif velocity_model == 'marmousi':
        filename = 'data/marmousi_random_training_PML.mat' if use_PML else 'data/marmousi_random_training.mat'

    mat_data = scipy.io.loadmat(filename)
    v_all = mat_data['v_all']
    xz_all = mat_data['xz_all']

    return {
        'U0_analytic': U0_analytic,
        'dU_2d': dU_2d,
        'xz_val': xz_val,
        's_xz': s_xz,
        'factor': factor,
        'v0': v0,
        'v_val': tf.reshape(v_val, (-1, 1)),
        'v_all': v_all,
        'xz_all': xz_all,
        'npts_z_val': npts_z_val,
        'npts_x_val': npts_x_val,
        'domain_bounds':domain_bounds
    }


def load_validation_data(frequency, velocity_model, dtype=tf.float32, use_PML=False):
    """
    Load validation and training data for Helmholtz or PINN-based simulation.

    Parameters:
    - frequency (int): Frequency in Hz
    - velocity_model (str): One of 'overthrust', 'simple', or 'marmousi'
    - dtype (tf.DType): TensorFlow data type (default: tf.float32)
    - use_PML (bool): Whether to use PML data for training

    Returns:
    - Dictionary with validation and training tensors
    """
    # Load validation data
    if velocity_model == 'overthrust':
        mat_data = scipy.io.loadmat(f'data/FD_results_{frequency}Hz_val_OVE_velocity_v0corrected.mat')
        a_x, b_x = 0.0, 12.5  # Bounds for x
        a_z, b_z = 0.0, 4 # Bounds for z
    elif velocity_model == 'simple':
        mat_data = scipy.io.loadmat(f'data/FD_results_{frequency}Hz_val_velocity.mat')
        a_x, b_x = 0., 2.5  # Bounds for x
        a_z, b_z = 0., 2.5 # Bounds for z
    elif velocity_model == 'marmousi':
        mat_data = scipy.io.loadmat(f'data/FD_results_{frequency}Hz_val_velocity_marmousi.mat')
        a_x, b_x = 0., 3.  # Bounds for x
        a_z, b_z = 0., 2. # Bounds for z
    else:
        raise ValueError(f"Unsupported velocity model: {velocity_model}")

    domain_bounds=a_x,b_x,a_z,b_z

    U0_analytic = tf.concat([
        tf.reshape(tf.math.real(mat_data['U0_analytic']), (-1, 1)),
        tf.reshape(tf.math.imag(mat_data['U0_analytic']), (-1, 1))
    ], axis=1)

    dU_2d_r = interpolator(np.real(mat_data['dU_2d']), domain_bounds, mat_data['xz_val'], dtype)
    dU_2d_i = interpolator(np.imag(mat_data['dU_2d']), domain_bounds, mat_data['xz_val'], dtype)
    dU_2d = tf.concat([dU_2d_r, dU_2d_i], axis=1)

    xz_val = tf.reshape(mat_data['xz_val'], (-1, 2))
    s_x = np.float32(np.squeeze(mat_data['s_x']))
    s_z = np.float32(np.squeeze(mat_data['s_z']))
    s_xz = tf.cast(tf.stack([s_x, s_z], axis=0), dtype=dtype)
    factor = np.float32(np.squeeze(mat_data['factor']))
    v0 = np.float32(np.squeeze(mat_data['v0']))

    v_val = mat_data['v_val']
    npts_z_val, npts_x_val = v_val.shape
    v_val = tf.reshape(v_val, (-1, 1))

    return {
        'U0_analytic': U0_analytic,
        'dU_2d': dU_2d,
        'xz_val': xz_val,
        's_xz': s_xz,
        'factor': factor,
        'v0': v0,
        'v_val': tf.reshape(v_val, (-1, 1)),
        'npts_z_val': npts_z_val,
        'npts_x_val': npts_x_val,
        'domain_bounds':domain_bounds
    }


# Define the background wavefield U0 in 2D
def compute_U0(xz, s_xz, v0, omega,factor=1.):
    """
    Compute the background wavefield U0 for the 2D Helmholtz equation.
    
    Args:
    x, z: Tensors of spatial coordinates (same shape).
    sx, sz: Source location (scalars).
    v0: Constant background velocity.
    omega: Angular frequency.
    factor: obtained by matching the analytical and finite-difference magnitudes
    
    Returns:
    U0: The background wavefield.
    """
    x, z = tf.unstack(xz, axis=-1)  # x and z will have shape [batch_size]
    x = tf.reshape(x, (-1, 1))  # Shape [batch_size, 1]
    z = tf.reshape(z, (-1, 1))  # Shape [batch_size, 1]
    sx, sz = tf.unstack(s_xz, axis=-1)
    # Compute the distance between the point and the source
    r = tf.sqrt((x - sx)**2 + (z - sz)**2)
    # print('r:',np.shape(r))
    # Compute the argument for the Hankel function
    # arg = omega * r / v0
    # Avoid division by zero by assigning a specific value when r is zero
    arg = tf.where(r == 0, tf.constant(1e-9, dtype=r.dtype), omega * r / v0)

    # Compute the background wavefield U0
    # U0 = (1j / 4) * hankel_0_second_kind(arg)
    U0 = factor*(1j / 4) *hankel2(0,arg)
    U0_real = tf.math.real(U0)  # Shape: [batch_size, 1]
    U0_imag = tf.math.imag(U0)  # Shape: [batch_size, 1]
    # print('U0:',np.shape(U0))
    U0=tf.concat([U0_real,U0_imag],axis=-1)
    # print('U0 stack real and imaginary:',np.shape(U0))
    return U0


    
def extend2d(V, npmlz, npmlx, Nz, Nx):
    # Initialize the output array Ve with zeros
    Ve = np.zeros((Nz, Nx))
    
    # Copy the central part of V into Ve
    Ve[npmlz:Nz-npmlz, npmlx:Nx-npmlx] = V
    
    # Extrapolate the z-boundaries (top and bottom)
    for ii in range(npmlz):
        Ve[ii, npmlx:Nx-npmlx] = V[0, :]  # Top boundary extrapolation
        Ve[Nz-1-ii, npmlx:Nx-npmlx] = V[-1, :]  # Bottom boundary extrapolation
    
    # Extrapolate the x-boundaries (left and right)
    for ii in range(npmlx):
        Ve[npmlz:Nz-npmlz, ii] = V[:, 0]  # Left boundary extrapolation
        Ve[npmlz:Nz-npmlz, Nx-1-ii] = V[:, -1]  # Right boundary extrapolation
    
    # Extrapolate the four corners
    for ix in range(npmlx):
        for iz in range(npmlz):
            Ve[iz, ix] = V[0, 0]  # Top-left corner
            Ve[iz, Nx-npmlx+ix] = V[0, -1]  # Top-right corner
            Ve[Nz-npmlz+iz, ix] = V[-1, 0]  # Bottom-left corner
            Ve[Nz-npmlz+iz, Nx-npmlx+ix] = V[-1, -1]  # Bottom-right corner
    
    return Ve

# Example usage
# V = np.random.rand(5, 5)  # Example input matrix
# npmlz, npmlx = 2, 2  # Number of boundary layers
# Nz, Nx = 9, 9  # Output matrix size

# Ve = extend2d(V, npmlz, npmlx, Nz, Nx)
# plt.figure()
# plt.imshow(Ve)





def save_model_and_history(epoch, formatted_time, u_model, Loss, Loss_val, Error_val, parameters):
    print('saving...')
    # Update only the training time
    parameters["Training_time"] = formatted_time

    # Dynamically include the epoch number in the model filename
    model_filename = f'Results/Models/u_model_epoch_{epoch}.keras'

    # Save the model and training history
    u_model.save(model_filename)
    history = {
        "training_loss": Loss,
        "validation_loss": Loss_val,
        "validation_error": Error_val,
        "parameters": parameters
    }
    np.save('Results/training_history.npy', history)
    print('\rSaved!       ')
    
#Plot real and imaginary parts of the wavefield
def plot_model_wavefield(wavefield, xz, npts_x, npts_z,domain_bounds):
    a_x,b_x,a_z,b_z=domain_bounds
    # Extract the real and imaginary parts
    u_real = wavefield[:, 0]  # Real part
    u_imag = wavefield[:, 1]  # Imaginary part

    # Reshape the real part wavefield back into a 2D grid
    u_real_grid = tf.reshape(u_real, (npts_z, npts_x))  # Shape [npts_z, npts_x]
    
    # Reshape the imaginary part wavefield back into a 2D grid
    u_imag_grid = tf.reshape(u_imag, (npts_z, npts_x))  # Shape [npts_z, npts_x]

    # Plot the real part as a 2D image
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(u_real_grid, extent=[a_x, b_x, b_z, a_z], origin='upper', cmap='viridis', aspect='auto')
    plt.colorbar(label='Real Part')
    plt.title("Real Part")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

    # Plot the imaginary part as a 2D image
    plt.subplot(122)
    plt.imshow(u_imag_grid, extent=[a_x, b_x,b_z, a_z], origin='upper', cmap='plasma', aspect='auto')
    plt.colorbar(label='Imaginary Part')
    plt.title("Imaginary Part")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    # plt.tight_layout()
    
def plot_collocation_points(xz,color='k'):
    # xz contains (x, z) coordinates
    plt.scatter(xz[:, 0], xz[:, 1], color=color, s=10)
    plt.gca().set_aspect('equal')
    plt.title("2D Collocation Points")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True)
    plt.show()
    
def sin_activation(x):
    return tf.sin(x)


def create_xz_reg(xz, s_xz, omega, v0,beta=1000, num_reg_points=500,max_bound=2.):
    """
    Select the closest 500 points in `xz` to `s_xz` for near source soft constraint.

    Args:
    xz: Tensor of coordinates (shape: [num_points, 2]).
    s_xz: Source location coordinates (shape: [1, 2]).
    omega: Angular frequency.
    v0: Background velocity.
    num_reg_points: Number of closest points to select for regularization (default: 500).

    Returns:
    xz_reg: The 500 closest points to `s_xz` (Tensor).
    factor_d: Regularization factors for the selected points (Tensor).
    """
    # Compute squared distances from each point in xz to the source location s_xz
    distance_squared =( tf.reduce_sum(tf.square(xz - s_xz), axis=-1))
    
    # Get the indices of the closest 500 points
    _, indices = tf.nn.top_k(-distance_squared, k=num_reg_points)  # Use negative to get smallest distances
    
    # Gather the closest points based on the indices
    xz_reg = tf.gather(xz, indices)

    # Define the maximum distance (squared) for regularization (lambda/2)
    max_distance_squared = tf.cast(( (v0 * 1* 3.141592 / omega) ** 2)/max_bound*2,dtype='float32')

    # Calculate the regularization factor for the selected points
    factor_d = tf.expand_dims(tf.gather(tf.nn.relu(max_distance_squared - distance_squared) * beta**2 * omega, indices),axis=-1)#1e6

    return xz_reg, factor_d,indices

