#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:02:10 2025

@author: mabedi

This code produces the results in the paper titled 
"Gabor-Enhanced Physics-Informed Neural Networks for Fast Simulations of Acoustic Wavefields"
By M.M. Abedi, D. Pardo, T. Alkhalifah


Constum layers we used are positional encoder, Gabor layer, and a 3D fully connected layer 
"""
import tensorflow as tf
import numpy as np
import keras

class EmbedderLayer(tf.keras.layers.Layer):
    def __init__(self, K=4, **kwargs):
        """
        Positional encoder, vectorized version
        
        Args:
            K (int): Number of frequency bands. Frequencies = [2^0, 2^1, ..., 2^(K-1)]
        """
        super(EmbedderLayer, self).__init__(**kwargs)
        self.K = K

    def build(self, input_shape):
        exponents = tf.range(self.K, dtype=tf.float32)  # [0, 1, ..., K-1]
        self.freq_bands = tf.pow(2.0, exponents)  # [1.0, 2.0, ..., 2^(K-1)]
        self.freq_bands = tf.reshape(self.freq_bands, (1, 1, self.K))  # Shape: [1, 1, K]

    def call(self, inputs):
        # inputs: [batch_size, input_dim]
        inputs = tf.expand_dims(inputs, axis=-1)  # [batch_size, input_dim, 1]
        
        scaled_inputs = inputs * self.freq_bands  # [batch_size, input_dim, K]
        scaled_inputs=tf.reshape(tf.transpose(scaled_inputs,perm=[0, 2, 1]), [-1, 2*self.K])

        sin_embed = tf.sin(scaled_inputs)
        cos_embed = tf.cos(scaled_inputs)

        return tf.concat([inputs[:, :, 0], sin_embed, cos_embed], axis=-1)

    def get_config(self):
        config = super(EmbedderLayer, self).get_config()
        config.update({"K": self.K})
        return config
    
    

class CustomLayer3D(keras.layers.Layer):
    # This layer has a weight matrix with shape [neurons_final, 1, 2], where 2 represents the real and imaginary parts
    # No bias
    def __init__(self, neurons_final, dtype=tf.float32, trainable=True,seed=1238, **kwargs):
        super(CustomLayer3D, self).__init__(**kwargs)
        self.dtype_ = dtype
        self.trainable_ = trainable
        self.neurons_final = neurons_final
        self.seed=seed

    def build(self, input_shape):
        # Initialize kernel weights here using GlorotNormal
        self.kernel = self.add_weight(shape=(2,self.neurons_final, 1), 
                                      initializer=keras.initializers.GlorotNormal(seed=self.seed), 
                                      trainable=self.trainable_)
    def call(self, inputs):
        # Perform the einsum operation
        output = tf.einsum('jbi,jik->bjk', inputs, self.kernel)
        # Remove the 2nd dimension to achieve the desired output shape (batch_size, 2)
        output = tf.squeeze(output, axis=2)
        return output
    def get_config(self):
        config = super(CustomLayer3D, self).get_config()
        config.update({
            "neurons_final": self.neurons_final,
            "dtype": self.dtype_,
            "trainable": self.trainable_
        })
        return config



    
class GaborFunctionLayer(keras.layers.Layer):
    def __init__(self, neurons, v, omega, dtype=tf.float32, **kwargs):
        super(GaborFunctionLayer, self).__init__(**kwargs)
        self.neurons = neurons  # Number of neurons
        self.v0 = v  # Fixed input parameter for velocity
        self.omega = omega  # Fixed input parameter for angular frequency
        self.dtype_ = dtype
        
        ##final traible:
        theta_init = tf.ones(self.neurons)*(-np.pi /4)
        self.theta = self.add_weight(
            shape=(self.neurons,), 
            initializer=keras.initializers.Constant(theta_init), 
            trainable=True, 
            dtype=self.dtype_, 
            name="theta")  # Rotation angle

        trainable_v_init = tf.linspace(self.v0, self.v0, self.neurons)
        self.trainable_v = self.add_weight(
            shape=(self.neurons,), 
            initializer=keras.initializers.Constant(trainable_v_init), 
            trainable=True, 
            dtype=self.dtype_, 
            name="trainable_v")  # v
        #NON-trainable:
        self.delta = tf.cast(10,self.dtype_)

    # @tf.function     
    def call(self, inputs):
        #fixed velocity in Gabor:
        d_input = inputs
        v_function = tf.cast(self.trainable_v,self.dtype_)
        d=d_input

        # Split inputs into dx and dz components
        d_split = tf.split(d, num_or_size_splits=2, axis=-1)  # Split into two parts
        dx = d_split[0]  # Shape: [batch_size, neurons]
        dz = d_split[1]  # Shape: [batch_size, neurons]

        # Compute transformed coordinates x_theta' and z_theta' for the Gabor function
        x_theta = dx * tf.cos(self.theta) + dz * tf.sin(self.theta)
        z_theta = -dx * tf.sin(self.theta) + dz * tf.cos(self.theta)

        # Compute D = (x_theta')^2 + (z_theta')^2
        D = tf.square(x_theta) + tf.square(z_theta)

        # # # Compute G_real and G_imag based on the Gabor function equations
        G_real =  tf.expand_dims(tf.cos(self.omega /v_function * x_theta) * tf.exp(-0.5 * D * tf.square(self.delta)), axis=0)
        G_imag =  tf.expand_dims(tf.sin(self.omega /v_function * x_theta) * tf.exp(-0.5 * D * tf.square(self.delta)), axis=0)

        G = tf.concat([G_real, G_imag], axis=0)

        return G
    def get_config(self):
        config = super(GaborFunctionLayer, self).get_config()
        config.update({
            "neurons": self.neurons,
            "v": self.v0,
            "omega": self.omega,
            "dtype": self.dtype_
        })
        return config
