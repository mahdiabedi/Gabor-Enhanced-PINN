# Gabor-Enhanced-PINN  
## Gabor-Enhanced Physics-Informed Neural Networks (PINNs) for Fast Simulations of Acoustic Wavefields  

This repository contains the implementation of **Gabor-Enhanced Physics-Informed Neural Networks (PINNs)** for solving the Helmholtz equation efficiently, as presented in the paper:

> ["Gabor-Enhanced Physics-Informed Neural Networks for Fast Simulations of Acoustic Wavefields"](https://arxiv.org/abs/2502.17134)  

---

## Overview  

Traditional PINNs converge slowly for wavefield simulations due to their low-frequency bias. This work introduces an improved PINN framework that integrates **explicit Gabor functions** into the network architecture, significantly enhancing convergence speed and accuracy for solving the **Scattered Helmholtz equation**. The framework is designed to work with realistic velocity models and includes a **Perfectly Matched Layer (PML)**.

The implementation supports training and validation using various benchmark models and provides pretrained models and visualization tools for streamlined experimentation and evaluation.

---

## Features  

- ✅ Implements both **Gabor-enhanced PINN** and **standard PINN** for comparison  
- ✅ Supports multiple benchmark **velocity models**, including:
  - Simple layered model  
  - Marmousi  
  - Overthrust  
- ✅ Includes **PML** to prevent wave reflections at domain boundaries
- ✅ Includes **Positional encoding** to better capture oscilatory wavefields 
- ✅ Uses an **exponentially decaying learning rate** for improved convergence  
- ✅ Provides **training and validation datasets** for all test models  
- ✅ Includes **pretrained models** saved at multiple epochs for fast inference and benchmarking  
- ✅ Offers **visualization tools** to plot:
  - Velocity models  
  - Reference wavefields (finite-difference)  
  - Predicted wavefields (PINN outputs)  

---

## Main Scripts  

- `Gabor_enhanced_PINN.py`  
  Main training script to train either Gabor-PINN or standard PINN on any of the supported velocity models.

- `Inference_Plotting.py`  
  Utility script to load pretrained models and plot velocity profiles, reference wavefields, and model predictions for easy comparison.

##Usage

Use Gabor_enhanced_PINN.py to train on your chosen velocity model. You can specify training settings within the script.

Use Inference_Plotting.py to:
Load a pretrained model (either Gabor-PINN or standard PINN)
Visualize the corresponding velocity model
Compare the predicted wavefield to the reference finite-difference solution

---

## Installation  

This code requires **Python 3.8+** and the following dependencies:

```bash
pip install numpy matplotlib scipy tensorflow 
