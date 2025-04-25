# Gabor-Enhanced-PINN
# Gabor-Enhanced Physics-Informed Neural Networks (PINNs) for Fast Simulations of Acoustic Wavefields  

This repository contains the implementation of **Gabor-Enhanced Physics-Informed Neural Networks (PINNs)** for solving the Helmholtz equation efficiently, as presented in the paper:  

> ["Gabor-Enhanced Physics-Informed Neural Networks for Fast Simulations of Acoustic Wavefields"](https://arxiv.org/abs/2502.17134)   

## **Overview**  

Traditional PINNs converge slowly for wavefield simulations due to their low-frequency bias. This work introduces an improved PINN framework that integrates **explicit Gabor functions** into the network structure, significantly enhancing convergence speed and accuracy for solving the **Helmholtz equation**. The implementation supports various **velocity models** and includes an efficient **Perfectly Matched Layer (PML) integration** for improved boundary behavior.  

## **Features**  
✔️ Implements **Gabor-enhanced PINN** and standard PINN for comparison.  
✔️ Supports differeny models.  
✔️ Includes **PML** to prevent wave reflections at domain boundaries.  
✔️ Uses an **exponentially decaying learning rate** for better convergence.  
✔️ Saves training history and model checkpoints.  

## **Installation**  

This code requires **Python 3.8+** and the following dependencies:  

```bash
pip install tensorflow numpy scipy matplotlib keras
