# Microcredencial AI Accelerators for Reconfigurable Technologies
<p align="center">
  <img src="./assets/ai-accelerators.jpeg" alt="Microcredential AI Accelerators for Reconfigurable Technologies" width="20%">
</p>

This document introduces the fundamentals of Artificial Intelligence (AI) development with a focus on Deep Learning (DL) and Neural Networks (NNs). The course provides a practical foundation for understanding how neural networks are trained, evaluated, and deployed on hardware accelerators, particularly reconfigurable technologies such as FPGAs.

Throughout the training sessions, it will be developed and trained neural networks to solve the MNIST handwritten digit recognition problem, evaluate model performance, and explore methods to accelerate inference using dedicated AI hardware. The course will also cover the process of integrating pre-trained models into accelerator platforms and comparing performance metrics across different implementations.

The MNIST dataset serves as an entry point for deep learning experiments. It consists of 60,000 training images and 10,000 validation images of handwritten digits (0–9), each represented as a 28×28 grayscale image. When building a simple two-layer perceptron, the network typically includes 784 input units (one per pixel) and 10 output units corresponding to the digit classes.
<p align="center">
  <img src="./assets/MNIST_dataset_example.png" alt="Microcredential AI Accelerators for Reconfigurable Technologies" width="50%">
</p>

## Learning Objectives

- Understand the principles of neural network design and training.
- Implement, train, and test models using the MNIST dataset.
- Evaluate model accuracy and performance metrics.
- Deploy and infer trained models on hardware accelerators.
- Analyze and compare software versus hardware performance.

## Requierements: 
- Python3
Packages:
- tensorflow

## MNIST Dataset

The MNIST dataset is a subset of a larger dataset provided by the National Institute of Standards and Technology (NIST):  
[MNIST LeCun (sometimes not available)](http://yann.lecun.com/exdb/mnist/)

Alternatively, you can download it from:
- [GitHub mirror repository](https://github.com/fgnt/mnist)  
- [Kaggle MNIST database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

✅ **Make sure to download the following files:**
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`
## [Module 1: AI model definition and traning](module1-aidev.md)
