# Microcredencial AI Accelerators for Reconfigurable Technologies: Introduction to AI development

This document will introduces Artificial Intelligent (AI) development based on Deep Learning (DL) and Neural Networks (NNs). For this purpose the MNIST database of handwritten digits will be employed. 

The MNIST dataset provides a training set of 60,000 handwritten digits and a validation set of 10,000 handwritten digits. The images have size 28 x 28 pixels. Therefore, when using a two-layer perceptron, we need 28 x 28 = 784 input units and 10 output units (representing the 10 different digits).

Requierements: 
- Python3
Packages:
- tensorflow

## MNIST Dataset

MNIST is a subset of a larger set available from NIST:  
[MNIST LeCun (sometimes not available)](http://yann.lecun.com/exdb/mnist/)

Alternatively, you can download it from:
- [GitHub mirror repository](https://github.com/fgnt/mnist)  
- [Kaggle MNIST database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

✅ **Make sure to download the following files:**
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## [A1: Introduction to TensorFlow](01_tensorflow.md)
## [A2: Introduction to MATLAB](02_matlab.md)