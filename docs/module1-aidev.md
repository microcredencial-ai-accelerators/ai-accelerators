# Microcredencial AI Accelerators for Reconfigurable Technologies: AI model definition and traning
## [Back to index](index.md)

This module introduces the fundamental concepts and practical workflow for developing and training Artificial Intelligence (AI) models based on Deep Learning. The process is demonstrated using both TensorFlow/Keras (Python) and MATLAB, focusing on the creation, training, and evaluation of neural networks for image classification tasks.

The MNIST handwritten digit dataset is used as the reference problem. It provides a clear and standardized environment for testing and comparing different network architectures and training configurations.

Two neural network architectures are implemented and analyzed:

1. Fully Connected Neural Network (FCNN)
Also known as a Multilayer Perceptron (MLP), this architecture connects every neuron in one layer to all neurons in the next. It is suitable for simple classification problems and serves as a foundation for understanding the basic operations of neural networks, including forward propagation, backpropagation, and loss optimization.

2. Convolutional Neural Network (CNN)
A specialized architecture for image processing tasks. It employs convolutional and pooling layers to extract spatial features directly from the input images, typically achieving improved performance and generalization in comparison to fully connected networks.

## Objectives:

- Understand the workflow for developing and training neural networks.
- Implement FCNN and CNN architectures using TensorFlow/Keras and MATLAB.
- Train both models on the MNIST dataset and evaluate their accuracy and loss.
- Compare the performance and efficiency between FCNN and CNN models.

## [A1: Introduction to TensorFlow](module1-aidev-tensorflow.md)
## [A2: Introduction to MATLAB](module1-aidev-matlab.md)
## [A3: Post-Training Quantization (PTQ) Tutorial with TensorFlow Lite](module1-aidev-ptq.md)
## [A4: Quantization-Aware Training (QAT) Tutorial with TensorFlow Lite](module1-aidev-qat.md)