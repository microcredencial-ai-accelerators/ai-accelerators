# TensorFlow
## [Back to index](index.md)

Required Python pakages
Packages:
- tensorflow

``pip install tensorflow tensorflow-datasets``

TensorFlow provides a simple method for Python to use the MNIST dataset

``
from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
``
