# TensorFlow
## [Back to index](index.md)


TensorFlow provides a simple and efficient way to load the MNIST dataset directly in Python, without needing to manually download and parse the raw IDX files.

### Required Python Packages

Make sure you have the following packages installed:

- `tensorflow`
- `tensorflow-datasets` (optional, for other datasets)

You can install them using pip:

```pip install tensorflow tensorflow-datasets```

TensorFlow provides a simple method for Python to use the MNIST dataset


### Reading MNIST dataset from TF
```
import tensorflow as tf
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

```
