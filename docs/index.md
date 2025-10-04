# Microcredencial AI Accelerators for Reconfigurable Technologies: Introduction to AI development

This document will introduces Artificial Intelligent (AI) development based on Deep Learning (DL) and Neural Networks (NNs). For this purpose the MNIST database of handwritten digits will be employed. 

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

### Import all the necessary packages 
```
import numpy as np
import struct
import matplotlib.pyplot as plt
```
### Reading MNIST dataset
```
# Function to read MNIST data
def read_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

# Function to read MNIST labels
def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Read files
train_data = read_images("data/train-images.idx3-ubyte")
train_labels = read_labels("data/train-labels.idx1-ubyte")
test_data = read_images("data/t10k-images.idx3-ubyte")
test_labels = read_labels("data/t10k-labels.idx1-ubyte")
```

### Display relevant dataset information
```
print('Train data length: ', len(train_data), 'images')
print('Train labels length: ', len(train_labels), 'labels')
print('Test data length: ', len(test_data), 'images')
print('Test labels length: ', len(test_labels), 'labels')
print('Data format: ', train_data[0].shape )
print('Train data shape:', train_data.shape)
print('Test data shape:', test_data.shape)
print('First label: ', train_labels[0])
print('Pixel value range:', train_data.min(), 'to', train_data.max())
print('Unique labels:', np.unique(train_labels))
```

### Display 10 MNIST training images with their labels
```
# Show 10 MNIST training images with their labels
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_data[i], cmap='gray')
    ax.set_title(f"Label: {train_labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```
![MNIST Example](./assets/mnist_10_examples.png)


## [A1: Introduction to TensorFlow](01_tensorflow.md)
## [A2: Introduction to MATLAB](02_matlab.md)