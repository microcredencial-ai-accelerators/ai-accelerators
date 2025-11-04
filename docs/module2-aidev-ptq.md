# Post-Training Quantization (PTQ) Tutorial with TensorFlow Lite
## [Back to Module 2](module2-aidev.md)

This tutorial demonstrates how to perform **Post-Training Quantization (PTQ)** on a trained MNIST model (either fully-connected or convolutional) using TensorFlow Lite.  
The process involves loading a pre-trained FP32 model, defining a representative dataset for calibration, converting the model to an INT8 TFLite version, and evaluating both accuracies.

---

## 1. Import Dependencies

```python
from importlib import import_module
from pathlib import Path
import sys
sys.path.insert(0, "../")
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Reload modules
import importlib
import models.fc
import models.cnn
importlib.reload(models.fc)
importlib.reload(models.cnn)

import data
importlib.reload(data)

from data import read_data, read_labels, normalize_img
from models.fc import build_fc_model
from models.cnn import build_cnn_model

```
## 2. Load and Normalize MNIST Dataset
```python
# Load dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

# Print pixel range before normalization
print('Raw data pixel value range:', train_data.min(), 'to', train_data.max())

# Normalize to [0, 1]
train_data, train_labels = normalize_img(train_data, train_labels)
test_data, test_labels = normalize_img(test_data, test_labels)

print('Normalized data type:', type(train_data))
print('Normalized pixel value range:', train_data.numpy().min(), 'to', train_data.numpy().max())

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
```

## 3. Load the Pre-Trained Model
Select the model type and load the corresponding FP32 Keras model.
```python
# Model type: choose 'fc' or 'cnn'
model_type = 'fc'  # Change to 'cnn' for convolutional network

# Path to saved model
OUTPUT_PATH = Path(f'./../../../saved_model/mnist_{model_type}') 

# Load model
model = load_model(OUTPUT_PATH / 'model.h5')
model.summary()

# Evaluate baseline FP32 accuracy
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"FP32 pretrained model accuracy: {test_acc:.4f}")
```

## 4. Define Representative Dataset Generator
The representative dataset provides a sample of input data for calibrating activations during quantization.
This step ensures better accuracy after converting to INT8.
```python
def representative_data_gen():
    for i in range(100):
        idx = np.random.randint(len(train_data))
        sample = train_data[idx:idx+1]
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()
        sample = sample.astype(np.float32)

        if model_type == 'cnn':
            if sample.ndim == 2:
                sample = np.expand_dims(sample, axis=-1)
            elif sample.ndim == 3 and sample.shape[0] == 1:
                sample = np.expand_dims(sample, axis=-1)
        elif model_type == 'fc':
            if sample.ndim > 2:
                sample = sample.reshape(1, -1)

        yield [sample]
```

## 5. Convert the Model to INT8 TensorFlow Lite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Ensure both input and output are quantized to int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Perform the conversion
tflite_model = converter.convert()

# Save model
OUTPUT_PATH = Path(f'./../../../saved_model/mnist_{model_type}_int8_ptq')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH / "model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```
## 6. Evaluate Quantized Model Accuracy

```python
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_PATH / "model_int8.tflite"))
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(x):
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)

    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        x = x / scale + zero_point
        x = np.clip(x, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    return output_data

# Convert test tensors to NumPy if needed
if isinstance(test_data, tf.Tensor):
    test_data = test_data.numpy()
if isinstance(test_labels, tf.Tensor):
    test_labels = test_labels.numpy()

# Compute accuracy
correct = 0
total = len(test_data)
for i in range(total):
    x = test_data[i:i+1].astype(np.float32)
    y_true = np.argmax(test_labels[i])
    y_pred = np.argmax(tflite_predict(x))
    correct += (y_true == y_pred)

accuracy = correct / total
print(f"Quantized model accuracy: {accuracy:.4f}")
print(f"FP32 pretrained model accuracy: {test_acc:.4f}")

```

Example results
```
Quantized model accuracy: 0.9704
FP32 pretrained model accuracy: 0.9705
```

## Conclusion:
This PTQ workflow shows that quantization can significantly reduce model size and inference latency while maintaining nearly the same accuracy.