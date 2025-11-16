# Quantization-Aware Training (QAT) Tutorial with TensorFlow Lite
## [Back to Module 2](module2-aidev.md)

## Introdiction
This tutorial demonstrates how to perform **Quantization-Aware Training (QAT)** using the **TensorFlow Model Optimization Toolkit**.  
Unlike Post-Training Quantization (PTQ), QAT simulates quantization during training, allowing the model to adapt to INT8 precision and recover most of the lost accuracy.

---

## 1. Install Dependencies

You need the `tensorflow_model_optimization` module.

```bash
pip install tensorflow-model-optimization
```
## 2. Import Libraries
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import quantize_model
from pathlib import Path
import numpy as np
```

## 3. Load and Prepare Data
We'll use the MNIST dataset, normalizing and one-hot encoding it as before.
```python
# Load dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize data to [0, 1]
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Add channel dimension if needed (for CNNs)
model_type = 'fc'  # Change to 'cnn' if using a convolutional model
if model_type == 'cnn':
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)
else:
    train_data = train_data.reshape((train_data.shape[0], -1))
    test_data = test_data.reshape((test_data.shape[0], -1))

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
```
## 4. Load the Pre-Trained Model
```python
# Path to the trained model
OUTPUT_PATH = Path(f'./../../../saved_model/mnist_{model_type}') 

# Load the original FP32 model
model = load_model(OUTPUT_PATH / 'model.h5')
model.summary()

# Evaluate FP32 model accuracy
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"FP32 pretrained model accuracy: {test_acc:.4f}")

```
## 5. Apply Quantization-Aware Training (QAT)

We now wrap the model with the quantization-aware training API.

```python
# Apply quantization-aware wrapper
quant_aware_model = quantize_model(model)

# Compile and fine-tune (a few epochs are usually enough)
quant_aware_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

# Fine-tune model
quant_aware_model.fit(train_data, train_labels,
                      epochs=3,
                      validation_data=(test_data, test_labels))
```

## 6. Convert to TensorFlow Lite (INT8)

After fine-tuning, convert the model to a quantized INT8 TensorFlow Lite model.
```python
converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save model
OUTPUT_PATH = Path(f'./../../../saved_model/mnist_{model_type}_int8_qat')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH / "model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

7. Evaluate Quantized Model

You can reuse the same evaluation function as in the PTQ tutorial to test the quantized model accuracy using TensorFlow Lite interpreter.

```python
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_PATH / "model_int8.tflite"))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(x):
    if model_type == 'cnn' and x.ndim == 3:
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

# Evaluate quantized model
correct = 0
for i in range(len(test_data)):
    x = test_data[i:i+1].astype(np.float32)
    y_true = np.argmax(test_labels[i])
    y_pred = np.argmax(tflite_predict(x))
    correct += (y_true == y_pred)

qat_acc = correct / len(test_data)
print(f"Quantized model accuracy: {qat_acc:.4f}")
print(f"FP32 pretrained model accuracy: {test_acc:.4f}")
```

Example results:
```
Quantized model accuracy: 0.9739
FP32 pretrained model accuracy: 0.9705
```
## Conclusion:
Quantization-Aware Training (QAT) simulates quantization during training, allowing the model to learn to be robust to quantization effects.
It typically yields higher post-quantization accuracy compared to PTQ, especially for more complex models or when accuracy sensitivity is high.