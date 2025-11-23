# Pruning Model Tutorial with TensorFlow
## [Back to Module 2](module2-aidev.md)

## Introduction
This tutorial demonstrates how to apply weight pruning to a neural network trained on the MNIST dataset using TensorFlow and the TensorFlow Model Optimization Toolkit (TF-MOT). The goal is to reduce the model size and increase sparsity while maintaining acceptable accuracy.

---

## 1. Import Dependencies

```python
import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model

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

## 4. Define the Pruning Strategy
We use polynomial decay pruning, gradually increasing sparsity from 0% to 50% throughout training.
```python
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,   # 50% of weights pruned
        begin_step=0,
        end_step=1000
    )
}

```
## 5. Apply Pruning to the Model
The pruning wrapper is applied to the model using:
```python
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model, **pruning_params)
```
Make sure model is already created before this step.

## 6. Compile and Train the Pruned Model

```python
pruned_model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS_FUNCTION,
    metrics=METRICS
)

pruned_model.summary()
```
Add the pruning callback and begin training:
```python
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

history = pruned_model.fit(
    train_data, 
    train_labels, 
    epochs=EPOCHS, 
    validation_data=(test_data, test_labels), 
    callbacks=callbacks
)

```
## 7. Evaluate the Model
```python
test_loss, test_acc = pruned_model.evaluate(test_data, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
```

## 8. Save the Pruned Model
The pruned model is saved in both TensorFlow SavedModel and HDF5 formats:
```python
OUTPUT_PATH = Path(f'./saved_model/mnist_{model_type}_pruned')

pruned_model.save(OUTPUT_PATH / 'SavedModel')
pruned_model.save(OUTPUT_PATH / 'model.h5', save_format='h5')
```
## 9. Compare Model Sizes
```python
def get_gzipped_model_size(file):
    import zipfile
    import tempfile
    import gzip
    import shutil

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file) / 1e6  # in MB

pruned_model_path = Path(f'./saved_model/mnist_{model_type}_pruned')
base_model_path = Path(f'./saved_model/mnist_{model_type}')


print(f"Baseline model size: {os.path.getsize(pruned_model_path / 'model.h5') / 1e6:.2f} MB")
print(f"Pruned model size: {os.path.getsize(base_model_path / 'model.h5') / 1e6:.2f} MB")
```
## 10. Calculate Model Sparsity
The sparsity of a model is calculated as the percentage of zero-valued weights:
```python
def calculate_sparsity(model):
    total = 0
    zeros = 0
    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            total += w.size
            zeros += np.sum(w == 0)
    sparsity = 100.0 * zeros / total
    return sparsity

model = load_model(base_model_path / 'model.h5')
print(f"Baseline sparsity: {calculate_sparsity(model):.2f}%")
print(f"Pruned model sparsity: {calculate_sparsity(pruned_model):.2f}%")
```
## Application to FPGA Deployment
This pruned model is ideal for edge and FPGA-based accelerators. By reducing the number of active parameters, pruning significantly reduces:
- Memory usage
- Compute complexity
- Power consumption
This makes the model more suitable for hardware implementation.
