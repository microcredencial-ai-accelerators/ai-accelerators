# %%
from importlib import import_module
from pathlib import Path
import sys
import tensorflow as tf
import numpy as np
import time

from data import read_data, read_labels, normalize_img
from models.fc import build_fc_model
from models.cnn import build_cnn_model
# %%
# Read MNIST database
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

# %%
# Peprocessing (Normalization)
print('Raw data pixel value range:', train_data.min(), 'to', train_data.max())
train_data, train_labels = normalize_img(train_data, train_labels)
test_data, test_labels = normalize_img(test_data, test_labels)

print('Normalized datatye: ', type(train_data))
print('Normalized data pixel value range:', train_data.numpy().min(), 'to', train_data.numpy().max())

# %%
# Define model
model_type = 'fc'  # 'fc' or 'cnn'

if model_type == 'fc':
    model = build_fc_model()
    # x_train_input, x_test_input = x_train_fc, x_test_fc
elif model_type == 'cnn':
    model = build_cnn_model()
    # x_train_input, x_test_input = x_train_cnn, x_test_cnn
else:
    print(f'{model_type} not supported')

# %%
# Training parameters
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']
EPOCHS = 5

model.compile(optimizer=OPTIMIZER,
              loss=LOSS_FUNCTION,
              metrics=METRICS)
model.summary()

# %%
# Train model
model.fit(train_data, train_labels, epochs=EPOCHS, validation_data=(test_data, test_labels))

# %%
# Evaluate model
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# %%
# Get predictions
start_time = time.time()
predictions = model.predict(test_data)
elapsed_time = time.time() - start_time
print(f"Time per inferecne: {elapsed_time/len(test_data)*1000:.4f} ms")

# %%
# Display prediction i and label
index = 1000
import matplotlib.pyplot as plt
plt.imshow(test_data[index].numpy().squeeze(), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[index])}, Label: {test_labels[index]}")
plt.axis('off')
plt.tight_layout()
plt.show()
# plt.savefig('pred_1000.png')

# %%
# Save model
OUTPUT_PATH = Path(f'saved_model/mnist_{model_type}')
model.save(OUTPUT_PATH / 'SavedModel')
model.save( OUTPUT_PATH / 'model.h5', save_format='h5')
