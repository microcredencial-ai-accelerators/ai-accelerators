# %%
from importlib import import_module
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# %%
# Select model
model_type = 'fc'

# %%
# Define the path where the model is saved
SAVEMODEL_PATH = Path(f'./saved_model/mnist_{model_type}_int8_ptq')
OUTPUT_PATH=Path(f'./weights/{model_type}_int8')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load the model
interpreter = tf.lite.Interpreter(model_path=str(SAVEMODEL_PATH /"model_int8.tflite"))
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_details = interpreter.get_tensor_details()

print(f"Loaded model: {SAVEMODEL_PATH.name}")
print(f"Total tensors: {len(model_details)}")

for t in model_details:
    name = t['name']
    shape = t['shape']
    dtype = t['dtype']
    print(f"{t['index']:3d}: {name:40s} {shape} {dtype}")

# %%
# Helper function to save dense layer weights

def save_dense(W, b, prefix, output_path):
    W_rm = W.astype('int8').T.copy()
    b32  = b.astype('int32').copy()
    W_rm.tofile(output_path / f'{prefix}_W.bin')
    b32.tofile(output_path / f'{prefix}_b.bin')

# %%
# Extract tensors
def get_tensor(idx):
    return interpreter.get_tensor(idx)

# Map layers
W0, b0 = get_tensor(7), get_tensor(6)
W1, b1 = get_tensor(5), get_tensor(4)
W2, b2 = get_tensor(3), get_tensor(2)

# %% 

# %%
# Save weights
save_dense(W0, b0, 'fc0', OUTPUT_PATH)
save_dense(W1, b1, 'fc1', OUTPUT_PATH)
save_dense(W2, b2, 'fc2', OUTPUT_PATH)