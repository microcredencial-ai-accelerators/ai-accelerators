# %%
from importlib import import_module
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# %%
# Select model
model_type = 'fc'  # Cambia a 'cnn' para usar la red convolucional

# %%
# Define the path where the model is saved
SAVEMODEL_PATH = Path(f'./saved_model/mnist_{model_type}') 

# Load the model
model = load_model(SAVEMODEL_PATH / 'model.h5')

model.summary()

# %%
# Get weights

W0,b0 = model.layers[1].get_weights()   # Dense(64)
W1,b1 = model.layers[3].get_weights()   # Dense(32)
W2,b2 = model.layers[5].get_weights()   # Dense(10)

# %%
def save_dense(W, b, prefix, output_path):
    # Keras: W [in_dim, out_dim] → transponer a [out_dim, in_dim]
    W_rm = W.T.astype('float32').copy()
    b32  = b.astype('float32').copy()
    W_rm.tofile(f'{output_path}/{prefix}_W.bin')
    b32.tofile(f'{output_path}/{prefix}_b.bin')

# %%
# Save weights
OUTPUT_PATH=Path(f'./weights/{model_type}_fp32')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
save_dense(W0,b0,'fc0', OUTPUT_PATH)   # 64 x 784 en fichero
save_dense(W1,b1,'fc1', OUTPUT_PATH)   # 32  x 128
save_dense(W2,b2,'fc2', OUTPUT_PATH)   # 10  x 32


