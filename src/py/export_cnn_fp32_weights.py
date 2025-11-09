# %%
from importlib import import_module
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# %%
# Select model
model_type = 'cnn'  # Cambia a 'cnn' para usar la red convolucional

# %%
# Define the path where the model is saved
SAVEMODEL_PATH = Path(f'./saved_model/mnist_{model_type}') 

# Load the model
model = load_model(SAVEMODEL_PATH / 'model.h5')

model.summary()

# %%
# Get weights

# Conv: Keras [Kh, Kw, Cin, Cout] -> necesitamos [Cout, Cin, Kh, Kw]
Wc, bc = model.layers[0].get_weights()     # Conv2D
Wc_o = np.transpose(Wc, (3,2,0,1)).astype('float32').copy()  # [16,1,3,3]
bc_o = bc.astype('float32').copy()

# Dense1 (2704->16): Keras [In, Out] -> necesitamos [Out, In]
W1, b1 = model.layers[4].get_weights()
W1_o = W1.T.astype('float32').copy()   # [16,2704]
b1_o = b1.astype('float32').copy()

# Dense2 (16->10): Keras [In, Out] -> [Out, In]
W2, b2 = model.layers[6].get_weights()
W2_o = W2.T.astype('float32').copy()   # [10,16]
b2_o = b2.astype('float32').copy()

# %%
# Save weights
OUTPUT_PATH=Path(f'./weights/{model_type}_fp32')
print(OUTPUT_PATH)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
Wc_o.tofile(OUTPUT_PATH / 'conv0_W.bin'); bc_o.tofile(OUTPUT_PATH / 'conv0_b.bin')
W1_o.tofile(OUTPUT_PATH / 'fc1_W.bin');   b1_o.tofile(OUTPUT_PATH / 'fc1_b.bin')
W2_o.tofile(OUTPUT_PATH / 'fc2_W.bin');   b2_o.tofile(OUTPUT_PATH / 'fc2_b.bin')


