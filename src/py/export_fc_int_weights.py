# %%
from importlib import import_module
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
# %%
# Select model
model_type = 'fc'

# %%
# Define the path where the model is saved
SAVEMODEL_PATH = Path(f'./saved_model/mnist_{model_type}_int8_ptq')
OUTPUT_PATH=Path(f'./weights/{model_type}_int')
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
    W_rm = W.astype('int8').copy()
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

# %% Quatization info
quant_info = {}

for t in model_details:
    name = t["name"]
    qparams = t["quantization_parameters"]
    scales = qparams["scales"]
    zero_points = qparams["zero_points"]

    if len(scales) > 0:  # tensor is quantized
        quant_info[name] = {
            "scales": [float(s) for s in scales],
            "zero_points": [int(z) for z in zero_points],
            "dtype": str(t["dtype"]),
            "shape": t["shape"].tolist(),
        }
for k, v in quant_info.items():
    print(f"{k:50s} scale={v['scales']} zp={v['zero_points']} dtype={v['dtype']}")

# %%
# Save weights
save_dense(W0, b0, 'fc0', OUTPUT_PATH)
save_dense(W1, b1, 'fc1', OUTPUT_PATH)
save_dense(W2, b2, 'fc2', OUTPUT_PATH)
with open(OUTPUT_PATH / "quant_params.json", "w") as f:
    json.dump(quant_info, f, indent=4)


# Function to sanitize tensor names for C macros
def sanitize_name(name):
    name = name.replace("/", "_").replace("-", "_").replace(";", "_").replace(":", "_")
    return name.upper()
# Open header for writing
with open(OUTPUT_PATH / f"quant_params_{model_type}_int.h", "w") as f:
    f.write("// Auto-generated quantization parameters\n")
    f.write("#pragma once\n\n")

    for tname, q in quant_info.items():
        base = sanitize_name(tname)

        # Only take first scale / zero_point if tensor is per-tensor quantized
        scale = q["scales"][0] if len(q["scales"]) > 0 else 1.0
        zp = q["zero_points"][0] if len(q["zero_points"]) > 0 else 0

        f.write(f"#define {base}_SCALE {scale:.8f}f\n")
        f.write(f"#define {base}_ZERO_POINT {zp}\n\n")
