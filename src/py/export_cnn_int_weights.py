# %%
from importlib import import_module
from pathlib import Path
import tensorflow as tf
import numpy as np
import json

# %%
# Select model
model_type = 'cnn'

# %%
# Paths
SAVEMODEL_PATH = Path(f'./saved_model/mnist_{model_type}_int8_ptq')
OUTPUT_PATH = Path(f'./weights/{model_type}_int')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=str(SAVEMODEL_PATH / "model_int8.tflite"))
interpreter.allocate_tensors()

# Get tensor details
model_details = interpreter.get_tensor_details()

print(f"Loaded model: {SAVEMODEL_PATH.name}")
print(f"Total tensors: {len(model_details)}")

for t in model_details:
    print(f"{t['index']:3d}: {t['name']:50s} {t['shape']} {t['dtype']}")

# %%
# Extract quant parameters
def get_qparams(t):
    q = t["quantization_parameters"]
    scale = float(q["scales"][0]) if len(q["scales"]) else 1.0
    zp    = int(q["zero_points"][0]) if len(q["zero_points"]) else 0
    return scale, zp

# Save binary tensor
def save_tensor(name, array):
    array.tofile(OUTPUT_PATH / f"{name}.bin")
    print(f"[OK] Saved {name}.bin   shape={array.shape}  dtype={array.dtype}")

# %%
quant_info = {}

# CNN layer storage
conv0_W = conv0_b = None
fc0_W = fc0_b = None
fc1_W = fc1_b = None

# %%
# Extract tensors from the model
for t in model_details:
    name = t["name"]
    idx  = t["index"]
    try:
        arr = interpreter.get_tensor(idx)
    except ValueError:
        # This tensor has no data (intermediate / unused)
        continue

    # Save quant params
    scale, zp = get_qparams(t)
    quant_info[name] = {
        "scale": scale,
        "zero_point": zp,
        "shape": arr.shape,
        "dtype": str(arr.dtype)
    }

    # ---- Conv2D weights: [16, 3, 3, 1] ----
    if arr.dtype == np.int8 and arr.ndim == 4:
        conv0_W = arr.copy()
        save_tensor("conv0_W", conv0_W)

    # ---- Conv2D bias ----
    if "conv2d/BiasAdd" in name and arr.dtype == np.int32:
        conv0_b = arr.copy()
        save_tensor("conv0_b", conv0_b)

    # ---- FC0: [16, 2704] ----
    if arr.dtype == np.int8 and arr.shape == (16, 2704):
        fc0_W = arr.copy()
        save_tensor("fc0_W", fc0_W)

    if arr.dtype == np.int32 and arr.shape == (16,) and "dense/BiasAdd" in name:
        fc0_b = arr.copy()
        save_tensor("fc0_b", fc0_b)

    # ---- FC1: [10, 16] ----
    if arr.dtype == np.int8 and arr.shape == (10, 16):
        fc1_W = arr.copy()
        save_tensor("fc1_W", fc1_W)

    if arr.dtype == np.int32 and arr.shape == (10,) and "dense_1/BiasAdd" in name:
        fc1_b = arr.copy()
        save_tensor("fc1_b", fc1_b)

# %%
# Save quantization info as JSON
with open(OUTPUT_PATH / "quant_params.json", "w") as f:
    json.dump(quant_info, f, indent=4)

print("\n[OK] Saved quant_params.json")

# %%
# ---------- EXPORT C HEADER WITH SCALES & ZERO POINTS ----------
def export_quant_header(quant_info, output_path):
    header_path = output_path / f"quant_params_{model_type}_int.h"

    with open(header_path, "w") as h:
        h.write("#ifndef QUANT_PARAMS_H_\n")
        h.write("#define QUANT_PARAMS_H_\n\n")

        h.write("// Auto-generated quantization parameters\n\n")

        for name, q in quant_info.items():
            cname = name.replace("/", "_").replace(";", "_").replace(":", "_")

            h.write(f"// Tensor: {name}\n")
            h.write(f"static const float {cname}_scale = {q['scale']:.9f}f;\n")
            h.write(f"static const int   {cname}_zp    = {q['zero_point']};\n\n")

        h.write("#endif // QUANT_PARAMS_H_\n")

    print(f"[OK] Exported header: {header_path}")

# %%
export_quant_header(quant_info, OUTPUT_PATH)

print("\n[DONE] All CNN weights + quantization params exported.")

