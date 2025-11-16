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

def save_bin(name, arr, dtype=None):
    if dtype is not None:
        arr = arr.astype(dtype)
    out = OUTPUT_PATH / f"{name}.bin"
    arr.tofile(out)
    print(f"[OK] {name:10s} shape={arr.shape} dtype={arr.dtype} -> {out}")



# %%
quant_info = {}

# CNN layer storage

def is_int8(a):  return a.dtype == np.int8
def is_int32(a): return a.dtype == np.int32

convW = convB = fc1W = fc1B = fc2W = fc2B = None

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

    
    # -------------------- Conv weights (expect 3x3x1x16) --------------------
    # TFLite typically stores Conv2D weights as [Kh, Kw, Cin, Cout] = [3,3,1,16] (NHWC kernels).
    # We need [Cout, Cin, Kh, Kw] = [16,1,3,3] for the OpenCL kernel.
    if is_int8(arr) and arr.ndim == 4:
        sh = arr.shape
        looks_like_conv_w = (
            ("conv2d/Conv2D" in name or name.endswith("/weights") or "sequential/conv2d/Conv2D" in name)
            and sorted(sh) == [1,3,3,16]  # elements are {3,3,1,16} in some order
        )
        if looks_like_conv_w or sh == (3,3,1,16) or sh == (16,3,3,1) or sh == (1,3,3,16):
            # Map to [Cout, Cin, Kh, Kw]
            if sh == (3,3,1,16):                # [Kh, Kw, Cin, Cout]
                convW = np.transpose(arr, (3,2,0,1))
            elif sh == (16,3,3,1):              # [Cout, Kh, Kw, Cin]
                convW = np.transpose(arr, (0,3,1,2))
            elif sh == (1,3,3,16):              # [Cin, Kh, Kw, Cout]
                convW = np.transpose(arr, (3,0,1,2))
            else:
                # Generic finder: identify axes by sizes {16,1,3,3}
                axes = list(range(4))
                ax_out = next(i for i,s in enumerate(sh) if s==16)
                ax_in  = next(i for i,s in enumerate(sh) if s==1)
                # take the remaining two axes as kh/kw (size=3)
                rem = [i for i in axes if i not in (ax_out, ax_in)]
                ax_kh, ax_kw = rem
                convW = np.transpose(arr, (ax_out, ax_in, ax_kh, ax_kw))
            if convW.size != 16*1*3*3:
                raise RuntimeError(f"conv0_W unexpected size: {convW.shape}")
            save_bin("conv0_W", convW, np.int8)
            continue  # move on; avoid catching this tensor again elsewhere

    # -------------------- Conv bias [16] --------------------
    if is_int32(arr) and arr.shape == (16,) and ("conv2d/BiasAdd" in name or "sequential/conv2d/BiasAdd" in name):
        convB = arr.copy()
        save_bin("conv0_b", convB, np.int32)
        continue

    # -------------------- FC1 weights: want [Out, In] = [16, 2704] --------------------
    if is_int8(arr) and arr.ndim == 2 and arr.size == 16*2704:
        if arr.shape == (2704,16):       # [In,Out] -> transpose
            fc1W = arr.T.copy()
        elif arr.shape == (16,2704):     # [Out,In]
            fc1W = arr.copy()
        else:
            # Make the axis with length 16 be 'Out'
            fc1W = arr if arr.shape[0] == 16 else arr.T.copy()
        save_bin("fc1_W", fc1W, np.int8)
        continue

    if is_int32(arr) and arr.shape == (16,) and ("dense/BiasAdd" in name or "dense_0/BiasAdd" in name):
        fc1B = arr.copy()
        save_bin("fc1_b", fc1B, np.int32)
        continue

    # -------------------- FC2 weights: want [Out, In] = [10, 16] --------------------
    if is_int8(arr) and arr.ndim == 2 and arr.size == 10*16:
        if arr.shape == (16,10):         # [In,Out] -> transpose
            fc2W = arr.T.copy()
        elif arr.shape == (10,16):       # [Out,In]
            fc2W = arr.copy()
        else:
            fc2W = arr if arr.shape[0] == 10 else arr.T.copy()
        save_bin("fc2_W", fc2W, np.int8)
        continue

    if is_int32(arr) and arr.shape == (10,) and ("dense_1/BiasAdd" in name):
        fc2B = arr.copy()
        save_bin("fc2_b", fc2B, np.int32)
        continue

print("\n[SUMMARY]")
for n, v in (("conv0_W", convW), ("conv0_b", convB),
             ("fc1_W",  fc1W),    ("fc1_b",  fc1B),
             ("fc2_W",  fc2W),    ("fc2_b",  fc2B)):
    print(f"{n:10s} ->", None if v is None else (v.shape, v.dtype))


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

