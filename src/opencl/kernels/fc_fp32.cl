// ============================================================================
// Fully-connected 3-layer network (FP32) for:
//   in_dim  → 64 → 32 → out_dim
//
// This kernel is written as a single-work-item pipeline (max_global_work_dim(0))
// which maps cleanly to hardware on Intel Cyclone V (DE10-Nano) using aoc.
//
// Weight layout expected (row-major by output neuron):
//   W0: [64, in_dim]   (Out, In)
//   W1: [32, 64]       (Out, In)
//   W2: [out_dim, 32]  (Out, In)
// Bias vectors:
//   b0: [64], b1: [32], b2: [out_dim]
//
// NOTE:
// - If you export from Keras, weights are [In, Out] → transpose to [Out, In].
// - Softmax is NOT required for prediction; do argmax on host.
// - Unroll pragmas are set to 1 to keep area low; increase for more parallelism.
// ============================================================================

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Fix the internal buffer sizes to your architecture for predictable fit.
// If you change hidden sizes, update these and recompile.
#define H1_DIM 64
#define H2_DIM 32

__attribute__((max_global_work_dim(0)))
__kernel void fc_64x32_infer_fp32(
    // Layer 1: in_dim → 64
    __global const float* restrict W0,  // [H1_DIM, in_dim]
    __global const float* restrict b0,  // [H1_DIM]

    // Layer 2: 64 → 32
    __global const float* restrict W1,  // [H2_DIM, H1_DIM]
    __global const float* restrict b1,  // [H2_DIM]

    // Layer 3 (output): 32 → out_dim
    __global const float* restrict W2,  // [out_dim, H2_DIM]
    __global const float* restrict b2,  // [out_dim]

    // Input / Output
    __global const float* restrict x_in,   // [in_dim]  (flattened)
    __global float* restrict logits,       // [out_dim] (no softmax)

    // Sizes (kept as runtime args so 'in_dim' and 'out_dim' are flexible)
    const int in_dim,
    const int out_dim
){
    // ---- Hidden activations (local on-chip) ----
    // Sized with compile-time constants to simplify hardware mapping.
    float a1[H1_DIM];
    float a2[H2_DIM];

    // ----------------------------
    // Layer 1: y1 = ReLU(W0 * x + b0)
    // ----------------------------
    for (int o = 0; o < H1_DIM; ++o) {
        float acc = b0[o];

        // Multiply-accumulate over input vector
        #pragma unroll 1 // Increase (e.g., 2/4/8) for more parallelism if resources allow
        for (int i = 0; i < in_dim; ++i) {
            // W0 is stored row-major by output neuron: index = o*in_dim + i
            acc += W0[o * in_dim + i] * x_in[i];
        }

        // ReLU activation
        a1[o] = (acc > 0.0f) ? acc : 0.0f;
    }

    // ----------------------------
    // Layer 2: y2 = ReLU(W1 * y1 + b1)
    // ----------------------------
    for (int o = 0; o < H2_DIM; ++o) {
        float acc = b1[o];

        // MAC across the previous hidden layer
        #pragma unroll 1
        for (int i = 0; i < H1_DIM; ++i) {
            // W1 index = o*H1_DIM + i  (row-major by output neuron)
            acc += W1[o * H1_DIM + i] * a1[i];
        }

        // ReLU
        a2[o] = (acc > 0.0f) ? acc : 0.0f;
    }

    // --------------------------------------
    // Output layer: logits = W2 * y2 + b2
    // --------------------------------------
    for (int o = 0; o < out_dim; ++o) {
        float acc = b2[o];

        #pragma unroll 1
        for (int i = 0; i < H2_DIM; ++i) {
            // W2 index = o*H2_DIM + i
            acc += W2[o * H2_DIM + i] * a2[i];
        }

        logits[o] = acc;  // Host will do argmax (and optional softmax if you need probabilities)
    }
}