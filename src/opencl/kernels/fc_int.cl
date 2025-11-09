// ============================================================================
// Fully-connected 3-layer network (INT8) for: in_dim -> 64 -> 32 -> out_dim
//
// Expected binary layout (row-major by output neuron = [Out, In]):
//   W0: int8  [64, in_dim]      b0: int32 [64]
//   W1: int8  [32, 64]          b1: int32 [32]
//   W2: int8  [out_dim, 32]     b2: int32 [out_dim]
//
// Quantization (per-layer, TFLite-style):
//   - Activations: int8 with per-tensor (scale, zero_point).
//   - Weights:     int8 with per-tensor (scale, zero_point).
//   - Bias:        int32 with scale = s_in * s_w (per layer).
//
// Requantization (per layer, scalar):
//   acc32 = sum_i ((x_q[i] - x_zp) * (w_q[o,i] - w_zp)) + b_q[o]
//   y_q   = y_zp + round( M * acc32 ), where M = (s_in * s_w) / s_out
//   ReLU  in quant domain: y_q = max(y_q, y_zp)
//
// Single-work-item kernel (max_global_work_dim(0)).
// ============================================================================

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

#define H1_DIM 64
#define H2_DIM 32

__kernel __attribute__((max_global_work_dim(0)))
void fc_64x32_infer_int8(
  // --- Layer 0: in_dim -> H1_DIM ---
  __global const char*  restrict W0,   // [H1_DIM, in_dim]  (int8)
  __global const int*   restrict b0,   // [H1_DIM]          (int32)
  const int                      w0_zp,
  const float                    M0,   // (s_in0 * s_w0) / s_out0
  const int                      x0_zp,
  const int                      y0_zp,

  // --- Layer 1: H1_DIM -> H2_DIM ---
  __global const char*  restrict W1,   // [H2_DIM, H1_DIM]
  __global const int*   restrict b1,   // [H2_DIM]
  const int                      w1_zp,
  const float                    M1,   // (s_in1 * s_w1) / s_out1
  const int                      x1_zp,
  const int                      y1_zp,

  // --- Layer 2: H2_DIM -> out_dim ---
  __global const char*  restrict W2,   // [out_dim, H2_DIM]
  __global const int*   restrict b2,   // [out_dim]
  const int                      w2_zp,
  const float                    M2,   // (s_in2 * s_w2) / s_out2
  const int                      x2_zp,
  const int                      y2_zp,

  // IO
  __global const char*  restrict x_in, // [in_dim]  int8
  __global char*        restrict y_out,// [out_dim] int8

  // Sizes
  const int in_dim,
  const int out_dim
){
  // On-chip hidden activations (quantized int8)
  char a1[H1_DIM];
  char a2[H2_DIM];

  // -------------------------
  // Layer 0: y1 = ReLU_q( Requant( W0*x + b0 ) )
  // -------------------------
  for (int o = 0; o < H1_DIM; ++o) {
    int acc = b0[o]; // bias in int32 (scale = s_in0 * s_w0)
    #pragma unroll 1
    for (int i = 0; i < in_dim; ++i) {
      const int xqi = ((int)x_in[i]) - x0_zp;                 // (int8 - zp)
      const int wqi = ((int)W0[o * in_dim + i]) - w0_zp;      // (int8 - zp)
      acc += xqi * wqi; // int32 MAC
    }
    // scalar requant + ReLU at zero in quant domain
    int yq = y0_zp + (int)rint(M0 * (float)acc);
    if (yq < y0_zp) yq = y0_zp;
    a1[o] = (char)clamp_int(yq, -128, 127);
  }

  // -------------------------
  // Layer 1: y2 = ReLU_q( Requant( W1*y1 + b1 ) )
  // -------------------------
  for (int o = 0; o < H2_DIM; ++o) {
    int acc = b1[o];
    #pragma unroll 1
    for (int i = 0; i < H1_DIM; ++i) {
      const int xqi = ((int)a1[i]) - x1_zp;
      const int wqi = ((int)W1[o * H1_DIM + i]) - w1_zp;
      acc += xqi * wqi;
    }
    int yq = y1_zp + (int)rint(M1 * (float)acc);
    if (yq < y1_zp) yq = y1_zp;
    a2[o] = (char)clamp_int(yq, -128, 127);
  }

  // -------------------------
  // Output layer: logits_q = Requant( W2*y2 + b2 )
  // (No ReLU; host will argmax on int8 logits)
  // -------------------------
  for (int o = 0; o < out_dim; ++o) {
    int acc = b2[o];
    #pragma unroll 1
    for (int i = 0; i < H2_DIM; ++i) {
      const int xqi = ((int)a2[i]) - x2_zp;
      const int wqi = ((int)W2[o * H2_DIM + i]) - w2_zp;
      acc += xqi * wqi;
    }
    int yq = y2_zp + (int)rint(M2 * (float)acc);
    y_out[o] = (char)clamp_int(yq, -128, 127);
  }
}