// ============================================================================
// Fully-connected 3-layer network (INT8) for: in_dim -> 64 -> 32 -> out_dim
// INT8 activations & weights; INT32 bias & accumulators; INT-only requant.
// Expected weight layout by default: [Out, In] (row-major by output neuron).
// If your .bin are [In, Out], compile with: -DFC_WEIGHTS_ARE_IN_OUT=1
// ============================================================================

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define H1_DIM 64
#define H2_DIM 32

#ifndef FC_WEIGHTS_ARE_IN_OUT
#define FC_WEIGHTS_ARE_IN_OUT 0
#endif

// ---- Weight index helpers (toggle between [Out,In] and [In,Out]) ----
inline int w_idx_fc0(int o, int i, int in_dim) {
#if FC_WEIGHTS_ARE_IN_OUT
  // [In,Out] with Out=H1_DIM (64)
  return i * H1_DIM + o;
#else
  // [Out,In]
  return o * in_dim + i;
#endif
}
inline int w_idx_fc1(int o, int i) {
#if FC_WEIGHTS_ARE_IN_OUT
  // [In,Out] with Out=H2_DIM (32)
  return i * H2_DIM + o;
#else
  // [Out,In]
  return o * H1_DIM + i;
#endif
}
inline int w_idx_fc2(int o, int i, int out_dim) {
#if FC_WEIGHTS_ARE_IN_OUT
  // [In,Out] with Out=out_dim
  return i * out_dim + o;
#else
  // [Out,In]
  return o * H2_DIM + i;
#endif
}

// ---- INT-only requantization: acc(int32) -> int with fixed-point mult/shift ----
inline int requantize_int32(int acc, int mult, int shift, int zp_out) {
  long prod = (long)acc * (long)mult;          // 64-bit intermediate
  long rq = (shift > 0) ? ((prod + (1L << (shift - 1))) >> shift) : prod; // round-to-nearest
  int yq = zp_out + (int)rq;
  return yq;
}

inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__kernel __attribute__((max_global_work_dim(0)))
void fc_64x32_infer_int8(
  // --- Layer 0: in_dim -> H1_DIM ---
  __global const char*  restrict W0,   // int8 [H1_DIM, in_dim] or [in_dim, H1_DIM]
  __global const int*   restrict b0,   // int32 [H1_DIM]
  const int                      w0_zp,
  const int                      M0_mult,
  const int                      M0_shift,
  const int                      x0_zp,
  const int                      y0_zp,

  // --- Layer 1: H1_DIM -> H2_DIM ---
  __global const char*  restrict W1,   // int8 [H2_DIM, H1_DIM] or [H1_DIM, H2_DIM]
  __global const int*   restrict b1,   // int32 [H2_DIM]
  const int                      w1_zp,
  const int                      M1_mult,
  const int                      M1_shift,
  const int                      x1_zp,
  const int                      y1_zp,

  // --- Layer 2: H2_DIM -> out_dim ---
  __global const char*  restrict W2,   // int8 [out_dim, H2_DIM] or [H2_DIM, out_dim]
  __global const int*   restrict b2,   // int32 [out_dim]
  const int                      w2_zp,
  const int                      M2_mult,
  const int                      M2_shift,
  const int                      x2_zp,
  const int                      y2_zp,

  // IO
  __global const char*  restrict x_in, // int8 [in_dim]
  __global char*        restrict y_out,// int8 [out_dim]

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
    int acc = b0[o]; // bias scale = s_in0 * s_w0 (int32)
    #pragma unroll 1
    for (int i = 0; i < in_dim; ++i) {
      const int xqi = (int)x_in[i] - x0_zp;
      const int wqi = (int)W0[w_idx_fc0(o, i, in_dim)] - w0_zp;
      acc += xqi * wqi;
    }
    int yq = requantize_int32(acc, M0_mult, M0_shift, y0_zp);
    if (yq < y0_zp) yq = y0_zp; // ReLU at zero in quant domain
    a1[o] = (char)clamp_int(yq, -128, 127);
  }

  // -------------------------
  // Layer 1: y2 = ReLU_q( Requant( W1*y1 + b1 ) )
  // -------------------------
  for (int o = 0; o < H2_DIM; ++o) {
    int acc = b1[o];
    #pragma unroll 1
    for (int i = 0; i < H1_DIM; ++i) {
      const int xqi = (int)a1[i] - x1_zp;
      const int wqi = (int)W1[w_idx_fc1(o, i)] - w1_zp;
      acc += xqi * wqi;
    }
    int yq = requantize_int32(acc, M1_mult, M1_shift, y1_zp);
    if (yq < y1_zp) yq = y1_zp;
    a2[o] = (char)clamp_int(yq, -128, 127);
  }

  // -------------------------
  // Output layer: logits_q = Requant( W2*y2 + b2 )
  // -------------------------
  for (int o = 0; o < out_dim; ++o) {
    int acc = b2[o];
    #pragma unroll 1
    for (int i = 0; i < H2_DIM; ++i) {
      const int xqi = (int)a2[i] - x2_zp;
      const int wqi = (int)W2[w_idx_fc2(o, i, out_dim)] - w2_zp;
      acc += xqi * wqi;
    }
    int yq = requantize_int32(acc, M2_mult, M2_shift, y2_zp);
    y_out[o] = (char)clamp_int(yq, -128, 127);
  }
}