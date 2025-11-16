// ============================================================================
// CNN INT8 pipeline: Conv(3x3 x16) + ReLU + MaxPool2x2  -> [16,13,13]
//                    Flatten [C,H,W]                     -> 2704
//                    FC1 2704->16 + ReLU                -> 16
//                    FC2 16->10 (logits INT8)
// INT8 activations & weights; INT32 bias & accumulators; INT-only requant.
// Weights layout:
//   - Conv: [Cout, Cin, Kh, Kw]
//   - FC : [Out, In] by default. If your .bin are [In,Out], build with:
//           -D FC_WEIGHTS_ARE_IN_OUT=1
// Tensor layout for conv features: [C, H, W] in memory.
//
// Based on your FP32 kernels (structure, loops, indices).
// ============================================================================

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define H0 28
#define W0 28
#define C0 1

#define Kh 3
#define Kw 3
#define Hcv (H0 - Kh + 1) // 26
#define Wcv (W0 - Kw + 1) // 26
#define H1  (Hcv/2)       // 13
#define W1  (Wcv/2)       // 13
#define C1  16

#define FC_IN (C1*H1*W1)  // 2704
#define FC_M  16
#define FC_O  10

#ifndef FC_WEIGHTS_ARE_IN_OUT
#define FC_WEIGHTS_ARE_IN_OUT 0
#endif

// ---- INT-only requant: acc(int32) -> int using fixed-point (mult, shift) ----
inline int requantize_int32(int acc, int mult, int shift, int zp_out) {
  long prod = (long)acc * (long)mult;  // 64-bit intermediate
  long rq   = (shift > 0) ? ((prod + (1L << (shift - 1))) >> shift) : prod; // round-to-nearest
  return zp_out + (int)rq;
}
inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- FC weight index helpers ([Out,In] vs [In,Out]) ----
inline int w_idx_fc1(int o, int i) {
#if FC_WEIGHTS_ARE_IN_OUT
  return i * FC_M + o;   // [In,Out]
#else
  return o * FC_IN + i;  // [Out,In]
#endif
}
inline int w_idx_fc2(int o, int i) {
#if FC_WEIGHTS_ARE_IN_OUT
  return i * FC_O + o;   // [In,Out]
#else
  return o * FC_M + i;   // [Out,In]
#endif
}

// ============================================================================
// KERNEL 1: Conv3x3 (valid) + ReLU + MaxPool2x2 -> y13 [C1,H1,W1] (INT8)
// Args:
//   Wc0:  int8  [C1, C0, Kh, Kw]
//   bc0:  int32 [C1]
//   w0_zp, M0_mult, M0_shift, x0_zp, y0_zp
//   x_in: int8  [C0,H0,W0] flattened as H0*W0 (C0=1)
//   y13:  int8  [C1,H1,W1]
// ============================================================================
__kernel __attribute__((max_global_work_dim(0)))
void cnn16_conv_relu_pool_int8(
  __global const char*  restrict Wc0,
  __global const int*   restrict bc0,
  const int                      w0_zp,
  const int                      M0_mult,
  const int                      M0_shift,
  const int                      x0_zp,
  const int                      y0_zp,
  __global const char*  restrict x_in,
  __global char*        restrict y13
){
  for (int oc = 0; oc < C1; ++oc){
    for (int pr = 0; pr < H1; ++pr){
      for (int pc = 0; pc < W1; ++pc){
        int vmax = -128; // pooled max in int8

        // 2x2 window over conv-valid map (26x26)
        for (int dr = 0; dr < 2; ++dr){
          for (int dc = 0; dc < 2; ++dc){
            int orow = 2*pr + dr;  // [0..25]
            int ocol = 2*pc + dc;  // [0..25]

            int acc = bc0[oc];     // int32 bias (scale = s_in0*s_w0)
            // Cin=1
            for (int ic = 0; ic < C0; ++ic){
              #pragma unroll 1
              for (int kr = 0; kr < Kh; ++kr){
                int ir = orow + kr; // [0..27]
                for (int kc = 0; kc < Kw; ++kc){
                  #pragma unroll 1
                  int icl = ocol + kc; // [0..27]
                  // W index: [Cout, Cin, Kh, Kw]
                  int w_idx = (((oc*C0 + ic)*Kh + kr)*Kw + kc);
                  const int wqi = (int)Wc0[w_idx] - w0_zp;
                  const int xqi = (int)x_in[ ic*(H0*W0) + ir*W0 + icl ] - x0_zp;
                  acc += xqi * wqi;
                }
              }
            }
            // Requantize & ReLU at zero in quant domain
            int yq = requantize_int32(acc, M0_mult, M0_shift, y0_zp);
            if (yq < y0_zp) yq = y0_zp;
            if (yq > vmax)  vmax = yq;
          }
        }
        y13[ oc*(H1*W1) + pr*W1 + pc ] = (char)clamp_int(vmax, -128, 127);
      }
    }
  }
}

// ============================================================================
// KERNEL 2: FC head (2704->16->10) with ReLU between FCs (INT8)
// Args:
//   Wfc1: int8  [FC_M, FC_IN] (or [FC_IN, FC_M] if FC_WEIGHTS_ARE_IN_OUT)
//   b1:   int32 [FC_M]
//   w1_zp, M1_mult, M1_shift, x1_zp, y1_zp
//   Wfc2: int8  [FC_O, FC_M] (or [FC_M, FC_O])
//   b2:   int32 [FC_O]
//   w2_zp, M2_mult, M2_shift, x2_zp, y2_zp
//   xin:  int8  [FC_IN]  (flattened [C1,H1,W1])
//   out:  int8  [FC_O]   (logits)
// ============================================================================
__kernel __attribute__((max_global_work_dim(0)))
void cnn16_head_fc_int8(
  // FC1
  __global const char*  restrict Wfc1,
  __global const int*   restrict b1,
  const int                      w1_zp,
  const int                      M1_mult,
  const int                      M1_shift,
  const int                      x1_zp,
  const int                      y1_zp,
  // FC2
  __global const char*  restrict Wfc2,
  __global const int*   restrict b2,
  const int                      w2_zp,
  const int                      M2_mult,
  const int                      M2_shift,
  const int                      x2_zp,
  const int                      y2_zp,

  __global const char*  restrict xin,   // [FC_IN]
  __global char*        restrict logits // [FC_O]
){
  // ---- FC1: 2704 -> 16 + ReLU ----
  char a1[FC_M];
  for (int o = 0; o < FC_M; ++o){
    int acc = b1[o];
    #pragma unroll 1
    for (int i = 0; i < FC_IN; ++i){
      const int xqi = (int)xin[i] - x1_zp;
      const int wqi = (int)Wfc1[w_idx_fc1(o,i)] - w1_zp;
      acc += xqi * wqi;
    }
    int yq = requantize_int32(acc, M1_mult, M1_shift, y1_zp);
    if (yq < y1_zp) yq = y1_zp;          // ReLU in quant domain
    a1[o] = (char)clamp_int(yq, -128, 127);
  }

  // ---- FC2: 16 -> 10 (logits) ----
  for (int o = 0; o < FC_O; ++o){
    int acc = b2[o];
    #pragma unroll 1
    for (int i = 0; i < FC_M; ++i){
      const int xqi = (int)a1[i] - x2_zp;
      const int wqi = (int)Wfc2[w_idx_fc2(o,i)] - w2_zp;
      acc += xqi * wqi;
    }
    int yq = requantize_int32(acc, M2_mult, M2_shift, y2_zp);
    logits[o] = (char)clamp_int(yq, -128, 127);
  }
}