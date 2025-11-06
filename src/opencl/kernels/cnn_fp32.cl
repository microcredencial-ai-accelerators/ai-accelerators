// kernels/cnn_small_fp32.cl
// Model: Conv(16, 3x3, valid) → ReLU → MaxPool2x2 → Flatten(2704) → Dense(16) → ReLU → Dense(10)
// Tensor layout is [C, H, W] in row-major: idx = c*(H*W) + h*W + w
// Conv weights: [Cout, Cin, Kh, Kw] ; FC weights: [Out, In]

// -----------------------------------------------------------------------------
// KERNEL 1: 3x3 Convolution (Cout=16, Cin=1), ReLU and inline MaxPool 2x2
// Input:   x_in  [1,28,28]  (flattened as H*W)
// Output:  yout  [16,13,13] (conv valid 26x26 then pool 2x2 → 13x13)
// -----------------------------------------------------------------------------
__attribute__((max_global_work_dim(0)))
__kernel void cnn16_conv_relu_pool(
    __global const float* restrict Wc0,   // [16,1,3,3] = 16*1*3*3
    __global const float* restrict bc0,   // [16]
    __global const float* restrict x_in,  // [1,28,28] = 784
    __global float* restrict yout         // [16,13,13] = 2704
){
    const int Cin  = 1,   Cout = 16;
    const int Hin  = 28,  Win  = 28;
    const int Kh   = 3,   Kw   = 3;
    const int Hcv  = Hin - Kh + 1; // 26
    const int Wcv  = Win - Kw + 1; // 26
    const int Hout = Hcv / 2;      // 13
    const int Wout = Wcv / 2;      // 13

    for (int oc = 0; oc < Cout; ++oc){
        for (int pr = 0; pr < Hout; ++pr){
            for (int pc = 0; pc < Wout; ++pc){
                float vmax = -3.4e38f;
                // 2x2 pooling window over the conv (valid) map
                for (int dr = 0; dr < 2; ++dr){
                    for (int dc = 0; dc < 2; ++dc){
                        int orow = 2*pr + dr; // [0..25]
                        int ocol = 2*pc + dc; // [0..25]
                        float acc = bc0[oc];

                        // Cin=1; keep the loop explicit for clarity
                        for (int ic = 0; ic < Cin; ++ic){
                            #pragma unroll 1
                            for (int kr = 0; kr < Kh; ++kr){
                                int ir = orow + kr;     // [0..27]
                                for (int kc = 0; kc < Kw; ++kc){
                                    #pragma unroll 1
                                    int icl = ocol + kc; // [0..27]
                                    float w = Wc0[ (((oc*Cin + ic)*Kh + kr)*Kw + kc) ];
                                    float x = x_in[ ic*(Hin*Win) + ir*Win + icl ];
                                    acc += w * x;
                                }
                            }
                        }
                        // ReLU
                        if (acc < 0.0f) acc = 0.0f;
                        if (acc > vmax) vmax = acc;
                    }
                }
                yout[ oc*(Hout*Wout) + pr*Wout + pc ] = vmax;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2: FC head (2704→16→10) with intermediate ReLU
// Input:   xin    [16,13,13] (=2704)  (treated as a flattened vector)
// Weights: W1 [16,2704], b1 [16]; W2 [10,16], b2 [10]
// Output:  logits [10]  (Softmax/argmax can be done on host if needed)
// -----------------------------------------------------------------------------
__attribute__((max_global_work_dim(0)))
__kernel void cnn16_head_fc(
    __global const float* restrict W1,   // [16,2704]
    __global const float* restrict b1,   // [16]
    __global const float* restrict W2,   // [10,16]
    __global const float* restrict b2,   // [10]
    __global const float* restrict xin,  // [16,13,13] (2704)
    __global float* restrict logits      // [10]
){
    const int FC_IN = 16 * 13 * 13; // 2704
    const int M     = 16;
    const int O     = 10;

    // FC1: 2704 -> 16 + ReLU
    float a1[16];
    for (int o = 0; o < M; ++o){
        float acc = b1[o];
        #pragma unroll 1
        for (int i = 0; i < FC_IN; ++i){
            acc += W1[o*FC_IN + i] * xin[i];
        }
        a1[o] = (acc > 0.0f) ? acc : 0.0f;
    }

    // FC2: 16 -> 10 (logits)
    for (int o = 0; o < O; ++o){
        float acc = b2[o];
        #pragma unroll 1
        for (int i = 0; i < M; ++i){
            acc += W2[o*M + i] * a1[i];
        }
        logits[o] = acc; // prediction = argmax(logits) (softmax not required)
    }
}