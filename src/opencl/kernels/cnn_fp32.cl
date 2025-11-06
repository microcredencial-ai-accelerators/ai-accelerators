// kernels/cnn_kernels_fp32.cl
// CNN MNIST: Conv3x3-32 → ReLU → MaxPool2 → Conv3x3-64 → ReLU → MaxPool2 → FC(64) → ReLU → FC(10)
// Layout tensores: [C, H, W] en row-major.
// Peso conv: [Cout, Cin, KH, KW] ; Peso FC: [Out, In].

// Para compilar: aoc -v -board=<tu_board> kernels/cnn_kernels_fp32.cl -o cnn_fp32.aocx
// Referencias de flujo/flags/reportes: Programming Guide del SDK OpenCL (aoc/aocl) [1](http://www.innovatefpga.com/cgi-bin/innovate/teams2018.pl?Id=PR065)

__attribute__((max_global_work_dim(0)))
__kernel void cnn_conv0_relu_pool(
    __global const float* restrict Wc0,   // [32,1,3,3] => 32*1*3*3
    __global const float* restrict bc0,   // [32]
    __global const float* restrict x_in,  // [1,28,28] => 28*28
    __global float* restrict y_pool0      // [32,13,13] => 32*13*13
){
    const int Cin = 1, Cout = 32;
    const int Hin = 28, Win = 28;
    const int Kh = 3, Kw = 3;
    const int Hconv = Hin - Kh + 1; // 26
    const int Wconv = Win - Kw + 1; // 26
    const int Hout = Hconv / 2;     // 13
    const int Wout = Wconv / 2;     // 13

    for (int oc = 0; oc < Cout; ++oc){
        for (int pr = 0; pr < Hout; ++pr){
            for (int pc = 0; pc < Wout; ++pc){
                float vmax = -3.4e38f;
                // 2x2 pooling window sobre el mapa de conv "valid"
                for (int dr = 0; dr < 2; ++dr){
                    for (int dc = 0; dc < 2; ++dc){
                        int orow = 2*pr + dr;
                        int ocol = 2*pc + dc;
                        float acc = bc0[oc];
                        // Cin = 1 (pero dejamos el bucle por claridad)
                        for (int ic = 0; ic < Cin; ++ic){
                            for (int kr = 0; kr < Kh; ++kr){
                                int ir = orow + kr;
                                for (int kc = 0; kc < Kw; ++kc){
                                    int icl = ocol + kc;
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
                y_pool0[ oc*(Hout*Wout) + pr*Wout + pc ] = vmax;
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
__kernel void cnn_conv1_relu_pool(
    __global const float* restrict Wc1,    // [64,32,3,3]
    __global const float* restrict bc1,    // [64]
    __global const float* restrict x_pool, // [32,13,13]
    __global float* restrict y_pool1       // [64,5,5]
){
    const int Cin = 32, Cout = 64;
    const int Hin = 13, Win = 13;
    const int Kh = 3, Kw = 3;
    const int Hconv = Hin - Kh + 1; // 11
    const int Wconv = Win - Kw + 1; // 11
    const int Hout = Hconv / 2;     // 5
    const int Wout = Wconv / 2;     // 5

    for (int oc = 0; oc < Cout; ++oc){
        for (int pr = 0; pr < Hout; ++pr){
            for (int pc = 0; pc < Wout; ++pc){
                float vmax = -3.4e38f;
                for (int dr = 0; dr < 2; ++dr){
                    for (int dc = 0; dc < 2; ++dc){
                        int orow = 2*pr + dr;
                        int ocol = 2*pc + dc;
                        float acc = bc1[oc];
                        for (int ic = 0; ic < Cin; ++ic){
                            for (int kr = 0; kr < Kh; ++kr){
                                int ir = orow + kr;
                                for (int kc = 0; kc < Kw; ++kc){
                                    int icl = ocol + kc;
                                    float w = Wc1[ (((oc*Cin + ic)*Kh + kr)*Kw + kc) ];
                                    float x = x_pool[ ic*(Hin*Win) + ir*Win + icl ];
                                    acc += w * x;
                                }
                            }
                        }
                        if (acc < 0.0f) acc = 0.0f; // ReLU
                        if (acc > vmax) vmax = acc;
                    }
                }
                y_pool1[ oc*(Hout*Wout) + pr*Wout + pc ] = vmax;
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
__kernel void cnn_head_fc(
    __global const float* restrict W3,  // [64,1600]
    __global const float* restrict b3,  // [64]
    __global const float* restrict W4,  // [10,64]
    __global const float* restrict b4,  // [10]
    __global const float* restrict x_f, // [1600]  (flatten de y_pool1)
    __global float* restrict logits     // [10]
){
    // Dense(1600->64) + ReLU
    float a3[64];
    for (int o=0; o<64; ++o){
        float acc = b3[o];
        #pragma unroll 4
        for (int i=0; i<1600; ++i){
            acc += W3[o*1600 + i] * x_f[i];
        }
        a3[o] = (acc > 0.0f) ? acc : 0.0f;
    }
    // Dense(64->10) => logits
    for (int o=0; o<10; ++o){
        float acc = b4[o];
        #pragma unroll 4
        for (int i=0; i<64; ++i){
            acc += W4[o*64 + i] * a3[i];
        }
        logits[o] = acc; // softmax/argmax en host
    }
}