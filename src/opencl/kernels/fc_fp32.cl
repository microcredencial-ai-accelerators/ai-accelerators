
// kernels/fc_fp32.cl
__kernel void fc_mnist(
    // Weights and biases for each layer
    __global const float* restrict W0, __global const float* restrict b0, // [128,784], [128]
    __global const float* restrict W1, __global const float* restrict b1, // [32,128], [32]
    __global const float* restrict W2, __global const float* restrict b2, // [10,32],  [10]
    __global const float* restrict x_in,   // [784]
    __global float* restrict y_out         // [10] logits
){
    // Intermediate buffers in global memory
    float a0[128];
    float a1[32];

    // FC0: 784 -> 128 + ReLU
    for (int o=0; o<128; ++o){
        float acc = b0[o];
        #pragma unroll 4
        for (int i=0; i<784; ++i){
            acc += W0[o*784 + i] * x_in[i];
        }
        // ReLU
        a0[o] = acc > 0.f ? acc : 0.f;
    }

    // FC1: 128 -> 32 + ReLU
    for (int o=0; o<32; ++o){
        float acc = b1[o];
        #pragma unroll 4
        for (int i=0; i<128; ++i){
            acc += W1[o*128 + i] * a0[i];
        }
        a1[o] = acc > 0.f ? acc : 0.f;
    }

    // FC2: 32 -> 10 (logits)
    for (int o=0; o<10; ++o){
        float acc = b2[o];
        #pragma unroll 4
        for (int i=0; i<32; ++i){
            acc += W2[o*32 + i] * a1[i];
        }
        y_out[o] = acc;
    }
}
