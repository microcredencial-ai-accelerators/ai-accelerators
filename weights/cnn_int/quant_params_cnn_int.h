#ifndef QUANT_PARAMS_H_
#define QUANT_PARAMS_H_

// Auto-generated quantization parameters

// Tensor: serving_default_conv2d_input:0
static const float serving_default_conv2d_input_0_scale = 0.003921569f;
static const int   serving_default_conv2d_input_0_zp    = -128;

// Tensor: sequential/flatten/Const
static const float sequential_flatten_Const_scale = 1.000000000f;
static const int   sequential_flatten_Const_zp    = 0;

// Tensor: sequential/dense_1/BiasAdd/ReadVariableOp
static const float sequential_dense_1_BiasAdd_ReadVariableOp_scale = 0.000536439f;
static const int   sequential_dense_1_BiasAdd_ReadVariableOp_zp    = 0;

// Tensor: sequential/dense_1/MatMul
static const float sequential_dense_1_MatMul_scale = 0.008457888f;
static const int   sequential_dense_1_MatMul_zp    = 0;

// Tensor: sequential/dense/BiasAdd/ReadVariableOp
static const float sequential_dense_BiasAdd_ReadVariableOp_scale = 0.000036407f;
static const int   sequential_dense_BiasAdd_ReadVariableOp_zp    = 0;

// Tensor: sequential/dense/MatMul
static const float sequential_dense_MatMul_scale = 0.005849662f;
static const int   sequential_dense_MatMul_zp    = 0;

// Tensor: sequential/conv2d/BiasAdd/ReadVariableOp
static const float sequential_conv2d_BiasAdd_ReadVariableOp_scale = 0.000006750f;
static const int   sequential_conv2d_BiasAdd_ReadVariableOp_zp    = 0;

// Tensor: sequential/conv2d/Conv2D
static const float sequential_conv2d_Conv2D_scale = 0.001721204f;
static const int   sequential_conv2d_Conv2D_zp    = 0;

// Tensor: sequential/max_pooling2d/MaxPool
static const float sequential_max_pooling2d_MaxPool_scale = 0.006223819f;
static const int   sequential_max_pooling2d_MaxPool_zp    = -128;

// Tensor: sequential/flatten/Reshape
static const float sequential_flatten_Reshape_scale = 0.006223819f;
static const int   sequential_flatten_Reshape_zp    = -128;

// Tensor: sequential/dense_1/MatMul;sequential/dense_1/BiasAdd
static const float sequential_dense_1_MatMul_sequential_dense_1_BiasAdd_scale = 0.170450300f;
static const int   sequential_dense_1_MatMul_sequential_dense_1_BiasAdd_zp    = 19;

// Tensor: StatefulPartitionedCall:0
static const float StatefulPartitionedCall_0_scale = 0.003906250f;
static const int   StatefulPartitionedCall_0_zp    = -128;

#endif // QUANT_PARAMS_H_
