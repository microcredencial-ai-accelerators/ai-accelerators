# Deploying Accelerator on a FPGA with OpenCL and Intel SDK for OpenCL
## [Back to Module 4](module4-designflows.md)

## 1Export Model Weights
Export weights from your trained TensorFlow/Keras model:

Conv2D: [O, I, kH, kW]
Dense layers: [Out, In]

Save as binary files:
```
conv0_W.bin, conv0_b.bin
fc1_W.bin, fc1_b.bin
fc2_W.bin, fc2_b.bin
```

## Host PC
Check OpenCL SDK:
```
aoc -version
```
List boards:
```
aoc -list-boards
```

Estimation
```
aoc -c -v -g -board=$BOARD src/opencl/kernels/cnn_fp32_nounroll.cl -o output/cnn_fp32_nounroll.aocx
```

Compile:
```
export BOARD=de10_nano_sharedonly
aoc -v -g -board=$BOARD src/opencl/kernels/cnn_fp32.cl -o output/cnn_fp32_unroll8.aocx
```
```
aoc: Environment checks are completed successfully.
aoc: Cached files in /var/tmp/aocl/ may be used to reduce compilation time
You are now compiling the full flow!!
aoc: Selected target board de10_nano_sharedonly
aoc: Running OpenCL parser....
aoc: OpenCL parser completed successfully.
aoc: Optimizing and doing static analysis of code...
aoc: Linking with IP library ...
Checking if memory usage is larger than 100%
Compiler Warning: Auto-unrolled loop at /home/aidev/repositories/ai-dev//home/aidev/repositories/ai-dev/src/opencl/kernels/cnn_fp32.cl:42
aoc: First stage compilation completed successfully.
Compiling for FPGA. This process may take a long time, please be patient.
aoc: Hardware generation completed successfully.
```


Crosscompile application
```
```

## Board 
```
source ./init_opencl.sh
aocl diagnose
```

Output:
```
aocl diagnose: Running diagnostic from /home/root/opencl_arm32_rte/board/c5soc/arm32/bin                                                               
                                                                                                                                                       
Verified that the kernel mode driver is installed on the host machine.                                                                                 
                                                                                                                                                       
Using platform: Intel(R) FPGA SDK for OpenCL(TM)                                                                                                       
Board vendor name: Intel(R) Corporation                                                                                                                
Board name: de10_nano_sharedonly : Cyclone V SoC Development Kit                                                                                       
                                                                                                                                                       
Buffer read/write test passed.                                                                                                                         
                                                                                                                                                       
DIAGNOSTIC_PASSED
```
Build:
```
g++ -O2 -std=c++11 opencl/host/main_cnn_fp32.cpp -o opencl/apps/cnn_fp32_host \
  $(aocl compile-config) \
  $(aocl link-config)

```
Run:
```

#!/bin/sh
set -e
cd /home/root
source ./init_opencl.sh
aocl diagnose
aocl program /dev/acl0 opencl/kernels/fc_fp32.aocx

# Ejecuta el host: aocx  imgs_u8  labels  weights_dir    out_dir_PGMs          save_k
opencl/apps/fc_fp32_host \
  opencl/kernels/fc_fp32_unroll8.aocx \
  opencl/data/test_images_u8.bin \
  opencl/data/test_labels.bin \
  opencl/weights/fc_fp32

```

```
[INFO] Batch: 10000 images (u8).
Reprogramming device [0] with handle 1
[  0] pred=7 label=7 time=0.423 ms -> OK
[  1] pred=2 label=2 time=0.372 ms -> OK
[  2] pred=1 label=1 time=0.363 ms -> OK
[  3] pred=0 label=0 time=0.362 ms -> OK
[  4] pred=4 label=4 time=0.359 ms -> OK
[  5] pred=1 label=1 time=0.356 ms -> OK
[  6] pred=4 label=4 time=0.359 ms -> OK
[  7] pred=9 label=9 time=0.358 ms -> OK
[  8] pred=6 label=5 time=0.357 ms -> FAIL
...
```
Performance summary:
```
[RESULT] Accuracy = 97.050%  (9705/10000)                                                                                                                        
[RESULT] Average kernel inference time = 0.359 ms  (min=0.355 ms, max=2.695 ms)  
```

opencl/apps/cnn_fp32_host \
  opencl/kernels/cnn_fp32.aocx \
  opencl/data/test_images_u8.bin \
  opencl/data/test_labels.bin \
  opencl/weights/cnn_fp32