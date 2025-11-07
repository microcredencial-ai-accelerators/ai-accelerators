Check OpenCL SDL:
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
aoc -v -g -board=$BOARD src/opencl/kernels/cnn_fp32_nounroll.cl -o output/cnn_fp32_nounroll.aocx
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
g++ -O2 -std=c++11 opencl/host/main_fc_fp32.cpp -o opencl/apps/fc_fp32_host \
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
  opencl/kernels/fc_fp32.aocx \
  opencl/data/test_images_u8_10.bin \
  opencl/data/test_labels_10.bin \
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