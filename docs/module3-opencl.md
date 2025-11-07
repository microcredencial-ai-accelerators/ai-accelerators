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
g++ -O2 -std=c++11 host/main_fc_batch_norm.cpp -o opencl/apps/fc_host \
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
  opencl/data/test_images_u8.bin \
  opencl/data/test_labels.bin \
  opencl/weights/fc_fp32 \
  opencl/data/raw_pgms \
  10000

```

opencl/apps/fc_fp32_host \
  opencl/kernels/fc_fp32.aocx \
  opencl/data/test_images_u8.bin \
  opencl/data/test_labels.bin \
  opencl/weights/fc_fp32 \
  opencl/data/raw_pgms \
  10000