Check OpenCL SDL:
```
aoc -version
```
List boards:
```
aoc -list-boards
```

Stimation
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
Build:
```
g++ -O2 -std=c++11 host/main_fc_batch_norm.cpp -lOpenCL -o /home/root/opencl/apps/fc_host
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
opencl/apps/fc_host \
  opencl/kernels/fc_fp32.aocx \
  opencl/data/test_images_u8_10.bin \
  opencl/data/test_labels_10.bin \
  opencl/weights/fc \
  opencl/data/raw_pgms \
  10

```