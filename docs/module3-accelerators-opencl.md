# Introduction to OpenCL and Intel FPGA SDK for OpenCL Using DE10-Nano (Cyclone V SoC)
## [Back to Module 3](module3-accelerators.md)

## Overview
This lab introduces OpenCL programming on Intel FPGAs using the Intel FPGA SDK for OpenCL.
You will learn the basics of OpenCL concepts, how to write and compile kernels for FPGA, and how to analyze performance using Quartus reports.

---

## OpenCL Basics

An OpenCL application is divided into two parts:
### 1. Host Code (CPU / ARM)
Controls the execution of kernels on the FPGA.
Key concepts:
- Platform and Device selection
- Context creation
- Command queue for submitting work
- Memory buffers for data transfer
- Program and kernel creation
- Execution and profiling
Common functions:
```
clGetPlatformIDs
clGetPlatformInfo
clGetDeviceIDs
clCreateContext
clCreateCommandQueue
clCreateProgramWithBinary
clBuildProgram
clCreateKernel
clCreateBuffer
clSetKernelArg
clEnqueueNDRangeKernel
clGetEventProfilingInfo
clReleaseEvent
clReleaseMemObject
clReleaseKernel
clReleaseProgram
clReleaseCommandQueue
clReleaseContext
```
### 2. Kernel Code (FPGA Accelerator)
Implements the core computation.
Kernels are written in OpenCL C and compiled with aoc into an FPGA bitstream (.aocx).

## Project Structure

```
project_root/
├── src/
│   └── opencl/
│   │   └── kernels/
│   │       ├── fc_fp32.cl
│   │       └── cnn_fp32.cl
│   └── host/
|       ├── main_fc_fp32.cpp
│       └── main_cnn_fp32.cpp
└── output/
    ├── fc_fp32/
    └── cnn_fp32/
```
Place kernel source files in:
```
./src/opencl/kernels/
```
Compilation output (bitstreams, logs, and reports) will go to:
```
./output/{kernel_name}/
```
## Part 1 — Fully Connected (FC) Kernel
File: src/opencl/kernels/fc_fp32.cl
Goal: Implement the matrix-vector multiplication and ReLU activation.
### Task: 
#### 1. Complete the multiply–accumulate (MAC) loop:
```
acc += W[o * in_dim + i] * x_in[i];
```
#### 2. Add ReLU activation:
```
y[o] = (acc > 0.0f) ? acc : 0.0f;
```
### Memory Layout
Weights are stored row-major by output neuron:
```
W0 = [ w(0,0), w(0,1), ..., w(0,in_dim-1),​

       w(1,0), w(1,1), ..., w(1,in_dim-1),​

       ...​

       w(H1_DIM-1, in_dim-1) ]
```
```
index = o * in_dim + i; // o = output neuron index, i = input feature index​
W0[index]
```
## Part 2 Part 2 — CNN Layer Kernel
File: src/opencl/kernels/cnn_fp32.cl
Goal: Complete CNN kernel implementation with the FC layers and the ReLU functions

## Compile both kernels
List the available boards and export the "BOARD" variable: 
```
aoc -list-boards
export BOARD=de10_nano_sharedonly
```
Compile the kernel for FPGA using the Intel FPGA SDK for OpenCL (aoc):
```
aoc -v -g -report -board=$BOARD src/opencl/kernels/fc_fp32.cl -o output/fc_fp32.aocx​
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
Compilation produces:
- .aocx — FPGA bitstream
- .html, .rpt — Reports (resource usage, performance)
- .log — Build logs

## Optimization with Pragmas

Experiment with Intel FPGA OpenCL pragmas inside your loops:
```
#pragma unroll 4
for (int i = 0; i < in_dim; ++i) {
    acc += W[o * in_dim + i] * x_in[i];
}
```

For each pragma configuration:
- Recompile the kernel
- Compare resource usage (DSPs, ALMs, M20Ks)

## Analyzing Quartus Reports

After compilation, open the HTML report (e.g. output/fc_fp32/report.html) and look for:
- Logic Utilization (ALMs)
- RAM Blocks​
- DSPs
- Block Memory bits​

## Deliverables 
- *.cl kernel files​
- Summary table with resource usage​
- .aocx bitstreams