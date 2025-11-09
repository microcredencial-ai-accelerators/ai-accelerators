#include <CL/cl.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <cmath>

// Include your cleaned quant header (coloned macros commented out)
#include "quant_params_fc_int.h"

// ---------- Utilities ----------
template<typename T>
std::vector<T> read_bin(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Unable to open: " + path);
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    if(nbytes % sizeof(T) != 0) throw std::runtime_error("Unexpected size at " + path);
    std::vector<T> v(nbytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), nbytes);
    return v;
}

std::vector<unsigned char> read_bytes_file(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("No aocx: " + path);
    f.seekg(0, std::ios::end); size_t n = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> b(n);
    f.read(reinterpret_cast<char*>(b.data()), n);
    return b;
}

cl_mem make_ro_buffer(cl_context ctx, size_t nbytes, const void* host, cl_int* err){
    return clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbytes, const_cast<void*>(host), err);
}
#define CL_CHECK(x) do{ cl_int _e=(x); if(_e!=CL_SUCCESS){ \
  std::cerr << "OpenCL err " << _e << " @ " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(1);} }while(0)

int main(int argc, char** argv){
  try{
    // Args:
    // 1: aocx
    // 2: images_u8.bin (N*784 bytes)
    // 3: labels.bin (N bytes)
    // 4: weights_dir
    const std::string aocx_path = (argc>1? argv[1] : "opencl/kernels/fc_int8.aocx");
    const std::string imgs_bin  = (argc>2? argv[2] : "opencl/data/test_images_u8.bin");
    const std::string labs_bin  = (argc>3? argv[3] : "opencl/data/test_labels.bin");
    const std::string wdir      = (argc>4? argv[4] : "weights/fc_int8");

    // Network: 784 -> 64 -> 32 -> 10
    const int in_dim  = 784;
    const int h1      = 64;
    const int h2      = 32;
    const int out_dim = 10;

    // 1) Load INT8 weights / INT32 biases (stored as [Out, In])
    auto W0 = read_bin<int8_t >(wdir + "/fc0_W.bin");
    auto b0 = read_bin<int32_t>(wdir + "/fc0_b.bin");
    auto W1 = read_bin<int8_t >(wdir + "/fc1_W.bin");
    auto b1 = read_bin<int32_t>(wdir + "/fc1_b.bin");
    auto W2 = read_bin<int8_t >(wdir + "/fc2_W.bin");
    auto b2 = read_bin<int32_t>(wdir + "/fc2_b.bin");

    if ((int)W0.size()!=h1*in_dim || (int)b0.size()!=h1 ||
        (int)W1.size()!=h2*h1    || (int)b1.size()!=h2 ||
        (int)W2.size()!=out_dim*h2 || (int)b2.size()!=out_dim) {
      std::cerr << "[ERR] Weights/bias sizes do not match with the NN.\n";
      return 1;
    }

    // 2) Load raw images (uint8) and labels
    std::ifstream fu(imgs_bin, std::ios::binary);
    if(!fu){ std::cerr<<"[ERR] Unable to open "<<imgs_bin<<"\n"; return 1; }
    fu.seekg(0, std::ios::end); size_t ib = size_t(fu.tellg()); fu.seekg(0, std::ios::beg); // <- fixed
    if(ib % (28*28) != 0){ std::cerr<<"[ERR] "<<imgs_bin<<" not x784 bytes\n"; return 1; }
    const int N = int( ib / (28*28) );
    std::vector<uint8_t> Xraw(N*in_dim);
    fu.read(reinterpret_cast<char*>(Xraw.data()), ib);

    auto Lall = read_bin<uint8_t>(labs_bin); // keep ONLY this declaration
    if ((int)Lall.size()!=N){
      std::cerr<<"[ERR] labels size ("<<Lall.size()<<") does not match images ("<<N<<")\n";
      return 1;
    }
    std::cout<<"[INFO] Batch: "<<N<<" images (u8).\n";

    // 3) OpenCL: platform / device / context / queue
    cl_uint np=0; CL_CHECK(clGetPlatformIDs(0,nullptr,&np));
    std::vector<cl_platform_id> plats(np); CL_CHECK(clGetPlatformIDs(np, plats.data(), nullptr));
    cl_platform_id plat = plats.empty()? nullptr : plats[0];
    for(auto p: plats){
      size_t sz=0; clGetPlatformInfo(p, CL_PLATFORM_NAME,0,nullptr,&sz);
      std::string name(sz,'\0'); clGetPlatformInfo(p, CL_PLATFORM_NAME,sz,&name[0],nullptr);
      if(name.find("Intel(R) FPGA")!=std::string::npos){ plat=p; break; }
    }
    if(!plat){ std::cerr<<"[ERR] OpenCL Platform not found.\n"; return 1; }
    cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr));
    cl_int err=CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CL_CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CL_CHECK(err);

    // 4) Load .aocx and build program
    auto aocx_vec = read_bytes_file(aocx_path);
    const unsigned char* bins[] = { aocx_vec.data() };
    size_t lens[] = { aocx_vec.size() };
    cl_int binst=0;
    cl_program prg = clCreateProgramWithBinary(ctx, 1, &dev, lens, bins, &binst, &err); CL_CHECK(err);
    CL_CHECK(clBuildProgram(prg, 0, nullptr, "", nullptr, nullptr));

    // 5) Kernel
    cl_kernel krn = clCreateKernel(prg, "fc_64x32_infer_int8", &err); CL_CHECK(err);

    // 6) Constant buffers (weights/bias)
    cl_mem dW0 = make_ro_buffer(ctx, W0.size()*sizeof(int8_t),  W0.data(), &err); CL_CHECK(err);
    cl_mem dB0 = make_ro_buffer(ctx, b0.size()*sizeof(int32_t), b0.data(), &err); CL_CHECK(err);
    cl_mem dW1 = make_ro_buffer(ctx, W1.size()*sizeof(int8_t),  W1.data(), &err); CL_CHECK(err);
    cl_mem dB1 = make_ro_buffer(ctx, b1.size()*sizeof(int32_t), b1.data(), &err); CL_CHECK(err);
    cl_mem dW2 = make_ro_buffer(ctx, W2.size()*sizeof(int8_t),  W2.data(), &err); CL_CHECK(err);
    cl_mem dB2 = make_ro_buffer(ctx, b2.size()*sizeof(int32_t), b2.data(), &err); CL_CHECK(err);

    // 7) Quant parameters from your header (use only identifier-safe symbols)
    const float INPUT_SCALE = SEQUENTIAL_2_FLATTEN_2_RESHAPE_SCALE;
    const int   INPUT_ZP    = SEQUENTIAL_2_FLATTEN_2_RESHAPE_ZERO_POINT;

    const float L0_OUT_SCALE = SEQUENTIAL_2_DENSE_6_MATMUL_SEQUENTIAL_2_RE_LU_4_RELU_SEQUENTIAL_2_DENSE_6_BIASADD_SCALE;
    const int   L0_OUT_ZP    = SEQUENTIAL_2_DENSE_6_MATMUL_SEQUENTIAL_2_RE_LU_4_RELU_SEQUENTIAL_2_DENSE_6_BIASADD_ZERO_POINT;
    const float L0_W_SCALE   = SEQUENTIAL_2_DENSE_6_MATMUL_SCALE;
    const int   L0_W_ZP      = SEQUENTIAL_2_DENSE_6_MATMUL_ZERO_POINT;

    const float L1_OUT_SCALE = SEQUENTIAL_2_DENSE_7_MATMUL_SEQUENTIAL_2_RE_LU_5_RELU_SEQUENTIAL_2_DENSE_7_BIASADD_SCALE;
    const int   L1_OUT_ZP    = SEQUENTIAL_2_DENSE_7_MATMUL_SEQUENTIAL_2_RE_LU_5_RELU_SEQUENTIAL_2_DENSE_7_BIASADD_ZERO_POINT;
    const float L1_W_SCALE   = SEQUENTIAL_2_DENSE_7_MATMUL_SCALE;
    const int   L1_W_ZP      = SEQUENTIAL_2_DENSE_7_MATMUL_ZERO_POINT;

    const float L2_OUT_SCALE = SEQUENTIAL_2_DENSE_8_MATMUL_SEQUENTIAL_2_DENSE_8_BIASADD_SCALE;
    const int   L2_OUT_ZP    = SEQUENTIAL_2_DENSE_8_MATMUL_SEQUENTIAL_2_DENSE_8_BIASADD_ZERO_POINT;
    const float L2_W_SCALE   = SEQUENTIAL_2_DENSE_8_MATMUL_SCALE;
    const int   L2_W_ZP      = SEQUENTIAL_2_DENSE_8_MATMUL_ZERO_POINT;

    // Scalar requantization multipliers (per layer)
    const float M0 = (INPUT_SCALE * L0_W_SCALE) / L0_OUT_SCALE;
    const float M1 = (L0_OUT_SCALE * L1_W_SCALE) / L1_OUT_SCALE;
    const float M2 = (L1_OUT_SCALE * L2_W_SCALE) / L2_OUT_SCALE;

    // 8) IO buffers
    std::vector<int8_t> x_q(in_dim), y_q(out_dim);
    cl_mem dX = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  sizeof(int8_t)*in_dim,  nullptr, &err); CL_CHECK(err);
    cl_mem dY = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int8_t)*out_dim, nullptr, &err); CL_CHECK(err);

    // 9) Set static kernel args (must match fc_64x32_infer_int8 signature)
    int a=0;
    // L0
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dW0));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dB0));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L0_W_ZP));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(float),   &M0));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &INPUT_ZP));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L0_OUT_ZP));
    // L1
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dW1));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dB1));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L1_W_ZP));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(float),   &M1));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L0_OUT_ZP)); // x1_zp
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L1_OUT_ZP)); // y1_zp
    // L2
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dW2));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dB2));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L2_W_ZP));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(float),   &M2));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L1_OUT_ZP)); // x2_zp
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &L2_OUT_ZP)); // y2_zp
    // IO + sizes
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dX));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(cl_mem),  &dY));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &in_dim));
    CL_CHECK(clSetKernelArg(krn, a++, sizeof(int),     &out_dim));

    // 10) Run per image: normalize -> quantize -> run -> read -> argmax
    size_t g=1;
    int correct=0;
    double sum_ms = 0.0, min_ms=1e100, max_ms=0.0;
    std::cout << std::fixed << std::setprecision(3);

    for(int n=0; n<N; ++n){
      const uint8_t* raw = &Xraw[n*in_dim];

      // Normalize to [0,1] then quantize to int8 using input params
      for(int i=0;i<in_dim;++i){
        float xf = float(raw[i]) / 255.0f;
        int qv = int(std::round(xf / INPUT_SCALE)) + INPUT_ZP;
        if(qv < -128) qv = -128; if(qv > 127) qv = 127;
        x_q[i] = (int8_t)qv;
      }

      // Copy input
      CL_CHECK(clEnqueueWriteBuffer(q, dX, CL_TRUE, 0, sizeof(int8_t)*in_dim, x_q.data(), 0, nullptr, nullptr));

      // Launch
      cl_event e;
      CL_CHECK(clEnqueueNDRangeKernel(q, krn, 1, nullptr, &g, nullptr, 0, nullptr, &e));
      CL_CHECK(clFinish(q));

      // Read output (int8)
      CL_CHECK(clEnqueueReadBuffer(q, dY, CL_TRUE, 0, sizeof(int8_t)*out_dim, y_q.data(), 0, nullptr, nullptr));

      // Timing
      cl_ulong t0=0, t1=0;
      clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
      clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
      clReleaseEvent(e);
      double ms = double(t1 - t0) / 1e6;
      sum_ms += ms; if(ms<min_ms) min_ms=ms; if(ms>max_ms) max_ms=ms;

      // Argmax on int8 logits
      int pred = int(std::max_element(y_q.begin(), y_q.end()) - y_q.begin());
      bool ok = (pred == int(Lall[n]));
      std::cout << "[" << std::setw(3) << n << "] pred=" << pred
                << " label=" << int(Lall[n])
                << " time=" << ms << " ms -> " << (ok ? "OK" : "FAIL") << "\n";
      if (ok) ++correct;
    }

    const double mean_ms = (N>0 ? sum_ms / N : 0.0);
    std::cout << "\n[RESULT] Accuracy = " << (100.0 * correct / N) << "% ("<<correct<<"/"<<N<<")\n";
    std::cout << "[RESULT] Average kernel inference time = " << mean_ms
              << " ms (min=" << min_ms << " ms, max=" << max_ms << " ms)\n";

    // Cleanup
    clReleaseMemObject(dW0); clReleaseMemObject(dB0);
    clReleaseMemObject(dW1); clReleaseMemObject(dB1);
    clReleaseMemObject(dW2); clReleaseMemObject(dB2);
    clReleaseMemObject(dX);  clReleaseMemObject(dY);
    clReleaseKernel(krn); clReleaseProgram(prg); clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
  }catch(const std::exception& e){
    std::cerr<<"Except: "<<e.what()<<"\n"; return 1;
  }
}