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

#include "quant_params_cnn_int.h"  // your identifier-safe header

// ---------- Utilities ----------
template<typename T>
std::vector<T> read_bin(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    if(nbytes % sizeof(T)) throw std::runtime_error("Unexpected size in: " + path);
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

// ---------- Integer requant: real M -> (mult, shift) ----------
static inline void QuantizeMultiplier(double M, int32_t& mult, int32_t& shift) {
    if (M <= 0.0) { mult = 0; shift = 0; return; }
    int exp;
    const double norm = std::frexp(M, &exp);         // M = norm * 2^exp, norm in [0.5,1)
    long long q31 = (long long)std::llround(norm * (1ll<<31)); // [2^30, 2^31]
    if (q31 == (1ll<<31)) { q31 >>= 1; ++exp; }      // handle rare round-up
    mult = (int32_t)q31;                             // Q31
    shift = 31 - exp;                                // keep shift >= 0
    if (shift < 0) { mult <<= (-shift); shift = 0; }
}

// ---------- Reorder FC1 columns: NHWC (H,W,C) -> NCHW (C,H,W), int8 ----------
static std::vector<int8_t>
reorder_fc1_columns_nhwc_to_nchw_i8(const std::vector<int8_t>& W1_src,
                                    int C, int H, int W, int Out)
{
    const int In = C * H * W;
    if ((int)W1_src.size() != Out * In) {
        throw std::runtime_error("[ERR] FC1 reorder: size mismatch");
    }
    std::vector<int8_t> W1_dst(Out * In);
    for (int c = 0; c < C; ++c){
        for (int h = 0; h < H; ++h){
            for (int w_ = 0; w_ < W; ++w_){
                const int i_nchw = ((c * H) + h) * W + w_; // (c,h,w)
                const int i_nhwc = ((h * W) + w_) * C + c; // (h,w,c)
                for (int o = 0; o < Out; ++o){
                    W1_dst[o * In + i_nchw] = W1_src[o * In + i_nhwc];
                }
            }
        }
    }
    return W1_dst;
}

int main(int argc, char** argv){
  try{
    // Args:
    // 1: aocx
    // 2: images_u8.bin (N*784 bytes)
    // 3: labels.bin (N bytes)
    // 4: weights_dir (int8/int32 bins)
    const std::string aocx_path = (argc>1? argv[1] : "opencl/kernels/cnn_int.aocx");
    const std::string imgs_bin  = (argc>2? argv[2] : "opencl/data/test_images_u8.bin");
    const std::string labs_bin  = (argc>3? argv[3] : "opencl/data/test_labels.bin");
    const std::string wdir      = (argc>4? argv[4] : "weights/cnn_int8");

    // Shapes
    const int H0=28, W0=28, C0=1;
    const int C1=16, H1=13, W1=13;     // <- keep these names for spatial dims
    const int FC_IN = C1*H1*W1;        // 2704
    const int FC_M  = 16;
    const int FC_O  = 10;

    // 1) Load INT8 weights / INT32 biases
    auto Wc0  = read_bin<int8_t >(wdir + "/conv0_W.bin");
    auto bc0  = read_bin<int32_t>(wdir + "/conv0_b.bin");
    auto Wfc1 = read_bin<int8_t >(wdir + "/fc1_W.bin");   // <- renamed
    auto b1   = read_bin<int32_t>(wdir + "/fc1_b.bin");
    auto Wfc2 = read_bin<int8_t >(wdir + "/fc2_W.bin");   // <- renamed
    auto b2   = read_bin<int32_t>(wdir + "/fc2_b.bin");

    if ((int)Wc0.size()!=C1*C0*3*3 ||
        (int)bc0.size()!=C1        ||
        (int)Wfc1.size()!=FC_M*FC_IN||
        (int)b1.size()!=FC_M       ||
        (int)Wfc2.size()!=FC_O*FC_M||
        (int)b2.size()!=FC_O) {
      std::cerr << "[ERR] Weight sizes do not match the network.\n";
      return 1;
    }

    // FC1 columns NHWC -> NCHW (to match conv feature memory layout)
    {
      std::vector<int8_t> W1_fixed = reorder_fc1_columns_nhwc_to_nchw_i8(Wfc1, C1, H1, W1, FC_M);
      Wfc1.swap(W1_fixed);
    }

    // 2) Load raw images (uint8) and labels
    std::ifstream fu(imgs_bin, std::ios::binary);
    if(!fu){ std::cerr<<"[ERR] Cannot open "<<imgs_bin<<"\n"; return 1; }
    fu.seekg(0, std::ios::end); size_t ib = size_t(fu.tellg()); fu.seekg(0, std::ios::beg);
    if(ib % (H0*W0) != 0){ std::cerr<<"[ERR] "<<imgs_bin<<" is not a multiple of 784 bytes\n"; return 1; }
    const int N = int( ib / (H0*W0) );
    std::vector<uint8_t> Xraw(N*H0*W0);
    fu.read(reinterpret_cast<char*>(Xraw.data()), ib);

    auto Lall = read_bin<uint8_t>(labs_bin);
    if ((int)Lall.size()!=N){
      std::cerr<<"[ERR] labels count ("<<Lall.size()<<") != images ("<<N<<")\n";
      return 1;
    }
    std::cout<<"[INFO] Batch: "<<N<<" images (u8).\n";

    // 3) OpenCL boilerplate
    cl_uint np=0; CL_CHECK(clGetPlatformIDs(0,nullptr,&np));
    std::vector<cl_platform_id> plats(np); CL_CHECK(clGetPlatformIDs(np, plats.data(), nullptr));
    cl_platform_id plat = plats.empty()? nullptr : plats[0];
    for(auto p: plats){
      size_t sz=0; clGetPlatformInfo(p, CL_PLATFORM_NAME,0,nullptr,&sz);
      std::string name(sz,'\0'); clGetPlatformInfo(p, CL_PLATFORM_NAME,sz,&name[0],nullptr);
      if(name.find("Intel(R) FPGA")!=std::string::npos){ plat=p; break; }
    }
    if(!plat){ std::cerr<<"[ERR] OpenCL platform not found.\n"; return 1; }
    cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr));
    cl_int err=CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CL_CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CL_CHECK(err);

    // 4) Load .aocx and program
    auto aocx_vec = read_bytes_file(aocx_path);
    const unsigned char* bins[] = { aocx_vec.data() };
    size_t lens[] = { aocx_vec.size() };
    cl_int binst=0;
    cl_program prg = clCreateProgramWithBinary(ctx, 1, &dev, lens, bins, &binst, &err); CL_CHECK(err);
    CL_CHECK(clBuildProgram(prg, 0, nullptr, "", nullptr, nullptr));

    // 5) Kernels
    cl_kernel k_conv = clCreateKernel(prg, "cnn16_conv_relu_pool_int8", &err); CL_CHECK(err);
    cl_kernel k_head = clCreateKernel(prg, "cnn16_head_fc_int8", &err);        CL_CHECK(err);

    // 6) Constant buffers
    cl_mem dWc0  = make_ro_buffer(ctx, Wc0 .size()*sizeof(int8_t),  Wc0 .data(), &err); CL_CHECK(err);
    cl_mem dBc0  = make_ro_buffer(ctx, bc0 .size()*sizeof(int32_t), bc0 .data(), &err); CL_CHECK(err);
    cl_mem dWfc1 = make_ro_buffer(ctx, Wfc1.size()*sizeof(int8_t),  Wfc1.data(), &err); CL_CHECK(err);
    cl_mem dB1   = make_ro_buffer(ctx, b1  .size()*sizeof(int32_t), b1  .data(), &err); CL_CHECK(err);
    cl_mem dWfc2 = make_ro_buffer(ctx, Wfc2.size()*sizeof(int8_t),  Wfc2.data(), &err); CL_CHECK(err);
    cl_mem dB2   = make_ro_buffer(ctx, b2  .size()*sizeof(int32_t), b2  .data(), &err); CL_CHECK(err);

    // 7) Quant params (from your header)
    const float X0_SCALE = serving_default_conv2d_input_0_scale;
    const int   X0_ZP    = serving_default_conv2d_input_0_zp;

    const float WC_SCALE = sequential_conv2d_Conv2D_scale;
    const int   WC_ZP    = sequential_conv2d_Conv2D_zp;
    const float Y0_SCALE = sequential_max_pooling2d_MaxPool_scale; // conv->relu->pool output
    const int   Y0_ZP    = sequential_max_pooling2d_MaxPool_zp;

    const float X1_SCALE = sequential_flatten_Reshape_scale;       // == Y0_SCALE
    const int   X1_ZP    = sequential_flatten_Reshape_zp;          // == Y0_ZP
    const float W1_SCALE = sequential_dense_MatMul_scale;
    const int   W1_ZP    = sequential_dense_MatMul_zp;

    const float W2_SCALE  = sequential_dense_1_MatMul_scale;
    const int   W2_ZP     = sequential_dense_1_MatMul_zp;
    const float Y2_SCALE  = sequential_dense_1_MatMul_sequential_dense_1_BiasAdd_scale; // logits
    const int   Y2_ZP     = sequential_dense_1_MatMul_sequential_dense_1_BiasAdd_zp;

    const float B2_SCALE  = sequential_dense_1_BiasAdd_ReadVariableOp_scale;
    const float X2_SCALE  = B2_SCALE / W2_SCALE;    // FC2 input scale = FC1 output activation scale
    const int   X2_ZP     = -128;                  // typical activation zp

    (void)sequential_dense_BiasAdd_ReadVariableOp_scale; // silence unused if present

    // 8) Compute integer requant multipliers
    int32_t M0_mult, M0_shift, M1_mult, M1_shift, M2_mult, M2_shift;
    QuantizeMultiplier((double)X0_SCALE * (double)WC_SCALE / (double)Y0_SCALE, M0_mult, M0_shift);
    QuantizeMultiplier((double)X1_SCALE * (double)W1_SCALE / (double)X2_SCALE, M1_mult, M1_shift);
    QuantizeMultiplier((double)X2_SCALE * (double)W2_SCALE / (double)Y2_SCALE, M2_mult, M2_shift);

    // 9) Activation buffers
    std::vector<int8_t> x_q(H0*W0), y_logits(FC_O);
    cl_int errc=CL_SUCCESS;
    cl_mem dX   = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  sizeof(int8_t)*H0*W0, nullptr, &errc); CL_CHECK(errc);
    cl_mem dY13 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int8_t)*C1*H1*W1, nullptr, &errc); CL_CHECK(errc);
    cl_mem dLOG = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int8_t)*FC_O,    nullptr, &errc); CL_CHECK(errc);

    // 10) Set static kernel args
    // conv
    int a=0;
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dWc0));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dBc0));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(int),    &WC_ZP));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(int),    &M0_mult));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(int),    &M0_shift));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(int),    &X0_ZP));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(int),    &Y0_ZP));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dX));
    CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dY13));

    // head (order must match cnn16_head_fc_int8 signature)
    a=0;
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dWfc1));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dB1));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &W1_ZP));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &M1_mult));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &M1_shift));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &X1_ZP));    // x1_zp = Y0_ZP
    const int Y1_ZP = X2_ZP; // fc1 output zp = fc2 input zp
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &Y1_ZP));

    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dWfc2));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dB2));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &W2_ZP));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &M2_mult));
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &M2_shift));
    const int X2_ZP_val = X2_ZP;
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &X2_ZP_val)); // x2_zp
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(int),    &Y2_ZP));     // logits zp

    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dY13));      // xin
    CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dLOG));      // out

    // 11) Run per image: quantize -> conv -> head -> read -> argmax
    size_t g=1;
    int correct=0;
    double sum_conv=0.0, sum_head=0.0, sum_tot=0.0;
    double min_tot=1e100, max_tot=0.0;
    std::cout << std::fixed << std::setprecision(3);

    for(int n=0; n<N; ++n){
      const uint8_t* raw = &Xraw[n*(H0*W0)];

      // Normalize to [0,1] then quantize to int8
      for(int i=0;i<H0*W0;++i){
        float xf = float(raw[i]) / 255.0f;
        int qv = int(std::round(xf / X0_SCALE)) + X0_ZP;
        if(qv < -128) qv = -128; if(qv > 127) qv = 127;
        x_q[i] = (int8_t)qv;
      }

      // H2D input
      CL_CHECK(clEnqueueWriteBuffer(q, dX, CL_TRUE, 0, sizeof(int8_t)*H0*W0, x_q.data(), 0, nullptr, nullptr));

      // conv
      cl_event e0;
      CL_CHECK(clEnqueueNDRangeKernel(q, k_conv, 1, nullptr, &g, nullptr, 0, nullptr, &e0));
      CL_CHECK(clFinish(q));

      // head
      cl_event e1;
      CL_CHECK(clEnqueueNDRangeKernel(q, k_head, 1, nullptr, &g, nullptr, 0, nullptr, &e1));
      CL_CHECK(clFinish(q));

      // D2H logits
      CL_CHECK(clEnqueueReadBuffer(q, dLOG, CL_TRUE, 0, sizeof(int8_t)*FC_O, y_logits.data(), 0, nullptr, nullptr));

      // Timings
      cl_ulong c0s, c0e, c1s, c1e;
      clGetEventProfilingInfo(e0, CL_PROFILING_COMMAND_START, sizeof(c0s), &c0s, nullptr);
      clGetEventProfilingInfo(e0, CL_PROFILING_COMMAND_END,   sizeof(c0e), &c0e, nullptr);
      clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof(c1s), &c1s, nullptr);
      clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_END,   sizeof(c1e), &c1e, nullptr);
      clReleaseEvent(e0); clReleaseEvent(e1);

      double ms_conv = double(c0e - c0s)/1e6;
      double ms_head = double(c1e - c1s)/1e6;
      double ms_tot  = ms_conv + ms_head;
      sum_conv += ms_conv; sum_head += ms_head; sum_tot += ms_tot;
      if(ms_tot < min_tot) min_tot = ms_tot;
      if(ms_tot > max_tot) max_tot = ms_tot;

      // Argmax on int8 logits
      int pred = int(std::max_element(y_logits.begin(), y_logits.end()) - y_logits.begin());
      bool ok = (pred == int(Lall[n]));
      if (ok) ++correct;

      std::cout << "["<<std::setw(3)<<n<<"] pred="<<pred
                << " label="<<int(Lall[n])
                << " t_conv="<<ms_conv<<" ms"
                << " t_head="<<ms_head<<" ms"
                << " t_total="<<ms_tot <<" ms -> " << (ok ? "OK" : "FAIL") << "\n";
    }

    double mean_conv = (N>0 ? sum_conv / N : 0.0);
    double mean_head = (N>0 ? sum_head / N : 0.0);
    double mean_tot  = (N>0 ? sum_tot  / N : 0.0);
    std::cout << "\n[RESULT] Accuracy = " << (100.0 * correct / N)
              << "% ("<<correct<<"/"<<N<<")\n";
    std::cout << "[RESULT] Mean times: conv=" << mean_conv
              << " ms, head=" << mean_head
              << " ms, total=" << mean_tot
              << " ms (min=" << min_tot << " ms, max=" << max_tot << " ms)\n";

    // Cleanup
    clReleaseMemObject(dWc0); clReleaseMemObject(dBc0);
    clReleaseMemObject(dWfc1); clReleaseMemObject(dB1);
    clReleaseMemObject(dWfc2); clReleaseMemObject(dB2);
    clReleaseMemObject(dX);  clReleaseMemObject(dY13); clReleaseMemObject(dLOG);
    clReleaseKernel(k_conv); clReleaseKernel(k_head);
    clReleaseProgram(prg); clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
  }catch(const std::exception& e){
    std::cerr<<"Exception: "<<e.what()<<"\n"; return 1;
  }
}