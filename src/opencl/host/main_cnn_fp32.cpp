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

#define CL_CHECK(x) do{ cl_int _e=(x); if(_e!=CL_SUCCESS){ \
  std::cerr << "OpenCL err " << _e << " @ " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(1);} }while(0)

// -------- Utilities --------

template<typename T>
std::vector<T> read_bin(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    if (nbytes % sizeof(T)) throw std::runtime_error("Unexpected size in: " + path);
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
    return clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          nbytes, const_cast<void*>(host), err);
}

// -------- Main --------

int main(int argc, char** argv){
    try{
        // Args:
        // 1: aocx
        // 2: images_u8.bin (N*784 bytes, raw MNIST)
        // 3: labels.bin     (N bytes)
        // 4: Wfceights_dir    (weights/cnn_small)
        const std::string aocx_path = (argc>1? argv[1] : "opencl/kernels/cnn_small_fp32.aocx");
        const std::string imgs_bin  = (argc>2? argv[2] : "opencl/data/test_images_u8_10.bin");
        const std::string labs_bin  = (argc>3? argv[3] : "opencl/data/test_labels_10.bin");
        const std::string wdir      = (argc>4? argv[4] : "opencl/weights/cnn_small");

        // Shapes
        const int H0=28, W0=28, C0=1;
        const int C1=16, H1=13, W1=13;                 // after conv+pool
        const int FC_IN = C1*H1*W1;                    // 2704
        const int FC_M  = 16;
        const int FC_O  = 10;

        // 1) Load weights
        // conv: [16,1,3,3], [16]
        // fc1 : [16,2704], [16]
        // fc2 : [10,16],   [10]
        auto Wc0  = read_bin<float>(wdir + "/conv0_W.bin");
        auto bc0  = read_bin<float>(wdir + "/conv0_b.bin");
        auto Wfc1 = read_bin<float>(wdir + "/fc1_W.bin");
        auto bfc1 = read_bin<float>(wdir + "/fc1_b.bin");
        auto Wfc2 = read_bin<float>(wdir + "/fc2_W.bin");
        auto bfc2 = read_bin<float>(wdir + "/fc2_b.bin");

        if ((int)Wc0.size()!=C1*C0*3*3 || (int)bc0.size()!=C1 ||
            (int)Wfc1.size()!=FC_M*FC_IN || (int)bfc1.size()!=FC_M ||
            (int)Wfc2.size()!=FC_O*FC_M  || (int)bfc2.size()!=FC_O){
            std::cerr << "[ERR] Wfceight sizes do not match the network.\n"; return 1;
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
        if((int)Lall.size()!=N){
            std::cerr<<"[ERR] labels count ("<<Lall.size()<<") != images ("<<N<<")\n"; return 1;
        }
        std::cout<<"[INFO] Batch: "<<N<<" images (uint8, not normalized).\n";

        // 3) OpenCL setup (profiling queue)
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

        // 4) Load .aocx and build
        auto aocx_vec = read_bytes_file(aocx_path);
        const unsigned char* bins[] = { aocx_vec.data() };
        size_t lens[] = { aocx_vec.size() };
        cl_int binst=0;
        cl_program prg = clCreateProgramWithBinary(ctx, 1, &dev, lens, bins, &binst, &err); CL_CHECK(err);
        CL_CHECK(clBuildProgram(prg, 0, nullptr, "", nullptr, nullptr));

        // 5) Kernels
        cl_kernel k_conv = clCreateKernel(prg, "cnn16_conv_relu_pool", &err); CL_CHECK(err);
        cl_kernel k_head = clCreateKernel(prg, "cnn16_head_fc",       &err); CL_CHECK(err);

        // 6) Constant buffers (weights/bias)
        cl_mem dWc0 = make_ro_buffer(ctx, Wc0.size()*sizeof(float), Wc0.data(), &err); CL_CHECK(err);
        cl_mem dBc0 = make_ro_buffer(ctx, bc0.size()*sizeof(float), bc0.data(), &err); CL_CHECK(err);
        cl_mem dW1  = make_ro_buffer(ctx, Wfc1 .size()*sizeof(float), Wfc1 .data(), &err); CL_CHECK(err);
        cl_mem dB1  = make_ro_buffer(ctx, bfc1 .size()*sizeof(float), bfc1 .data(), &err); CL_CHECK(err);
        cl_mem dW2  = make_ro_buffer(ctx, Wfc2 .size()*sizeof(float), Wfc2 .data(), &err); CL_CHECK(err);
        cl_mem dB2  = make_ro_buffer(ctx, bfc2 .size()*sizeof(float), bfc2 .data(), &err); CL_CHECK(err);

        // 7) Activation buffers
        std::vector<float> x_f32(H0*W0), logits(FC_O);
        cl_mem dX    = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  sizeof(float)*H0*W0,    nullptr, &err); CL_CHECK(err); // [1,28,28]
        cl_mem dY13  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float)*C1*H1*W1, nullptr, &err); CL_CHECK(err); // [16,13,13]
        cl_mem dLOG  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*FC_O,     nullptr, &err); CL_CHECK(err); // [10]

        // 8) Set static kernel args
        // conv
        int a=0;
        CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dWc0));
        CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dBc0));
        CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dX));
        CL_CHECK(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &dY13));
        // head
        a=0;
        CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dW1));
        CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dB1));
        CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dW2));
        CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dB2));
        CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dY13)); // xin = conv output
        CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dLOG));

        // 9) Run per image: save raw PGM, normalize, run both kernels, gather timings
        size_t g=1;
        int correct=0;
        double sum_ms_conv=0.0, sum_ms_head=0.0, sum_ms_total=0.0;
        double min_ms_tot=1e100, max_ms_tot=0.0;

        std::cout << std::fixed << std::setprecision(3);
        for(int n=0; n<N; ++n){
            const uint8_t* raw = &Xraw[n*(H0*W0)];
            
            // Normalize to [0,1]
            for(int i=0;i<H0*W0;++i) x_f32[i] = float(raw[i]) / 255.0f;

            // H2D input
            CL_CHECK(clEnqueueWriteBuffer(q, dX, CL_TRUE, 0, sizeof(float)*H0*W0, x_f32.data(), 0, nullptr, nullptr));

            // conv kernel
            cl_event e0;
            CL_CHECK(clEnqueueNDRangeKernel(q, k_conv, 1, nullptr, &g, nullptr, 0, nullptr, &e0));
            CL_CHECK(clFinish(q));

            // head kernel
            cl_event e1;
            CL_CHECK(clEnqueueNDRangeKernel(q, k_head, 1, nullptr, &g, nullptr, 0, nullptr, &e1));
            CL_CHECK(clFinish(q));

            // D2H logits
            CL_CHECK(clEnqueueReadBuffer(q, dLOG, CL_TRUE, 0, sizeof(float)*FC_O, logits.data(), 0, nullptr, nullptr));

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

            sum_ms_conv  += ms_conv;
            sum_ms_head  += ms_head;
            sum_ms_total += ms_tot;
            if(ms_tot < min_ms_tot) min_ms_tot = ms_tot;
            if(ms_tot > max_ms_tot) max_ms_tot = ms_tot;

            int pred = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            bool ok = (pred == int(Lall[n]));
            if (ok) ++correct;

            std::cout << "["<< std::setw(3) << n << "] "
                      << "pred=" << pred << " label=" << int(Lall[n])
                      << "  t_conv=" << ms_conv << " ms"
                      << "  t_head=" << ms_head << " ms"
                      << "  t_total=" << ms_tot  << " ms"
                      << "  -> " << (ok ? "OK" : "FAIL")
                      << "\n";
        }

        double mean_conv  = (N>0 ? sum_ms_conv  / N : 0.0);
        double mean_head  = (N>0 ? sum_ms_head  / N : 0.0);
        double mean_total = (N>0 ? sum_ms_total / N : 0.0);

        std::cout << "\n[RESULT] Accuracy = " << (100.0 * correct / N)
                  << "%  ("<<correct<<"/"<<N<<")\n";
        std::cout << "[RESULT] Mean times: conv=" << mean_conv
                  << " ms, head=" << mean_head
                  << " ms, total=" << mean_total
                  << " ms  (min=" << min_ms_tot << " ms, max=" << max_ms_tot << " ms)\n";

        // Cleanup
        clReleaseMemObject(dWc0); clReleaseMemObject(dBc0);
        clReleaseMemObject(dW1);  clReleaseMemObject(dB1);
        clReleaseMemObject(dW2);  clReleaseMemObject(dB2);
        clReleaseMemObject(dX);   clReleaseMemObject(dY13); clReleaseMemObject(dLOG);
        clReleaseKernel(k_conv);  clReleaseKernel(k_head);
        clReleaseProgram(prg);    clReleaseCommandQueue(q); clReleaseContext(ctx);
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Exception: "<<e.what()<<"\n"; return 1;
    }
}