// host/main_cnn.cpp
#include <CL/cl.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ===== Utilidades simples =====
template<typename T>
std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("No se puede abrir: " + path); }
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<T> buf(nbytes / sizeof(T));
    f.read(reinterpret_cast<char*>(buf.data()), nbytes);
    return buf;
}

std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("No se puede abrir: " + path); }
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(nbytes);
    f.read(reinterpret_cast<char*>(buf.data()), nbytes);
    return buf;
}

#define CL_CHECK(x) do{ cl_int _err = (x); if(_err!=CL_SUCCESS){ \
  std::cerr<<"OpenCL error "<<_err<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

// Carga PGM (P5) 8-bit y devuelve float32 normalizado [0,1], 28x28.
// Si falla, devuelve vector vacío.
std::vector<float> load_pgm28x28_as_float(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::string magic; f >> magic;
    if (magic != "P5") return {};
    int w,h,maxv; f >> w >> h >> maxv; f.get(); // consumir salto de línea
    if (w!=28 || h!=28 || maxv<=0) return {};
    std::vector<unsigned char> pix(28*28);
    f.read(reinterpret_cast<char*>(pix.data()), pix.size());
    std::vector<float> out(28*28);
    for (int i=0;i<28*28;++i) out[i] = float(pix[i]) / float(maxv);
    return out;
}

// ====== Main ======
int main(int argc, char** argv) {
    try {
        // Parámetros y rutas
        const std::string aocx_path   = (argc>1 ? argv[1] : "cnn_fp32.aocx");
        const std::string img_path    = (argc>2 ? argv[2] : "images/test.pgm");
        const std::string weights_dir = (argc>3 ? argv[3] : "weights");

        // Dimensiones fijas (tu CNN)
        const int H0=28, W0=28, C0=1;
        const int C1=32, H1=13, W1=13;    // tras conv0+pool
        const int C2=64, H2=5,  W2=5;     // tras conv1+pool
        const int FC_IN = C2*H2*W2;       // 1600
        const int FC_M  = 64;
        const int FC_O  = 10;

        // 1) Cargar pesos/bias
        auto Wc0 = read_bin<float>(weights_dir + "/conv0_W.bin");   // [32,1,3,3] = 288
        auto bc0 = read_bin<float>(weights_dir + "/conv0_b.bin");   // [32]
        auto Wc1 = read_bin<float>(weights_dir + "/conv1_W.bin");   // [64,32,3,3] = 18432
        auto bc1 = read_bin<float>(weights_dir + "/conv1_b.bin");   // [64]
        auto W3  = read_bin<float>(weights_dir + "/fc0_W.bin");     // [64,1600]
        auto b3  = read_bin<float>(weights_dir + "/fc0_b.bin");     // [64]
        auto W4  = read_bin<float>(weights_dir + "/fc1_W.bin");     // [10,64]
        auto b4  = read_bin<float>(weights_dir + "/fc1_b.bin");     // [10]

        // Sanity check tamaños
        if (Wc0.size() != size_t(C1*C0*3*3) || bc0.size()!=size_t(C1) ||
            Wc1.size() != size_t(C2*C1*3*3) || bc1.size()!=size_t(C2) ||
            W3.size()  != size_t(FC_M*FC_IN) || b3.size()!=size_t(FC_M) ||
            W4.size()  != size_t(FC_O*FC_M)  || b4.size()!=size_t(FC_O)) {
            std::cerr<<"Tamaños de pesos/bias inesperados. Revisa exportación.\n";
            return 1;
        }

        // 2) Cargar imagen 28x28 -> float32
        std::vector<float> x28 = load_pgm28x28_as_float(img_path);
        if (x28.empty()) {
            // como alternativa, si images/test.bin contiene 784 floats:
            try { x28 = read_bin<float>("images/test.bin"); }
            catch(...) {}
        }
        if (x28.size()!=size_t(H0*W0)) {
            std::cerr<<"No se pudo cargar imagen 28x28. Usando ceros.\n";
            x28.assign(H0*W0, 0.0f);
        }

        // 3) OpenCL: plataforma Intel FPGA, device, contexto, cola (con profiling)
        cl_uint np=0; CL_CHECK(clGetPlatformIDs(0,nullptr,&np));
        std::vector<cl_platform_id> plats(np); CL_CHECK(clGetPlatformIDs(np, plats.data(), nullptr));
        cl_platform_id plat = nullptr;
        for (auto p: plats) {
            size_t sz=0; clGetPlatformInfo(p, CL_PLATFORM_NAME, 0,nullptr,&sz);
            std::string name(sz, '\0'); clGetPlatformInfo(p, CL_PLATFORM_NAME, sz, &name[0], nullptr);
            if (name.find("Intel(R) FPGA")!=std::string::npos) { plat=p; break; }
        }
        if (!plat) { plat = plats[0]; } // fallback

        cl_device_id dev; CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr));
        cl_int err=CL_SUCCESS;
        cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CL_CHECK(err);
        cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CL_CHECK(err);

        // 4) Programar la FPGA desde el host (opcional): cargar .aocx
        auto aocx = read_bytes(aocx_path);
        const unsigned char* bins[] = { aocx.data() };
        size_t bin_len[] = { aocx.size() };
        cl_int bin_status=0;
        cl_program prg = clCreateProgramWithBinary(ctx, 1, &dev, bin_len, bins, &bin_status, &err); CL_CHECK(err);
        CL_CHECK(clBuildProgram(prg, 0, nullptr, "", nullptr, nullptr));

        // 5) Crear kernels
        cl_kernel k_conv0 = clCreateKernel(prg, "cnn_conv0_relu_pool", &err); CL_CHECK(err);
        cl_kernel k_conv1 = clCreateKernel(prg, "cnn_conv1_relu_pool", &err); CL_CHECK(err);
        cl_kernel k_head  = clCreateKernel(prg, "cnn_head_fc",        &err); CL_CHECK(err);

        // 6) Buffers de pesos/bias
        auto make_ro = size_t nbytes, const void* host{
            return clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbytes, const_cast<void*>(host), &err);
        };
        cl_mem dWc0 = make_ro(Wc0.size()*sizeof(float), Wc0.data());
        cl_mem dbc0 = make_ro(bc0.size()*sizeof(float), bc0.data());
        cl_mem dWc1 = make_ro(Wc1.size()*sizeof(float), Wc1.data());
        cl_mem dbc1 = make_ro(bc1.size()*sizeof(float), bc1.data());
        cl_mem dW3  = make_ro(W3 .size()*sizeof(float), W3 .data());
        cl_mem db3  = make_ro(b3 .size()*sizeof(float), b3 .data());
        cl_mem dW4  = make_ro(W4 .size()*sizeof(float), W4 .data());
        cl_mem db4  = make_ro(b4 .size()*sizeof(float), b4 .data());

        // 7) Buffers de activaciones
        cl_mem dx    = make_ro(x28.size()*sizeof(float), x28.data());        // [1,28,28] lineal
        cl_mem dy13  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float)*C1*H1*W1, nullptr, &err); CL_CHECK(err);
        cl_mem dy5   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float)*C2*H2*W2, nullptr, &err); CL_CHECK(err);
        cl_mem dlog  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*FC_O,   nullptr, &err); CL_CHECK(err);

        // 8) Set args y ejecutar kernels
        auto setargs_conv0 = &{
            int a=0;
            CL_CHECK(clSetKernelArg(k_conv0, a++, sizeof(cl_mem), &dWc0));
            CL_CHECK(clSetKernelArg(k_conv0, a++, sizeof(cl_mem), &dbc0));
            CL_CHECK(clSetKernelArg(k_conv0, a++, sizeof(cl_mem), &dx));
            CL_CHECK(clSetKernelArg(k_conv0, a++, sizeof(cl_mem), &dy13));
        };
        auto setargs_conv1 = &{
            int a=0;
            CL_CHECK(clSetKernelArg(k_conv1, a++, sizeof(cl_mem), &dWc1));
            CL_CHECK(clSetKernelArg(k_conv1, a++, sizeof(cl_mem), &dbc1));
            CL_CHECK(clSetKernelArg(k_conv1, a++, sizeof(cl_mem), &dy13));
            CL_CHECK(clSetKernelArg(k_conv1, a++, sizeof(cl_mem), &dy5));
        };
        auto setargs_head = &{
            int a=0;
            CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dW3));
            CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &db3));
            CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dW4));
            CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &db4));
            CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dy5));   // ya es "flatten": [64*5*5]
            CL_CHECK(clSetKernelArg(k_head, a++, sizeof(cl_mem), &dlog));
        };

        setargs_conv0();
        setargs_conv1();
        setargs_head();

        size_t g = 1; // single work-item kernels

        cl_event e0,e1,e2;
        CL_CHECK(clEnqueueNDRangeKernel(q, k_conv0, 1, nullptr, &g, nullptr, 0, nullptr, &e0));
        CL_CHECK(clEnqueueNDRangeKernel(q, k_conv1, 1, nullptr, &g, nullptr, 0, nullptr, &e1));
        CL_CHECK(clEnqueueNDRangeKernel(q, k_head , 1, nullptr, &g, nullptr, 0, nullptr, &e2));
        CL_CHECK(clFinish(q));

        // 9) Leer logits y argmax
        std::vector<float> logits(FC_O);
        CL_CHECK(clEnqueueReadBuffer(q, dlog, CL_TRUE, 0, sizeof(float)*FC_O, logits.data(), 0, nullptr, nullptr));

        int pred = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::cout << "Predicción CNN: " << pred << "\n";

        // 10) (opcional) tiempos por kernel
        auto ns = [&](cl_event ev){ cl_ulong t0,t1; clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0lptr);
                                    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
                                    return (t1 - t0)/1e6; };
        std::cout << "t(conv0+pool) = " << ns(e0) << " ms\n";
        std::cout << "t(conv1+pool) = " << ns(e1) << " ms\n";
        std::cout << "t(head FC)    = " << ns(e2) << " ms\n";

        // Limpieza mínima (omito releases repetitivos por brevedad)
        clReleaseEvent(e0); clReleaseEvent(e1); clReleaseEvent(e2);
        clReleaseMemObject(dWc0); clReleaseMemObject(dbc0); clReleaseMemObject(dWc1); clReleaseMemObject(dbc1);
        clReleaseMemObject(dW3);  clReleaseMemObject(db3);  clReleaseMemObject(dW4);  clReleaseMemObject(db4);
        clReleaseMemObject(dx); clReleaseMemObject(dy13); clReleaseMemObject(dy5); clReleaseMemObject(dlog);
        clReleaseKernel(k_conv0); clReleaseKernel(k_conv1); clReleaseKernel(k_head);
        clReleaseProgram(prg); clReleaseCommandQueue(q); clReleaseContext(ctx);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Excepción: " << e.what() << "\n";
        return 1;
    }
}