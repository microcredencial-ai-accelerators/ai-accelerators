
// host/main_fc_fp32.cpp
#include <CL/cl.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

template<typename T>
std::vector<T> read_bin(const char* p){
  std::ifstream f(p, std::ios::binary); f.seekg(0,std::ios::end);
  size_t n=f.tellg(); f.seekg(0,std::ios::beg);
  std::vector<T> v(n/sizeof(T)); f.read((char*)v.data(), n); return v;
}
std::vector<unsigned char> read_bytes(const char* p){
  std::ifstream f(p,std::ios::binary); f.seekg(0,std::ios::end);
  size_t n=f.tellg(); f.seekg(0,std::ios::beg);
  std::vector<unsigned char> v(n); f.read((char*)v.data(), n); return v;
}

int main(){
  cl_int err; cl_uint np=0; clGetPlatformIDs(0,nullptr,&np);
  std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np, plats.data(), nullptr);

  cl_platform_id plat = nullptr; // buscar plataforma Intel FPGA
  for (auto p: plats){
    size_t sz=0; clGetPlatformInfo(p, CL_PLATFORM_NAME, 0,nullptr,&sz);
    std::string name(sz, '\0'); clGetPlatformInfo(p, CL_PLATFORM_NAME, sz, name.data(), nullptr);
    if (name.find("Intel(R) FPGA")!=std::string::npos) { plat=p; break; }
  }
  if (!plat){ std::cerr<<"Plataforma FPGA no encontrada\n"; return 1; }
  cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr);
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);

  // Cargar bitstream .aocx
  auto aocx = read_bytes("fc_fp32.aocx");
  const unsigned char* bins[] = { aocx.data() }; size_t lens[] = { aocx.size() }; cl_int binst;
  cl_program prg = clCreateProgramWithBinary(ctx, 1, &dev, lens, bins, &binst, &err);
  err = clBuildProgram(prg, 0, nullptr, "", nullptr, nullptr);
  cl_kernel krn = clCreateKernel(prg, "fc_mnist", &err);

  // Cargar pesos
  auto W0=read_bin<float>("weights/fc0_W.bin");
  auto b0=read_bin<float>("weights/fc0_b.bin");
  auto W1=read_bin<float>("weights/fc1_W.bin");
  auto b1=read_bin<float>("weights/fc1_b.bin");
  auto W2=read_bin<float>("weights/fc2_W.bin");
  auto b2=read_bin<float>("weights/fc2_b.bin");

  // Entrada (28x28 FP32 normalizada a [0,1])
  std::vector<float> x(784, 0.f); // TODO: cargar imagen
  std::vector<float> y(10, 0.f);

  // Buffers
  cl_mem dW0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*W0.size(), W0.data(), &err);
  cl_mem dB0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*b0.size(), b0.data(), &err);
  cl_mem dW1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*W1.size(), W1.data(), &err);
  cl_mem dB1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*b1.size(), b1.data(), &err);
  cl_mem dW2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*W2.size(), W2.data(), &err);
  cl_mem dB2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*b2.size(), b2.data(), &err);
  cl_mem dX  = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*x.size(),  x.data(),  &err);
  cl_mem dY  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*y.size(), nullptr, &err);

  // Args
  int a=0;
  clSetKernelArg(krn, a++, sizeof(cl_mem), &dW0); clSetKernelArg(krn, a++, sizeof(cl_mem), &dB0);
  clSetKernelArg(krn, a++, sizeof(cl_mem), &dW1); clSetKernelArg(krn, a++, sizeof(cl_mem), &dB1);
  clSetKernelArg(krn, a++, sizeof(cl_mem), &dW2); clSetKernelArg(krn, a++, sizeof(cl_mem), &dB2);
  clSetKernelArg(krn, a++, sizeof(cl_mem), &dX);
  clSetKernelArg(krn, a++, sizeof(cl_mem), &dY);

  size_t g=1; err = clEnqueueNDRangeKernel(q, krn, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
  clFinish(q);
  clEnqueueReadBuffer(q, dY, CL_TRUE, 0, sizeof(float)*y.size(), y.data(), 0, nullptr, nullptr);

  // Argmax
  int best = int(std::max_element(y.begin(), y.end()) - y.begin());
  std::cout << "Predicción (FC): " << best << "\n";

  // Limpieza omitida por brevedad...
  return 0;
}
