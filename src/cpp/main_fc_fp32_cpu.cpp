#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>

#define H1_DIM 64
#define H2_DIM 32

template<typename T>
std::vector<T> read_bin(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<T> v(nbytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), nbytes);
    return v;
}

void fc_forward(
    const std::vector<float>& W0, const std::vector<float>& b0,
    const std::vector<float>& W1, const std::vector<float>& b1,
    const std::vector<float>& W2, const std::vector<float>& b2,
    const float* x,
    float* y,
    int in_dim,
    int out_dim
){
    float a1[H1_DIM];
    float a2[H2_DIM];

    for(int o=0;o<H1_DIM;++o){
        float acc = b0[o];
        for(int i=0;i<in_dim;++i)
            acc += W0[o*in_dim+i] * x[i];
        a1[o] = acc > 0.0f ? acc : 0.0f;
    }

    for(int o=0;o<H2_DIM;++o){
        float acc = b1[o];
        for(int i=0;i<H1_DIM;++i)
            acc += W1[o*H1_DIM+i] * a1[i];
        a2[o] = acc > 0.0f ? acc : 0.0f;
    }

    for(int o=0;o<out_dim;++o){
        float acc = b2[o];
        for(int i=0;i<H2_DIM;++i)
            acc += W2[o*H2_DIM+i] * a2[i];
        y[o] = acc;
    }
}

int main(int argc, char** argv){
    try{
        std::string imgs_bin = (argc>1? argv[1] : "opencl/data/test_images_u8.bin");
        std::string labs_bin = (argc>2? argv[2] : "opencl/data/test_labels.bin");
        std::string wdir     = (argc>3? argv[3] : "opencl/weights/fc_fp32");

        const int in_dim=784, out_dim=10;

        std::vector<float> W0 = read_bin<float>(wdir + "/fc0_W.bin");
        std::vector<float> b0 = read_bin<float>(wdir + "/fc0_b.bin");
        std::vector<float> W1 = read_bin<float>(wdir + "/fc1_W.bin");
        std::vector<float> b1 = read_bin<float>(wdir + "/fc1_b.bin");
        std::vector<float> W2 = read_bin<float>(wdir + "/fc2_W.bin");
        std::vector<float> b2 = read_bin<float>(wdir + "/fc2_b.bin");

        std::vector<uint8_t> Xraw = read_bin<uint8_t>(imgs_bin);
        std::vector<uint8_t> Lall = read_bin<uint8_t>(labs_bin);

        int N = Lall.size();

        std::vector<float> x(in_dim), y(out_dim);

        int correct=0;
        double total_ms=0;

        for(int n=0;n<N;++n){
            for(int i=0;i<in_dim;++i)
                x[i] = float(Xraw[n*in_dim+i]) / 255.0f;

            auto t0 = std::chrono::high_resolution_clock::now();

            fc_forward(W0,b0,W1,b1,W2,b2,x.data(),y.data(),in_dim,out_dim);

            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();

            total_ms += ms;

            int pred = std::max_element(y.begin(), y.end()) - y.begin();
            if(pred == Lall[n]) correct++;
        }

        std::cout << "Accuracy: " << (100.0*correct/N) << "%\n";
        std::cout << "Mean: " << (total_ms/N)
                  << "\n";

    }catch(std::exception& e){
        std::cerr << e.what() << "\n";
        return 1;
    }
}