#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>

// Dimensiones
const int IMG_H = 28;
const int IMG_W = 28;

const int C1 = 16;
const int OUT_H = 13;
const int OUT_W = 13;

const int FC_IN = C1 * OUT_H * OUT_W; // 2704

// -------- Utils --------
template<typename T>
std::vector<T> read_bin(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t nbytes = size_t(f.tellg());
    f.seekg(0, std::ios::beg);

    if(nbytes % sizeof(T))
        throw std::runtime_error("Bad size: " + path);

    std::vector<T> v(nbytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), nbytes);
    return v;
}

// -------- Reorder FC1 (igual que OpenCL host) --------
std::vector<float>
reorder_fc1_columns_nhwc_to_nchw(const std::vector<float>& W1_src,
                                 int C, int H, int W, int Out)
{
    int In = C*H*W;
    std::vector<float> W1_dst(Out * In);

    for(int c=0;c<C;++c)
    for(int h=0;h<H;++h)
    for(int w=0;w<W;++w){
        int i_nchw = ((c * H) + h) * W + w;
        int i_nhwc = ((h * W) + w) * C + c;

        for(int o=0;o<Out;++o)
            W1_dst[o * In + i_nchw] = W1_src[o * In + i_nhwc];
    }

    return W1_dst;
}

// -------- Forward CNN --------
void cnn_forward(
    const std::vector<float>& Wc0,
    const std::vector<float>& bc0,
    const std::vector<float>& W1,
    const std::vector<float>& b1,
    const std::vector<float>& W2,
    const std::vector<float>& b2,
    const float* x,
    float* logits
){
    float conv_out[FC_IN];

    // ---- Conv + ReLU + MaxPool ----
    for(int oc=0; oc<C1; ++oc){
        for(int pr=0; pr<OUT_H; ++pr){
            for(int pc=0; pc<OUT_W; ++pc){

                float vmax = -1e30f;

                for(int dr=0; dr<2; ++dr){
                    for(int dc=0; dc<2; ++dc){

                        int orow = 2*pr + dr;
                        int ocol = 2*pc + dc;

                        float acc = bc0[oc];

                        for(int kr=0; kr<3; ++kr){
                            for(int kc=0; kc<3; ++kc){

                                int ir = orow + kr;
                                int ic = ocol + kc;

                                float w = Wc0[oc*9 + kr*3 + kc];
                                float xv = x[ir*IMG_W + ic];

                                acc += w * xv;
                            }
                        }

                        // ReLU
                        if(acc < 0.0f) acc = 0.0f;

                        if(acc > vmax) vmax = acc;
                    }
                }

                conv_out[oc*(OUT_H*OUT_W) + pr*OUT_W + pc] = vmax;
            }
        }
    }

    // ---- FC1 ----
    float a1[16];

    for(int o=0;o<16;++o){
        float acc = b1[o];
        for(int i=0;i<FC_IN;++i){
            acc += W1[o*FC_IN+i] * conv_out[i];
        }
        a1[o] = (acc > 0.0f) ? acc : 0.0f;
    }

    // ---- FC2 ----
    for(int o=0;o<10;++o){
        float acc = b2[o];
        for(int i=0;i<16;++i){
            acc += W2[o*16+i] * a1[i];
        }
        logits[o] = acc;
    }
}

// -------- Main --------
int main(int argc, char** argv){

    std::string imgs = (argc>1? argv[1] : "opencl/data/test_images_u8.bin");
    std::string labs = (argc>2? argv[2] : "opencl/data/test_labels.bin");
    std::string wdir = (argc>3? argv[3] : "opencl/weights/cnn_fp32");

    auto Wc0 = read_bin<float>(wdir + "/conv0_W.bin");
    auto bc0 = read_bin<float>(wdir + "/conv0_b.bin");
    auto W1  = read_bin<float>(wdir + "/fc1_W.bin");
    auto b1  = read_bin<float>(wdir + "/fc1_b.bin");
    auto W2  = read_bin<float>(wdir + "/fc2_W.bin");
    auto b2  = read_bin<float>(wdir + "/fc2_b.bin");

    // Reordenar como en OpenCL
    W1 = reorder_fc1_columns_nhwc_to_nchw(W1, C1, OUT_H, OUT_W, 16);

    auto Xraw = read_bin<uint8_t>(imgs);
    auto Lall = read_bin<uint8_t>(labs);

    int N = Lall.size();

    std::vector<float> x(IMG_H*IMG_W);
    std::vector<float> logits(10);

    int correct = 0;
    double total_ms = 0.0;

    for(int n=0;n<N;++n){

        // Normalize
        for(int i=0;i<IMG_H*IMG_W;++i)
            x[i] = float(Xraw[n*IMG_H*IMG_W + i]) / 255.0f;

        auto t0 = std::chrono::high_resolution_clock::now();

        cnn_forward(Wc0,bc0,W1,b1,W2,b2,x.data(),logits.data());

        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        total_ms += ms;

        int pred = std::max_element(logits.begin(), logits.end()) - logits.begin();

        if(pred == Lall[n]) correct++;
    }

    std::cout << "Accuracy: " << (100.0 * correct / N) << "%\n";
    std::cout << "Mean time: " << (total_ms / N) << " ms\n";

    return 0;
}