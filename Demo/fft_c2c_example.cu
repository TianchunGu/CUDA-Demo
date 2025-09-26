// fft_c2c_example.cu
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                         \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CHECK_CUFFT(call)                                                    \
  do {                                                                       \
    cufftResult res = (call);                                                \
    if (res != CUFFT_SUCCESS) {                                              \
      fprintf(stderr, "cuFFT error %d at %s:%d\n", res, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

int main() {
  const int N = 8;

  // 准备主机端输入：简单的复数序列（实部为索引，虚部为0）
  std::vector<cufftComplex> h_in(N);
  for (int i = 0; i < N; ++i) {
    h_in[i].x = static_cast<float>(i);  // real
    h_in[i].y = 0.0f;                   // imag
  }

  // 设备端内存
  cufftComplex* d_data = nullptr;
  CHECK_CUDA(cudaMalloc(&d_data, sizeof(cufftComplex) * N));
  CHECK_CUDA(cudaMemcpy(d_data, h_in.data(), sizeof(cufftComplex) * N,
                        cudaMemcpyHostToDevice));

  // 创建 1D C2C FFT 计划
  cufftHandle plan;
  CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

  // 前向FFT（in-place）
  CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

  // 将频域数据拷回主机并打印
  std::vector<cufftComplex> h_freq(N);
  CHECK_CUDA(cudaMemcpy(h_freq.data(), d_data, sizeof(cufftComplex) * N,
                        cudaMemcpyDeviceToHost));
  printf("Forward FFT result:\n");
  for (int i = 0; i < N; ++i) {
    printf("k=%d: (%f, %f)\n", i, h_freq[i].x, h_freq[i].y);
  }

  // 逆向FFT（in-place）
  CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

  // 拷回并归一化（cuFFT 的逆变换未归一化，需要除以 N）
  std::vector<cufftComplex> h_back(N);
  CHECK_CUDA(cudaMemcpy(h_back.data(), d_data, sizeof(cufftComplex) * N,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    h_back[i].x /= N;
    h_back[i].y /= N;
  }

  printf("\nInverse FFT (normalized) result:\n");
  for (int i = 0; i < N; ++i) {
    printf("n=%d: (%f, %f)\n", i, h_back[i].x, h_back[i].y);
  }

  // 清理
  CHECK_CUFFT(cufftDestroy(plan));
  CHECK_CUDA(cudaFree(d_data));

  return 0;
}

// 编译命令：nvcc -std=c++17 fft_c2c_example.cu -lcufft -o fft_c2c_example

// ❯ nvcc -std=c++17 fft_c2c_example.cu -lcufft -o fft_c2c_example
// ❯ ./fft_c2c_example
// Forward FFT result:
// k=0: (28.000000, 0.000000)
// k=1: (-4.000000, 9.656855)
// k=2: (-4.000000, 4.000000)
// k=3: (-4.000000, 1.656854)
// k=4: (-4.000000, 0.000000)
// k=5: (-4.000000, -1.656854)
// k=6: (-4.000000, -4.000000)
// k=7: (-4.000000, -9.656854)

// Inverse FFT (normalized) result:
// n=0: (0.000000, 0.000000)
// n=1: (1.000000, 0.000000)
// n=2: (2.000000, 0.000000)
// n=3: (3.000000, -0.000000)
// n=4: (4.000000, 0.000000)
// n=5: (5.000000, -0.000000)
// n=6: (6.000000, -0.000000)
// n=7: (7.000000, 0.000000)