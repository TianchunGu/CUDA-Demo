#include <util/utils.cuh>

namespace util {
void __cudaCheck(cudaError_t err, const char* file, const int line) {
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}

void __kernelCheck(const char* file, const int line) {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}

void __cufftCheck(const char* file, const int line) {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}

}  // namespace util