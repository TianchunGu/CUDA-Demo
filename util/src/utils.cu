#include <util/utils.cuh>

namespace util {
static void __cudaCheck(cudaError_t err, const char* file, const int line) {
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}

static void __kernelCheck(const char* file, const int line) {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}
}  // namespace util