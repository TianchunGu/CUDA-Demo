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

void __cufftCHECK(cufftResult err, const char* file, const int line) {
  if (err != CUFFT_SUCCESS) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE:%d, DETAIL:%s\n", err, cufftGetErrorString(err));
    exit(1);
  }
}

const char* cufftGetErrorString(cufftResult err) {
  switch (err) {
    case CUFFT_SUCCESS:
      return "The cuFFT operation was successful";
    case CUFFT_INVALID_PLAN:
      return "cuFFT was passed an invalid plan handle";
    case CUFFT_ALLOC_FAILED:
      return "cuFFT failed to allocate GPU or CPU memory";
    case CUFFT_INVALID_TYPE:
      return "The cuFFT type provided is unsupported";
    case CUFFT_INVALID_VALUE:
      return "User specified an invalid pointer or parameter";
    case CUFFT_INTERNAL_ERROR:
      return "Driver or internal cuFFT library error";
    case CUFFT_EXEC_FAILED:
      return "Failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED:
      return "The cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE:
      return "User specified an invalid transform size";
    case CUFFT_UNALIGNED_DATA:
      return "Not currently in use";
    case CUFFT_INVALID_DEVICE:
      return "Execution of a plan was on different GPU than plan creation";
    case CUFFT_NO_WORKSPACE:
      return "No workspace has been provided prior to plan execution";
    case CUFFT_NOT_IMPLEMENTED:
      return "Function does not implement functionality for given parameters";
    case CUFFT_NOT_SUPPORTED:
      return "Operation is not supported for parameters given";
    default:
      return "Unknown error code";
  }
}

}  // namespace util