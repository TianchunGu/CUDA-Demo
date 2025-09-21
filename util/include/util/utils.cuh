#pragma once
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHECK(call) __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call) __kernelCheck(__FILE__, __LINE__)

namespace util {

void __cudaCheck(cudaError_t err, const char* file, const int line);

void __kernelCheck(const char* file, const int line);

}  // namespace util