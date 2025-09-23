#pragma once
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

// 用于检查返回 cudaError_t 的 CUDA Runtime API 函数
#define CUDA_CHECK(call) __cudaCheck(call, __FILE__, __LINE__)
// 用于检查异步核函数启动（<<<...>>>）后是否发生错误
#define LAST_KERNEL_CHECK(call) __kernelCheck(__FILE__, __LINE__)
// 用于检查返回 cufftResult 的 cuFFT 库函数
#define CUFFT_CHECK(call) __cufftCHECK(call, __FILE__, __LINE__)
namespace util {
const char* cufftGetErrorString(cufftResult err);
void __cudaCheck(cudaError_t err, const char* file, const int line);

void __kernelCheck(const char* file, const int line);

void __cufftCHECK(cufftResult err, const char* file, const int line);

}  // namespace util