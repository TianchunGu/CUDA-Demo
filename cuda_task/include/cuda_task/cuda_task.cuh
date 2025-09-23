#pragma once
#include <cuda_runtime.h>  // 包含 cuda_runtime.h 以便使用 cudaGetErrorString
#include <cufft.h>         // 包含 cufft.h 以便使用 cufftResult
#include <iostream>
#include <util/utils.cuh>

namespace cuda_task {
__global__ void kernel_1(void);

__global__ void kernel_2(int N);

void perform_cufft_transform();

void run_vector_add();

}  // namespace cuda_task