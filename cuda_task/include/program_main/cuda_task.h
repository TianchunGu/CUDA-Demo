#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <util/utils.cuh>

namespace cuda_task {
__global__ void kernel_1(void);

__global__ void kernel_2(int N);

void run_vector_add();

}  // namespace cuda_task