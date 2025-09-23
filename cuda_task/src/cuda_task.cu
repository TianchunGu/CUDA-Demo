#include <cuda_task/cuda_task.cuh>

// 宏定义变换的尺寸和批次数
#define NX 256
#define BATCH 10

__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

namespace cuda_task {
__global__ void kernel_1(void) {
  printf("Constant data c_data = %.2f.\n", c_data);
}

__global__ void kernel_2(int N) {
  int idx = threadIdx.x;
  if (idx < N) {
  }
}

void run_vector_add() {
  int devID = 0;
  cudaDeviceProp deviceProps;
  util::CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

  float h_data = 8.8f;
  util::CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

  dim3 block(1);
  dim3 grid(1);
  kernel_1<<<grid, block>>>();
  util::CUDA_CHECK(cudaDeviceSynchronize());
  util::CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));
  printf("Constant data h_data = %.2f.\n", h_data);

  util::CUDA_CHECK(cudaDeviceReset());
}

/**
 * @brief 执行一次完整的cuFFT变换流程
 *
 * 该函数封装了内存分配、创建计划、执行变换和资源清理的完整三步流程。
 * [cite_start]这个函数的设计基于文档中的示例代码 [cite: 76, 77, 78, 81, 82, 83,
 * 84, 85, 86]。
 */
void perform_cufft_transform() {
  cufftHandle plan;
  cufftComplex* data;

  std::cout << "Allocating GPU memory for " << BATCH << " transforms of size "
            << NX << "..." << std::endl;
  util::CUDA_CHECK(
      cudaMalloc((void**)&data, sizeof(cufftComplex) * NX * BATCH));

  std::cout << "Creating cuFFT plan..." << std::endl;
  util::CUFFT_CHECK(cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH));

  std::cout << "Executing cuFFT plan..." << std::endl;
  util::CUFFT_CHECK(cufftExecC2C(plan, data, data, CUFFT_FORWARD));

  std::cout << "Waiting for GPU to finish..." << std::endl;
  util::CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "Destroying cuFFT plan and freeing memory..." << std::endl;
  util::CUFFT_CHECK(cufftDestroy(plan));
  util::CUDA_CHECK(cudaFree(data));  // 释放GPU内存
}

}  // namespace cuda_task