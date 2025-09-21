#include <cuda_task/cuda_task.cuh>

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

}  // namespace cuda_task