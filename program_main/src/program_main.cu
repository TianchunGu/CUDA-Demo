#include <iostream>
#include <util/utils.cuh>  // 包含 util 库的头文件

__global__ void saxpy_kernel(float* y, float* x, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  float h_x[N], h_y[N];
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(i);
    h_y[i] = static_cast<float>(i * 2);
  }

  float *d_x, *d_y;

  // 分配设备内存，并使用宏进行错误检查
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

  // 将数据从主机复制到设备
  CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  // 执行内核
  saxpy_kernel<<<blocks, threads>>>(d_y, d_x, 2.0f);
  // 检查内核执行是否出错
  KERNEL_CHECK();

  // 将结果从设备复制回主机
  CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

  // 打印部分结果以验证
  for (int i = 0; i < 10; ++i) {
    std::cout << "h_y[" << i << "] = " << h_y[i] << std::endl;
  }

  // 释放设备内存
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  std::cout << "Program finished successfully." << std::endl;

  return 0;
}