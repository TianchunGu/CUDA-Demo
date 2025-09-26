#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                         \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// DBF 核：输入为四子阵按行拼接的大矩阵 concat( (4R) x C )
// 输出为 R x C 的三路：和通道、方位差、俯仰差
/**
 * @brief 在 GPU 上执行数字波束形成(DBF)中的和差波束合成。
 *
 * 该 Kernel 从一个由四路子阵(A, B, C, D)数据拼接而成的大数组 `concat`
 * 中读取数据，
 * 并行计算出和通道(Sum)、方位差通道(Azimuth)和俯仰差通道(Elevation)。
 *
 * @param concat    指向拼接后的四路输入数据的指针 [输入]。总尺寸为 (4*R) x C。
 * @param R         单个子阵列的行数。
 * @param C         单个子阵列的列数。
 * @param out_sum   指向和通道结果的输出指针 [输出]。尺寸为 R x C。
 * @param out_az    指向方位差通道结果的输出指针 [输出]。尺寸为 R x C。
 * @param out_el    指向俯仰差通道结果的输出指针 [输出]。尺寸为 R x C。
 */
__global__ void dbf_abcd_concat_rows(const cufftComplex* __restrict__ concat,
                                     int R,
                                     int C,
                                     cufftComplex* __restrict__ out_sum,
                                     cufftComplex* __restrict__ out_az,
                                     cufftComplex* __restrict__ out_el) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;  // 列
  int iy = blockIdx.y * blockDim.y + threadIdx.y;  // 行
  if (ix >= C || iy >= R)
    return;  // 越界保护 [34]

  // 四个子阵各自的行起始索引（按行拼接）
  int rowA = iy;
  int rowB = iy + R;
  int rowC = iy + R * 2;
  int rowD = iy + R * 3;

  // 线性索引（行主序）
  int idxA = rowA * C + ix;
  int idxB = rowB * C + ix;
  int idxC = rowC * C + ix;
  int idxD = rowD * C + ix;

  // 读取四路元素（复数）
  float2 a = reinterpret_cast<const float2*>(concat)[idxA];
  float2 b = reinterpret_cast<const float2*>(concat)[idxB];
  float2 c = reinterpret_cast<const float2*>(concat)[idxC];
  float2 d = reinterpret_cast<const float2*>(concat)[idxD];

  // 和通道 S = A + B + C + D
  float2 s;
  s.x = a.x + b.x + c.x + d.x;
  s.y = a.y + b.y + c.y + d.y;

  // 方位差 Az = (A + C) - (B + D)
  float2 az;
  az.x = (a.x + c.x) - (b.x + d.x);
  az.y = (a.y + c.y) - (b.y + d.y);

  // 俯仰差 El = (A + B) - (C + D)
  float2 el;
  el.x = (a.x + b.x) - (c.x + d.x);
  el.y = (a.y + b.y) - (c.y + d.y);

  // 写回结果
  int outIdx = iy * C + ix;
  reinterpret_cast<float2*>(out_sum)[outIdx] = s;
  reinterpret_cast<float2*>(out_az)[outIdx] = az;
  reinterpret_cast<float2*>(out_el)[outIdx] = el;
}

int main() {
  // 设备端输入：四子阵按行拼接后的矩阵，形状 (4R) x C
  int R = /* 每个子阵的行数 */;
  int C = /* 每个子阵的列数 */;
  size_t bytes_concat = sizeof(cufftComplex) * (size_t)(4 * R) * (size_t)C;

  cufftComplex* d_concat = nullptr;
  CHECK_CUDA(cudaMalloc(&d_concat, bytes_concat));  // 分配设备内存 [2]

  // 设备端输出：三路 R x C
  size_t bytes_out = sizeof(cufftComplex) * (size_t)R * (size_t)C;
  cufftComplex* d_sum = nullptr;
  cufftComplex* d_az = nullptr;
  cufftComplex* d_el = nullptr;
  CHECK_CUDA(cudaMalloc(&d_sum, bytes_out));  // 分配设备内存 [2]
  CHECK_CUDA(cudaMalloc(&d_az, bytes_out));   // 分配设备内存 [2]
  CHECK_CUDA(cudaMalloc(&d_el, bytes_out));   // 分配设备内存 [2]

  // ... 此处省略将数据准备并拷贝到 d_concat 的过程（如需从主机拷贝可用
  // cudaMemcpyHostToDevice）[2]
  // 若前一脉冲压缩步骤已在设备端得到按行拼接结果，直接传入 d_concat
  // 即可，无需回主机。

  // 设置执行配置：二维网格 + 二维块，采用向上取整覆盖全部元素 [4][34][5][21]
  dim3 block(32, 8);
  dim3 grid((C + block.x - 1) / block.x, (R + block.y - 1) / block.y);

  // 发射核函数（保持结果在 GPU 显存，不回拷）[2][30]
  dbf_abcd_concat_rows<<<grid, block>>>(d_concat, R, C, d_sum, d_az, d_el);
  CHECK_CUDA(cudaGetLastError());  // 发射后错误检查，及早发现配置问题 [24]

  // ... 此处不执行 cudaMemcpyDeviceToHost，结果 d_sum/d_az/d_el
  // 保留在设备端以便后续处理 [2]

  // 资源清理（若后续还需继续使用，可不立即释放）
  // CHECK_CUDA(cudaFree(d_concat));
  // CHECK_CUDA(cudaFree(d_sum));
  // CHECK_CUDA(cudaFree(d_az));
  // CHECK_CUDA(cudaFree(d_el));
  return 0;
}