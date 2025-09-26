// pulse_compression_rows.cu
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                         \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CHECK_CUFFT(call)                                                    \
  do {                                                                       \
    cufftResult res = (call);                                                \
    if (res != CUFFT_SUCCESS) {                                              \
      fprintf(stderr, "cuFFT error %d at %s:%d\n", res, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// 频域逐点乘（按行广播系数）：freq[row, k] *= coeff[k]
/**
 * @brief 在 GPU 上并行地将多行频域数据与一个系数向量进行元素级复数乘法。
 * * 这个 Kernel 将 `freq` 数组中的每一行都与 `coeff` 向量进行逐元素的复数乘法。
 * `freq` 数组被视为一个 `rows` 行、`fftnum` 列的矩阵。
 * `coeff` 数组是一个长度为 `fftnum` 的向量。
 * 操作是就地的，结果直接写入 `freq` 数组。
 * * @param freq 指向待处理的频域数据数组的指针 [输入/输出]。
 * @param coeff 指向系数数组的指针 [输入]。
 * @param fftnum `freq` 数组的行宽，也是 `coeff` 数组的长度。
 * @param rows `freq` 数组的行数。
 */
__global__ void mul_rows_by_coeff(cufftComplex* __restrict__ freq,
                                  const cufftComplex* __restrict__ coeff,
                                  int fftnum,
                                  int rows) {
  int total = fftnum * rows;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = i; idx < total; idx += stride) {
    int k = idx % fftnum;
    float2 a = reinterpret_cast<const float2*>(coeff)[k];
    float2 b = reinterpret_cast<float2*>(freq)[idx];
    // 复数乘法 (b *= a)
    float2 c;
    c.x = b.x * a.x - b.y * a.y;
    c.y = b.x * a.y + b.y * a.x;
    reinterpret_cast<float2*>(freq)[idx] = c;
  }
}

// 就地归一化（乘以 1/N）——cuFFT 不归一化，需用户缩放 [20]
/**
 * @brief 在 GPU 上并行地对一维复数数组进行就地缩放。
 *
 * 这个 Kernel 会遍历一个包含 'total' 个元素的复数数组 'data'，
 * 并将每个元素的实部和虚部分别乘以一个标量 'scale'。
 * 操作是就地的，结果直接写回 'data' 数组。
 *
 * @param data  指向待缩放的复数数组的指针 [输入/输出]。
 * @param total 数组中复数元素的总个数。
 * @param scale 用于缩放的浮点数乘子。
 */
__global__ void scale_inplace(cufftComplex* __restrict__ data,
                              int total,
                              float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = i; idx < total; idx += stride) {
    float2 v = reinterpret_cast<float2*>(data)[idx];
    v.x *= scale;
    v.y *= scale;
    reinterpret_cast<float2*>(data)[idx] = v;
  }
}

// 按行截取有效区间：从 start 开始取 samplenumber 点到结果矩阵 pc[row, j]
/**
 * @brief 在 GPU 上并行地从一个二维数组的每一行中裁剪出一个子片段。
 *
 * 这个 Kernel 将源数组 `src` (逻辑上为 `rows` 行, `fftnum` 列) 中的每一行
 * 从 `start` 列开始，复制 `samplenumber` 个元素到目标数组 `dst` 中。
 * `dst` 数组的尺寸将是 `rows` 行, `samplenumber` 列。
 *
 * @param src           指向源数据数组的指针 [输入]。
 * @param dst           指向目标数据数组的指针 [输出]。
 * @param fftnum        源数组中每一行的宽度。
 * @param rows          需要处理的总行数。
 * @param start         裁剪的起始列索引。
 * @param samplenumber  裁剪的样本数量（即目标数组的行宽）。
 */
__global__ void crop_rows(const cufftComplex* __restrict__ src,
                          cufftComplex* __restrict__ dst,
                          int fftnum,
                          int rows,
                          int start,
                          int samplenumber) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int j0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  for (int j = j0; j < samplenumber; j += blockDim.x * gridDim.x) {
    int src_idx = row * fftnum + (start + j);
    int dst_idx = row * samplenumber + j;
    reinterpret_cast<float2*>(dst)[dst_idx] =
        reinterpret_cast<const float2*>(src)[src_idx];
  }
}

int main() {
  // ---------------- 参数设置（根据实际数据替换） ----------------
  const int PulseNumber = 1024;   // 行数（脉冲数）
  const int fftnum = 4096;        // 每行 FFT 长度（零填充后列数）
  const int samplenumber = 2048;  // 截取长度
  const int number =
      512;  // 截取起点（0 基，MATLAB 的 number+1 对应这里的 number）

  // ---------------- 分配与准备数据（示例） ----------------
  // 设备端输入矩阵 Echo，尺寸 PulseNumber x fftnum（已零填充）
  cufftComplex* d_echo = nullptr;
  CHECK_CUDA(cudaMalloc(&d_echo, sizeof(cufftComplex) * PulseNumber * fftnum));

  // 设备端匹配滤波器频域系数 coeff_fft，长度 fftnum
  cufftComplex* d_coeff = nullptr;
  CHECK_CUDA(cudaMalloc(&d_coeff, sizeof(cufftComplex) * fftnum));

  // 设备端结果矩阵 pc，尺寸 PulseNumber x samplenumber
  cufftComplex* d_pc = nullptr;
  CHECK_CUDA(
      cudaMalloc(&d_pc, sizeof(cufftComplex) * PulseNumber * samplenumber));

  // 如需在主机侧初始化并拷贝，可按需使用 cudaMemcpy（输入/输出需 GPU
  // 可见内存）[9] 此处省略主机数据准备与拷贝代码，假设 d_echo 与 d_coeff
  // 已填充完毕。

  // ---------------- 创建 batched 1D FFT 计划（按行） ----------------
  // 使用 cufftPlanMany，设置
  // rank=1，n[0]=fftnum，batch=PulseNumber；行内步幅=1，行间距离=fftnum
  // [10][25]
  cufftHandle plan;
  int n[1] = {fftnum};
  int istride = 1, ostride = 1;
  int idist = fftnum, odist = fftnum;
  int* inembed = nullptr;  // 连续布局可设为 nullptr
  int* onembed = nullptr;
  CHECK_CUFFT(cufftPlanMany(&plan, /*rank=*/1, n, inembed, istride, idist,
                            onembed, ostride, odist, CUFFT_C2C,
                            /*batch=*/PulseNumber));  // [10][25]

  // ---------------- 前向 FFT（按行） ----------------
  // 就地执行：时域 -> 频域；方向参数为 CUFFT_FORWARD [3][8]
  CHECK_CUFFT(cufftExecC2C(plan, d_echo, d_echo, CUFFT_FORWARD));  // [3][8]

  // ---------------- 频域逐点乘匹配滤波器系数 ----------------
  {
    int total = PulseNumber * fftnum;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mul_rows_by_coeff<<<blocks, threads>>>(d_echo, d_coeff, fftnum,
                                           PulseNumber);
    CHECK_CUDA(cudaGetLastError());
  }
  // 上述流程（FFT → 点操作 →
  // IFFT）是常见的频域处理序列，无需在两次执行之间重排数据 [1]

  // ---------------- 逆向 FFT（按行） ----------------
  // 频域 -> 时域；就地执行；方向参数为 CUFFT_INVERSE [3][8][20]
  CHECK_CUFFT(cufftExecC2C(plan, d_echo, d_echo, CUFFT_INVERSE));  // [3][8][20]

  // ---------------- 设备端就地归一化（乘以 1/fftnum） ----------------
  // cuFFT 保留频率信息但不做幅度归一化，缩放留给用户按数据长度进行 [6][20]
  {
    int total = PulseNumber * fftnum;
    float invN = 1.0f / static_cast<float>(fftnum);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scale_inplace<<<blocks, threads>>>(d_echo, total, invN);
    CHECK_CUDA(cudaGetLastError());
  }

  // ---------------- 截取有效区间到结果矩阵（按行） ----------------
  // 每行从 number 起，取 samplenumber 点，写入 d_pc
  {
    dim3 block(32, 8);
    dim3 grid((samplenumber + block.x - 1) / block.x,
              (PulseNumber + block.y - 1) / block.y);
    crop_rows<<<grid, block>>>(d_echo, d_pc, fftnum, PulseNumber, number,
                               samplenumber);
    CHECK_CUDA(cudaGetLastError());
  }

  // （可选）如需取回主机侧验证：cudaMemcpy(..., cudaMemcpyDeviceToHost) [4][9]

  // ---------------- 清理资源 ----------------
  CHECK_CUFFT(cufftDestroy(plan));  // 计划可复用，完成后销毁 [29]
  CHECK_CUDA(cudaFree(d_echo));
  CHECK_CUDA(cudaFree(d_coeff));
  CHECK_CUDA(cudaFree(d_pc));
  return 0;
}