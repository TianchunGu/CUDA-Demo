#include <cmath>  // For std::log2, std::ceil, std::abs, std::isinf, std::isnan
#include <iostream>
#include <limits>  // For std::numeric_limits

/**
 * @brief 严格模拟 MATLAB 的 nextpow2 函数处理浮点数的行为。
 *
 * 该函数计算满足 2^p >= |x| 的最小整数 p。
 *
 * @param x 输入的 double 类型数值。
 * @return double 类型的结果，精确模拟 MATLAB 的返回值，包括对特殊值的处理。
 */
double matlab_nextpow2(double x) {
  // 1. 处理 NaN (非数值)
  // MATLAB: nextpow2(NaN) -> NaN
  if (std::isnan(x)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // 2. 处理 Inf (无穷大)
  // MATLAB: nextpow2(Inf) -> Inf
  // MATLAB: nextpow2(-Inf) -> Inf
  if (std::isinf(x)) {
    return std::numeric_limits<double>::infinity();
  }

  // 3. 处理 0
  // MATLAB: nextpow2(0) -> -Inf
  if (x == 0.0) {
    return -std::numeric_limits<double>::infinity();
  }

  // 4. 标准计算流程
  // 使用 std::abs 处理负数，然后取 log2，最后向上取整
  return std::ceil(std::log2(std::abs(x)));
}

int main() {
  // --- 测试案例 ---
  std::cout << "严格模拟 MATLAB nextpow2 函数\n" << std::endl;

  // 普通数值
  std::cout << "nextpow2(1023)    = " << matlab_nextpow2(1023)
            << " (MATLAB: 10)" << std::endl;
  std::cout << "nextpow2(1024)    = " << matlab_nextpow2(1024)
            << " (MATLAB: 10)" << std::endl;
  std::cout << "nextpow2(1025)    = " << matlab_nextpow2(1025)
            << " (MATLAB: 11)" << std::endl;
  std::cout << "nextpow2(-50)     = " << matlab_nextpow2(-50) << " (MATLAB: 6)"
            << std::endl;

  // 小于 1 的数值
  std::cout << "nextpow2(0.6)     = " << matlab_nextpow2(0.6) << " (MATLAB: -0)"
            << std::endl;  // C++可能显示-0，但数值上等于0
  std::cout << "nextpow2(0.5)     = " << matlab_nextpow2(0.5) << " (MATLAB: -1)"
            << std::endl;
  std::cout << "nextpow2(0.25)    = " << matlab_nextpow2(0.25)
            << " (MATLAB: -2)" << std::endl;
  std::cout << "nextpow2(0.24)    = " << matlab_nextpow2(0.24)
            << " (MATLAB: -2)" << std::endl;

  std::cout << "\n--- 特殊值测试 ---\n" << std::endl;

  // 测试 0
  std::cout << "nextpow2(0.0)     = " << matlab_nextpow2(0.0)
            << " (MATLAB: -Inf)" << std::endl;

  // 测试无穷大
  double pos_inf = std::numeric_limits<double>::infinity();
  double neg_inf = -std::numeric_limits<double>::infinity();
  std::cout << "nextpow2(Inf)     = " << matlab_nextpow2(pos_inf)
            << " (MATLAB: Inf)" << std::endl;
  std::cout << "nextpow2(-Inf)    = " << matlab_nextpow2(neg_inf)
            << " (MATLAB: Inf)" << std::endl;

  // 测试 NaN
  double q_nan = std::numeric_limits<double>::quiet_NaN();
  std::cout << "nextpow2(NaN)     = " << matlab_nextpow2(q_nan)
            << " (MATLAB: NaN)" << std::endl;

  return 0;
}