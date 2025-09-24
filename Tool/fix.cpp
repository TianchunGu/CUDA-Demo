#include <cmath>  // For std::trunc, std::isinf, std::isnan
#include <iostream>
#include <limits>  // For std::numeric_limits

/**
 * @brief 严格模拟 MATLAB 的 fix 函数处理浮点数的行为。
 *
 * 该函数通过向零舍入来将输入值转换为整数值（但返回类型仍为 double）。
 * 这与 C++ 标准库中的 std::trunc 函数行为完全一致。
 *
 * @param x 输入的 double 类型数值。
 * @return double 类型的结果，即向零取整后的值。
 */
double matlab_fix(double x) {
  // std::trunc 函数的行为与 MATLAB 的 fix 函数完全相同，
  // 包括对正数、负数、零、Inf、-Inf 和 NaN 的处理。
  return std::trunc(x);
}

int main() {
  // --- 测试案例 ---
  std::cout << "严格模拟 MATLAB fix 函数\n" << std::endl;

  // 普通数值
  std::cout << "fix(3.14)      = " << matlab_fix(3.14) << " (MATLAB: 3)"
            << std::endl;
  std::cout << "fix(3.99)      = " << matlab_fix(3.99) << " (MATLAB: 3)"
            << std::endl;
  std::cout << "fix(-3.14)     = " << matlab_fix(-3.14) << " (MATLAB: -3)"
            << std::endl;
  std::cout << "fix(-3.99)     = " << matlab_fix(-3.99) << " (MATLAB: -3)"
            << std::endl;

  // 边界和整数值
  std::cout << "fix(5.0)       = " << matlab_fix(5.0) << " (MATLAB: 5)"
            << std::endl;
  std::cout << "fix(-5.0)      = " << matlab_fix(-5.0) << " (MATLAB: -5)"
            << std::endl;
  std::cout << "fix(0.0)       = " << matlab_fix(0.0) << " (MATLAB: 0)"
            << std::endl;
  std::cout << "fix(-0.0)      = " << matlab_fix(-0.0) << " (MATLAB: 0)"
            << std::endl;

  std::cout << "\n--- 特殊值测试 ---\n" << std::endl;

  // 测试无穷大
  double pos_inf = std::numeric_limits<double>::infinity();
  double neg_inf = -std::numeric_limits<double>::infinity();
  std::cout << "fix(Inf)       = " << matlab_fix(pos_inf) << " (MATLAB: Inf)"
            << std::endl;
  std::cout << "fix(-Inf)      = " << matlab_fix(neg_inf) << " (MATLAB: -Inf)"
            << std::endl;

  // 测试 NaN
  double q_nan = std::numeric_limits<double>::quiet_NaN();
  std::cout << "fix(NaN)       = " << matlab_fix(q_nan) << " (MATLAB: NaN)"
            << std::endl;

  return 0;
}

// fix 函数的功能是向零取整（"round towards zero"），即简单地去掉小数部分。

// MATLAB fix 函数行为分析
// 首先，我们明确 fix 函数在 MATLAB 中的具体行为：

// 正数：fix(3.7) 返回 3。它会舍弃小数部分。

// 负数：fix(-3.7) 返回 -3。同样是舍弃小数部分，向零靠近。

// 整数：fix(5.0) 返回 5。

// 零：fix(0.0) 返回 0。

// 无穷大 (Infinity)：

// fix(Inf) 返回 Inf。

// fix(-Inf) 返回 -Inf。

// 非数值 (NaN)：fix(NaN) 返回 NaN。

// 从 C/C++ 的角度看，MATLAB 的 fix 函数的行为与标准库 <cmath> 中的 trunc
// (truncate) 函数完全一致。trunc
// 函数就是用来截断数值，即移除小数部分，使其向零舍入。

// C++ 实现
// 利用 <cmath> 中的 trunc 函数，我们可以非常简单且精确地实现 fix
// 的功能。同时，为了保证严格模拟，我们还需要确保对 Inf 和 NaN
// 的处理是正确的。幸运的是，C++标准中的 trunc 函数对这些特殊值的处理行为与
// MATLAB 的 fix 函数是完全一致的。