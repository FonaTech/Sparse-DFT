#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sdft {

// Cubic interpolation with a natural right boundary and either a natural
// left boundary or a specified left derivative. Coefficients are built once;
// evaluating a table must not repeat the tridiagonal solve inside H assembly.
class TabulatedSpline {
 public:
  TabulatedSpline() = default;
  TabulatedSpline(std::vector<double> x, std::vector<double> y,
                  std::optional<double> left_derivative = std::nullopt)
      : x_(std::move(x)), y_(std::move(y)) {
    if (x_.size() != y_.size() || x_.size() < 2)
      throw std::invalid_argument("spline requires at least two matching samples");
    const auto n = x_.size();
    for (std::size_t i = 0; i < n; ++i)
      if (!std::isfinite(x_[i]) || !std::isfinite(y_[i]) ||
          (i && x_[i] <= x_[i - 1]))
        throw std::invalid_argument("spline samples must be finite and increasing");
    if (left_derivative && !std::isfinite(*left_derivative))
      throw std::invalid_argument("nonfinite spline boundary derivative");
    // Unknowns are the node slopes s_i (Hermite form). With h_i = x_{i+1}-x_i and
    // secant t_i = (y_{i+1}-y_i)/h_i, C2 continuity at an interior node gives
    //   h_i s_{i-1} + 2(h_{i-1}+h_i) s_i + h_{i-1} s_{i+1} = 3(h_i t_{i-1} + h_{i-1} t_i).
    // A zero second derivative at an end gives 2 s_0 + s_1 = 3 t_0 (left) and
    // s_{n-2} + 2 s_{n-1} = 3 t_{n-2} (right); a clamped left end fixes s_0.
    // The system is strictly diagonally dominant, so elimination needs no pivoting.
    std::vector<double> h(n - 1), secant(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
      h[i] = x_[i + 1] - x_[i];
      secant[i] = (y_[i + 1] - y_[i]) / h[i];
    }
    std::vector<double> lower(n, 0.0), diag(n, 0.0), upper(n, 0.0), load(n, 0.0);
    if (left_derivative) {
      diag[0] = 1.0;
      load[0] = *left_derivative;
    } else {
      diag[0] = 2.0;
      upper[0] = 1.0;
      load[0] = 3.0 * secant[0];
    }
    for (std::size_t i = 1; i + 1 < n; ++i) {
      lower[i] = h[i];
      diag[i] = 2.0 * (h[i - 1] + h[i]);
      upper[i] = h[i - 1];
      load[i] = 3.0 * (h[i] * secant[i - 1] + h[i - 1] * secant[i]);
    }
    lower[n - 1] = 1.0;
    diag[n - 1] = 2.0;
    load[n - 1] = 3.0 * secant[n - 2];
    for (std::size_t i = 1; i < n; ++i) {  // forward elimination
      const double ratio = lower[i] / diag[i - 1];
      diag[i] -= ratio * upper[i - 1];
      load[i] -= ratio * load[i - 1];
    }
    std::vector<double> slope(n);
    slope[n - 1] = load[n - 1] / diag[n - 1];
    for (std::size_t i = n - 1; i-- > 0;)  // back substitution
      slope[i] = (load[i] - upper[i] * slope[i + 1]) / diag[i];
    // Power-form coefficients about x_i of the Hermite cubic on [x_i, x_{i+1}].
    b_.resize(n - 1); c_.resize(n - 1); d_.resize(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
      b_[i] = slope[i];
      c_[i] = (3.0 * secant[i] - 2.0 * slope[i] - slope[i + 1]) / h[i];
      d_[i] = (slope[i] + slope[i + 1] - 2.0 * secant[i]) / (h[i] * h[i]);
    }
  }
  bool empty() const { return x_.empty(); }
  double value(double x) const {
    const auto i = interval(x);
    const double dx = x - x_[i];
    return y_[i] + dx * (b_[i] + dx * (c_[i] + dx * d_[i]));
  }
  double derivative(double x) const {
    const auto i = interval(x);
    const double dx = x - x_[i];
    return b_[i] + dx * (2.0 * c_[i] + 3.0 * dx * d_[i]);
  }
 private:
  std::size_t interval(double x) const {
    if (empty() || !std::isfinite(x)) throw std::invalid_argument("invalid spline evaluation");
    const auto it = std::upper_bound(x_.begin(), x_.end(), x);
    if (it == x_.begin()) return 0;
    return std::min(static_cast<std::size_t>(it - x_.begin() - 1), x_.size() - 2);
  }
  std::vector<double> x_, y_, b_, c_, d_;
};
} // namespace sdft
