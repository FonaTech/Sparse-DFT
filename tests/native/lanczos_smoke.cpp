#include "pw_core.hpp"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  const std::size_t n = 8;
  std::vector<double> h(n * n, 0.0), s(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    h[i * n + i] = 0.5 + static_cast<double>(i);
    s[i * n + i] = 1.0 + 0.1 * static_cast<double>(i);
    if (i + 1 < n) h[i * n + i + 1] = h[(i + 1) * n + i] = -0.05;
  }
  const auto dense = sdft::solve_generalized_symmetric(h, s, n);
  const auto iterative = sdft::solve_generalized_symmetric_iterative(h, s, n, 3, 64, 1e-11);
  if (iterative.eigenvalues.size() != 3) return 2;
  for (std::size_t i = 0; i < 3; ++i) {
    if (std::abs(iterative.eigenvalues[i] - dense.eigenvalues[i]) > 1e-8) return 3;
  }
  std::cout << "lanczos_smoke PASS roots=3 max_error=1e-8\n";
  return 0;
}
