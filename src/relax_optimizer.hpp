#pragma once
#include <vector>
#include <cstddef>

namespace sdft {
struct LbfgsPair { std::vector<double> displacement, gradient_change; };
struct RelaxDirection {
  std::vector<double> step;
  std::size_t used_pairs{}, rejected_pairs{};
  double directional_derivative{};
  bool steepest_descent{};
};
// Two-loop recursion, O(ndof * history), with a Euclidean trust bound.
// The caller performs an actual energy line search before accepting a step.
RelaxDirection lbfgs_direction(const std::vector<double>& gradient,
                              const std::vector<LbfgsPair>& history,
                              double initial_scale, double max_step);
}
