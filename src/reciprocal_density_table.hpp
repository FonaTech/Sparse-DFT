#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sdft {
struct RadialDensitySample { double value{}, derivative{}; };

// Cubic through four equally spaced samples f(-1), f(0), f(1), f(2), evaluated
// at t in node units, normally 0 <= t <= 1 (four-point Lagrange formula).
// `slope` is df/dt; multiply by the inverse node spacing for df/dq.
struct FourNodeCubic { double value{}, slope{}; };

inline FourNodeCubic four_node_lagrange(double fm, double f0, double f1, double f2, double t) {
  const double a = t + 1.0, b = t, c = t - 1.0, d = t - 2.0;
  const double value = -fm * b * c * d / 6.0 + f0 * a * c * d / 2.0 -
      f1 * a * b * d / 2.0 + f2 * a * b * c / 6.0;
  const double slope = -fm * (c * d + b * d + b * c) / 6.0 + f0 * (c * d + a * d + a * c) / 2.0 -
      f1 * (b * d + a * d + a * b) / 2.0 + f2 * (b * c + a * c + a * b) / 6.0;
  return {value, slope};
}

// Scalar densities on the uniform reciprocal grid q_k = k*qmax/N.  At q = 0 the
// stored sample (the total charge) is returned exactly.  Elsewhere the cubic
// through nodes k-1..k+2 is used, with k >= 1 so that no parity ghost is needed
// (below q_1 the first four samples are extrapolated).  The last three samples
// bound the stencil, so q >= (N-3) qmax/N gives zero.  Projector form factors
// use the parity-extended stencil instead (projector_table_cubic in scf_core).
inline RadialDensitySample reciprocal_density_sample(const std::vector<double>& values,
                                                     double q, double qmax) {
  if(values.size()<4 || !std::isfinite(q) || !std::isfinite(qmax) || qmax<=0)
    throw std::invalid_argument("invalid reciprocal density table or coordinate");
  const double n=static_cast<double>(values.size()), scale=n/qmax;
  if(q<0 || q>=qmax*(1.-3./n))return {};
  if(q==0)return {values.front(),0.};
  const double u=q*scale;
  const auto k=std::max<std::size_t>(1,static_cast<std::size_t>(std::floor(u)));
  const auto cubic=four_node_lagrange(values[k-1],values[k],values[k+1],values[k+2],u-static_cast<double>(k));
  return {cubic.value,scale*cubic.slope};
}
} // namespace sdft
