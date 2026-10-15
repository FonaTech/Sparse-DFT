#include "pw_core.hpp"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
  std::array<sdft::Vec3, 3> reciprocal{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  auto g = sdft::enumerate_g_vectors(reciprocal, 30.0);
  assert(!g.empty() && g.front().h == 0 && g.front().k == 0 && g.front().l == 0);
  auto shifted = sdft::enumerate_g_vectors(reciprocal, 30.0, {0.5, 0.0, 0.0});
  assert(std::any_of(shifted.begin(), shifted.end(), [](const auto& gv) { return gv.h == -1 && gv.k == 0 && gv.l == 0; }));
  for (const auto& gv : shifted) assert(gv.kinetic_ev <= 30.0 + 1e-10);
  // An index precisely on the cutoff must survive compression in fixed-basis
  // mode, while its Cartesian vector and kinetic energy change physically.
  auto compressed = reciprocal;
  compressed[0].x *= 1.01;
  const double boundary_cutoff = 27.211386245988 / 2;
  const auto reference = sdft::enumerate_g_vectors(reciprocal, boundary_cutoff);
  const auto fixed = sdft::enumerate_g_vectors_fixed_reference(compressed, reciprocal, boundary_cutoff);
  const auto variable = sdft::enumerate_g_vectors(compressed, boundary_cutoff);
  if (fixed.size()!=reference.size() || variable.size()>=fixed.size()) return 3;
  bool above_cutoff=false;
  for (std::size_t i=0; i<fixed.size(); ++i) {
    const auto& a=reference[i]; const auto& b=fixed[i];
    if (a.h!=b.h || a.k!=b.k || a.l!=b.l || std::abs(b.cartesian.x-1.01*a.cartesian.x)>1e-14) return 4;
    above_cutoff |= b.kinetic_ev>boundary_cutoff;
  }
  if (!above_cutoff) return 5;
  const sdft::Vec3 kpoint{.23,.11,-.07};
  const auto bloch=sdft::enumerate_g_vectors_fixed_reference(compressed,reciprocal,30.,kpoint);
  for (const auto& gv : bloch) {
    const double x=gv.cartesian.x+1.01*kpoint.x, y=gv.cartesian.y+kpoint.y, z=gv.cartesian.z+kpoint.z;
    if (std::abs(gv.kinetic_ev-.5*(x*x+y*y+z*z)*27.211386245988)>1e-12) return 6;
  }
  const std::size_t n = 2;
  std::vector<double> h{1.0, 0.0, 0.0, 2.0};
  std::vector<double> s{1.0, 0.0, 0.0, 1.0};
  auto result = sdft::solve_generalized_symmetric(h, s, n);
  assert(result.eigenvalues.size() == n);
  assert(std::abs(result.eigenvalues[0] - 1.0) < 1e-12);
  assert(std::abs(result.eigenvalues[1] - 2.0) < 1e-12);
  sdft::PawProjectorOperator p{1, 2, {1.0, 2.0}, {3.0}, {0.5}};
  sdft::add_paw_low_rank_operator(h, s, n, p);
  assert(std::abs(h[0] - 4.0) < 1e-12 && std::abs(h[1] - 6.0) < 1e-12);
  std::cout << "sdft_pw_smoke PASS\n";
}
