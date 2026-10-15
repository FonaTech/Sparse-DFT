#include "paw_onecenter.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

int main() {
  sdft::PawNumericCache c;
  c.radial_rmax = 1.0;
  c.radial_grid = {0.05, 0.15, 0.30, 0.50, 0.75, 1.0};
  c.radial_points = static_cast<unsigned>(c.radial_grid.size());
  c.pseudo_wavefunctions = {{0.05,0.14,0.25,0.31,0.28,0.20}};
  c.ae_wavefunctions = {{0.05,0.15,0.29,0.38,0.34,0.24}};
  std::vector<int> l{0}, np{1};
  Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(1,1);
  auto z = sdft::paw_one_center_spherical(c, zero, l, np, 2);
  if (std::abs(z.electron_count) > 1e-12 || !std::isfinite(z.spherical_dij_ev(0,0)))
    return 2;
  Eigen::MatrixXd onsite(1,1); onsite(0,0) = 1.0;
  auto r = sdft::paw_one_center_spherical(c, onsite, l, np, 2);
  if (!std::isfinite(r.double_counting_ae_ev) || !std::isfinite(r.double_counting_ps_ev) ||
      !std::isfinite(r.spherical_dij_ev(0,0)) || r.electron_count <= 0.0)
    return 3;
  std::cout << "paw_onecenter_smoke PASS electron_count=" << r.electron_count << "\n";
  return 0;
}
