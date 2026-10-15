#include "pbe_xc.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  auto uniform = sdft::pbe_unpolarized(0.5, 0.0);
  auto graded = sdft::pbe_unpolarized(0.5, 0.2);
  assert(std::isfinite(uniform.energy_density));
  assert(std::isfinite(graded.energy_density));
  assert(uniform.exchange_energy_density < 0.0);
  assert(std::abs(graded.energy_density - uniform.energy_density) > 1e-12);
  assert(std::abs(graded.energy_density - (-0.32588649164864247)) < 1e-12);

  const double dn = 1e-5;
  const double dgrad = 1e-5;
  const double fd_density = (sdft::pbe_unpolarized(0.5 + dn, 0.2).energy_density -
                             sdft::pbe_unpolarized(0.5 - dn, 0.2).energy_density) / (2.0 * dn);
  const double fd_gradient = (sdft::pbe_unpolarized(0.5, 0.2 + dgrad).energy_density -
                               sdft::pbe_unpolarized(0.5, 0.2 - dgrad).energy_density) / (2.0 * dgrad);
  assert(std::abs(fd_density - graded.exchange_potential - graded.correlation_potential) < 1e-8);
  assert(std::abs(fd_gradient - graded.gradient_squared_coefficient * 0.2) < 1e-8);
  // The PBE linear-response constraint cancels the exchange/correlation
  // sigma derivative at zero gradient; avoid subtracting O(n^-4/3) terms.
  for(double n:{1e-10,1e-8,.01,.5,20.})assert(sdft::pbe_unpolarized(n,0).gradient_squared_coefficient==0.);
  // Nitrogen PS radial tail can be clipped to 1e-99 with a nonzero mesh
  // gradient. The LDA tail drops the reduced gradients here; no 0/0 is permitted.
  for(double n:{1e-99,1e-60,1e-20,1e-10})for(double g:{0.,.166195}){
    const auto radial=sdft::pbe_unpolarized_lda_tail(n,g);
    assert(std::isfinite(radial.energy_density));assert(std::isfinite(radial.exchange_potential));
    assert(std::isfinite(radial.correlation_potential));assert(radial.gradient_squared_coefficient==0.);
  }
  std::cout << "sdft_pbe_smoke PASS\n";
}
