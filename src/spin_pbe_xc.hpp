#pragma once
#include <array>

namespace sdft {
// Collinear PBE with PW92 spin interpolation, atomic units. Potentials are
// partial derivatives at fixed gradients; the full Euler potential is
// density_potential[s] - div(gradient_response[s]). Core is supplied equally
// to the two density channels by the caller, never counted as valence charge.
struct SpinPbePoint {
  double energy_density{}, exchange_energy_density{}, correlation_energy_density{};
  std::array<double,2> density_potential{};
  std::array<std::array<double,3>,2> gradient_response{};
};
SpinPbePoint pbe_collinear(double up, double down,
    const std::array<double,3>& gradient_up,
    const std::array<double,3>& gradient_down, bool lda_low_density_tail=false);
}
