#pragma once

namespace sdft {

struct PbeXcPoint {
  double energy_density{};  // Hartree / bohr^3
  double energy_per_particle{};
  double exchange_energy_density{};
  double correlation_energy_density{};
  double exchange_potential{};
  double correlation_potential{};
  double gradient_squared_coefficient{};  // 2 * dE_xc/d|grad(n)|^2
};

PbeXcPoint pbe_unpolarized(double density, double gradient_magnitude);
// Low-density convention used for matched VASP references: reduced gradients
// s=t=0 for n<=1e-10, i.e. LDA exchange and PW92 correlation (Perdew & Wang,
// PRB 45, 13244 (1992)) without forming vanishing GGA ratios.
PbeXcPoint pbe_unpolarized_lda_tail(double density, double gradient_magnitude);

}
