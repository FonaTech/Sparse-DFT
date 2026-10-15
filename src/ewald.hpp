#pragma once
#include "sdft_core.hpp"
#include <vector>

namespace sdft {
struct EwaldResult {
  double energy{};  // Hartree
  std::vector<Vec3> forces;  // Hartree/bohr, negative position gradient
  std::array<Vec3, 3> stress{};  // dE/d(strain)/volume, Hartree/bohr^3
  double eta{};
  double tail_tolerance{};
};
// Neutralizing uniform background; no electronic G=0 Coulomb term. Reciprocal
// vectors include 2*pi. The fixed-eta derivative includes the background term;
// eta is free to adapt between calls because the complete sum is eta invariant.
EwaldResult evaluate_ewald(const std::array<Vec3, 3>& lattice_bohr,
    const std::vector<Vec3>& fractional, const std::vector<double>& charges,
    double eta = 0.0, double tail_tolerance = 1e-14);
}
