#include "ionic_ewald.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
double& component(sdft::Vec3& r, int i) { return i == 0 ? r.x : i == 1 ? r.y : r.z; }
double component(const sdft::Vec3& r, int i) { return i == 0 ? r.x : i == 1 ? r.y : r.z; }
}

int main() {
  using namespace sdft;
  const std::array<Vec3, 3> lattice{{{8.0, 0.4, 0.1}, {0.7, 9.0, -0.2}, {-0.1, 0.3, 10.0}}};
  const std::vector<IonicEwaldAtom> atoms{{{0.13, 0.21, 0.08}, 2.0},
      {{0.47, 0.61, 0.73}, -1.0}, {{0.69, 0.24, 0.45}, 0.4}};
  const auto ref = evaluate_ionic_ewald(lattice, atoms);
  const double dx = 1e-5;
  const double det = determinant(lattice);
  const std::array<Vec3, 3> inverse{{cross(lattice[1], lattice[2]),
      cross(lattice[2], lattice[0]), cross(lattice[0], lattice[1])}};
  double force_error = 0.0, stress_error = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) for (int axis = 0; axis < 3; ++axis) {
    auto plus = atoms, minus = atoms;
    for (int f = 0; f < 3; ++f) {
      component(plus[i].fractional, f) += dx * component(inverse[f], axis) / det;
      component(minus[i].fractional, f) -= dx * component(inverse[f], axis) / det;
    }
    const double ep = evaluate_ionic_ewald(lattice, plus).energy_hartree;
    const double em = evaluate_ionic_ewald(lattice, minus).energy_hartree;
    const double fd = -(ep - em) / (2.0 * dx);
    force_error = std::max(force_error, std::abs(fd - component(ref.forces_hartree_per_bohr[i], axis)));
  }
  for (int alpha = 0; alpha < 3; ++alpha) for (int beta = 0; beta < 3; ++beta) {
    auto plus = lattice, minus = lattice;
    for (int v = 0; v < 3; ++v) {
      component(plus[v], alpha) += dx * component(lattice[v], beta);
      component(minus[v], alpha) -= dx * component(lattice[v], beta);
    }
    const double ep = evaluate_ionic_ewald(plus, atoms).energy_hartree;
    const double em = evaluate_ionic_ewald(minus, atoms).energy_hartree;
    const double fd = (ep - em) / (2.0 * dx) / std::abs(det);
    stress_error = std::max(stress_error, std::abs(fd - component(ref.stress_hartree_per_bohr3[alpha], beta)));
  }
  double force_sum = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    double total = 0.0;
    for (const auto& f : ref.forces_hartree_per_bohr) total += component(f, axis);
    force_sum = std::max(force_sum, std::abs(total));
  }
  const auto other_eta = evaluate_ionic_ewald(lattice, atoms, 0.43);
  const double eta_energy_error = std::abs(ref.energy_hartree - other_eta.energy_hartree);
  std::cout << "ionic_ewald_smoke force_error=" << force_error << " stress_error=" << stress_error
            << " force_sum=" << force_sum << " eta_energy_error=" << eta_energy_error << "\n";
  if (force_error > 5e-9 || stress_error > 5e-10 || force_sum > 1e-12 || eta_energy_error > 3e-8) return 1;
}
