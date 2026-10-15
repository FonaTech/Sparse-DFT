#pragma once

#include "sdft_core.hpp"
#include <array>
#include <vector>

namespace sdft {

// Independent periodic point-charge Ewald evaluator. Lattice vectors and
// fractional coordinates are in bohr; charge is in units of |e|. The
// implementation matches the neutralizing-background convention used by the
// SCF ionic term: the electronic G=0 term is omitted and the ionic background
// correction is retained. Forces are -dE/dr in Hartree/bohr and stress is
// dE/d epsilon / volume in Hartree/bohr^3 for fixed fractional coordinates.
struct IonicEwaldAtom {
  Vec3 fractional{};
  double charge{};
};

struct IonicEwaldResult {
  double energy_hartree{};
  std::vector<Vec3> forces_hartree_per_bohr;
  std::array<Vec3, 3> stress_hartree_per_bohr3{};
};

IonicEwaldResult evaluate_ionic_ewald(
    const std::array<Vec3, 3>& lattice_bohr,
    const std::vector<IonicEwaldAtom>& atoms,
    double eta_bohr_inv = 0.55,
    int real_cutoff = 8,
    int reciprocal_cutoff = 16);

} // namespace sdft
