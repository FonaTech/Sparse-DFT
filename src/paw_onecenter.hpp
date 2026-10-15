#pragma once

#include "sdft_core.hpp"
#include <Eigen/Dense>
#include <vector>

namespace sdft {

/**
 * Density-dependent PAW one-centre correction.
 *
 * The projector density matrix is in the expanded (radial-projector,m)
 * ordering used by scf_core.  The routine evaluates a spherical reference
 * density on the POTCAR radial mesh, builds Hartree+PBE potentials for the AE
 * and PS densities, and projects their difference onto AE/PS partial waves.
 * It is deliberately a pure function so it can be cached and tested against
 * finite-difference derivatives independently from the plane-wave solver.
 */
struct PawOneCenterResult {
  Eigen::MatrixXd spherical_dij_ev;              // correction to D_ij, in eV
  double double_counting_ae_ev{};
  double double_counting_ps_ev{};
  double electron_count{};
  double max_potential_ev{};
  // Variational energy correction E_AE-E_PS for the supplied on-site
  // density matrix.  This is the quantity that must be combined with the
  // band-energy subtraction when spherical_dij_ev is inserted in H.
  double energy_difference_ev{};
  double potential_expectation_ev{};
};

PawOneCenterResult paw_one_center_spherical(const PawNumericCache& cache,
                                            const Eigen::MatrixXd& onsite,
                                            const std::vector<int>& channel_l,
                                            const std::vector<int>& channel_nproj,
                                            int spin_multiplicity = 1);

} // namespace sdft
