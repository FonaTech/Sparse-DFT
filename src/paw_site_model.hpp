#pragma once
#include "paw_framework.hpp"
#include "upf_beta_cache.hpp"
#include "paw_hubbard.hpp"
#include <functional>
#include <memory>

namespace sdft {
class PawCompensationOperator;
// Format adapters supply atomic data in one common atomic-unit convention.
// The SCF loop owns the smooth density, Hxc, occupations, metric solver and
// mixer. No adapter runs its own SCF or changes another atom's PAW dataset.
struct PawSiteClosure {
  double energy_hartree{};
  std::array<std::vector<std::complex<double>>,2> derivative_hartree;
};

struct PawSiteGeometry {
  PawFramework framework{PawFramework::Upf};
  std::string element, record_id;
  int numeric_cache_version{};
  bool projector_q_bohr{};
  std::vector<int> radial_channel_l;
  std::vector<std::complex<double>> atomic_onsite;
  // POTCAR has a separate EATOM/cutoff/core-XC ledger and an excluded G=0
  // ionic potential. UPF keeps its original local G=0 inside H instead.
  double zval{}, frozen_core_energy_hartree{}, reference_energy_hartree{}, excluded_local_g0_form_hartree{};
  std::vector<double> local, local_derivative, core, core_derivative, atomic_density;
  std::vector<ComplexPawOperator> bare_at_origin;
  std::vector<UpfBetaGeometry> beta_geometry;
  // Trace convention: rho_aug(G) = sum_ij P_ij Q_ji(G)/volume.
  std::vector<std::vector<std::complex<double>>> qform;
  std::vector<std::array<std::vector<std::complex<double>>,3>> qgradient;
  // Keep POTCAR's exact small multipole transform and [G,lm] geometry.
  // Construct only per-atom translation phases; do not expand [G,i,j].
  std::function<std::shared_ptr<PawCompensationOperator>(Vec3)> compact_compensation;
  std::function<PawSiteClosure(const std::array<std::vector<std::complex<double>>,2>&,int)> closure;
  std::function<std::vector<double>(int)> hubbard_overlap;
  std::size_t dimension() const { return bare_at_origin.at(0).projectors; }
};

std::shared_ptr<PawSiteGeometry> prepare_potcar_site_geometry(
    const ScfInput&, std::size_t atom,
    const std::vector<std::tuple<int,int,int>>& density_keys,
    const std::vector<Vec3>& reciprocal_vectors,
    const std::vector<std::vector<Vec3>>& k_vectors);
ComplexPawOperator prepare_potcar_bloch_block(const ScfInput&,std::size_t atom,
    const std::vector<Vec3>& vectors);
} // namespace sdft
