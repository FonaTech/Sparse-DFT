#pragma once
#include "atomic_units.hpp"

#include "sdft_core.hpp"
#include "density_functionals.hpp"
#include <array>
#include <complex>
#include <vector>
#include <cstddef>

namespace sdft {

// PAW one-centre augmentation data (Kresse & Joubert, PRB 59, 1758 (1999)).
// aug_moment(i,j,L) = int (phi_i phi_j - phi~_i phi~_j) r^L dr is the radial
// L-moment of the AE-PS partial-wave pair density.  compensation_r2[L][ir]
// stores g_L(r) r^2, the normalized compensation shape that multiplies the
// (L,M) moment to give the radial compensation density.
struct PawAugmentationData {
  std::vector<int> channel_l;
  int max_l{};
  double rmax{};
  double rdep{};
  std::vector<double> radial_grid;
  std::vector<double> aug_moment;
  // Overlap correction q_ij = aug_moment(i,j,0) for equal-l pairs, the term in
  // S = 1 + sum |p_i> q_ij <p_j|.  The AE norm truncated at the RDEP radius is
  // a different quantity and does not enter the generalized overlap.
  std::vector<double> overlap_correction;
  std::vector<std::vector<double>> compensation_r2;
  std::vector<std::array<double, 2>> compensation_q;
  std::vector<std::array<double, 2>> compensation_amplitudes;
  // Compensation-shape normalization.  false (default) uses the exact
  // continuous L-moment of the Bessel pair; true uses the discrete radial
  // Simpson weights below rmax, an older compatibility convention.  The flag
  // is kept so regression tests cannot confuse the two constructions.
  bool mesh_compensation_norm{false};

  std::size_t channel_count() const { return channel_l.size(); }
  std::size_t aug_moment_index(std::size_t i, std::size_t j, int l) const;
  double moment(std::size_t i, std::size_t j, int l) const;
  double overlap_moment(std::size_t i, std::size_t j) const;
};

// Build the complete radial PAW moment tensor and the compensation shapes
// g_L = a1 j_L(q1 r) + a2 j_L(q2 r), which vanish with zero slope at rmax and
// have unit L-moment, from the decoded POTCAR radial waves.
PawAugmentationData build_paw_augmentation(const PawNumericCache& cache,
                                           const std::vector<int>& channel_l,
                                           int max_l = -1);

// Contract a complex projector density matrix with aug_moment and the real
// Gaunt coefficients over all ordered channel pairs: the (L,M) moments of
// the compensation charge.
std::vector<std::complex<double>> paw_lm_moments_from_density(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension);

// Real and imaginary parts of the aug_moment-weighted (L,M) moments, summed
// over the channel upper triangle with off-diagonal pairs doubled.  The
// imaginary part uses the antisymmetric m<m' sign, so it keeps the part of a
// complex occupation matrix that the real charge discards.
std::vector<double> paw_lm_moments_real(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension);
std::vector<double> paw_lm_moments_imag(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension);

// Gaunt-projected (L,M) components per channel pair, before the aug_moment
// weight; the one-centre coordinate of the density mixer.  Blocks follow the
// channel upper triangle (i<=j), each covering L=|li-lj|..li+lj, and the
// vector is padded to expanded_dimension^2 to fill the occupation slot.
std::vector<double> paw_packed_channel_lm_real(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension);
std::vector<double> paw_packed_channel_lm_imag(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension);
std::vector<std::complex<double>> paw_packed_channel_lm_complex(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension);
std::vector<std::complex<double>> paw_density_from_packed_channel_lm(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& packed_channel_lm,
    std::size_t expanded_dimension);

// One non-zero real-harmonic Gaunt coefficient G(l1 m1, l2 m2 | L M) =
// \int Y_{l1 m1} Y_{l2 m2} Y_{L M} dOmega.  `inverse` maps an (L, M) moment back
// onto the (m1, m2) occupation entries (G divided by sum_{m1 m2} G^2).  Exposing
// the coefficients lets the angular convention be tested against quadrature.
struct PawGauntTerm {
  int L{};
  int M{};
  double value{};
  double inverse{};
};

std::vector<PawGauntTerm> paw_real_gaunt_terms(int l1, int l2, int m1, int m2);

// All coefficients up to lmax in compressed-row form.  Row r enumerates
// (l1, l2, m1, m2) with l1 outermost and m2 innermost; its terms occupy
// [row_offset[r], row_offset[r + 1]).  lm_index is L(L+1)+M.
struct PawGauntTable {
  int lmax{};
  std::vector<std::size_t> row_offset;
  std::vector<int> lm_index;
  std::vector<int> L;
  std::vector<double> value;
  std::vector<double> inverse;
};

PawGauntTable paw_real_gaunt_table(int lmax);

// Reconstruct compensation charge components on the radial mesh.  The outer
// index is the flattened (L,M) index L*L+M+L; values are densities multiplied
// by r^2, the radial convention used throughout the one-centre code.
std::vector<std::vector<std::complex<double>>> paw_compensation_density(
    const PawAugmentationData& augmentation,
    const std::vector<std::complex<double>>& lm_moments);

// Variational one-centre PAW Hartree/PBE-XC correction (reference-atom kinetic
// and frozen-core potentials remain the caller's responsibility). The cache
// normally contains Angstrom units; the optional unit flag is for validated
// callers with already converted radial arrays. lm_moments are returned in the
// archive's Angstrom^L convention in either case. The returned matrix is in the
// expanded (radial-projector,m) ordering used by the plane-wave Hamiltonian
// and is in eV.  It is built from the same current on-site density that is
// used for the AE/PS radial energies.
struct PawOneCenterClosureResult {
  std::vector<std::complex<double>> d_matrix_ev;
  std::vector<std::complex<double>> lm_moments;
  // `lm_moments` is the physical augmentation charge entering the radial
  // Hartree/XC functional.  This companion keeps the imaginary (L,M) transform
  // of a complex occupation matrix for diagnostics and reconstruction without
  // injecting an imaginary charge into that functional.
  std::vector<double> lm_moment_imag;
  // Diagnostic arrays use the archive radial mesh, density coefficients in
  // e/Angstrom and potential coefficients in eV. Core is excluded from rho.
  std::vector<std::vector<double>> rho_ae_r2, rho_ps_r2;
  std::vector<std::vector<double>> potential_ae_ev, potential_ps_ev;
  std::vector<std::vector<double>> hartree_ae_ev, hartree_ps_ev;
  std::vector<std::vector<double>> xc_potential_ae_ev, xc_potential_ps_ev;
  std::vector<std::complex<double>> reference_d_matrix_ev;
  double ae_energy_ev{};
  double ps_energy_ev{};
  double ae_hartree_ev{};
  double ps_hartree_ev{};
  double ae_xc_ev{};
  double ps_xc_ev{};
  double energy_difference_ev{};
  double potential_expectation_ev{};
  double reference_linear_energy_ev{};
  // Frozen-core XC energy of the reference core; subtracted once from the AE
  // one-centre energy by the periodic assembler.
  double core_xc_energy_ev{};
  double ae_potential_expectation_ev{}, ps_potential_expectation_ev{};
  double ae_double_counting_ev{}, ps_double_counting_ev{};
  double max_potential_ev{};
  // These describe which density channels participate in the discrete XC
  // energy/derivative pair; they do not certify agreement with reference data.
  bool spherical_gga_variational{false};
  bool aspherical_gga_variational{false};
  bool mesh_compensation_norm{false};
  std::size_t ae_clipped_density_samples{};
  std::size_t ps_clipped_density_samples{};
  bool direct_radial_potential{false};
};

enum class PawRadialLengthUnit { Angstrom, Bohr };
// RadialDirect and AsphericalDirect build the XC potential from the continuum
// Euler-Lagrange form (finite-difference radial gradient applied twice) with an
// inward-integrated Hartree potential, as used for matched VASP references.
// DiscreteVariational and AsphericalAdjoint use the exact adjoint of the energy
// quadrature. Keep the convention explicit: agreement of either matrix alone
// does not certify that its forces are derivatives of the discrete energy.
enum class PawRadialPotentialMode { DiscreteVariational, RadialDirect, AsphericalDirect,
  AsphericalAdjoint }; // Aspherical energy quadrature, exact discrete derivative.

// Signed spherical reference potentials -V_Hxc of the isolated reference atom,
// added to the one-centre AE/PS potentials; not the positive physical potential.
// The archive's stored atomic potential is not a substitute: it is not
// recomputed with this functional.
struct PawOneCenterReferencePotentials {
  // Imported reference-code arrays may leave this unspecified; internally
  // generated references must record and enforce their constants convention.
  AtomicUnitConvention constants{AtomicUnitConvention::MatchRadialMode};
  std::vector<double> ae_reference_potential_ev, ps_reference_potential_ev;
  // Exact projection of the same discrete functional at the reference-atom
  // occupations. This preserves the response at a zero-weight endpoint of an
  // even Simpson mesh.
  std::vector<std::complex<double>> projected_reference_d_matrix_ev;
  double ae_atomic_energy_ev{}, ps_atomic_energy_ev{};
  double core_xc_energy_ev{};
};

struct PawRadialProjectionResult {
  std::vector<std::complex<double>> ae_partial_ev;
  std::vector<std::complex<double>> ps_partial_ev;
  std::vector<std::complex<double>> ps_augmentation_ev;
  std::vector<std::complex<double>> total_ev;
};

// Project unweighted AE/PS radial potential coefficients (eV) through the
// radial partial waves and the aug_moment-weighted compensation shapes.  This
// independent path checks the one-centre D_ij layer by layer.
PawRadialProjectionResult paw_project_radial_potentials(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    const std::vector<std::vector<double>>& potential_ae_ev,
    const std::vector<std::vector<double>>& potential_ps_ev);

// Same radial integrals per (L,M), before the Gaunt contraction to (m,m')
// blocks, in packed upper-triangle channel storage.  Diagnostic use only.
std::vector<std::complex<double>> paw_project_radial_potentials_packed(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    const std::vector<std::vector<double>>& potential_ae_ev,
    const std::vector<std::vector<double>>& potential_ps_ev);

// Project a smooth periodic Fourier potential through the complete PAW
// one-centre response.  Only the compensation shapes are integrated against
// the potential; the result is contracted with aug_moment and the Gaunt
// coefficients.
// This is the positive compensation-density adjoint. The returned matrix is
// in eV and uses the expanded (radial-projector,m) ordering.
std::vector<std::complex<double>> paw_project_fourier_potential(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    const std::array<Vec3, 3>& lattice_bohr, Vec3 fractional_position,
    const FourierDensity& potential_hartree);

std::vector<std::complex<double>> paw_atomic_reference_density(
    const PawNumericCache& cache, const std::vector<int>& channel_l);

// Construct -V_Hxc of the reference atom and its exact projected matrix from
// the production radial functional. The potentials and matrix are in eV, and
// core_xc_energy_ev is the frozen-core PBE energy of the same functional. No
// reference-code dump or empirical shift is used. Cache units follow radial_unit.
PawOneCenterReferencePotentials paw_atomic_reference_potentials(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    PawRadialLengthUnit radial_unit = PawRadialLengthUnit::Angstrom,
    PawRadialPotentialMode potential_mode = PawRadialPotentialMode::DiscreteVariational,
    AtomicUnitConvention constants = AtomicUnitConvention::MatchRadialMode);

PawOneCenterClosureResult paw_one_center_closure(
    const PawNumericCache& cache,
    const std::vector<int>& channel_l,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension,
    int spin_multiplicity = 1,
    PawRadialLengthUnit radial_unit = PawRadialLengthUnit::Angstrom,
    const PawOneCenterReferencePotentials* reference = nullptr,
    PawRadialPotentialMode potential_mode = PawRadialPotentialMode::DiscreteVariational,
    AtomicUnitConvention constants = AtomicUnitConvention::MatchRadialMode);

std::array<PawOneCenterClosureResult,2> paw_collinear_one_center_closure(
    const PawNumericCache&,const std::vector<int>& channel_l,
    const std::array<std::vector<std::complex<double>>,2>& density,std::size_t dimension,
    PawRadialLengthUnit radial_unit=PawRadialLengthUnit::Angstrom,
    const PawOneCenterReferencePotentials* reference=nullptr,
    PawRadialPotentialMode potential_mode=PawRadialPotentialMode::AsphericalDirect,
    AtomicUnitConvention constants=AtomicUnitConvention::MatchRadialMode);

} // namespace sdft
