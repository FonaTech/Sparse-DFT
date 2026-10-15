#pragma once

#include "sdft_core.hpp"
#include "pw_core.hpp"
#include "atomic_units.hpp"
#include "paw_hubbard.hpp"
#include "density_functionals.hpp"
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <complex>
#include <optional>

namespace sdft {

// Canonical option values. Earlier releases spelled these with a "vasp_"
// prefix; those spellings remain accepted input aliases.
inline std::string canonical_projector_interpolation(const std::string& value) {
  if (value == "vasp_cubic") return "cubic";
  if (value == "vasp_spline") return "spline";
  return value;
}
inline std::string canonical_paw_xc_mode(const std::string& value) {
  if (value == "vasp_aspherical_adjoint") return "aspherical_adjoint";
  if (value == "vasp_aspherical") return "aspherical";
  if (value == "vasp_radial") return "radial";
  return value;
}
inline std::string canonical_onecenter_radial_mode(const std::string& value) {
  if (value == "vasp_aspherical_gga") return "aspherical_gga";
  if (value == "vasp_radial_gga") return "radial_gga";
  return value;
}

struct ScfAtom {
  std::string symbol;
  Vec3 fractional{};
  double zval{};
  std::filesystem::path paw_cache;
  double initial_magnetic_moment{};
  HubbardSpec hubbard;
};

struct ScfInput {
  int spin_channels{1}; // 1 unpolarized, 2 real collinear channels
  std::array<Vec3, 3> lattice_bohr{};
  std::vector<ScfAtom> atoms;
  double cutoff_ev{300.0};
  // Independent compensation/core Fourier support. Zero selects 16*ENCUT;
  // this is a starting value, not proof of density-cutoff convergence.
  double density_cutoff_ev{};
  // Explicit finite-basis derivative mode; both wavefunction and density
  // indices are selected in this cell, with all operators in lattice_bohr.
  std::optional<std::array<Vec3, 3>> basis_reference_lattice_bohr;
  int max_iterations{240};
  double density_tolerance{1e-7};
  double onecenter_density_tolerance{1e-7};
  double mixing{0.2};
  int bands{0};
  Vec3 kpoint_fractional{};
  struct Kpoint { Vec3 fractional{}; double weight{}; };
  std::vector<Kpoint> kpoints; // empty preserves the single-k interface
  double smearing_ev{}; // Fermi-Dirac k_B*T, zero-T filling when zero
  AtomicUnitConvention unit_convention{AtomicUnitConvention::Codata};
  // Reciprocal projector table interpolation: "cubic" (Lagrange) or "spline".
  std::string projector_interpolation{"cubic"};
  std::filesystem::path iteration_log;
  std::filesystem::path geometry_guess_input,geometry_guess_output;
};

// Frozen input to the final diagnostic SCF Hamiltonian. onsite_dij_ev includes
// the dataset's bare D_ij and the density-dependent closure correction, in
// expanded (radial,m) order.
struct ScfOneCenterState {
  std::size_t atom_index{};
  std::size_t projector_dimension{};
  std::string symbol;
  Vec3 fractional{};
  std::filesystem::path paw_cache;
  std::string record_id;
  std::uint32_t numeric_cache_version{};
  double numeric_cache_checksum{};
  std::string numeric_cache_sha256;
  double projector_scale{1.0};
  bool projector_q_bohr{};
  bool reciprocal_linear{};
  AtomicUnitConvention unit_convention{AtomicUnitConvention::Codata};
  std::vector<int> radial_channel_l;
  std::vector<std::complex<double>> density_matrix;
  std::vector<std::complex<double>> onsite_dij_ev;
  std::vector<std::complex<double>> closure_correction_ev;
  std::vector<std::complex<double>> compensation_correction_ev;
  std::vector<double> lm_moment_imag;
  double ae_energy_ev{};
  double ps_energy_ev{};
  double energy_difference_ev{};
  double potential_expectation_ev{};
  double core_xc_energy_ev{};
};

struct ScfResult {
  std::string paw_framework{"potcar_paw"};
  int spin_channels{1};
  double magnetic_moment{},hubbard_energy_ev{},hubbard_potential_expectation_ev{},hubbard_occupation_residual{};
  std::string hubbard_occupation_convention{"ae_partial_wave_rdep_angstrom_v1"};
  std::vector<int> kpoint_spin_channels; // flattened spin-major k spectra
  std::array<FourierDensity,2> spin_density_q,spin_hamiltonian_density_q;
  std::array<std::vector<ScfOneCenterState>,2> spin_onecenter_states;
  std::vector<HubbardResult> hubbard_states;
  std::vector<HubbardSpec> hubbard_specs;
  std::optional<DensityFunctionals> frozen_channel_functional; // internal NSCF channel view only
  std::string projector_interpolation{"cubic"};
  bool converged{};
  bool geometry_guess_used{};
  double geometry_guess_retained_fraction{};
  bool onecenter_closure_diagnostic{};
  bool onecenter_diagnostic_output_enabled{true};
  std::string onecenter_residual_definition{"max_atom_frobenius_output_minus_input_over_max_one_output_frobenius"};
  std::string onecenter_radial_mode{"inactive"};
  bool total_energy_valid{false};
  int iterations{};
  std::size_t plane_waves{};
  int occupied_bands{};
  // Bands actually solved; bands_requested is input.bands (0 = automatic).
  // An automatic count grows (restarting SCF) until the smeared top band is empty.
  int bands{},bands_requested{},empty_band_extensions{};
  double electron_count{};
  double cutoff_ev{};
  double density_cutoff_ev{};
  std::size_t density_reciprocal_modes{};
  std::size_t compensation_geometry_groups{},compensation_geometry_bytes{},compensation_atom_bytes{};
  std::size_t compensation_unshared_equivalent_bytes{};
  std::string density_support_index_sha256;
  std::optional<std::array<Vec3, 3>> basis_reference_lattice_bohr;
  double integrated_electron_count{};
  double compensation_electron_count{};
  double fermi_energy_ev{};
  double density_residual{};
  double onecenter_density_residual{};
  std::vector<double> onecenter_density_residuals;
  double onecenter_density_tolerance{};
  std::string onecenter_mixing_method;
  double band_energy_ev{};
  double hartree_energy_ev{};
  double xc_energy_ev{};
  double vxc_density_ev{};
  double local_g0_energy_ev{};
  double ion_ion_energy_ev{};
  double atomic_reference_energy_ev{};
  double atomic_cutoff_correction_ev{};
  double total_energy_ev{};
  double paw_spherical_dij_norm_ev{};
  double paw_double_counting_ae_ev{};
  double paw_double_counting_ps_ev{};
  double paw_onecenter_energy_ev{};
  double paw_onecenter_potential_expectation_ev{};
  // Sum of per-atom frozen-core XC energies of the reference cores.
  double paw_onecenter_core_xc_ev{};
  // Electronic Hellmann--Feynman derivative assembled from the converged
  // generalized eigenvectors.  The ion-ion/Ewald and full PAW one-centre
  // terms are reported separately in approximation_flags until validated.
  std::vector<Vec3> hf_electronic_forces_ev_per_angstrom;
  std::vector<Vec3> ewald_forces_ev_per_angstrom;
  std::vector<Vec3> hf_total_forces_ev_per_angstrom;
  std::array<Vec3, 3> hf_electronic_stress_ev_per_angstrom3{};
  std::array<Vec3, 3> ewald_stress_ev_per_angstrom3{}, hf_total_stress_ev_per_angstrom3{};
  // Disjoint contributions to the same stationary free-energy derivative.
  // Their tensor sum must reproduce hf_total_stress exactly; this ledger is
  // used to identify reference differences, not to fit a stress correction.
  std::map<std::string, std::array<Vec3, 3>> stress_components_ev_per_angstrom3;
  bool analytic_stress_available{};
  std::vector<double> eigenvalues_ev;
  std::vector<ScfInput::Kpoint> kpoints;
  std::vector<std::size_t> plane_waves_per_kpoint;
  std::vector<std::string> plane_wave_index_sha256;
  std::vector<std::vector<double>> kpoint_eigenvalues_ev;
  std::vector<std::vector<double>> kpoint_occupations;
  double entropy_energy_ev{};
  double internal_energy_ev{};
  std::string solver_method;
  std::size_t max_fft_grid_points{}, max_xc_grid_points{}, max_operator_storage_bytes{};
  AtomicUnitConvention unit_convention{AtomicUnitConvention::Codata};
  double eigensolver_max_residual{};
  double eigensolver_history_max_residual{};
  bool adaptive_search{};
  std::string mixer_method;
  bool multisecant_backtracking{};
  double final_mixing_step_scale{1.};
  bool trial_search_enabled{},final_guarded_search{true};
  std::size_t trial_manifold_updates{},guarded_manifold_updates{};
  int precision_refinement_iteration{};
  double scf_acceptance_margin{1.};
  double effective_density_tolerance{},effective_onecenter_density_tolerance{};
  std::vector<double> density_fourier;
  std::map<std::tuple<int, int, int>, std::complex<double>> density_q;
  std::map<std::tuple<int, int, int>, std::complex<double>> hamiltonian_density_q;
  std::map<std::tuple<int, int, int>, std::complex<double>> frozen_core_q;
  std::array<Vec3, 3> onecenter_lattice_bohr{};
  std::vector<ScfOneCenterState> onecenter_states;
  std::vector<GVector> g_vectors;
};

ScfResult run_scf(const ScfInput& input);
// Read-only spectral-tail probe of the PAW hard densities (JSON text).
std::string density_spectrum_probe_json(const ScfInput& input);
// Audited, converged frozen PAW state for NSCF. The text payload must match
// its SHA-256 sidecar and the caller's structure, units, cutoff and PAW data.
ScfResult read_frozen_scf(const std::filesystem::path& path, const ScfInput& input);
// Independent fixed-H/S derivative diagnostic. Does not change production
// stress or mark a finite-displacement reference as an analytic derivative.
void write_projector_stress_diagnostic(const ScfInput& input, const ScfResult& scf,
                                      const std::filesystem::path& output);

struct BandPathResult {
  int spin_channels{1};
  std::array<std::vector<double>,2> spin_eigenvalues_ev,spin_projector_weights,spin_residuals;
  std::array<std::vector<ScfOneCenterState>,2> spin_onecenter_states;
  std::vector<Vec3> kpoints;
  std::vector<double> eigenvalues_ev;
  std::vector<double> eigensolver_max_residuals_hartree;
  std::vector<double> seed_transport_retained_fraction;
  std::size_t warm_start_kpoints{},identical_k_spectrum_reuses{};
  int bands{};
  std::size_t nkpoints{};
  std::string solver_method;
  std::size_t max_fft_grid_points{}, max_xc_grid_points{}, max_operator_storage_bytes{};
  double fermi_energy_ev{};
  bool onecenter_dij_reused{};
  std::size_t onecenter_dij_applications{};
  std::vector<std::string> projection_orbitals,projection_atom_symbols;
  // Raw sum |B psi|^2, not normalized orbital probabilities or occupancies.
  std::vector<double> projector_weights;
  // Same sum split by real harmonic m, labelled in VASP PROCAR order
  // (s; py pz px; dxy dyz dz2 dxz dx2-y2; f...). Layout [k][band][atom][lm].
  // Empty when the projector path only resolves l.
  std::vector<std::string> projection_lm_orbitals;
  std::string projection_lm_weight_definition;
  std::vector<double> projector_lm_weights;
  std::array<std::vector<double>,2> spin_projector_lm_weights;
  bool spin_projection_available{};
  std::vector<ScfOneCenterState> onecenter_states;
};

BandPathResult run_nscf(const ScfInput& input, const ScfResult& scf,
                        const std::vector<Vec3>& kpoints_fractional,
                        int bands);

}  // namespace sdft
