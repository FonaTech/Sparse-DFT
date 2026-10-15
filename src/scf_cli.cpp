#include "scf_core.hpp"
#include "compute_backend.hpp"
#include "density_json.hpp"
#include "parallel_context.hpp"
#include <memory>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace {
using namespace sdft;

struct Job {
  ScfInput input;
  std::vector<Vec3> path;
};

std::string json_string(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : text) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20)
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << c;
    }
  }
  out << '"';
  return out.str();
}

void write_complex_matrix(std::ostream& f, const std::vector<std::complex<double>>& values,
                          std::size_t dimension) {
  if (values.size() != dimension * dimension)
    throw std::runtime_error("cannot serialize PAW matrix with invalid dimensions");
  f << '[';
  for (std::size_t row = 0; row < dimension; ++row) {
    if (row) f << ", ";
    f << '[';
    for (std::size_t col = 0; col < dimension; ++col) {
      if (col) f << ", ";
      const auto value = values[row * dimension + col];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::runtime_error("cannot serialize non-finite PAW matrix");
      f << '[' << value.real() << ", " << value.imag() << ']';
    }
    f << ']';
  }
  f << ']';
}

void write_onecenter_states(std::ostream& f, const std::vector<ScfOneCenterState>& states) {
  f << '[';
  for (std::size_t atom = 0; atom < states.size(); ++atom) {
    if (atom) f << ", ";
    const auto& state = states[atom];
    f << "{\"schema_version\": \"sparse-dft-paw-frozen-onsite-dij-2\", \"atom_index\": " << state.atom_index
      << ", \"symbol\": " << json_string(state.symbol)
      << ", \"fractional_position\": [" << state.fractional.x << ", " << state.fractional.y << ", " << state.fractional.z << ']'
      << ", \"projector_dimension\": " << state.projector_dimension
      << ", \"ordering\": \"radial_projector_then_m_minus_l_to_l_real_harmonics\""
      << ", \"complex_encoding\": \"row_major_matrix_of_real_imag_pairs\""
      << ", \"matrix_units\": \"eV\", \"onsite_dij_includes_bare_dij\": true"
      << ", \"density_source\": \"final_scf_hamiltonian_input_onsite_occupation\""
      << ", \"validation_status\": \"diagnostic_only\""
      << ", \"paw_cache\": " << json_string(state.paw_cache.string())
      << ", \"paw_record_id\": " << json_string(state.record_id)
      << ", \"numeric_cache_version\": " << state.numeric_cache_version
      << ", \"numeric_cache_checksum\": " << state.numeric_cache_checksum
      << ", \"numeric_cache_sha256\": " << json_string(state.numeric_cache_sha256)
      << ", \"projector_scale\": " << state.projector_scale
      << ", \"projector_q_bohr\": " << (state.projector_q_bohr ? "true" : "false")
      << ", \"reciprocal_linear\": " << (state.reciprocal_linear ? "true" : "false")
      << ", \"atomic_units\": " << json_string(std::string(atomic_unit_name(state.unit_convention)))
      << ", \"radial_channel_l\": [";
    for (std::size_t i = 0; i < state.radial_channel_l.size(); ++i)
      f << (i ? ", " : "") << state.radial_channel_l[i];
    f << "], \"onsite_occupation\": ";
    write_complex_matrix(f, state.density_matrix, state.projector_dimension);
    f << ", \"onsite_dij_ev\": ";
    write_complex_matrix(f, state.onsite_dij_ev, state.projector_dimension);
    f << ", \"closure_correction_ev\": ";
    write_complex_matrix(f, state.closure_correction_ev, state.projector_dimension);
    f << ", \"compensation_correction_ev\": ";
    write_complex_matrix(f, state.compensation_correction_ev, state.projector_dimension);
    f << ", \"lm_moment_imag\": [";
    for (std::size_t i = 0; i < state.lm_moment_imag.size(); ++i)
      f << (i ? ", " : "") << state.lm_moment_imag[i];
    f << ']';
    f << ", \"closure_ae_energy_ev\": " << state.ae_energy_ev
      << ", \"closure_ps_energy_ev\": " << state.ps_energy_ev
      << ", \"closure_energy_difference_ev\": " << state.energy_difference_ev
      << ", \"closure_potential_expectation_ev\": " << state.potential_expectation_ev
      << ", \"closure_core_xc_energy_ev\": " << state.core_xc_energy_ev << '}';
  }
  f << ']';
}

void write_density_q(std::ostream& f, const std::map<std::tuple<int, int, int>, std::complex<double>>& density) {
  write_density_fourier_json(f,density);
}

// Input files are UTF-8; on Windows std::filesystem::path(std::string)
// would decode with the active code page (GBK), so convert explicitly.
std::filesystem::path utf8_path(const std::string& s) {
  return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

Job read_job(const std::string& filename) {
  std::ifstream f(utf8_path(filename));
  if (!f) throw std::runtime_error("cannot open Sparse-DFT job input: " + filename);
  Job job;
  std::string tag;
  if (!(f >> tag) || tag != "SDFT_SCF_INPUT_V1") throw std::runtime_error("invalid Sparse-DFT SCF input header");
  if (!(f >> tag) || tag != "lattice_bohr") throw std::runtime_error("missing lattice_bohr");
  for (auto& v : job.input.lattice_bohr) if (!(f >> v.x >> v.y >> v.z)) throw std::runtime_error("invalid lattice");
  std::size_t natoms = 0;
  if (!(f >> tag >> natoms) || tag != "atoms") throw std::runtime_error("missing atoms");
  job.input.atoms.resize(natoms);
  for (auto& atom : job.input.atoms) {
    // Paths may be written with std::quoted (spaces, non-ASCII); a bare
    // token still parses because std::quoted falls back to >> without '"'.
    std::string cache;
    if (!(f >> tag >> atom.symbol >> atom.fractional.x >> atom.fractional.y >> atom.fractional.z >> atom.zval >> std::quoted(cache)) || tag != "atom")
      throw std::runtime_error("invalid atom record");
    atom.paw_cache = utf8_path(cache);
  }
  while (f >> tag) {
    if (tag == "cutoff_ev") f >> job.input.cutoff_ev;
    else if (tag == "geometry_guess_input") { std::string v; f >> std::quoted(v); job.input.geometry_guess_input = utf8_path(v); }
    else if (tag == "geometry_guess_output") { std::string v; f >> std::quoted(v); job.input.geometry_guess_output = utf8_path(v); }
    else if (tag == "density_cutoff_ev") f >> job.input.density_cutoff_ev;
    else if (tag == "basis_reference_lattice_bohr") {
      std::array<Vec3,3> reference{};
      for (auto& v : reference) f >> v.x >> v.y >> v.z;
      job.input.basis_reference_lattice_bohr = reference;
    }
    else if (tag == "max_iterations") f >> job.input.max_iterations;
    else if (tag == "density_tolerance") f >> job.input.density_tolerance;
    else if (tag == "onecenter_density_tolerance") f >> job.input.onecenter_density_tolerance;
    else if (tag == "mixing") f >> job.input.mixing;
    else if (tag == "bands") f >> job.input.bands;
    else if (tag == "smearing_ev") f >> job.input.smearing_ev;
    else if (tag == "spin_channels") f >> job.input.spin_channels;
    else if (tag == "initial_moments") {
      std::size_t count;f>>count;if(count!=job.input.atoms.size())throw std::runtime_error("initial moment atom count mismatch");
      for(auto& atom:job.input.atoms)f>>atom.initial_magnetic_moment;
    }
    else if (tag == "hubbard_ueff") {
      std::size_t count;f>>count;if(count!=job.input.atoms.size())throw std::runtime_error("Hubbard atom count mismatch");
      for(auto& atom:job.input.atoms)f>>atom.hubbard.angular_momentum>>atom.hubbard.ueff_ev;
    }
    else if (tag == "projector_interpolation") {
      f >> job.input.projector_interpolation;
      job.input.projector_interpolation=canonical_projector_interpolation(job.input.projector_interpolation);
      if(job.input.projector_interpolation!="cubic" && job.input.projector_interpolation!="spline" && job.input.projector_interpolation!="upf_original_radial_quadrature" && job.input.projector_interpolation!="per_site_original_paw")
        throw std::runtime_error("projector_interpolation must be cubic, spline or upf_original_radial_quadrature");
    }
    else if (tag == "atomic_units") {
      std::string units;f>>units;
      if(units=="codata")job.input.unit_convention=AtomicUnitConvention::Codata;
      else if(units=="vasp_legacy")job.input.unit_convention=AtomicUnitConvention::VaspLegacy;
      else throw std::runtime_error("atomic_units must be codata or vasp_legacy");
    }
    else if (tag == "kpoint_fractional") f >> job.input.kpoint_fractional.x >> job.input.kpoint_fractional.y >> job.input.kpoint_fractional.z;
    else if (tag == "scf_kpoints") {
      std::size_t count=0;f>>count;
      if(!f || count==0 || count>200000)throw std::runtime_error("invalid SCF k-point count");
      job.input.kpoints.resize(count);
      for(auto& k:job.input.kpoints)f>>k.fractional.x>>k.fractional.y>>k.fractional.z>>k.weight;
    }
    else if (tag == "path") {
      std::size_t count = 0; f >> count; job.path.resize(count);
      for (auto& k : job.path) f >> k.x >> k.y >> k.z;
    } else throw std::runtime_error("unknown Sparse-DFT SCF input key: " + tag);
    if(!f)throw std::runtime_error("malformed Sparse-DFT SCF value for key: "+tag);
  }
  return job;
}

void write_scf(const std::string& filename, const ScfResult& r) {
  sdft::PerformanceTimer io_timer("scf_result_serialization");
  std::ofstream f(filename);
  if (!f) throw std::runtime_error("cannot write Sparse-DFT SCF output: " + filename);
  f << std::setprecision(std::numeric_limits<double>::max_digits10) << "{\n"
    << "  \"paw_framework\": " << json_string(r.paw_framework) << ",\n"
    << "  \"onecenter_diagnostic_output_enabled\": " << (r.onecenter_diagnostic_output_enabled?"true":"false") << ",\n"
    << "  \"spin_channels\": " << r.spin_channels << ",\n"
    << "  \"magnetic_moment_muB\": " << r.magnetic_moment << ",\n"
    << "  \"hubbard_energy_ev\": " << r.hubbard_energy_ev << ",\n"
    << "  \"hubbard_potential_expectation_ev\": " << r.hubbard_potential_expectation_ev << ",\n"
    << "  \"hubbard_occupation_convention\": " << json_string(r.hubbard_occupation_convention) << ",\n"
    << "  \"hubbard_occupation_residual\": " << r.hubbard_occupation_residual << ",\n"
    << "  \"projector_interpolation\": " << json_string(r.projector_interpolation) << ",\n"
    << "  \"schema_version\": \"sparse-dft-scf-1\",\n"
    << "  \"converged\": " << (r.converged ? "true" : "false") << ",\n"
    << "  \"geometry_guess_used\": " << (r.geometry_guess_used ? "true" : "false") << ",\n"
    << "  \"geometry_guess_retained_fraction\": " << r.geometry_guess_retained_fraction << ",\n"
    << "  \"total_energy_valid\": " << (r.total_energy_valid ? "true" : "false") << ",\n"
    << "  \"onecenter_closure_diagnostic\": " << (r.onecenter_closure_diagnostic ? "true" : "false") << ",\n"
    << "  \"onecenter_radial_mode\": " << json_string(r.onecenter_radial_mode) << ",\n"
    << "  \"iterations\": " << r.iterations << ",\n"
    << "  \"plane_waves\": " << r.plane_waves << ",\n"
    << "  \"scf_kpoint_count\": " << r.kpoints.size() << ",\n"
    << "  \"solver_method\": " << json_string(r.solver_method) << ",\n"
    << "  \"max_fft_grid_points\": " << r.max_fft_grid_points << ",\n"
    << "  \"max_xc_grid_points\": " << r.max_xc_grid_points << ",\n"
    << "  \"max_operator_storage_bytes\": " << r.max_operator_storage_bytes << ",\n"
    << "  \"atomic_units\": " << json_string(std::string(atomic_unit_name(r.unit_convention))) << ",\n"
    << "  \"eigensolver_max_residual_hartree\": " << r.eigensolver_max_residual << ",\n"
    << "  \"eigensolver_history_max_residual_hartree\": " << r.eigensolver_history_max_residual << ",\n"
    << "  \"adaptive_search\": " << (r.adaptive_search?"true":"false") << ",\n"
    << "  \"trial_search_enabled\": " << (r.trial_search_enabled?"true":"false") << ",\n"
    << "  \"final_guarded_search\": " << (r.final_guarded_search?"true":"false") << ",\n"
    << "  \"trial_manifold_updates\": " << r.trial_manifold_updates << ",\n"
    << "  \"guarded_manifold_updates\": " << r.guarded_manifold_updates << ",\n"
    << "  \"mixer_method\": " << json_string(r.mixer_method) << ",\n"
    << "  \"multisecant_backtracking\": " << (r.multisecant_backtracking?"true":"false") << ",\n"
    << "  \"final_mixing_step_scale\": " << r.final_mixing_step_scale << ",\n"
    << "  \"precision_refinement_iteration\": " << r.precision_refinement_iteration << ",\n"
    << "  \"scf_acceptance_margin\": " << r.scf_acceptance_margin << ",\n"
    << "  \"effective_density_tolerance\": " << r.effective_density_tolerance << ",\n"
    << "  \"effective_onecenter_density_tolerance\": " << r.effective_onecenter_density_tolerance << ",\n"
    << "  \"occupied_bands\": " << r.occupied_bands << ",\n"
    << "  \"solved_bands\": " << r.bands << ",\n"
    << "  \"bands_requested\": " << r.bands_requested << ",\n"
    << "  \"empty_band_extensions\": " << r.empty_band_extensions << ",\n"
    << "  \"electron_count\": " << r.electron_count << ",\n"
    << "  \"cutoff_ev\": " << r.cutoff_ev << ",\n"
    << "  \"density_cutoff_ev\": " << r.density_cutoff_ev << ",\n"
    << "  \"density_reciprocal_modes\": " << r.density_reciprocal_modes << ",\n"
    << "  \"compensation_geometry_groups\": " << r.compensation_geometry_groups << ",\n"
    << "  \"compensation_geometry_bytes\": " << r.compensation_geometry_bytes << ",\n"
    << "  \"compensation_atom_bytes\": " << r.compensation_atom_bytes << ",\n"
    << "  \"compensation_unshared_equivalent_bytes\": " << r.compensation_unshared_equivalent_bytes << ",\n"
    << "  \"density_support_index_sha256\": " << json_string(r.density_support_index_sha256) << ",\n"
    << "  \"basis_policy\": " << json_string(r.basis_reference_lattice_bohr ? "fixed_reference_lattice" : "current_lattice_cutoff") << ",\n"
    << "  \"integrated_electron_count\": " << r.integrated_electron_count << ",\n"
    << "  \"compensation_electron_count\": " << r.compensation_electron_count << ",\n"
    << "  \"fermi_energy_ev\": " << r.fermi_energy_ev << ",\n"
    << "  \"density_residual\": " << r.density_residual << ",\n"
    << "  \"onecenter_density_residual\": " << r.onecenter_density_residual << ",\n"
    << "  \"onecenter_density_tolerance\": " << r.onecenter_density_tolerance << ",\n"
    << "  \"basis_reference_lattice_bohr\": ";
  if (r.basis_reference_lattice_bohr) {
    f << '[';
    for (std::size_t i=0; i<3; ++i) {
      const auto& v = r.basis_reference_lattice_bohr->at(i);
      f << (i ? ", " : "") << '[' << v.x << ", " << v.y << ", " << v.z << ']';
    }
    f << ']';
  } else f << "null";
  f << ",\n  \"onecenter_density_residuals\": [";
  for (std::size_t i = 0; i < r.onecenter_density_residuals.size(); ++i)
    f << (i ? ", " : "") << r.onecenter_density_residuals[i];
  f << "],\n"
    << "  \"onecenter_mixing_method\": " << json_string(r.onecenter_mixing_method) << ",\n"
    << "  \"onecenter_residual_provenance\": {\"definition\": " << json_string(r.onecenter_residual_definition) << ", \"units\": \"dimensionless\", \"absolute_density_units\": \"projected_electron_occupancy\"},\n"
    << "  \"band_energy_ev\": " << r.band_energy_ev << ",\n"
    << "  \"entropy_energy_ev\": " << r.entropy_energy_ev << ",\n"
    << "  \"internal_energy_ev\": " << r.internal_energy_ev << ",\n"
    << "  \"hartree_energy_ev\": " << r.hartree_energy_ev << ",\n"
    << "  \"xc_energy_ev\": " << r.xc_energy_ev << ",\n"
    << "  \"vxc_density_ev\": " << r.vxc_density_ev << ",\n"
    << "  \"local_g0_energy_ev\": " << r.local_g0_energy_ev << ",\n"
    << "  \"ion_ion_energy_ev\": " << r.ion_ion_energy_ev << ",\n"
    << "  \"atomic_reference_energy_ev\": " << r.atomic_reference_energy_ev << ",\n"
    << "  \"atomic_cutoff_correction_ev\": " << r.atomic_cutoff_correction_ev << ",\n"
    << "  \"paw_spherical_dij_norm_ev\": " << r.paw_spherical_dij_norm_ev << ",\n"
    << "  \"paw_double_counting_ae_ev\": " << r.paw_double_counting_ae_ev << ",\n"
    << "  \"paw_double_counting_ps_ev\": " << r.paw_double_counting_ps_ev << ",\n"
    << "  \"paw_onecenter_energy_ev\": " << r.paw_onecenter_energy_ev << ",\n"
    << "  \"paw_onecenter_potential_expectation_ev\": " << r.paw_onecenter_potential_expectation_ev << ",\n"
    << "  \"paw_onecenter_core_xc_ev\": " << r.paw_onecenter_core_xc_ev << ",\n"
    << "  \"total_energy_ev\": " << r.total_energy_ev << ",\n"
    << "  \"eigenvalues_ev\": [";
  for (std::size_t i = 0; i < r.eigenvalues_ev.size(); ++i) f << (i ? ", " : "") << r.eigenvalues_ev[i];
  f << "],\n  \"scf_kpoints\": [";
  for(std::size_t k=0;k<r.kpoints.size();++k) {
    if(k)f<<", ";
    const auto& kp=r.kpoints[k];
    f<<"{\"fractional\": ["<<kp.fractional.x<<", "<<kp.fractional.y<<", "<<kp.fractional.z
      <<"], \"spin_channel\": "<<r.kpoint_spin_channels.at(k)
      <<", \"weight\": "<<kp.weight<<", \"plane_waves\": "<<r.plane_waves_per_kpoint[k]
      <<", \"plane_wave_index_sha256\": "<<json_string(r.plane_wave_index_sha256.at(k))
      <<", \"eigenvalues_ev\": [";
    for(std::size_t band=0;band<r.kpoint_eigenvalues_ev[k].size();++band)
      f<<(band?", ":"")<<r.kpoint_eigenvalues_ev[k][band];
    f<<"], \"occupations\": [";
    for(std::size_t band=0;band<r.kpoint_occupations[k].size();++band)
      f<<(band?", ":"")<<r.kpoint_occupations[k][band];
    f<<"]}";
  }
  f << "],\n  \"approximation_flags\": [" << json_string(r.spin_channels==2?"collinear_spin_pbe_paw":"spin_unpolarized") << ", \"k_weighted_scf_density\", \"k_shifted_plane_wave_cutoff\", \"paw_framework_original_reciprocal_sampling\", \"pbe_pw92_xc_spectral_derivative\", \"paw_nonlocal_low_rank\", \"ewald_adaptive_tail_1e-14\", \"paw_energy_derivatives_pending_validation\"],\n"
    << "  \"hf_electronic_forces_ev_per_angstrom\": [";
  for (std::size_t i = 0; i < r.hf_electronic_forces_ev_per_angstrom.size(); ++i) {
    if (i) f << ", ";
    const auto& force = r.hf_electronic_forces_ev_per_angstrom[i];
    f << "[" << force.x << ", " << force.y << ", " << force.z << "]";
  }
  f << "],\n  \"ewald_forces_ev_per_angstrom\": [";
  for (std::size_t i=0;i<r.ewald_forces_ev_per_angstrom.size();++i) {
    const auto& force=r.ewald_forces_ev_per_angstrom[i];
    f << (i?", ":"") << '[' << force.x << ", " << force.y << ", " << force.z << ']';
  }
  f << "],\n  \"hf_total_forces_ev_per_angstrom\": [";
  for (std::size_t i=0;i<r.hf_total_forces_ev_per_angstrom.size();++i) {
    const auto& force=r.hf_total_forces_ev_per_angstrom[i];
    f << (i?", ":"") << '[' << force.x << ", " << force.y << ", " << force.z << ']';
  }
  f << "],\n";
  for(const auto& item:std::vector<std::pair<std::string,std::array<sdft::Vec3,3>>>{
      {"hf_electronic_stress_ev_per_angstrom3",r.hf_electronic_stress_ev_per_angstrom3},
      {"ewald_stress_ev_per_angstrom3",r.ewald_stress_ev_per_angstrom3},
      {"hf_total_stress_ev_per_angstrom3",r.hf_total_stress_ev_per_angstrom3}}) {
    f<<"  \""<<item.first<<"\": ";
    if(!r.analytic_stress_available){f<<"null,\n";continue;}
    f<<'[';
    for(int a=0;a<3;++a){const auto row=item.second[a];f<<(a?", ":"")<<'['<<row.x<<", "<<row.y<<", "<<row.z<<']';}
    f<<"],\n";
  }
  f<<"  \"stress_components_ev_per_angstrom3\": ";
  if(!r.analytic_stress_available)f<<"null,\n";
  else {
    f<<'{';bool comma=false;
    for(const auto& [name,tensor]:r.stress_components_ev_per_angstrom3) {
      f<<(comma?", ":"")<<json_string(name)<<": [";comma=true;
      for(int a=0;a<3;++a){const auto row=tensor[a];f<<(a?", ":"")<<'['<<row.x<<", "<<row.y<<", "<<row.z<<']';}
      f<<']';
    }
    f<<"},\n";
  }
  f<<"  \"stress_provenance\": {\"analytic\": "<<(r.analytic_stress_available?"true":"false")
    <<", \"validated\": false, \"convention\": \"dE_dstrain_over_volume_positive_tension\", \"scope\": \"fixed_plane_wave_indices_paw_kinetic_local_projector_overlap_compensation_nlcc_ewald\", \"cutoff_pulay_extrapolation\": false},\n";
  f << "  \"force_provenance\": {\"method\": \"stationary_paw_projector_overlap_compensation_nlcc_ewald\", \"analytic\": true, \"validated\": false, \"onecenter_dij_reused\": "
    << (r.onecenter_closure_diagnostic?"true":"false")
    << ", \"status\": \"diagnostic_pending_finite_difference_and_vasp\"},\n"
    << "  \"density_fourier_values\": [";
  for (std::size_t i = 0; i < r.density_fourier.size(); ++i) f << (i ? ", " : "") << r.density_fourier[i];
  f << "],\n  \"density_q\": ";
  write_density_q(f, r.density_q);
  f << ",\n  \"hamiltonian_density_q\": ";
  write_density_q(f, r.hamiltonian_density_q);
  f<<",\n  \"hubbard_specs\": [";
  for(std::size_t a=0;a<r.hubbard_specs.size();++a){if(a)f<<",";f<<"{\"l\":"<<r.hubbard_specs[a].angular_momentum<<",\"ueff_ev\":"<<r.hubbard_specs[a].ueff_ev<<"}";}
  f<<"]";
  if(r.spin_channels==2){
    f<<",\n  \"spin_density_q\": [";write_density_q(f,r.spin_density_q[0]);f<<",";write_density_q(f,r.spin_density_q[1]);f<<"]";
    f<<",\n  \"spin_hamiltonian_density_q\": [";write_density_q(f,r.spin_hamiltonian_density_q[0]);f<<",";write_density_q(f,r.spin_hamiltonian_density_q[1]);f<<"]";
  }
  f << ",\n  \"frozen_core_q\": ";
  write_density_q(f, r.frozen_core_q);
  f << ",\n  \"onecenter_states\": ";
  write_onecenter_states(f, r.onecenter_states);
  if(r.spin_channels==2){
    f<<",\n  \"spin_onecenter_states\": [";write_onecenter_states(f,r.spin_onecenter_states[0]);f<<",";write_onecenter_states(f,r.spin_onecenter_states[1]);f<<"]";
    f<<",\n  \"hubbard_states\": [";
    for(std::size_t atom=0;atom<r.hubbard_states.size();++atom){if(atom)f<<",";const auto& u=r.hubbard_states[atom];
      f<<"{\"atom_index\":"<<atom<<",\"correlated_dimension\":"<<u.correlated_dimension<<",\"energy_ev\":"<<u.energy_ev<<",\"occupations\":[";
      if(u.correlated_dimension){write_complex_matrix(f,u.occupations[0],u.correlated_dimension);f<<",";write_complex_matrix(f,u.occupations[1],u.correlated_dimension);}f<<"]}";}
    f<<"]";
  }
  f << ",\n  \"onecenter_lattice_bohr\": [";
  for (std::size_t axis = 0; axis < r.onecenter_lattice_bohr.size(); ++axis) {
    if (axis) f << ", ";
    const auto& vector = r.onecenter_lattice_bohr[axis];
    f << '[' << vector.x << ", " << vector.y << ", " << vector.z << ']';
  }
  f << ']';
  f << ",\n  \"onecenter_provenance\": {\"validation_status\": \"diagnostic_only\", "
    << "\"hamiltonian_density_source\": \"final_scf_hamiltonian_input_density\", "
    << "\"double_counting_validated\": false, \"analytic_derivatives_validated\": false}\n}\n";
}

void write_bands(const std::string& filename, const BandPathResult& r,
                 const ScfInput& input, const ScfResult& scf) {
  std::ofstream f(filename);
  if (!f) throw std::runtime_error("cannot write Sparse-DFT bands output: " + filename);
  const auto reciprocal = std::array<Vec3, 3>{
      sdft::cross(input.lattice_bohr[1], input.lattice_bohr[2]),
      sdft::cross(input.lattice_bohr[2], input.lattice_bohr[0]),
      sdft::cross(input.lattice_bohr[0], input.lattice_bohr[1])};
  // The input lattice is in bohr; serialize reciprocal vectors in 1/Angstrom
  // to match vasprun.xml and standard band-structure plotting conventions.
  const double reciprocal_scale = 2.0 * 3.14159265358979323846 * atomic_units(input.unit_convention).bohr_per_angstrom /
                                  sdft::determinant(input.lattice_bohr);
  std::array<Vec3, 3> reciprocal_basis = reciprocal;
  for (auto& vector : reciprocal_basis) {
    vector.x *= reciprocal_scale;
    vector.y *= reciprocal_scale;
    vector.z *= reciprocal_scale;
  }
  f << std::setprecision(std::numeric_limits<double>::max_digits10) << "{\n  \"schema_version\": \"sparse-dft-bands-1\",\n"
    << "  \"onecenter_radial_mode\": " << json_string(scf.onecenter_radial_mode) << ",\n"
    << "  \"fermi_energy_ev\": " << r.fermi_energy_ev << ",\n  \"kpoints\": [";
  for (std::size_t i = 0; i < r.kpoints.size(); ++i) {
    if (i) f << ", ";
    f << "[" << r.kpoints[i].x << ", " << r.kpoints[i].y << ", " << r.kpoints[i].z << "]";
  }
  f << "],\n  \"eigenvalues\": [";
  for (std::size_t i = 0; i < r.kpoints.size(); ++i) {
    if (i) f << ", ";
    f << "[";
    for (int b = 0; b < r.bands; ++b) f << (b ? ", " : "") << r.eigenvalues_ev[i * static_cast<std::size_t>(r.bands) + static_cast<std::size_t>(b)];
    f << "]";
  }
  f << "]";
  if(r.spin_channels==2){
    f<<",\n  \"spin_channels\": {";
    for(int c=0;c<2;++c){f<<(c?",":"")<<json_string(c?"down":"up")<<":[";
      for(std::size_t k=0;k<r.nkpoints;++k){f<<(k?",":"")<<"[";
        for(int band=0;band<r.bands;++band)f<<(band?",":"")<<r.spin_eigenvalues_ev[c].at(k*r.bands+band);
        f<<"]";}f<<"]";}
    f<<"},\n  \"spin_channel_count\": 2,\n  \"projections\": {\"schema_version\":\"sparse-dft-projector-2\","
      <<"\"spin_projection_available\":true,\"weight_definition\":\"raw_sum_abs_Bpsi_squared_not_normalized_probability\","
      <<"\"projection_orbitals\":[";
    for(std::size_t o=0;o<r.projection_orbitals.size();++o)f<<(o?",":"")<<json_string(r.projection_orbitals[o]);
    f<<"],\"projection_atom_symbols\":[";
    for(std::size_t a=0;a<r.projection_atom_symbols.size();++a)f<<(a?",":"")<<json_string(r.projection_atom_symbols[a]);
    f<<"],\"projection_weights\":{";
    for(int c=0;c<2;++c){f<<(c?",":"")<<json_string(c?"down":"up")<<":[";std::size_t offset=0;
      for(std::size_t k=0;k<r.nkpoints;++k){f<<(k?",":"")<<"[";
        for(int b=0;b<r.bands;++b){f<<(b?",":"")<<"[";
          for(std::size_t a=0;a<r.projection_atom_symbols.size();++a){f<<(a?",":"")<<"[";
            for(std::size_t o=0;o<r.projection_orbitals.size();++o)f<<(o?",":"")<<r.spin_projector_weights[c].at(offset++);
            f<<"]";}f<<"]";}f<<"]";}f<<"]";}
    f<<"}";
    const std::size_t nlm=r.projection_lm_orbitals.size();
    if(nlm&&r.spin_projector_lm_weights[0].size()==r.nkpoints*r.bands*r.projection_atom_symbols.size()*nlm&&
       r.spin_projector_lm_weights[1].size()==r.spin_projector_lm_weights[0].size()){
      f<<",\"projection_lm_orbitals\":[";
      for(std::size_t o=0;o<nlm;++o)f<<(o?",":"")<<json_string(r.projection_lm_orbitals[o]);
      f<<"],\"projection_lm_weight_definition\":"<<json_string(r.projection_lm_weight_definition)<<",\"projection_lm_weights\":{";
      for(int c=0;c<2;++c){f<<(c?",":"")<<json_string(c?"down":"up")<<":[";std::size_t offset=0;
        for(std::size_t k=0;k<r.nkpoints;++k){f<<(k?",":"")<<"[";
          for(int b=0;b<r.bands;++b){f<<(b?",":"")<<"[";
            for(std::size_t a=0;a<r.projection_atom_symbols.size();++a){f<<(a?",":"")<<"[";
              for(std::size_t o=0;o<nlm;++o)f<<(o?",":"")<<r.spin_projector_lm_weights[c].at(offset++);
              f<<"]";}f<<"]";}f<<"]";}f<<"]";}
      f<<"}";
    }
    f<<"}";
  }
  f << ",\n  \"eigensolver_max_residuals_hartree\": [";
  for (std::size_t k=0; k<r.eigensolver_max_residuals_hartree.size(); ++k)
    f << (k ? ", " : "") << r.eigensolver_max_residuals_hartree[k];
  f << "],\n  \"warm_start_kpoints\": " << r.warm_start_kpoints
    << ",\n  \"identical_k_spectrum_reuses\": " << r.identical_k_spectrum_reuses
    << ",\n  \"seed_transport_retained_fraction\": [";
  for(std::size_t k=0;k<r.seed_transport_retained_fraction.size();++k)
    f << (k?", ":"") << r.seed_transport_retained_fraction[k];
  f << "],\n  \"projection_orbitals\": [";
  for (std::size_t i=0; i<r.projection_orbitals.size(); ++i)
    f << (i ? ", " : "") << json_string(r.projection_orbitals[i]);
  f << "],\n  \"projection_atom_symbols\": [";
  for (std::size_t i=0; i<r.projection_atom_symbols.size(); ++i)
    f << (i ? ", " : "") << json_string(r.projection_atom_symbols[i]);
  f << "],\n  \"projector_weights\": [";
  for (std::size_t i=0; i<r.projector_weights.size(); ++i)
    f << (i ? ", " : "") << r.projector_weights[i];
  // AE-sphere (VASP LORBIT=11) lm weights, PROCAR column order; [k][band][atom][lm].
  f << "],\n  \"projection_lm_orbitals\": [";
  for (std::size_t i=0; i<r.projection_lm_orbitals.size(); ++i)
    f << (i ? ", " : "") << json_string(r.projection_lm_orbitals[i]);
  f << "],\n  \"projection_lm_weight_definition\": " << json_string(r.projection_lm_weight_definition)
    << ",\n  \"projector_lm_weights\": [";
  for (std::size_t i=0; i<r.projector_lm_weights.size(); ++i)
    f << (i ? ", " : "") << r.projector_lm_weights[i];
  f << "],\n  \"spin_projection_available\": " << (r.spin_projection_available ? "true" : "false")
    << ",\n  \"projection_weight_definition\": \"raw_sum_abs_Bpsi_squared_not_normalized_probability\""
    << ",\n  \"metadata\": {\"engine\": \"Sparse-DFT C++\", \"density_source\": \"converged_scf\"," 
    << " \"scf_converged\": " << (scf.converged ? "true" : "false")
    << ", \"solver_method\": " << json_string(r.solver_method)
    << ", \"max_fft_grid_points\": " << r.max_fft_grid_points
    << ", \"max_xc_grid_points\": " << r.max_xc_grid_points
    << ", \"max_operator_storage_bytes\": " << r.max_operator_storage_bytes
    << ", \"scf_kpoint_count\": " << scf.kpoints.size()/scf.spin_channels
    << ", \"scf_iterations\": " << scf.iterations
    << ", \"scf_density_residual\": " << scf.density_residual
    << ", \"scf_onecenter_density_residual\": " << scf.onecenter_density_residual
    << ", \"scf_onecenter_density_tolerance\": " << scf.onecenter_density_tolerance
    << ", \"scf_integrated_electron_count\": " << scf.integrated_electron_count
    << ", \"scf_compensation_electron_count\": " << scf.compensation_electron_count
    << ", \"density_cutoff_ev\": " << scf.density_cutoff_ev
    << ", \"total_energy_valid\": " << (scf.total_energy_valid ? "true" : "false")
    << ", \"onecenter_closure_diagnostic\": " << (scf.onecenter_closure_diagnostic ? "true" : "false")
    << ", \"onecenter_dij_reused\": " << (r.onecenter_dij_reused ? "true" : "false")
    << ", \"onecenter_dij_applications\": " << r.onecenter_dij_applications
    << ", \"onecenter_hamiltonian_density_source\": "
    << json_string(r.onecenter_dij_reused ? "final_scf_hamiltonian_input_density" : "scf_mixed_output_density")
    << ", \"onecenter_states\": ";
  write_onecenter_states(f, r.onecenter_states);
  f
    << ", \"reciprocal_basis\": [["
    << reciprocal_basis[0].x << ", " << reciprocal_basis[0].y << ", " << reciprocal_basis[0].z
    << "], [" << reciprocal_basis[1].x << ", " << reciprocal_basis[1].y << ", " << reciprocal_basis[1].z
    << "], [" << reciprocal_basis[2].x << ", " << reciprocal_basis[2].y << ", " << reciprocal_basis[2].z
    << "]] }\n}\n";
}
}

int main(int argc, char** argv) {
  auto& mpi = sdft::parallel();
  try {
    mpi.initialize(&argc, &argv);
    if(argc==2&&std::string(argv[1])=="--backend-info"){
      auto capabilities=sdft::compiled_compute_capabilities();
      // Append the process-parallel layout to the capability JSON object.
      capabilities.insert(capabilities.size()-1,",\"parallel\":"+mpi.description());
      std::cout<<capabilities<<'\n';mpi.finalize();return 0;
    }
    auto total_timer=std::make_unique<sdft::PerformanceTimer>("process_compute_total");
    if(const char* precision=std::getenv("SDFT_METAL_PRECISION"))
      if(std::string(precision)!="fp32x2")throw std::invalid_argument("SCF/NSCF requires fp32x2; direct fp32 is a standalone arithmetic benchmark only");
    sdft::compute_backend_name();
    if (argc < 4) throw std::runtime_error("usage: sdft_scf scf|bands|nscf|projector-stress|probe INPUT OUTPUT [FROZEN_SCF]");
    const std::string mode(argv[1]);
    if(mode=="probe") {
      const auto job=read_job(argv[2]);
      std::ofstream probe;
      if(mpi.is_root()){probe.open(argv[3]);if(!probe)throw std::runtime_error("cannot write density spectrum probe");}
      const auto text=sdft::density_spectrum_probe_json(job.input);
      if(mpi.is_root())probe<<text;
      mpi.finalize();
      return 0;
    }
    if(mode!="scf"&&mode!="bands"&&mode!="nscf"&&mode!="projector-stress")
      throw std::runtime_error("unknown calculation mode; use scf, bands, nscf or projector-stress");
    auto job = read_job(argv[2]);
    // Only world rank 0 writes the iteration log and result files.
    if(mpi.is_root())job.input.iteration_log=std::string(argv[3])+".iterations.jsonl";
    const bool projector_stress=std::string(argv[1])=="projector-stress";
    const bool frozen_mode=std::string(argv[1])=="nscf" || projector_stress;
    if(frozen_mode && argc!=5)throw std::runtime_error("frozen mode requires a frozen SCF payload");
    const auto scf = frozen_mode ? sdft::read_frozen_scf(argv[4],job.input) : sdft::run_scf(job.input);
    if (projector_stress) { if(mpi.is_root())sdft::write_projector_stress_diagnostic(job.input,scf,argv[3]); }
    else if (std::string(argv[1]) == "scf") { if(mpi.is_root())write_scf(argv[3], scf); }
    else if (std::string(argv[1]) == "bands" || frozen_mode) {
      if (job.path.empty()) throw std::runtime_error("bands mode requires a non-empty path");
      // Preserve the exact frozen potential state alongside the band data so
      // the SCF/NSCF handoff can be audited without rerunning the SCF.
      if(!frozen_mode&&mpi.is_root())write_scf((std::filesystem::path(argv[3]).parent_path() / "scf.json").string(), scf);
      // NSCF k points are independent, but run_nscf is not distributed
      // yet: rank 0 computes the path, other ranks only wait.
      if(mpi.is_root())write_bands(argv[3], sdft::run_nscf(job.input, scf, job.path, job.input.bands), job.input, scf);
    } else throw std::runtime_error("mode must be scf or bands");
    total_timer.reset();
    if(mpi.is_root()){
      sdft::record_performance("mpi_world_size",0,static_cast<std::size_t>(mpi.world_size()));
      sdft::write_performance_report(std::string(argv[3])+".performance.json");
    }
    mpi.barrier();
    mpi.finalize();
    return scf.converged ? 0 : 3;
  } catch (const std::exception& e) {
    std::cerr << "SDFT_ERROR: " << e.what() << '\n';
    // Any rank failing takes the whole job down with a nonzero code.
    if (mpi.enabled() && mpi.world_size() > 1) mpi.abort(2);
    mpi.finalize();
    return 2;
  }
}
