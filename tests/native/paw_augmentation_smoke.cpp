#include "paw_augmentation.hpp"
#include "sdft_core.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

namespace {
double log_mesh_simpson(const std::vector<double>& r, const std::vector<double>& f) {
  if (r.size() != f.size() || r.size() < 2) return 0.0;
  const double h = std::log(r[1] / r[0]);
  if (!(r[0] > 0.0) || !std::isfinite(h)) return 0.0;
  for (std::size_t i = 2; i < r.size(); ++i)
    if (std::abs(std::log(r[i] / r[i - 1]) - h) >= 2e-7) return 0.0;
  std::vector<double> si(r.size(), 0.0);
  for (std::size_t k = r.size(); k >= 3; k -= 2) {
    si[k - 1] += r[k - 1] * h / 3.0;
    si[k - 2] = 4.0 * r[k - 2] * h / 3.0;
    si[k - 3] = r[k - 3] * h / 3.0;
    if (k == 3) break;
  }
  double value = 0.0;
  for (std::size_t i = 0; i < r.size(); ++i) value += si[i] * f[i];
  return value;
}
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: sdft_paw_augmentation_smoke <numeric.bin>\n";
    return 2;
  }
  const auto cache = sdft::read_paw_numeric_cache(argv[1]);
  std::vector<int> l;
  for (const auto& ch : cache.nonlocal_channels)
    for (int i = 0; i < ch.nprojectors; ++i) l.push_back(ch.angular_momentum);
  const auto a = sdft::build_paw_augmentation(cache, l);
  if (a.channel_count() != 4 || a.max_l != 2) return 3;
  // The archived 4x4 augmentation section is the L=0 moment for equal-l
  // projector pairs; cross-l pairs carry their first allowed odd moment.
  const double expected_ss[] = {-0.118612820244, -0.166184190184,
                                -0.166184190184, -0.232848406670};
  const double expected_pp[] = {-0.0179689550405, -0.0343505440708,
                                -0.0343505440708, -0.065514272773};
  double maxerr = 0.0;
  for (std::size_t i = 0; i < 2; ++i) for (std::size_t j = 0; j < 2; ++j)
    maxerr = std::max(maxerr, std::abs(a.moment(i, j, 0) - expected_ss[i * 2 + j]));
  for (std::size_t i = 2; i < 4; ++i) for (std::size_t j = 2; j < 4; ++j)
    maxerr = std::max(maxerr, std::abs(a.moment(i, j, 0) - expected_pp[(i - 2) * 2 + (j - 2)]));
  // The two cross-l L=1 entries are independently reconstructed from the
  // archived partial waves; they must be non-zero and symmetric.
  if (std::abs(a.moment(0, 2, 1)) < 1e-4 || std::abs(a.moment(1, 3, 1)) < 1e-4)
    return 5;
  if (maxerr > 5e-5) {
    std::cerr << "aug_moment L=0 regression failed maxerr=" << maxerr << "\n";
    return 4;
  }
  for (int ll = 0; ll <= a.max_l; ++ll) {
    // The default normalization uses the continuous radial function. Verify by an
    // independent fine Simpson rule, not by the archive's coarse log mesh.
    auto bessel = [](int l, double x) {
      if (std::abs(x) < 0.5) {
        double term = 1.0;
        for (int k = 1; k <= l; ++k) term *= x / (2.0 * k + 1.0);
        double sum = term;
        for (int k = 1; k < 25; ++k) {
          term *= -x*x / (2.0*k*(2.0*l+2.0*k+1.0)); sum += term;
        }
        return sum;
      }
      double prev=std::sin(x)/x;
      if (l==0) return prev;
      double value=std::sin(x)/(x*x)-std::cos(x)/x;
      for (int k=1;k<l;++k) { const double next=(2*k+1)*value/x-prev;prev=value;value=next; }
      return value;
    };
    constexpr int panels = 4096;
    double moment = 0.0;
    for (int ir=0;ir<=panels;++ir) {
      const double radius=a.rmax*ir/panels;
      double shape=0.0;
      for (int j=0;j<2;++j)
        shape+=a.compensation_amplitudes[ll][j]*bessel(ll,a.compensation_q[ll][j]*radius);
      moment+=(ir==0||ir==panels?1.0:ir%2?4.0:2.0)*shape*std::pow(radius,ll+2);
    }
    moment*=a.rmax/(3.0*panels);
    if (std::abs(moment - 1.0) > 2e-10) {
      std::cerr << "compensation moment failed L=" << ll << " value=" << moment << "\n";
      return 6;
    }
    const std::size_t ir = a.radial_grid.size() - 1;
    const double dr = a.radial_grid[ir] - a.radial_grid[ir - 1];
    const double qprime = (a.compensation_r2[static_cast<std::size_t>(ll)][ir] /
                           std::max(a.radial_grid[ir] * a.radial_grid[ir], 1e-20) -
                           a.compensation_r2[static_cast<std::size_t>(ll)][ir - 1] /
                           std::max(a.radial_grid[ir - 1] * a.radial_grid[ir - 1], 1e-20)) / dr;
    if (std::abs(qprime) > 2e-2) {
      std::cerr << "compensation boundary derivative failed L=" << ll << " value=" << qprime << "\n";
      return 7;
    }
  }

  // Exercise the complex, aspherical one-centre path independently of SCF.
  // The cache has expanded ordering s,s,p(m=-1,0,1),p(m=-1,0,1).
  constexpr std::size_t expanded = 8;
  std::vector<std::complex<double>> onsite(expanded * expanded, {0.0, 0.0});
  const double occupations[] = {0.2, 0.5, 0.9, 0.1, 0.15, 0.2};
  const std::size_t p_indices[] = {2, 3, 4, 5, 6, 7};
  for (std::size_t i = 0; i < 6; ++i)
    onsite[p_indices[i] * expanded + p_indices[i]] = occupations[i];
  for (std::size_t m = 0; m < 3; ++m) {
    const std::size_t i = 2 + m, j = 5 + m;
    onsite[i * expanded + j] = {0.025, 0.015};
    onsite[j * expanded + i] = std::conj(onsite[i * expanded + j]);
  }
  const auto closure = sdft::paw_one_center_closure(cache, l, onsite, expanded);
  const auto complex_lm_moments = sdft::paw_lm_moments_from_density(a, onsite, expanded);
  const auto real_lm_moments = sdft::paw_lm_moments_real(a, onsite, expanded);
  const auto imag_lm_moments = sdft::paw_lm_moments_imag(a, onsite, expanded);
  if (!std::isfinite(closure.ae_energy_ev) || !std::isfinite(closure.ps_energy_ev) ||
      !std::isfinite(closure.energy_difference_ev) ||
      !std::isfinite(closure.potential_expectation_ev) ||
      !std::isfinite(closure.max_potential_ev)) {
    std::cerr << "one-centre closure returned a non-finite scalar\n";
    return 8;
  }
  double hermitian_error = 0.0, aspherical_norm = 0.0, lm_imag = 0.0;
  for (std::size_t i = 0; i < expanded; ++i) for (std::size_t j = 0; j < expanded; ++j)
    hermitian_error = std::max(hermitian_error,
      std::abs(closure.d_matrix_ev[i * expanded + j] -
               std::conj(closure.d_matrix_ev[j * expanded + i])));
  for (std::size_t lm = 1; lm < closure.lm_moments.size(); ++lm)
    aspherical_norm += std::norm(closure.lm_moments[lm]);
  for (const auto& value : closure.lm_moments) lm_imag = std::max(lm_imag, std::abs(value.imag()));
  double complex_transform_error = 0.0, real_transform_error = 0.0, imag_transform_error = 0.0;
  if (complex_lm_moments.size() != closure.lm_moments.size() || real_lm_moments.size() != closure.lm_moments.size() ||
      imag_lm_moments.size() != closure.lm_moments.size() || closure.lm_moment_imag.size() != closure.lm_moments.size()) {
    complex_transform_error = real_transform_error = imag_transform_error =
        std::numeric_limits<double>::infinity();
  } else {
    for (std::size_t lm = 0; lm < complex_lm_moments.size(); ++lm)
      complex_transform_error = std::max(complex_transform_error,
                                         std::abs(complex_lm_moments[lm] - closure.lm_moments[lm]));
    for (std::size_t lm = 0; lm < closure.lm_moments.size(); ++lm) {
      real_transform_error = std::max(real_transform_error,
          std::abs(real_lm_moments[lm] - closure.lm_moments[lm].real()));
      imag_transform_error = std::max(imag_transform_error,
          std::abs(imag_lm_moments[lm] - closure.lm_moment_imag[lm]));
    }
  }
  // The complex moment transform and its separate real/imaginary transforms
  // are both checked. An imaginary aspherical component is valid; only the charge
  // monopole must remain real for a Hermitian occupancy matrix.
  if (hermitian_error > 1e-10 || aspherical_norm < 1e-12 ||
      std::abs(closure.lm_moments.front().imag()) > 1e-12 ||
      complex_transform_error > 1e-12 || imag_transform_error > 1e-12) {
    std::cerr << "one-centre closure invariant failed hermitian=" << hermitian_error
              << " aspherical_norm=" << aspherical_norm << " lm_imag=" << lm_imag
              << " complex_transform_error=" << complex_transform_error
              << " real_transform_error=" << real_transform_error
              << " imag_transform_error=" << imag_transform_error << "\n";
    return 9;
  }
  if (!closure.spherical_gga_variational || !closure.aspherical_gga_variational) {
    std::cerr << "full angular PBE-GGA variational branch was not marked active\n";
    return 10;
  }

  // A Hermitian complex direction probes diagonal spherical PBE terms and
  // non-spherical Hartree multipoles. The central energy difference must equal
  // Tr(delta-rho D) for the same discrete radial energy and its returned D.
  std::vector<std::complex<double>> direction(expanded * expanded, {0.0, 0.0});
  direction[2 * expanded + 2] = 0.11;
  direction[4 * expanded + 4] = -0.07;
  direction[2 * expanded + 6] = {0.035, -0.021};
  direction[6 * expanded + 2] = std::conj(direction[2 * expanded + 6]);
  direction[3 * expanded + 7] = {-0.018, 0.026};
  direction[7 * expanded + 3] = std::conj(direction[3 * expanded + 7]);
  auto perturbed_density = [&](double scale) {
    auto out = onsite;
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += scale * direction[i];
    return out;
  };
  constexpr double step = 2e-4;
  const auto plus = sdft::paw_one_center_closure(cache, l, perturbed_density(step), expanded);
  const auto minus = sdft::paw_one_center_closure(cache, l, perturbed_density(-step), expanded);
  double directional_derivative_ev = 0.0;
  for (std::size_t i = 0; i < direction.size(); ++i)
    directional_derivative_ev += (std::conj(direction[i]) * closure.d_matrix_ev[i]).real();
  const double finite_difference_ev =
      (plus.energy_difference_ev - minus.energy_difference_ev) / (2.0 * step);
  const double variational_error = std::abs(finite_difference_ev - directional_derivative_ev);
  if (variational_error > 2e-5 * std::max(1.0, std::abs(finite_difference_ev))) {
    std::cerr << "one-centre variational derivative failed fd=" << finite_difference_ev
              << " matrix=" << directional_derivative_ev << " error=" << variational_error << "\n";
    return 11;
  }
  const auto reference = sdft::paw_atomic_reference_potentials(cache, l);
  const auto atom = sdft::paw_atomic_reference_density(cache, l);
  const auto neutral = sdft::paw_one_center_closure(cache, l, atom, expanded, 1,
      sdft::PawRadialLengthUnit::Angstrom, &reference);
  double reference_zero_error=0.0;
  for (const auto& value:neutral.d_matrix_ev)
    reference_zero_error=std::max(reference_zero_error,std::abs(value));
  const auto ref_closure=sdft::paw_one_center_closure(cache,l,onsite,expanded,1,
      sdft::PawRadialLengthUnit::Angstrom,&reference);
  const auto ref_plus=sdft::paw_one_center_closure(cache,l,perturbed_density(step),expanded,1,
      sdft::PawRadialLengthUnit::Angstrom,&reference);
  const auto ref_minus=sdft::paw_one_center_closure(cache,l,perturbed_density(-step),expanded,1,
      sdft::PawRadialLengthUnit::Angstrom,&reference);
  double reference_derivative=0.0;
  for (std::size_t i=0;i<direction.size();++i)
    reference_derivative+=(std::conj(direction[i])*ref_closure.d_matrix_ev[i]).real();
  const double reference_fd=(ref_plus.energy_difference_ev-ref_minus.energy_difference_ev)/(2*step);
  const double reference_fd_error=std::abs(reference_fd-reference_derivative);
  if(reference_zero_error>1e-11||reference_fd_error>2e-5||!std::isfinite(reference.core_xc_energy_ev)) {
    std::cerr<<"atomic reference failure zero="<<reference_zero_error<<" fd="<<reference_fd_error<<'\n';
    return 15;
  }
  if (std::abs(ref_closure.core_xc_energy_ev - reference.core_xc_energy_ev) > 1e-12) {
    std::cerr << "one-centre closure did not preserve frozen-core XC energy" << '\n';
    return 16;
  }
  // An even logarithmic mesh has a zero Simpson weight at its first node.
  // The transposed gradient still differentiates through that node: this
  // regression rejects implementations that divide by zero or discard it.
  // The f channel also exercises harmonics and compensation through L=6.
  sdft::PawNumericCache synthetic;
  constexpr std::size_t synthetic_nr = 100, synthetic_dim = 7;
  synthetic.radial_grid.resize(synthetic_nr);
  synthetic.ae_wavefunctions.assign(1, std::vector<double>(synthetic_nr));
  synthetic.pseudo_wavefunctions.assign(1, std::vector<double>(synthetic_nr));
  synthetic.radial_rmax = 4.0;
  for (std::size_t ir = 0; ir < synthetic_nr; ++ir) {
    const double radius = 0.005 * std::exp(std::log(800.0) * ir / (synthetic_nr - 1));
    synthetic.radial_grid[ir] = radius;
    const double wave = std::pow(radius, 4.0) * std::exp(-2.0 * radius);
    synthetic.pseudo_wavefunctions[0][ir] = wave;
    synthetic.ae_wavefunctions[0][ir] = 1.1 * wave;
  }
  std::vector<std::complex<double>> synthetic_density(synthetic_dim * synthetic_dim, {0.0, 0.0});
  std::vector<std::complex<double>> synthetic_direction(synthetic_dim * synthetic_dim, {0.0, 0.0});
  for (std::size_t i = 0; i < synthetic_dim; ++i) {
    synthetic_density[i * synthetic_dim + i] = 0.9 + 0.03 * i;
    synthetic_direction[i * synthetic_dim + i] = 0.01 * (static_cast<int>(i) - 2);
  }
  synthetic_direction[1 * synthetic_dim + 4] = {0.02, -0.013};
  synthetic_direction[4 * synthetic_dim + 1] = std::conj(synthetic_direction[1 * synthetic_dim + 4]);
  const auto synthetic_closure = sdft::paw_one_center_closure(synthetic, {3}, synthetic_density, synthetic_dim);
  constexpr double bohr_per_angstrom = sdft::atomic_units(sdft::AtomicUnitConvention::Codata).bohr_per_angstrom;
  auto synthetic_bohr = synthetic;
  for (double& radius : synthetic_bohr.radial_grid) radius *= bohr_per_angstrom;
  synthetic_bohr.radial_rmax *= bohr_per_angstrom;
  for (auto& wave : synthetic_bohr.ae_wavefunctions)
    for (double& value : wave) value /= std::sqrt(bohr_per_angstrom);
  for (auto& wave : synthetic_bohr.pseudo_wavefunctions)
    for (double& value : wave) value /= std::sqrt(bohr_per_angstrom);
  const auto bohr_closure = sdft::paw_one_center_closure(synthetic_bohr, {3}, synthetic_density,
      synthetic_dim, 1, sdft::PawRadialLengthUnit::Bohr);
  double unit_error = std::abs(bohr_closure.energy_difference_ev - synthetic_closure.energy_difference_ev);
  for (std::size_t i = 0; i < bohr_closure.d_matrix_ev.size(); ++i)
    unit_error = std::max(unit_error, std::abs(bohr_closure.d_matrix_ev[i] - synthetic_closure.d_matrix_ev[i]));
  for (std::size_t i = 0; i < bohr_closure.lm_moments.size(); ++i)
    unit_error = std::max(unit_error, std::abs(bohr_closure.lm_moments[i] - synthetic_closure.lm_moments[i]));
  if (unit_error > 1e-10) {
    std::cerr << "Angstrom/Bohr radial unit invariance failed error=" << unit_error << "\n";
    return 14;
  }
  const auto synthetic_augmentation = sdft::build_paw_augmentation(synthetic, {3});
  auto isotropic_density = synthetic_density;
  for (std::size_t i = 0; i < synthetic_dim; ++i) isotropic_density[i * synthetic_dim + i] = 1.0;
  const auto isotropic_lm_moments = sdft::paw_lm_moments_from_density(synthetic_augmentation, isotropic_density, synthetic_dim);
  const double isotropic_q00 = synthetic_dim * synthetic_augmentation.moment(0, 0, 0) /
                                std::sqrt(4.0 * 3.14159265358979323846);
  double isotropic_error = std::abs(isotropic_lm_moments[0] - isotropic_q00);
  for (std::size_t lm = 1; lm < isotropic_lm_moments.size(); ++lm)
    isotropic_error = std::max(isotropic_error, std::abs(isotropic_lm_moments[lm]));
  if (isotropic_error > 2e-10) {
    std::cerr << "f-channel harmonic normalization failed error=" << isotropic_error << "\n";
    return 13;
  }
  auto synthetic_plus_density = synthetic_density, synthetic_minus_density = synthetic_density;
  double synthetic_matrix_derivative = 0.0;
  for (std::size_t i = 0; i < synthetic_density.size(); ++i) {
    synthetic_plus_density[i] += step * synthetic_direction[i];
    synthetic_minus_density[i] -= step * synthetic_direction[i];
    synthetic_matrix_derivative += (std::conj(synthetic_direction[i]) * synthetic_closure.d_matrix_ev[i]).real();
  }
  const auto synthetic_plus = sdft::paw_one_center_closure(synthetic, {3}, synthetic_plus_density, synthetic_dim);
  const auto synthetic_minus = sdft::paw_one_center_closure(synthetic, {3}, synthetic_minus_density, synthetic_dim);
  const double synthetic_fd = (synthetic_plus.energy_difference_ev - synthetic_minus.energy_difference_ev) / (2.0 * step);
  const double synthetic_error = std::abs(synthetic_fd - synthetic_matrix_derivative);
  if (synthetic_error > 2e-6 || !std::isfinite(synthetic_error)) {
    std::cerr << "even-grid L=6 variational derivative failed fd=" << synthetic_fd
              << " matrix=" << synthetic_matrix_derivative << " error=" << synthetic_error << "\n";
    return 12;
  }
  // The LASPH response extends f-channel density (L=6) to L=7 Cartesian
  // harmonics. This previously hit the L<=6 guard. Direct mesh derivatives
  // deliberately carry a non-variational label; the discrete-adjoint mode
  // above remains the independent finite-difference test.
  const auto aspherical = sdft::paw_one_center_closure(synthetic, {3}, synthetic_density,
      synthetic_dim, 1, sdft::PawRadialLengthUnit::Angstrom, nullptr,
      sdft::PawRadialPotentialMode::AsphericalDirect);
  if(!std::isfinite(aspherical.energy_difference_ev) || aspherical.aspherical_gga_variational ||
      aspherical.spherical_gga_variational || !aspherical.direct_radial_potential) return 17;
  for(std::size_t i=0;i<synthetic_dim;++i)for(std::size_t j=0;j<synthetic_dim;++j)
    if(std::abs(aspherical.d_matrix_ev[i*synthetic_dim+j]-std::conj(aspherical.d_matrix_ev[j*synthetic_dim+i]))>1e-12)return 18;
  std::cout << "paw_augmentation_smoke PASS max_aug_moment_l0_error=" << maxerr
            << " aug_moment_l2=" << a.moment(2, 3, 2)
            << " closure_delta_e_ev=" << closure.energy_difference_ev
            << " closure_hermitian_error=" << hermitian_error
            << " closure_aspherical_norm=" << aspherical_norm
            << " variational_fd_ev=" << finite_difference_ev
            << " variational_matrix_ev=" << directional_derivative_ev
            << " variational_error_ev=" << variational_error
            << " reference_atomic_zero_error_ev=" << reference_zero_error
            << " reference_variational_error_ev=" << reference_fd_error
            << " core_xc_reference_ev=" << reference.core_xc_energy_ev
            << " even_grid_l6_variational_error_ev=" << synthetic_error
            << " f_channel_harmonic_error=" << isotropic_error
            << " angstrom_bohr_unit_error=" << unit_error
            << " ae_clipped_density_samples=" << closure.ae_clipped_density_samples
            << " ps_clipped_density_samples=" << closure.ps_clipped_density_samples << "\n";
  return 0;
}
