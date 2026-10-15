#pragma once

#include "sdft_core.hpp"
#include <complex>
#include <map>
#include <tuple>
#include <vector>

namespace sdft {
using FourierDensity = std::map<std::tuple<int, int, int>, std::complex<double>>;

// Band-major plane-wave coefficients. Computes the smooth density by FFT,
// with an alias-free product grid and the requested shared Fourier support.
// Bloch k phases cancel in |psi|^2. Occupations exclude k-point weights.
FourierDensity plane_wave_density_fft(
    const std::vector<std::tuple<int,int,int>>& g,
    const std::vector<std::complex<double>>& coefficients,
    const std::vector<double>& occupations, double volume_bohr3,
    const FourierDensity& support);
// Exact integer G-G' support via FFT autocorrelation of the basis mask.
// Avoids enumerating NPW^2 pairs while preserving the old spherical support.
FourierDensity plane_wave_difference_support(const std::vector<std::tuple<int,int,int>>& g);

struct DensityFunctionals {
  FourierDensity potential;
  FourierDensity xc_potential;
  double hartree_energy{};
  double xc_energy{};
  double vxc_density{};
  // dE/d(strain)/volume for a fixed fractional Fourier density whose
  // coefficients scale as 1/volume. Atomic shape responses are additional.
  std::array<Vec3,3> hartree_stress{}, xc_stress{};
  double sampled_electrons{};
  // Minimum of the valence+core density sampled by XC.
  double minimum_density{};
  std::size_t clipped_density_points{};
  std::size_t grid_points{};
};

// Atomic units, Fourier convention n(r)=sum_G n_G exp(+iG.r). Hartree
// and vxc_density use density only; PBE uses density+frozen_pseudo_core.
// Both potentials cover the union of the two Fourier supports, permitting
// a core-only mode in the explicit derivative of the frozen-core XC energy.
// The G=0 Coulomb term is excluded for the neutral periodic background.
DensityFunctionals evaluate_density_functionals(const FourierDensity& density,
    const std::array<Vec3, 3>& lattice_bohr,
    const FourierDensity* frozen_pseudo_core = nullptr, int quadrature_padding = 0, bool allow_accelerator = false);

// Each channel has the common Hartree potential and its own spin-XC Euler
// potential. Energies/stresses are split equally so summing channels counts
// Hartree and XC once; vxc_density contracts the actual channel density.
std::array<DensityFunctionals,2> evaluate_collinear_density_functionals(
    const std::array<FourierDensity,2>& density,
    const std::array<Vec3,3>& lattice_bohr,
    const FourierDensity* frozen_pseudo_core=nullptr,int quadrature_padding=0);

inline DensityFunctionals evaluate_density_functionals(const FourierDensity& density,
    const std::array<Vec3, 3>& lattice_bohr, const FourierDensity& frozen_pseudo_core, int quadrature_padding = 0, bool allow_accelerator = false) {
  return evaluate_density_functionals(density, lattice_bohr, &frozen_pseudo_core, quadrature_padding, allow_accelerator);
}
}
