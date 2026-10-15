#pragma once

#include "density_functionals.hpp"
#include "paw_augmentation.hpp"
#include <memory>

namespace sdft {

// Linear map from the one-centre occupation to the Fourier compensation
// density, via aug_moment and g_L, and its adjoint. Radial arrays retain PAW54 Angstrom
// units; Fourier densities, potentials and positions use atomic units.
class PawCompensationOperator {
 public:
  PawCompensationOperator(const PawAugmentationData& augmentation,
      const std::array<Vec3, 3>& lattice_bohr, Vec3 fractional_position,
      const std::vector<std::tuple<int, int, int>>& reciprocal_keys,
      AtomicUnitConvention constants = AtomicUnitConvention::Codata,
      const PawCompensationOperator* shared_geometry = nullptr);

  FourierDensity density(const std::vector<std::complex<double>>& onsite) const;
  // All atoms use the same reciprocal support. Contract each G independently
  // in deterministic atom order, then build one map instead of one per atom.
  static FourierDensity density_sum(const std::vector<PawCompensationOperator>& atoms,
      const std::vector<std::vector<std::complex<double>>>& onsite);
  std::vector<std::complex<double>> project_potential(
      const FourierDensity& potential_hartree) const;
  // Sample V[-G] once for all atoms; parallelize independent atoms while
  // retaining the serial G/lm contraction order inside each atom.
  static std::vector<std::vector<std::complex<double>>> project_potential_batch(
      const std::vector<PawCompensationOperator>& atoms,
      const FourierDensity& potential_hartree);
  // Explicit position derivative at fixed on-site density and grid potential;
  // this is one force contribution, not the full PAW force.
  Vec3 position_energy_gradient(
      const std::vector<std::complex<double>>& onsite,
      const FourierDensity& potential_hartree) const;
  // Strain derivative of the radial/angular shape, excluding the universal
  // 1/volume term already included in smooth-functional stress. Phase G.R
  // is invariant under a homogeneous strain at fixed fractional positions.
  std::array<Vec3,3> shape_strain_energy_gradient(
      const std::vector<std::complex<double>>& onsite,
      const FourierDensity& potential_hartree) const;
  struct Derivatives {
    Vec3 position;
    std::array<Vec3,3> shape;
  };
  // Sample the true V[-G] support once, then preserve each atom's original
  // G/lm summation order. Missing Fourier modes remain missing.
  static std::vector<Derivatives> energy_derivatives_batch(
      const std::vector<PawCompensationOperator>& atoms,
      const std::vector<std::vector<std::complex<double>>>& onsite,
      const FourierDensity& potential_hartree);
  const std::vector<std::tuple<int,int,int>>& reciprocal_keys() const { return geometry_->keys; }
  const std::vector<Vec3>& reciprocal_vectors() const { return geometry_->reciprocal_vectors; }
  const std::vector<std::complex<double>>& translation_phases() const { return phases_; }
  std::size_t dimension() const { return dimension_; }
  std::size_t form_factor_count() const;
  std::size_t geometry_storage_bytes() const;
  std::size_t atom_storage_bytes() const;
  bool shares_geometry_with(const PawCompensationOperator& other) const;
  // Origin-centred Q_ij(G), without translation or 1/volume. This exposes
  // the original linear compensation map to the shared multi-format SCF;
  // it does not re-integrate or approximate any PAW radial data.
  std::vector<std::complex<double>> reciprocal_matrix(std::size_t g, int derivative_axis=-1) const;

 private:
  PawAugmentationData augmentation_;
  std::vector<std::complex<double>> multipoles(
      const std::vector<std::complex<double>>& onsite) const;
  double volume_{};
  std::size_t dimension_{}, multipole_count_{};
  struct Geometry {
    AtomicUnitConvention constants;
    std::array<Vec3,3> lattice;
    std::vector<std::tuple<int,int,int>> keys;
    std::vector<Vec3> reciprocal_vectors;
    std::vector<std::complex<double>> form_factors;
    std::array<std::vector<std::complex<double>>,3> shape_gradients;
    // exp(-i G.r) angular phase is exactly (-i)^L; the nonzero
    // component is still FP64 and is signed. This is lossless storage,
    // independent of whether onsite density or the potential is complex.
    std::vector<double> compact_form;
    std::array<std::vector<double>,3> compact_gradients;
    std::vector<unsigned char> imaginary_channel;
    std::vector<std::complex<double>> contract_density_tile(std::size_t first,
        std::size_t count,const std::vector<std::complex<double>>& coefficients,
        std::size_t columns,int axis=-1) const;
    std::vector<std::complex<double>> contract_potential_tile(std::size_t first,
        std::size_t count,const std::vector<std::complex<double>>& phased,
        std::size_t columns) const;
    std::complex<double> form(std::size_t index,std::size_t lm) const {
      if(compact_form.empty())return form_factors[index];
      return imaginary_channel[lm] ? std::complex<double>(0,compact_form[index]) : std::complex<double>(compact_form[index],0);
    }
    std::complex<double> gradient(int axis,std::size_t index,std::size_t lm) const {
      if(compact_form.empty())return shape_gradients[axis][index];
      return imaginary_channel[lm] ? std::complex<double>(0,compact_gradients[axis][index]) : std::complex<double>(compact_gradients[axis][index],0);
    }
  };
  std::shared_ptr<const Geometry> geometry_;
  static std::vector<std::vector<std::size_t>> geometry_groups(
      const std::vector<PawCompensationOperator>& atoms);
  // Translation is the only per-atom grid-sized array. The same immutable
  // PAW/angular/radial geometry is reused at all matching atom positions.
  std::vector<std::complex<double>> phases_;
  // [lm, i, j], small and density-independent; [G,lm] form factors are cached.
  // Keep the complex transform: a complex plane-wave solver represents the
  // real and imaginary (L,M) moments together. Dropping the imaginary part here
  // is only valid at Gamma and corrupts non-Gamma augmentation.
  std::vector<std::complex<double>> density_transform_;
};

// Real harmonics, m=-l..l with sin(|m|phi) for m<0 and cos(m phi) for m>0,
// without the Condon-Shortley phase, through L=6; for l<=3 this is the VASP
// PROCAR column order.
double paw_real_spherical_harmonic(int l, int m, Vec3 direction);
// Radial compensation shape g_L(q) for L=0..max_l, q in 1/Angstrom, with
// g_0(0)=1. Diagnostic only (density-spectrum probe).
std::vector<double> paw_compensation_radial_form(const PawAugmentationData& augmentation,double q_angstrom);
Vec3 paw_real_spherical_harmonic_gradient(int l, int m, Vec3 direction);

} // namespace sdft
