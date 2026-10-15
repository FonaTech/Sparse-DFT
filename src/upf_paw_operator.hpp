#pragma once
#include "upf_paw_data.hpp"
#include "plane_wave_operator.hpp"
#include "paw_augmentation.hpp"
#include <memory>
#include <tuple>

namespace sdft {

// Geometry/operator interface for the original UPF PAW representation.
// H and S use the common B^dagger D B / B^dagger Q B operator blocks, but
// radial beta and pair-dependent QijL are kept in Bohr/Ry UPF conventions.
// An operator object alone does not implement the PAW SCF energy closure.
class UpfPawOperator {
 public:
  UpfPawOperator(std::shared_ptr<const UpfPawData> data);
  std::size_t dimension() const { return expanded_.size(); }
  ComplexPawOperator nonlocal_block(const std::vector<Vec3>& g_plus_k,
      double volume_bohr3,Vec3 position_bohr) const;
  // Integral of Q_ab(r) exp(-iG.r), before position phase/cell volume.
  std::vector<std::complex<double>> augmentation_form(Vec3 g) const;
  std::array<std::vector<std::complex<double>>,3> augmentation_form_gradient(Vec3 g) const;
  struct RadialForms {
    double magnitude{};
    std::map<std::tuple<std::size_t,std::size_t,int>,std::pair<double,double>> pairs;
  };
  // Immutable exact-radius data. No interpolation, radius rounding or grid
  // truncation: the same scalar integrals are shared by angular directions.
  RadialForms radial_forms(double magnitude) const;
  std::vector<std::complex<double>> augmentation_form(Vec3 g,const RadialForms&) const;
  std::array<std::vector<std::complex<double>>,3> augmentation_form_gradient(Vec3 g,const RadialForms&) const;
  // Radial Y_lm density coefficients already multiplied by r^2. The UPF
  // pseudo one-center density contains PS partial-wave products plus QijL.
  std::vector<std::vector<double>> radial_density(
      const std::vector<std::complex<double>>& onsite,bool all_electron) const;
  // Exact transpose of radial_density for weighted radial potential values.
  std::vector<std::complex<double>> project_radial_potential(
      const std::vector<std::vector<double>>& weighted_potential,bool all_electron) const;
  int density_lmax() const { return 2*maximum_l_; }
 private:
  struct Expanded {std::size_t radial;int l,m;};
  struct Coupling {std::size_t a,b,i,j;int l,m;double gaunt;};
  std::shared_ptr<const UpfPawData> data_;
  std::vector<Expanded> expanded_;
  std::vector<Coupling> coupling_;
  struct RadialPair {
    std::vector<double> augmentation,all_electron,pseudo;
  };
  std::map<std::tuple<std::size_t,std::size_t,int>,RadialPair> radial_pairs_;
  std::vector<double> radial_weights_;
  int maximum_l_{};
};
} // namespace sdft
