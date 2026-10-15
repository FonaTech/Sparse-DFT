#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace sdft {

// This model carries original UPF Bohr/Ry/r*phi definitions. In particular,
// the QijL arrays are pair-dependent, unlike the POTCAR compensation shape.
// A successful read is data validation only, never a material-SCF verdict.
struct UpfPawData {
  std::string element, functional, relativistic, original_sha256;
  double z_valence{}, core_energy_ry{},augmentation_epsilon{};
  std::size_t augmentation_count{}, integration_count{};
  std::vector<int> projector_l;
  // Original PP_CHI attributes, typed in native format v2 for QE pseudo
  // Hubbard subspace selection. V1 remains readable for U=0 calculations.
  std::vector<int> atomic_l;
  std::vector<double> atomic_occupations;
  std::vector<std::size_t> projector_cutoff;
  std::map<std::string,std::vector<double>> arrays;
  std::string original_xml, metadata_json;

  const std::vector<double>& array(const std::string& path) const;
  std::size_t projector_count() const { return projector_l.size(); }
};

UpfPawData read_native_upf_paw(const std::filesystem::path& path);
std::vector<double> upf_simpson_weights(const std::vector<double>& rab,std::size_t count);
double upf_spherical_bessel(int l,double x);
double upf_spherical_bessel_derivative(int l,double x);
double upf_projector_form(const UpfPawData& data,std::size_t i,double g,bool derivative=false);
std::vector<double> upf_augmentation_radial(const UpfPawData& data,std::size_t i,std::size_t j,int l);
double upf_augmentation_form(const UpfPawData& data,std::size_t i,std::size_t j,int l,double g,bool derivative=false);
double upf_local_form_ry(const UpfPawData& data,double g,bool derivative=false);

} // namespace sdft
