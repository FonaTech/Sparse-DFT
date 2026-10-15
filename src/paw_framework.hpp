#pragma once
#include "scf_core.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

namespace sdft {
enum class PawFramework { Potcar, Upf, Mixed };
inline PawFramework paw_framework(const std::filesystem::path& path) {
  std::ifstream stream(path,std::ios::binary);char magic[8]{};
  stream.read(magic,8);
  if(!stream)throw std::invalid_argument("cannot read PAW framework: "+path.string());
  const std::string tag(magic,8);
  if(tag=="SDFTPAW1")return PawFramework::Potcar;
  if(tag=="SDFTUPF1")return PawFramework::Upf;
  throw std::invalid_argument("unknown PAW container; prepare original POTCAR/UPF through the Sparse-DFT input adapter: "+path.string());
}
inline PawFramework homogeneous_paw_framework(const ScfInput& input) {
  if(input.atoms.empty())throw std::invalid_argument("SCF needs atoms");
  const auto framework=paw_framework(input.atoms.front().paw_cache);
  for(const auto& atom:input.atoms)
    if(paw_framework(atom.paw_cache)!=framework)
      throw std::invalid_argument("mixed UPF/POTCAR physical adapter not yet validated; refusing to substitute either atom's PAW");
  return framework;
}
inline PawFramework selected_paw_framework(const ScfInput& input) {
  if(input.atoms.empty())throw std::invalid_argument("SCF needs atoms");
  const auto first=paw_framework(input.atoms.front().paw_cache);
  for(const auto& atom:input.atoms)if(paw_framework(atom.paw_cache)!=first)return PawFramework::Mixed;
  return first;
}
inline AtomicUnits shared_paw_atomic_units(const ScfInput& input) {
  // UPF radii, beta and D/Q tables are stored in their original atomic units.
  // POTCAR tables use Angstrom/eV and also support the vasp_legacy constants.
  // A mixed model keeps the UPF convention; its POTCAR adapter converts to it.
  if(input.unit_convention!=AtomicUnitConvention::Codata &&
     (input.unit_convention!=AtomicUnitConvention::VaspLegacy ||
      selected_paw_framework(input)!=PawFramework::Potcar))
    throw std::invalid_argument("UPF/mixed PAW requires original CODATA Bohr/Ry units; VASP legacy units require pure POTCAR");
  return atomic_units(input.unit_convention);
}
inline bool use_shared_site_kernel(const ScfInput& input) {
  const auto framework=selected_paw_framework(input);
  const char* flag=std::getenv("SDFT_PAW_SHARED_KERNEL");
  if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")
    throw std::invalid_argument("SDFT_PAW_SHARED_KERNEL must be 0 or 1");
  // The override is for pure-POTCAR limit validation against the preserved
  // implementation. UPF and mixed inputs always select their site adapters.
  return framework!=PawFramework::Potcar||(flag&&std::string(flag)=="1");
}
ScfResult run_upf_scf(const ScfInput& input);
BandPathResult run_upf_nscf(const ScfInput&,const ScfResult&,const std::vector<Vec3>&,int);
}
