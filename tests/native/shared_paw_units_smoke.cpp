#include "paw_framework.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
  using namespace sdft;
  const auto directory=std::filesystem::temp_directory_path()/
      ("sparse-dft-paw-unit-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  struct Cleanup {std::filesystem::path p;~Cleanup(){std::filesystem::remove_all(p);}} cleanup{directory};
  const auto potcar=directory/"potcar",upf=directory/"upf";
  std::ofstream(potcar,std::ios::binary)<<"SDFTPAW1";
  std::ofstream(upf,std::ios::binary)<<"SDFTUPF1";
  ScfInput input;input.atoms.resize(1);input.atoms[0].paw_cache=potcar;
  for(const auto convention:{AtomicUnitConvention::Codata,AtomicUnitConvention::VaspLegacy}) {
    input.unit_convention=convention;
    const auto units=shared_paw_atomic_units(input);
    const double q=2*3.14159265358979323846/(6*units.bohr_per_angstrom);
    const std::array<Vec3,3> rec{Vec3{q,0,0},Vec3{0,q,0},Vec3{0,0,q}};
    const Vec3 k{.1,.07,.03};
    const auto basis=enumerate_g_vectors_fixed_reference(rec,rec,80.,k,units.ev_per_hartree);
    for(const auto& g:basis) {
      const Vec3 shifted{q*(g.h+k.x),q*(g.k+k.y),q*(g.l+k.z)};
      if(std::abs(g.kinetic_ev-.5*dot(shifted,shifted)*units.ev_per_hartree)>1e-12)return 1;
    }
  }
  auto rejected=[&]() {
    try {shared_paw_atomic_units(input);return false;}
    catch(const std::invalid_argument&) {return true;}
  };
  input.atoms[0].paw_cache=upf;
  if(!rejected())return 2;
  input.atoms.push_back(input.atoms[0]);input.atoms[1].paw_cache=potcar;
  if(!rejected())return 3;
  input.unit_convention=AtomicUnitConvention::Codata;
  if(shared_paw_atomic_units(input).ev_per_hartree!=27.211386245988)return 4;
  input.unit_convention=AtomicUnitConvention::MatchRadialMode;
  if(!rejected())return 5;
  std::cout<<"PASS pure POTCAR unit conventions, Bloch kinetic conversion, UPF/mixed unit rejection\n";
}
