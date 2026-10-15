#pragma once
#include "pw_core.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace sdft {
struct TransportedPlaneWaveSeeds {
  std::vector<std::complex<double>> coefficients;
  double minimum_retained_norm_fraction{};
  std::size_t common_indices{};
};

// Transfer periodic u_k coefficients by integer G, never by kinetic-energy
// order. This is an initial guess only. H, S, B*psi, occupations and D_ij
// MUST be recomputed in the destination geometry/k-point.
inline TransportedPlaneWaveSeeds transport_plane_wave_seeds(
    const std::vector<GVector>& old_g, const std::vector<GVector>& new_g,
    const std::vector<std::complex<double>>& coefficients, std::size_t columns) {
  using Key=std::tuple<int,int,int>;
  if(old_g.empty()||new_g.empty()||!columns||columns>coefficients.size()/old_g.size()||
      coefficients.size()!=old_g.size()*columns)
    throw std::invalid_argument("invalid plane-wave seed transport dimensions");
  std::map<Key,std::size_t> source;
  for(std::size_t i=0;i<old_g.size();++i)
    if(!source.emplace(Key{old_g[i].h,old_g[i].k,old_g[i].l},i).second)
      throw std::invalid_argument("duplicate source G index");
  std::map<Key,bool> destination;
  TransportedPlaneWaveSeeds out;
  out.coefficients.assign(new_g.size()*columns,{});
  for(std::size_t i=0;i<new_g.size();++i) {
    const Key key{new_g[i].h,new_g[i].k,new_g[i].l};
    if(!destination.emplace(key,true).second)throw std::invalid_argument("duplicate destination G index");
    const auto found=source.find(key);
    if(found==source.end())continue;
    ++out.common_indices;
    for(std::size_t j=0;j<columns;++j)
      out.coefficients[j*new_g.size()+i]=coefficients[j*old_g.size()+found->second];
  }
  out.minimum_retained_norm_fraction=1.;
  for(std::size_t j=0;j<columns;++j) {
    long double before=0,after=0;
    for(std::size_t i=0;i<old_g.size();++i) {
      const auto v=coefficients[j*old_g.size()+i];
      if(!std::isfinite(v.real())||!std::isfinite(v.imag()))
        throw std::invalid_argument("nonfinite source wavefunction coefficient");
      before+=static_cast<long double>(v.real())*v.real()+static_cast<long double>(v.imag())*v.imag();
    }
    for(std::size_t i=0;i<new_g.size();++i) {
      const auto v=out.coefficients[j*new_g.size()+i];
      after+=static_cast<long double>(v.real())*v.real()+static_cast<long double>(v.imag())*v.imag();
    }
    if(!(before>0))throw std::invalid_argument("zero source wavefunction seed");
    out.minimum_retained_norm_fraction=std::min(out.minimum_retained_norm_fraction,double(after/before));
  }
  return out;
}
} // namespace sdft
