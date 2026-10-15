#pragma once
#include "density_functionals.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sdft {
struct AlignedFourierState {
  std::vector<std::tuple<int,int,int>> keys;
  std::vector<std::complex<double>> prior, output, residual;
  double maximum_residual{};
};
// Ordered merge of the actual maps, including newly introduced or absent
// modes. No fixed-support assumption, pointer cache, or discarded coefficient.
inline AlignedFourierState align_fourier_states(const FourierDensity& old_state,
                                               const FourierDensity& new_state) {
  AlignedFourierState result;
  const auto count=std::max(old_state.size(),new_state.size());
  result.keys.reserve(count);result.prior.reserve(count);
  result.output.reserve(count);result.residual.reserve(count);
  auto old=old_state.begin(),next=new_state.begin();
  while(old!=old_state.end()||next!=new_state.end()) {
    const auto key=next==new_state.end()||(old!=old_state.end()&&old->first<next->first)
        ?old->first:next->first;
    std::complex<double> prior{},output{};
    if(old!=old_state.end()&&old->first==key){prior=old->second;++old;}
    if(next!=new_state.end()&&next->first==key){output=next->second;++next;}
    if(!std::isfinite(prior.real())||!std::isfinite(prior.imag())||
       !std::isfinite(output.real())||!std::isfinite(output.imag()))
      throw std::invalid_argument("nonfinite Fourier mixer state");
    const auto error=output-prior;
    result.keys.push_back(key);result.prior.push_back(prior);
    result.output.push_back(output);result.residual.push_back(error);
    result.maximum_residual=std::max(result.maximum_residual,std::abs(error));
  }
  return result;
}
}
