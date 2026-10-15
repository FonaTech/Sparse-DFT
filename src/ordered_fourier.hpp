#pragma once
#include "density_functionals.hpp"
#include "thread_budget.hpp"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace sdft {
// Ordered merges give the same sums as tree inserts (bitwise on Si2, Si8,
// Ni at CPU8, 2026-10-15) with no per-key tree search. Default on;
// SDFT_ORDERED_FOURIER=0 or SDFT_DETERMINISTIC=1 restores tree accumulation.
inline bool ordered_fourier_enabled() {
  const char* value=std::getenv("SDFT_ORDERED_FOURIER");
  if(value&&std::string(value)!="0"&&std::string(value)!="1")
    throw std::invalid_argument("SDFT_ORDERED_FOURIER must be 0 or 1");
  if(!value)return !deterministic_mode();
  return std::string(value)=="1";
}

// Preserve source/k-point arithmetic order. Only tree searches/allocations
// change; no modes, imaginary components or small values are discarded.
inline void accumulate_ordered_fourier(FourierDensity& target,
    const FourierDensity& source,double weight=1.) {
  if(!std::isfinite(weight))throw std::invalid_argument("nonfinite Fourier accumulation weight");
  auto dst=target.begin();
  for(const auto& [key,value]:source) {
    while(dst!=target.end()&&dst->first<key)++dst;
    const auto scaled=weight*value;
    if(dst!=target.end()&&dst->first==key)dst->second+=scaled;
    else dst=target.emplace_hint(dst,key,std::complex<double>{}+scaled);
  }
}

inline void consume_ordered_fourier(FourierDensity& target,FourierDensity&& source,
    double weight=1.) {
  if(&target==&source)throw std::invalid_argument("aliased Fourier consume");
  if(!std::isfinite(weight))throw std::invalid_argument("nonfinite Fourier accumulation weight");
  if(target.empty()) {
    for(auto& [key,value]:source){(void)key;value=std::complex<double>{}+weight*value;}
    target=std::move(source);
  }else accumulate_ordered_fourier(target,source,weight);
}

inline double opposite_fourier_contract(const FourierDensity& left,
    const FourierDensity& right,double volume) {
  auto opposite=right.rbegin();double result=0;
  for(const auto& [key,rho]:left) {
    const auto [h,k,l]=key;const auto wanted=std::make_tuple(-h,-k,-l);
    while(opposite!=right.rend()&&opposite->first>wanted)++opposite;
    if(opposite!=right.rend()&&opposite->first==wanted)
      result+=volume*(rho*opposite->second).real();
  }
  return result;
}
}
