#pragma once
#include "upf_paw_data.hpp"
#include "paw_hubbard.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sdft {
// Pseudo-wave Hubbard projector, matching QE's "pseudo" Hubbard option:
// select the last occupied PP_CHI of l, identify its bound beta by its
// ORIGINAL PS wave, and use Q P Q^H. This is not the POTCAR AE-overlap
// Hubbard definition.
inline std::vector<double> upf_pseudo_hubbard_overlap(const UpfPawData& data,int l) {
  if(l<0||l>3||data.atomic_l.empty()||
     data.atomic_l.size()!=data.atomic_occupations.size())
    throw std::invalid_argument("UPF pseudo Hubbard requires original typed PP_CHI metadata");
  const auto& rab=data.array("UPF/PP_MESH/PP_RAB");
  const auto& radii=data.array("UPF/PP_MESH/PP_R");
  // The overlap is taken on a truncated mesh, not the full raw UPF mesh:
  // up to the first point past 10 Bohr, rounded down to an odd count.
  // Scattering partial waves can grow in the raw asymptotic tail, so using
  // the entire mesh falsely rejects a genuine bound 3d CHI (e.g. Fe).
  std::size_t match_count=radii.size();
  for(std::size_t k=0;k<radii.size();++k)if(radii[k]>10.){match_count=k+1;break;}
  match_count=2*((match_count+1)/2)-1;
  const auto weights=upf_simpson_weights(rab,match_count);
  std::size_t selected=data.atomic_l.size();
  for(std::size_t i=0;i<data.atomic_l.size();++i)
    if(data.atomic_l[i]==l&&data.atomic_occupations[i]>=0.)selected=i;
  if(selected==data.atomic_l.size())throw std::invalid_argument("UPF has no requested bound Hubbard orbital");
  const auto& chi=data.array("UPF/PP_PSWFC/PP_CHI."+std::to_string(selected+1));
  if(chi.size()!=rab.size())throw std::invalid_argument("UPF Hubbard CHI/mesh mismatch");
  const auto nr=data.projector_count();std::size_t bound=nr;
  for(std::size_t i=0;i<nr;++i)if(data.projector_l[i]==l) {
    const auto& ps=data.array("UPF/PP_FULL_WFC/PP_PSWFC."+std::to_string(i+1));
    double projection=0.;for(std::size_t k=0;k<rab.size();++k)projection+=weights[k]*ps[k]*chi[k];
    const double sign=projection<0?-1.:1.;double difference=0.;
    for(std::size_t k=0;k<rab.size();++k)difference+=weights[k]*std::pow(ps[k]-sign*chi[k],2);
    if(std::abs(difference)<=1e-9)bound=i;
  }
  if(bound==nr)throw std::invalid_argument("UPF Hubbard bound beta does not match original CHI");
  const auto& ae_bound=data.array("UPF/PP_FULL_WFC/PP_AEWFC."+std::to_string(bound+1));
  std::vector<double> q(nr,0.);
  for(std::size_t i=0;i<nr;++i)if(data.projector_l[i]==l) {
    // QE fills the upper triangle nb<=mb using kbeta(nb), and mirrors it.
    // Preserve that discrete extent, rather than inventing a common RDEP.
    const auto cutoff=data.projector_cutoff[std::min(i,bound)];
    const auto w=upf_simpson_weights(rab,cutoff);
    const auto& ae=data.array("UPF/PP_FULL_WFC/PP_AEWFC."+std::to_string(i+1));
    for(std::size_t k=0;k<rab.size();++k)q[i]+=w[k]*ae[k]*ae_bound[k];
  }
  std::vector<double> result(nr*nr,0.);
  for(std::size_t i=0;i<nr;++i)for(std::size_t j=0;j<nr;++j)result[i*nr+j]=q[i]*q[j];
  return result;
}
} // namespace sdft
