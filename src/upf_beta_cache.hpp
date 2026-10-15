#pragma once
#include "upf_geometry_derivatives_v2.hpp"
#include <map>
#include <exception>
#include <mutex>
#include "compute_backend.hpp"

namespace sdft {
// Exact original quadrature, shared across m channels and identical |G+k|.
// The position-independent beta and its Cartesian derivative use the same
// radial values. Each atom adds only its actual Bloch position phase.
struct UpfBetaGeometry {
  std::size_t dimension{},npw{};
  std::vector<std::complex<double>> amplitudes;
  std::array<std::vector<std::complex<double>>,3> gradients;
};

inline UpfBetaGeometry prepare_upf_beta_geometry(const UpfPawData& data,
    const std::vector<Vec3>& vectors,double volume) {
  PerformanceTimer timer("upf_shared_beta_and_shape_setup");
  if(!std::isfinite(volume)||volume<=0)throw std::invalid_argument("invalid UPF beta volume");
  using C=std::complex<double>;
  constexpr double pi=3.1415926535897932384626433832795;
  UpfBetaGeometry result;result.npw=vectors.size();
  for(int l:data.projector_l)result.dimension+=2*l+1;
  result.amplitudes.resize(result.dimension*result.npw);
  for(auto& row:result.gradients)row.resize(result.dimension*result.npw);
  std::map<double,std::vector<std::size_t>> groups;
  for(std::size_t g=0;g<vectors.size();++g) {
    const double q=std::sqrt(dot(vectors[g],vectors[g]));
    if(!std::isfinite(q))throw std::invalid_argument("nonfinite UPF beta vector");
    groups[q].push_back(g);
  }
  std::vector<std::pair<double,std::vector<std::size_t>>> work;
  for(auto& group:groups)work.push_back(group);
  std::exception_ptr error;std::mutex error_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(std::ptrdiff_t group=0;group<static_cast<std::ptrdiff_t>(work.size());++group)try {
    const auto& entry=work[group];const double q=entry.first;
    std::size_t a=0;
    for(std::size_t radial=0;radial<data.projector_count();++radial) {
      const int l=data.projector_l[radial];
      const double f=upf_projector_form(data,radial,q),df=upf_projector_form(data,radial,q,true);
      const auto prefactor=4*pi/std::sqrt(volume)*std::pow(C(0,1),l);
      for(int m=-l;m<=l;++m,++a)for(const auto g:entry.second) {
        const auto v=vectors[g];const auto offset=a*result.npw+g;
        result.amplitudes[offset]=prefactor*upf_real_spherical_harmonic(l,m,v)*f;
        if(q==0) {
          if(l==1)for(int axis=0;axis<3;++axis) {
            const Vec3 direction=axis==0?Vec3{1,0,0}:axis==1?Vec3{0,1,0}:Vec3{0,0,1};
            result.gradients[axis][offset]=prefactor*df*upf_real_spherical_harmonic(l,m,direction);
          }
        } else {
          const auto dy=paw_real_spherical_harmonic_gradient(l,m,v);
          const double y=upf_real_spherical_harmonic(l,m,v);
          for(int axis=0;axis<3;++axis)result.gradients[axis][offset]=prefactor*
              (upf_component(dy,axis)*f+y*df*upf_component(v,axis)/q);
        }
      }
    }
  } catch(...) {std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=std::current_exception();}
  if(error)std::rethrow_exception(error);
  record_performance("upf_beta_unique_exact_radii",0.,work.size());
  return result;
}

inline std::vector<std::complex<double>> phased_upf_beta(const UpfBetaGeometry& geometry,
    const std::vector<Vec3>& vectors,Vec3 position) {
  if(vectors.size()!=geometry.npw)throw std::invalid_argument("UPF beta geometry dimensions changed");
  auto result=geometry.amplitudes;
  for(std::size_t g=0;g<geometry.npw;++g) {
    const auto phase=std::exp(std::complex<double>(0,dot(vectors[g],position)));
    for(std::size_t a=0;a<geometry.dimension;++a)result[a*geometry.npw+g]*=phase;
  }
  return result;
}
}
