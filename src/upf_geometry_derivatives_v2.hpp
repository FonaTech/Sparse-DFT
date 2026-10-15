#pragma once
#include "upf_paw_operator.hpp"
#include "paw_compensation.hpp"
#include <cmath>

namespace sdft {
inline double upf_component(Vec3 v,int axis){return axis==0?v.x:axis==1?v.y:v.z;}
// Cartesian solid-harmonic recurrence avoids subtracting 1-z^2/r^2 near
// the polar axis. It has the same real-harmonic normalization and phases
// as the established derivative recurrence, including non-CS convention.
inline double upf_real_spherical_harmonic(int l,int m,Vec3 g){
  constexpr double pi=3.1415926535897932384626433832795;
  using C=std::complex<double>;
  const double r2=dot(g,g),r=std::sqrt(r2);const int order=std::abs(m);
  if(l<0||l>6||order>l)throw std::invalid_argument("unsupported UPF harmonic");
  if(r==0)return l==0?1/std::sqrt(4*pi):0.;
  C value=1,previous=0;const C xy(g.x,g.y);
  for(int k=1;k<=order;++k)value*=(2*k-1.)*xy;
  if(l>order){previous=value;value*=(2*order+1.)*g.z;
    for(int k=order+2;k<=l;++k){const auto next=((2*k-1.)*g.z*value-(k+order-1.)*r2*previous)/double(k-order);previous=value;value=next;}}
  const double pref=std::sqrt((2*l+1.)/(4*pi)*std::exp(std::lgamma(l-order+1.)-std::lgamma(l+order+1.)))*(order?std::sqrt(2.):1.)/std::pow(r,l);
  return pref*(m<0?value.imag():value.real());
}

// Derivative of the original beta bra with respect to Cartesian G+k,
// at fixed position phase. Position and strain phase responses are handled
// separately; fractional phase is invariant under a homogeneous strain.
inline std::array<std::vector<std::complex<double>>,3> upf_beta_shape_gradient(
    const UpfPawData& data,const std::vector<Vec3>& vectors,double volume,Vec3 position) {
  constexpr double pi=3.1415926535897932384626433832795;
  using C=std::complex<double>;
  std::size_t dimension=0;for(int l:data.projector_l)dimension+=2*l+1;
  std::array<std::vector<C>,3> result;
  for(auto& row:result)row.resize(dimension*vectors.size());
  std::size_t a=0;
  for(std::size_t radial=0;radial<data.projector_count();++radial){const int l=data.projector_l[radial];
    for(int m=-l;m<=l;++m,++a)for(std::size_t k=0;k<vectors.size();++k){
      const auto g=vectors[k];const double q=std::sqrt(dot(g,g));
      const auto prefactor=4*pi/std::sqrt(volume)*std::pow(C(0,1),l)*std::exp(C(0,dot(g,position)));
      const double f=upf_projector_form(data,radial,q),df=upf_projector_form(data,radial,q,true);
      if(q==0){
        // l=1 beta form is linear at the origin; evaluate the product's
        // analytic radial limit along each axis, avoiding singular dY/dG.
        if(l==1)for(int axis=0;axis<3;++axis){Vec3 direction{};
          direction=axis==0?Vec3{1,0,0}:axis==1?Vec3{0,1,0}:Vec3{0,0,1};
          result[axis][a*vectors.size()+k]=prefactor*df*upf_real_spherical_harmonic(l,m,direction);}
        continue;
      }
      const double y=upf_real_spherical_harmonic(l,m,g);const auto dy=paw_real_spherical_harmonic_gradient(l,m,g);
      for(int axis=0;axis<3;++axis)result[axis][a*vectors.size()+k]=prefactor*(upf_component(dy,axis)*f+y*df*upf_component(g,axis)/q);
    }
  }
  return result;
}
}
