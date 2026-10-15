#include "paw_hubbard.hpp"
#include <cmath>
#include <stdexcept>

namespace sdft {
HubbardResult dudarev_paw(const std::vector<int>& channels,
    const std::vector<double>& overlap,HubbardSpec spec,
    const std::array<std::vector<std::complex<double>>,2>& density){
  using C=std::complex<double>;
  if(spec.angular_momentum<0||spec.angular_momentum>3||!std::isfinite(spec.ueff_ev)||spec.ueff_ev<0)
    throw std::invalid_argument("Dudarev requires l in [0,3] and nonnegative finite Ueff");
  const auto nr=channels.size();std::vector<std::size_t> offset(nr);std::size_t dim=0;
  bool selected=false;
  for(std::size_t i=0;i<nr;++i){
    if(channels[i]<0||channels[i]>3)throw std::invalid_argument("invalid PAW Hubbard channel");
    offset[i]=dim;dim+=2*channels[i]+1;selected|=channels[i]==spec.angular_momentum;
  }
  if(!selected||overlap.size()!=nr*nr)throw std::invalid_argument("correlated PAW channel or radial overlap missing");
  for(std::size_t i=0;i<nr;++i)for(std::size_t j=0;j<nr;++j)
    if(!std::isfinite(overlap[i*nr+j])||std::abs(overlap[i*nr+j]-overlap[j*nr+i])>1e-12)
      throw std::invalid_argument("AE Hubbard overlap is not finite symmetric");
  HubbardResult out;const auto nm=std::size_t(2*spec.angular_momentum+1);out.correlated_dimension=nm;
  for(int spin=0;spin<2;++spin){
    const auto& rho=density[spin];
    if(rho.size()!=dim*dim)throw std::invalid_argument("Hubbard onsite dimension mismatch");
    for(std::size_t i=0;i<dim;++i)for(std::size_t j=0;j<dim;++j)
      if(!std::isfinite(rho[i*dim+j].real())||!std::isfinite(rho[i*dim+j].imag())||
         std::abs(rho[i*dim+j]-std::conj(rho[j*dim+i]))>1e-10)
        throw std::invalid_argument("Hubbard onsite must be finite Hermitian");
    auto& n=out.occupations[spin];n.assign(nm*nm,C{});
    for(std::size_t i=0;i<nr;++i)if(channels[i]==spec.angular_momentum)
      for(std::size_t j=0;j<nr;++j)if(channels[j]==spec.angular_momentum)
        for(std::size_t m=0;m<nm;++m)for(std::size_t p=0;p<nm;++p)
          n[m*nm+p]+=overlap[i*nr+j]*rho[(offset[i]+m)*dim+offset[j]+p];
    double trace=0,square=0;
    for(std::size_t m=0;m<nm;++m){trace+=n[m*nm+m].real();
      for(std::size_t p=0;p<nm;++p)square+=std::norm(n[m*nm+p]);}
    out.energy_ev+=.5*spec.ueff_ev*(trace-square);
    auto& potential=out.projector_potential_ev[spin];potential.assign(dim*dim,C{});
    for(std::size_t i=0;i<nr;++i)if(channels[i]==spec.angular_momentum)
      for(std::size_t j=0;j<nr;++j)if(channels[j]==spec.angular_momentum)
        for(std::size_t m=0;m<nm;++m)for(std::size_t p=0;p<nm;++p)
          potential[(offset[i]+m)*dim+offset[j]+p]=spec.ueff_ev*overlap[i*nr+j]*
              ((m==p?C(.5):C{})-n[m*nm+p]);
    for(std::size_t i=0;i<rho.size();++i)
      out.potential_expectation_ev+=(std::conj(rho[i])*potential[i]).real();
  }
  out.band_double_counting_ev=out.energy_ev-out.potential_expectation_ev;
  return out;
}
}
