#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sdft {

// Numerov discretization of each logarithmic-mesh radial Poisson channel.
// The potential map and its exact transpose define one quadratic energy.
// Keeping both is essential: a separately discretized potential would break
// the energy/force identity when the PAW occupation changes with geometry.
class UpfRadialPoisson {
  std::vector<double> r_, sqrt_r_, diagonal_, edge_;
  std::array<double,4> c2_{},c3_{};
  double ch_{},ei_{},first_{},last_{},origin_{},delta2_{},delta3_{};

  static std::array<double,4> cubic(const std::array<double,4>& f,
                                  const std::vector<double>& r) {
    const double a=r[0],b=r[1],c=r[2],d=r[3];
    const double d21=(f[1]-f[0])/(b-a),d32=(f[2]-f[1])/(c-b),d43=(f[3]-f[2])/(d-c);
    const double dd42=(d43-d32)/(d-b),dd31=(d32-d21)/(c-a);
    const double b3=(dd42-dd31)/(d-a),b2=dd31-b3*(a+b+c);
    const double b1=d21-b2*(a+b)-b3*(a*a+b*b+a*b);
    return {f[0]-a*(b1+a*(b2+a*b3)),b1,b2,b3};
  }

  std::vector<double> solve(std::vector<double> value) const {
    if(value.size()!=diagonal_.size())throw std::invalid_argument("radial Poisson RHS size");
    for(std::size_t i=1;i<value.size();++i)value[i]-=edge_[i-1]/diagonal_[i-1]*value[i-1];
    value.back()/=diagonal_.back();
    for(std::size_t i=value.size()-1;i-->0;)value[i]=(value[i]-edge_[i]*value[i+1])/diagonal_[i];
    return value;
  }

public:
  UpfRadialPoisson(const std::vector<double>& r,int l):r_(r) {
    constexpr double pi=3.1415926535897932384626433832795;
    if(r.size()<5||r.front()<=0||l<0||l>16)throw std::invalid_argument("UPF radial Poisson grid/angular channel");
    const double dx=std::log(r[1]/r[0]);
    if(!std::isfinite(dx)||dx<=0)throw std::invalid_argument("UPF radial Poisson logarithmic step");
    for(std::size_t i=1;i<r.size();++i)
      if(!std::isfinite(r[i])||std::abs(std::log(r[i]/r[i-1])-dx)>1e-9)
        throw std::invalid_argument("UPF high-order Poisson requires original uniform logarithmic mesh");
    for(double x:r)sqrt_r_.push_back(std::sqrt(x));
    ch_=dx*dx/12;const double x=ch_*(l+.5)*(l+.5);ei_=1-x;
    const int k=2*l+1,nst=2*l+2,nk=nst-l-1;
    first_=std::pow(sqrt_r_[0]/sqrt_r_[1],k);
    last_=std::pow(sqrt_r_[r.size()-2]/sqrt_r_.back(),k);
    origin_=std::pow(sqrt_r_[0],k);
    delta2_=r[0]*r[0]-r[1]*r[1];delta3_=r[0]*r[0]*r[0]-r[1]*r[1]*r[1];
    if(nk<3)for(int j=0;j<4;++j) {
      std::array<double,4> f{};f[j]=-4*pi/std::pow(r[j],nst);
      const auto b=cubic(f,r);
      if(nk==1)c2_[j]=b[0]/(4*l+6);
      c3_[j]=b[2-nk]/(6*l+12);
    }
    diagonal_.assign(r.size()-2,2+10*x);edge_.assign(r.size()-3,-ei_);
    diagonal_.front()-=ei_*first_;diagonal_.back()-=ei_*last_;
    for(std::size_t i=1;i<diagonal_.size();++i) {
      if(!(diagonal_[i-1]>0))throw std::runtime_error("radial Poisson factorization not positive");
      diagonal_[i]-=edge_[i-1]*edge_[i-1]/diagonal_[i-1];
    }
    if(!(diagonal_.back()>0))throw std::runtime_error("radial Poisson final pivot not positive");
  }

  std::vector<double> potential(const std::vector<double>& rho) const {
    constexpr double pi=3.1415926535897932384626433832795;
    const auto n=r_.size();if(rho.size()!=n)throw std::invalid_argument("radial Poisson density size");
    double c2=0,c3=0;for(int j=0;j<4;++j){c2+=c2_[j]*rho[j];c3+=c3_[j]*rho[j];}
    std::vector<double> aux(n),rhs(n-2),v(n);
    for(std::size_t i=0;i<n;++i)aux[i]=4*pi*ch_*sqrt_r_[i]*rho[i];
    for(std::size_t i=1;i+1<n;++i)rhs[i-1]=aux[i-1]+10*aux[i]+aux[i+1];
    rhs[0]+=ei_*origin_*(c2*delta2_+c3*delta3_);
    const auto u=solve(std::move(rhs));
    for(std::size_t i=1;i+1<n;++i)v[i]=u[i-1];
    v[0]=first_*v[1]+origin_*(c2*delta2_+c3*delta3_);
    v[n-1]=last_*v[n-2];
    for(std::size_t i=0;i<n;++i)v[i]/=sqrt_r_[i];
    return v;
  }

  std::vector<double> transpose(const std::vector<double>& response) const {
    constexpr double pi=3.1415926535897932384626433832795;
    const auto n=r_.size();if(response.size()!=n)throw std::invalid_argument("radial Poisson adjoint size");
    std::vector<double> bar(n),rhs(n-2),gradient(n);
    for(std::size_t i=0;i<n;++i)bar[i]=response[i]/sqrt_r_[i];
    double c2=bar[0]*origin_*delta2_,c3=bar[0]*origin_*delta3_;
    for(std::size_t i=1;i+1<n;++i)rhs[i-1]=bar[i];
    rhs[0]+=first_*bar[0];rhs.back()+=last_*bar.back();
    const auto adj=solve(std::move(rhs));
    c2+=adj[0]*ei_*origin_*delta2_;c3+=adj[0]*ei_*origin_*delta3_;
    for(std::size_t i=1;i+1<n;++i){gradient[i-1]+=adj[i-1];gradient[i]+=10*adj[i-1];gradient[i+1]+=adj[i-1];}
    for(std::size_t i=0;i<n;++i)gradient[i]*=4*pi*ch_*sqrt_r_[i];
    for(int j=0;j<4;++j)gradient[j]+=c2*c2_[j]+c3*c3_[j];
    return gradient;
  }
};

struct UpfRadialHartree {
  double energy{};
  std::vector<std::vector<double>> derivative;
};

inline UpfRadialHartree upf_log_radial_hartree(const std::vector<double>& r,
    const std::vector<double>& w,const std::vector<std::vector<double>>& rho) {
  if(r.size()!=w.size())throw std::invalid_argument("UPF Hartree integration dimensions");
  UpfRadialHartree result;result.derivative.resize(rho.size());
  for(std::size_t lm=0;lm<rho.size();++lm) {
    const UpfRadialPoisson poisson(r,int(std::sqrt(double(lm))));
    const auto v=poisson.potential(rho[lm]);std::vector<double> weighted(r.size());
    for(std::size_t i=0;i<r.size();++i)weighted[i]=w[i]*rho[lm][i];
    const auto adj=poisson.transpose(weighted);result.derivative[lm].resize(r.size());
    for(std::size_t i=0;i<r.size();++i){
      result.energy+=.5*weighted[i]*v[i];
      result.derivative[lm][i]=.5*(w[i]*v[i]+adj[i]);
    }
  }
  return result;
}
} // namespace sdft
