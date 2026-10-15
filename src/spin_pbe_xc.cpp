#include "spin_pbe_xc.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sdft {
namespace {
constexpr double pi=3.1415926535897932384626433832795;
constexpr double gamma=.031090690869654895,beta=.06672455060314922;
constexpr double mu=.2195149727645171,kappa=.804;
// Independent five-variable forward derivatives: nu,nd,sigma_uu,sigma_ud,
// sigma_dd. The cross gradient therefore has a single derivative, not two
// separately counted variables. This is derived from the PBE energy formula.
struct D {
  double v{};std::array<double,5> g{};
  D()=default;D(double x):v(x){}
  static D variable(double x,int k){D d(x);d.g[k]=1;return d;}
};
D operator+(D a,D b){D r(a.v+b.v);for(int i=0;i<5;++i)r.g[i]=a.g[i]+b.g[i];return r;}
D operator-(D a,D b){D r(a.v-b.v);for(int i=0;i<5;++i)r.g[i]=a.g[i]-b.g[i];return r;}
D operator-(D a){return D(0)-a;}
D operator*(D a,D b){D r(a.v*b.v);for(int i=0;i<5;++i)r.g[i]=a.g[i]*b.v+a.v*b.g[i];return r;}
D operator/(D a,D b){D r(a.v/b.v);for(int i=0;i<5;++i)r.g[i]=(a.g[i]-r.v*b.g[i])/b.v;return r;}
D power(D a,double p){
  if(a.v<=0)throw std::invalid_argument("spin PBE positive power argument required");
  D r(std::pow(a.v,p));for(int i=0;i<5;++i)r.g[i]=p*r.v/a.v*a.g[i];return r;
}
D logone(D a){D r(std::log1p(a.v));for(int i=0;i<5;++i)r.g[i]=a.g[i]/(1+a.v);return r;}
D expminus(D a){D r(std::expm1(a.v));for(int i=0;i<5;++i)r.g[i]=(1+r.v)*a.g[i];return r;}
D pw(D r,double a,double alpha,double b1,double b2,double b3,double b4){
  auto root=power(r,.5);auto poly=b1*root+b2*r+b3*r*root+b4*r*r;
  return -2*a*(1+alpha*r)*logone(1/(2*a*poly));
}
D exchange(D n,D sigma,bool low){
  // Spin scaling: .5 E_x[2 n_s, 2 grad n_s].
  const auto doubled=2*n;auto root=power(doubled,1./3.);
  const auto kf=std::cbrt(3*pi*pi)*root;
  // Algebraically identical bounded enhancement. Avoid first creating
  // sigma/n^(8/3), whose derivative overflows in an almost empty channel
  // even though the final exchange response has a finite saturation limit.
  const auto denominator=4*kf*kf*doubled*doubled;
  const auto numerator=low?D(0):4*(mu/kappa)*sigma;
  const auto fx=1+kappa*numerator/(denominator+numerator);
  return -.375*std::cbrt(3/pi)*doubled*root*fx;
}
}

SpinPbePoint pbe_collinear(double up,double down,const std::array<double,3>& gu,
    const std::array<double,3>& gd,bool radial_low){
  if(!std::isfinite(up)||!std::isfinite(down)||up<0||down<0)
    throw std::invalid_argument("spin PBE densities must be finite and nonnegative");
  double uu=0,ud=0,dd=0;
  for(int a=0;a<3;++a){
    if(!std::isfinite(gu[a])||!std::isfinite(gd[a]))throw std::invalid_argument("nonfinite spin PBE gradient");
    uu+=gu[a]*gu[a];ud+=gu[a]*gd[a];dd+=gd[a]*gd[a];
  }
  if(up+down<1e-99)return {};
  // Exact empty channels have zero exchange. A positive density floor for
  // correlation avoids subtracting zeta from one at complete polarization.
  const auto nu=D::variable(std::max(up,1e-99),0),nd=D::variable(std::max(down,1e-99),1);
  const auto su=D::variable(uu,2),sud=D::variable(ud,3),sd=D::variable(dd,4);
  const auto n=nu+nd,root=power(n,1./3.),rs=std::cbrt(3/(4*pi))/root;
  auto z=(nu-nd)/n;
  // The PBE polarization endpoint has a singular phi derivative. Match
  // the documented reference regularization, retaining the response
  // derivatives at the bounded evaluation point rather than zeroing them.
  z.v=std::clamp(z.v,-.9999999999999,.9999999999999);
  const auto z2=z*z,z4=z2*z2;
  const auto pu=1+z,pd=1-z;
  constexpr double denominator=0.5198420997897463295;
  constexpr double fzz=8/(9*denominator);
  const auto f=(power(pu,4./3.)+power(pd,4./3.)-2)/denominator;
  const auto eu=pw(rs,.0310907,.21370,7.5957,3.5876,1.6382,.49294);
  const auto ep=pw(rs,.01554535,.20548,14.1189,6.1977,3.3662,.62517);
  const auto alpha=-pw(rs,.0168869,.11125,10.357,3.6231,.88026,.49671);
  const auto ec=eu+alpha*f/fzz*(1-z4)+(ep-eu)*f*z4;
  auto phi=.5*(power(pu,2./3.)+power(pd,2./3.));
  // Original PBE potential regularizes dphi/dzeta with eta=1e-12.
  // An unbounded AD derivative at a clipped empty channel produced a
  // finite but enormous PAW potential and destroyed Davidson's search.
  constexpr double eta=1e-12;
  const double gz=(std::pow(pu.v*pu.v+eta,-1./6.)-std::pow(pd.v*pd.v+eta,-1./6.))/3.;
  for(int i=0;i<5;++i)phi.g[i]=gz*z.g[i];
  const auto phi3=phi*phi*phi;
  const bool low=radial_low&&n.v<=1e-10;
  const auto sigma=low?D(0):su+2*sud+sd;
  const auto kf=std::cbrt(3*pi*pi)*root,ks2=4*kf/pi;
  const auto t2=sigma/(4*phi*phi*ks2*n*n);
  const auto a=(beta/gamma)/expminus(-ec/(gamma*phi3));
  const auto at=a*t2;
  const auto h=gamma*phi3*logone((beta/gamma)*t2*(1+at)/(1+at+at*at));
  auto ex=up==0?D(0):exchange(nu,su,low);
  if(down!=0)ex=ex+exchange(nd,sd,low);
  const auto co=n*(ec+h),total=ex+co;
  SpinPbePoint out;out.energy_density=total.v;out.exchange_energy_density=ex.v;
  out.correlation_energy_density=co.v;out.density_potential={total.g[0],total.g[1]};
  for(int axis=0;axis<3;++axis){
    out.gradient_response[0][axis]=2*total.g[2]*gu[axis]+total.g[3]*gd[axis];
    out.gradient_response[1][axis]=2*total.g[4]*gd[axis]+total.g[3]*gu[axis];
    if(!std::isfinite(out.gradient_response[0][axis])||!std::isfinite(out.gradient_response[1][axis]))
      throw std::runtime_error("nonfinite spin PBE gradient response");
  }
  if(!std::isfinite(out.energy_density)||!std::isfinite(out.density_potential[0])||
      !std::isfinite(out.density_potential[1]))throw std::runtime_error("nonfinite spin PBE response");
  return out;
}
}
