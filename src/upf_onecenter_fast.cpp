#include "upf_onecenter_fast.hpp"
#include <map>
#include <memory>
#include <mutex>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
#include "paw_compensation.hpp"
#include "upf_radial_poisson.hpp"
#include "pbe_xc.hpp"
#include "spin_pbe_xc.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sdft {
namespace {
constexpr double pi=3.1415926535897932384626433832795;
using Radial=std::vector<std::vector<double>>;
struct Functional {double energy{};Radial derivative;std::size_t negative{};};
// Use the high-order radial energy and its exact discrete adjoint together.
Functional hartree(const std::vector<double>& r,const std::vector<double>& w,const Radial& rho) {
  const auto value=upf_log_radial_hartree(r,w,rho);
  return {value.energy,value.derivative,0};
}
struct GradientRow {std::array<std::size_t,5> index;std::array<double,5> weight;};
// QE style=0 derivatives on the actual mesh. The
// transpose below differentiates the same PBE discrete energy exactly.
std::vector<GradientRow> gradients(const std::vector<double>& r) {
  if(r.size()<5)throw std::invalid_argument("UPF PBE radial mesh too short");
  std::vector<GradientRow> rows(r.size());
  // Three-point nonuniform-mesh first derivative on the original UPF mesh,
  // the convention QE uses for its PAW one-centre GGA.
  // The energy derivative below uses its exact transpose, so the change
  // preserves the discrete variational identity instead of copying a
  // separately discretized continuum divergence.
  for(std::size_t k=1;k+1<r.size();++k){
    for(int j=0;j<3;++j){const auto a=k-1+j;rows[k].index[j]=a;double d=0.;
      for(int m=0;m<3;++m)if(m!=j){const auto b=k-1+m;double term=1./(r[a]-r[b]);
        for(int t=0;t<3;++t)if(t!=j&&t!=m){const auto c=k-1+t;term*=(r[k]-r[c])/(r[a]-r[c]);}d+=term;}
      rows[k].weight[j]=d;
    }
  }
  const double extrapolate=(r[0]-r[1])/(r[2]-r[1]);
  for(int j=0;j<4;++j){rows[0].index[j]=j;double w=0.;
    for(int t=0;t<3;++t){if(rows[1].index[t]==std::size_t(j))w+=(1-extrapolate)*rows[1].weight[t];
      if(rows[2].index[t]==std::size_t(j))w+=extrapolate*rows[2].weight[t];}
    rows[0].weight[j]=w;
  }
  // QE assumes a vanishing asymptotic density and sets the last gradient 0.

  return rows;
}
struct Direction {double weight;Vec3 direction;std::vector<double> y;std::vector<Vec3> gradient;};
std::vector<Direction> angular(int lmax,int order) {
  if(order<8||order>64)throw std::invalid_argument("invalid UPF angular quadrature");
  std::vector<Direction> points;
  for(int i=0;i<order;++i){double z=std::cos(pi*(i+.75)/(order+.5)),derivative=0;
    for(int it=0;it<40;++it){double prev=1,current=z;for(int n=2;n<=order;++n){double next=((2*n-1)*z*current-(n-1)*prev)/n;prev=current;current=next;}
      derivative=order*(z*current-prev)/(z*z-1);const auto step=current/derivative;z-=step;if(std::abs(step)<3e-16)break;}
    const double weight=2/((1-z*z)*derivative*derivative),s=std::sqrt(1-z*z);
    for(int j=0;j<2*order;++j){double phi=2*pi*j/(2*order);Direction p{weight*2*pi/(2*order),{s*std::cos(phi),s*std::sin(phi),z},{},{}};
      for(int l=0;l<=lmax;++l)for(int m=-l;m<=l;++m){p.y.push_back(paw_real_spherical_harmonic(l,m,p.direction));p.gradient.push_back(paw_real_spherical_harmonic_gradient(l,m,p.direction));}points.push_back(std::move(p));}
  }return points;
}
// The quadrature is immutable and keyed by its actual L/order. It is never
// truncated or replaced with a spherical approximation for performance.
std::shared_ptr<const std::vector<Direction>> cached_angular(int lmax,int order) {
  static std::mutex mutex;
  static std::map<std::pair<int,int>,std::shared_ptr<const std::vector<Direction>>> values;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key=std::make_pair(lmax,order);
  auto it=values.find(key);
  if(it==values.end())it=values.emplace(key,std::make_shared<const std::vector<Direction>>(angular(lmax,order))).first;
  return it->second;
}
Functional xc(const std::vector<double>& r,const std::vector<double>& w,const Radial& rho,int lmax,int order) {
  const auto rows=gradients(r);const auto cached=cached_angular(lmax,order);
  const auto& directions=*cached;const auto mesh=r.size(),nm=rho.size();
  for(const auto& row:rho)for(double x:row)
    if(!std::isfinite(x))throw std::invalid_argument("nonfinite UPF radial density");
  std::vector<double> inv_r(mesh),inv_r2(mesh),inv_r3(mesh);
  for(std::size_t k=0;k<mesh;++k){inv_r[k]=1/r[k];inv_r2[k]=inv_r[k]*inv_r[k];inv_r3[k]=inv_r2[k]*inv_r[k];}
  // Eight fixed angular partitions and a fixed reduction order produce the
  // same result at OMP_NUM_THREADS=1 and 8. No atomic floating-point adds.
  constexpr int partitions=8;
  std::array<Functional,partitions> parts;
  for(auto& f:parts)f.derivative=Radial(nm,std::vector<double>(mesh));
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
  for(int part=0;part<partitions;++part){auto& f=parts[part];
    std::vector<double> density(mesh),density_response(mesh);
    std::vector<Vec3> tangent(mesh),response(mesh);
    const auto begin=directions.size()*part/partitions,end=directions.size()*(part+1)/partitions;
    for(std::size_t q=begin;q<end;++q){const auto& p=directions[q];
      std::fill(density.begin(),density.end(),0.);
      std::fill(tangent.begin(),tangent.end(),Vec3{});
      std::fill(density_response.begin(),density_response.end(),0.);
      for(std::size_t k=0;k<mesh;++k)for(std::size_t lm=0;lm<nm;++lm){const double n=rho[lm][k]*inv_r2[k];
        density[k]+=p.y[lm]*n;const double t=n*inv_r[k];
        tangent[k].x+=p.gradient[lm].x*t;tangent[k].y+=p.gradient[lm].y*t;tangent[k].z+=p.gradient[lm].z*t;
      }
      for(std::size_t k=0;k<mesh;++k){double radial=0.;
        for(int j=0;j<5;++j)radial+=rows[k].weight[j]*density[rows[k].index[j]];
        const Vec3 g{radial*p.direction.x+tangent[k].x,radial*p.direction.y+tangent[k].y,radial*p.direction.z+tangent[k].z};
        if(density[k]<-1e-10)++f.negative;
        const auto value=pbe_unpolarized(std::max(density[k],1e-18),std::sqrt(dot(g,g)));
        const double volume=w[k]*r[k]*r[k]*p.weight;
        f.energy+=volume*value.energy_density;
        const double v=density[k]>1e-18?value.exchange_potential+value.correlation_potential:0.;
        density_response[k]+=volume*v;
        response[k]={volume*value.gradient_squared_coefficient*g.x,
                     volume*value.gradient_squared_coefficient*g.y,
                     volume*value.gradient_squared_coefficient*g.z};
        const double radial_response=dot(response[k],p.direction);
        // Apply the radial-gradient transpose once per direction, then
        // transform back to Y_lm. This is the exact original discrete adjoint.
        for(int j=0;j<5;++j)density_response[rows[k].index[j]]+=radial_response*rows[k].weight[j];
      }
      for(std::size_t lm=0;lm<nm;++lm)for(std::size_t k=0;k<mesh;++k)
        f.derivative[lm][k]+=p.y[lm]*density_response[k]*inv_r2[k]
            +dot(response[k],p.gradient[lm])*inv_r3[k];
    }
  }
  Functional result;result.derivative=Radial(nm,std::vector<double>(mesh));
  for(const auto& part:parts){result.energy+=part.energy;result.negative+=part.negative;
    for(std::size_t lm=0;lm<nm;++lm)for(std::size_t k=0;k<mesh;++k)result.derivative[lm][k]+=part.derivative[lm][k];}
  return result;
}
struct SpinFunctional {
  double energy{};
  std::array<Radial,2> derivative;
  std::size_t negative{};
};
SpinFunctional spin_xc(const std::vector<double>& r,const std::vector<double>& w,
    const std::array<Radial,2>& rho,int lmax,int order) {
  const auto rows=gradients(r);const auto directions=cached_angular(lmax,order);
  const auto mesh=r.size(),nm=rho[0].size();
  for(const auto& channel:rho) {
    if(channel.size()!=nm)throw std::invalid_argument("UPF spin angular density dimensions differ");
    for(const auto& row:channel) {
      if(row.size()!=mesh)throw std::invalid_argument("UPF spin radial density dimensions differ");
      for(double x:row)if(!std::isfinite(x))throw std::invalid_argument("nonfinite UPF spin density");
    }
  }
  constexpr int partitions=8;
  std::array<SpinFunctional,partitions> parts;
  for(auto& part:parts)for(auto& d:part.derivative)d=Radial(nm,std::vector<double>(mesh));
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
  for(int part=0;part<partitions;++part) {
    auto& f=parts[part];std::array<std::vector<double>,2> density,response;
    std::array<std::vector<Vec3>,2> tangent,gradient_response;
    for(int s=0;s<2;++s) {density[s].resize(mesh);response[s].resize(mesh);
      tangent[s].resize(mesh);gradient_response[s].resize(mesh);}
    for(std::size_t q=directions->size()*part/partitions;q<directions->size()*(part+1)/partitions;++q) {
      const auto& p=directions->at(q);
      for(int s=0;s<2;++s) {
        std::fill(density[s].begin(),density[s].end(),0.);
        std::fill(tangent[s].begin(),tangent[s].end(),Vec3{});
        std::fill(response[s].begin(),response[s].end(),0.);
        for(std::size_t k=0;k<mesh;++k)for(std::size_t lm=0;lm<nm;++lm) {
          const double n=rho[s][lm][k]/(r[k]*r[k]);density[s][k]+=p.y[lm]*n;
          const double t=n/r[k];tangent[s][k].x+=p.gradient[lm].x*t;
          tangent[s][k].y+=p.gradient[lm].y*t;tangent[s][k].z+=p.gradient[lm].z*t;
        }
      }
      for(std::size_t k=0;k<mesh;++k) {
        std::array<std::array<double,3>,2> g{};
        for(int s=0;s<2;++s) {
          double radial=0.;for(int j=0;j<5;++j)radial+=rows[k].weight[j]*density[s][rows[k].index[j]];
          g[s]={radial*p.direction.x+tangent[s][k].x,radial*p.direction.y+tangent[s][k].y,
                radial*p.direction.z+tangent[s][k].z};
          if(density[s][k]<-1e-10)++f.negative;
        }
        const auto value=pbe_collinear(std::max(density[0][k],1e-18),std::max(density[1][k],1e-18),g[0],g[1]);
        const double volume=w[k]*r[k]*r[k]*p.weight;f.energy+=volume*value.energy_density;
        for(int s=0;s<2;++s) {
          if(density[s][k]>1e-18)response[s][k]+=volume*value.density_potential[s];
          const auto& v=value.gradient_response[s];gradient_response[s][k]={volume*v[0],volume*v[1],volume*v[2]};
          const double radial=dot(gradient_response[s][k],p.direction);
          for(int j=0;j<5;++j)response[s][rows[k].index[j]]+=radial*rows[k].weight[j];
        }
      }
      for(int s=0;s<2;++s)for(std::size_t lm=0;lm<nm;++lm)for(std::size_t k=0;k<mesh;++k)
        f.derivative[s][lm][k]+=p.y[lm]*response[s][k]/(r[k]*r[k])+
            dot(gradient_response[s][k],p.gradient[lm])/(r[k]*r[k]*r[k]);
    }
  }
  SpinFunctional result;for(auto& d:result.derivative)d=Radial(nm,std::vector<double>(mesh));
  for(const auto& part:parts) {result.energy+=part.energy;result.negative+=part.negative;
    for(int s=0;s<2;++s)for(std::size_t lm=0;lm<nm;++lm)for(std::size_t k=0;k<mesh;++k)
      result.derivative[s][lm][k]+=part.derivative[s][lm][k];}
  return result;
}
}
UpfOneCenterResult upf_onecenter_fast(const UpfPawData& data,const UpfPawOperator& op,const std::vector<std::complex<double>>& density,int order) {
  const auto& r=data.array("UPF/PP_MESH/PP_R");
  if(r.front()<=0)throw std::invalid_argument("UPF onecenter requires strictly positive radial mesh");
  const auto w=upf_simpson_weights(data.array("UPF/PP_MESH/PP_RAB"),r.size());
  auto ae=op.radial_density(density,true),ps=op.radial_density(density,false);
  const auto ha=hartree(r,w,ae),hp=hartree(r,w,ps);
  const auto& ca=data.array("UPF/PP_PAW/PP_AE_NLCC");
  const auto found=data.arrays.find("UPF/PP_NLCC");
  // PP_NLCC and PP_AE_NLCC store physical spherical density (not r^2 rho).
  // Frozen cores participate in XC only; PAW Hartree uses valence density.
  for(std::size_t k=0;k<r.size();++k){ae[0][k]+=std::sqrt(4*pi)*r[k]*r[k]*ca[k];
    if(found!=data.arrays.end())ps[0][k]+=std::sqrt(4*pi)*r[k]*r[k]*found->second[k];}
  const auto xa=xc(r,w,ae,op.density_lmax(),order),xp=xc(r,w,ps,op.density_lmax(),order);
  auto va=ha.derivative,vp=hp.derivative;
  for(std::size_t lm=0;lm<va.size();++lm)for(std::size_t k=0;k<r.size();++k){va[lm][k]+=xa.derivative[lm][k];vp[lm][k]+=xp.derivative[lm][k];}
  auto da=op.project_radial_potential(va,true),dp=op.project_radial_potential(vp,false);
  UpfOneCenterResult result;result.hartree_ae=ha.energy;result.hartree_ps=hp.energy;result.xc_ae=xa.energy;result.xc_ps=xp.energy;result.energy=ha.energy-hp.energy+xa.energy-xp.energy;
  result.derivative.resize(da.size());for(std::size_t k=0;k<da.size();++k)result.derivative[k]=da[k]-dp[k];
  result.negative_density_samples=xa.negative+xp.negative;return result;
}
UpfSpinOneCenterResult upf_onecenter_collinear(const UpfPawData& data,const UpfPawOperator& op,
    const std::array<std::vector<std::complex<double>>,2>& density,int order) {
  const auto& r=data.array("UPF/PP_MESH/PP_R");
  if(r.front()<=0)throw std::invalid_argument("UPF spin onecenter requires positive radial mesh");
  const auto w=upf_simpson_weights(data.array("UPF/PP_MESH/PP_RAB"),r.size());
  std::array<Radial,2> ae,ps;
  for(int s=0;s<2;++s){ae[s]=op.radial_density(density[s],true);ps[s]=op.radial_density(density[s],false);}
  auto ae_total=ae[0],ps_total=ps[0];
  for(std::size_t lm=0;lm<ae_total.size();++lm)for(std::size_t k=0;k<r.size();++k) {
    ae_total[lm][k]+=ae[1][lm][k];ps_total[lm][k]+=ps[1][lm][k];
  }
  const auto ha=hartree(r,w,ae_total),hp=hartree(r,w,ps_total);
  const auto& ca=data.array("UPF/PP_PAW/PP_AE_NLCC");const auto core=data.arrays.find("UPF/PP_NLCC");
  // Frozen core is unpolarized: exactly half in each XC spin channel.
  // Hartree contains the sum of valence channels once and excludes core.
  for(int s=0;s<2;++s)for(std::size_t k=0;k<r.size();++k) {
    ae[s][0][k]+=.5*std::sqrt(4*pi)*r[k]*r[k]*ca[k];
    if(core!=data.arrays.end())ps[s][0][k]+=.5*std::sqrt(4*pi)*r[k]*r[k]*core->second[k];
  }
  const auto xa=spin_xc(r,w,ae,op.density_lmax(),order),xp=spin_xc(r,w,ps,op.density_lmax(),order);
  UpfSpinOneCenterResult result;result.energy=ha.energy-hp.energy+xa.energy-xp.energy;
  result.hartree_ae=ha.energy;result.hartree_ps=hp.energy;result.xc_ae=xa.energy;result.xc_ps=xp.energy;
  for(int s=0;s<2;++s) {
    auto va=ha.derivative,vp=hp.derivative;
    for(std::size_t lm=0;lm<va.size();++lm)for(std::size_t k=0;k<r.size();++k) {
      va[lm][k]+=xa.derivative[s][lm][k];vp[lm][k]+=xp.derivative[s][lm][k];
    }
    const auto da=op.project_radial_potential(va,true),dp=op.project_radial_potential(vp,false);
    result.derivative[s].resize(da.size());
    for(std::size_t k=0;k<da.size();++k)result.derivative[s][k]=da[k]-dp[k];
  }
  result.negative_density_samples=xa.negative+xp.negative;
  return result;
}
} // namespace sdft
