#include "spin_pbe_xc.hpp"
#include "pbe_xc.hpp"
#include "paw_hubbard.hpp"
#include "joint_multisecant.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void close(double a,double b,double tol,const char* name){
  if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tol*std::max(1.,std::abs(b)))
    throw std::runtime_error(std::string(name)+" error="+std::to_string(std::abs(a-b)));
}
}
int main(){
  using namespace sdft;
  for(double n:{1e-6,.001,.05,1.,100.})for(double g:{0.,1e-6,.001,.1}){
    const auto p=pbe_collinear(n/2,n/2,{g/2,0,0},{g/2,0,0});
    const auto q=pbe_unpolarized(n,g);
    close(p.energy_density,q.energy_density,3e-13,"unpolarized energy");
    close(p.density_potential[0],q.exchange_potential+q.correlation_potential,3e-12,"unpolarized potential");
    close(p.gradient_response[0][0],q.gradient_squared_coefficient*g,3e-11,"unpolarized gradient");
  }
  // PAW mixing can expose a clipped empty minority channel while the
  // other channel and its gradients remain finite. This used to create
  // inf/inf in AD before any final physical response was evaluated.
  for(double minor:{1e-99,1e-70,1e-30,1e-12})for(double major:{1e-8,.01,1.}) {
    const auto x=pbe_collinear(major,minor,{.001,.003,0},{.002,-.001,0},true);
    const auto y=pbe_collinear(minor,major,{.002,-.001,0},{.001,.003,0},true);
    close(x.energy_density,y.energy_density,2e-14,"empty minority spin exchange symmetry");
    close(x.density_potential[0],y.density_potential[1],2e-12,"empty minority potential symmetry");
    for(const auto& g:x.gradient_response)for(double v:g)if(!std::isfinite(v))throw std::runtime_error("nonfinite minority gradient");
  }
  for(double n:{1e-99,1e-70,1e-30})pbe_collinear(n,n,{.001,0,0},{-.001,0,0},true);
  double maximum=0;
  for(double n:{.003,.1,4.})for(double polarization:{-.8,-.2,0.,.3,.9}){
    double nu=n*(1+polarization)/2,nd=n*(1-polarization)/2;
    std::array<double,3> gu{.007,.003,-.002},gd{-.001,.004,.003};
    const auto p=pbe_collinear(nu,nd,gu,gd),swap=pbe_collinear(nd,nu,gd,gu);
    close(p.energy_density,swap.energy_density,2e-14,"spin swap energy");
    close(p.density_potential[0],swap.density_potential[1],2e-12,"spin swap potential");
    for(int s=0;s<2;++s){
      const double h=(s?nd:nu)*1e-4;
      const auto a=pbe_collinear(nu+(s?0:h),nd+(s?h:0),gu,gd);
      const auto b=pbe_collinear(nu-(s?0:h),nd-(s?h:0),gu,gd);
      const auto fd=(a.energy_density-b.energy_density)/(2*h);
      maximum=std::max(maximum,std::abs(fd-p.density_potential[s]));
      close(fd,p.density_potential[s],5e-8,"spin density derivative");
      for(int axis=0;axis<3;++axis){
        auto ug=gu,dg=gd;auto& active=s?dg:ug;active[axis]+=1e-6;
        const double plus=pbe_collinear(nu,nd,ug,dg).energy_density;
        active[axis]-=2e-6;
        const double minus=pbe_collinear(nu,nd,ug,dg).energy_density;
        close((plus-minus)/2e-6,p.gradient_response[s][axis],5e-8,"spin gradient derivative");
      }
    }
  }
  using C=std::complex<double>;
  const std::vector<int> channels{1,1};const std::vector<double> overlap{.8,.15,.15,.4};
  constexpr int dim=6;std::array<std::vector<C>,2> rho;
  for(int s=0;s<2;++s){rho[s].resize(dim*dim);for(int i=0;i<dim;++i)rho[s][i*dim+i]=.1+.02*i+.03*s;
    rho[s][1]=C(.008,.003);rho[s][dim]=std::conj(rho[s][1]);}
  std::vector<C> positive{.4,C(.1,.2),C(.1,-.2),.6};
  if(!onsite_positive_semidefinite(positive,{2}))throw std::runtime_error("physical complex onsite rejected");
  auto invalid=positive;invalid[0]=-.1;
  if(onsite_positive_semidefinite(invalid,{2}))throw std::runtime_error("negative onsite accepted");
  auto linear=positive;
  // A separate PSD convex oracle uses two independent positive matrices.
  const std::vector<C> diagonal{.7,0.,0.,.2};
  for(std::size_t i=0;i<linear.size();++i)linear[i]=.2*positive[i]+.8*diagonal[i];
  if(!onsite_positive_semidefinite(linear,{2}))throw std::runtime_error("convex positive mixture rejected");
  auto u=dudarev_paw(channels,overlap,{1,4},rho);
  for(int s=0;s<2;++s)for(int i=0;i<dim;++i)for(int j=i;j<dim;++j){
    auto perturbed=rho;const double h=1e-6;
    perturbed[s][i*dim+j]+=h;if(i!=j)perturbed[s][j*dim+i]+=h;
    const double plus=dudarev_paw(channels,overlap,{1,4},perturbed).energy_ev;
    perturbed[s][i*dim+j]-=2*h;if(i!=j)perturbed[s][j*dim+i]-=2*h;
    const double minus=dudarev_paw(channels,overlap,{1,4},perturbed).energy_ev;
    close((plus-minus)/(2*h),(i==j?1:2)*u.projector_potential_ev[s][i*dim+j].real(),1e-8,"U real Hermitian derivative");
    if(i!=j){perturbed=rho;perturbed[s][i*dim+j]+=C(0,h);perturbed[s][j*dim+i]-=C(0,h);
      const double ep=dudarev_paw(channels,overlap,{1,4},perturbed).energy_ev;
      perturbed[s][i*dim+j]-=C(0,2*h);perturbed[s][j*dim+i]+=C(0,2*h);
      const double em=dudarev_paw(channels,overlap,{1,4},perturbed).energy_ev;
      close((ep-em)/(2*h),2*u.projector_potential_ev[s][i*dim+j].imag(),1e-8,"U imaginary Hermitian derivative");}
  }
  close(dudarev_paw(channels,overlap,{1,0},rho).energy_ev,0,0,"U zero limit");
  std::cout<<"SPIN_PBE_HUBBARD_ORACLE_PASS maximum_density_fd_error="<<maximum<<"\n";
}
