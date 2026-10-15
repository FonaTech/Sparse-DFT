#include "upf_radial_poisson.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

int main() { try {
  constexpr double pi=3.1415926535897932384626433832795;
  std::vector<double> r(2001),w(r.size()),rho(r.size()),direction(r.size()),response(r.size());
  for(std::size_t i=0;i<r.size();++i) {
    r[i]=1e-6*std::exp(.01*i);
    w[i]=.01*r[i]/3*(i==0||i+1==r.size()?1:i%2?4:2);
    rho[i]=std::sqrt(4*pi)*r[i]*r[i]*std::exp(-2*r[i])/pi;
    direction[i]=rho[i]*(.1+.03*std::sin(r[i]));
    response[i]=std::sin(.17*i)*std::exp(-r[i]);
  }
  const auto base=sdft::upf_log_radial_hartree(r,w,{rho});
  const double analytic=.3125,error=std::abs(base.energy-analytic);
  if(error>2e-8)throw std::runtime_error("hydrogenic radial Coulomb analytic gate: "+std::to_string(error));
  double adjoint_error=0,fd_error=0,minimum_energy=base.energy;
  for(int l=0;l<=4;++l) {
    sdft::UpfRadialPoisson op(r,l);
    const auto v=op.potential(direction),a=op.transpose(response);
    double lhs=0,rhs=0;
    for(std::size_t i=0;i<r.size();++i){lhs+=response[i]*v[i];rhs+=direction[i]*a[i];}
    adjoint_error=std::max(adjoint_error,std::abs(lhs-rhs));
    const std::size_t lm=l*l;
    std::vector<std::vector<double>> density(lm+1,std::vector<double>(r.size()));density[lm]=rho;
    const auto result=sdft::upf_log_radial_hartree(r,w,density);
    minimum_energy=std::min(minimum_energy,result.energy);
    double projected=0;for(std::size_t i=0;i<r.size();++i)projected+=direction[i]*result.derivative[lm][i];
    for(double h:{2e-4,1e-4}) {
      auto plus=density,minus=density;
      for(std::size_t i=0;i<r.size();++i){plus[lm][i]+=h*direction[i];minus[lm][i]-=h*direction[i];}
      const double central=(sdft::upf_log_radial_hartree(r,w,plus).energy-sdft::upf_log_radial_hartree(r,w,minus).energy)/(2*h);
      fd_error=std::max(fd_error,std::abs(central-projected));
    }
  }
  if(adjoint_error>1e-10||fd_error>1e-8||minimum_energy<=0)
    throw std::runtime_error("radial Poisson exact transpose/energy derivative/positive smooth channel gate");
  bool rejected=false;try {auto wrong=r;wrong[8]*=1.001;sdft::UpfRadialPoisson op(wrong,0);}
    catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("nonlog mesh negative control failed");
  std::cout<<std::setprecision(17)<<"{\"status\":\"PASS_MATH_ONLY\",\"analytic_hartree_error_ha\":"<<error
      <<",\"adjoint_error\":"<<adjoint_error<<",\"energy_fd_error_ha\":"<<fd_error
      <<",\"minimum_smooth_channel_energy_ha\":"<<minimum_energy<<"}\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
