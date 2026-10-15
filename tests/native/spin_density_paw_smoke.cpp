#include "density_functionals.hpp"
#include "paw_augmentation.hpp"
#include "occupations.hpp"
#include "paw_header_radius.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace sdft;
void check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(int argc,char** argv){try{
  const auto radius=paw_header_rdep_angstrom("   RDEP = 2.442 radius for radial grids");
  check(radius && std::abs(*radius-1.292250842058)<1e-14,"RDEP file units");
  check(!paw_header_rdep_angstrom("   RDEPT = 1.890 core radius for aug-charge"),"RDEPT confused with RDEP");
  check(!paw_header_rdep_angstrom("   OTHER_RDEP = 2.442"),"non-RDEP key accepted");
  for(const auto& text:{"RDEP = nan","RDEP = -1","RDEP = invalid"}){
    bool rejected=false;try{paw_header_rdep_angstrom(text);}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"invalid RDEP accepted");
  }
  std::array<Vec3,3> lattice{{{7.1,0,0},{.3,7.3,0},{.2,.1,7.7}}};
  FourierDensity n{{{0,0,0},.02},{{1,0,0},{.001,.0002}},{{-1,0,0},{.001,-.0002}},{{0,1,0},.0004},{{0,-1,0},.0004}};
  FourierDensity core{{{0,0,0},.003},{{1,0,0},.0002},{{-1,0,0},.0002}};
  std::array<FourierDensity,2> spin{n,n};for(auto& ch:spin)for(auto& [k,v]:ch)v*=.5;
  const auto old=evaluate_density_functionals(n,lattice,&core);
  const auto equal=evaluate_collinear_density_functionals(spin,lattice,&core);
  check(std::abs(old.hartree_energy-equal[0].hartree_energy-equal[1].hartree_energy)<1e-12,"Hartree counted twice");
  check(std::abs(old.xc_energy-equal[0].xc_energy-equal[1].xc_energy)<1e-10,"spin XC nonmag energy");
  for(int s=0;s<2;++s)for(const auto& [k,v]:old.potential)check(std::abs(v-equal[s].potential.at(k))<1e-10,"spin nonmag Euler potential");
  spin[0][{0,0,0}]*=1.3;spin[1][{0,0,0}]*=.7;
  const auto f=evaluate_collinear_density_functionals(spin,lattice,&core);
  const double h=1e-7,volume=std::abs(determinant(lattice));
  auto energy=[&](const auto& rho){const auto v=evaluate_collinear_density_functionals(rho,lattice,&core);return v[0].hartree_energy+v[1].hartree_energy+v[0].xc_energy+v[1].xc_energy;};
  double maxfd=0;
  for(int s=0;s<2;++s){auto plus=spin,minus=spin;plus[s][{1,0,0}]+=h;plus[s][{-1,0,0}]+=h;minus[s][{1,0,0}]-=h;minus[s][{-1,0,0}]-=h;
    const double fd=(energy(plus)-energy(minus))/(2*h),analytic=2*volume*f[s].potential.at({1,0,0}).real();maxfd=std::max(maxfd,std::abs(fd-analytic));}
  check(maxfd<1e-6,"spin grid energy/potential derivative");
  const auto fill=fill_collinear_occupations({{0.,2.,8.},{1.,3.,9.}},{1.},2.5,.1);
  check(std::abs(fill.electron_count-2.5)<1e-10,"shared chemical potential charge");
  for(const auto& ch:fill.occupations)for(double v:ch)check(v>=0&&v<=1,"spin occupation cap");
  if(argc==2){const auto cache=read_paw_numeric_cache(argv[1]);std::vector<int> l;std::size_t dim=0;
    for(const auto& c:cache.nonlocal_channels)for(int p=0;p<c.nprojectors;++p){l.push_back(c.angular_momentum);dim+=2*c.angular_momentum+1;}
    const auto density=paw_atomic_reference_density(cache,l);std::array<std::vector<std::complex<double>>,2> rho{density,density};for(auto& ch:rho)for(auto& v:ch)v*=.5;
    const auto mode=PawRadialPotentialMode::AsphericalDirect;
    const auto ref=paw_atomic_reference_potentials(cache,l,PawRadialLengthUnit::Angstrom,mode,AtomicUnitConvention::VaspLegacy);
    const auto a=paw_one_center_closure(cache,l,density,dim,1,PawRadialLengthUnit::Angstrom,&ref,mode,AtomicUnitConvention::VaspLegacy);
    const auto b=paw_collinear_one_center_closure(cache,l,rho,dim,PawRadialLengthUnit::Angstrom,&ref,mode,AtomicUnitConvention::VaspLegacy);
    double matrix=0;for(int s=0;s<2;++s)for(std::size_t i=0;i<a.d_matrix_ev.size();++i)matrix=std::max(matrix,std::abs(a.d_matrix_ev[i]-b[s].d_matrix_ev[i]));
    std::cout<<"PAW nonmag matrix="<<matrix<<" energy="<<a.energy_difference_ev-b[0].energy_difference_ev-b[1].energy_difference_ev<<'\n';
    check(matrix<1e-7,"PAW spin nonmag matrix");
    check(std::abs(a.energy_difference_ev-b[0].energy_difference_ev-b[1].energy_difference_ev)<1e-7,"PAW spin nonmag ledger");
  }
  std::cout<<"SPIN_DENSITY_PAW_ORACLE_PASS max_grid_derivative_error="<<maxfd<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
