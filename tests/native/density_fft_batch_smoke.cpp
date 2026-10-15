#include "env_cache.hpp"
#include "density_functionals.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

double compare(const sdft::DensityFunctionals& a,const sdft::DensityFunctionals& b){
  if(a.grid_points!=b.grid_points||a.potential.size()!=b.potential.size()||
      a.xc_potential.size()!=b.xc_potential.size()||a.clipped_density_points!=b.clipped_density_points)
    throw std::runtime_error("batched density support/quadrature changed");
  double error=std::max({std::abs(a.xc_energy-b.xc_energy),std::abs(a.hartree_energy-b.hartree_energy),
    std::abs(a.vxc_density-b.vxc_density),std::abs(a.sampled_electrons-b.sampled_electrons)});
  for(const auto& [key,value]:a.potential)error=std::max(error,std::abs(value-b.potential.at(key)));
  for(const auto& [key,value]:a.xc_potential)error=std::max(error,std::abs(value-b.xc_potential.at(key)));
  for(int i=0;i<3;++i){const auto x=a.xc_stress[i],y=b.xc_stress[i];
    error=std::max({error,std::abs(x.x-y.x),std::abs(x.y-y.y),std::abs(x.z-y.z)});}
  return error;
}

int main(int argc,char** argv){
  using namespace sdft;
  try{
    const bool benchmark=argc>1&&std::string(argv[1])=="--benchmark";
    const int extent=benchmark&&argc>2?std::stoi(argv[2]):16;
    if(extent<2||extent>64)throw std::invalid_argument("FFT density benchmark extent in [2,64]");
    setenv("SDFT_DENSITY_TILED_REDUCTION","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_DENSITY_FUSE_PBE_TILES","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_FROZEN_CORE_FFT_REUSE","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_DENSITY_FFT_PLAN","estimate",1);sdft::refresh_environment_cache();
    setenv("SDFT_FAST_DENSITY_GRID","1",1);sdft::refresh_environment_cache();
    const std::array<Vec3,3> cell{{{3.1,.2,.1},{.1,3.3,.3},{.2,.1,3.7}}};
    FourierDensity density{{{0,0,0},.5}},core{{{0,0,0},.1},{{0,1,0},.002},{{0,-1,0},.002}};
    for(int h=-extent;h<=extent;++h)for(int k=-extent;k<=extent;++k)for(int l=-extent;l<=extent;++l){
      const int square=h*h+k*k+l*l;
      if(square&&square<=extent*extent){
        const double amplitude=1e-5/std::pow(1.+square,2.);
        const double phase=.17*h+.23*k-.19*l;
        density[{h,k,l}]={amplitude*std::cos(phase),amplitude*std::sin(phase)};
      }
    }
    // Core-only reciprocal keys must remain in XC quadrature and output
    // support even when absent from the valence/Hartree support.
    core[{extent+1,0,0}]={1e-6,2e-7};core[{-extent-1,0,0}]={1e-6,-2e-7};
    double error=0.;std::vector<double> reference_seconds,batched_seconds;
    for(int variant=0;variant<3;++variant){
      auto lattice=cell;if(variant==2)lattice[0].y+=.13;
      auto rho=density;if(variant)for(auto& [key,value]:rho){
        const auto [h,k,l]=key;value*=std::exp(std::complex<double>(0,.071*h-.043*k+.037*l));}
      const FourierDensity* ncore=variant==1?nullptr:&core;
      setenv("SDFT_DENSITY_FLAT_SUPPORT","0",1);sdft::refresh_environment_cache();
      setenv("SDFT_DENSITY_BATCHED_FFT","0",1);sdft::refresh_environment_cache();
      const auto original=evaluate_density_functionals(rho,lattice,ncore);
      setenv("SDFT_DENSITY_BATCHED_FFT","1",1);sdft::refresh_environment_cache();
      const auto batched=evaluate_density_functionals(rho,lattice,ncore);
      error=std::max(error,compare(original,batched));
      setenv("SDFT_DENSITY_FLAT_SUPPORT","1",1);sdft::refresh_environment_cache();
      error=std::max(error,compare(original,evaluate_density_functionals(rho,lattice,ncore)));
      setenv("SDFT_DENSITY_BATCHED_FFT","0",1);sdft::refresh_environment_cache();
      error=std::max(error,compare(original,evaluate_density_functionals(rho,lattice,ncore)));
      setenv("SDFT_DENSITY_FLAT_SUPPORT","0",1);sdft::refresh_environment_cache();
      if(benchmark&&variant==0)for(int rep=0;rep<3;++rep)for(int pass=0;pass<2;++pass){
        const bool use_batch=(rep+pass)%2;
        setenv("SDFT_DENSITY_BATCHED_FFT",use_batch?"1":"0",1);sdft::refresh_environment_cache();
        const auto start=std::chrono::steady_clock::now();
        const auto result=evaluate_density_functionals(rho,lattice,ncore);
        (use_batch?batched_seconds:reference_seconds).push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        error=std::max(error,compare(original,result));
      }
    }
    if(error>1e-12)throw std::runtime_error("batched density FFT energy/potential/stress original tolerance failed: "+std::to_string(error));
    std::cout<<std::setprecision(17)<<"{\"scope\":\"density FFT/PBE/Hartree kernel, not complete SCF\",\"extent\":"<<extent
      <<",\"maximum_error\":"<<error<<",\"limit\":1e-12,\"passed\":true";
    if(benchmark){std::sort(reference_seconds.begin(),reference_seconds.end());std::sort(batched_seconds.begin(),batched_seconds.end());
      std::cout<<",\"reference_seconds\":"<<reference_seconds[1]<<",\"batched_seconds\":"<<batched_seconds[1]
        <<",\"kernel_speedup\":"<<reference_seconds[1]/batched_seconds[1];}
    std::cout<<"}\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
