#include "paw_augmentation.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::runtime_error("usage: paw_reference_smoke numeric.bin");
    using namespace sdft;
    const auto cache=read_paw_numeric_cache(argv[1]);
    std::vector<int> channels;std::size_t dim=0;
    for(const auto& c:cache.nonlocal_channels)for(int p=0;p<c.nprojectors;++p) {
      channels.push_back(c.angular_momentum);dim+=2*c.angular_momentum+1;
    }
    const auto density=paw_atomic_reference_density(cache,channels);
    for(auto mode:{PawRadialPotentialMode::DiscreteVariational,PawRadialPotentialMode::RadialDirect}) {
      const auto ref=paw_atomic_reference_potentials(cache,channels,PawRadialLengthUnit::Angstrom,mode);
      const auto closure=paw_one_center_closure(cache,channels,density,dim,1,PawRadialLengthUnit::Angstrom,&ref,mode);
      double max_d=0;
      for(const auto& value:closure.d_matrix_ev)max_d=std::max(max_d,std::abs(value));
      std::cout<<"atomic mode="<<(mode==PawRadialPotentialMode::RadialDirect?"radial_direct":"discrete")
        <<" reference_cancellation_ev="<<max_d<<" core_xc_ev="<<ref.core_xc_energy_ev<<'\n';
      if(max_d>1e-10)throw std::runtime_error("atomic-reference matrix fails cancellation in its own convention");
      if(mode==PawRadialPotentialMode::RadialDirect&&
         std::abs(ref.core_xc_energy_ev-(-527.3399904487892))>1e-6)
        throw std::runtime_error("Si core XC differs from the matched VASP reference value");
    }
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
