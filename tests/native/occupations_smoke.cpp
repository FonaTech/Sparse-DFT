#include "occupations.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    const auto z=sdft::fill_occupations({{-2,1,2},{-.5,.1,3}},{.25,.75},3);
    if(std::abs(z.occupations[1][1]-4./3)>1e-12 || z.occupations[0][1]!=0 ||
       std::abs(z.electron_count-3)>1e-12)throw std::runtime_error("global metallic filling failed");
    const auto deg=sdft::fill_occupations({{0,0,2},{0,1,3}},{.5,.5},2);
    if(std::abs(deg.occupations[0][0]-4./3)>1e-12 ||
       std::abs(deg.occupations[0][1]-4./3)>1e-12 ||
       std::abs(deg.occupations[1][0]-4./3)>1e-12)
      throw std::runtime_error("degenerate filling breaks symmetry");
    const auto thermal=sdft::fill_occupations({{-.5,.5,10}},{1},2,.1);
    if(std::abs(thermal.chemical_potential_ev)>1e-10 ||
       std::abs(thermal.occupations[0][0]-2/(1+std::exp(-5)))>1e-12 ||
       thermal.entropy_energy_ev>=0)
      throw std::runtime_error("Fermi-Dirac analytic benchmark failed");
    std::cout<<"occupations PASS weighted_count="<<z.electron_count
      <<" mu_ev="<<thermal.chemical_potential_ev
      <<" entropy_energy_ev="<<thermal.entropy_energy_ev<<'\n';
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
