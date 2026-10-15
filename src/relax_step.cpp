#include "relax_optimizer.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
int main() {
  try {
    std::string magic;std::size_t n,m;double scale,trust;
    std::cin>>magic>>n>>m>>scale>>trust;
    if(!std::cin||magic!="SDFT_LBFGS_V1"||!n||n>10000000||m>64)
      throw std::invalid_argument("invalid Sparse-DFT L-BFGS input");
    std::vector<double> g(n);for(auto& v:g)std::cin>>v;
    std::vector<sdft::LbfgsPair> pairs(m);
    for(auto& p:pairs){p.displacement.resize(n);p.gradient_change.resize(n);
      for(auto& v:p.displacement)std::cin>>v;for(auto& v:p.gradient_change)std::cin>>v;}
    std::string trailing;if(!std::cin||(std::cin>>trailing))throw std::invalid_argument("truncated or trailing L-BFGS input");
    const auto r=sdft::lbfgs_direction(g,pairs,scale,trust);
    std::cout<<std::setprecision(17)<<"{\"step\":[";
    for(std::size_t i=0;i<n;++i)std::cout<<(i?",":"")<<r.step[i];
    std::cout<<"],\"used_pairs\":"<<r.used_pairs<<",\"rejected_pairs\":"<<r.rejected_pairs
      <<",\"directional_derivative\":"<<r.directional_derivative
      <<",\"steepest_descent\":"<<(r.steepest_descent?"true":"false")<<"}\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
