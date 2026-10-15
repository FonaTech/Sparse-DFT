#include "relax_optimizer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
  using namespace sdft;
  // The known curvature of a quadratic must recover its Newton direction.
  auto r=lbfgs_direction({2.,6.},{{{1.,0.},{2.,0.}},{{0.,1.},{0.,3.}}},.1,10.);
  if(std::abs(r.step[0]+1)>1e-12||std::abs(r.step[1]+2)>1e-12||r.directional_derivative>=0)
    throw std::runtime_error("L-BFGS quadratic curvature regression failed");
  r=lbfgs_direction({2.,6.},{{{1.,0.},{-2.,0.}}},.1,.05);
  if(r.used_pairs||r.rejected_pairs!=1||std::hypot(r.step[0],r.step[1])>.050000000001||r.directional_derivative>=0)
    throw std::runtime_error("L-BFGS trust/negative curvature regression failed");
  bool rejected=false;try{lbfgs_direction({NAN},{},1.,1.);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("L-BFGS accepted a nonfinite gradient");
  std::cout<<"L-BFGS curvature, descent, trust and nonfinite gates passed\n";
}
