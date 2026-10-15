#include "pw_seed_transport.hpp"
#include <iostream>
#include <limits>

int main() {
  using namespace sdft;
  using C=std::complex<double>;
  const std::vector<GVector> old_g{{1,0,0},{0,0,0},{-1,0,0}};
  const std::vector<GVector> new_g{{-1,0,0},{2,0,0},{1,0,0}};
  const auto moved=transport_plane_wave_seeds(old_g,new_g,{{1,2},{3,4},{5,6}},1);
  if(moved.coefficients!=std::vector<C>{{5,6},{0,0},{1,2}}||moved.common_indices!=2||
     std::abs(moved.minimum_retained_norm_fraction-66./91.)>1e-15)return 1;
  auto duplicate=new_g;duplicate[1]=duplicate[0];
  try {transport_plane_wave_seeds(old_g,duplicate,{{1,0},{2,0},{3,0}},1);return 2;}
  catch(const std::invalid_argument&) {}
  try {transport_plane_wave_seeds(old_g,new_g,{{std::numeric_limits<double>::quiet_NaN(),0},{2,0},{3,0}},1);return 3;}
  catch(const std::invalid_argument&) {}
  // At a band crossing the transported old lowest subspace can now be
  // excited. Independent guarded search must still find the new lowest
  // roots; a small residual of the old manifold alone is insufficient.
  constexpr std::size_t n=96,roots=4;
  HermitianOperator h,s;h.dimension=s.dimension=n;
  h.diagonal.resize(n);s.diagonal.assign(n,1.);
  for(std::size_t i=0;i<n;++i)h.diagonal[i]=i<roots?5.+i:i<2*roots?-2.+.25*(i-roots):10.+i;
  h.apply=[&](const auto& x,auto& y){y.resize(n);for(std::size_t i=0;i<n;++i)y[i]=h.diagonal[i]*x[i];};
  s.apply=[](const auto& x,auto& y){y=x;};
  std::vector<C> wrong(n*roots);
  for(std::size_t j=0;j<roots;++j)wrong[j*n+j]=1.;
  const auto result=solve_generalized_hermitian_davidson(h,s,roots,256,1e-12,wrong);
  if(!result.converged)return 4;
  for(std::size_t j=0;j<roots;++j)
    if(std::abs(result.eigenvalues[j]-(-2.+.25*j))>1e-11||result.residual_norms[j]>1e-12)return 5;
  std::cout<<"integer-G transport, invalid data, and guarded crossing PASS\n";
}
