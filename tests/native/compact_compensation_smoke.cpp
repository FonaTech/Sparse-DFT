#include "env_cache.hpp"
#include "paw_compensation.hpp"
#include "sdft_core.hpp"
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace sdft;
using Complex=std::complex<double>;
void control(const char* name,const char* value) {
#ifdef _WIN32
  _putenv_s(name,value);sdft::refresh_environment_cache();
#else
  setenv(name,value,1);sdft::refresh_environment_cache();
#endif
}
void require(bool passed,const char* message) {
  if(!passed)throw std::runtime_error(message);
}
bool equal(Vec3 a,Vec3 b) {return a.x==b.x && a.y==b.y && a.z==b.z;}
bool equal(const PawCompensationOperator::Derivatives& a,const PawCompensationOperator::Derivatives& b) {
  if(!equal(a.position,b.position))return false;
  for(int i=0;i<3;++i)if(!equal(a.shape[i],b.shape[i]))return false;
  return true;
}
void compare(const PawAugmentationData& augmentation,const std::vector<std::tuple<int,int,int>>& keys) {
  const std::array<Vec3,3> cell{{{8,.1,0},{.3,7.5,0},{0,-.2,9}}};
  const Vec3 position{.137,.217,.081},translated{.398,.169,.275};
  std::size_t dim=0;
  for(int l:augmentation.channel_l)dim+=2*l+1;
  std::vector<Complex> p(dim*dim);
  for(std::size_t i=0;i<dim;++i) {
    p[i*dim+i]=.3+.02*i;
    for(std::size_t j=0;j<i;++j) {
      p[i*dim+j]=Complex(.02*std::sin(i+3*j),.013*std::cos(2*i+j));
      p[j*dim+i]=std::conj(p[i*dim+j]);
    }
  }
  FourierDensity sparse;
  for(std::size_t i=0;i<keys.size();i+=3) {
    auto [h,k,l]=keys[i];
    sparse[{-h,-k,-l}]=Complex(.03*std::cos(i),.01*std::sin(i));
  }
  for(const char* batched:{"0","1"}) {
    control("SDFT_BATCH_COMPENSATION_GEOMETRY",batched);
    control("SDFT_COMPACT_COMPENSATION","0");
    std::vector<PawCompensationOperator> reference;
    reference.reserve(2);
    reference.emplace_back(augmentation,cell,position,keys);
    reference.emplace_back(augmentation,cell,translated,keys,AtomicUnitConvention::Codata,&reference[0]);
    control("SDFT_COMPACT_COMPENSATION","1");
    std::vector<PawCompensationOperator> compact;
    compact.reserve(2);
    compact.emplace_back(augmentation,cell,position,keys);
    compact.emplace_back(augmentation,cell,translated,keys,AtomicUnitConvention::Codata,&compact[0]);
    PawCompensationOperator independent(augmentation,cell,translated,keys);
    require(compact[1].shares_geometry_with(compact[0]),"compact geometry was not shared");
    // Reading uses the object's immutable representation after mode changes.
    control("SDFT_COMPACT_COMPENSATION","0");
    PawCompensationOperator mixed(augmentation,cell,translated,keys,AtomicUnitConvention::Codata,&compact[0]);
    require(mixed.shares_geometry_with(compact[0]),"shared compact representation lost identity");
    const auto lm=static_cast<std::size_t>((augmentation.max_l+1)*(augmentation.max_l+1));
    require(reference[0].form_factor_count()==keys.size()*lm,"legacy form count mismatch");
    require(compact[0].form_factor_count()==reference[0].form_factor_count(),"compact form count mismatch");
    const auto saved=reference[0].geometry_storage_bytes()-compact[0].geometry_storage_bytes();
    require(saved+lm==keys.size()*lm*4*sizeof(double),"compact storage lost a signed FP64 component");
    require(reference[0].atom_storage_bytes()==compact[0].atom_storage_bytes(),"atom arrays changed");
    for(std::size_t a=0;a<2;++a)require(reference[a].density(p)==compact[a].density(p),"complex charge changed");
    require(independent.density(p)==mixed.density(p),"mode change corrupted compact shared charge");
    const std::vector<std::vector<Complex>> onsite{p,p};
    require(PawCompensationOperator::density_sum(reference,onsite)==PawCompensationOperator::density_sum(compact,onsite),"batched complex charge changed");
    for(const auto& potential:{sparse,FourierDensity{},FourierDensity{{{0,0,0},Complex(.37,0)}}}) {
      for(std::size_t a=0;a<2;++a) {
        require(reference[a].project_potential(potential)==compact[a].project_potential(potential),"adjoint changed");
        require(equal(reference[a].position_energy_gradient(p,potential),compact[a].position_energy_gradient(p,potential)),"position derivative changed");
        const auto x=reference[a].shape_strain_energy_gradient(p,potential),y=compact[a].shape_strain_energy_gradient(p,potential);
        for(int i=0;i<3;++i)require(equal(x[i],y[i]),"shape derivative changed");
      }
      require(PawCompensationOperator::project_potential_batch(reference,potential)==PawCompensationOperator::project_potential_batch(compact,potential),"batched adjoint changed");
      const auto a=PawCompensationOperator::energy_derivatives_batch(reference,onsite,potential);
      const auto b=PawCompensationOperator::energy_derivatives_batch(compact,onsite,potential);
      for(std::size_t i=0;i<a.size();++i)require(equal(a[i],b[i]),"batched derivative changed");
    }
    std::cout<<"exact compact compensation PASS L="<<augmentation.max_l<<" batch="<<batched
             <<" G="<<keys.size()<<" legacy_bytes="<<reference[0].geometry_storage_bytes()
             <<" compact_bytes="<<compact[0].geometry_storage_bytes()<<'\n';
  }
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("usage: compact_compensation_smoke Si_numeric_cache");
    const auto cache=sdft::read_paw_numeric_cache(argv[1]);
    std::vector<int> channels;
    for(const auto& ch:cache.nonlocal_channels)
      for(int i=0;i<ch.nprojectors;++i)channels.push_back(ch.angular_momentum);
    std::vector<std::tuple<int,int,int>> keys;
    for(int h=-8;h<=8;++h)for(int k=-8;k<=8;++k)for(int l=-8;l<=8;++l)keys.emplace_back(h,k,l);
    std::reverse(keys.begin(),keys.end());
    compare(sdft::build_paw_augmentation(cache,channels),keys);
    // Synthetic f-channel fixture exercises every L<=6 including odd phases.
    // It validates representation, not reference-potential accuracy.
    auto high=sdft::build_paw_augmentation(cache,channels,6);
    high.channel_l={0,1,2,3};
    const auto n=high.channel_l.size();
    high.aug_moment.assign(n*n*7,0.);high.overlap_correction.assign(n*n,0.);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j) {
      if(high.channel_l[i]==high.channel_l[j])high.overlap_correction[i*n+j]=.1/(1+i+j);
      for(int L=std::abs(high.channel_l[i]-high.channel_l[j]);L<=high.channel_l[i]+high.channel_l[j];L+=2)
        high.aug_moment[high.aug_moment_index(i,j,L)]=.013*(L+1)/(1+i+j);
    }
    compare(high,keys);
    control("SDFT_COMPACT_COMPENSATION","invalid");
    bool rejected=false;
    try {PawCompensationOperator invalid(high,{{{8,0,0},{0,8,0},{0,0,8}}},{},keys);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid compact mode accepted");
    return 0;
  }catch(const std::exception& error) {
    std::cerr<<"compact_compensation_smoke FAIL: "<<error.what()<<'\n';return 1;
  }
}
