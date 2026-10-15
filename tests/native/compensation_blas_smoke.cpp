#include "env_cache.hpp"
#include "paw_compensation.hpp"
#include "compute_backend.hpp"
#include "sdft_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace sdft;using C=std::complex<double>;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double error(C a,C b){return std::abs(a-b)/std::max(1.,std::abs(b));}
double error(Vec3 a,Vec3 b){return std::max({error(a.x,b.x),error(a.y,b.y),error(a.z,b.z)});}
void check_real_blas() {
  const std::size_t m=7,n=9,k=11,ldc=m+3;
  for(auto ao:{MatrixOp::Normal,MatrixOp::Adjoint})for(auto bo:{MatrixOp::Normal,MatrixOp::Adjoint})for(bool add:{false,true}) {
    const auto ar=ao==MatrixOp::Normal?m:k,ac=ao==MatrixOp::Normal?k:m;
    const auto br=bo==MatrixOp::Normal?k:n,bc=bo==MatrixOp::Normal?n:k;
    const auto lda=ar+2,ldb=br+4;
    std::vector<double> a(lda*ac,-900.),b(ldb*bc,-800.),out(ldc*n,123.456);
    for(std::size_t j=0;j<ac;++j)for(std::size_t i=0;i<ar;++i)a[j*lda+i]=std::sin(.2*(i+1)*(j+2));
    for(std::size_t j=0;j<bc;++j)for(std::size_t i=0;i<br;++i)b[j*ldb+i]=std::cos(.17*(i+3)*(j+1));
    cpu_real_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,out.data(),ldc,add);
    for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<ldc;++i) {
      if(i>=m){require(out[j*ldc+i]==123.456,"real BLAS padding overwritten");continue;}
      long double expected=add?123.456:0.;
      for(std::size_t l=0;l<k;++l)expected+=static_cast<long double>(ao==MatrixOp::Normal?a[l*lda+i]:a[i*lda+l])*
          (bo==MatrixOp::Normal?b[j*ldb+l]:b[l*ldb+j]);
      require(error(out[j*ldc+i],double(expected))<2e-13,"real BLAS scalar stride/transposition/beta mismatch");
    }
  }
}

void compare(const PawAugmentationData& data) {
  const std::array<Vec3,3> cell{{{8,0,0},{0,9,0},{0,0,10}}};
  std::vector<std::tuple<int,int,int>> keys;
  for(int h=-10;h<=10;++h)for(int k=-10;k<=10;++k)for(int l=-10;l<=10;++l)keys.emplace_back(h,k,l);
  std::reverse(keys.begin(),keys.end()); // 9261 modes: one full tile and a tail
  std::size_t dim=0;for(int l:data.channel_l)dim+=2*l+1;
  std::vector<C> p(dim*dim);
  for(std::size_t i=0;i<dim;++i) {
    p[i*dim+i]=.3+.01*i;
    for(std::size_t j=0;j<i;++j){p[i*dim+j]=C(.012*std::sin(i+2*j),.01*std::cos(2*i+j));p[j*dim+i]=std::conj(p[i*dim+j]);}
  }
  std::vector<PawCompensationOperator> atoms;atoms.reserve(3);
  atoms.emplace_back(data,cell,Vec3{.13,.21,.08},keys);
  atoms.emplace_back(data,cell,Vec3{.39,.16,.27},keys,AtomicUnitConvention::Codata,&atoms[0]);
  atoms.emplace_back(data,cell,Vec3{.68,.73,.45},keys); // independent group, same PAW
  auto p2=p;for(auto& z:p2)z*=.7;
  const std::vector<std::vector<C>> onsite{p,p2,p};
  FourierDensity sparse;
  for(std::size_t i=0;i<keys.size();i+=3){auto [h,k,l]=keys[i];sparse[{-h,-k,-l}]=C(.004*std::cos(i),.003*std::sin(i));}
  double largest=0;
  for(const auto& potential:{sparse,FourierDensity{},FourierDensity{{{0,0,0},C(.37,0)}}}) {
    setenv("SDFT_COMPENSATION_BLAS","0",1);sdft::refresh_environment_cache();
    const auto charge=PawCompensationOperator::density_sum(atoms,onsite);
    const auto adjoint=PawCompensationOperator::project_potential_batch(atoms,potential);
    const auto derivatives=PawCompensationOperator::energy_derivatives_batch(atoms,onsite,potential);
    setenv("SDFT_COMPENSATION_BLAS","1",1);sdft::refresh_environment_cache();
    const auto got_charge=PawCompensationOperator::density_sum(atoms,onsite);
    const auto got_adjoint=PawCompensationOperator::project_potential_batch(atoms,potential);
    const auto got_derivatives=PawCompensationOperator::energy_derivatives_batch(atoms,onsite,potential);
    require(charge.size()==got_charge.size(),"compensation BLAS dropped modes");
    for(const auto& [key,value]:charge)largest=std::max(largest,error(got_charge.at(key),value));
    for(std::size_t a=0;a<atoms.size();++a) {
      for(std::size_t i=0;i<adjoint[a].size();++i)largest=std::max(largest,error(got_adjoint[a][i],adjoint[a][i]));
      largest=std::max(largest,error(got_derivatives[a].position,derivatives[a].position));
      for(int i=0;i<3;++i)largest=std::max(largest,error(got_derivatives[a].shape[i],derivatives[a].shape[i]));
    }
  }
  require(largest<5e-11,"compensation BLAS full complex scalar operator/adjoint/derivative mismatch");
  // Independent variational check on a small support, using the NEW batched
  // charge implementation on both sides of a Cartesian/strain perturbation.
  const std::vector<std::tuple<int,int,int>> small{{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{1,1,0},{-1,-1,0}};
  FourierDensity v{{{0,0,0},C(.31,0)},{{1,0,0},C(.08,.03)},{{-1,0,0},C(.08,-.03)},
      {{0,1,0},C(-.04,.02)},{{0,-1,0},C(-.04,-.02)},{{1,1,0},C(.018,-.011)},{{-1,-1,0},C(.018,.011)}};
  const auto energy=[&](const auto& lattice,Vec3 position){
    std::vector<PawCompensationOperator> a;a.emplace_back(data,lattice,position,small);
    const auto rho=PawCompensationOperator::density_sum(a,{p});double result=0;
    for(const auto& [key,z]:rho){auto [h,k,l]=key;if(auto it=v.find({-h,-k,-l});it!=v.end())result+=std::abs(determinant(lattice))*(it->second*z).real();}
    return result;
  };
  std::vector<PawCompensationOperator> a;a.emplace_back(data,cell,Vec3{.13,.21,.08},small);
  const auto derivative=PawCompensationOperator::energy_derivatives_batch(a,{p},v).front();
  const double delta=2e-5;
  const double fd=(energy(cell,{.13+delta/8,.21,.08})-energy(cell,{.13-delta/8,.21,.08}))/(2*delta);
  require(error(fd,derivative.position.x)<2e-7,"compensation BLAS Cartesian finite difference mismatch");
  auto upper=cell,lower=cell;upper[0].y+=delta*8;lower[0].y-=delta*8;
  const double strain=(energy(upper,{.13,.21,.08})-energy(lower,{.13,.21,.08}))/(2*delta);
  require(error(strain,derivative.shape[0].y)<2e-7,"compensation BLAS strain finite difference mismatch");
  std::cout<<"BLAS compensation L="<<data.max_l<<" G="<<keys.size()<<" scalar relative error="<<largest<<" PASS\n";
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("usage: compensation_blas_smoke Si_numeric_cache");
    check_real_blas();setenv("SDFT_COMPACT_COMPENSATION","1",1);sdft::refresh_environment_cache();setenv("SDFT_BATCH_COMPENSATION_GEOMETRY","1",1);sdft::refresh_environment_cache();
    const auto cache=sdft::read_paw_numeric_cache(argv[1]);std::vector<int> channels;
    for(const auto& ch:cache.nonlocal_channels)for(int i=0;i<ch.nprojectors;++i)channels.push_back(ch.angular_momentum);
    compare(sdft::build_paw_augmentation(cache,channels));
    auto high=sdft::build_paw_augmentation(cache,channels,6);high.channel_l={0,1,2,3};
    const auto n=high.channel_l.size();high.aug_moment.assign(n*n*7,0.);high.overlap_correction.assign(n*n,0.);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j) {
      if(high.channel_l[i]==high.channel_l[j])high.overlap_correction[i*n+j]=.1/(1+i+j);
      for(int l=std::abs(high.channel_l[i]-high.channel_l[j]);l<=high.channel_l[i]+high.channel_l[j];l+=2)
        high.aug_moment[high.aug_moment_index(i,j,l)]=.013*(l+1)/(1+i+j);
    }
    compare(high);return 0;
  }catch(const std::exception& e){std::cerr<<"compensation BLAS smoke FAIL: "<<e.what()<<'\n';return 1;}
}
