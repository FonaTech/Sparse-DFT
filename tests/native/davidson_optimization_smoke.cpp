#include "env_cache.hpp"
#include "pw_core.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

int main() {
  using C=std::complex<double>;using M=Eigen::MatrixXcd;
  constexpr std::size_t n=256,roots=16;
  for(double e:{0.,.01,1.,100.})for(double t:{0.,.001,1.,100.,1e100}) {
    const double scale=1.5*std::max(e,2./27.211386245988),x=t/scale;
    const double factor=sdft::kinetic_preconditioner(t,e);
    if(!std::isfinite(factor)||factor<0||factor>2./scale)return 4;
    if(x<1e20) {
      const double p=27.+18.*x+12.*x*x+8.*x*x*x;
      const double expected=(2./scale)*p/(p+16.*x*x*x*x);
      if(std::abs(factor-expected)>1e-13*expected)return 5;
    }
  }
  for(double invalid:{-1.,std::numeric_limits<double>::infinity()}){
    try{sdft::kinetic_preconditioner(invalid,1.);return 6;}catch(const std::invalid_argument&){}
  }
  std::mt19937_64 rng(847392);std::normal_distribution<double> normal;
  M source(n,n),coupling(12,n);
  for(auto* matrix:{&source,&coupling})for(Eigen::Index i=0;i<matrix->size();++i)
    matrix->data()[i]=C(normal(rng),normal(rng))/std::sqrt(double(n));
  const M unitary=Eigen::HouseholderQR<M>(source).householderQ();
  const M s=M::Identity(n,n)+coupling.adjoint()*coupling;
  const M l=Eigen::LLT<M>(s).matrixL();
  Eigen::VectorXd levels(n);
  for(std::size_t i=0;i<n;++i)levels[i]=-2.+.05*double(i/2); // exact multiplicities
  M h=(l*unitary*levels.asDiagonal()*unitary.adjoint()*l.adjoint()).eval();
  std::vector<C> hv(n*n),sv(n*n),wrong(n*roots),near_dependent(n*roots);
  const M eigenvectors=l.adjoint().triangularView<Eigen::Upper>().solve(unitary);
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){hv[i*n+j]=h(i,j);sv[i*n+j]=s(i,j);}
  for(std::size_t j=0;j<roots;++j)for(std::size_t i=0;i<n;++i){
    wrong[j*n+i]=eigenvectors(i,n-roots+j);
    near_dependent[j*n+i]=eigenvectors(i,0)+1e-8*eigenvectors(i,j);
  }
  auto check=[&](const auto& result,const M& problem,const Eigen::VectorXd& expected) {
    if(!result.converged){std::cerr<<"not converged: iterations="<<result.iterations<<" residual="<<*std::max_element(result.residual_norms.begin(),result.residual_norms.end())<<'\n';return false;}
    const Eigen::Map<const M> x(result.eigenvectors.data(),n,roots);
    const M gram=x.adjoint()*s*x;
    if((gram-M::Identity(roots,roots)).cwiseAbs().maxCoeff()>1e-11){std::cerr<<"Gram error="<<(gram-M::Identity(roots,roots)).cwiseAbs().maxCoeff()<<'\n';return false;}
    for(std::size_t j=0;j<roots;++j) {
      if(std::abs(result.eigenvalues[j]-expected[j])>1e-10 ||
          (problem*x.col(j)-result.eigenvalues[j]*s*x.col(j)).norm()>1e-12){std::cerr<<"band="<<j<<" eigen error="<<std::abs(result.eigenvalues[j]-expected[j])<<" fresh residual="<<(problem*x.col(j)-result.eigenvalues[j]*s*x.col(j)).norm()<<'\n';return false;}
    }
    return true;
  };
  setenv("SDFT_BLOCK_ORTHOGONALIZE","1",1);sdft::refresh_environment_cache();
  for(const char* cached:{"0","1"}) {
    setenv("SDFT_CACHED_METRIC_ORTHO",cached,1);sdft::refresh_environment_cache();
    for(const auto& seed:{std::vector<C>{},wrong,near_dependent}) {
      const auto result=sdft::solve_generalized_hermitian_davidson(hv,sv,n,roots,256,1e-12,seed);
      if(!check(result,h,levels)){std::cerr<<"incorrect spectrum/residual: cached_metric="<<cached<<" seeds="<<seed.size()<<'\n';return 1;}
    }
    // Change the actual operator, split degeneracies and bring an unoccupied
    // state below the old occupied manifold. A warm subspace is not proof
    // that it still contains the lowest roots.
    const auto old=sdft::solve_generalized_hermitian_davidson(hv,sv,n,roots,256,1e-12);
    Eigen::VectorXd changed_levels=levels;changed_levels[roots+3]=-2.5;
    for(std::size_t i=0;i<n;++i)changed_levels[i]+=.00001*i;
    const M changed=(l*unitary*changed_levels.asDiagonal()*unitary.adjoint()*l.adjoint()).eval();
    auto expected=changed_levels;std::sort(expected.data(),expected.data()+n);
    std::vector<C> changed_h(n*n);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)changed_h[i*n+j]=changed(i,j);
    for(const char* guards:{"32","16"}) {
      setenv("SDFT_WARM_GUARD_VECTORS",guards,1);sdft::refresh_environment_cache();
      const auto result=sdft::solve_generalized_hermitian_davidson(changed_h,sv,n,roots,256,1e-12,old.eigenvectors);
      if(!check(result,changed,expected)){std::cerr<<"missed changed warm roots: cached="<<cached<<" guards="<<guards<<'\n';return 2;}
    }
    unsetenv("SDFT_WARM_GUARD_VECTORS");sdft::refresh_environment_cache();
  }
  setenv("SDFT_CACHED_METRIC_ORTHO","bad",1);sdft::refresh_environment_cache();
  try{sdft::solve_generalized_hermitian_davidson(hv,sv,n,roots,256,1e-12);return 3;}
  catch(const std::invalid_argument&){}
  unsetenv("SDFT_CACHED_METRIC_ORTHO");sdft::refresh_environment_cache();
  std::cout<<"Davidson optimized/control independent generalized spectrum, degenerate, wrong warm, near-dependent and crossing gates PASS\n";
}
