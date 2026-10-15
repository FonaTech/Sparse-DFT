#include "env_cache.hpp"
#include "pw_core.hpp"
#include "compute_backend.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using C=std::complex<double>;using M=Eigen::MatrixXcd;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
sdft::HermitianOperator op(const M& a) {
  sdft::HermitianOperator out;out.dimension=a.rows();out.diagonal.resize(a.rows());
  for(Eigen::Index i=0;i<a.rows();++i)out.diagonal[i]=a(i,i).real();
  out.apply=[a](const auto& x,auto& y){const Eigen::VectorXcd v=a*Eigen::Map<const Eigen::VectorXcd>(x.data(),a.rows());y.assign(v.data(),v.data()+v.size());};
  out.apply_block_into=[a](const C* x,std::size_t cols,C* y,bool){Eigen::Map<M>(y,a.rows(),cols).noalias()=a*Eigen::Map<const M>(x,a.rows(),cols);};
  return out;
}
std::vector<C> payload(const M& x){return {x.data(),x.data()+x.size()};}
void check(const sdft::ComplexGeneralizedEigenResult& r,const M& h,const M& s,double tol,std::size_t budget) {
  require(r.iterations<=budget,"RMM exceeded total original iteration budget");
  require(r.trial_manifold!=r.converged,"RMM trial status ambiguous");
  const Eigen::Map<const M> x(r.eigenvectors.data(),h.rows(),r.eigenvalues.size());
  require((x.adjoint()*s*x-M::Identity(x.cols(),x.cols())).cwiseAbs().maxCoeff()<=1e-10,"RMM fresh S Gram failed");
  for(Eigen::Index j=0;j<x.cols();++j)
    require((h*x.col(j)-r.eigenvalues[j]*s*x.col(j)).norm()<=tol,"RMM original requested residual was loosened");
  if(r.converged){Eigen::GeneralizedSelfAdjointEigenSolver<M> exact(h,s);
    for(Eigen::Index j=0;j<x.cols();++j)require(std::abs(r.eigenvalues[j]-exact.eigenvalues()[j])<=1e-9,"RMM fallback missed lowest spectrum");}
}
}
int main(){try{
  constexpr Eigen::Index n=96,roots=12;
  std::mt19937_64 rng(204991);std::normal_distribution<double> normal;
  M random(n,n),b(7,n),noise(n,roots);
  for(auto* x:{&random,&b,&noise})for(Eigen::Index i=0;i<x->size();++i)x->data()[i]=C(normal(rng),normal(rng))/std::sqrt(double(n));
  const M u=Eigen::HouseholderQR<M>(random).householderQ();
  const M s=M::Identity(n,n)+b.adjoint()*b,l=Eigen::LLT<M>(s).matrixL();
  Eigen::VectorXd energies(n);for(Eigen::Index i=0;i<n;++i)energies[i]=-.6+.015*(i/2);
  const M h=l*u*energies.asDiagonal()*u.adjoint()*l.adjoint();
  auto ho=op(h),so=op(s);ho.scf_trial_update=true;ho.kinetic.resize(n);
  for(Eigen::Index i=0;i<n;++i)ho.kinetic[i]=.1+.003*i;
  Eigen::GeneralizedSelfAdjointEigenSolver<M> exact(h,s);
  const auto low=payload(M(exact.eigenvectors().leftCols(roots))),wrong=payload(M(exact.eigenvectors().rightCols(roots)));
  for(const auto& seed:{low,wrong,payload(M(exact.eigenvectors().leftCols(roots)+2e-3*noise))}) {
    const auto r=sdft::solve_generalized_hermitian_rmm_trial(ho,so,roots,256,1e-5,seed);
    check(r,h,s,1e-5,256);
    if(seed==wrong)require(r.trial_manifold&&!r.converged,"wrong exact upper RMM manifold falsely certified");
  }
  // Rank-deficient warm guesses fall back to the original protected search;
  // neither orthogonalization failure nor a zero residual produces success.
  const auto zero=sdft::solve_generalized_hermitian_rmm_trial(ho,so,roots,256,1e-5,std::vector<C>(n*roots));
  require(zero.converged&&!zero.trial_manifold,"zero RMM warm did not invoke protected search");check(zero,h,s,1e-5,256);
  setenv("SDFT_SCF_RMM_TRIAL","1",1);sdft::refresh_environment_cache();
  const auto dispatched=sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-5,low);
  check(dispatched,h,s,1e-5,256);require(dispatched.trial_manifold,"SCF-only RMM dispatch not used");
  ho.scf_trial_update=false;
  const auto strict=sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-12,wrong);
  check(strict,h,s,1e-12,256);require(strict.converged&&!strict.trial_manifold,"strict/NSCF search escaped into trial");
  const auto reject=[&](auto action){bool rejected=false;try{action();}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid RMM scope/precision accepted");};
  reject([&]{sdft::solve_generalized_hermitian_rmm_trial(ho,so,roots,256,1e-5,low);});
  ho.scf_trial_update=true;
  reject([&]{sdft::solve_generalized_hermitian_rmm_trial(ho,so,roots,256,1e-12,low);});
  reject([&]{sdft::solve_generalized_hermitian_rmm_trial(ho,so,roots,256,1e-5,{});});
  setenv("SDFT_SCF_RMM_TRIAL","bad",1);sdft::refresh_environment_cache();
  reject([&]{sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-5,low);});
  std::cout<<"Coarse residual minimization: complex noncommuting PAW S, degeneracy, wrong upper status, fresh original tolerance, rank-deficient fallback, strict lowest-spectrum search and budget PASS\n";
  return 0;
}catch(const std::exception& error){std::cerr<<"RMM trial FAIL: "<<error.what()<<'\n';return 1;}}
