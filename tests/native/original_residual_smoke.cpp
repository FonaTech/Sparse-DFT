#include "pw_core.hpp"
#include "compute_backend.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using C=std::complex<double>;using M=Eigen::MatrixXcd;
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
sdft::HermitianOperator operation(const M& original,const M& search) {
  sdft::HermitianOperator out;out.dimension=original.rows();out.diagonal.resize(original.rows());
  for(Eigen::Index i=0;i<original.rows();++i)out.diagonal[i]=original(i,i).real();
  out.apply=[original](const auto& x,auto& y){const Eigen::VectorXcd v=original*Eigen::Map<const Eigen::VectorXcd>(x.data(),original.rows());y.assign(v.data(),v.data()+v.size());};
  out.apply_block_into=[original,search](const C* x,std::size_t cols,C* y,bool exact){
    Eigen::Map<M>(y,original.rows(),cols).noalias()=(exact?original:search)*Eigen::Map<const M>(x,original.rows(),cols);
  };
  return out;
}
void check(const sdft::ComplexGeneralizedEigenResult& result,const M& h,const M& s,double tolerance) {
  require(result.converged&&!result.trial_manifold,"original callback problem failed protected convergence");
  require(result.iterations<=256,"fresh original check expanded original budget");
  Eigen::GeneralizedSelfAdjointEigenSolver<M> reference(h,s);
  const Eigen::Map<const M> x(result.eigenvectors.data(),h.rows(),result.eigenvalues.size());
  require((x.adjoint()*s*x-M::Identity(x.cols(),x.cols())).cwiseAbs().maxCoeff()<1e-10,"fresh original S norm failed");
  for(Eigen::Index j=0;j<x.cols();++j) {
    require((h*x.col(j)-result.eigenvalues[j]*s*x.col(j)).norm()<=tolerance,"approximate cached residual falsely accepted as original");
    require(std::abs(reference.eigenvalues()[j]-result.eigenvalues[j])<1e-6,"original low spectrum mismatch");
  }
}
}
int main(){try{
  constexpr Eigen::Index n=128,roots=8;
  M h=M::Zero(n,n),approx=M::Zero(n,n),s=M::Identity(n,n),seed=M::Zero(n,roots);
  for(Eigen::Index i=0;i<n;++i){h(i,i)=.03*i;approx(i,i)=.03*i;}
  // Search operator deliberately suppresses a coupling by four orders of
  // magnitude. Its residual can meet coarse tolerance while the TRUE
  // operator residual fails. This is an adversarial callback contract,
  // not a simulated scientific result or a substitute accelerator.
  for(Eigen::Index j=0;j<roots;++j) {
    const auto remote=96+j;const C coupling(.025,.007);
    h(j,remote)=coupling;h(remote,j)=std::conj(coupling);
    approx(j,remote)=1e-4*coupling;approx(remote,j)=1e-4*std::conj(coupling);
    seed(j,j)=1.;
  }
  auto ho=operation(h,approx),so=operation(s,s);ho.approximate_search=true;
  ho.kinetic.resize(n);for(Eigen::Index i=0;i<n;++i)ho.kinetic[i]=.1+.03*i;
  const std::vector<C> initial(seed.data(),seed.data()+seed.size());
  const auto result=sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-4,initial);
  check(result,h,s,1e-4);
  std::random_device nonce;std::filesystem::path directory;
  for(int i=0;i<128;++i){auto p=std::filesystem::temp_directory_path()/("sparse-dft-original-residual-"+std::to_string(nonce())+"-"+std::to_string(nonce()));
    if(std::filesystem::create_directory(p)){directory=p;break;}}
  require(!directory.empty(),"cannot create isolated original-residual evidence");
  struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code error;std::filesystem::remove_all(p,error);}} cleanup{directory};
  const auto report=directory/"performance.json";sdft::write_performance_report(report.string());
  std::ifstream input(report);std::stringstream counters;counters<<input.rdbuf();
  require(counters.str().find("davidson_coarse_original_residual_rejected")!=std::string::npos,
      "adversarial test did not exercise original coarse residual rejection");
  check(sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-12,initial),h,s,1e-12);
  std::cout<<"Approximate coarse search rejected by fresh original H/S; original tolerance, strict lowest spectrum and budget PASS\n";
  return 0;
}catch(const std::exception& error){std::cerr<<"Original residual FAIL: "<<error.what()<<'\n';return 1;}}
