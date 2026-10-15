#include "env_cache.hpp"
#include "pw_core.hpp"
#include "solver_search_state.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {
using C=std::complex<double>;using M=Eigen::MatrixXcd;
void require(bool b,const char* reason){if(!b)throw std::runtime_error(reason);}
sdft::HermitianOperator operation(const M& a) {
  sdft::HermitianOperator out;out.dimension=a.rows();out.diagonal.resize(a.rows());
  for(Eigen::Index i=0;i<a.rows();++i)out.diagonal[i]=a(i,i).real();
  out.apply=[a](const auto& x,auto& y) {
    const Eigen::VectorXcd v=a*Eigen::Map<const Eigen::VectorXcd>(x.data(),a.rows());
    y.assign(v.data(),v.data()+v.size());
  };
  out.apply_block_into=[a](const C* x,std::size_t columns,C* y,bool) {
    Eigen::Map<M>(y,a.rows(),columns).noalias()=a*Eigen::Map<const M>(x,a.rows(),columns);
  };
  return out;
}
void check(const sdft::ComplexGeneralizedEigenResult& result,const M& h,const M& s) {
  const auto roots=result.eigenvalues.size();
  require(result.converged,"search-state solver not converged");
  Eigen::GeneralizedSelfAdjointEigenSolver<M> ref(h,s);
  require(ref.info()==Eigen::Success,"independent generalized reference failed");
  const Eigen::Map<const M> x(result.eigenvectors.data(),h.rows(),roots);
  require((x.adjoint()*s*x-M::Identity(roots,roots)).cwiseAbs().maxCoeff()<1e-11,"search-state Gram error");
  for(std::size_t j=0;j<roots;++j) {
    require(std::abs(ref.eigenvalues()[j]-result.eigenvalues[j])<1e-10,"missed lower eigenvalue");
    require((h*x.col(j)-result.eigenvalues[j]*s*x.col(j)).norm()<=1e-12,"fresh original H/S residual failed");
  }
}
}
int main() {
  try {
    constexpr Eigen::Index n=128,roots=8;
    std::mt19937_64 rng(99531);std::normal_distribution<double> normal;
    M random(n,n),projectors(9,n);
    for(auto* a:{&random,&projectors})for(Eigen::Index i=0;i<a->size();++i)
      a->data()[i]=C(normal(rng),normal(rng))/std::sqrt(double(n));
    const M u=Eigen::HouseholderQR<M>(random).householderQ();
    const M s=M::Identity(n,n)+projectors.adjoint()*projectors;
    const M l=Eigen::LLT<M>(s).matrixL();
    Eigen::VectorXd energies(n);
    for(Eigen::Index i=0;i<n;++i)energies[i]=-2.+.05*(i/2);
    const auto matrix=[&](const Eigen::VectorXd& e)->M{return l*u*e.asDiagonal()*u.adjoint()*l.adjoint();};
    const M h=matrix(energies);auto ho=operation(h);const auto so=operation(s);
    setenv("SDFT_RECYCLE_SPECTRAL_GUARDS","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_BLOCK_ORTHOGONALIZE","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_BAND_GROUP_SOLVER","0",1);sdft::refresh_environment_cache();
    const auto old=sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-12);
    check(old,h,s);require(old.search_guard_vectors.size()==n*roots,"upper Ritz state not exported");
    for(int problem=0;problem<3;++problem) {
      auto values=energies;
      if(problem==1)values[roots+3]=-2.4; // crossing inside recycled sector
      if(problem==2)values[n-5]=-2.7;     // crossing outside recycled sector
      for(Eigen::Index i=0;i<n;++i)values[i]+=1e-5*i;
      const M changed=matrix(values);auto next=operation(changed);
      next.initial_search_guards=old.search_guard_vectors;
      const auto result=sdft::solve_generalized_hermitian_davidson(next,so,roots,256,1e-12,old.eigenvectors);
      check(result,changed,s);
      // Completely wrong upper states are guesses, never a certificate.
      const M true_vectors=l.adjoint().triangularView<Eigen::Upper>().solve(u);
      next.initial_search_guards.assign(true_vectors.data()+(n-roots)*n,true_vectors.data()+n*n);
      check(sdft::solve_generalized_hermitian_davidson(next,so,roots,256,1e-12,old.eigenvectors),changed,s);
    }
    const auto invalid=[&](std::vector<C> guards) {
      auto bad=ho;bad.initial_search_guards=std::move(guards);
      try{sdft::solve_generalized_hermitian_davidson(bad,so,roots,256,1e-12,old.eigenvectors);return false;}
      catch(const std::invalid_argument&){return true;}
    };
    require(invalid(std::vector<C>(n+1)),"unaligned guard input accepted");
    require(invalid(std::vector<C>(n*(roots+1))),"excessive guard input accepted");
    auto nonfinite=old.search_guard_vectors;nonfinite[0]={INFINITY,0};
    require(invalid(nonfinite),"nonfinite guard input accepted");
    require(invalid(std::vector<C>(n*roots)),"zero guard input accepted");
    // Combine the new search state with the existing two-group solver.
    // This catches accidental forwarding of 16 global guards to 8-root groups.
    constexpr Eigen::Index split_roots=16;
    const auto wide=sdft::solve_generalized_hermitian_davidson(ho,so,split_roots,256,1e-12);
    auto grouped=ho;grouped.initial_search_guards=wide.search_guard_vectors;
    setenv("SDFT_BAND_GROUP_SOLVER","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_BAND_GROUP_SWEEP","1",1);sdft::refresh_environment_cache();
    const auto combined=sdft::solve_generalized_hermitian_davidson(grouped,so,split_roots,256,1e-12,wide.eigenvectors);
    check(combined,h,s);
    setenv("SDFT_BAND_GROUP_SOLVER","0",1);sdft::refresh_environment_cache();
    unsetenv("SDFT_BAND_GROUP_SWEEP");sdft::refresh_environment_cache();
    const std::string sha(64,'a');
    const auto path=std::filesystem::temp_directory_path()/
      ("sparse-dft-search-state-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove(p,e);}}cleanup{path};
    sdft::save_search_guards(path.string(),sha,n,old.search_guard_vectors);
    require(sdft::load_search_guards(path.string(),sha,n,roots)==old.search_guard_vectors,"search snapshot changed complex seeds");
    bool rejected=false;
    try{sdft::load_search_guards(path.string(),std::string(64,'b'),n,roots);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"wrong parent operator identity accepted");
    std::filesystem::resize_file(path,std::filesystem::file_size(path)-1);rejected=false;
    try{sdft::load_search_guards(path.string(),sha,n,roots);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"truncated complex seed payload accepted");
    std::cout<<"Spectral search continuity, generalized degeneracies, inside/outside crossings, wrong guards, fresh FP64 and snapshot identity PASS\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"spectral_guard_smoke FAIL: "<<e.what()<<'\n';return 1;}
}
