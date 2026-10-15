#include "env_cache.hpp"
#include "plane_wave_operator.hpp"
#include "compute_backend.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  using namespace sdft;
  using C=std::complex<double>;using M=Eigen::MatrixXcd;
  std::vector<GVector> g;
  for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j)for(int k=-2;k<=2;++k)
    g.push_back({i,j,k,{double(i)+.13,double(j)-.11,double(k)+.21},
        27.211386245988*(.6+.5*i*i+.7*j*j+.8*k*k)});
  const int n=g.size(),np=6,roots=12;
  FourierDensity v{{{0,0,0},.2},{{1,-1,0},C(.12,.09)},{{-1,1,0},C(.12,-.09)},
      {{0,1,-1},C(-.06,.11)},{{0,-1,1},C(-.06,-.11)}};
  ComplexPawOperator p;p.projectors=np;
  M b(np,n),d=M::Zero(np,np),q=M::Zero(np,np);
  for(int i=0;i<np;++i)for(int j=0;j<n;++j)
    b(i,j)=C(.02*std::sin(.33*(i+1)*(j+3)),.025*std::cos(.31*(i+2)*(j+1)));
  for(int i=0;i<np;++i) {
    d(i,i)=.09*(i-2);q(i,i)=.04+.01*i;
    if(i+1<np){d(i,i+1)=C(.02,-.01);d(i+1,i)=std::conj(d(i,i+1));
      q(i,i+1)=C(.002,.003);q(i+1,i)=std::conj(q(i,i+1));}
    for(int j=0;j<n;++j)p.amplitudes.push_back(b(i,j));
    for(int j=0;j<np;++j){p.d.push_back(d(i,j));p.q.push_back(q(i,j));}
  }
  // Fill after all off-diagonal entries have been assigned.
  p.d.clear();p.q.clear();
  for(int i=0;i<np;++i)for(int j=0;j<np;++j){p.d.push_back(d(i,j));p.q.push_back(q(i,j));}
  M h=M::Zero(n,n),s=(1+1e-12)*M::Identity(n,n);
  for(int i=0;i<n;++i)for(int j=0;j<n;++j){
    auto it=v.find({g[i].h-g[j].h,g[i].k-g[j].k,g[i].l-g[j].l});
    if(it!=v.end())h(i,j)=it->second;
  }
  for(int i=0;i<n;++i)h(i,i)+=g[i].kinetic_ev/27.211386245988;
  h+=b.adjoint()*d*b;s+=b.adjoint()*q*b;
  Eigen::GeneralizedSelfAdjointEigenSolver<M> ref(h,s);
  if(ref.info()!=Eigen::Success)return 1;
  PlaneWaveOperator op(g,v,{p},27.211386245988);
  setenv("SDFT_BLOCK_ORTHOGONALIZE","1",1);sdft::refresh_environment_cache();
  setenv("SDFT_GPU_BLOCK_ORTHO","1",1);sdft::refresh_environment_cache();
  setenv("SDFT_PROJECTOR_STATE","1",1);sdft::refresh_environment_cache();
  setenv("SDFT_PROJECTOR_METRIC","1",1);sdft::refresh_environment_cache();
  setenv("SDFT_PRECONDITIONER","kinetic",1);sdft::refresh_environment_cache();
  setenv("SDFT_CACHED_METRIC_ORTHO","0",1);sdft::refresh_environment_cache();
  const auto check=[&](const ComplexGeneralizedEigenResult& result) {
    if(!result.converged)return false;
    const Eigen::Map<const M> x(result.eigenvectors.data(),n,roots);
    if((x.adjoint()*s*x-M::Identity(roots,roots)).cwiseAbs().maxCoeff()>1e-11)return false;
    for(int j=0;j<roots;++j) {
      const double residual=(h*x.col(j)-result.eigenvalues[j]*s*x.col(j)).norm();
      if(residual>1e-12||std::abs(result.eigenvalues[j]-ref.eigenvalues()[j])>1e-10){
        std::cerr<<"band "<<j<<" fresh residual="<<residual<<'\n';return false;
      }
    }
    return true;
  };
  const M wrong=ref.eigenvectors().rightCols(roots);
  const std::vector<C> wrong_seed(wrong.data(),wrong.data()+wrong.size());
  for(const char* setting:{"0","1"}) {
    setenv("SDFT_GPU_FACTORED_METRIC",setting,1);sdft::refresh_environment_cache();
    for(const auto& seed:{std::vector<C>{},wrong_seed}) {
      const auto result=solve_generalized_hermitian_davidson(op.hamiltonian(),op.overlap(),roots,256,1e-12,seed);
      if(!check(result)){std::cerr<<"incorrect lowest spectrum: GPU metric="<<setting<<'\n';return 2;}
    }
  }
  setenv("SDFT_GPU_FACTORED_METRIC","bad",1);sdft::refresh_environment_cache();
  bool rejected=false;
  try{solve_generalized_hermitian_davidson(op.hamiltonian(),op.overlap(),roots,256,1e-12);}
  catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)return 3;
  if(argc==2)write_performance_report(argv[1]);
  std::cout<<"factored metric search/control, independent complex PAW dense spectrum, wrong warm and original fresh FP64 gate PASS\n";
}
