#include "pw_core.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <random>

int main() {
  using M=Eigen::MatrixXcd;using C=std::complex<double>;
  constexpr int n=72,roots=8;
  std::mt19937_64 rng(19671001);std::normal_distribution<double> random;
  M seed(n,n);for(int i=0;i<seed.size();++i)seed.data()[i]=C(random(rng),random(rng));
  const M u=Eigen::HouseholderQR<M>(seed).householderQ();
  Eigen::VectorXd signs=Eigen::VectorXd::Ones(n),levels(n);
  signs[0]=-.008;signs[1]=-.3;levels[0]=-100;levels[1]=-.7;
  for(int i=2;i<n;++i)levels[i]=-2.+.05*((i-2)/2);
  const M s=u*signs.asDiagonal()*u.adjoint(),h=u*(signs.array()*levels.array()).matrix().asDiagonal()*u.adjoint();
  sdft::HermitianOperator ho,so;ho.dimension=so.dimension=n;so.indefinite_metric=true;
  for(int i=0;i<n;++i){ho.diagonal.push_back(h(i,i).real());so.diagonal.push_back(s(i,i).real());}
  ho.apply=[&](const auto& input,auto& output){const auto result=(h*Eigen::Map<const Eigen::VectorXcd>(input.data(),n)).eval();output.assign(result.data(),result.data()+n);};
  so.apply=[&](const auto& input,auto& output){const auto result=(s*Eigen::Map<const Eigen::VectorXcd>(input.data(),n)).eval();output.assign(result.data(),result.data()+n);};
  std::vector<C> wrong(n*roots);for(int j=0;j<roots;++j)std::copy_n(u.col(n-roots+j).data(),n,wrong.data()+j*n);
  for(const auto& initial:{std::vector<C>{},wrong}) {
    const auto result=sdft::solve_generalized_hermitian_davidson(ho,so,roots,256,1e-12,initial);
    if(!result.converged){std::cerr<<"not converged "<<result.iterations<<" residual="<<*std::max_element(result.residual_norms.begin(),result.residual_norms.end())<<'\n';return 1;}
    const Eigen::Map<const M> x(result.eigenvectors.data(),n,roots);
    double residual=0,spectrum=0;
    for(int j=0;j<roots;++j){residual=std::max(residual,(h*x.col(j)-result.eigenvalues[j]*s*x.col(j)).norm());spectrum=std::max(spectrum,std::abs(result.eigenvalues[j]-levels[j+2]));}
    const double orth=(x.adjoint()*s*x-M::Identity(roots,roots)).cwiseAbs().maxCoeff();
    std::cout<<"analytic indefinite pencil: iterations="<<result.iterations<<" residual="<<residual<<" spectrum="<<spectrum<<" S-Gram="<<orth<<'\n';
    if(residual>1e-12||spectrum>1e-10||orth>1e-10)return 2;
  }
  std::cout<<"negative ghost below physical roots, degenerate positive roots and wrong-warm analytic original-pencil oracle PASS\n";
}
