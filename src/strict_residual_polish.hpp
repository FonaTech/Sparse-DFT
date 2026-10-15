#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace sdft {
// The local tangent step uses an SPD inverse scale even if H_ii-lambda S_ii
// changes sign. This is only a direction; acceptance uses unchanged H/S.
inline double strict_positive_diagonal_scale(double h,double value,double s) {
  const double shifted=h-value*s;
  if(!std::isfinite(shifted))throw std::invalid_argument("nonfinite strict diagonal scale");
  return std::max(.1,std::abs(shifted));
}

// A small residual can be obscured by the backward error of a projected
// eigensolve. Refine the ORIGINAL Hx-lambda*Sx, using a tangent direction
// and its Rayleigh curvature; never accept the cached Galerkin residual.
inline double compensated_real_inner(const Eigen::VectorXcd& a,
                                     const Eigen::VectorXcd& b) {
  double sum=0., correction=0.;
  for(Eigen::Index i=0;i<a.size();++i)
    for(double term:{a[i].real()*b[i].real(),a[i].imag()*b[i].imag()}) {
      const double next=sum+term;
      correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;
      sum=next;
    }
  return sum+correction;
}

inline std::complex<double> compensated_complex_inner(const Eigen::VectorXcd& a,
                                                       const Eigen::VectorXcd& b) {
  double re=0.,im=0.,re_error=0.,im_error=0.;
  const auto add=[](double term,double& sum,double& error){const double next=sum+term;
    error+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;};
  for(Eigen::Index i=0;i<a.size();++i){
    add(a[i].real()*b[i].real(),re,re_error);add(a[i].imag()*b[i].imag(),re,re_error);
    add(a[i].real()*b[i].imag(),im,im_error);add(-a[i].imag()*b[i].real(),im,im_error);
  }
  return {re+re_error,im+im_error};
}

// Replace stale requested roots while preserving every independent old
// search direction in their S-orthogonal complement. QR acts only on the
// small coefficient matrix, never on a dense plane-wave Hamiltonian.
inline Eigen::MatrixXcd strict_preserved_search_subspace(
    const Eigen::MatrixXcd& x,const Eigen::MatrixXcd& sx,
    const Eigen::MatrixXcd& basis) {
  using M=Eigen::MatrixXcd;
  if(!x.rows()||!x.cols()||sx.rows()!=x.rows()||sx.cols()!=x.cols()||
      basis.rows()!=x.rows()||basis.cols()<x.cols()||!x.allFinite()||
      !sx.allFinite()||!basis.allFinite())
    throw std::invalid_argument("invalid corrected Davidson search subspace");
  M retained(x.rows(),basis.cols());retained.leftCols(x.cols())=x;
  if(basis.cols()>x.cols()) {
    const M cross=(sx.adjoint()*basis).adjoint();
    Eigen::HouseholderQR<M> qr(cross);
    const M q=qr.householderQ()*M::Identity(basis.cols(),basis.cols());
    retained.rightCols(basis.cols()-x.cols())=basis*q.rightCols(basis.cols()-x.cols());
  }
  return retained;
}

// Tangent corrections remove every requested root. They therefore cannot
// repair a small spurious rotation BETWEEN requested eigenvectors. Solve a
// fresh generalized Ritz problem within that complete manifold, whose
// spectral scale excludes the high-energy search guards. Accept only an
// improved original-operator certificate and the original S constraints.
template<class ApplyPair,class SolveSmall>
bool strict_root_manifold_step(Eigen::MatrixXcd& x,Eigen::MatrixXcd& sx,
    Eigen::VectorXd& values,Eigen::MatrixXcd& residual,double tolerance,
    ApplyPair apply_original_pair,SolveSmall solve_small,
    const Eigen::MatrixXcd& locked_s=Eigen::MatrixXcd{}) {
  using M=Eigen::MatrixXcd;const auto n=x.rows(),roots=x.cols();
  if(!n||!roots||sx.rows()!=n||sx.cols()!=roots||values.size()!=roots||
      residual.rows()!=n||residual.cols()!=roots||!std::isfinite(tolerance)||tolerance<=0||
      (locked_s.cols()&&locked_s.rows()!=n)||!x.allFinite()||!sx.allFinite()||
      !values.allFinite()||!residual.allFinite()||!locked_s.allFinite())
    throw std::invalid_argument("invalid strict root-manifold state");
  const double old_max=residual.colwise().norm().maxCoeff();
  if(old_max<=tolerance)return false;
  M hx,original_s;apply_original_pair(x,hx,original_s);
  M h(roots,roots),s(roots,roots);
  for(Eigen::Index i=0;i<roots;++i)for(Eigen::Index j=0;j<roots;++j){
    h(i,j)=compensated_complex_inner(x.col(i),hx.col(j));
    s(i,j)=compensated_complex_inner(x.col(i),original_s.col(j));
  }
  h=(.5*(h+h.adjoint())).eval();s=(.5*(s+s.adjoint())).eval();
  const auto solved=solve_small(h,s);
  if(solved.first.size()!=roots||solved.second.rows()!=roots||solved.second.cols()!=roots||
      !solved.first.allFinite()||!solved.second.allFinite())
    throw std::runtime_error("strict root-manifold eigensolve dimensions differ");
  M trial=x*solved.second,th,ts;apply_original_pair(trial,th,ts);
  for(Eigen::Index root=0;root<roots;++root){
    const double norm=compensated_real_inner(trial.col(root),ts.col(root));
    if(!std::isfinite(norm)||norm<=0)return false;
    trial.col(root)/=std::sqrt(norm);
  }
  apply_original_pair(trial,th,ts);
  Eigen::VectorXd energies(roots);M errors(n,roots);
  for(Eigen::Index root=0;root<roots;++root){
    energies[root]=compensated_real_inner(trial.col(root),th.col(root))/
        compensated_real_inner(trial.col(root),ts.col(root));
    for(Eigen::Index i=0;i<n;++i)errors(i,root)={
      std::fma(-energies[root],ts(i,root).real(),th(i,root).real()),
      std::fma(-energies[root],ts(i,root).imag(),th(i,root).imag())};
  }
  const double new_max=errors.colwise().norm().maxCoeff();
  const M gram=trial.adjoint()*ts;
  if(!errors.allFinite()||!energies.allFinite()||!gram.allFinite()||
      (gram-M::Identity(roots,roots)).cwiseAbs().maxCoeff()>1e-12||
      (locked_s.cols()&&(locked_s.adjoint()*trial).cwiseAbs().maxCoeff()>1e-12)||
      !(new_max<=tolerance||new_max<.999*old_max))return false;
  x=std::move(trial);sx=std::move(ts);values=std::move(energies);residual=std::move(errors);
  return true;
}

template<class ApplyPair>
bool strict_residual_polish_step(Eigen::MatrixXcd& x, Eigen::MatrixXcd& sx,
    Eigen::VectorXd& values, Eigen::MatrixXcd& residual,
    const Eigen::MatrixXcd& preconditioned_direction, double tolerance,
    ApplyPair apply_original_pair,
    const Eigen::MatrixXcd& locked=Eigen::MatrixXcd{},
    const Eigen::MatrixXcd& locked_s=Eigen::MatrixXcd{}) {
  using M=Eigen::MatrixXcd;
  const auto n=x.rows(), roots=x.cols();
  if(!n||!roots||sx.rows()!=n||sx.cols()!=roots||values.size()!=roots||
      residual.rows()!=n||residual.cols()!=roots||
      preconditioned_direction.rows()!=n||preconditioned_direction.cols()!=roots||
      !std::isfinite(tolerance)||tolerance<=0||
      locked.cols()!=locked_s.cols()||
      (locked.cols()&&(locked.rows()!=n||locked_s.rows()!=n)))
    throw std::invalid_argument("invalid strict residual polish dimensions");
  if(!x.allFinite()||!sx.allFinite()||!values.allFinite()||!residual.allFinite()||
      !preconditioned_direction.allFinite()||!locked.allFinite()||!locked_s.allFinite())
    throw std::invalid_argument("nonfinite strict residual polish state");
  const double old_max=residual.colwise().norm().maxCoeff();
  if(old_max<=tolerance)return false;
  M direction=preconditioned_direction;
  // Remove both requested roots and completed lower bands in the physical
  // S metric. A local band group is not the global lowest-root manifold:
  // ignoring its locked bands admits lower-energy correction directions.
  // Two passes keep a near-degenerate partner out of the correction.
  for(int pass=0;pass<2;++pass) {
    if(locked.cols())direction-=(locked*(locked_s.adjoint()*direction)).eval();
    direction-=(x*(sx.adjoint()*direction)).eval();
  }
  M dh,ds;
  apply_original_pair(direction,dh,ds);
  M trial=x;
  bool moved=false;
  for(Eigen::Index root=0;root<roots;++root) {
    if(residual.col(root).norm()<=.5*tolerance)continue;
    const double gradient=compensated_real_inner(direction.col(root),residual.col(root));
    const double curvature=compensated_real_inner(direction.col(root),dh.col(root))-
        values[root]*compensated_real_inner(direction.col(root),ds.col(root));
    if(!(curvature>0.)||!std::isfinite(curvature)||!std::isfinite(gradient))continue;
    const double alpha=-gradient/curvature;
    // This is a final local correction, not a new spectral search or a
    // fallback for an incorrectly selected eigenstate.
    if(!(alpha>0.)||!std::isfinite(alpha)||alpha*direction.col(root).norm()>1e-4)continue;
    for(Eigen::Index i=0;i<n;++i)
      trial(i,root)={std::fma(alpha,direction(i,root).real(),x(i,root).real()),
                     std::fma(alpha,direction(i,root).imag(),x(i,root).imag())};
    moved=true;
  }
  if(!moved)return false;
  M th,ts;
  apply_original_pair(trial,th,ts);
  for(Eigen::Index root=0;root<roots;++root) {
    const double metric=compensated_real_inner(trial.col(root),ts.col(root));
    if(!(metric>0.)||!std::isfinite(metric))return false;
    trial.col(root)/=std::sqrt(metric);
  }
  // Reapply after normalization, so the accepted vectors themselves have
  // a fresh FP64 certificate. Algebraic scaling of an old image is not it.
  apply_original_pair(trial,th,ts);
  Eigen::VectorXd energies(roots);
  M errors(n,roots);
  for(Eigen::Index root=0;root<roots;++root) {
    const double metric=compensated_real_inner(trial.col(root),ts.col(root));
    energies[root]=compensated_real_inner(trial.col(root),th.col(root))/metric;
    for(Eigen::Index i=0;i<n;++i)
      errors(i,root)={std::fma(-energies[root],ts(i,root).real(),th(i,root).real()),
                      std::fma(-energies[root],ts(i,root).imag(),th(i,root).imag())};
  }
  const double new_max=errors.colwise().norm().maxCoeff();
  const M gram=trial.adjoint()*ts;
  if(!errors.allFinite()||!energies.allFinite()||!gram.allFinite()||
      (gram-M::Identity(roots,roots)).cwiseAbs().maxCoeff()>1e-12||
      (locked.cols()&&(locked_s.adjoint()*trial).cwiseAbs().maxCoeff()>1e-12)||
      !(new_max<=tolerance||new_max<.999*old_max))return false;
  x=std::move(trial);sx=std::move(ts);values=std::move(energies);
  residual=std::move(errors);
  return true;
}
} // namespace sdft
