#include "strict_residual_polish.hpp"
#include <iostream>

int main() {
  using C=std::complex<double>;using M=Eigen::MatrixXcd;
  constexpr int n=24,roots=6;
  M h=M::Zero(n,n),s=M::Zero(n,n),x=M::Zero(n,roots);
  for(int i=0;i<n;++i) {
    const double metric=1.+.07*i, energy=i<roots?-.8+.01*(i/2):1.+.4*i;
    s(i,i)=metric;h(i,i)=metric*energy;
  }
  // Each root is an independent two-level generalized eigenproblem with
  // known exact vector e_j/sqrt(S_jj). Include degenerate partners and a
  // complex error; no numerical eigensolver defines the reference answer.
  for(int j=0;j<roots;++j) {
    x(j,j)=1./std::sqrt(s(j,j).real());
    x(roots+j,j)=C(3e-10,-2e-10);
    x.col(j)/=std::sqrt((x.col(j).adjoint()*s*x.col(j))(0,0).real());
  }
  M sx=s*x,residual=h*x;
  Eigen::VectorXd energies(roots);
  for(int j=0;j<roots;++j) {
    energies[j]=(x.col(j).adjoint()*h*x.col(j))(0,0).real();
    residual.col(j)-=energies[j]*sx.col(j);
  }
  M direction=-(s.diagonal().cwiseInverse().asDiagonal()*residual);
  auto original=[&](const M& input,M& hx,M& sy){hx=h*input;sy=s*input;};
  if(!sdft::strict_residual_polish_step(x,sx,energies,residual,direction,5e-14,original))return 1;
  const M fresh=h*x-s*x*energies.asDiagonal();
  if(fresh.colwise().norm().maxCoeff()>5e-14||
      (x.adjoint()*s*x-M::Identity(roots,roots)).cwiseAbs().maxCoeff()>1e-12)return 2;
  for(int j=0;j<roots;++j) {
    Eigen::VectorXcd exact=Eigen::VectorXcd::Zero(n);
    exact[j]=1./std::sqrt(s(j,j).real());
    if((x.col(j)-exact).norm()>1e-13||std::abs(energies[j]-h(j,j).real()/s(j,j).real())>1e-14)return 3;
  }
  // Exact vectors stay untouched. The primitive cannot turn a residual
  // from a different Hamiltonian into a valid certificate.
  const M solved=x;
  // Internal contamination is invisible to a tangent correction that
  // projects away the entire requested manifold. The exact diagonal H/S
  // fixes both the reference roots and the needed complex rotation.
  M rotation=M::Identity(roots,roots);const double angle=2e-8;
  rotation(0,0)=rotation(2,2)=std::cos(angle);
  rotation(0,2)=rotation(2,0)=C(0.,std::sin(angle));
  M internal=solved*rotation,internal_s=s*internal,internal_r=h*internal;
  Eigen::VectorXd internal_e(roots);
  for(int j=0;j<roots;++j){
    internal_e[j]=sdft::compensated_real_inner(internal.col(j),internal_r.col(j));
    internal_r.col(j)-=internal_e[j]*internal_s.col(j);
  }
  if(internal_r.colwise().norm().maxCoeff()<1e-11)return 11;
  auto small=[](const M& a,const M& b){Eigen::GeneralizedSelfAdjointEigenSolver<M> eig(a,b);
    if(eig.info()!=Eigen::Success)throw std::runtime_error("small analytic-manifold solve failed");
    return std::make_pair(Eigen::VectorXd(eig.eigenvalues()),M(eig.eigenvectors()));};
  if(!sdft::strict_root_manifold_step(internal,internal_s,internal_e,internal_r,5e-14,original,small)||
      (h*internal-s*internal*internal_e.asDiagonal()).colwise().norm().maxCoeff()>5e-14||
      (internal.adjoint()*s*internal-M::Identity(roots,roots)).cwiseAbs().maxCoeff()>1e-12)return 12;
  const M unchanged_internal=internal;
  if(sdft::strict_root_manifold_step(internal,internal_s,internal_e,internal_r,5e-14,original,small)||
      internal!=unchanged_internal)return 13;
  if(sdft::strict_residual_polish_step(x,sx,energies,residual,direction,5e-14,original)||
      x!=solved)return 4;
  x(roots,0)+=1e-8;sx=s*x;
  residual=h*x-sx*energies.asDiagonal();direction=s.diagonal().cwiseInverse().asDiagonal()*residual;
  const M untouched=x;
  if(sdft::strict_residual_polish_step(x,sx,energies,residual,direction,5e-14,original)||
      x!=untouched)return 5;
  direction(0,0)={NAN,0.};
  try {sdft::strict_residual_polish_step(x,sx,energies,residual,direction,5e-14,original);return 6;}
  catch(const std::invalid_argument&) {}
  // Independent higher-root band group with already locked lower states.
  // Deliberately contaminate its preconditioner direction with those states.
  // The correction must solve the original H/S, retain all S constraints,
  // and never collapse into the lower-energy excluded manifold.
  constexpr int completed=2;
  M local=M::Zero(n,roots),locked=M::Zero(n,completed);
  for(int j=0;j<completed;++j)locked(j,j)=1./std::sqrt(s(j,j).real());
  for(int j=0;j<roots;++j) {
    const int index=completed+j;
    local(index,j)=1./std::sqrt(s(index,index).real());
    local(completed+roots+j,j)=C(3e-10,-2e-10);
    local.col(j)/=std::sqrt((local.col(j).adjoint()*s*local.col(j))(0,0).real());
  }
  M local_s=s*local,local_r=h*local;
  Eigen::VectorXd local_e(roots);
  for(int j=0;j<roots;++j) {
    local_e[j]=(local.col(j).adjoint()*h*local.col(j))(0,0).real();
    local_r.col(j)-=local_e[j]*local_s.col(j);
  }
  M local_d=-(s.diagonal().cwiseInverse().asDiagonal()*local_r);
  for(int j=0;j<roots;++j)local_d.col(j)+=C(1e-7,-2e-7)*locked.col(j%completed);
  const M locked_s=s*locked;
  if(!sdft::strict_residual_polish_step(local,local_s,local_e,local_r,local_d,
      5e-14,original,locked,locked_s))return 7;
  if((h*local-s*local*local_e.asDiagonal()).colwise().norm().maxCoeff()>5e-14||
      (locked_s.adjoint()*local).cwiseAbs().maxCoeff()>1e-14)return 8;
  for(int j=0;j<roots;++j) {
    Eigen::VectorXcd exact=Eigen::VectorXcd::Zero(n);
    exact[completed+j]=1./std::sqrt(s(completed+j,completed+j).real());
    if((local.col(j)-exact).norm()>1e-13)return 9;
  }
  // A pre-existing forbidden lower-band component cannot receive a
  // certificate merely because the other requested roots are orthogonal.
  local(0,0)+=1e-8;local_s=s*local;
  local_r=h*local-local_s*local_e.asDiagonal();
  local_d=-(s.diagonal().cwiseInverse().asDiagonal()*local_r);
  const M forbidden=local;
  if(sdft::strict_residual_polish_step(local,local_s,local_e,local_r,local_d,
      5e-14,original,locked,locked_s)||local!=forbidden)return 10;
  if(sdft::strict_root_manifold_step(local,local_s,local_e,local_r,5e-14,original,small,locked_s)||
      local!=forbidden)return 14;
  // An accepted tiny out-of-search correction must survive, while the
  // known six empty search directions remain present. Prematurely retaining
  // only occupied roots fails this independent exact-vector challenge.
  M search=M::Zero(n,2*roots),corrected=M::Zero(n,roots);
  for(int j=0;j<2*roots;++j)search(j,j)=1./std::sqrt(s(j,j).real());
  corrected=search.leftCols(roots);
  for(int j=0;j<roots;++j)corrected(2*roots+j,j)=C(1e-10,-2e-10);
  const M preserved=sdft::strict_preserved_search_subspace(corrected,s*corrected,search);
  if(preserved.cols()!=2*roots||preserved.leftCols(roots)!=corrected)return 15;
  if((corrected.adjoint()*s*preserved.rightCols(roots)).cwiseAbs().maxCoeff()>1e-13)return 16;
  const M guard=search.rightCols(roots),gram=preserved.adjoint()*s*preserved;
  Eigen::LLT<M> chol(gram);
  if(chol.info()!=Eigen::Success||
      (guard-preserved*chol.solve(preserved.adjoint()*s*guard)).norm()>1e-13)return 17;
  // Independent rotated three-level counterexample: for the second root,
  // the signed H_ii-lambda denominator sends the projected residual uphill.
  // The positive local inverse repairs that direction without changing H.
  M u=M::Identity(3,3);
  u(0,0)=-std::sqrt(2./3.);u(2,0)=std::sqrt(1./3.);
  u(0,2)=std::sqrt(1./3.);u(2,2)=std::sqrt(2./3.);
  Eigen::Vector3d levels;levels<<-3.,-1.,2.;
  const M dense=u*levels.asDiagonal()*u.adjoint();
  M challenge=u.leftCols(2);challenge.col(1)+=C(1e-10,-2e-10)*u.col(2);
  challenge.col(1).normalize();M cs=challenge,cr=dense*challenge;
  Eigen::VectorXd ce(2);
  for(int j=0;j<2;++j){ce[j]=sdft::compensated_real_inner(challenge.col(j),cr.col(j));cr.col(j)-=ce[j]*challenge.col(j);}
  M signed_d=cr,positive_d=cr;
  for(int j=0;j<2;++j)for(int i=0;i<3;++i){
    double signed_scale=dense(i,i).real()-ce[j];
    if(std::abs(signed_scale)<.1)signed_scale=std::copysign(.1,signed_scale);
    signed_d(i,j)/=-signed_scale;
    positive_d(i,j)/=-sdft::strict_positive_diagonal_scale(dense(i,i).real(),ce[j],1.);
  }
  auto dense_original=[&](const M& v,M& hv,M& sv){hv=dense*v;sv=v;};
  const M before_challenge=challenge;
  if(sdft::strict_residual_polish_step(challenge,cs,ce,cr,signed_d,5e-14,dense_original)||challenge!=before_challenge)return 18;
  if(!sdft::strict_residual_polish_step(challenge,cs,ce,cr,positive_d,5e-14,dense_original)||
      (dense*challenge-challenge*ce.asDiagonal()).colwise().norm().maxCoeff()>5e-14)return 19;
  std::cout<<"analytic complex generalized two-level roots, degeneracy, fresh residual and rejection gates PASS\n";
}
