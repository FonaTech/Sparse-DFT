#include "pw_core.hpp"
#include "compute_backend.hpp"
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
#include "apple_modern_blas.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#ifdef SDFT_HAS_EIGEN
#include <Eigen/Dense>
#include "strict_residual_polish.hpp"
#endif
#ifdef SDFT_HAS_LAPACK
extern "C" void zggev_(const char*,const char*,const int*,std::complex<double>*,const int*,
  std::complex<double>*,const int*,std::complex<double>*,std::complex<double>*,
  std::complex<double>*,const int*,std::complex<double>*,const int*,
  std::complex<double>*,const int*,double*,int*);
#endif

namespace sdft {
#ifdef SDFT_HAS_EIGEN
namespace {
using C=std::complex<double>;using M=Eigen::MatrixXcd;using V=Eigen::VectorXcd;
struct PencilRitz {Eigen::VectorXd values;M vectors,ghosts;};
PencilRitz pencil_ritz(M h,M s) {
  const M original_s=s,original_h=h;const int n=h.rows();M vectors(n,n);V alpha(n),beta(n);
#ifdef SDFT_HAS_LAPACK
  int info=0;
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
  apple_modern_pencil(n,h.data(),s.data(),alpha.data(),beta.data(),vectors.data());
#else
  const char no='N',yes='V';const int one=1;int lwork=-1;C unused,query;
  std::vector<double> realwork(8*n);
  zggev_(&no,&yes,&n,h.data(),&n,s.data(),&n,alpha.data(),beta.data(),&unused,&one,
    vectors.data(),&n,&query,&lwork,realwork.data(),&info);
  if(info||!std::isfinite(query.real())||query.real()>std::numeric_limits<int>::max())
    throw std::runtime_error("original-pencil QZ workspace query failed");
  lwork=std::max(2*n,int(std::ceil(query.real())));std::vector<C> work(lwork);
  zggev_(&no,&yes,&n,h.data(),&n,s.data(),&n,alpha.data(),beta.data(),&unused,&one,
    vectors.data(),&n,work.data(),&lwork,realwork.data(),&info);
  if(info)throw std::runtime_error("original-pencil QZ solve failed");
#endif
#else
  throw std::runtime_error("indefinite original PAW pencil requires LAPACK QZ");
#endif
  std::vector<std::pair<double,int>> positive;std::vector<int> ghosts;
  for(int j=0;j<n;++j) {
    if(std::abs(beta[j])<1e-30)continue;
    const C value=alpha[j]/beta[j];const V sy=original_s*vectors.col(j);
    const double norm=std::real(vectors.col(j).dot(sy));
    const double bound=1e-12*vectors.col(j).norm()*sy.norm();
    if(!std::isfinite(value.real())||!std::isfinite(value.imag())||
       std::abs(value.imag())>1e-9*std::max(1.,std::abs(value.real())))continue;
    if(norm>bound)positive.push_back({value.real(),j});
    else if(norm < -bound)ghosts.push_back(j);
  }
  std::sort(positive.begin(),positive.end());
  PencilRitz out;out.values.resize(positive.size());out.vectors.resize(n,positive.size());
  for(std::size_t j=0;j<positive.size();++j) {
    const auto [value,index]=positive[j];const V sy=original_s*vectors.col(index);
    out.values[j]=value;out.vectors.col(j)=vectors.col(index)/std::sqrt(std::real(vectors.col(index).dot(sy)));
  }
  // QZ individual eigenvectors need not be S-orthogonal inside a repeated
  // root. Re-diagonalize the unchanged pencil within its positive-signature
  // invariant manifold, preserving degenerate multiplicity and original S.
  if(out.values.size()) {
    const M ph=out.vectors.adjoint()*original_h*out.vectors;
    const M ps=out.vectors.adjoint()*original_s*out.vectors;
    Eigen::GeneralizedSelfAdjointEigenSolver<M> refine((.5*(ph+ph.adjoint())).eval(),(.5*(ps+ps.adjoint())).eval());
    if(refine.info()!=Eigen::Success)throw std::runtime_error("positive-signature QZ manifold refinement failed");
    out.values=refine.eigenvalues();out.vectors=(out.vectors*refine.eigenvectors()).eval();
  }
  out.ghosts.resize(n,ghosts.size());
  for(std::size_t j=0;j<ghosts.size();++j)out.ghosts.col(j)=vectors.col(ghosts[j]);
  record_performance("original_pencil_negative_signature_ritz",0.,ghosts.size());
  return out;
}
}
#endif

ComplexGeneralizedEigenResult solve_generalized_hermitian_indefinite_davidson(
    const HermitianOperator& h,const HermitianOperator& s,std::size_t requested,
    std::size_t max_iterations,double tolerance,const std::vector<std::complex<double>>& initial) {
#if defined(SDFT_HAS_EIGEN) && defined(SDFT_HAS_LAPACK)
  PerformanceTimer timer("davidson_original_indefinite_pencil");
  const Eigen::Index n=h.dimension,roots=requested;
  if(n<roots||roots<1||s.dimension!=h.dimension||!h.apply||!s.apply||
     h.diagonal.size()!=std::size_t(n)||s.diagonal.size()!=std::size_t(n)||
     !max_iterations||!(tolerance>0)||!std::isfinite(tolerance)||
     (!initial.empty()&&initial.size()!=std::size_t(n*roots))||!h.excluded_vectors.empty())
    throw std::invalid_argument("invalid original indefinite pencil problem");
  const Eigen::Index capacity=std::min(n,std::max<Eigen::Index>(16*roots,128));
  M basis(n,capacity),hb(n,capacity),sb(n,capacity);Eigen::Index size=0,computed=0;
  const auto apply=[&](const M& x,M& hx,M& sx) {
    hx.resize(n,x.cols());sx.resize(n,x.cols());
    if(h.pair_identity&&h.pair_identity==s.pair_identity&&h.apply_pair_into)
      h.apply_pair_into(x.data(),x.cols(),hx.data(),sx.data(),true);
    else {
      for(const auto* op:{&h,&s}) {
        M& output=op==&h?hx:sx;
        if(op->apply_block_into)op->apply_block_into(x.data(),x.cols(),output.data(),true);
        else for(Eigen::Index j=0;j<x.cols();++j) {
          std::vector<C> input(x.col(j).data(),x.col(j).data()+n),value;
          op->apply(input,value);if(value.size()!=std::size_t(n))throw std::runtime_error("pencil operator dimension mismatch");
          output.col(j)=Eigen::Map<const V>(value.data(),n);
        }
      }
    }
    if(!hx.allFinite()||!sx.allFinite())throw std::runtime_error("nonfinite original-pencil H/S images");
  };
  const auto append=[&](V v) {
    if(size==capacity)return false;
    const double norm=v.norm();if(!std::isfinite(norm))throw std::runtime_error("nonfinite pencil search direction");
    if(norm==0)return false;v/=norm;
    // Euclidean CGS2 permits both signatures. S is used only in the actual
    // generalized Galerkin pencil and final certificate, never replaced.
    for(int pass=0;pass<2;++pass)if(size) {
      const V projection=basis.leftCols(size).adjoint()*v;
      v-=basis.leftCols(size)*projection;
    }
    const double remaining=v.norm();if(remaining<1e-12)return false;
    basis.col(size++)=v/remaining;return true;
  };
  std::vector<Eigen::Index> order(n);std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return h.diagonal[a]/s.diagonal[a]<h.diagonal[b]/s.diagonal[b];});
  std::uint64_t rng=0x6a09e667f3bcc909ULL;
  const auto random=[&]() {std::uint64_t z=(rng+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;z^=z>>31;return double(z>>11)*0x1.0p-53-.5;};
  const auto broad=[&](){V v(n);for(Eigen::Index i=0;i<n;++i)v[i]=C(random(),random());v.normalize();return v;};
  if(!initial.empty())for(Eigen::Index j=0;j<roots;++j)append(Eigen::Map<const V>(initial.data()+j*n,n)+1e-10*broad());
  for(Eigen::Index j=0;j<std::min(n,3*roots);++j){V v=.05*broad();v[order[j]]+=1.;append(v);}
  ComplexGeneralizedEigenResult out;out.dimension=n;out.residual_norms.resize(roots);
  M x,sx,hx,residual;Eigen::VectorXd values;unsigned polish_attempts=0;
  const auto real_dot=[](const V& a,const V& b) {
    double total=0,correction=0;for(Eigen::Index i=0;i<a.size();++i)for(const double t:{a[i].real()*b[i].real(),a[i].imag()*b[i].imag()}) {
      const double next=total+t;correction+=std::abs(total)>=std::abs(t)?(total-next)+t:(t-next)+total;total=next;
    }return total+correction;
  };
  const auto certificate=[&]() {
    apply(x,hx,sx);out.converged=true;residual.resize(n,roots);
    for(Eigen::Index j=0;j<roots;++j) {
      const double norm=real_dot(x.col(j),sx.col(j));
      if(!(norm>0)||!std::isfinite(norm))throw std::runtime_error("nonpositive final physical pencil root");
      x.col(j)/=std::sqrt(norm);
    }
    apply(x,hx,sx);
    for(Eigen::Index j=0;j<roots;++j) {
      values[j]=real_dot(x.col(j),hx.col(j))/real_dot(x.col(j),sx.col(j));
      for(Eigen::Index i=0;i<n;++i)residual(i,j)=C(std::fma(-values[j],sx(i,j).real(),hx(i,j).real()),std::fma(-values[j],sx(i,j).imag(),hx(i,j).imag()));
      out.residual_norms[j]=residual.col(j).norm();out.converged &= out.residual_norms[j]<=tolerance;
    }
    const M gram=x.adjoint()*sx;
    out.converged &= (gram-M::Identity(roots,roots)).cwiseAbs().maxCoeff()<=1e-10;
  };
  for(std::size_t iteration=0;iteration<max_iterations;++iteration) {
    if(computed<size){M hh,ss;apply(M(basis.middleCols(computed,size-computed)),hh,ss);hb.middleCols(computed,size-computed)=hh;sb.middleCols(computed,size-computed)=ss;computed=size;}
    M ph=basis.leftCols(size).adjoint()*hb.leftCols(size),ps=basis.leftCols(size).adjoint()*sb.leftCols(size);
    const auto rr=pencil_ritz((.5*(ph+ph.adjoint())).eval(),(.5*(ps+ps.adjoint())).eval());
    out.iterations=iteration+1;
    if(rr.values.size()<roots) {
      bool added=false;for(Eigen::Index j=0;j<n&&size<capacity;++j){V v=V::Zero(n);v[order[j]]=1.;if(append(v)){added=true;break;}}
      if(!added)throw std::runtime_error("original pencil lacks requested positive-signature roots");continue;
    }
    values=rr.values.head(roots);x=basis.leftCols(size)*rr.vectors.leftCols(roots);
    hx=hb.leftCols(size)*rr.vectors.leftCols(roots);sx=sb.leftCols(size)*rr.vectors.leftCols(roots);
    residual=hx-sx*values.asDiagonal();
    if(residual.colwise().norm().maxCoeff()<std::max(1e-11,10*tolerance)||iteration+1==max_iterations) {
      certificate();
      if(!out.converged&&polish_attempts<16&&iteration+1<max_iterations) {
        M direction=residual;
        for(Eigen::Index j=0;j<roots;++j){double kinetic=0.;if(h.kinetic.size()==std::size_t(n))for(Eigen::Index i=0;i<n;++i)kinetic+=std::norm(x(i,j))*h.kinetic[i];
          for(Eigen::Index i=0;i<n;++i)direction(i,j)*=h.kinetic.size()==std::size_t(n)?-kinetic_preconditioner(h.kinetic[i],kinetic):-1./std::max(.1,std::abs(h.diagonal[i]-values[j]*s.diagonal[i]));}
        ++polish_attempts;++iteration;out.iterations=iteration+1;
        if(strict_residual_polish_step(x,sx,values,residual,direction,tolerance,apply,M{},M{}))certificate();
      }
      if(out.converged)break;
    }
    if(size+roots>capacity) {
      const Eigen::Index keep=std::min<Eigen::Index>(2*roots,rr.values.size());
      M retained=basis.leftCols(size)*rr.vectors.leftCols(keep);
      M ghosts=basis.leftCols(size)*rr.ghosts.leftCols(std::min<Eigen::Index>(4,rr.ghosts.cols()));
      size=0;computed=0;for(Eigen::Index j=0;j<keep;++j)append(retained.col(j));
      for(Eigen::Index j=0;j<ghosts.cols();++j)append(ghosts.col(j));
    }
    const auto before=size;
    for(Eigen::Index j=0;j<roots&&size<capacity;++j) {
      if(residual.col(j).norm()<=.5*tolerance)continue;
      V direction=residual.col(j);double kinetic=0.;if(h.kinetic.size()==std::size_t(n))for(Eigen::Index i=0;i<n;++i)kinetic+=std::norm(x(i,j))*h.kinetic[i];
      for(Eigen::Index i=0;i<n;++i)direction[i]*=h.kinetic.size()==std::size_t(n)?-kinetic_preconditioner(h.kinetic[i],kinetic):-1./std::max(.1,std::abs(h.diagonal[i]-values[j]*s.diagonal[i]));
      append(direction);
    }
    if(size==before)for(Eigen::Index j=0;j<n;++j){V v=V::Zero(n);v[order[j]]=1.;if(append(v))break;}
  }
  if(x.cols()!=roots)throw std::runtime_error("original pencil produced no complete physical root manifold");
  certificate();out.eigenvalues.assign(values.data(),values.data()+roots);out.eigenvectors.assign(x.data(),x.data()+x.size());
  return out;
#else
  throw std::runtime_error("original indefinite pencil requires Eigen and LAPACK");
#endif
}
}
