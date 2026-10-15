#include "pw_core.hpp"
#include "compute_backend.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef SDFT_HAS_LAPACK
extern "C" void zhegv_(const int*,const char*,const char*,const int*,std::complex<double>*,const int*,
    std::complex<double>*,const int*,double*,std::complex<double>*,const int*,double*,int*);
#endif

namespace sdft {
namespace {
using C=std::complex<double>;
using M=Eigen::MatrixXcd;
struct Images {M x,h,s;};

M product(const M& a,const M& b,bool adjoint=false) {
  M out(adjoint?a.cols():a.rows(),b.cols());
  complex_gemm(adjoint?MatrixOp::Adjoint:MatrixOp::Normal,MatrixOp::Normal,
      out.rows(),out.cols(),b.rows(),a.data(),a.rows(),b.data(),b.rows(),out.data(),out.rows(),true);
  return out;
}

Images apply(const HermitianOperator& h,const HermitianOperator& s,M x,bool exact) {
  Images out{std::move(x),M{},M{}};
  // The images own their complete lifetime.
  out.h.resize(out.x.rows(),out.x.cols());out.s.resize(out.x.rows(),out.x.cols());
  if(h.pair_identity&&h.pair_identity==s.pair_identity&&h.apply_pair_into) {
    if(h.projector_rows&&h.projector_rows==s.projector_rows&&h.project_into&&h.apply_projected_pair_into) {
      M amplitudes(h.projector_rows,out.x.cols());
      h.project_into(out.x.data(),out.x.cols(),amplitudes.data(),exact);
      h.apply_projected_pair_into(out.x.data(),out.x.cols(),amplitudes.data(),out.h.data(),out.s.data(),exact);
      record_performance("rmm_shared_projector_columns",0,out.x.cols());
    }else h.apply_pair_into(out.x.data(),out.x.cols(),out.h.data(),out.s.data(),exact);
  }else {
    const auto one=[&](const HermitianOperator& op,M& result) {
      if(op.apply_block_into)op.apply_block_into(out.x.data(),out.x.cols(),result.data(),exact);
      else if(op.apply_block){std::vector<C> input(out.x.data(),out.x.data()+out.x.size()),values;
        op.apply_block(input,out.x.cols(),values,exact);
        if(values.size()!=std::size_t(out.x.size()))throw std::runtime_error("RMM block operator size mismatch");
        result=Eigen::Map<const M>(values.data(),out.x.rows(),out.x.cols());
      }else for(Eigen::Index j=0;j<out.x.cols();++j){std::vector<C> input(out.x.col(j).data(),out.x.col(j).data()+out.x.rows()),values;
        op.apply(input,values);if(values.size()!=std::size_t(out.x.rows()))throw std::runtime_error("RMM scalar operator size mismatch");
        result.col(j)=Eigen::Map<const Eigen::VectorXcd>(values.data(),out.x.rows());}
    };
    one(h,out.h);one(s,out.s);
  }
  if(!out.x.allFinite()||!out.h.allFinite()||!out.s.allFinite())throw std::runtime_error("nonfinite RMM original operator output");
  record_performance("rmm_hs_columns",0,out.x.cols());
  return out;
}

M rotation(M h,M s) {
  h=(.5*(h+h.adjoint())).eval();s=(.5*(s+s.adjoint())).eval();
#ifdef SDFT_HAS_LAPACK
  const int n=h.rows(),type=1;const char vectors='V',lower='L';int info=0,workspace=-1;C query;
  std::vector<double> values(n),rwork(std::max(1,3*n-2));
  zhegv_(&type,&vectors,&lower,&n,h.data(),&n,s.data(),&n,values.data(),&query,&workspace,rwork.data(),&info);
  if(info||!std::isfinite(query.real()))throw std::runtime_error("RMM global rotation query failed");
  workspace=std::max(1,int(query.real()));std::vector<C> work(workspace);
  zhegv_(&type,&vectors,&lower,&n,h.data(),&n,s.data(),&n,values.data(),work.data(),&workspace,rwork.data(),&info);
  if(info)throw std::runtime_error("RMM global rotation overlap is not positive");
  return h;
#else
  Eigen::GeneralizedSelfAdjointEigenSolver<M> solve(h,s);
  if(solve.info()!=Eigen::Success)throw std::runtime_error("RMM global rotation failed");
  return solve.eigenvectors();
#endif
}

void rotate(Images& v,const M& u) {
  v.x=product(v.x,u);v.h=product(v.h,u);v.s=product(v.s,u);
  record_performance("rmm_rotated_hs_columns_reused",0,u.cols());
}

bool full_rank(const Images& v) {
  const M gram=product(v.x,v.s,true);
  Eigen::SelfAdjointEigenSolver<M> check((.5*(gram+gram.adjoint())).eval(),Eigen::EigenvaluesOnly);
  return check.info()==Eigen::Success&&check.eigenvalues().allFinite()&&
      check.eigenvalues().minCoeff()>1e-13*std::max(1.,check.eigenvalues().maxCoeff());
}

void project(Images& d,const Images& x) {
  // Owned exact H/S images follow every linear update. They belong to this
  // unchanged operator only; none crosses an SCF/geometry/precision boundary.
  for(int pass=0;pass<2;++pass){const M c=product(x.x,d.s,true);
    d.x-=product(x.x,c);d.h-=product(x.h,c);d.s-=product(x.s,c);}
}

void normalize(Images& v) {
  for(Eigen::Index j=0;j<v.x.cols();++j) {
    const double norm=std::real(v.x.col(j).dot(v.s.col(j)));
    if(norm>1e-28&&std::isfinite(norm)) {
      const double scale=1./std::sqrt(norm);v.x.col(j)*=scale;v.h.col(j)*=scale;v.s.col(j)*=scale;
    }else{v.x.col(j).setZero();v.h.col(j).setZero();v.s.col(j).setZero();}
  }
}

double residual(const Images& v,std::vector<double>& values,std::vector<double>& norms,M* output=nullptr) {
  double maximum=0;values.resize(v.x.cols());norms.resize(v.x.cols());
  if(output)output->resize(v.x.rows(),v.x.cols());
  for(Eigen::Index j=0;j<v.x.cols();++j) {
    long double numerator=0,denominator=0,square=0;
    for(Eigen::Index i=0;i<v.x.rows();++i) {
      numerator+=static_cast<long double>(v.x(i,j).real())*v.h(i,j).real()+static_cast<long double>(v.x(i,j).imag())*v.h(i,j).imag();
      denominator+=static_cast<long double>(v.x(i,j).real())*v.s(i,j).real()+static_cast<long double>(v.x(i,j).imag())*v.s(i,j).imag();
    }
    if(!(denominator>0))throw std::runtime_error("RMM nonpositive wavefunction metric");
    const double energy=double(numerator/denominator);values[j]=energy;
    for(Eigen::Index i=0;i<v.x.rows();++i) {
      const double re=std::fma(-energy,v.s(i,j).real(),v.h(i,j).real()),im=std::fma(-energy,v.s(i,j).imag(),v.h(i,j).imag());
      square+=static_cast<long double>(re)*re+static_cast<long double>(im)*im;
      if(output)(*output)(i,j)=C(re,im);
    }
    norms[j]=std::sqrt(double(square));maximum=std::max(maximum,norms[j]);
  }
  return maximum;
}
}

ComplexGeneralizedEigenResult solve_generalized_hermitian_rmm_trial(
    const HermitianOperator& h,const HermitianOperator& s,std::size_t roots,
    std::size_t max_iterations,double tolerance,const std::vector<C>& initial) {
  PerformanceTimer timer("rmm_trial_total");
  const auto n=h.dimension;
  if(!h.scf_trial_update||h.band_group_local||tolerance<=1e-9||!std::isfinite(tolerance)||
      !n||!roots||roots>n||!max_iterations||initial.size()!=n*roots||s.dimension!=n||
      !h.apply||!s.apply||h.kinetic.size()!=n||h.diagonal.size()!=n||s.diagonal.size()!=n)
    throw std::invalid_argument("RMM trial requires a coarse explicit SCF warm request with physical kinetic energies");
  for(double value:h.kinetic)if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid RMM kinetic diagonal");
  for(std::size_t i=0;i<n;++i)
    if(!std::isfinite(h.diagonal[i])||!std::isfinite(s.diagonal[i])||s.diagonal[i]<=0)
      throw std::invalid_argument("invalid RMM H/S diagonals");
  ComplexGeneralizedEigenResult result;result.dimension=n;
  std::size_t used=0;
  const auto fallback=[&]() {
    HermitianOperator original=h;original.scf_trial_update=false;original.band_group_local=true;
    original.subspace_capacity=0;original.local_seed_only=false;
    const auto guess=result.eigenvectors.empty()?initial:result.eigenvectors;
    record_performance("rmm_original_guarded_fallback",0);
    auto out=solve_generalized_hermitian_davidson(original,s,roots,max_iterations-used,tolerance,guess);
    out.iterations+=used;return out;
  };
  // Keep at least half the unchanged 256 budget for original protected
  // global search. Each RMM update applies only NBANDS new directions.
  const auto limit=std::min<std::size_t>(16,max_iterations/2);
  if(!limit)return fallback();
  Images x=apply(h,s,Eigen::Map<const M>(initial.data(),n,roots),true),previous;
  if(!full_rank(x))return fallback();
  for(std::size_t step=0;step<limit;++step) {
    ++used;
    {PerformanceTimer rotation_timer("rmm_nbands_global_rotation");
      const M u=rotation(product(x.x,x.h,true),product(x.x,x.s,true));rotate(x,u);
      if(previous.x.size())rotate(previous,u);}
    M r;
    const double maximum=residual(x,result.eigenvalues,result.residual_norms,&r);
    if(maximum<=tolerance) {
      // Fresh FP64 original operators, independent of all reused images.
      x=apply(h,s,std::move(x.x),true);
      const double fresh=residual(x,result.eigenvalues,result.residual_norms,&r);
      const double gram_error=(product(x.x,x.s,true)-M::Identity(roots,roots)).cwiseAbs().maxCoeff();
      if(fresh<=tolerance&&gram_error<=1e-10) {
        result.eigenvectors.assign(x.x.data(),x.x.data()+x.x.size());
        result.iterations=used;result.converged=false;result.trial_manifold=true;
        record_performance("rmm_residual_checked_nonconverged_trial",0);
        return result;
      }
    }
    M direction(n,roots);
    for(std::size_t j=0;j<roots;++j) {
      double kinetic=0;for(std::size_t i=0;i<n;++i)kinetic+=std::norm(x.x(i,j))*h.kinetic[i];
      for(std::size_t i=0;i<n;++i)direction(i,j)=-kinetic_preconditioner(h.kinetic[i],kinetic)*r(i,j);
    }
    Images d=apply(h,s,std::move(direction),tolerance<1e-6);
    project(d,x);normalize(d);
    if(previous.x.size()){project(previous,x);normalize(previous);}
    Images next=x;
    for(std::size_t j=0;j<roots;++j) {
      const double energy=result.eigenvalues[j];
      Eigen::Matrix<C,Eigen::Dynamic,2> corrections(n,2);
      corrections.col(0)=d.h.col(j)-energy*d.s.col(j);
      if(previous.x.size())corrections.col(1)=previous.h.col(j)-energy*previous.s.col(j);
      else corrections.col(1).setZero();
      Eigen::Matrix2cd gram=corrections.adjoint()*corrections;
      const double scale=gram.diagonal().real().maxCoeff();
      if(!(scale>0)||!std::isfinite(scale))continue;
      gram/=scale;gram.diagonal().array()+=1e-14;
      const Eigen::Vector2cd rhs=-(corrections.adjoint()*r.col(j))/scale;
      const Eigen::Vector2cd coeff=gram.ldlt().solve(rhs);
      if(!coeff.allFinite()||coeff.cwiseAbs().maxCoeff()>1e6)continue;
      next.x.col(j)+=coeff[0]*d.x.col(j);next.h.col(j)+=coeff[0]*d.h.col(j);next.s.col(j)+=coeff[0]*d.s.col(j);
      if(previous.x.size()){
        next.x.col(j)+=coeff[1]*previous.x.col(j);next.h.col(j)+=coeff[1]*previous.h.col(j);next.s.col(j)+=coeff[1]*previous.s.col(j);
      }
    }
    normalize(next);
    if(!full_rank(next))break;
    x=std::move(next);previous=std::move(d);
    result.eigenvectors.assign(x.x.data(),x.x.data()+x.x.size());
    record_performance("rmm_small_band_residual_minimizations",0,roots);
  }
  return fallback();
}
}
