#include "pw_core.hpp"
#include "compute_backend.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#ifdef SDFT_HAS_LAPACK
extern "C" void zhegv_(const int*,const char*,const char*,const int*,std::complex<double>*,const int*,
    std::complex<double>*,const int*,double*,std::complex<double>*,const int*,double*,int*);
#endif

namespace sdft {
namespace {
using C=std::complex<double>;using M=Eigen::MatrixXcd;
M product(const M& a,const M& b,bool adjoint=false) {
  M c(adjoint?a.cols():a.rows(),b.cols());
  if(!a.cols()||!b.rows()){c.setZero();return c;}
  complex_gemm(adjoint?MatrixOp::Adjoint:MatrixOp::Normal,MatrixOp::Normal,c.rows(),c.cols(),b.rows(),
      a.data(),a.rows(),b.data(),b.rows(),c.data(),c.rows(),true);
  return c;
}
M apply_original(const HermitianOperator& op,const M& x,bool exact) {
  M y(x.rows(),x.cols());
  if(op.apply_block_into)op.apply_block_into(x.data(),x.cols(),y.data(),exact);
  else if(op.apply_block){std::vector<C> a(x.data(),x.data()+x.size()),b;op.apply_block(a,x.cols(),b,exact);
    if(b.size()!=std::size_t(x.size()))throw std::runtime_error("band-group block dimension mismatch");y=Eigen::Map<const M>(b.data(),x.rows(),x.cols());}
  else for(Eigen::Index j=0;j<x.cols();++j){std::vector<C> a(x.col(j).data(),x.col(j).data()+x.rows()),b;op.apply(a,b);
    if(b.size()!=std::size_t(x.rows()))throw std::runtime_error("band-group scalar dimension mismatch");y.col(j)=Eigen::Map<const Eigen::VectorXcd>(b.data(),x.rows());}
  if(!y.allFinite())throw std::runtime_error("nonfinite band-group operator");return y;
}
bool control(const char* name) {
  const char* setting=std::getenv(name);if(!setting)return false;
  if(std::string(setting)!="0"&&std::string(setting)!="1")throw std::invalid_argument(std::string(name)+" must be 0 or 1");
  return std::string(setting)=="1";
}
// Only NBANDS x NBANDS, independent of the local 5*group_size search space.
M global_rotation(M h,M s) {
#ifdef SDFT_HAS_LAPACK
  const int size=h.rows(),type=1;const char vectors='V',triangle='L';int info=0,work_size=-1;
  std::vector<double> values(size),real_work(std::max(1,3*size-2));C query;
  zhegv_(&type,&vectors,&triangle,&size,h.data(),&size,s.data(),&size,values.data(),&query,&work_size,real_work.data(),&info);
  if(info)throw std::runtime_error("band-group LAPACK workspace query failed");
  work_size=std::max(1,int(query.real()));std::vector<C> work(work_size);
  zhegv_(&type,&vectors,&triangle,&size,h.data(),&size,s.data(),&size,values.data(),work.data(),&work_size,real_work.data(),&info);
  if(info)throw std::runtime_error("band-group LAPACK generalized rotation failed");
  return h;
#else
  Eigen::GeneralizedSelfAdjointEigenSolver<M> solver(h,s);
  if(solver.info()!=Eigen::Success)throw std::runtime_error("band-group global rotation failed");
  return solver.eigenvectors();
#endif
}
}

ComplexGeneralizedEigenResult solve_generalized_hermitian_band_groups(
    const HermitianOperator& h,const HermitianOperator& s,std::size_t roots,
    std::size_t max_iterations,double tolerance,const std::vector<C>& initial) {
  PerformanceTimer timer("band_group_total");
  const std::size_t n=h.dimension;
  if(!n||!roots||roots>n||!max_iterations||s.dimension!=n||!h.apply||!s.apply||
      h.diagonal.size()!=n||s.diagonal.size()!=n||!std::isfinite(tolerance)||tolerance<=0||
      (!initial.empty()&&initial.size()!=n*roots))throw std::invalid_argument("invalid bounded band-group problem");
  if(std::getenv("SDFT_BAND_GROUP_LOCAL_STEPS"))throw std::invalid_argument("fixed local-step experiment failed independent regression; use the validated residual-driven group solver");
  if(const char* value=std::getenv("SDFT_WARM_GUARD_VECTORS")){
    const std::string v(value);std::size_t end=0;
    if(v.empty()||v[0]=='-')throw std::invalid_argument("invalid warm guard count");
    const auto count=std::stoull(v,&end);
    if(end!=v.size()||count<1||count>std::min(n,2*roots))throw std::invalid_argument("warm guard count outside global band range");
  }
  std::size_t group_size=8;
  if(const char* value=std::getenv("SDFT_BAND_GROUP_SIZE")){
    const std::string v(value);std::size_t end=0;
    if(v.empty()||v[0]=='-')throw std::invalid_argument("invalid band group size");
    group_size=std::stoull(v,&end);
    if(end!=v.size()||!group_size||group_size>64)throw std::invalid_argument("SDFT_BAND_GROUP_SIZE must be in [1,64]");
  }
  const bool swept=control("SDFT_BAND_GROUP_SWEEP");
  const bool reuse_images=control("SDFT_BAND_GROUP_IMAGE_REUSE")&&s.pair_identity&&s.pair_identity==h.pair_identity;
  // A short local solve is a search step, never a convergence declaration.
  // Reserve half the existing iteration budget for original global refinement
  // if sweeps do not attain every original generalized residual. No extra
  // iterations or weakened tolerance are hidden in this candidate.
  const std::size_t local_limit=3,groups=(roots+group_size-1)/group_size;
  const std::size_t local_budget=max_iterations/2;
  if(swept&&groups*local_limit>local_budget) {
    HermitianOperator original=h;original.band_group_local=true;
    record_performance("band_group_small_budget_original_solve",0);
    return solve_generalized_hermitian_davidson(original,s,roots,max_iterations,tolerance,initial);
  }
  M x=M::Zero(n,roots);
  if(!initial.empty())x=Eigen::Map<const M>(initial.data(),n,roots);
  M cached_h=M::Zero(n,roots),cached_s=M::Zero(n,roots),cached_p=M::Zero(reuse_images&&h.project_into?h.projector_rows:0,roots);
  ComplexGeneralizedEigenResult out;out.dimension=n;out.converged=false;
  out.eigenvalues.resize(roots);out.residual_norms.resize(roots);
  // Complete sweeps retain global rotations across groups and degeneracies.
  // Every local solve is bounded at 5*group_size; no 8*NBANDS search basis.
  const std::size_t sweep_limit=swept?local_budget/(groups*local_limit):2;
  for(std::size_t sweep=0;sweep<sweep_limit;++sweep){
    for(std::size_t first=0;first<roots;first+=group_size){
      const auto count=std::min(group_size,roots-first);
      HermitianOperator gh=h,gs=s;
      // Global guards have NBANDS columns; local groups have fewer roots.
      // Keep them for global refinement, never forward this array to a local solve.
      gh.initial_search_guards.clear();
      gh.band_group_local=true;gh.subspace_capacity=std::min(n,5*group_size);
      // Constrain the search basis, never shift or modify the operator.
      // Fresh original residuals include the completed-band error as well.
      if(first){
        M locked=x.leftCols(first),sl=reuse_images?M(cached_s.leftCols(first)):apply_original(s,locked,true);
        M gram=product(locked,sl,true);gram=(.5*(gram+gram.adjoint())).eval();
        if(!reuse_images||(gram-M::Identity(first,first)).cwiseAbs().maxCoeff()>1e-12) {
          Eigen::LLT<M> chol(gram);
          if(chol.info()!=Eigen::Success)throw std::runtime_error("nonpositive completed-band overlap");
          const M rotation=chol.matrixU().solve(M::Identity(first,first));
          locked=product(locked,rotation);x.leftCols(first)=locked;
          if(reuse_images) {
            sl=product(sl,rotation);cached_s.leftCols(first)=sl;
            cached_h.leftCols(first)=product(M(cached_h.leftCols(first)),rotation);
            if(cached_p.rows())cached_p.leftCols(first)=product(M(cached_p.leftCols(first)),rotation);
          }
        }
        gh.excluded_vectors.assign(locked.data(),locked.data()+locked.size());
        if(reuse_images) {
          gh.excluded_image_identity=s.pair_identity;
          gh.excluded_overlap_images.assign(sl.data(),sl.data()+sl.size());
          if(cached_p.rows()) {
            const M amplitudes=cached_p.leftCols(first);
            gh.excluded_projector_images.assign(amplitudes.data(),amplitudes.data()+amplitudes.size());
          }
        }
      }
      std::vector<C> seed;
      if(sweep||!initial.empty()){const M block=x.middleCols(first,count);seed.assign(block.data(),block.data()+block.size());}
      const auto result=solve_generalized_hermitian_davidson(gh,gs,count,swept?local_limit:max_iterations,tolerance,seed);
      if(std::getenv("SDFT_SOLVER_TRACE"))std::cerr<<"band_group sweep="<<sweep<<" first="<<first<<" count="<<count
          <<" iterations="<<result.iterations<<" residual="<<*std::max_element(result.residual_norms.begin(),result.residual_norms.end())
          <<" converged="<<result.converged<<'\n';
      out.iterations+=result.iterations;
      x.middleCols(first,count)=Eigen::Map<const M>(result.eigenvectors.data(),n,count);
      if(reuse_images) {
        const M changed=x.middleCols(first,count);
        // A changed local band set has fresh images; subsequent band groups
        // use those images rather than reconstructing B*psi for locked bands.
        if(cached_p.rows()&&h.apply_projected_into&&s.apply_projected_into){
          h.project_into(changed.data(),count,cached_p.data()+first*cached_p.rows(),true);
          h.apply_projected_into(changed.data(),count,cached_p.data()+first*cached_p.rows(),cached_h.data()+first*n,true);
          s.apply_projected_into(changed.data(),count,cached_p.data()+first*cached_p.rows(),cached_s.data()+first*n,true);
        }else if(h.apply_pair_into){
          h.apply_pair_into(changed.data(),count,cached_h.data()+first*n,cached_s.data()+first*n,true);
          if(cached_p.rows())h.project_into(changed.data(),count,cached_p.data()+first*cached_p.rows(),true);
        }else{
          cached_h.middleCols(first,count)=apply_original(h,changed,true);
          cached_s.middleCols(first,count)=apply_original(s,changed,true);
        }
        record_performance("band_group_updated_image_columns",0,count);
      }
      record_performance("band_group_local_solves",0);record_performance("band_group_local_ritz_capacity",0,gh.subspace_capacity);
    }
    // Global NBANDS-only rotation resolves inter-group crossings. It is not
    // a large Davidson expansion. Verify every root with new original H/S.
    M hx=reuse_images?cached_h:apply_original(h,x,true),sx=reuse_images?cached_s:apply_original(s,x,true);
    M small_h=product(x,hx,true),small_s=product(x,sx,true);
    small_h=(.5*(small_h+small_h.adjoint())).eval();small_s=(.5*(small_s+small_s.adjoint())).eval();
    M rotation;
    if(swept) {
      PerformanceTimer global_timer("band_group_global_ritz");rotation=global_rotation(small_h,small_s);
    } else {
      Eigen::GeneralizedSelfAdjointEigenSolver<M> global(small_h,small_s);
      if(global.info()!=Eigen::Success)throw std::runtime_error("band-group global rotation failed");
      rotation=global.eigenvectors();
    }
    x=product(x,rotation);sx=product(sx,rotation);
    if(reuse_images&&cached_p.rows())cached_p=product(cached_p,rotation);
    for(std::size_t j=0;j<roots;++j){const double metric=std::real(x.col(j).dot(sx.col(j)));
      if(!(metric>0))throw std::runtime_error("band-group invalid final metric");
      const double factor=1/std::sqrt(metric);x.col(j)*=factor;
      if(reuse_images&&cached_p.rows())cached_p.col(j)*=factor;
    }
    // The certification uses fresh exact original H/S, never rotated images.
    // Joint application reuses the immediate PAW projection for H and S.
    if(h.apply_pair_into&&h.pair_identity&&h.pair_identity==s.pair_identity){
      h.apply_pair_into(x.data(),roots,hx.data(),sx.data(),true);
    }else{hx=apply_original(h,x,true);sx=apply_original(s,x,true);}
    if(reuse_images){cached_h=hx;cached_s=sx;}
    // Only this new original global H/S residual can certify the spectrum.
    out.converged=true;
    for(std::size_t j=0;j<roots;++j){
      long double numerator=0,denominator=0,square=0;
      for(std::size_t i=0;i<n;++i){numerator+=static_cast<long double>(x(i,j).real())*hx(i,j).real()+static_cast<long double>(x(i,j).imag())*hx(i,j).imag();
        denominator+=static_cast<long double>(x(i,j).real())*sx(i,j).real()+static_cast<long double>(x(i,j).imag())*sx(i,j).imag();}
      out.eigenvalues[j]=double(numerator/denominator);
      for(std::size_t i=0;i<n;++i){const double re=std::fma(-out.eigenvalues[j],sx(i,j).real(),hx(i,j).real()),im=std::fma(-out.eigenvalues[j],sx(i,j).imag(),hx(i,j).imag());square+=static_cast<long double>(re)*re+static_cast<long double>(im)*im;}
      out.residual_norms[j]=std::sqrt(double(square));out.converged &= out.residual_norms[j]<=tolerance;
    }
    if(out.converged)break;
  }
  if(swept&&!out.converged) {
    HermitianOperator original=h;original.band_group_local=true;
    const std::size_t remaining=max_iterations-out.iterations;
    record_performance("band_group_global_refinement",0);
    const auto search_steps=out.iterations;
    const std::vector<C> seed(x.data(),x.data()+x.size());
    out=solve_generalized_hermitian_davidson(original,s,roots,remaining,tolerance,seed);
    out.iterations+=search_steps;
    record_performance("band_group_global_refinement_iterations",0,out.iterations-search_steps);
    return out;
  }
  out.eigenvectors.assign(x.data(),x.data()+x.size());return out;
}
}
