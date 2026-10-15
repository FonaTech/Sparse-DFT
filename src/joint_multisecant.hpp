#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace sdft {
// A soft SCF mode has R(X) approximately -lambda*(X-X*), lambda << 1.
// Capping an inverse-response step by a multiple of |R| destroys the very
// amplification learned by the secants. Bound displacement by the physical
// state scale instead; positivity and measured residual feedback still apply.
inline double secant_state_trust_scale(double prior_norm_squared,
    double step_norm_squared,double relative_radius=.1) {
  if(!std::isfinite(prior_norm_squared)||prior_norm_squared<0||
     !std::isfinite(step_norm_squared)||step_norm_squared<0||
     !std::isfinite(relative_radius)||relative_radius<=0||relative_radius>1)
    throw std::invalid_argument("invalid secant state trust radius");
  const double bound_squared=relative_radius*relative_radius*std::max(1.,prior_norm_squared);
  return step_norm_squared>bound_squared ? std::sqrt(bound_squared/step_norm_squared) : 1.;
}

// Residual-feedback contraction; no trial SCF or claim of a line-search rollback.
class AdaptiveSecantStep {
  bool enabled_; double prior_=-1., scale_=1., recovery_anchor_=-1.;
  unsigned stable_decreases_=0;
 public:
  explicit AdaptiveSecantStep(bool enabled):enabled_(enabled){}
  double observe(double residual) {
    if(!std::isfinite(residual)||residual<0)throw std::invalid_argument("invalid secant feedback residual");
    if(enabled_&&prior_>0) {
      if(residual>1.25*prior_) {
        scale_=std::max(.125,.5*scale_);stable_decreases_=0;recovery_anchor_=-1.;
      } else if(residual<.7*prior_) {
        scale_=std::min(1.,1.25*scale_);stable_decreases_=0;recovery_anchor_=-1.;
      } else if(residual<=.98*prior_) {
        // With an exact secant proposal, scale=.125 leaves .875 of the
        // error. A per-iteration .7 test alone can never recover the step.
        // Require sustained measured progress, then cautiously grow it.
        if(!stable_decreases_)recovery_anchor_=prior_;
        ++stable_decreases_;
        if(stable_decreases_>=4&&residual<=.8*recovery_anchor_) {
          scale_=std::min(1.,1.25*scale_);stable_decreases_=0;recovery_anchor_=-1.;
        }
      } else {stable_decreases_=0;recovery_anchor_=-1.;}
    }
    prior_=residual;return scale_;
  }
};

// Every on-site output is a positive semidefinite sum of occupied PAW
// amplitudes. An inverse-response extrapolation need not preserve that cone.
// Reject invalid candidates without modifying the output wavefunctions,
// Hamiltonian, occupations or physical convergence gates.
inline bool onsite_positive_semidefinite(const std::vector<std::complex<double>>& values,
    const std::vector<std::size_t>& dimensions,double tolerance=1e-10) {
  std::size_t offset=0;
  for(const auto n:dimensions){
    if(!n||n>1000||offset+n*n>values.size())throw std::invalid_argument("invalid onsite positivity dimensions");
    Eigen::MatrixXcd matrix(n,n);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){
      const auto z=values[offset+i*n+j];
      if(!std::isfinite(z.real())||!std::isfinite(z.imag()))return false;
      matrix(i,j)=z;
    }
    const double scale=std::max(1.,matrix.norm());
    if((matrix-matrix.adjoint()).norm()>tolerance*scale)return false;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig(matrix,Eigen::EigenvaluesOnly);
    if(eig.info()!=Eigen::Success||eig.eigenvalues().minCoeff()<-tolerance*scale)return false;
    offset+=n*n;
  }
  if(offset!=values.size())throw std::invalid_argument("trailing onsite positivity entries");
  return true;
}

// Validate the actual contracted proposal. Testing the full extrapolation
// first can discard a physical damped step and repeatedly erase its history.
// All charge, magnetic and onsite coordinates share the same contraction.
inline bool finalize_collinear_mixing_step(
    std::vector<std::complex<double>>& smooth,
    std::vector<std::complex<double>>& onsite,
    const std::vector<std::complex<double>>& smooth_prior,
    const std::vector<std::complex<double>>& onsite_prior,
    const std::vector<std::complex<double>>& smooth_residual,
    const std::vector<std::complex<double>>& onsite_residual,
    const std::vector<std::size_t>& dimensions,
    double contraction,double linear_mixing) {
  if(smooth.size()!=smooth_prior.size()||smooth.size()!=smooth_residual.size()||
      onsite.size()!=onsite_prior.size()||onsite.size()!=onsite_residual.size()||
      !std::isfinite(contraction)||contraction<=0||contraction>1||
      !std::isfinite(linear_mixing)||linear_mixing<=0||linear_mixing>1)
    throw std::invalid_argument("invalid coupled spin mixing step");
  for(std::size_t q=0;q<smooth.size();++q)
    smooth[q]=smooth_prior[q]+contraction*(smooth[q]-smooth_prior[q]);
  for(std::size_t q=0;q<onsite.size();++q)
    onsite[q]=onsite_prior[q]+contraction*(onsite[q]-onsite_prior[q]);
  if(onsite_positive_semidefinite(onsite,dimensions))return false;
  for(std::size_t q=0;q<smooth.size();++q)
    smooth[q]=smooth_prior[q]+contraction*linear_mixing*smooth_residual[q];
  for(std::size_t q=0;q<onsite.size();++q)
    onsite[q]=onsite_prior[q]+contraction*linear_mixing*onsite_residual[q];
  if(!onsite_positive_semidefinite(onsite,dimensions))
    throw std::runtime_error("spin linear mixture violates onsite density positivity; inspect prior/output state");
  return true;
}

struct JointConeStep {
  double scale=1.;
  bool convex_fallback=false;
};

// A nonpositive full extrapolation need not invalidate its direction or the
// measured secants. First retract the WHOLE coupled displacement along that
// direction. Never project individual eigenvalues, atoms or spin channels.
// Limit the search to useful finite steps: a direction outside the tangent
// cone at a singular prior must fall back to the physical convex image.
inline JointConeStep retract_joint_paw_cone(
    std::vector<std::complex<double>>& smooth,
    std::vector<std::complex<double>>& onsite,
    const std::vector<std::complex<double>>& smooth_prior,
    const std::vector<std::complex<double>>& onsite_prior,
    const std::vector<std::complex<double>>& smooth_output,
    const std::vector<std::complex<double>>& onsite_output,
    const std::vector<std::size_t>& dimensions,double mixing) {
  if(smooth.size()!=smooth_prior.size()||smooth.size()!=smooth_output.size()||
     onsite.size()!=onsite_prior.size()||onsite.size()!=onsite_output.size()||
     !std::isfinite(mixing)||mixing<=0||mixing>1)
    throw std::invalid_argument("invalid joint PAW cone retraction");
  for(const auto* values:std::initializer_list<const std::vector<std::complex<double>>*>{
                         &smooth,&onsite,&smooth_prior,&onsite_prior,
                         &smooth_output,&onsite_output})
    for(const auto& z:*values)
      if(!std::isfinite(z.real())||!std::isfinite(z.imag()))
        throw std::invalid_argument("nonfinite joint PAW cone state");
  const auto direction=smooth,site_direction=onsite;
  double scale=1.;
  for(int trial=0;trial<4;++trial,scale*=.5) {
    for(std::size_t i=0;i<smooth.size();++i)
      smooth[i]=smooth_prior[i]+scale*(direction[i]-smooth_prior[i]);
    for(std::size_t i=0;i<onsite.size();++i)
      onsite[i]=onsite_prior[i]+scale*(site_direction[i]-onsite_prior[i]);
    if(onsite_positive_semidefinite(onsite,dimensions))return {scale,false};
  }
  for(std::size_t i=0;i<smooth.size();++i)
    smooth[i]=smooth_prior[i]+mixing*(smooth_output[i]-smooth_prior[i]);
  for(std::size_t i=0;i<onsite.size();++i)
    onsite[i]=onsite_prior[i]+mixing*(onsite_output[i]-onsite_prior[i]);
  if(!onsite_positive_semidefinite(onsite,dimensions))
    throw std::runtime_error("convex joint PAW image is nonpositive; inspect prior/output");
  return {1.,true};
}

struct JointSecants {
  using History=std::vector<std::vector<std::complex<double>>>;
  bool usable{};
  int rank{};
  std::vector<double> coefficients;
};

// Rank-revealing Anderson inverse-response update in a real density metric.
// A shared real metric couples smooth charge to PAW onsite density. This
// modifies the solver trajectory only, never H/S, occupations or final gates.
inline JointSecants joint_secant_coefficients(
    const JointSecants::History& smooth, const JointSecants::History& onsite,
    double smooth_weight, double onsite_weight) {
  if(smooth.size()<2) return {};
  if(!onsite.empty() && onsite.size()!=smooth.size())
    throw std::invalid_argument("joint secant history counts disagree");
  if(!(smooth_weight>0) || !std::isfinite(smooth_weight) ||
     !std::isfinite(onsite_weight) || onsite_weight<0)
    throw std::invalid_argument("invalid joint secant metric");
  const int count=static_cast<int>(smooth.size())-1;
  const auto check=[](const JointSecants::History& h) {
    for(const auto& row:h) {
      if(row.size()!=h.front().size()) throw std::invalid_argument("joint secant vector sizes disagree");
      for(const auto& z:row) if(!std::isfinite(z.real())||!std::isfinite(z.imag()))
        throw std::invalid_argument("nonfinite joint secant residual");
    }
  };
  check(smooth);if(!onsite.empty())check(onsite);
  // Fit the realified rectangular secants directly. Normal equations
  // square the condition number: filtering Gram eigenvalues at 1e-8
  // discarded legitimate response singular values as large as 1e-4.
  // That removes weak orbital modes even when their response is resolved.
  const std::size_t smooth_size=smooth.front().size();
  const std::size_t onsite_size=onsite.empty()?0:onsite.front().size();
  const std::size_t rows=2*(smooth_size+onsite_size);
  if(!rows)return {};
  Eigen::MatrixXd directions(rows,count);
  Eigen::VectorXd rhs(rows);
  const auto accumulate=[&](const JointSecants::History& h,double weight,std::size_t offset) {
    if(h.empty())return;
    const double factor=std::sqrt(weight);
    for(std::size_t q=0;q<h.front().size();++q) {
      rhs[offset+2*q]=factor*h.back()[q].real();
      rhs[offset+2*q+1]=factor*h.back()[q].imag();
      for(int i=0;i<count;++i) {
        const auto delta=factor*(h[i+1][q]-h[i][q]);
        directions(offset+2*q,i)=delta.real();
        directions(offset+2*q+1,i)=delta.imag();
      }
    }
  };
  accumulate(smooth,smooth_weight,0);accumulate(onsite,onsite_weight,2*smooth_size);
  double scale=0.;
  Eigen::VectorXd norms(count);
  for(int i=0;i<count;++i){norms[i]=directions.col(i).norm();scale=std::max(scale,norms[i]);}
  if(!(scale>0)||!std::isfinite(scale))return {};
  for(int i=0;i<count;++i){norms[i]=std::max(norms[i],scale*1e-10);directions.col(i)/=norms[i];}
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(directions,Eigen::ComputeThinU|Eigen::ComputeThinV);
  if(svd.info()!=Eigen::Success||!svd.singularValues().size())return {};
  const double largest=svd.singularValues()[0];
  if(!(largest>0)||!std::isfinite(largest))return {};
  Eigen::VectorXd projected=svd.matrixU().adjoint()*rhs;
  JointSecants result;result.coefficients.resize(count);
  for(Eigen::Index i=0;i<projected.size();++i) {
    if(svd.singularValues()[i]>largest*1e-8){projected[i]/=svd.singularValues()[i];++result.rank;}
    else projected[i]=0.;
  }
  const Eigen::VectorXd solved=svd.matrixV()*projected;
  result.usable=result.rank>0;
  for(int i=0;i<count;++i) {
    result.coefficients[i]=solved[i]/norms[i];
    if(!std::isfinite(result.coefficients[i]))result.usable=false;
  }
  return result;
}

inline void apply_joint_secants(std::vector<std::complex<double>>& mixed,
    const JointSecants::History& outputs, const JointSecants::History& residuals,
    double damping,const JointSecants& secants) {
  if(!secants.usable)return;
  if(!std::isfinite(damping)||damping<=0||damping>1)
    throw std::invalid_argument("invalid joint secant damping");
  if(outputs.size()!=residuals.size() || outputs.size()!=secants.coefficients.size()+1)
    throw std::invalid_argument("joint secant image history counts disagree");
  for(std::size_t i=0;i<secants.coefficients.size();++i) {
    if(outputs[i].size()!=mixed.size()||outputs[i+1].size()!=mixed.size()||
       residuals[i].size()!=mixed.size()||residuals[i+1].size()!=mixed.size())
      throw std::invalid_argument("joint secant image dimensions disagree");
    for(std::size_t q=0;q<mixed.size();++q)
      mixed[q]-=secants.coefficients[i]*((outputs[i+1][q]-outputs[i][q])-
          (1-damping)*(residuals[i+1][q]-residuals[i][q]));
  }
}
} // namespace sdft
