#pragma once
#include "joint_multisecant.hpp"
#include <cstdlib>
#include <string>
#include <cerrno>

namespace sdft {
// Common inverse-response primitives and actual-step positivity guard from
// the main SCF kernel. This class controls trajectory, never the PAW model.
class JointPawMixer {
  using C=std::complex<double>;
  JointSecants::History residuals_,outputs_,site_residuals_,site_outputs_;
  std::vector<std::size_t> dimensions_;
  double mixing_,smooth_weight_,site_weight_,prior_feedback_=-1.,trust_radius_;
  bool multisecant_,full_diis_;
  // The image is evaluated by the next SCF iteration. Rejected trials must
  // never become secants or the base of a newly extrapolated proposal.
  std::vector<C> accepted_smooth_,accepted_sites_,accepted_residual_,accepted_site_residual_;
  std::vector<C> trial_smooth_direction_,trial_site_direction_;
  double accepted_feedback_=-1.,trial_fraction_=1.;
  std::vector<double> accepted_feedback_window_;
  bool trial_pending_=false,linear_trial_=false;
  static double trust_radius() {
    const char* text=std::getenv("SDFT_MULTISECANT_TRUST_RADIUS");
    if(!text)return .1;
    errno=0;char* end=nullptr;const double value=std::strtod(text,&end);
    if(errno||end==text||*end||!std::isfinite(value)||value<=0||value>1)
      throw std::invalid_argument("SDFT_MULTISECANT_TRUST_RADIUS must be in (0,1]");
    return value;
  }
  static bool contraction() {
    const char* p=std::getenv("SDFT_MULTISECANT_BACKTRACK");
    if(p&&std::string(p)!="0"&&std::string(p)!="1")
      throw std::invalid_argument("SDFT_MULTISECANT_BACKTRACK must be 0 or 1");
    return !p||std::string(p)=="1";
  }
  static std::string method() {
    const char* p=std::getenv("SDFT_MIXER");const std::string value=p?p:"multisecant";
    if(value!="multisecant"&&value!="full_diis"&&value!="damped_diis"&&value!="linear")
      throw std::invalid_argument("SDFT_MIXER must be multisecant, full_diis, damped_diis or linear");
    return value;
  }
 public:
  int rank=0;double step_scale=1.,proposal_scale=1.,positivity_scale=1.;bool positivity_fallback=false,history_restarted=false,trial_rejected=false;double feedback_norm=0.,accepted_feedback=0.,feedback_bound=0.;
  explicit JointPawMixer(double mixing,std::size_t keys,double mean_density,
      std::vector<std::size_t> dimensions):dimensions_(std::move(dimensions)),mixing_(mixing),
      smooth_weight_(1./(std::max(std::size_t(1),keys)*mean_density*mean_density)),
      site_weight_(1.),trust_radius_(trust_radius()),multisecant_(method()=="multisecant"),full_diis_(method()=="full_diis") {
    std::size_t total=0;for(auto d:dimensions_)total+=d*d;
    site_weight_=1./std::max(std::size_t(1),total);
  }
  std::string name() const {return method();}
  bool adaptive_enabled() const {return multisecant_&&contraction();}
  std::size_t history_size() const {return residuals_.size();}
  void clear_inexact_history() {
    residuals_.clear();outputs_.clear();site_residuals_.clear();site_outputs_.clear();
    prior_feedback_=-1.;rank=0;
    trial_pending_=false;accepted_feedback_=-1.;accepted_feedback_window_.clear();
  }
  void update(std::vector<C>& smooth,std::vector<C>& sites,
      const std::vector<C>& next,const std::vector<C>& next_sites) {
    if(smooth.size()!=next.size()||sites.size()!=next_sites.size())
      throw std::invalid_argument("coupled PAW mixer dimensions changed");
    std::vector<C> residual(smooth.size()),site_residual(sites.size());
    for(std::size_t i=0;i<smooth.size();++i)residual[i]=next[i]-smooth[i];
    for(std::size_t i=0;i<sites.size();++i)site_residual[i]=next_sites[i]-sites[i];
    // Use the same metric as the inverse-response fit. The maximum
    // component is still the caller's strict final acceptance certificate,
    // but rotating a residual at constant norm must not contract its step.
    double squared_norm=0.;
    for(const auto& z:residual)squared_norm+=smooth_weight_*std::norm(z);
    for(const auto& z:site_residual)squared_norm+=site_weight_*std::norm(z);
    feedback_norm=std::sqrt(squared_norm);
    if(!std::isfinite(feedback_norm))throw std::invalid_argument("nonfinite joint residual metric");
    trial_rejected=false;positivity_fallback=false;history_restarted=false;positivity_scale=1.;
    accepted_feedback=accepted_feedback_;
    // A finite accepted-history envelope permits an intermediate response
    // to rotate or oscillate while new secants are learned. A single tiny
    // residual sample must not permanently pin the anchor near roundoff.
    // Rejected samples never enlarge this envelope; it expires in eight
    // accepted steps, the same finite window used by the inverse fit.
    feedback_bound=accepted_feedback_window_.empty()?accepted_feedback_:
        *std::max_element(accepted_feedback_window_.begin(),accepted_feedback_window_.end());
    if(multisecant_&&contraction()&&trial_pending_&&
        feedback_bound>0&&feedback_norm>1.25*feedback_bound) {
      trial_rejected=true;
      trial_fraction_*=.5;
      // Reuse the actual attempted direction, relative to its accepted
      // anchor. A smaller convex segment preserves an already valid cone.
      // After six unsuccessful halves rebuild from the physical B0 image.
      if(!linear_trial_&&trial_fraction_<1./64.) {
        residuals_.clear();outputs_.clear();site_residuals_.clear();site_outputs_.clear();
        history_restarted=true;rank=0;trial_fraction_=1.;linear_trial_=true;
        for(std::size_t i=0;i<smooth.size();++i)trial_smooth_direction_[i]=mixing_*.125*accepted_residual_[i];
        for(std::size_t i=0;i<sites.size();++i)trial_site_direction_[i]=mixing_*.125*accepted_site_residual_[i];
      }
      if(trial_fraction_<std::ldexp(1.,-30))
        throw std::runtime_error("no measured descent on coupled convex B0 fallback");
      for(std::size_t i=0;i<smooth.size();++i)smooth[i]=accepted_smooth_[i]+trial_fraction_*trial_smooth_direction_[i];
      for(std::size_t i=0;i<sites.size();++i)sites[i]=accepted_sites_[i]+trial_fraction_*trial_site_direction_[i];
      if(!onsite_positive_semidefinite(sites,dimensions_))
        throw std::runtime_error("retracted coupled trial violates accepted PAW cone");
      step_scale=trial_fraction_;proposal_scale=1.;
      return;
    }
    // Only an evaluated, accepted point is appended to the inverse fit.
    accepted_smooth_=smooth;accepted_sites_=sites;
    accepted_residual_=residual;accepted_site_residual_=site_residual;
    accepted_feedback_=feedback_norm;accepted_feedback=feedback_norm;
    accepted_feedback_window_.push_back(feedback_norm);
    if(accepted_feedback_window_.size()>8)accepted_feedback_window_.erase(accepted_feedback_window_.begin());
    history_restarted=prior_feedback_>0&&feedback_norm>3*prior_feedback_;
    if(history_restarted) {
      residuals_.clear();outputs_.clear();site_residuals_.clear();site_outputs_.clear();
    }
    prior_feedback_=feedback_norm;
    residuals_.push_back(residual);outputs_.push_back(next);
    site_residuals_.push_back(site_residual);site_outputs_.push_back(next_sites);
    if(residuals_.size()>8) {
      residuals_.erase(residuals_.begin());outputs_.erase(outputs_.begin());
      site_residuals_.erase(site_residuals_.begin());site_outputs_.erase(site_outputs_.begin());
    }
    auto proposed=smooth,proposed_sites=sites;
    for(std::size_t i=0;i<smooth.size();++i)proposed[i]+=mixing_*residual[i];
    for(std::size_t i=0;i<sites.size();++i)proposed_sites[i]+=mixing_*site_residual[i];
    rank=0;proposal_scale=1.;
    if(multisecant_||method()=="damped_diis"||full_diis_) {
      const auto coefficients=joint_secant_coefficients(residuals_,site_residuals_,smooth_weight_,site_weight_);
      rank=coefficients.rank;
      if(coefficients.usable) {
        // Anderson learns the inverse response only inside its history span.
        // Keep the requested B0 in the complementary, unlearned directions.
        // Unit response there is full DIIS, not a safe multisecant baseline.
        const double damping=full_diis_?1.:mixing_;
        proposed=next;proposed_sites=next_sites;
        if(damping<1.) {
          for(std::size_t i=0;i<proposed.size();++i)proposed[i]-=(1-damping)*residual[i];
          for(std::size_t i=0;i<proposed_sites.size();++i)proposed_sites[i]-=(1-damping)*site_residual[i];
        }
        apply_joint_secants(proposed,outputs_,residuals_,damping,coefficients);
        apply_joint_secants(proposed_sites,site_outputs_,site_residuals_,damping,coefficients);
        if(multisecant_) {
          double state=0.,step=0.;
          for(std::size_t i=0;i<smooth.size();++i) {state+=smooth_weight_*std::norm(smooth[i]);step+=smooth_weight_*std::norm(proposed[i]-smooth[i]);}
          for(std::size_t i=0;i<sites.size();++i) {state+=site_weight_*std::norm(sites[i]);step+=site_weight_*std::norm(proposed_sites[i]-sites[i]);}
          proposal_scale=secant_state_trust_scale(state,step,trust_radius_);
          for(std::size_t i=0;i<smooth.size();++i)proposed[i]=smooth[i]+proposal_scale*(proposed[i]-smooth[i]);
          for(std::size_t i=0;i<sites.size();++i)proposed_sites[i]=sites[i]+proposal_scale*(proposed_sites[i]-sites[i]);
        }
      }
    }
    // Evaluated retraction is the adaptive contraction controller. Applying
    // another lagged residual-feedback scale here would damp an already
    // accepted nonmonotone response a second time and pin it at .125.
    step_scale=1.;
    const auto cone=retract_joint_paw_cone(proposed,proposed_sites,smooth,sites,
        next,next_sites,dimensions_,mixing_);
    positivity_scale=cone.scale;step_scale=cone.scale;positivity_fallback=cone.convex_fallback;
    if(positivity_fallback) {
      residuals_.clear();outputs_.clear();site_residuals_.clear();site_outputs_.clear();
      history_restarted=true;
    }
    smooth=std::move(proposed);sites=std::move(proposed_sites);
    trial_smooth_direction_.resize(smooth.size());trial_site_direction_.resize(sites.size());
    for(std::size_t i=0;i<smooth.size();++i)trial_smooth_direction_[i]=smooth[i]-accepted_smooth_[i];
    for(std::size_t i=0;i<sites.size();++i)trial_site_direction_[i]=sites[i]-accepted_sites_[i];
    trial_fraction_=1.;linear_trial_=false;trial_pending_=multisecant_&&contraction();
  }
};
}
