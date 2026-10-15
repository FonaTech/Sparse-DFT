#pragma once
#include "joint_multisecant.hpp"

namespace sdft {
// Kerker inverse response A*G^2/(G^2+B^2) (Kerker, PRB 23, 3082 (1981)),
// floored at amin, with reciprocal vectors in inverse Å.
// G=0 is assigned the smallest nonzero-G response by the caller; it is not
// a constraint on total magnetization. Only the charge integral is fixed.
inline double kerker_response(double g2,double amix,double bmix,double amin) {
  if(!std::isfinite(g2)||g2<0||!std::isfinite(amix)||amix<=0||
     !std::isfinite(bmix)||bmix<0||!std::isfinite(amin)||amin<=0||amin>amix)
    throw std::invalid_argument("invalid Kerker inverse response");
  return bmix==0?amix:std::max(amin,amix*g2/(g2+bmix*bmix));
}

// Spin-up then spin-down -> charge then magnetization. Both atom channels
// have exactly the same packing, including heterogeneous projector sizes.
inline std::vector<std::complex<double>> collinear_charge_magnetization(
    const std::vector<std::complex<double>>& spin) {
  if(spin.size()%2)throw std::invalid_argument("odd collinear onsite size");
  auto result=spin;const auto n=spin.size()/2;
  for(std::size_t q=0;q<n;++q){result[q]=spin[q]+spin[n+q];result[n+q]=spin[q]-spin[n+q];}
  return result;
}
inline std::vector<std::complex<double>> collinear_spin_blocks(
    const std::vector<std::complex<double>>& charge_magnetization) {
  if(charge_magnetization.size()%2)throw std::invalid_argument("odd collinear onsite size");
  auto result=charge_magnetization;const auto n=result.size()/2;
  for(std::size_t q=0;q<n;++q){result[q]=.5*(charge_magnetization[q]+charge_magnetization[n+q]);
    result[n+q]=.5*(charge_magnetization[q]-charge_magnetization[n+q]);}
  return result;
}

// Anderson update with a diagonal initial inverse response B0:
// X+B0 R - sum c_i (delta X_i+B0 delta R_i). The coefficients may use a
// separate physical metric. B0 belongs in every history image, not merely
// in the fallback, and it never changes the self-consistent fixed points.
inline std::vector<std::complex<double>> preconditioned_secant_image(
    const std::vector<std::complex<double>>& prior,
    const JointSecants::History& outputs,const JointSecants::History& residuals,
    const std::vector<double>& response,const JointSecants& secants) {
  if(outputs.empty()||outputs.size()!=residuals.size()||prior.size()!=response.size())
    throw std::invalid_argument("invalid preconditioned secant history");
  const auto n=prior.size();auto image=prior;
  for(std::size_t q=0;q<n;++q){
    if(outputs.back().size()!=n||residuals.back().size()!=n||
       !std::isfinite(response[q])||response[q]<=0)
      throw std::invalid_argument("invalid preconditioned secant response");
    image[q]+=response[q]*residuals.back()[q];
  }
  if(!secants.usable)return image;
  if(secants.coefficients.size()+1!=outputs.size())
    throw std::invalid_argument("preconditioned secant count mismatch");
  for(std::size_t i=0;i<secants.coefficients.size();++i){
    if(outputs[i].size()!=n||outputs[i+1].size()!=n||residuals[i].size()!=n||residuals[i+1].size()!=n)
      throw std::invalid_argument("preconditioned secant dimension mismatch");
    for(std::size_t q=0;q<n;++q)
      image[q]-=secants.coefficients[i]*((outputs[i+1][q]-outputs[i][q])-
          (1-response[q])*(residuals[i+1][q]-residuals[i][q]));
  }
  return image;
}

// Check the contracted coupled state. If extrapolation leaves the physical
// cone, use B0 R and reduce that entire charge/spin/onsite step until both
// spin matrices are positive. Never clip individual atoms or spin channels.
inline bool finalize_preconditioned_spin_step(
    std::vector<std::complex<double>>& smooth,std::vector<std::complex<double>>& onsite,
    const std::vector<std::complex<double>>& smooth_prior,
    const std::vector<std::complex<double>>& onsite_prior,
    const std::vector<std::complex<double>>& linear_smooth,
    const std::vector<std::complex<double>>& linear_onsite,
    const std::vector<std::size_t>& dimensions,double contraction,double& positivity_scale,
    const std::vector<std::complex<double>>& output_smooth={},
    const std::vector<std::complex<double>>& output_onsite={},double convex_mixing=.2,
    bool* convex_fallback=nullptr) {
  if(smooth.size()!=smooth_prior.size()||smooth.size()!=linear_smooth.size()||
     onsite.size()!=onsite_prior.size()||onsite.size()!=linear_onsite.size()||
     !std::isfinite(contraction)||contraction<=0||contraction>1)
    throw std::invalid_argument("invalid preconditioned positivity step");
  positivity_scale=1.;if(convex_fallback)*convex_fallback=false;
  for(std::size_t q=0;q<smooth.size();++q)smooth[q]=smooth_prior[q]+contraction*(smooth[q]-smooth_prior[q]);
  for(std::size_t q=0;q<onsite.size();++q)onsite[q]=onsite_prior[q]+contraction*(onsite[q]-onsite_prior[q]);
  if(onsite_positive_semidefinite(onsite,dimensions))return false;
  // A tiny positive step on a cone boundary can satisfy a roundoff
  // tolerance while producing no useful SCF progress. Try the same finite
  // contraction range as the adaptive controller, then change direction.
  for(int trial=0;trial<4;++trial){
    const double step=contraction*positivity_scale;
    for(std::size_t q=0;q<smooth.size();++q)smooth[q]=smooth_prior[q]+step*(linear_smooth[q]-smooth_prior[q]);
    for(std::size_t q=0;q<onsite.size();++q)onsite[q]=onsite_prior[q]+step*(linear_onsite[q]-onsite_prior[q]);
    if(onsite_positive_semidefinite(onsite,dimensions))return true;
    positivity_scale*=.5;
  }
  // A singular prior may have a nullspace in both spin channels. Distinct
  // charge/magnetic B0 can point outside that cone at every positive scale.
  // Its direction must be replaced, not continually halved or clipped.
  // The original occupied-subspace output is positive, so the same scalar
  // convex interpolation in every smooth/onsite spin coordinate is safe.
  if(output_smooth.size()!=smooth.size()||output_onsite.size()!=onsite.size()||
     !std::isfinite(convex_mixing)||convex_mixing<=0||convex_mixing>1)
    throw std::runtime_error("preconditioned spin step has no positive coupled density; inspect prior/output state");
  positivity_scale=1.;
  for(std::size_t q=0;q<smooth.size();++q)
    smooth[q]=smooth_prior[q]+contraction*convex_mixing*(output_smooth[q]-smooth_prior[q]);
  for(std::size_t q=0;q<onsite.size();++q)
    onsite[q]=onsite_prior[q]+contraction*convex_mixing*(output_onsite[q]-onsite_prior[q]);
  if(!onsite_positive_semidefinite(onsite,dimensions))
    throw std::runtime_error("convex collinear mixture is nonpositive; inspect prior/output state");
  if(convex_fallback)*convex_fallback=true;
  return true;
}
} // namespace sdft
