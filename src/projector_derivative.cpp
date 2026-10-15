#include "projector_derivative.hpp"
#include "compute_backend.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sdft {
PawDerivativeResponse paw_derivative_response(std::size_t p,std::size_t n,std::size_t bands,
    const Complex64* beta,const Complex64* coeff,const Complex64* d,const Complex64* q,
    const double* eigenvalues,const double* occupations) {
  PerformanceTimer timer("analytic_projector_band_contraction");
  if(!p||!n||!bands||!beta||!coeff||!d||!q||!eigenvalues||!occupations||
     p>std::numeric_limits<std::size_t>::max()/std::max(n,bands))
    throw std::invalid_argument("invalid PAW derivative contraction dimensions");
  std::vector<Complex64> projected(p*bands),response(p*bands),overlap(p*bands);
  complex_gemm(MatrixOp::Normal,MatrixOp::Normal,p,bands,n,beta,p,coeff,n,projected.data(),p,true);
  complex_gemm(MatrixOp::Normal,MatrixOp::Normal,p,bands,p,d,p,projected.data(),p,response.data(),p,true);
  complex_gemm(MatrixOp::Normal,MatrixOp::Normal,p,bands,p,q,p,projected.data(),p,overlap.data(),p,true);
  for(std::size_t band=0;band<bands;++band) {
    if(!std::isfinite(eigenvalues[band])||!std::isfinite(occupations[band])||occupations[band]<0)
      throw std::invalid_argument("invalid PAW derivative eigenvalue/occupation");
    for(std::size_t row=0;row<p;++row)response[row+band*p]=
        occupations[band]*(response[row+band*p]-eigenvalues[band]*overlap[row+band*p]);
  }
  PawDerivativeResponse result{p,n,std::vector<Complex64>(p*n)};
  // W_pG=sum_b f_b [(D-epsilon_b Q) B psi_b]_p conj(psi_Gb).
  // The expensive band contraction is shared by all position/strain axes.
  complex_gemm(MatrixOp::Normal,MatrixOp::Adjoint,p,n,bands,response.data(),p,
      coeff,n,result.sensitivity.data(),p,true);
  record_performance("analytic_projector_response_columns",0,bands);
  return result;
}
double paw_derivative_contract(const PawDerivativeResponse& state,const Complex64* derivative) {
  PerformanceTimer timer("analytic_projector_derivative_contract");
  if(!derivative||!state.projectors||!state.plane_waves||
     state.projectors>std::numeric_limits<std::size_t>::max()/state.plane_waves||
     state.sensitivity.size()!=state.projectors*state.plane_waves)
    throw std::invalid_argument("invalid PAW derivative response");
  long double sum=0;
  for(std::size_t i=0;i<state.sensitivity.size();++i) {
    const auto a=derivative[i],b=state.sensitivity[i];
    sum+=static_cast<long double>(a.real())*b.real()+static_cast<long double>(a.imag())*b.imag();
  }
  const double result=2*double(sum);
  if(!std::isfinite(result))throw std::runtime_error("nonfinite PAW derivative contraction");
  return result;
}
namespace {
void validate_geometry_response(const PawDerivativeResponse& state,
    const Complex64* beta,const std::vector<Vec3>& wavevectors) {
  if(!beta||!state.projectors||!state.plane_waves||
     state.projectors>std::numeric_limits<std::size_t>::max()/state.plane_waves||
     state.sensitivity.size()!=state.projectors*state.plane_waves||
     wavevectors.size()!=state.plane_waves)
    throw std::invalid_argument("invalid PAW geometry derivative response");
  for(const auto& q:wavevectors)
    if(!std::isfinite(q.x)||!std::isfinite(q.y)||!std::isfinite(q.z))
      throw std::invalid_argument("nonfinite PAW derivative wavevector");
}
double checked_response(long double sum) {
  const double result=2*double(sum);
  if(!std::isfinite(result))throw std::runtime_error("nonfinite PAW geometry derivative contraction");
  return result;
}
}
Vec3 paw_derivative_position_contract(const PawDerivativeResponse& state,
    const Complex64* beta,const std::vector<Vec3>& wavevectors) {
  PerformanceTimer timer("analytic_projector_position_fused");
  validate_geometry_response(state,beta,wavevectors);
  std::array<long double,3> sums{};
  for(std::size_t g=0;g<state.plane_waves;++g) {
    const double q[]{wavevectors[g].x,wavevectors[g].y,wavevectors[g].z};
    for(std::size_t p=0;p<state.projectors;++p) {
      const auto i=p+g*state.projectors;
      const auto response=state.sensitivity[i];
      for(int axis=0;axis<3;++axis) {
        // Preserve the original double-precision derivative formation before
        // the original long-double accumulation, including complex phases.
        const auto derivative=beta[i]*Complex64(0,q[axis]);
        sums[axis]+=static_cast<long double>(derivative.real())*response.real()+
            static_cast<long double>(derivative.imag())*response.imag();
      }
    }
  }
  return {checked_response(sums[0]),checked_response(sums[1]),checked_response(sums[2])};
}
std::array<Vec3,3> paw_derivative_strain_contract(const PawDerivativeResponse& state,
    const Complex64* beta,const std::array<const Complex64*,3>& gradients,
    const std::vector<Vec3>& wavevectors) {
  PerformanceTimer timer("analytic_projector_strain_fused");
  validate_geometry_response(state,beta,wavevectors);
  for(const auto* gradient:gradients)if(!gradient)
    throw std::invalid_argument("null PAW momentum gradient");
  std::array<std::array<long double,3>,3> sums{};
  for(std::size_t g=0;g<state.plane_waves;++g) {
    const double q[]{wavevectors[g].x,wavevectors[g].y,wavevectors[g].z};
    for(std::size_t p=0;p<state.projectors;++p) {
      const auto i=p+g*state.projectors;
      const auto response=state.sensitivity[i];
      for(int axis=0;axis<3;++axis)for(int component=0;component<3;++component) {
        auto derivative=gradients[axis][i]*(-q[component]);
        if(axis==component)derivative-=.5*beta[i];
        sums[axis][component]+=static_cast<long double>(derivative.real())*response.real()+
            static_cast<long double>(derivative.imag())*response.imag();
      }
    }
  }
  std::array<Vec3,3> result;
  for(int a=0;a<3;++a)result[a]={checked_response(sums[a][0]),
      checked_response(sums[a][1]),checked_response(sums[a][2])};
  return result;
}
}
