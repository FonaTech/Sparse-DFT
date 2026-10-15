#pragma once
#include "pbe_xc.hpp"
#include <cmath>
namespace sdft {
// FP64 forward density derivative, with a cancellation-free analytic sigma
// response. Constants and functional match pbe_xc.cpp; no TF32 conversion.
struct CudaPbeDual {double v,n;};
__device__ inline CudaPbeDual operator+(CudaPbeDual a,CudaPbeDual b){return {a.v+b.v,a.n+b.n};}
__device__ inline CudaPbeDual operator-(CudaPbeDual a,CudaPbeDual b){return {a.v-b.v,a.n-b.n};}
__device__ inline CudaPbeDual operator*(CudaPbeDual a,CudaPbeDual b){return {a.v*b.v,a.n*b.v+a.v*b.n};}
__device__ inline CudaPbeDual operator/(CudaPbeDual a,CudaPbeDual b){const double v=a.v/b.v;return {v,(a.n-v*b.n)/b.v};}
__device__ inline CudaPbeDual operator+(CudaPbeDual a,double b){return {a.v+b,a.n};}
__device__ inline CudaPbeDual operator-(CudaPbeDual a,double b){return {a.v-b,a.n};}
__device__ inline CudaPbeDual operator*(CudaPbeDual a,double b){return {a.v*b,a.n*b};}
__device__ inline CudaPbeDual operator/(CudaPbeDual a,double b){return {a.v/b,a.n/b};}
__device__ inline CudaPbeDual operator/(double a,CudaPbeDual b){return CudaPbeDual{a,0}/b;}
__device__ inline CudaPbeDual cuda_pbe_log(CudaPbeDual a){return {::log(a.v),a.n/a.v};}
__device__ inline CudaPbeDual cuda_pbe_log1p(CudaPbeDual a){return {::log1p(a.v),a.n/(1.+a.v)};}
__device__ inline CudaPbeDual cuda_pbe_exp(CudaPbeDual a){const auto v=::exp(a.v);return {v,v*a.n};}
__device__ inline CudaPbeDual cuda_pbe_expm1(CudaPbeDual a){const auto v=::expm1(a.v);return {v,(1.+v)*a.n};}
__device__ inline CudaPbeDual cuda_pbe_root(CudaPbeDual a,double p){const auto v=::pow(a.v,p);return {v,p*v/a.v*a.n};}
__global__ void cuda_pbe_points(const double* density,const double* gradient,std::size_t count,PbeXcPoint* output){
  const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  constexpr double pi=3.1415926535897932384626433832795,mu=.2195149727645171,kappa=.804;
  constexpr double beta=.06672455060314922,gamma=.031090690869654895,A=.0310907;
  CudaPbeDual n{density[i],density[i]},sigma{gradient[i]*gradient[i],0};
  const auto root=cuda_pbe_root(n,1./3.),rs=::cbrt(3./(4*pi))/root;
  const auto kf=root*::cbrt(3*pi*pi),s2=sigma/(kf*kf*n*n*4.);
  const auto exchange=n*(root*(-.75*::cbrt(3./pi)))*(CudaPbeDual{1+kappa,0}-kappa/(s2*(mu/kappa)+1.));
  const auto rroot=cuda_pbe_root(rs,.5);
  const auto poly=rroot*7.5957+rs*3.5876+rs*rroot*1.6382+rs*rs*.49294;
  const auto eps=(rs*.21370+1.)*cuda_pbe_log1p(1./(poly*(2*A)))*(-2*A);
  const auto ks2=kf*(4./pi),t2=sigma/(ks2*n*n*4.);
  const auto factor=(beta/gamma)/cuda_pbe_expm1(eps*(-1./gamma));
  const auto at=factor*t2;
  CudaPbeDual ratio;
  if(at.v>1.){const auto y=1./at;ratio=((beta/gamma)/factor)*(y+1.)/(y+y*y+1.);}
  else ratio=(t2*(beta/gamma)*(at+1.))/(at+at*at+1.);
  const auto h=cuda_pbe_log1p(ratio)*gamma;
  const auto correlation=n*(eps+h),total=exchange+correlation;
  const double u=mu/kappa*s2.v,x=at.v,bt=beta/gamma*t2.v;
  double difference;
  if(u>.25||x>.25){
    double cf;
    if(x>1.){const double inverse=1./x,den=1.+inverse+inverse*inverse;
      cf=(2.+inverse)*inverse*inverse*inverse/(den*(den+(beta/gamma)/factor.v*(1.+inverse)));}
    else{const double den=1.+x+x*x;cf=(1.+2*x)/(den*(den+bt*(1.+x)));}
    const double inverse=1./(1.+u);difference=cf-inverse*inverse;
  }else{const double den=1.+x+x*x;
    difference=u*(2.+u)/((1.+u)*(1.+u))-(x*x*(3.+x*(2.+x))+bt*(1.+x)*den)/(den*(den+bt*(1.+x)));}
  auto& point=output[i];point.energy_density=total.v;point.energy_per_particle=total.v/n.v;
  point.exchange_energy_density=exchange.v;point.correlation_energy_density=correlation.v;
  point.exchange_potential=exchange.n/n.v;point.correlation_potential=correlation.n/n.v;
  point.gradient_squared_coefficient=beta/(2*ks2.v*n.v)*difference;
}
}
