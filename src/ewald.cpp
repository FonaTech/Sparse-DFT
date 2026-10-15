#include "ewald.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace sdft {
namespace {
constexpr double pi=3.1415926535897932384626433832795;
Vec3 mul(Vec3 v,double t){return {v.x*t,v.y*t,v.z*t};}
Vec3 plus(Vec3 x,Vec3 y){return {x.x+y.x,x.y+y.y,x.z+y.z};}
Vec3 cart(const std::array<Vec3,3>& a,Vec3 f){return plus(plus(mul(a[0],f.x),mul(a[1],f.y)),mul(a[2],f.z));}
double length(Vec3 v){return std::sqrt(dot(v,v));}
}
EwaldResult evaluate_ewald(const std::array<Vec3,3>& a,const std::vector<Vec3>& f,
    const std::vector<double>& z,double eta,double tol){
  const double det=determinant(a),vol=std::abs(det);
  if(!std::isfinite(vol)||vol<1e-12||f.empty()||f.size()!=z.size()||
      !std::isfinite(eta)||eta<0||!std::isfinite(tol)||tol<=0||tol>=1)
    throw std::invalid_argument("invalid Ewald lattice, charges, eta or tolerance");
  for(std::size_t i=0;i<f.size();++i)
    if(!std::isfinite(f[i].x)||!std::isfinite(f[i].y)||!std::isfinite(f[i].z)||!std::isfinite(z[i]))
      throw std::invalid_argument("nonfinite Ewald ion");
  if(eta==0)eta=std::sqrt(pi)/std::cbrt(vol);
  const std::array<Vec3,3> b{mul(cross(a[1],a[2]),2*pi/det),mul(cross(a[2],a[0]),2*pi/det),mul(cross(a[0],a[1]),2*pi/det)};
  // Conservative tails for both sums. Bounds use reciprocal norms, so skew
  // cells do not lose translations even when their basis vectors are long.
  const double arg=std::sqrt(-std::log(tol))+1.0;
  const double rcut=arg/eta,gcut=2*eta*arg;
  int nr[3],ng[3];
  for(int axis=0;axis<3;++axis){
    nr[axis]=static_cast<int>(std::ceil(rcut*length(b[axis])/(2*pi)+.5));
    ng[axis]=static_cast<int>(std::ceil(gcut*length(a[axis])/(2*pi)));
  }
  EwaldResult out;out.forces.resize(f.size());out.eta=eta;out.tail_tolerance=tol;
  double derivative[3][3]{};
  const auto outer=[&](Vec3 v,double t){const double x[]{v.x,v.y,v.z};for(int i=0;i<3;++i)for(int j=0;j<3;++j)derivative[i][j]+=t*x[i]*x[j];};
  double sumz=0,sumz2=0;
  for(double q:z){sumz+=q;sumz2+=q*q;}
  for(std::size_t i=0;i<f.size();++i)for(std::size_t j=i;j<f.size();++j){
    Vec3 df{f[i].x-f[j].x,f[i].y-f[j].y,f[i].z-f[j].z};
    df.x-=std::floor(df.x+.5);df.y-=std::floor(df.y+.5);df.z-=std::floor(df.z+.5);
    const double count=i==j?.5:1.0,zz=z[i]*z[j];
    for(int h=-nr[0];h<=nr[0];++h)for(int k=-nr[1];k<=nr[1];++k)for(int l=-nr[2];l<=nr[2];++l){
      if(i==j&&h==0&&k==0&&l==0)continue;
      Vec3 r=cart(a,{df.x+h,df.y+k,df.z+l});const double d=length(r);
      if(d>=rcut)continue;
      if(d<1e-10)throw std::invalid_argument("coincident periodic ions in Ewald sum");
      const double er=std::erfc(eta*d);
      out.energy+=count*zz*er/d;
      const double factor=zz*(er/(d*d*d)+2*eta/std::sqrt(pi)*std::exp(-eta*eta*d*d)/(d*d));
      if(i!=j){out.forces[i]=plus(out.forces[i],mul(r,factor));out.forces[j]=plus(out.forces[j],mul(r,-factor));}
      outer(r,-count*factor);
    }
  }
  for(int h=-ng[0];h<=ng[0];++h)for(int k=-ng[1];k<=ng[1];++k)for(int l=-ng[2];l<=ng[2];++l){
    if(h==0&&k==0&&l==0)continue;
    const Vec3 g=cart(b,{double(h),double(k),double(l)});const double g2=dot(g,g);
    if(g2>=gcut*gcut)continue;
    double cs=0,ss=0;
    for(std::size_t i=0;i<f.size();++i){const double phase=2*pi*(h*f[i].x+k*f[i].y+l*f[i].z);cs+=z[i]*std::cos(phase);ss+=z[i]*std::sin(phase);}
    const double pref=2*pi/vol*std::exp(-g2/(4*eta*eta))/g2;
    const double e=pref*(cs*cs+ss*ss);out.energy+=e;
    for(std::size_t i=0;i<f.size();++i){const double phase=2*pi*(h*f[i].x+k*f[i].y+l*f[i].z);out.forces[i]=plus(out.forces[i],mul(g,2*pref*z[i]*(cs*std::sin(phase)-ss*std::cos(phase))));}
    outer(g,e*(2/g2+1/(2*eta*eta)));
    for(int axis=0;axis<3;++axis)derivative[axis][axis]-=e;
  }
  const double background=-pi*sumz*sumz/(2*eta*eta*vol);
  out.energy+=background-eta/std::sqrt(pi)*sumz2;
  for(int i=0;i<3;++i){derivative[i][i]-=background;out.stress[i]={derivative[i][0]/vol,derivative[i][1]/vol,derivative[i][2]/vol};}
  return out;
}
}
