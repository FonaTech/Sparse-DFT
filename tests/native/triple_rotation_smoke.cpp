#include "env_cache.hpp"
#include "compute_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
using namespace sdft;
namespace {
double independent(std::size_t m,std::size_t n,std::size_t k) {
  const std::size_t lda=m+5,ldb=k+3,ldc=m+7;
  const Complex64 sentinel(1234.,-2345.);
  std::array<std::vector<Complex64>,3> inputs,outputs;
  std::vector<Complex64> right(ldb*n);
  std::array<const Complex64*,3> a;std::array<Complex64*,3> c;
  for(int image=0;image<3;++image) {
    inputs[image].resize(lda*k);outputs[image].assign(ldc*n,sentinel);
    a[image]=inputs[image].data();c[image]=outputs[image].data();
  }
  double maximum=0.;
  for(int variant=0;variant<3;++variant) {
    for(int image=0;image<3;++image)for(std::size_t p=0;p<k;++p)for(std::size_t i=0;i<m;++i)
      inputs[image][i+p*lda]={std::sin(.07*(i+1)*(p+2+image+variant)),.3*std::cos(.13*(i+2)*(p+1+image))};
    for(std::size_t j=0;j<n;++j)for(std::size_t p=0;p<k;++p)
      right[p+j*ldb]={std::cos(.09*(p+2)*(j+1+variant)),std::sin(.03*(p+1)*(j+3+variant))};
    complex_gemm_triple_right(m,n,k,a,lda,right.data(),ldb,c,ldc,false);
    for(int image=0;image<3;++image)for(std::size_t j=0;j<n;++j) {
      for(std::size_t i=0;i<m;++i) {
        std::complex<long double> sum{};;
        for(std::size_t p=0;p<k;++p) {
          const auto av=inputs[image][i+p*lda],bv=right[p+j*ldb];
          sum+=std::complex<long double>(av.real(),av.imag())*std::complex<long double>(bv.real(),bv.imag());
        }
        const Complex64 reference(double(sum.real()),double(sum.imag()));
        const auto actual=outputs[image][i+j*ldc];
        if(!std::isfinite(actual.real())||!std::isfinite(actual.imag()))throw std::runtime_error("nonfinite triple output");
        maximum=std::max(maximum,std::abs(actual-reference)/std::max(1.,std::abs(reference)));
      }
      for(std::size_t i=m;i<ldc;++i)
        if(outputs[image][i+j*ldc]!=sentinel)throw std::runtime_error("triple output overwrote leading-dimension padding");
    }
  }
  bool rejected=false;
  try {complex_gemm_triple_right(m,n,k,a,lda,right.data(),ldb,{c[0],c[0],c[2]},ldc,true);}
  catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("aliased triple outputs accepted");
  rejected=false;
  try {complex_gemm_triple_right(m,n,k,{a[0],nullptr,a[2]},lda,right.data(),ldb,c,ldc,true);}
  catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("null triple input accepted");
  return maximum;
}
}
int main(int argc,char** argv) {
  try {
    bool gpu=false,benchmark=false;
    for(int i=1;i<argc;++i) {
      const std::string option=argv[i];
      if(option=="--require-gpu")gpu=true;
      else if(option=="--benchmark")benchmark=true;
      else throw std::invalid_argument("unknown triple rotation test option");
    }
    if(gpu) {
      if(compute_backend_name()!="mps")throw std::runtime_error("actual fused Metal rotation required");
      setenv("SDFT_GPU_MIN_WORK","0",1);sdft::refresh_environment_cache();
    }
    const double error=std::max({independent(23,19,37),independent(7,43,53),
        independent(65,17,2053),independent(513,3,41)});
    if(error>2e-11)throw std::runtime_error("independent joint rotation oracle exceeds unchanged 2e-11 limit");
    std::cout<<std::setprecision(17)<<"{\"status\":\"PASSED_ROTATION_ONLY\",\"independent_scaled_error\":"<<error
        <<",\"required_fused_metal\":"<<(gpu?"true":"false")<<",\"timings\":[";
    if(benchmark) {
      constexpr std::size_t m=13353,n=96,k=384;
      std::array<std::vector<Complex64>,3> a,c,reference;
      std::array<const Complex64*,3> ap;std::array<Complex64*,3> cp;
      std::vector<Complex64> b(k*n);
      for(std::size_t i=0;i<b.size();++i)b[i]={std::sin(.09*i),std::cos(.07*i)};
      for(int image=0;image<3;++image) {
        a[image].resize(m*k);c[image].resize(m*n);reference[image].resize(m*n);
        for(std::size_t i=0;i<a[image].size();++i)a[image][i]={std::sin(.03*(i+1+image)),std::cos(.017*(i+3*image))};
        ap[image]=a[image].data();cp[image]=c[image].data();
        complex_gemm(MatrixOp::Normal,MatrixOp::Normal,m,n,k,ap[image],m,b.data(),k,reference[image].data(),m,true);
      }
      bool comma=false;
      for(int rep=1;rep<=3;++rep)for(int order=0;order<2;++order) {
        const bool joint=(order+rep)%2;
        const auto start=std::chrono::steady_clock::now();
        if(joint)complex_gemm_triple_right(m,n,k,ap,m,b.data(),k,cp,m,false);
        else for(int image=0;image<3;++image)
          complex_gemm(MatrixOp::Normal,MatrixOp::Normal,m,n,k,ap[image],m,b.data(),k,cp[image],m,false);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        double difference=0.,scale=1.;
        for(int image=0;image<3;++image)for(std::size_t i=0;i<c[image].size();++i) {
          difference=std::max(difference,std::abs(c[image][i]-reference[image][i]));scale=std::max(scale,std::abs(reference[image][i]));
        }
        if(difference/scale>2e-11)throw std::runtime_error("large rotation FP64 reference disagreement");
        std::cout<<(comma?",":"")<<"{\"joint\":"<<(joint?"true":"false")<<",\"rep\":"<<rep<<",\"seconds\":"<<seconds<<",\"scaled_error\":"<<difference/scale<<"}";comma=true;
      }
    }
    std::cout<<"],\"full_scf_speedup_verified\":false}\n";
    write_performance_report("triple_rotation.performance.json");
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
