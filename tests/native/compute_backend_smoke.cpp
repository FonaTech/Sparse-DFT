#include "compute_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
// Deliberately does not call BLAS or the production GEMM. Exercise column
// splitting, conjugate operands, padded strides, beta=0 and beta=1.
double independent_gemm_check(std::size_t m,std::size_t n,std::size_t k){
  using namespace sdft;
  double maximum=0;
  for(auto ao:{MatrixOp::Normal,MatrixOp::Adjoint})for(auto bo:{MatrixOp::Normal,MatrixOp::Adjoint}){
    const auto ar=ao==MatrixOp::Normal?m:k,ac=ao==MatrixOp::Normal?k:m;
    const auto br=bo==MatrixOp::Normal?k:n,bc=bo==MatrixOp::Normal?n:k;
    const auto lda=ar+3,ldb=br+5,ldc=m+7;
    const Complex64 sentinel(9876.,-5432.);
    std::vector<Complex64> a(lda*ac,sentinel),b(ldb*bc,sentinel),ref(m*n);
    for(std::size_t j=0;j<ac;++j)for(std::size_t i=0;i<ar;++i)
      a[i+j*lda]={std::sin(.13*(i+2)*(j+1)),std::cos(.17*(i+4)*(j+2))};
    for(std::size_t j=0;j<bc;++j)for(std::size_t i=0;i<br;++i)
      b[i+j*ldb]={std::cos(.21*(i+1)*(j+3)),std::sin(.31*(i+3)*(j+2))};
    double scale=1.;
    for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i){
      Complex64 sum{};
      for(std::size_t p=0;p<k;++p)
        sum+=(ao==MatrixOp::Normal?a[i+p*lda]:std::conj(a[p+i*lda]))*
             (bo==MatrixOp::Normal?b[p+j*ldb]:std::conj(b[j+p*ldb]));
      ref[i+j*m]=sum;scale=std::max(scale,std::abs(sum));
    }
    for(bool exact:{true,false}){
      std::vector<Complex64> c(ldc*n,sentinel);
      for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i)
        c[i+j*ldc]={std::numeric_limits<double>::quiet_NaN(),0.};
      complex_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,c.data(),ldc,exact,false);
      for(bool add:{false,true}){
        if(add){
          for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i)c[i+j*ldc]={.01*(i+1),-.02*(j+1)};
          complex_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,c.data(),ldc,exact,true);
        }
        for(std::size_t j=0;j<n;++j){
          for(std::size_t i=0;i<m;++i){
            const auto expected=ref[i+j*m]+(add?Complex64(.01*(i+1),-.02*(j+1)):Complex64{});
            const auto error=std::abs(c[i+j*ldc]-expected)/scale;
            if(!std::isfinite(error))throw std::runtime_error("GEMM read beta=0 output or produced nonfinite data");
            maximum=std::max(maximum,error);
          }
          for(std::size_t i=m;i<ldc;++i)if(c[i+j*ldc]!=sentinel)
            throw std::runtime_error("GEMM overwrote output stride padding");
        }
      }
    }
  }
  return maximum;
}
double immutable_operand_check(){
  using namespace sdft;double error=0;
  constexpr std::size_t m=43,k=67,n=17;
  for(auto ao:{MatrixOp::Normal,MatrixOp::Adjoint}){
    const auto ar=ao==MatrixOp::Normal?m:k,ac=ao==MatrixOp::Normal?k:m;
    const std::size_t lda=ar+5,ldb=k+3,ldc=m+7;
    std::vector<Complex64> a(lda*ac),b(ldb*n),c(ldc*n),ref(ldc*n);
    std::uint64_t key=new_immutable_matrix_identity();
    for(int rep=0;rep<4;++rep){
      // Same host storage gets a new identity when its contents change;
      // dynamic B changes every call. A pointer-only cache would fail here.
      if(rep==0||rep==2){if(rep==2)key=new_immutable_matrix_identity();
        for(std::size_t j=0;j<ac;++j)for(std::size_t i=0;i<ar;++i)a[i+j*lda]={std::sin(.17*(i+1)*(j+2+rep)),std::cos(.31*(j+1)*(i+3))};}
      for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<k;++i)b[i+j*ldb]={std::cos(.19*(i+rep+1)*(j+3)),std::sin(.23*(j+1)*(i+2))};
      for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i){Complex64 sum{};
        for(std::size_t p=0;p<k;++p)sum+=(ao==MatrixOp::Normal?a[i+p*lda]:std::conj(a[p+i*lda]))*b[p+j*ldb];
        ref[i+j*ldc]=sum;c[i+j*ldc]=rep%2?Complex64(.1,-.2):Complex64{};}
      complex_gemm_immutable_left(key,ao,m,n,k,a.data(),lda,b.data(),ldb,c.data(),ldc,false,rep%2);
      for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i)
        error=std::max(error,std::abs(c[i+j*ldc]-ref[i+j*ldc]-(rep%2?Complex64(.1,-.2):Complex64{}))/std::max(1.,std::abs(ref[i+j*ldc])));
    }
  }
  return error;
}
}
int main(int argc,char** argv){
  using namespace sdft;
  try {
    const bool benchmark=argc>1&&std::string(argv[1])=="--benchmark";
    const std::size_t size=benchmark&&argc>2?std::stoull(argv[2]):512;
    const std::size_t m=benchmark?size:23,n=benchmark?size:19,k=benchmark?size:37;
    double maximum=benchmark?0.:std::max({independent_gemm_check(23,19,37),
        independent_gemm_check(127,89,367),independent_gemm_check(4097,1,1031),immutable_operand_check()});
    const bool fp32=std::getenv("SDFT_METAL_PRECISION")&&std::string(std::getenv("SDFT_METAL_PRECISION"))=="fp32";
    const double limit=fp32?2e-5:2e-11;
    const auto selected=compute_backend_name();
    std::cout<<"{\"backend\":\""<<selected<<"\",\"independent_reference_scaled_error\":"<<maximum<<",\"results\":[";bool comma=false;
    for(auto ao:{MatrixOp::Normal,MatrixOp::Adjoint})for(auto bo:{MatrixOp::Normal,MatrixOp::Adjoint}){
      const auto ar=ao==MatrixOp::Normal?m:k,ac=ao==MatrixOp::Normal?k:m;
      const auto br=bo==MatrixOp::Normal?k:n,bc=bo==MatrixOp::Normal?n:k;
      const std::size_t lda=ar+3,ldb=br+5,ldc=m+7;
      std::vector<Complex64> a(lda*ac),b(ldb*bc),c(ldc*n),ref(ldc*n);
      for(std::size_t j=0;j<ac;++j)for(std::size_t i=0;i<ar;++i)a[i+j*lda]={std::sin(.33*(i+2)*(j+1)),std::cos(.11*(i+4)*(j+2))};
      for(std::size_t j=0;j<bc;++j)for(std::size_t i=0;i<br;++i)b[i+j*ldb]={std::cos(.19*(i+1)*(j+3)),std::sin(.71*(i+3)*(j+2))};
      complex_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,ref.data(),ldc,true);
      // warm backend separately; all elapsed times include packing and wait.
      complex_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,c.data(),ldc);
      auto measure=[&](bool exact){std::vector<double> times;
        for(int rep=0;rep<(benchmark?3:1);++rep){const auto start=std::chrono::steady_clock::now();
          complex_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,c.data(),ldc,exact);
          times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());}
        std::sort(times.begin(),times.end());return times[times.size()/2];};
      const double cpu=measure(true),candidate=measure(false);double error=0,scale=1;
      for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i){error=std::max(error,std::abs(c[i+j*ldc]-ref[i+j*ldc]));scale=std::max(scale,std::abs(ref[i+j*ldc]));}
      maximum=std::max(maximum,error/scale);
      // Accumulating must not require valid previous output for beta=0.
      complex_gemm(ao,bo,m,n,k,a.data(),lda,b.data(),ldb,c.data(),ldc,false,true);
      for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<m;++i)maximum=std::max(maximum,std::abs(c[i+j*ldc]-2.*ref[i+j*ldc])/scale);
      std::cout<<(comma?",":"")<<"{\"a_adjoint\":"<<(ao==MatrixOp::Adjoint?"true":"false")<<",\"b_adjoint\":"<<(bo==MatrixOp::Adjoint?"true":"false")
        <<",\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"cpu_seconds\":"<<cpu<<",\"candidate_seconds\":"<<candidate
        <<",\"speedup\":"<<cpu/candidate<<",\"scaled_error\":"<<error/scale<<"}";comma=true;
    }
    std::cout<<"],\"precision\":\""<<(fp32?"fp32":"fp32x2_or_cpu_fp64")<<"\",\"max_scaled_error\":"<<maximum<<",\"arithmetic_limit\":"<<limit<<",\"passed\":"<<(maximum<limit?"true":"false")<<"}\n";
    write_performance_report("compute_backend.performance.json");
    return maximum<limit?0:3;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
