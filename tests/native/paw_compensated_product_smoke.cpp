#include "paw_compensated_product.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {
using C=std::complex<double>;
// The original one-output Neumaier loop, kept verbatim as the bitwise
// reference for the tiled, branch-free implementation.
struct ReferenceSum {
  double sum=0.,correction=0.;
  void add(double value){const double next=sum+value;
    correction+=std::abs(sum)>=std::abs(value)?(sum-next)+value:(value-next)+sum;sum=next;}
  void product(double a,double b){const double r=a*b;add(r);add(std::fma(a,b,-r));}
};
void reference_product(sdft::MatrixOp op,std::size_t rows,std::size_t columns,std::size_t inner,
    const C* a,std::size_t lda,const C* b,std::size_t ldb,C* c,std::size_t ldc,bool add){
  for(std::size_t index=0;index<rows*columns;++index){
    const std::size_t row=index%rows,column=index/rows;ReferenceSum re,im;
    if(add){re.add(c[row+column*ldc].real());im.add(c[row+column*ldc].imag());}
    for(std::size_t k=0;k<inner;++k){
      const auto l=op==sdft::MatrixOp::Normal?a[row+k*lda]:std::conj(a[k+row*lda]);const auto r=b[k+column*ldb];
      re.product(l.real(),r.real());re.product(-l.imag(),r.imag());im.product(l.real(),r.imag());im.product(l.imag(),r.real());}
    c[row+column*ldc]={re.sum+re.correction,im.sum+im.correction};
  }
}
// Random factors spanning 40 binades, with planted cancelling pairs.
bool bitwise_equal_random(){
  std::mt19937_64 rng(20261015);std::uniform_real_distribution<double> u(-1,1),e(-20,20);
  auto draw=[&]{return C(u(rng)*std::exp2(e(rng)),u(rng)*std::exp2(e(rng)));};
  for(auto op:{sdft::MatrixOp::Normal,sdft::MatrixOp::Adjoint})
  for(std::size_t rows:{1,3,4,5,13,64})for(std::size_t columns:{1,7,20})for(std::size_t inner:{1,9,517})
  for(bool add:{false,true}){
    const std::size_t lda=(op==sdft::MatrixOp::Normal?rows:inner)+1,ldb=inner+2,ldc=rows+3;
    std::vector<C> a(lda*(op==sdft::MatrixOp::Normal?inner:rows)),b(ldb*columns),c(ldc*columns);
    for(auto& x:a)x=draw();for(auto& x:b)x=draw();for(auto& x:c)x=draw();
    if(inner>2)for(std::size_t r=0;r<rows;++r){auto& p=op==sdft::MatrixOp::Normal?a[r]:a[r*lda];auto& q=op==sdft::MatrixOp::Normal?a[r+lda]:a[1+r*lda];q=-p;}
    auto expected=c,actual=c;
    reference_product(op,rows,columns,inner,a.data(),lda,b.data(),ldb,expected.data(),ldc,add);
    sdft::compensated_paw_product(op,rows,columns,inner,a.data(),lda,b.data(),ldb,actual.data(),ldc,add);
    for(std::size_t i=0;i<c.size();++i)
      if(std::memcmp(&expected[i],&actual[i],sizeof(C))){std::cerr<<"bitwise mismatch rows "<<rows<<" columns "<<columns<<" inner "<<inner<<"\n";return false;}
  }
  return true;
}
}  // namespace

int main(){try {
  if(!bitwise_equal_random())throw std::runtime_error("tiled compensated product differs from the reference loop");
  // The exact binary result survives a cancelled pair of 1e16 terms.
  // Real and imaginary cancellation, accumulation into an existing local
  // image, padded leading dimensions and adjoint layout are separate gates.
  std::vector<C> a={C(1e16,1e16),C(7,0),C(1,2),C(7,0),C(-1e16,-1e16),C(7,0)};
  std::vector<C> b={C(1),C(1),C(1)},c(4,C(31,37));
  sdft::compensated_paw_product(sdft::MatrixOp::Normal,1,1,3,a.data(),2,b.data(),3,c.data(),4);
  if(c[0]!=C(1,2)||c[1]!=C(31,37))throw std::runtime_error("exact complex cancellation or padding failed");
  c[0]=C(-1,-2);
  sdft::compensated_paw_product(sdft::MatrixOp::Normal,1,1,3,a.data(),2,b.data(),3,c.data(),4,true);
  if(c[0]!=C(0))throw std::runtime_error("local/nonlocal compensated addition failed");
  std::vector<C> adjoint={C(1e16,1e16),C(1,2),C(-1e16,-1e16)};
  sdft::compensated_paw_product(sdft::MatrixOp::Adjoint,1,1,3,adjoint.data(),3,b.data(),3,c.data(),4);
  if(c[0]!=C(1,-2))throw std::runtime_error("compensated adjoint layout failed");
  // Rounded multiplication alone loses the exact -2^-104 result. FMA
  // product recovery must retain it, independently of accumulation order.
  const double epsilon=std::ldexp(1.,-52);
  std::vector<C> p={C(1+epsilon),C(-1)},q={C(1-epsilon),C(1)};
  sdft::compensated_paw_product(sdft::MatrixOp::Normal,1,1,2,p.data(),1,q.data(),2,c.data(),4);
  if(c[0]!=C(-std::ldexp(1.,-104)))throw std::runtime_error("FMA product rounding recovery failed");
  // A larger batched product exercises the actual OpenMP output loop with
  // exactly representable values and both layouts at CPU1 and CPU8.
  constexpr std::size_t rows=13,inner=1139,columns=12;
  std::vector<C> large(rows*inner,C(1,.5)),right(inner*columns,C(.5,-1)),result(rows*columns);
  sdft::compensated_paw_product(sdft::MatrixOp::Normal,rows,columns,inner,
      large.data(),rows,right.data(),inner,result.data(),rows);
  for(auto value:result)if(value!=C(inner,-.75*inner))throw std::runtime_error("parallel complex product failed");
  std::cout<<"Original PAW factor cancellation, FMA product recovery, adjoint and batched threading PASS (bitwise equal to the reference loop)\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
