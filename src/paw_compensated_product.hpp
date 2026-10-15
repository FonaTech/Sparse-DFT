#pragma once
#include "compute_backend.hpp"
#include "thread_budget.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SDFT_COMPENSATED_NEON 1
#endif

namespace sdft {
// A large PAW projector is not itself an ill-conditioned overlap. Radial
// channels can cancel inside B^H D B even when S has a modest condition
// number. Retain the original factors and compensate their dot products.
// FMA recovers each product's rounding error; Neumaier accumulation keeps
// small terms through cancellation. No reliance on wider long double on ARM.
class PawCompensatedSum {
  double sum_=0., correction_=0.;
 public:
  // Branch-free TwoSum. In round-to-nearest it yields the same exact
  // rounding error as the |a|>=|b| (Fast2Sum/Neumaier) branch it replaces,
  // so every partial sum and correction is bitwise unchanged.
  void add(double value) {
    const double next=sum_+value;
    const double back=next-sum_;
    correction_ += (sum_-(next-back))+(value-back);
    sum_=next;
  }
  void product(double a,double b) {
    const double rounded=a*b;
    add(rounded);add(std::fma(a,b,-rounded));
  }
  double value() const {return sum_+correction_;}
};

namespace detail {
// Scalar reference order for one output: c, then per k re*re, -im*im (real)
// and re*im, im*re (imaginary). Every path below reproduces it exactly.
inline void compensated_paw_output(MatrixOp operation,std::size_t row,std::size_t column,
    std::size_t inner,const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,
    Complex64* c,std::size_t ldc,bool add) {
  PawCompensatedSum real,imaginary;
  if(add){real.add(c[row+column*ldc].real());imaginary.add(c[row+column*ldc].imag());}
  for(std::size_t k=0;k<inner;++k) {
    const auto left=operation==MatrixOp::Normal?a[row+k*lda]:std::conj(a[k+row*lda]);
    const auto right=b[k+column*ldb];
    real.product(left.real(),right.real());real.product(-left.imag(),right.imag());
    imaginary.product(left.real(),right.imag());imaginary.product(left.imag(),right.real());
  }
  c[row+column*ldc]={real.value(),imaginary.value()};
}
#ifdef SDFT_COMPENSATED_NEON
// Two outputs per 128-bit lane pair, two pairs per tile. Lane-wise IEEE
// add/sub/mul/fma equal their scalar counterparts, so each lane follows
// the scalar sequence above bit for bit.
struct LaneSum {
  float64x2_t sum=vdupq_n_f64(0.),correction=vdupq_n_f64(0.);
  void add(float64x2_t value) {
    const float64x2_t next=vaddq_f64(sum,value);
    const float64x2_t back=vsubq_f64(next,sum);
    correction=vaddq_f64(correction,vaddq_f64(vsubq_f64(sum,vsubq_f64(next,back)),vsubq_f64(value,back)));
    sum=next;
  }
  void product(float64x2_t a,float64x2_t b) {
    const float64x2_t rounded=vmulq_f64(a,b);
    add(rounded);add(vfmaq_f64(vnegq_f64(rounded),a,b));
  }
  float64x2_t value() const {return vaddq_f64(sum,correction);}
};
inline void compensated_paw_tile4(MatrixOp operation,std::size_t row,std::size_t column,
    std::size_t inner,const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,
    Complex64* c,std::size_t ldc,bool add) {
  LaneSum real[2],imaginary[2];
  Complex64* out=c+row+column*ldc;
  if(add)for(int v=0;v<2;++v){
    const float64x2x2_t existing=vld2q_f64(reinterpret_cast<const double*>(out+2*v));
    real[v].add(existing.val[0]);imaginary[v].add(existing.val[1]);
  }
  const Complex64* right=b+column*ldb;
  const bool normal=operation==MatrixOp::Normal;
  for(std::size_t k=0;k<inner;++k) {
    const float64x2_t rr=vdupq_n_f64(right[k].real()),ri=vdupq_n_f64(right[k].imag());
    for(int v=0;v<2;++v) {
      float64x2_t lr,li;
      if(normal) {
        const float64x2x2_t pair=vld2q_f64(reinterpret_cast<const double*>(a+row+2*v+k*lda));
        lr=pair.val[0];li=pair.val[1];
      } else {
        const Complex64 x=a[k+(row+2*v)*lda],y=a[k+(row+2*v+1)*lda];
        lr=vsetq_lane_f64(y.real(),vdupq_n_f64(x.real()),1);
        li=vnegq_f64(vsetq_lane_f64(y.imag(),vdupq_n_f64(x.imag()),1));
      }
      real[v].product(lr,rr);real[v].product(vnegq_f64(li),ri);
      imaginary[v].product(lr,ri);imaginary[v].product(li,rr);
    }
  }
  for(int v=0;v<2;++v){
    float64x2x2_t result;result.val[0]=real[v].value();result.val[1]=imaginary[v].value();
    vst2q_f64(reinterpret_cast<double*>(out+2*v),result);
  }
}
#endif
}  // namespace detail

inline void compensated_paw_product(MatrixOp operation,std::size_t rows,
    std::size_t columns,std::size_t inner,const Complex64* a,std::size_t lda,
    const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add=false) {
  if(!rows||!columns||!inner||!a||!b||!c||a==c||b==c||
      ldb<inner||ldc<rows||lda<(operation==MatrixOp::Normal?rows:inner))
    throw std::invalid_argument("invalid compensated PAW product dimensions");
  constexpr std::size_t tile=4;
  const std::size_t row_tiles=(rows+tile-1)/tile,tasks=row_tiles*columns;
  int threads=1;
#ifdef SDFT_HAS_OPENMP
  // Outputs are independent and each is summed in a fixed order, so the
  // thread count never changes a bit. Inside a k-point team this uses the
  // team's inner budget instead of silently going serial.
  if(rows*columns*inner>=32768)threads=static_cast<int>(std::min<std::size_t>(kernel_threads(),tasks));
#pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
#endif
  for(std::size_t task=0;task<tasks;++task) {
    const std::size_t row=(task%row_tiles)*tile,column=task/row_tiles;
#ifdef SDFT_COMPENSATED_NEON
    if(rows-row>=tile){detail::compensated_paw_tile4(operation,row,column,inner,a,lda,b,ldb,c,ldc,add);continue;}
#endif
    for(std::size_t r=row;r<std::min(rows,row+tile);++r)
      detail::compensated_paw_output(operation,r,column,inner,a,lda,b,ldb,c,ldc,add);
  }
  (void)threads;
}
} // namespace sdft
