#pragma once
#include <complex>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include "pbe_xc.hpp"

namespace sdft {
using Complex64 = std::complex<double>;
enum class MatrixOp { Normal, Adjoint };
// Real FP64 CPU BLAS. Adjoint is transpose for real operands. This is not
// a GPU dispatch and never changes the compute backend's search precision.
void cpu_real_gemm(MatrixOp aop,MatrixOp bop,std::size_t m,std::size_t n,
    std::size_t k,const double* a,std::size_t lda,const double* b,
    std::size_t ldb,double* c,std::size_t ldc,bool add=false);

// Column-major, leading dimensions are explicit. Requested accelerators fail
// if unavailable; CPU dispatch below the work threshold is counted separately.
void complex_gemm(MatrixOp aop, MatrixOp bop, std::size_t m, std::size_t n,
                  std::size_t k, const Complex64* a, std::size_t lda,
                  const Complex64* b, std::size_t ldb, Complex64* c,
                  std::size_t ldc, bool exact = false, bool add = false);
// Disjoint A_i * B products, shared dynamic right operand, normal views.
void complex_gemm_triple_right(std::size_t m,std::size_t n,std::size_t k,
                  const std::array<const Complex64*,3>& a,std::size_t lda,
                  const Complex64* b,std::size_t ldb,
                  const std::array<Complex64*,3>& c,std::size_t ldc,bool exact=false);
// Identity is a unique lifetime token, not a host address. Values/shapes of
// the left operand must remain immutable for that identity. Normal/adjoint
// device views may persist; dynamic right operands are always uploaded anew.
std::uint64_t new_immutable_matrix_identity();
void complex_gemm_immutable_left(std::uint64_t identity, MatrixOp aop,
                  std::size_t m,std::size_t n,std::size_t k,
                  const Complex64* a,std::size_t lda,const Complex64* b,
                  std::size_t ldb,Complex64* c,std::size_t ldc,bool exact=false,bool add=false);
// Nonzero token promises an immutable shape/index/potential for that operator
// lifetime. Zero performs content comparisons. Kinetic/input remain dynamic.
bool accelerated_local_hamiltonian(const std::array<int,3>& shape,
    const std::vector<std::size_t>& indices,const std::vector<double>& kinetic,
    const std::vector<double>& potential,const Complex64* input,
    std::size_t columns,Complex64* output,bool exact,std::uint64_t operator_identity=0);
bool accelerated_pbe_points(const std::vector<double>& density,
    const std::vector<double>& gradient,std::vector<PbeXcPoint>& output);
bool accelerated_paw_pair(std::uint64_t identity,std::size_t npw,std::size_t projectors,
    std::size_t columns,const Complex64* b,const Complex64* d,const Complex64* q,
    const Complex64* input,Complex64* h,Complex64* s,bool exact,std::uint64_t projector_identity=0);
std::string compute_backend_name();
std::string compiled_compute_capabilities();
void record_performance(const std::string& name, double seconds, std::size_t calls=1);
void write_performance_report(const std::string& filename);

class PerformanceTimer {
 public:
  explicit PerformanceTimer(std::string name);
  ~PerformanceTimer();
 private:
  std::string name_;
  double start_;
};
class Accelerator {
 public:
  virtual ~Accelerator()=default;
  virtual std::string device() const=0;
  virtual void gemm(MatrixOp,MatrixOp,std::size_t,std::size_t,std::size_t,
      const Complex64*,std::size_t,const Complex64*,std::size_t,Complex64*,std::size_t,bool)=0;
  virtual void gemm_immutable_left(std::uint64_t,MatrixOp ao,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add) {
    gemm(ao,MatrixOp::Normal,m,n,k,a,lda,b,ldb,c,ldc,add);
  }
  virtual void gemm_triple_right(std::size_t m,std::size_t n,std::size_t k,
      const std::array<const Complex64*,3>& a,std::size_t lda,
      const Complex64* b,std::size_t ldb,const std::array<Complex64*,3>& c,std::size_t ldc) {
    // Compatibility only; a backend that retains three calls must not
    // claim fused numerical execution.
    for(std::size_t i=0;i<3;++i)gemm(MatrixOp::Normal,MatrixOp::Normal,m,n,k,a[i],lda,b,ldb,c[i],ldc,false);
  }
  virtual void local_hamiltonian(const std::array<int,3>&,
      const std::vector<std::size_t>&,const std::vector<double>&,
      const std::vector<double>&,const Complex64*,std::size_t,Complex64*,std::uint64_t)=0;
  virtual void pbe_points(const double*,const double*,std::size_t,PbeXcPoint*) {
    throw std::runtime_error("native PBE is not implemented for this accelerator");
  }
  virtual bool paw_pair(std::uint64_t,std::size_t,std::size_t,std::size_t,
      const Complex64*,const Complex64*,const Complex64*,const Complex64*,Complex64*,Complex64*,std::uint64_t) {return false;}
};
std::unique_ptr<Accelerator> make_metal_accelerator();
std::unique_ptr<Accelerator> make_cuda_accelerator();
}
