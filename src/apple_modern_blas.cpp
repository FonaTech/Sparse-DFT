#include "apple_modern_blas.hpp"
#ifndef ACCELERATE_NEW_LAPACK
#error This translation unit requires the current Apple BLAS API.
#endif
#ifdef ACCELERATE_LAPACK_ILP64
#error Sparse-DFT's current BLAS dimensions use the checked LP64 API.
#endif
#include <vecLib/cblas_new.h>
#include <vecLib/lapack.h>
#include <vecLib/thread_api.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sdft {
namespace {
void single_thread_policy() {
  if(__builtin_available(macOS 15.0,*)) {
    thread_local const int policy=BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    if(policy)throw std::runtime_error("Apple BLAS single-thread policy was rejected");
  }
}
int workspace(double value) {
  if(!std::isfinite(value)||value<1||value>std::numeric_limits<int>::max())
    throw std::runtime_error("current Apple LAPACK workspace out of LP64 range");
  return static_cast<int>(std::ceil(value));
}
}
void apple_modern_dgemm(MatrixOp ao,MatrixOp bo,int m,int n,int k,
    const double* a,int lda,const double* b,int ldb,double* c,int ldc,bool add) {
  single_thread_policy();
  cblas_dgemm(CblasColMajor,ao==MatrixOp::Normal?CblasNoTrans:CblasTrans,
      bo==MatrixOp::Normal?CblasNoTrans:CblasTrans,m,n,k,1.,a,lda,b,ldb,
      add?1.:0.,c,ldc);
}
void apple_modern_zgemm(MatrixOp ao,MatrixOp bo,int m,int n,int k,
    const Complex64* a,int lda,const Complex64* b,int ldb,
    Complex64* c,int ldc,bool add) {
  static_assert(sizeof(__LAPACK_int)==sizeof(int));
  static_assert(sizeof(__LAPACK_double_complex)==sizeof(Complex64));
  static_assert(alignof(__LAPACK_double_complex)==alignof(Complex64));
  // The documented setting is thread-local. Every OpenMP worker explicitly
  // requests one BLAS thread, preventing an internal team per output tile.
  single_thread_policy();
  const Complex64 one(1.),beta(add?1.:0.);
  cblas_zgemm(CblasColMajor,ao==MatrixOp::Normal?CblasNoTrans:CblasConjTrans,
      bo==MatrixOp::Normal?CblasNoTrans:CblasConjTrans,m,n,k,
      reinterpret_cast<const __LAPACK_double_complex*>(&one),
      reinterpret_cast<const __LAPACK_double_complex*>(a),lda,
      reinterpret_cast<const __LAPACK_double_complex*>(b),ldb,
      reinterpret_cast<const __LAPACK_double_complex*>(&beta),
      reinterpret_cast<__LAPACK_double_complex*>(c),ldc);
}
void apple_modern_pencil(int n,Complex64* h,Complex64* s,Complex64* alpha,Complex64* beta,Complex64* vectors) {
  if(n<1||!h||!s||!alpha||!beta||!vectors)throw std::invalid_argument("invalid original-pencil QZ input");
  single_thread_policy();const char no='N',yes='V';const int one=1;int info=0,lwork=-1;
  Complex64 unused,query;std::vector<double> rw(8*n);
  zggev_(&no,&yes,&n,h,&n,s,&n,alpha,beta,&unused,&one,vectors,&n,&query,&lwork,rw.data(),&info);
  if(info)throw std::runtime_error("current Apple original-pencil QZ query failed");
  lwork=std::max(2*n,workspace(query.real()));std::vector<Complex64> work(lwork);
  zggev_(&no,&yes,&n,h,&n,s,&n,alpha,beta,&unused,&one,vectors,&n,work.data(),&lwork,rw.data(),&info);
  if(info)throw std::runtime_error("current Apple original-pencil QZ failed: "+std::to_string(info));
}
int apple_modern_ritz(int n,int requested,const std::string& choice,
    Complex64* h,Complex64* s,double* values,Complex64* vectors) {
  if(n<1||!h||!s||!values||!vectors)throw std::invalid_argument("invalid current LAPACK Ritz input");
  single_thread_policy();
  const int itype=1;const char job='V',triangle='L';int info=0,lwork=-1;
  if(choice=="subset" && requested>0 && requested<n) {
    const char range='I';const int first=1,last=requested;const double unused=0,abstol=0;int found=0;
    std::vector<double> rw(7*n);std::vector<int> iw(5*n),failed(n);Complex64 query;
    zhegvx_(&itype,&job,&range,&triangle,&n,h,&n,s,&n,&unused,&unused,&first,&last,&abstol,
        &found,values,vectors,&n,&query,&lwork,rw.data(),iw.data(),failed.data(),&info);
    if(info)throw std::runtime_error("current Apple selected Ritz query failed: "+std::to_string(info));
    lwork=std::max(2*n,workspace(query.real()));std::vector<Complex64> w(lwork);
    zhegvx_(&itype,&job,&range,&triangle,&n,h,&n,s,&n,&unused,&unused,&first,&last,&abstol,
        &found,values,vectors,&n,w.data(),&lwork,rw.data(),iw.data(),failed.data(),&info);
    if(info||found!=last)throw std::runtime_error("current Apple selected Ritz failed: "+std::to_string(info));
    return found;
  }
  if(choice=="divide_conquer"||(choice=="auto"&&n>=64)) {
    Complex64 cq;double rq=0;int iq=0,lrwork=-1,liwork=-1;
    zhegvd_(&itype,&job,&triangle,&n,h,&n,s,&n,values,&cq,&lwork,&rq,&lrwork,&iq,&liwork,&info);
    if(info)throw std::runtime_error("current Apple divide-conquer Ritz query failed");
    lwork=workspace(cq.real());lrwork=workspace(rq);liwork=workspace(iq);
    std::vector<Complex64> w(lwork);std::vector<double> rw(lrwork);std::vector<int> iw(liwork);
    zhegvd_(&itype,&job,&triangle,&n,h,&n,s,&n,values,w.data(),&lwork,rw.data(),&lrwork,iw.data(),&liwork,&info);
    if(info)throw std::runtime_error("current Apple divide-conquer Ritz failed: "+std::to_string(info));
  }else {
    std::vector<double> rw(std::max(1,3*n-2));Complex64 query;
    zhegv_(&itype,&job,&triangle,&n,h,&n,s,&n,values,&query,&lwork,rw.data(),&info);
    if(info)throw std::runtime_error("current Apple QR Ritz query failed");
    lwork=std::max(2*n-1,workspace(query.real()));std::vector<Complex64> w(lwork);
    zhegv_(&itype,&job,&triangle,&n,h,&n,s,&n,values,w.data(),&lwork,rw.data(),&info);
    if(info)throw std::runtime_error("current Apple QR Ritz failed: "+std::to_string(info));
  }
  std::copy_n(h,std::size_t(n)*n,vectors);return n;
}
}
