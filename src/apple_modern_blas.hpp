#pragma once
#include "compute_backend.hpp"
#include <string>
namespace sdft {
// Separately compiled against Apple's typed, current LP64 CBLAS API.
// No dlsym/ABI guesses or external runtime dependency.
void apple_modern_zgemm(MatrixOp ao,MatrixOp bo,int m,int n,int k,
    const Complex64* a,int lda,const Complex64* b,int ldb,
    Complex64* c,int ldc,bool add);
void apple_modern_dgemm(MatrixOp ao,MatrixOp bo,int m,int n,int k,
    const double* a,int lda,const double* b,int ldb,double* c,int ldc,bool add);
// Same projected generalized H/S, index selection and workspace convention.
// h/s are working copies; vectors are column-major n x returned roots.
void apple_modern_pencil(int n,Complex64* h,Complex64* s,Complex64* alpha,Complex64* beta,Complex64* vectors);
int apple_modern_ritz(int n,int requested,const std::string& choice,
    Complex64* h,Complex64* s,double* values,Complex64* vectors);
}
