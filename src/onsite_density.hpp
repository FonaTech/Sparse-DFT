#pragma once
#include "compute_backend.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sdft {
// Return the row-major Hermitian one-centre occupation matrix
// rho_ij = sum_n f_n <p_i|psi_n><psi_n|p_j>. The input is an atom's
// row view in a single fresh, all-atom FP64 B*psi projection for this k point.
inline std::vector<Complex64> onsite_density_from_amplitudes(std::size_t rows,
    std::size_t bands,const Complex64* amplitudes,std::size_t stride,
    const double* occupations) {
  if(!rows||!bands||!amplitudes||!occupations||stride<rows||
     rows>std::numeric_limits<std::size_t>::max()/std::max(rows,bands))
    throw std::invalid_argument("invalid projected onsite density dimensions");
  std::vector<Complex64> weighted(rows*bands),density(rows*rows),output(rows*rows);
  for(std::size_t band=0;band<bands;++band) {
    const double occupation=occupations[band];
    if(!std::isfinite(occupation)||occupation<0||occupation>2)
      throw std::invalid_argument("invalid projected onsite occupation");
    for(std::size_t row=0;row<rows;++row) {
      const auto value=amplitudes[row+band*stride];
      if(!std::isfinite(value.real())||!std::isfinite(value.imag()))
        throw std::invalid_argument("nonfinite onsite projection");
      weighted[row+band*rows]=occupation*value;
    }
  }
  complex_gemm(MatrixOp::Normal,MatrixOp::Adjoint,rows,rows,bands,
      weighted.data(),rows,amplitudes,stride,density.data(),rows,true);
  for(std::size_t row=0;row<rows;++row)for(std::size_t col=0;col<rows;++col)
    output[row*rows+col]=density[row+col*rows];
  return output;
}
}
