#pragma once
#include "sdft_core.hpp"
#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace sdft {
// Fixed-wavefunction PAW derivative. All matrices are column-major. The
// response contracts occupied bands once; no altered D/Q or fitted force.
struct PawDerivativeResponse {
  std::size_t projectors{},plane_waves{};
  std::vector<std::complex<double>> sensitivity;
};
PawDerivativeResponse paw_derivative_response(std::size_t projectors,std::size_t plane_waves,
    std::size_t bands,const std::complex<double>* beta,const std::complex<double>* coefficients,
    const std::complex<double>* d,const std::complex<double>* q,
    const double* eigenvalues,const double* occupations);
double paw_derivative_contract(const PawDerivativeResponse&,
    const std::complex<double>* derivative_beta);
// Same fixed-wavefunction derivatives, in the original column-major sum
// order. No allocated derivative matrices and no parallel reduction tree.
Vec3 paw_derivative_position_contract(const PawDerivativeResponse&,
    const std::complex<double>* beta, const std::vector<Vec3>& wavevectors);
std::array<Vec3,3> paw_derivative_strain_contract(const PawDerivativeResponse&,
    const std::complex<double>* beta,
    const std::array<const std::complex<double>*,3>& momentum_gradients,
    const std::vector<Vec3>& wavevectors);
}
