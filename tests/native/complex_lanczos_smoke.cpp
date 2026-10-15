#include "env_cache.hpp"
#include "pw_core.hpp"
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <cstdlib>

int main() {
  using C = std::complex<double>;
  const std::size_t n = 14;
  Eigen::MatrixXcd a = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  Eigen::MatrixXcd b = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      const double re = 0.17 * static_cast<double>((i + 3) * (j + 5) % 11);
      const double im = (i == j) ? 0.0 : 0.031 * static_cast<double>((i + 2 * j + 1) % 7);
      a(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = C(re, im);
      a(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = std::conj(C(re, im));
      b(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = C(0.04 * (i + 1), 0.013 * (j + 1));
      b(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = std::conj(b(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)));
    }
  }
  Eigen::MatrixXcd s = b.adjoint() * b + 0.8 * Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> dense(a, s);
  if (dense.info() != Eigen::Success) return 2;
  std::vector<C> hv(n * n), sv(n * n);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
    hv[i * n + j] = a(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    sv[i * n + j] = s(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  }
  const auto iterative = sdft::solve_generalized_hermitian_iterative(hv, sv, n, 5, 128, 1e-10);
  if (iterative.eigenvalues.size() != 5 || !iterative.converged) return 3;
  double max_error = 0.0, max_residual = 0.0;
  for (std::size_t i = 0; i < 5; ++i) {
    max_error = std::max(max_error, std::abs(iterative.eigenvalues[i] - dense.eigenvalues()(static_cast<Eigen::Index>(i))));
    max_residual = std::max(max_residual, iterative.residual_norms[i]);
  }
  if (max_error > 1e-8 || max_residual > 1e-8) return 4;
  const auto davidson=sdft::solve_generalized_hermitian_davidson(hv,sv,n,5,256,1e-10);
  if(!davidson.converged)return 5;
  for(std::size_t i=0;i<5;++i)
    if(std::abs(davidson.eigenvalues[i]-dense.eigenvalues()[i])>1e-8)return 6;
  // Independently verify both small-space LAPACK algorithms against Eigen
  // and fresh generalized residuals before testing degeneracies/warm starts.
  for(const char* method : {"qr","divide_conquer"}) {
    setenv("SDFT_RITZ_SOLVER",method,1);sdft::refresh_environment_cache();
    const auto check=sdft::solve_generalized_hermitian_davidson(hv,sv,n,5,256,1e-12);
    if(!check.converged)return 15;
    for(std::size_t i=0;i<5;++i) {
      if(std::abs(check.eigenvalues[i]-dense.eigenvalues()[i])>1e-10)return 16;
      const Eigen::Map<const Eigen::VectorXcd> vector(check.eigenvectors.data()+i*n,n);
      if((a*vector-check.eigenvalues[i]*s*vector).norm()>1e-12 ||
         std::abs(vector.dot(s*vector)-C(1.))>1e-12)return 17;
    }
  }
  unsetenv("SDFT_RITZ_SOLVER");sdft::refresh_environment_cache();
  // Exact multiplicities and complex basis rotations are a separate gate.
  Eigen::MatrixXcd unitary=Eigen::HouseholderQR<Eigen::MatrixXcd>(a).householderQ();
  Eigen::VectorXd spectrum(n);
  for(std::size_t i=0;i<n;++i)spectrum[i]=double(i/3);
  Eigen::MatrixXcd clustered=unitary*spectrum.asDiagonal()*unitary.adjoint();
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){hv[i*n+j]=clustered(i,j);sv[i*n+j]=i==j?1.:0.;}
  const auto degenerate=sdft::solve_generalized_hermitian_davidson(hv,sv,n,7,256,1e-10);
  if(!degenerate.converged)return 7;
  for(std::size_t i=0;i<7;++i)if(std::abs(degenerate.eigenvalues[i]-spectrum[i])>1e-8)return 8;
  // Warm starts must keep exact multiplicities when the operator changes.
  for(std::size_t i=0;i<n;++i)hv[i*n+i]+=0.001*(i+1);
  Eigen::MatrixXcd changed=clustered;
  for(std::size_t i=0;i<n;++i)changed(i,i)+=0.001*(i+1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> expected(changed);
  const auto warm=sdft::solve_generalized_hermitian_davidson(hv,sv,n,7,256,1e-10,degenerate.eigenvectors);
  if(!warm.converged)return 9;
  for(std::size_t i=0;i<7;++i)
    if(std::abs(warm.eigenvalues[i]-expected.eigenvalues()[i])>1e-8)return 10;
  // More than half the full basis is requested: warm seeds plus canonical
  // augmentation must respect capacity, including when 2*roots > NPW.
  const auto many= sdft::solve_generalized_hermitian_davidson(hv,sv,n,10,256,1e-12);
  const auto many_warm= sdft::solve_generalized_hermitian_davidson(hv,sv,n,10,256,1e-12,many.eigenvectors);
  if(!many.converged||!many_warm.converged)return 11;
  for(std::size_t i=0;i<10;++i)
    if(std::abs(many_warm.eigenvalues[i]-expected.eigenvalues()[i])>1e-10)return 12;
  // A low eigenvalue can live in a disconnected block whose diagonal is
  // high. Canonical low-diagonal seeds alone have exactly zero overlap with
  // it and return the wrong roots with a zero residual. Exercise both cold
  // and misleading, already-converged warm starts.
  const std::size_t hidden_n=80, hidden_roots=4;
  std::vector<C> hidden_h(hidden_n*hidden_n), hidden_s(hidden_n*hidden_n), misleading(hidden_roots*hidden_n);
  for (std::size_t i=0; i<hidden_n; ++i) {
    hidden_h[i*hidden_n+i]=1.+i;
    hidden_s[i*hidden_n+i]=1.;
  }
  hidden_h[70*hidden_n+70]=hidden_h[71*hidden_n+71]=20.;
  hidden_h[70*hidden_n+71]=C(0.,19.75);
  hidden_h[71*hidden_n+70]=C(0.,-19.75);
  for (std::size_t i=0; i<hidden_roots; ++i) misleading[i*hidden_n+i]=1.;
  for (const auto& seeds : {std::vector<C>{}, misleading}) {
    const auto hidden=sdft::solve_generalized_hermitian_davidson(hidden_h,hidden_s,
        hidden_n,hidden_roots,256,1e-10,seeds);
    const double expected_hidden[]{.25,1.,2.,3.};
    if (!hidden.converged) return 13;
    for (std::size_t i=0; i<hidden_roots; ++i) if (std::abs(hidden.eigenvalues[i]-expected_hidden[i])>1e-9) {
      std::cerr << "Davidson missed a lowest root outside the canonical seed sector: root=" << i
                << " actual=" << hidden.eigenvalues[i] << " expected=" << expected_hidden[i] << '\n';
      return 14;
    }
  }
  std::cout << "complex_lanczos_smoke PASS roots=5 max_error=" << max_error
            << " max_residual=" << max_residual << " iterations=" << iterative.iterations
            << " davidson_iterations=" << davidson.iterations << " degenerate_davidson=PASS\n";
  return 0;
}
