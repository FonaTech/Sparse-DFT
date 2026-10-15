#include "env_cache.hpp"
#include "thread_budget.hpp"
#include "pw_core.hpp"
#include "compute_backend.hpp"
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
#include "apple_modern_blas.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <iomanip>

#ifdef SDFT_HAS_EIGEN
#include <Eigen/Dense>
#include "strict_residual_polish.hpp"
#endif

#ifdef SDFT_HAS_LAPACK
extern "C" void zhegv_(const int*,const char*,const char*,const int*,std::complex<double>*,const int*,
    std::complex<double>*,const int*,double*,std::complex<double>*,const int*,double*,int*);
extern "C" void zhegvd_(const int*,const char*,const char*,const int*,std::complex<double>*,const int*,
    std::complex<double>*,const int*,double*,std::complex<double>*,const int*,double*,const int*,int*,const int*,int*);
extern "C" void zhegvx_(const int*,const char*,const char*,const char*,const int*,std::complex<double>*,const int*,
    std::complex<double>*,const int*,const double*,const double*,const int*,const int*,const double*,int*,double*,
    std::complex<double>*,const int*,std::complex<double>*,const int*,double*,int*,int*,int*);
#endif

namespace sdft {

double kinetic_preconditioner(double kinetic,double wavefunction_kinetic) {
  if(!std::isfinite(kinetic)||kinetic<0||!std::isfinite(wavefunction_kinetic)||wavefunction_kinetic<0)
    throw std::invalid_argument("kinetic preconditioner requires nonnegative Hartree energies");
  const double scale=1.5*std::max(wavefunction_kinetic,2./27.211386245988);
  const double x=kinetic/scale;
  // Evaluate the rational polynomial in inverse x for large arguments;
  // this preserves the 1/T tail without overflowing x^4.
  if(x>1.) {
    const double inv=1./x;
    const double p=inv*(8.+inv*(12.+inv*(18.+27.*inv)));
    return (2./scale)*p/(p+16.);
  }
  const double p=27.+x*(18.+x*(12.+8.*x));
  return (2./scale)*p/(p+16.*x*x*x*x);
}

namespace {
constexpr double hartree_to_ev = 27.211386245988;
constexpr double two_pi = 6.283185307179586476925286766559;
Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 scale(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
#ifdef SDFT_HAS_EIGEN
struct RitzSpectrum {Eigen::VectorXd values;Eigen::MatrixXcd vectors;};
RitzSpectrum rayleigh_ritz(Eigen::MatrixXcd h,Eigen::MatrixXcd s,Eigen::Index requested=0, bool strict_refinement=false) {
#ifdef SDFT_HAS_LAPACK
  // Use the platform LAPACK only for the small projected space. The full
  // plane-wave operators stay matrix free, including their overlap metric.
  const int n=h.rows(),itype=1;const char job='V',triangle='L';int info=0,lwork=-1;
  const std::string configured=std::getenv("SDFT_RITZ_SOLVER")?std::getenv("SDFT_RITZ_SOLVER"):"auto";
  // A guarded search may be converged while the projected eigensolve's
  // backward error still exceeds a strict ORIGINAL-operator certificate.
  // QR is restricted to that final lane; explicit solver choices are kept.
  const std::string choice=strict_refinement&&configured=="auto"?"qr":configured;
  if(choice!="auto" && choice!="qr" && choice!="divide_conquer" && choice!="subset")
    throw std::invalid_argument("SDFT_RITZ_SOLVER must be auto, qr, divide_conquer or subset");
  const std::string interface=std::getenv("SDFT_CPU_LAPACK_INTERFACE")?std::getenv("SDFT_CPU_LAPACK_INTERFACE"):"legacy";
  if(interface!="legacy"&&interface!="modern")throw std::invalid_argument("SDFT_CPU_LAPACK_INTERFACE must be legacy or modern");
  if(interface=="modern") {
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
    PerformanceTimer timer("rayleigh_ritz_current_apple");
    const int columns=choice=="subset"&&requested>0&&requested<n?int(requested):n;
    Eigen::VectorXd values(n);Eigen::MatrixXcd vectors(n,columns);
    const int found=apple_modern_ritz(n,int(requested),choice,h.data(),s.data(),values.data(),vectors.data());
    values.conservativeResize(found);
    return {std::move(values),std::move(vectors)};
#else
    throw std::runtime_error("current Apple LAPACK was not compiled");
#endif
  }
  if(choice=="subset" && requested>0 && requested<n) {
    PerformanceTimer timer("rayleigh_ritz_subset");
    // Preserve the complete generalized metric. Select the low roots and
    // thick-restart guards by index; neither S=I nor a spectral shift is used.
    const char range='I';const int first=1,last=static_cast<int>(requested);
    const double unused=0,abstol=0;int found=0;
    Eigen::VectorXd values(n);Eigen::MatrixXcd vectors(n,requested);
    std::vector<double> real_work(7*n);std::vector<int> int_work(5*n),failed(n);
    std::complex<double> query;
    zhegvx_(&itype,&job,&range,&triangle,&n,h.data(),&n,s.data(),&n,&unused,&unused,&first,&last,&abstol,
        &found,values.data(),vectors.data(),&n,&query,&lwork,real_work.data(),int_work.data(),failed.data(),&info);
    if(info||!std::isfinite(query.real())||query.real()<1||query.real()>std::numeric_limits<int>::max())
      throw std::runtime_error("LAPACK selected Ritz workspace query failed: info="+std::to_string(info));
    lwork=std::max(2*n,static_cast<int>(std::ceil(query.real())));
    std::vector<std::complex<double>> work(lwork);
    zhegvx_(&itype,&job,&range,&triangle,&n,h.data(),&n,s.data(),&n,&unused,&unused,&first,&last,&abstol,
        &found,values.data(),vectors.data(),&n,work.data(),&lwork,real_work.data(),int_work.data(),failed.data(),&info);
    if(info||found!=last)throw std::runtime_error("LAPACK selected generalized Ritz failed: info="+std::to_string(info));
    values.conservativeResize(found);
    record_performance("rayleigh_ritz_unused_vectors_avoided",0,n-found);
    return {std::move(values),std::move(vectors)};
  }
  if(choice=="divide_conquer" || (choice=="auto" && n>=64)) {
    PerformanceTimer timer("rayleigh_ritz_divide_conquer");
    Eigen::VectorXd values(n);std::complex<double> work_query;double real_query=0.;int int_query=0;
    int lrwork=-1,liwork=-1;
    zhegvd_(&itype,&job,&triangle,&n,h.data(),&n,s.data(),&n,values.data(),&work_query,&lwork,
        &real_query,&lrwork,&int_query,&liwork,&info);
    if(info)throw std::runtime_error("LAPACK divide-conquer workspace query failed: info="+std::to_string(info));
    const auto workspace=[](double size) {
      if(!std::isfinite(size) || size<1 || size>std::numeric_limits<int>::max())
        throw std::runtime_error("LAPACK divide-conquer workspace exceeds integer range");
      return static_cast<int>(std::ceil(size));
    };
    lwork=workspace(work_query.real());lrwork=workspace(real_query);liwork=workspace(int_query);
    std::vector<std::complex<double>> work(lwork);std::vector<double> real_work(lrwork);std::vector<int> int_work(liwork);
    zhegvd_(&itype,&job,&triangle,&n,h.data(),&n,s.data(),&n,values.data(),work.data(),&lwork,
        real_work.data(),&lrwork,int_work.data(),&liwork,&info);
    if(info)throw std::runtime_error("LAPACK divide-conquer Rayleigh-Ritz failed: info="+std::to_string(info));
    return {std::move(values),std::move(h)};
  }
  PerformanceTimer timer("rayleigh_ritz_qr");
  Eigen::VectorXd values(n);std::vector<double> real_work(std::max(1,3*n-2));std::complex<double> query;
  zhegv_(&itype,&job,&triangle,&n,h.data(),&n,s.data(),&n,values.data(),&query,&lwork,real_work.data(),&info);
  if(info)throw std::runtime_error("LAPACK projected workspace query failed");
  lwork=std::max(2*n-1,static_cast<int>(query.real()));std::vector<std::complex<double>> work(lwork);
  zhegv_(&itype,&job,&triangle,&n,h.data(),&n,s.data(),&n,values.data(),work.data(),&lwork,real_work.data(),&info);
  if(info)throw std::runtime_error("LAPACK generalized Rayleigh-Ritz failed: info="+std::to_string(info));
  return {std::move(values),std::move(h)};
#else
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> solver(h,s);
  if(solver.info()!=Eigen::Success)throw std::runtime_error("Davidson Rayleigh-Ritz failed");
  return {solver.eigenvalues(),solver.eigenvectors()};
#endif
}
#endif
}

std::vector<GVector> enumerate_g_vectors(const std::array<Vec3, 3>& reciprocal,
                                         double cutoff_ev, Vec3 kpoint_fractional,
                                         double hartree_to_ev) {
  if (cutoff_ev <= 0.0 || !std::isfinite(cutoff_ev)) throw std::invalid_argument("plane-wave cutoff must be finite and positive");
  if(hartree_to_ev<=0 || !std::isfinite(hartree_to_ev))throw std::invalid_argument("invalid Hartree conversion");
  const double gmax = std::sqrt(2.0 * cutoff_ev / hartree_to_ev);
  const double det = determinant(reciprocal);
  if (std::abs(det) < 1e-12 || !std::isfinite(det)) throw std::invalid_argument("reciprocal lattice is singular");
  const Vec3 dual0 = scale(cross(reciprocal[1], reciprocal[2]), 1.0 / det);
  const Vec3 dual1 = scale(cross(reciprocal[2], reciprocal[0]), 1.0 / det);
  const Vec3 dual2 = scale(cross(reciprocal[0], reciprocal[1]), 1.0 / det);
  const int hmax = static_cast<int>(std::ceil(gmax * std::sqrt(dot(dual0, dual0)) + std::abs(kpoint_fractional.x))) + 1;
  const int kmax = static_cast<int>(std::ceil(gmax * std::sqrt(dot(dual1, dual1)) + std::abs(kpoint_fractional.y))) + 1;
  const int lmax = static_cast<int>(std::ceil(gmax * std::sqrt(dot(dual2, dual2)) + std::abs(kpoint_fractional.z))) + 1;
  const Vec3 shift = add(add(scale(reciprocal[0], kpoint_fractional.x),
                              scale(reciprocal[1], kpoint_fractional.y)),
                              scale(reciprocal[2], kpoint_fractional.z));
  std::vector<GVector> vectors;
  for (int h = -hmax; h <= hmax; ++h) for (int k = -kmax; k <= kmax; ++k) for (int l = -lmax; l <= lmax; ++l) {
    Vec3 g = add(add(scale(reciprocal[0], h), scale(reciprocal[1], k)), scale(reciprocal[2], l));
    const Vec3 shifted_g = add(g, shift);
    const double g2 = dot(shifted_g, shifted_g);
    if (0.5 * g2 <= cutoff_ev / hartree_to_ev + 1e-12) vectors.push_back({h, k, l, g, 0.5 * g2 * hartree_to_ev});
  }
  std::sort(vectors.begin(), vectors.end(), [](const auto& a, const auto& b) {
    if (a.kinetic_ev != b.kinetic_ev) return a.kinetic_ev < b.kinetic_ev;
    if (a.h != b.h) return a.h < b.h;
    if (a.k != b.k) return a.k < b.k;
    return a.l < b.l;
  });
  return vectors;
}

std::vector<GVector> enumerate_g_vectors_fixed_reference(
    const std::array<Vec3, 3>& reciprocal,
    const std::array<Vec3, 3>& reference_reciprocal,
    double cutoff_ev, Vec3 kpoint_fractional, double hartree_to_ev) {
  const double det = determinant(reciprocal);
  if (!std::isfinite(det) || std::abs(det) < 1e-12)
    throw std::invalid_argument("current reciprocal lattice is singular");
  auto vectors = enumerate_g_vectors(reference_reciprocal, cutoff_ev,
                                     kpoint_fractional, hartree_to_ev);
  const Vec3 shift = add(add(scale(reciprocal[0], kpoint_fractional.x),
                            scale(reciprocal[1], kpoint_fractional.y)),
                            scale(reciprocal[2], kpoint_fractional.z));
  for (auto& g : vectors) {
    g.cartesian = add(add(scale(reciprocal[0], g.h), scale(reciprocal[1], g.k)),
                      scale(reciprocal[2], g.l));
    const auto shifted = add(g.cartesian, shift);
    g.kinetic_ev = .5 * dot(shifted, shifted) * hartree_to_ev;
    if (!std::isfinite(g.kinetic_ev))
      throw std::invalid_argument("nonfinite fixed-reference kinetic energy");
  }
  // Retain the reference ordering: warm starts and strain differences must
  // address the same coefficient for every integer reciprocal index.
  return vectors;
}

void add_paw_low_rank_operator(std::vector<double>& hamiltonian,
                               std::vector<double>& overlap,
                               std::size_t dimension,
                               const PawProjectorOperator& projectors) {
  if (hamiltonian.size() != dimension * dimension || overlap.size() != dimension * dimension)
    throw std::invalid_argument("dense operator dimensions do not match basis");
  if (projectors.amplitudes.size() != projectors.nprojectors * dimension ||
      projectors.d_matrix.size() != projectors.nprojectors * projectors.nprojectors ||
      projectors.overlap_matrix.size() != projectors.nprojectors * projectors.nprojectors)
    throw std::invalid_argument("PAW projector dimensions are inconsistent");
  // H += P^T D P and S += P^T Q P for a real Gamma-point representation.
  for (std::size_t i = 0; i < dimension; ++i) for (std::size_t j = 0; j < dimension; ++j) {
    double h = 0.0, s = 0.0;
    for (std::size_t a = 0; a < projectors.nprojectors; ++a) for (std::size_t b = 0; b < projectors.nprojectors; ++b) {
      const double left = projectors.amplitudes[a * dimension + i];
      const double right = projectors.amplitudes[b * dimension + j];
      h += left * projectors.d_matrix[a * projectors.nprojectors + b] * right;
      s += left * projectors.overlap_matrix[a * projectors.nprojectors + b] * right;
    }
    hamiltonian[i * dimension + j] += h;
    overlap[i * dimension + j] += s;
  }
}

DenseGeneralizedEigenResult solve_generalized_symmetric(
    const std::vector<double>& hamiltonian,
    const std::vector<double>& overlap,
    std::size_t dimension) {
  if (dimension == 0 || hamiltonian.size() != dimension * dimension || overlap.size() != dimension * dimension)
    throw std::invalid_argument("invalid generalized eigenproblem dimensions");
#ifdef SDFT_HAS_EIGEN
  Eigen::MatrixXd h(dimension, dimension), s(dimension, dimension);
  for (std::size_t i = 0; i < dimension; ++i) for (std::size_t j = 0; j < dimension; ++j) {
    h(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = hamiltonian[i * dimension + j];
    s(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = overlap[i * dimension + j];
  }
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(h, s);
  if (solver.info() != Eigen::Success) throw std::runtime_error("generalized symmetric eigensolver failed");
  DenseGeneralizedEigenResult result;
  result.dimension = dimension;
  result.eigenvalues.resize(dimension);
  result.eigenvectors.resize(dimension * dimension);
  for (std::size_t i = 0; i < dimension; ++i) {
    result.eigenvalues[i] = solver.eigenvalues()(static_cast<Eigen::Index>(i));
    for (std::size_t j = 0; j < dimension; ++j)
      result.eigenvectors[i * dimension + j] = solver.eigenvectors()(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  }
  return result;
#else
  throw std::runtime_error("Sparse-DFT generalized eigensolver requires Eigen3 at build time");
#endif
}

DenseGeneralizedEigenResult solve_generalized_symmetric_iterative(
    const std::vector<double>& hamiltonian,
    const std::vector<double>& overlap,
    std::size_t dimension,
    std::size_t n_roots,
    std::size_t max_iterations,
    double tolerance) {
  if (dimension == 0 || hamiltonian.size() != dimension * dimension ||
      overlap.size() != dimension * dimension || n_roots == 0 ||
      n_roots > dimension || max_iterations == 0 || tolerance <= 0.0)
    throw std::invalid_argument("invalid iterative generalized eigenproblem dimensions");
#ifdef SDFT_HAS_EIGEN
  Eigen::MatrixXd h(dimension, dimension), s(dimension, dimension);
  for (std::size_t i = 0; i < dimension; ++i) for (std::size_t j = 0; j < dimension; ++j) {
    h(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = hamiltonian[i * dimension + j];
    s(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = overlap[i * dimension + j];
  }
  Eigen::LLT<Eigen::MatrixXd> llt(s);
  if (llt.info() != Eigen::Success) throw std::runtime_error("iterative generalized eigensolver requires positive-definite overlap");
  const Eigen::MatrixXd l = llt.matrixL();
  const Eigen::MatrixXd transformed = l.triangularView<Eigen::Lower>().solve(h) *
                                      l.transpose().triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(dimension, dimension));
  // The multiplication above is written as two triangular solves rather than
  // an explicit inverse.  Symmetrize to remove roundoff from the solve order.
  const Eigen::MatrixXd a = 0.5 * (transformed + transformed.transpose());
  Eigen::VectorXd q = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(dimension));
  q.normalize();
  Eigen::VectorXd previous = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
  double beta_previous = 0.0;
  std::vector<Eigen::VectorXd> basis;
  std::vector<double> alpha, beta;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> tri_solver;
  Eigen::MatrixXd ritz_vectors;
  Eigen::VectorXd ritz_values;
  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    // Full reorthogonalization avoids ghost roots when the PAW overlap makes
    // the low spectrum clustered.
    for (const auto& v : basis) q -= v.dot(q) * v;
    const double qnorm = q.norm();
    if (qnorm < 1e-14) break;
    q /= qnorm;
    basis.push_back(q);
    Eigen::VectorXd w = a * q - beta_previous * previous;
    const double aa = q.dot(w);
    w -= aa * q;
    for (const auto& v : basis) w -= v.dot(w) * v;
    const double bb = w.norm();
    alpha.push_back(aa);
    beta.push_back(bb);
    const Eigen::Index m = static_cast<Eigen::Index>(basis.size());
    Eigen::MatrixXd t = Eigen::MatrixXd::Zero(m, m);
    for (Eigen::Index i = 0; i < m; ++i) {
      t(i, i) = alpha[static_cast<std::size_t>(i)];
      if (i + 1 < m) t(i, i + 1) = t(i + 1, i) = beta[static_cast<std::size_t>(i)];
    }
    tri_solver.compute(t);
    if (tri_solver.info() != Eigen::Success) throw std::runtime_error("Lanczos tridiagonal eigensolver failed");
    ritz_values = tri_solver.eigenvalues();
    ritz_vectors = tri_solver.eigenvectors();
    const std::size_t roots = std::min(n_roots, static_cast<std::size_t>(m));
    bool converged = m >= static_cast<Eigen::Index>(n_roots);
    if (converged && bb > 0.0) for (std::size_t root = 0; root < roots; ++root)
      converged = converged && std::abs(bb * ritz_vectors(m - 1, static_cast<Eigen::Index>(root))) < tolerance;
    if (converged || bb < 1e-14 || iteration + 1 == max_iterations) break;
    previous = q;
    q = w;
    beta_previous = bb;
  }
  const std::size_t roots = std::min(n_roots, static_cast<std::size_t>(ritz_values.size()));
  DenseGeneralizedEigenResult result;
  result.dimension = dimension;
  result.eigenvalues.resize(roots);
  result.eigenvectors.assign(roots * dimension, 0.0);
  Eigen::MatrixXd qmatrix(dimension, static_cast<Eigen::Index>(basis.size()));
  for (std::size_t j = 0; j < basis.size(); ++j) qmatrix.col(static_cast<Eigen::Index>(j)) = basis[j];
  for (std::size_t root = 0; root < roots; ++root) {
    result.eigenvalues[root] = ritz_values(static_cast<Eigen::Index>(root));
    const Eigen::VectorXd y = qmatrix * ritz_vectors.col(static_cast<Eigen::Index>(root));
    const Eigen::VectorXd x = l.transpose().triangularView<Eigen::Upper>().solve(y);
    for (std::size_t component = 0; component < dimension; ++component)
      result.eigenvectors[root * dimension + component] = x(static_cast<Eigen::Index>(component));
  }
  return result;
#else
  throw std::runtime_error("Sparse-DFT iterative generalized eigensolver requires Eigen3 at build time");
#endif
}

ComplexGeneralizedEigenResult solve_generalized_hermitian_iterative(
    const std::vector<std::complex<double>>& hamiltonian,
    const std::vector<std::complex<double>>& overlap,
    std::size_t dimension,
    std::size_t n_roots,
    std::size_t max_iterations,
    double tolerance) {
  if (dimension == 0 || hamiltonian.size() != dimension * dimension ||
      overlap.size() != dimension * dimension || n_roots == 0 ||
      n_roots > dimension || max_iterations == 0 ||
      !(tolerance > 0.0) || !std::isfinite(tolerance))
    throw std::invalid_argument("invalid complex iterative generalized eigenproblem dimensions");
#ifdef SDFT_HAS_EIGEN
  using C = std::complex<double>;
  using MC = Eigen::MatrixXcd;
  using VC = Eigen::VectorXcd;
  MC h(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
  MC s(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
  for (std::size_t i = 0; i < dimension; ++i) {
    for (std::size_t j = 0; j < dimension; ++j) {
      h(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = hamiltonian[i * dimension + j];
      s(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = overlap[i * dimension + j];
    }
  }
  // Plane-wave assembly accumulates roundoff in conjugate matrix elements.
  // Symmetrizing here preserves the Hermitian problem required by Lanczos and
  // makes the numerical contract explicit instead of silently using a
  // non-Hermitian eigensolver.
  h = C(0.5, 0.0) * (h + h.adjoint());
  s = C(0.5, 0.0) * (s + s.adjoint());
  Eigen::LLT<MC> llt(s);
  if (llt.info() != Eigen::Success)
    throw std::runtime_error("complex iterative generalized eigensolver requires positive-definite overlap");
  const MC l = llt.matrixL();
  // Keep the adjoint in a named matrix.  Passing ``l.adjoint()`` directly to
  // triangularView creates a temporary expression whose lifetime can end
  // before Eigen finishes evaluating solve() on some libc++/Eigen versions.
  const MC ldag = l.adjoint();
  auto solve_upper = [&](const VC& rhs) {
    VC out = rhs;
    ldag.template triangularView<Eigen::Upper>().solveInPlace(out);
    return out;
  };
  auto solve_lower = [&](const VC& rhs) {
    VC out = rhs;
    l.template triangularView<Eigen::Lower>().solveInPlace(out);
    return out;
  };
  auto apply_a = [&](const VC& x) {
    // A x = L^{-1} H L^{-H} x.  No explicit inverse is formed.
    const VC y = solve_upper(x);
    const VC z = h * y;
    return solve_lower(z);
  };

  VC q(static_cast<Eigen::Index>(dimension));
  for (std::size_t i = 0; i < dimension; ++i) {
    const double phase = 0.17320508075688773 * static_cast<double>(i + 1);
    q(static_cast<Eigen::Index>(i)) = C(std::cos(phase), std::sin(phase));
  }
  q.normalize();
  VC previous = VC::Zero(static_cast<Eigen::Index>(dimension));
  double beta_previous = 0.0;
  std::vector<VC> basis;
  basis.reserve(max_iterations);
  std::vector<double> alpha;
  std::vector<double> beta;
  alpha.reserve(max_iterations);
  beta.reserve(max_iterations);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> tri_solver;
  Eigen::VectorXd ritz_values;
  Eigen::MatrixXd ritz_vectors;
  std::vector<double> residuals;
  bool converged = false;
  std::size_t used_iterations = 0;

  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    // Full reorthogonalization is deliberate: PAW overlap eigenvalues can be
    // tightly clustered, where a three-term recurrence otherwise develops
    // ghost roots.  A second pass controls loss of orthogonality.
    for (const auto& v : basis) q -= v.dot(q) * v;
    for (const auto& v : basis) q -= v.dot(q) * v;
    const double qnorm = q.norm();
    if (qnorm < 1e-14) break;
    q /= qnorm;
    basis.push_back(q);
    VC w = apply_a(q);
    if (iteration > 0) w -= beta_previous * previous;
    const double aa = std::real(q.dot(w));
    w -= aa * q;
    for (const auto& v : basis) w -= v.dot(w) * v;
    for (const auto& v : basis) w -= v.dot(w) * v;
    const double bb = w.norm();
    alpha.push_back(aa);
    beta.push_back(bb);
    const Eigen::Index m = static_cast<Eigen::Index>(basis.size());
    Eigen::MatrixXd t = Eigen::MatrixXd::Zero(m, m);
    for (Eigen::Index i = 0; i < m; ++i) {
      t(i, i) = alpha[static_cast<std::size_t>(i)];
      if (i + 1 < m) t(i, i + 1) = t(i + 1, i) = beta[static_cast<std::size_t>(i)];
    }
    tri_solver.compute(t);
    if (tri_solver.info() != Eigen::Success)
      throw std::runtime_error("complex Lanczos tridiagonal eigensolver failed");
    ritz_values = tri_solver.eigenvalues();
    ritz_vectors = tri_solver.eigenvectors();
    const std::size_t roots = std::min(n_roots, static_cast<std::size_t>(m));
    residuals.assign(roots, std::numeric_limits<double>::infinity());
    // The tridiagonal residual estimate is exact for an unrestarted Lanczos
    // recurrence.  We additionally compute the full residual below before
    // returning to protect against finite precision/reorthogonalization drift.
    if (bb == 0.0) {
      for (std::size_t root = 0; root < roots; ++root) residuals[root] = 0.0;
    } else {
      for (std::size_t root = 0; root < roots; ++root)
        residuals[root] = std::abs(bb * ritz_vectors(m - 1, static_cast<Eigen::Index>(root)));
    }
    used_iterations = iteration + 1;
    converged = roots == n_roots;
    for (double r : residuals) converged = converged && r <= tolerance;
    if (converged || bb < 1e-14) break;
    previous = q;
    q = w;
    beta_previous = bb;
  }

  const std::size_t roots = std::min(n_roots, static_cast<std::size_t>(ritz_values.size()));
  if (roots < n_roots)
    throw std::runtime_error("complex Lanczos terminated before requested roots were generated");
  Eigen::MatrixXcd qmatrix(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(basis.size()));
  for (std::size_t j = 0; j < basis.size(); ++j)
    qmatrix.col(static_cast<Eigen::Index>(j)) = basis[j];
  ComplexGeneralizedEigenResult result;
  result.dimension = dimension;
  result.iterations = used_iterations;
  result.converged = converged;
  result.eigenvalues.resize(roots);
  result.eigenvectors.assign(roots * dimension, C(0.0, 0.0));
  result.residual_norms.assign(roots, 0.0);
  for (std::size_t root = 0; root < roots; ++root) {
    result.eigenvalues[root] = ritz_values(static_cast<Eigen::Index>(root));
    const Eigen::VectorXd coeff = ritz_vectors.col(static_cast<Eigen::Index>(root));
    const VC y = qmatrix * coeff.cast<C>();
    const VC x = solve_upper(y);
    const VC residual = h * x - result.eigenvalues[root] * (s * x);
    const double snorm = std::sqrt(std::max(0.0, std::real(x.dot(s * x))));
    result.residual_norms[root] = residual.norm() / std::max(1.0, snorm);
    for (std::size_t component = 0; component < dimension; ++component)
      result.eigenvectors[root * dimension + component] = x(static_cast<Eigen::Index>(component));
  }
  result.converged = true;
  for (double r : result.residual_norms) result.converged = result.converged && r <= tolerance;
  return result;
#else
  throw std::runtime_error("Sparse-DFT complex iterative eigensolver requires Eigen3 at build time");
#endif
}

ComplexGeneralizedEigenResult solve_generalized_hermitian_davidson(
    const std::vector<std::complex<double>>& hamiltonian,
    const std::vector<std::complex<double>>& overlap,
    std::size_t dimension, std::size_t n_roots,
    std::size_t max_iterations, double tolerance,
    const std::vector<std::complex<double>>& initial_vectors) {
#ifdef SDFT_HAS_EIGEN
  using M=Eigen::MatrixXcd;
  const Eigen::Index n=static_cast<Eigen::Index>(dimension);
  if(!dimension||hamiltonian.size()!=dimension*dimension||overlap.size()!=dimension*dimension)
    throw std::invalid_argument("invalid dense Davidson dimensions");
  M h(n,n),s(n,n);
  for(Eigen::Index i=0;i<n;++i)for(Eigen::Index j=0;j<n;++j) {
    h(i,j)=hamiltonian[i*n+j];s(i,j)=overlap[i*n+j];
  }
  if(!h.allFinite()||!s.allFinite() || (h-h.adjoint()).norm()>1e-10*std::max(1.0,h.norm()) ||
      (s-s.adjoint()).norm()>1e-10*std::max(1.0,s.norm()))
    throw std::invalid_argument("Davidson requires finite Hermitian operators");
  h=(0.5*(h+h.adjoint())).eval();s=(0.5*(s+s.adjoint())).eval();
  HermitianOperator ho,so;ho.dimension=so.dimension=dimension;
  for(Eigen::Index i=0;i<n;++i){ho.diagonal.push_back(h(i,i).real());so.diagonal.push_back(s(i,i).real());}
  auto callback=[](const M& matrix, const std::vector<std::complex<double>>& x,
      std::vector<std::complex<double>>& y) {
    y.resize(x.size());
    Eigen::Map<const Eigen::VectorXcd> input(x.data(),x.size());
    Eigen::Map<Eigen::VectorXcd> output(y.data(),y.size());output.noalias()=matrix*input;
  };
  ho.apply=[&](const auto& x,auto& y){callback(h,x,y);};
  so.apply=[&](const auto& x,auto& y){callback(s,x,y);};
  return solve_generalized_hermitian_davidson(ho,so,n_roots,max_iterations,tolerance,initial_vectors);
#else
  throw std::runtime_error("Davidson requires bundled Eigen headers");
#endif
}

ComplexGeneralizedEigenResult solve_generalized_hermitian_davidson(
    const HermitianOperator& hamiltonian, const HermitianOperator& overlap,
    std::size_t n_roots, std::size_t max_iterations, double tolerance,
    const std::vector<std::complex<double>>& initial_vectors) {
#ifdef SDFT_HAS_EIGEN
  using C=std::complex<double>;using M=Eigen::MatrixXcd;using V=Eigen::VectorXcd;
  PerformanceTimer solve_timer("davidson_total");
  const std::size_t dimension=hamiltonian.dimension;
  const Eigen::Index n=static_cast<Eigen::Index>(dimension), roots=static_cast<Eigen::Index>(n_roots);
  M excluded,excluded_s,excluded_p;
  if(!dimension||!n_roots||n_roots>dimension||!max_iterations||overlap.dimension!=dimension||
      !hamiltonian.apply||!overlap.apply||hamiltonian.diagonal.size()!=dimension||overlap.diagonal.size()!=dimension||
      (!initial_vectors.empty()&&initial_vectors.size()!=dimension*n_roots)||
      !std::isfinite(tolerance)||tolerance<=0)
    throw std::invalid_argument("invalid matrix-free Davidson problem");
  for(std::size_t i=0;i<dimension;++i)
    if(!std::isfinite(hamiltonian.diagonal[i])||!std::isfinite(overlap.diagonal[i])||overlap.diagonal[i]<=0)
      throw std::invalid_argument("invalid Davidson operator diagonal");
  if(hamiltonian.initial_search_guards.size()%dimension ||
      hamiltonian.initial_search_guards.size()/dimension>n_roots)
    throw std::invalid_argument("invalid spectral search guard dimensions");
  for(const auto value:hamiltonian.initial_search_guards)
    if(!std::isfinite(value.real())||!std::isfinite(value.imag()))
      throw std::invalid_argument("nonfinite spectral search guard");
  if(overlap.indefinite_metric)
    return solve_generalized_hermitian_indefinite_davidson(hamiltonian,overlap,n_roots,max_iterations,tolerance,initial_vectors);
  const char* rmm_flag=std::getenv("SDFT_SCF_RMM_TRIAL");
  if(rmm_flag&&std::string(rmm_flag)!="0"&&std::string(rmm_flag)!="1")
    throw std::invalid_argument("SDFT_SCF_RMM_TRIAL must be 0 or 1");
  if(rmm_flag&&std::string(rmm_flag)=="1"&&hamiltonian.scf_trial_update&&
      !hamiltonian.band_group_local&&!initial_vectors.empty())
    return solve_generalized_hermitian_rmm_trial(hamiltonian,overlap,n_roots,max_iterations,tolerance,initial_vectors);
  const char* group_flag=std::getenv("SDFT_BAND_GROUP_SOLVER");
  if(group_flag&&std::string(group_flag)!="0"&&std::string(group_flag)!="1")throw std::invalid_argument("SDFT_BAND_GROUP_SOLVER must be 0 or 1");
  if(group_flag&&std::string(group_flag)=="1"&&!hamiltonian.band_group_local)
    return solve_generalized_hermitian_band_groups(hamiltonian,overlap,n_roots,max_iterations,tolerance,initial_vectors);
  auto apply=[&](const HermitianOperator& op,const V& x) -> V {
    std::vector<C> input(x.data(),x.data()+n),output;
    op.apply(input,output);
    if(output.size()!=dimension)throw std::runtime_error("Davidson callback dimension mismatch");
    V result=Eigen::Map<const V>(output.data(),n);
    if(!result.allFinite())throw std::runtime_error("nonfinite Davidson operator result");
    return result;
  };
  auto apply_s=[&](const V& x){return apply(overlap,x);};
  // Compensated FP64 dot products and fused subtraction avoid injecting the
  // normalization error into a near-machine-precision Rayleigh residual.
  const auto real_dot=[](const auto& a,const auto& b){
    double sum=0,correction=0;
    for(Eigen::Index i=0;i<a.size();++i)for(const double term:{a[i].real()*b[i].real(),a[i].imag()*b[i].imag()}){
      const double next=sum+term;
      correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;
    }
    return sum+correction;
  };
  const auto rayleigh_residual=[&](const V& v,const V& hv,const V& sv,double& value){
    const double metric=real_dot(v,sv);
    if(!(metric>0))throw std::runtime_error("invalid Davidson Rayleigh metric");
    value=real_dot(v,hv)/metric;V r(n);
    for(Eigen::Index i=0;i<n;++i)r[i]={std::fma(-value,sv[i].real(),hv[i].real()),std::fma(-value,sv[i].imag(),hv[i].imag())};
    return r;
  };
  bool exact_mode=compute_backend_name()=="cpu"&&!hamiltonian.approximate_search;
  const auto boolean_control=[](const char* name,bool fallback) {
    const char* flag=std::getenv(name);if(!flag)return fallback;
    if(std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument(std::string(name)+" must be 0 or 1");
    return std::string(flag)=="1";
  };
  // Real PAW A/B showed that fewer S applications can be outweighed by
  // additional search iterations. Preserve the measured original default;
  // keep reuse as an explicit, independently validated candidate control.
  const bool cached_metric_ortho=boolean_control("SDFT_CACHED_METRIC_ORTHO",false);
  const bool joint_hs=boolean_control("SDFT_JOINT_HS",true);
  const bool certify_once=boolean_control("SDFT_CERTIFY_ONCE",false)&&!hamiltonian.band_group_local;
  const bool restart_only_ritz=boolean_control("SDFT_RITZ_RESTART_ONLY",false);
  const bool adaptive_reorthogonalize=boolean_control("SDFT_ADAPTIVE_REORTHOGONALIZE",false);
  const bool gpu_factored_metric=boolean_control("SDFT_GPU_FACTORED_METRIC",false);
  const bool recycle_guards=boolean_control("SDFT_RECYCLE_SPECTRAL_GUARDS",false)&&
      !(hamiltonian.band_group_local&&hamiltonian.subspace_capacity);
  const bool fresh_residual_polish=boolean_control("SDFT_FRESH_RESIDUAL_POLISH",true);
  const bool root_manifold_refinement=boolean_control("SDFT_STRICT_ROOT_MANIFOLD",true);
  bool projector_state=boolean_control("SDFT_PROJECTOR_STATE",false)&&
      hamiltonian.pair_identity&&hamiltonian.pair_identity==overlap.pair_identity&&
      hamiltonian.projector_rows&&hamiltonian.projector_rows==overlap.projector_rows&&
      hamiltonian.project_into&&hamiltonian.apply_projected_into&&overlap.apply_projected_into;
  const std::string preconditioner=std::getenv("SDFT_PRECONDITIONER")?std::getenv("SDFT_PRECONDITIONER"):"auto";
  if(preconditioner!="auto"&&preconditioner!="diagonal"&&preconditioner!="kinetic")
    throw std::invalid_argument("SDFT_PRECONDITIONER must be auto, diagonal or kinetic");
  // Real T(G+k) provides a bounded positive inverse for both spectral
  // expansion and local descent. A shifted approximate H diagonal may be
  // indefinite and stagnate near PAW spectral/degeneracy crossings. Dense
  // operators without physical T keep their established diagonal path.
  // Honor explicit diagonal/kinetic choices; never alter H/S or tolerances.
  const bool kinetic_precondition=preconditioner=="kinetic"||
      (preconditioner=="auto"&&!hamiltonian.kinetic.empty());
  if(preconditioner=="auto"&&kinetic_precondition)
    record_performance("davidson_positive_kinetic_auto_selected",0);
  if(kinetic_precondition) {
    if(hamiltonian.kinetic.size()!=dimension)throw std::invalid_argument("kinetic preconditioner needs physical T(G+k)");
    for(double t:hamiltonian.kinetic)if(!std::isfinite(t)||t<0)throw std::invalid_argument("invalid physical kinetic diagonal");
  }
  const auto project=[&](const auto& x,bool exact)->M {
    M p(hamiltonian.projector_rows,x.cols());
    hamiltonian.project_into(x.data(),x.cols(),p.data(),exact);
    if(!p.allFinite())throw std::runtime_error("nonfinite persistent PAW projection");
    return p;
  };
  const bool low_rank_metric=boolean_control("SDFT_PROJECTOR_METRIC",false)&&overlap.projector_metric_into;
  const bool projected_pair=boolean_control("SDFT_PROJECTED_PAIR",false)&&hamiltonian.apply_projected_pair_into;
  if(low_rank_metric&&(!std::isfinite(overlap.metric_identity_scale)||overlap.metric_identity_scale<=0))
    throw std::invalid_argument("invalid PAW metric identity scale");
  const auto q_project=[&](const M& p)->M {
    M result(p.rows(),p.cols());overlap.projector_metric_into(p.data(),p.cols(),result.data());
    if(!result.allFinite())throw std::runtime_error("nonfinite PAW metric projector image");return result;
  };
  const auto apply_projected=[&](const HermitianOperator& op,const auto& x,const auto& p,bool exact)->M {
    M result(n,x.cols());op.apply_projected_into(x.data(),x.cols(),p.data(),result.data(),exact);
    if(!result.allFinite())throw std::runtime_error("nonfinite projected H/S result");
    record_performance(&op==&hamiltonian?"davidson_h_columns":"davidson_s_columns",0,x.cols());
    record_performance("davidson_reused_projector_columns",0,x.cols());return result;
  };
  auto block_apply=[&](const HermitianOperator& op,const auto& x,bool exact)->M {
    record_performance(&op==&hamiltonian?"davidson_h_columns":"davidson_s_columns",0,x.cols());
    if(op.apply_block_into && x.outerStride()==x.rows()) {
      M result(x.rows(),x.cols());op.apply_block_into(x.data(),x.cols(),result.data(),exact);
      if(!result.allFinite())throw std::runtime_error("nonfinite Davidson direct block result");
      record_performance("davidson_direct_block_apply",0);return result;
    }
    if(op.apply_block){
      std::vector<C> input(x.data(),x.data()+x.size()),output;
      op.apply_block(input,x.cols(),output,exact);
      if(output.size()!=static_cast<std::size_t>(x.size()))throw std::runtime_error("Davidson block callback dimension mismatch");
      M result=Eigen::Map<const M>(output.data(),x.rows(),x.cols());
      if(!result.allFinite())throw std::runtime_error("nonfinite Davidson block result");return result;
    }
    M result(x.rows(),x.cols());for(Eigen::Index i=0;i<x.cols();++i)result.col(i)=apply(op,x.col(i));return result;
  };
  auto multiply=[&](const auto& a,const auto& b,bool adjoint=false)->M {
    M result(adjoint?a.cols():a.rows(),b.cols());
    complex_gemm(adjoint?MatrixOp::Adjoint:MatrixOp::Normal,MatrixOp::Normal,result.rows(),result.cols(),b.rows(),
        a.data(),a.outerStride(),b.data(),b.outerStride(),result.data(),result.outerStride(),exact_mode);
    return result;
  };
  if(hamiltonian.excluded_vectors.size()%dimension)throw std::invalid_argument("invalid completed-band dimensions");
  const Eigen::Index excluded_count=hamiltonian.excluded_vectors.size()/dimension;
  if(excluded_count){
    excluded=Eigen::Map<const M>(hamiltonian.excluded_vectors.data(),n,excluded_count);
    const bool reuse_images=hamiltonian.excluded_image_identity &&
        hamiltonian.excluded_image_identity==overlap.pair_identity;
    if(reuse_images&&!hamiltonian.excluded_overlap_images.empty()) {
      if(hamiltonian.excluded_overlap_images.size()!=dimension*excluded_count)
        throw std::invalid_argument("invalid completed-band overlap image dimensions");
      excluded_s=Eigen::Map<const M>(hamiltonian.excluded_overlap_images.data(),n,excluded_count);
      if(!excluded_s.allFinite())throw std::invalid_argument("nonfinite completed-band overlap images");
      record_performance("band_group_saved_excluded_s_columns",0,excluded_count);
    } else excluded_s=block_apply(overlap,excluded,true);
    if(projector_state) {
      if(reuse_images&&!hamiltonian.excluded_projector_images.empty()) {
        if(hamiltonian.excluded_projector_images.size()!=hamiltonian.projector_rows*excluded_count)
          throw std::invalid_argument("invalid completed-band projector image dimensions");
        excluded_p=Eigen::Map<const M>(hamiltonian.excluded_projector_images.data(),hamiltonian.projector_rows,excluded_count);
        if(!excluded_p.allFinite())throw std::invalid_argument("nonfinite completed-band projector images");
        record_performance("band_group_saved_excluded_projection_columns",0,excluded_count);
      } else excluded_p=project(excluded,true);
    }
    const M gram=multiply(excluded,excluded_s,true);
    if((gram-M::Identity(excluded_count,excluded_count)).cwiseAbs().maxCoeff()>1e-10)
      throw std::invalid_argument("completed bands must be S orthonormal");
  }
  const auto remove_excluded=[&](M& vectors,M* p){
    if(!excluded_count)return;
    M coefficients(excluded_count,vectors.cols()),removal(n,vectors.cols());
    complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,excluded_count,vectors.cols(),n,
        excluded_s.data(),n,vectors.data(),n,coefficients.data(),excluded_count,true);
    complex_gemm(MatrixOp::Normal,MatrixOp::Normal,n,vectors.cols(),excluded_count,
        excluded.data(),n,coefficients.data(),excluded_count,removal.data(),n,true);
    vectors-=removal;
    if(p){M removed_p(p->rows(),p->cols());
      complex_gemm(MatrixOp::Normal,MatrixOp::Normal,p->rows(),p->cols(),excluded_count,
          excluded_p.data(),excluded_p.rows(),coefficients.data(),excluded_count,removed_p.data(),p->rows(),true);*p-=removed_p;}
  };
  const auto block_apply_pair=[&](const auto& input,M& hx,M& sx,bool exact) {
    if(joint_hs&&hamiltonian.pair_identity&&
        hamiltonian.pair_identity==overlap.pair_identity&&hamiltonian.apply_pair_into&&
        input.outerStride()==input.rows()) {
      hx.resize(input.rows(),input.cols());sx.resize(input.rows(),input.cols());
      hamiltonian.apply_pair_into(input.data(),input.cols(),hx.data(),sx.data(),exact);
      if(!hx.allFinite()||!sx.allFinite())throw std::runtime_error("nonfinite Davidson joint H/S result");
      record_performance("davidson_h_columns",0,input.cols());
      record_performance("davidson_s_columns",0,input.cols());
      record_performance("davidson_joint_hs_calls",0);
    } else {
      sx=block_apply(overlap,input,exact);hx=block_apply(hamiltonian,input,exact);
    }
  };
  if(projector_state&&tolerance<1e-9&&!initial_vectors.empty()){
    const M warm=Eigen::Map<const M>(initial_vectors.data(),n,roots);M wh,ws;
    block_apply_pair(warm,wh,ws,true);double maximum=0;
    for(Eigen::Index j=0;j<roots;++j){double value=0;maximum=std::max(maximum,rayleigh_residual(warm.col(j),wh.col(j),ws.col(j),value).norm());}
    if(maximum<=1e-8){projector_state=false;record_performance("davidson_projector_strict_warm_bypass",0,roots);}
  }
  if(hamiltonian.subspace_capacity&&hamiltonian.subspace_capacity<std::min(dimension,2*n_roots))
    throw std::invalid_argument("Davidson local subspace must hold roots and guard states");
  Eigen::Index subspace_factor=8;
  if(const char* setting=std::getenv("SDFT_DAVIDSON_SUBSPACE_FACTOR")) {
    const std::string value(setting);std::size_t end=0;
    if(value.empty()||value[0]=='-')throw std::invalid_argument("invalid Davidson subspace factor");
    const auto factor=std::stoull(value,&end);
    // Warm roots plus 2*NBANDS sector guards must fit, with one correction block.
    if(end!=value.size()||factor<4||factor>16)throw std::invalid_argument("SDFT_DAVIDSON_SUBSPACE_FACTOR must be in [4,16]");
    subspace_factor=factor;
  }
  const bool reuse_restart_images=boolean_control("SDFT_RESTART_IMAGES",false);
  bool rotated_images=false;
  const Eigen::Index capacity=std::min(n,hamiltonian.subspace_capacity?Eigen::Index(hamiltonian.subspace_capacity):std::max<Eigen::Index>(subspace_factor*roots,64));
  record_performance("davidson_search_capacity",0,capacity);
  M basis=M::Zero(n,capacity),hb=M::Zero(n,capacity),sb=M::Zero(n,capacity);
  M pb=M::Zero(projector_state?hamiltonian.projector_rows:0,capacity);
  Eigen::Index size=0;
  Eigen::Index computed_h=0;
  Eigen::Index projected_size=0;
  M cached_h=M::Zero(capacity,capacity),cached_s=M::Zero(capacity,capacity);
  // The final FP64 lane requests residuals comparable to the backward
  // error of a projected solve. Building Galerkin slabs incrementally
  // carries search-history rounding into the invariant manifold. Start
  // strict solves with the complete unchanged H/S projection; preserve
  // incremental reuse for the looser search lane and explicit controls.
  bool incremental_ritz=tolerance>=1e-12;
  bool strict_projection_refinement=false;
  // Local descent is a rounding-floor aid, not the spectral search. Bound
  // all refinement attempts over this solve, including rejected ones, and
  // let Davidson expand between residual levels. A slightly improved local
  // vector must not repeatedly replace the global Ritz search.
  unsigned strict_polish_attempts=0,strict_manifold_attempts=0;
  double strict_refinement_entry=std::numeric_limits<double>::infinity();
  if(const char* flag=std::getenv("SDFT_INCREMENTAL_RITZ")) {
    if(std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("SDFT_INCREMENTAL_RITZ must be 0 or 1");
    incremental_ritz=std::string(flag)=="1";
  }
  const auto update_projected=[&](const M& applied,M& cached) {
    const Eigen::Index first=incremental_ritz?projected_size:0;
    if(first==size)return;
    const auto columns=size-first;
    // Old basis columns and their original H/S images are immutable until a
    // restart or precision transition. Recompute only the new slabs. Compute
    // both off-diagonal directions before Hermitian averaging, preserving
    // the original finite-precision Galerkin convention.
    complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,size,columns,n,
        basis.data(),basis.outerStride(),applied.data()+first*n,applied.outerStride(),
        cached.data()+first*cached.outerStride(),cached.outerStride(),exact_mode);
    if(first)complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,columns,first,n,
        basis.data()+first*n,basis.outerStride(),applied.data(),applied.outerStride(),
        cached.data()+first,cached.outerStride(),exact_mode);
  };
  auto flush_h=[&](){if(computed_h<size){
    const auto b=basis.middleCols(computed_h,size-computed_h);
    const char* chain=cached_getenv("SDFT_GPU_PAW_CHAIN");
    if(chain&&std::string(chain)=="1"&&!exact_mode&&!projector_state){
      M hy,sy;block_apply_pair(b,hy,sy,false);
      hb.middleCols(computed_h,size-computed_h)=hy;sb.middleCols(computed_h,size-computed_h)=sy;
    }else hb.middleCols(computed_h,size-computed_h)=projector_state?
        apply_projected(hamiltonian,b,pb.middleCols(computed_h,size-computed_h),exact_mode):block_apply(hamiltonian,b,exact_mode);
    computed_h=size;
  }};
  auto append=[&](V v){
    if(size>=capacity)return false;
    // Residual directions shrink with convergence. An absolute rejection
    // threshold before scaling incorrectly discards the corrections needed
    // for a strict solve. Reject linear dependence after unit scaling instead.
    const double euclidean_norm=v.norm();
    if(!std::isfinite(euclidean_norm))throw std::runtime_error("nonfinite Davidson direction");
    if(euclidean_norm==0)return false;
    v/=euclidean_norm;
    V sv;M pv;
    if(projector_state)pv=project(M(v),true);
    const auto fresh_s=[&]()->V{return projector_state?V(apply_projected(overlap,M(v),pv,true)):apply_s(v);};
    if(!cached_metric_ortho)sv=fresh_s();
    for(int pass=0;pass<2;++pass) {
      if(excluded_count){M vv=v;remove_excluded(vv,projector_state?&pv:nullptr);v=vv.col(0);if(!cached_metric_ortho)sv=fresh_s();}
      if(cached_metric_ortho) {
        // S is Hermitian and the existing basis/S images are immutable in
        // this solve: (S*B)^H v == B^H(S*v). CGS2 therefore needs no new
        // PAW application between passes. Recompute the final S*v below;
        // never propagate a cancelled S image into the normalization.
        if(size) {
          V projections(size),removal(n);
          complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,size,1,n,sb.data(),sb.outerStride(),v.data(),n,projections.data(),size,true);
          complex_gemm(MatrixOp::Normal,MatrixOp::Normal,n,1,size,basis.data(),basis.outerStride(),projections.data(),size,removal.data(),n,true);
          v-=removal;
          if(projector_state)pv-=multiply(pb.leftCols(size),M(projections));
        }
        continue;
      }
      if(pass)sv=fresh_s();
      if(size<8){for(Eigen::Index j=0;j<size;++j){const C projection=basis.col(j).dot(sv);v-=projection*basis.col(j);sv-=projection*sb.col(j);if(projector_state)pv.col(0)-=projection*pb.col(j);}}
      else {
        // CGS2: two reorthogonalization passes, each using BLAS matrix/vector
        // blocks. Recompute S*v between passes and after cancellation.
        V projections(size);
        complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,size,1,n,basis.data(),basis.outerStride(),sv.data(),n,projections.data(),size,true);
        V removal(n);
        complex_gemm(MatrixOp::Normal,MatrixOp::Normal,n,1,size,basis.data(),basis.outerStride(),projections.data(),size,removal.data(),n,true);
        v-=removal;
        if(projector_state)pv-=multiply(pb.leftCols(size),M(projections));
      }
    }
    // Recompute S*v after cancellation. Algebraically updating an old S*v
    // loses relative precision for nearly dependent directions and can
    // invent an indefinite Rayleigh-Ritz metric during a warm start.
    // Cancellation invalidates transformed amplitudes, even though the
    // transform is algebraically exact. Reproject before rank revelation.
    if(projector_state)pv=project(M(v),true);
    sv=fresh_s();
    const double norm2=std::real(v.dot(sv));
    if(!std::isfinite(norm2)||norm2 < -1e-10)
      throw std::runtime_error("Davidson encountered non-positive overlap direction");
    if(norm2<1e-24)return false;
    v/=std::sqrt(norm2);sv/=std::sqrt(norm2);
    basis.col(size)=v;sb.col(size)=sv;
    if(projector_state)pb.col(size)=pv.col(0)/std::sqrt(norm2);
    ++size;return true;
  };
  // Block CGS2 is the default: the sparsity sweep (Si2..Ni, Al slab) gave
  // results unchanged to the certificate with 7-10x fewer S applications,
  // and it removes the per-vector serial GEMV chain. SDFT_BLOCK_ORTHOGONALIZE=0
  // or SDFT_DETERMINISTIC=1 restores one-vector CGS2.
  const bool block_orthogonalize=[]{
    const char* flag=std::getenv("SDFT_BLOCK_ORTHOGONALIZE");
    if(flag)return std::string(flag)=="1";
    return !deterministic_mode();
  }();
  // On a GPU backend the inexact-search CGS2 GEMMs go to the device by
  // default (Si64 replay 53.0 s -> 49.8 s, certificate unchanged); exact
  // (certification) orthogonalization always stays in FP64 on the CPU.
  const bool gpu_orthogonalize=[]{
    const char* flag=std::getenv("SDFT_GPU_BLOCK_ORTHO");
    if(flag)return std::string(flag)=="1";
    return !deterministic_mode()&&compute_backend_name()!="cpu";
  }();
  const auto append_block=[&](M vectors,const M* supplied_projection=nullptr){
    const Eigen::Index before=size;
    if(vectors.cols()>capacity-size)vectors.conservativeResize(n,capacity-size);
    if(!vectors.cols())return false;
    const auto serial=[&](){for(Eigen::Index j=0;j<vectors.cols();++j)append(vectors.col(j));return size>before;};
    if(!block_orthogonalize||vectors.cols()==1)return serial();
    PerformanceTimer timer("davidson_block_orthogonalization");
    const bool exact_orthogonalize=exact_mode || !gpu_orthogonalize;
    M pv;
    if(projector_state)pv=supplied_projection?M(supplied_projection->leftCols(vectors.cols())):project(vectors,exact_orthogonalize);
    for(Eigen::Index j=0;j<vectors.cols();++j){const double length=vectors.col(j).norm();
      if(!(length>0)||!std::isfinite(length))return serial();vectors.col(j)/=length;if(projector_state)pv.col(j)/=length;}
    const M original=vectors;
    M metric_vectors;
    const auto metric_apply=[&](){return projector_state?apply_projected(overlap,vectors,pv,exact_orthogonalize):block_apply(overlap,vectors,exact_orthogonalize);};
    const bool factored_metric=projector_state&&low_rank_metric;
    if(!cached_metric_ortho&&!factored_metric)metric_vectors=metric_apply();
    // DGKS norm-loss criterion in the actual positive PAW S metric, not the
    // Euclidean norm. It only removes a redundant second pass in FP64 when
    // every direction retains at least 1/sqrt(2) of its original norm.
    // Excluded-band local solves and approximate GPU passes keep full CGS2.
    const bool dgks=adaptive_reorthogonalize && exact_orthogonalize && !excluded_count;
    const auto metric_norms=[&]() {
      Eigen::VectorXd norms(vectors.cols());
      if(factored_metric) {
        const M qp=q_project(pv);
        for(Eigen::Index j=0;j<vectors.cols();++j)
          norms[j]=overlap.metric_identity_scale*real_dot(vectors.col(j),vectors.col(j))+
              real_dot(pv.col(j),qp.col(j));
      }else {
        if(cached_metric_ortho)metric_vectors=metric_apply();
        for(Eigen::Index j=0;j<vectors.cols();++j)norms[j]=real_dot(vectors.col(j),metric_vectors.col(j));
      }
      return norms;
    };
    const Eigen::VectorXd original_metric_norms=dgks?metric_norms():Eigen::VectorXd{};
    // An empty basis has no projections to remove. Avoid recomputing S*V
    // twice on initial seeds/restarts; retain both fresh passes otherwise.
    for(int pass=0;(size||excluded_count) && pass<2;++pass){
      if(pass==1 && dgks) {
        const auto remaining=metric_norms();
        bool stable=remaining.allFinite()&&original_metric_norms.allFinite();
        for(Eigen::Index j=0;j<remaining.size();++j)
          stable &= original_metric_norms[j]>0 && remaining[j]>=.5*original_metric_norms[j];
        // Near-cancelled amplitudes always require the existing fresh
        // projection and both passes, even if a metric dot is optimistic.
        stable &= vectors.colwise().norm().minCoeff()>=1e-3;
        if(stable){record_performance("davidson_dgks_second_pass_saved",0,vectors.cols());break;}
        record_performance("davidson_dgks_second_pass_required",0,vectors.cols());
      }
      remove_excluded(vectors,projector_state?&pv:nullptr);
      if(excluded_count&&!cached_metric_ortho&&!factored_metric)metric_vectors=metric_apply();
      if(size){
        M projections(size,vectors.cols()),removal(n,vectors.cols());
        if(factored_metric){
          // The large Euclidean part may use the same search precision as
          // the other block-orthogonalization products. The small PAW Q
          // contribution stays FP64. The original operators are reapplied
          // in FP64 at the existing refinement/certification boundary.
          // Never enable an approximate pass during exact refinement.
          const bool exact_projection=exact_orthogonalize||!gpu_factored_metric;
          complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,size,vectors.cols(),n,basis.data(),basis.outerStride(),vectors.data(),n,projections.data(),size,exact_projection);
          record_performance(exact_projection?"davidson_factored_projection_fp64":"davidson_factored_projection_search",0,vectors.cols());
          projections*=overlap.metric_identity_scale;
          const M qp=q_project(pv);
          complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,size,vectors.cols(),pb.rows(),pb.data(),pb.outerStride(),qp.data(),qp.rows(),projections.data(),size,true,true);
          record_performance("davidson_low_rank_metric_saved_s_columns",0,vectors.cols());
        }else{
          const auto* left=cached_metric_ortho?sb.data():basis.data();
          const auto* right=cached_metric_ortho?vectors.data():metric_vectors.data();
          complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,size,vectors.cols(),n,left,basis.outerStride(),right,n,projections.data(),size,exact_orthogonalize);
        }
        complex_gemm(MatrixOp::Normal,MatrixOp::Normal,n,vectors.cols(),size,basis.data(),basis.outerStride(),projections.data(),size,removal.data(),n,exact_orthogonalize);
        vectors-=removal;
        if(projector_state) {
          M projected_removal(pb.rows(),vectors.cols());
          complex_gemm(MatrixOp::Normal,MatrixOp::Normal,pb.rows(),vectors.cols(),size,
              pb.data(),pb.outerStride(),projections.data(),size,projected_removal.data(),pb.rows(),exact_orthogonalize);
          pv-=projected_removal;
          // Reuse is safe for well-scaled directions. Near cancellation
          // requires new B*V, never an address-based cache hit.
          if(vectors.colwise().norm().minCoeff()<1e-3){pv=project(vectors,exact_orthogonalize);record_performance("davidson_projector_cancellation_refresh",0,vectors.cols());}
        }
      }
      if(!cached_metric_ortho&&!factored_metric)metric_vectors=metric_apply();
    }
    if(cached_metric_ortho&&!factored_metric) {
      metric_vectors=metric_apply();
      record_performance("davidson_cached_metric_orthogonalization",0,vectors.cols());
    }
    M gram(vectors.cols(),vectors.cols());
    if(factored_metric){
      gram=overlap.metric_identity_scale*multiply(vectors,vectors,true);
      const M qp=q_project(pv);gram+=multiply(pv,qp,true);
      record_performance("davidson_low_rank_metric_saved_s_columns",0,vectors.cols());
    }else complex_gemm(MatrixOp::Adjoint,MatrixOp::Normal,vectors.cols(),vectors.cols(),n,vectors.data(),n,metric_vectors.data(),n,gram.data(),gram.rows(),exact_orthogonalize);
    gram=(.5*(gram+gram.adjoint())).eval();
    Eigen::SelfAdjointEigenSolver<M> metric_solver(gram);
    if(metric_solver.info()!=Eigen::Success)throw std::runtime_error("block Davidson metric decomposition failed");
    double largest=metric_solver.eigenvalues().maxCoeff(),smallest=metric_solver.eigenvalues().minCoeff();
    if(projector_state&&(smallest < -1e-10||!(largest>0)||smallest<1e-10*largest)){
      // A transformed projection is an optimization, never evidence that
      // the physical overlap became indefinite. Recompute with the actual
      // original FP64 operator before rank decisions or error reporting.
      metric_vectors=block_apply(overlap,vectors,true);
      gram=multiply(vectors,metric_vectors,true);gram=(.5*(gram+gram.adjoint())).eval();
      metric_solver.compute(gram);
      if(metric_solver.info()!=Eigen::Success)throw std::runtime_error("fresh PAW metric decomposition failed");
      largest=metric_solver.eigenvalues().maxCoeff();smallest=metric_solver.eigenvalues().minCoeff();
      pv=project(vectors,true);record_performance("davidson_projector_rank_refresh",0,vectors.cols());
    }
    if(smallest < -1e-10)throw std::runtime_error("block Davidson encountered non-positive overlap");
    // Ill-conditioned correction blocks need sequential rank revelation to
    // avoid squaring their condition number in the small Gram matrix. Keep
    // every valid correction rather than discarding it via a loose cutoff.
    if(!(largest>0)||smallest<1e-10*largest){vectors=original;record_performance("davidson_block_rank_fallback",0);return serial();}
    const M normalization=metric_solver.eigenvectors()*metric_solver.eigenvalues().array().sqrt().inverse().matrix().asDiagonal();
    M normalized(n,vectors.cols()),normalized_s(n,vectors.cols());
    complex_gemm(MatrixOp::Normal,MatrixOp::Normal,n,vectors.cols(),vectors.cols(),vectors.data(),n,normalization.data(),normalization.rows(),normalized.data(),n,exact_orthogonalize);
    if(!factored_metric)complex_gemm(MatrixOp::Normal,MatrixOp::Normal,n,vectors.cols(),vectors.cols(),metric_vectors.data(),n,normalization.data(),normalization.rows(),normalized_s.data(),n,exact_orthogonalize);
    if(projector_state) {
      M normalized_p(pb.rows(),vectors.cols());
      complex_gemm(MatrixOp::Normal,MatrixOp::Normal,pb.rows(),vectors.cols(),vectors.cols(),pv.data(),pv.rows(),normalization.data(),normalization.rows(),normalized_p.data(),pb.rows(),exact_orthogonalize);
      pb.middleCols(size,vectors.cols())=normalized_p;
      if(smallest<1e-4*largest){
        pb.middleCols(size,vectors.cols())=project(normalized,true);
        normalized_s=apply_projected(overlap,normalized,pb.middleCols(size,vectors.cols()),true);
        record_performance("davidson_projector_normalization_refresh",0,vectors.cols());
      }
      if(projected_pair&&computed_h==size&&(exact_mode||!factored_metric)) {
        // Every old H image is already valid. Store both images for these
        // normalized new columns, sharing their solver-owned amplitudes.
        M normalized_h(n,vectors.cols());
        hamiltonian.apply_projected_pair_into(normalized.data(),vectors.cols(),pb.data()+size*pb.outerStride(),
            normalized_h.data(),normalized_s.data(),exact_orthogonalize);
        if(!normalized_h.allFinite()||!normalized_s.allFinite())throw std::runtime_error("nonfinite paired projected H/S");
        hb.middleCols(size,vectors.cols())=normalized_h;computed_h=size+vectors.cols();
        record_performance("davidson_h_columns",0,vectors.cols());record_performance("davidson_s_columns",0,vectors.cols());
        record_performance("davidson_projected_pair_columns",0,vectors.cols());
      }else if(factored_metric)normalized_s=apply_projected(overlap,normalized,pb.middleCols(size,vectors.cols()),true);
    }
    basis.middleCols(size,vectors.cols())=normalized;sb.middleCols(size,vectors.cols())=normalized_s;size+=vectors.cols();return true;
  };
  // Canonical seeds cover distinct kinetic states, including degenerate
  // partners. Unlike scalar Lanczos, a block can represent their multiplicity.
  std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
  for(Eigen::Index i=0;i<n;++i){
    if(overlap.diagonal[i]<=0)throw std::runtime_error("non-positive Davidson overlap diagonal");
    order[static_cast<std::size_t>(i)]=i;
  }
  std::stable_sort(order.begin(),order.end(),[&](auto i,auto j){return hamiltonian.diagonal[i]/overlap.diagonal[i]<hamiltonian.diagonal[j]/overlap.diagonal[j];});
  // Canonical low-diagonal seeds may lie entirely inside an invariant
  // symmetry sector. Its Ritz residual can be zero even when a lower band
  // exists in the orthogonal sector (observed on Si W-L). Add a reproducible
  // full-support block so every sector can enter the search, including when
  // the supplied warm start already consists of exact, but wrong, roots.
  std::uint64_t seed_state=0x6a09e667f3bcc909ULL;
  auto signed_uniform=[&]() {
    std::uint64_t z=(seed_state+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    z^=z>>31;
    return static_cast<double>(z>>11)*0x1.0p-53-.5;
  };
  auto broad_seed=[&]() -> V {
    V seed(n);
    for(Eigen::Index i=0;i<n;++i) {
      const double re=signed_uniform(), im=signed_uniform();
      seed[i]=C(re,im);
    }
    seed.normalize();
    return seed;
  };
  // Pure canonical/exact warm vectors can themselves converge before the
  // broad block has discovered the missing sector. Give every seed full
  // support, rather than just appending unrelated random guard vectors.
  // Perturbations affect initial guesses only; H/S and residual thresholds
  // are unchanged, and the final eigenvectors are recomputed normally.
  std::vector<V> seeds;
  // A fixed 1e-6 perturbation forces an already accurate late-SCF subspace
  // back through many refinement iterations. Full support is still required
  // to reveal missing symmetry sectors: keep its amplitude above the solve
  // tolerance, without contaminating a strict warm start unnecessarily.
  double warm_perturbation=std::clamp(100.*tolerance,1e-10,1e-6);
  if(const char* setting=std::getenv("SDFT_WARM_START_PERTURBATION")){
    std::size_t end=0;const std::string value(setting);warm_perturbation=std::stod(value,&end);
    if(end!=value.size() || !std::isfinite(warm_perturbation) || warm_perturbation<=0 || warm_perturbation>.1)
      throw std::invalid_argument("SDFT_WARM_START_PERTURBATION must be finite and in (0,0.1]");
  }
  const bool warm_start=!initial_vectors.empty() && 2*roots<n;
  if(hamiltonian.local_seed_only&&
      (!hamiltonian.band_group_local||!hamiltonian.subspace_capacity||!warm_start))
    throw std::invalid_argument("seed-only update requires a bounded local warm search");
  if(warm_start)for(Eigen::Index j=0;j<roots;++j) {
    V seed(n);
    for(Eigen::Index i=0;i<n;++i)seed[i]=initial_vectors[j*dimension+i];
    if(!seed.allFinite())throw std::invalid_argument("nonfinite Davidson initial subspace");
    seed+=warm_perturbation*seed.norm()*broad_seed();
    seeds.push_back(std::move(seed));
  }
  Eigen::Index guard_count=std::min(n,2*roots);
  if(warm_start) {
    // Keep the original 2*NBANDS guard default. Real PAW A/B found the
    // reduced guard block increased iterations enough to lose wall time.
    // The bounded explicit control remains for material-specific studies.
    if(const char* flag=std::getenv("SDFT_WARM_GUARD_VECTORS")) {
      const std::string setting(flag);std::size_t end=0;
      if(setting.empty()||setting[0]=='-')throw std::invalid_argument("invalid SDFT_WARM_GUARD_VECTORS");
      const auto value=std::stoull(setting,&end);
      if(end!=setting.size()||value<1||value>static_cast<std::size_t>(hamiltonian.band_group_local?n:std::min(n,2*roots)))
        throw std::invalid_argument("SDFT_WARM_GUARD_VECTORS must be in [1,min(NPW,2*NBANDS)]");
      guard_count=std::min<Eigen::Index>(value,2*roots);
    }
  }
  if(hamiltonian.local_seed_only) {
    // These steps only improve a supplied manifold. Full-spectrum safety
    // belongs to the original global guarded solve in band_group_solver.
    // Repeating low canonical guards in every local group reintroduced
    // already excluded low bands and wasted projection/orthogonalization.
    record_performance("band_group_predictor_guard_columns_saved",0,guard_count);
    guard_count=0;
  }
  record_performance(warm_start?"davidson_warm_seed_columns":"davidson_cold_seed_columns",0,seeds.size()+guard_count);
  // Keep the same guard count and full-support perturbations. Approximate
  // upper Ritz states replace at most half of the original canonical guards;
  // the remaining canonical/broad seeds still search missing symmetry sectors.
  // Every state is normalized, orthogonalized and applied to the NEW H/S.
  // Retaining old operator images across SCF iterations would be incorrect.
  const Eigen::Index resident_count=warm_start&&recycle_guards ?
      std::min<Eigen::Index>(guard_count/2,hamiltonian.initial_search_guards.size()/dimension):0;
  for(Eigen::Index j=0;j<guard_count && size<capacity;++j) {
    V seed;
    if(j<resident_count) {
      seed=Eigen::Map<const V>(hamiltonian.initial_search_guards.data()+j*dimension,n);
      const double norm=seed.norm();
      if(!(norm>0)||!std::isfinite(norm))throw std::invalid_argument("zero or invalid spectral search guard");
      seed+=warm_perturbation*norm*broad_seed();
    } else {seed=.05*broad_seed();seed[order[j-resident_count]]+=1.;}
    seeds.push_back(std::move(seed));
  }
  record_performance("davidson_resident_spectral_guard_columns",0,resident_count);
  M seed_block(n,seeds.size());for(std::size_t j=0;j<seeds.size();++j)seed_block.col(j)=seeds[j];append_block(std::move(seed_block));
  if(hamiltonian.local_seed_only&&size<roots) {
    // Invalid/rank-deficient warm manifolds are guesses, not guarantees.
    // Supply only missing directions so the short update remains defined;
    // no success here can bypass the mandatory global guarded solve.
    for(Eigen::Index j=0;j<n&&size<roots;++j) {
      if(append(broad_seed()))record_performance("band_group_predictor_rank_repair",0);
    }
    if(size<roots)throw std::runtime_error("warm predictor could not form an independent local manifold");
  }
  ComplexGeneralizedEigenResult out;out.dimension=dimension;
  const bool triple_rotation=boolean_control("SDFT_RITZ_TRIPLE_ROTATION",false);
  M x,sx,residual,final_ritz_vectors;Eigen::VectorXd values;bool certified=false;
  const auto certify_roots=[&](){
    PerformanceTimer timer("davidson_fp64_certification");
    // Certify the original operators in FP64 as a band block. Scalar calls
    // turn PAW GEMM into GEMV and prevent the FFT block/thread path from
    // sharing work; batching changes neither H/S nor the residual threshold.
    M hx;
    if(certify_once)block_apply_pair(x,hx,sx,true);
    else sx=block_apply(overlap,x,true);
    bool renormalize=!certify_once;
    if(certify_once)for(Eigen::Index root=0;root<roots;++root)
      renormalize |= std::abs(real_dot(x.col(root),sx.col(root))-1.)>1e-13;
    for(Eigen::Index root=0;root<roots;++root){
      const double norm2=std::real(x.col(root).dot(sx.col(root)));
      if(!std::isfinite(norm2)||!(norm2>0))throw std::runtime_error("invalid Davidson Ritz S norm");
      if(renormalize)x.col(root)/=std::sqrt(norm2);
    }
    if(renormalize)block_apply_pair(x,hx,sx,true);
    else record_performance("davidson_certification_saved_overlap_columns",0,roots);
    residual.resize(n,roots);out.converged=true;
    for(Eigen::Index root=0;root<roots;++root){
      residual.col(root)=rayleigh_residual(x.col(root),hx.col(root),sx.col(root),values[root]);
      out.residual_norms[root]=residual.col(root).norm();
      out.converged &= out.residual_norms[root]<=tolerance;
    }
  };
  for(std::size_t iteration=0;iteration<max_iterations;++iteration){
    flush_h();
    {
      PerformanceTimer timer("davidson_incremental_galerkin");
      update_projected(hb,cached_h);update_projected(sb,cached_s);
      projected_size=size;
    }
    M projected=cached_h.topLeftCorner(size,size);
    projected=(0.5*(projected+projected.adjoint())).eval();
    // Do not assume the computed S-orthogonal basis has an exactly identity
    // Gram matrix. Near-degenerate PAW roots amplify small Gram errors into
    // occupation-density noise even when individual residuals are tiny.
    M metric=cached_s.topLeftCorner(size,size);
    metric=(0.5*(metric+metric.adjoint())).eval();
    PerformanceTimer iteration_timer("davidson_iteration");
    // Extra empty Ritz vectors are only consumed by a thick restart. Solve
    // for them at that boundary rather than on every search iteration.
    const Eigen::Index ritz_roots=restart_only_ritz && size+roots<=capacity ? roots : std::min(size,2*roots);
    const auto rr=[&](){PerformanceTimer timer("rayleigh_ritz");return rayleigh_ritz(projected,metric,ritz_roots,strict_projection_refinement);}();
    values=rr.values.head(roots);
    const M rotation=rr.vectors.leftCols(roots);
    if(triple_rotation) {
      PerformanceTimer timer("davidson_triple_ritz_rotation");
      x.resize(n,roots);sx.resize(n,roots);residual.resize(n,roots);
      complex_gemm_triple_right(n,roots,size,{basis.data(),sb.data(),hb.data()},basis.outerStride(),
          rotation.data(),rotation.outerStride(),{x.data(),sx.data(),residual.data()},n,exact_mode);
      residual-=sx*values.asDiagonal();
      record_performance("davidson_triple_ritz_products",0,3);
    }else {
      x=multiply(basis.leftCols(size),rotation);sx=multiply(sb.leftCols(size),rotation);
      residual=multiply(hb.leftCols(size),rotation)-sx*values.asDiagonal();
    }
    out.residual_norms.resize(n_roots);out.converged=true;out.iterations=iteration+1;
    for(Eigen::Index root=0;root<roots;++root){out.residual_norms[root]=residual.col(root).norm();out.converged &= out.residual_norms[root]<=tolerance;}
    if(!exact_mode&&(*std::max_element(out.residual_norms.begin(),out.residual_norms.end())<1e-8||iteration+1==max_iterations)){
      // Approximate GPU arithmetic is only a search accelerator. Refresh the
      // entire subspace with original FP64 H/S before accepting strict roots.
      exact_mode=true;record_performance("fp64_subspace_refresh",0);
      if(projector_state&&tolerance<1e-9)projector_state=false;
      projected_size=0;
      M refreshed_h,refreshed_s;
      block_apply_pair(basis.leftCols(size),refreshed_h,refreshed_s,true);
      if(projector_state)pb.leftCols(size)=project(basis.leftCols(size),true);
      hb.leftCols(size)=refreshed_h;sb.leftCols(size)=refreshed_s;computed_h=size;
      rotated_images=false;
      continue;
    }
    if(projector_state&&tolerance<1e-9&&*std::max_element(out.residual_norms.begin(),out.residual_norms.end())<=1e-8&&
        *std::max_element(out.residual_norms.begin(),out.residual_norms.end())>8*tolerance){
      // Rotate/project reuse is useful during search, but not at the FP64
      // rounding floor. Hand over before its error becomes the residual.
      projector_state=false;M fresh_h,fresh_s;
      block_apply_pair(basis.leftCols(size),fresh_h,fresh_s,true);
      hb.leftCols(size)=fresh_h;sb.leftCols(size)=fresh_s;
      rotated_images=false;
      projected_size=0;computed_h=size;
      record_performance("davidson_projector_refinement_handoff",0,size);continue;
    }
    bool roots_freshly_checked=false;
    if(exact_mode&&*std::max_element(out.residual_norms.begin(),out.residual_norms.end())<=8*tolerance) {
      // Cached H*B rotated into the Ritz space has a different rounding
      // error from a fresh FFT/low-rank application. Accept the actual
      // generalized residual, and refine it if it still exceeds tolerance.
      certify_roots();
      roots_freshly_checked=true;
      if(!out.converged&&(projector_state||rotated_images)){
        // At the final rounding floor, a rotated B*psi can no longer be
        // distinguished reliably from a fresh projection. Retire the reuse
        // state for this solve and refresh the unchanged original operators.
        // This is strict refinement, not a relaxed convergence gate.
        projector_state=false;M fresh_h,fresh_s;
        block_apply_pair(basis.leftCols(size),fresh_h,fresh_s,true);
        hb.leftCols(size)=fresh_h;sb.leftCols(size)=fresh_s;
        rotated_images=false;
        projected_size=0;computed_h=size;
        record_performance("davidson_projector_final_precision_refresh",0,size);
        continue;
      }
    }
    if(out.converged&&!roots_freshly_checked) {
      // A coarse request can be met before the fixed GPU->FP64 transition.
      // A cached/rotated approximate residual is not an original residual.
      // Check only the returned roots first, avoiding a full refresh when
      // their actual requested FP64 H/S tolerance already holds.
      certify_roots();roots_freshly_checked=true;
      record_performance("davidson_coarse_original_residual_checked",0,roots);
      if(!out.converged&&!exact_mode) {
        exact_mode=true;
        if(projector_state&&tolerance<1e-9)projector_state=false;
        M fresh_h,fresh_s;block_apply_pair(basis.leftCols(size),fresh_h,fresh_s,true);
        if(projector_state)pb.leftCols(size)=project(basis.leftCols(size),true);
        hb.leftCols(size)=fresh_h;sb.leftCols(size)=fresh_s;
        rotated_images=false;projected_size=0;computed_h=size;
        record_performance("davidson_coarse_original_residual_rejected",0,roots);
        continue;
      }
    }
    if(fresh_residual_polish&&exact_mode&&!out.converged&&tolerance<=1e-12&&
        !strict_projection_refinement&&
        *std::max_element(out.residual_norms.begin(),out.residual_norms.end())<=1e-10) {
      // Incremental Galerkin slabs and a full product have different FP64
      // accumulation errors. At this floor rebuild H/S and the WHOLE
      // projected pair, instead of repeatedly returning the same inaccurate
      // Ritz rotation. This changes no physical operator or root selection.
      // It also repairs components INSIDE the requested-root manifold that
      // a tangent correction, orthogonal to all those roots, cannot change.
      if(!roots_freshly_checked)certify_roots();
      if(!out.converged) {
        M fresh_h,fresh_s;
        block_apply_pair(basis.leftCols(size),fresh_h,fresh_s,true);
        hb.leftCols(size)=fresh_h;sb.leftCols(size)=fresh_s;
        if(projector_state)projector_state=false;
        rotated_images=false;computed_h=size;projected_size=0;
        incremental_ritz=false;strict_projection_refinement=true;
        record_performance("davidson_strict_projected_pair_rebuild",0,size);
        continue;
      }
    }
    // A capped local band group is not a complete invariant manifold.
    // Its approximate locked bands and later global rotation must remain
    // inside the original constrained Davidson algorithm. Refine fresh
    // residuals only for the complete block, including the global fallback.
    bool strict_polished=false;
    if(fresh_residual_polish&&(!hamiltonian.band_group_local||!hamiltonian.subspace_capacity)&&
        exact_mode&&!out.converged&&tolerance<=1e-12&&
        *std::max_element(out.residual_norms.begin(),out.residual_norms.end())<=1e-10&&
        *std::max_element(out.residual_norms.begin(),out.residual_norms.end())<.5*strict_refinement_entry&&
        (strict_polish_attempts<16||(root_manifold_refinement&&strict_manifold_attempts<4))) {
      if(!roots_freshly_checked)certify_roots();
      strict_refinement_entry=residual.colwise().norm().maxCoeff();
      if(root_manifold_refinement&&!out.converged&&strict_manifold_attempts<4&&iteration+1<max_iterations){
        ++strict_manifold_attempts;
        ++iteration;out.iterations=iteration+1;
        const bool improved=strict_root_manifold_step(x,sx,values,residual,tolerance,
            [&](const M& input,M& hx,M& sy){block_apply_pair(input,hx,sy,true);},
            [&](M h,M s){const auto small=rayleigh_ritz(std::move(h),std::move(s),roots,true);
              return std::make_pair(small.values,small.vectors);},excluded_s);
        strict_polished |= improved;
        record_performance(improved?"davidson_root_manifold_refinement_accepted":
            "davidson_root_manifold_refinement_rejected",0);
        out.converged=true;
        for(Eigen::Index root=0;root<roots;++root){
          out.residual_norms[root]=residual.col(root).norm();
          out.converged &= out.residual_norms[root]<=tolerance;
        }
      }
      // Correct the original operator residual directly once the guarded
      // Ritz search has resolved the lowest-root manifold. Count every
      // correction inside the existing total 256-iteration budget.
      for(;!out.converged&&strict_polish_attempts<16&&iteration+1<max_iterations;++strict_polish_attempts) {
        M direction=residual;
        for(Eigen::Index root=0;root<roots;++root) {
          double kinetic=0.;
          if(kinetic_precondition)for(Eigen::Index i=0;i<n;++i)
            kinetic+=std::norm(x(i,root))*hamiltonian.kinetic[i];
          for(Eigen::Index i=0;i<n;++i) {
            if(kinetic_precondition)direction(i,root)*=-kinetic_preconditioner(hamiltonian.kinetic[i],kinetic);
            else {
              // Davidson may use an indefinite inverse approximation, but
              // a local energy descent needs a positive metric. Otherwise
              // its projected gradient can point uphill and be rejected.
              const double denominator=strict_positive_diagonal_scale(
                  hamiltonian.diagonal[i],values[root],overlap.diagonal[i]);
              direction(i,root)/=-denominator;
            }
          }
        }
        ++iteration;out.iterations=iteration+1;
        const double before_polish=residual.colwise().norm().maxCoeff();
        const bool improved=strict_residual_polish_step(x,sx,values,residual,direction,tolerance,
            [&](const M& input,M& hx,M& sy){block_apply_pair(input,hx,sy,true);},
            excluded,excluded_s);
        strict_polished |= improved;
        if(std::getenv("SDFT_TRACE_STRICT_POLISH"))
          std::cerr<<std::setprecision(17)<<"strict_polish iteration="<<iteration+1
            <<" before="<<before_polish<<" after="<<residual.colwise().norm().maxCoeff()
            <<" accepted="<<improved<<"\n";
        record_performance(improved?"davidson_fresh_residual_polish_accepted":
            "davidson_fresh_residual_polish_rejected",0);
        if(!improved){++strict_polish_attempts;break;}
        out.converged=true;
        for(Eigen::Index root=0;root<roots;++root) {
          out.residual_norms[root]=residual.col(root).norm();
          out.converged &= out.residual_norms[root]<=tolerance;
        }
      }
    }
    if(out.converged){certified=true;if(recycle_guards)final_ritz_vectors=rr.vectors;break;}
    if(strict_polished&&iteration+1<max_iterations){
      // Keep the accepted original-H/S correction without erasing the
      // expanding Davidson complement. Only the existing capacity boundary
      // may perform a thick restart and discard high search directions.
      M retained;
      const bool at_capacity=size+roots>capacity;
      const Eigen::Index keep=at_capacity?std::min(size,2*roots):size;
      if(at_capacity) {
        retained.resize(n,keep);retained.leftCols(roots)=x;
        if(keep>roots)retained.rightCols(keep-roots)=
            multiply(basis.leftCols(size),rr.vectors.middleCols(roots,keep-roots));
      } else {
        retained=strict_preserved_search_subspace(x,sx,M(basis.leftCols(size)));
      }
      size=0;computed_h=0;projected_size=0;rotated_images=false;
      append_block(std::move(retained));
      record_performance(at_capacity?"davidson_preserved_fresh_correction_restart":
          "davidson_preserved_fresh_correction_complement",0,keep);
    }
    if(size+roots>capacity){
      // Thick restart retains nearby empty Ritz states as well as requested
      // roots. Keeping only the occupied block can repeatedly erase a band
      // crossing that has just begun entering the search space.
      const Eigen::Index keep=std::min(size,2*roots);
      const M retained=multiply(basis.leftCols(size),rr.vectors.leftCols(keep));
      const M retained_p=projector_state?multiply(pb.leftCols(size),rr.vectors.leftCols(keep)):M{};
      if(reuse_restart_images) {
        const M retained_h=multiply(hb.leftCols(size),rr.vectors.leftCols(keep));
        const M retained_s=multiply(sb.leftCols(size),rr.vectors.leftCols(keep));
        const M gram=multiply(retained,retained_s,true);
        const double error=(gram-M::Identity(keep,keep)).cwiseAbs().maxCoeff();
        if(std::isfinite(error)&&error<=1e-12) {
          basis.leftCols(keep)=retained;hb.leftCols(keep)=retained_h;sb.leftCols(keep)=retained_s;
          if(projector_state)pb.leftCols(keep)=retained_p;
          size=keep;computed_h=keep;projected_size=0;rotated_images=true;
          record_performance("davidson_restart_saved_hs_columns",0,keep);
        }else{
          size=0;computed_h=0;projected_size=0;
          append_block(retained,projector_state?&retained_p:nullptr);
          record_performance("davidson_restart_metric_refresh",0,keep);
        }
      }else{
        size=0;computed_h=0;projected_size=0;
        append_block(retained,projector_state?&retained_p:nullptr);
      }
    }
    const Eigen::Index previous_size=size;
    std::vector<V> directions;
    for(Eigen::Index root=0;root<roots && size<capacity;++root){
      if(out.residual_norms[root]<=0.5*tolerance)continue;
      V direction=residual.col(root);
      double wavefunction_kinetic=0;
      if(kinetic_precondition)for(Eigen::Index i=0;i<n;++i)wavefunction_kinetic+=std::norm(x(i,root))*hamiltonian.kinetic[i];
      for(Eigen::Index i=0;i<n;++i){
        if(kinetic_precondition){direction[i]*=-kinetic_preconditioner(hamiltonian.kinetic[i],wavefunction_kinetic);continue;}
        double denominator=hamiltonian.diagonal[i]-values[root]*overlap.diagonal[i];
        // Bounded diagonal preconditioner; the full residual is the
        // acceptance criterion, so this never changes the solved operator.
        if(std::abs(denominator)<0.1)denominator=std::copysign(0.1,denominator==0?1:denominator);
        direction[i]/=-denominator;
      }
      directions.push_back(std::move(direction));
    }
    M correction_block(n,directions.size());for(std::size_t j=0;j<directions.size();++j)correction_block.col(j)=directions[j];append_block(std::move(correction_block));
    if(size==previous_size){
      bool added=false;
      for(Eigen::Index j=0;j<n && !added;++j){V seed=V::Zero(n);seed[order[j]]=1;added=append(seed);}
      if(!added)break;
    }
  }
  out.eigenvalues.resize(n_roots);out.eigenvectors.resize(n_roots*dimension);
  // Recheck against the original generalized operators and the S norm.
  if(!certified)certify_roots();
  if(std::getenv("SDFT_TRACE_ROOT_METRIC")) {
    double metric_error=0.;
    for(Eigen::Index i=0;i<roots;++i)for(Eigen::Index j=0;j<roots;++j)
      metric_error=std::max(metric_error,std::abs(compensated_complex_inner(x.col(i),sx.col(j))-(i==j?C(1.):C(0.))));
    std::cerr<<std::setprecision(17)<<"SDFT_ROOT_METRIC_AUDIT tolerance="<<tolerance
      <<" error="<<metric_error<<" fresh="<<*std::max_element(out.residual_norms.begin(),out.residual_norms.end())<<"\n";
  }
  for(Eigen::Index root=0;root<roots;++root){
    // A vector certified against a fresh original H/S application above is
    // already normalized. Repeating normalization adds a new rounding error
    // without improving the metric or the certified residual.
    out.eigenvalues[root]=values[root];
    for(Eigen::Index i=0;i<n;++i)out.eigenvectors[root*dimension+i]=x(i,root);
  }
  if(out.converged && recycle_guards && final_ritz_vectors.cols()>roots) {
    PerformanceTimer timer("davidson_export_spectral_guards");
    const auto count=std::min(roots,final_ritz_vectors.cols()-roots);
    const M guards=multiply(basis.leftCols(size),final_ritz_vectors.middleCols(roots,count));
    if(!guards.allFinite())throw std::runtime_error("nonfinite exported spectral search state");
    out.search_guard_vectors.assign(guards.data(),guards.data()+guards.size());
    record_performance("davidson_export_spectral_guard_columns",0,count);
  }
  return out;
#else
  throw std::runtime_error("Davidson requires bundled Eigen headers");
#endif
}

}
