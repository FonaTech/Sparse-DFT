#pragma once

#include "sdft_core.hpp"
#include <cstddef>
#include <complex>
#include <functional>
#include <vector>

namespace sdft {

struct GVector {
  int h{}, k{}, l{};
  Vec3 cartesian{};
  double kinetic_ev{};
};

std::vector<GVector> enumerate_g_vectors(const std::array<Vec3, 3>& reciprocal,
                                         double cutoff_ev, Vec3 kpoint_fractional = {},
                                         double hartree_to_ev = 27.211386245988);

// Select integer indices once in a reference cell, then evaluate their
// Cartesian vectors and kinetic energies in the current cell. This defines
// the differentiable finite-basis energy used by analytic strain checks.
std::vector<GVector> enumerate_g_vectors_fixed_reference(
    const std::array<Vec3, 3>& reciprocal,
    const std::array<Vec3, 3>& reference_reciprocal,
    double cutoff_ev, Vec3 kpoint_fractional = {},
    double hartree_to_ev = 27.211386245988);

struct PawProjectorOperator {
  // Row-major projector amplitudes: nprojectors x nbasis.
  std::size_t nprojectors{};
  std::size_t nbasis{};
  std::vector<double> amplitudes;
  std::vector<double> d_matrix;
  std::vector<double> overlap_matrix;
};

struct DenseGeneralizedEigenResult {
  std::vector<double> eigenvalues;
  std::vector<double> eigenvectors;
  std::size_t dimension{};
};

DenseGeneralizedEigenResult solve_generalized_symmetric(
    const std::vector<double>& hamiltonian,
    const std::vector<double>& overlap,
    std::size_t dimension);

// Lowest-root iterative path for large plane-wave matrices.  It applies a
// fully re-orthogonalized Lanczos recurrence to the Cholesky-transformed
// standard problem and returns the same eigenvalue/eigenvector container as
// the dense solver.  ``n_roots`` may be smaller than ``dimension``; callers
// should use the dense path for full spectra or as a reference check.
DenseGeneralizedEigenResult solve_generalized_symmetric_iterative(
    const std::vector<double>& hamiltonian,
    const std::vector<double>& overlap,
    std::size_t dimension,
    std::size_t n_roots,
    std::size_t max_iterations = 128,
    double tolerance = 1e-10);

// Complex Hermitian generalized eigensolver for Bloch (non-Gamma) plane-wave
// Hamiltonians.  The overlap is factorized as S=L L^H and the Lanczos
// recurrence is applied to A=L^{-1} H L^{-H} through matrix-vector products.
// Eigenvectors are returned row-major as n_roots x dimension and are
// normalized with respect to S.  This path avoids a dense generalized
// eigendecomposition; callers should inspect residual_norms when deciding
// whether a result is production quality.
struct ComplexGeneralizedEigenResult {
  std::vector<double> eigenvalues;
  std::vector<std::complex<double>> eigenvectors;
  std::vector<double> residual_norms;
  std::size_t dimension{};
  bool converged{};
  std::size_t iterations{};
  // Approximate upper Ritz states, for the next unchanged-basis SCF search.
  // These are neither converged roots nor H/S images and must be reapplied.
  std::vector<std::complex<double>> search_guard_vectors;
  // An early-SCF trial manifold can meet its coarse original H/S residual
  // without proving that it spans the lowest bands. Such output ALWAYS
  // has converged=false, and must never be used for NSCF or final acceptance.
  bool trial_manifold{};
};

ComplexGeneralizedEigenResult solve_generalized_hermitian_iterative(
    const std::vector<std::complex<double>>& hamiltonian,
    const std::vector<std::complex<double>>& overlap,
    std::size_t dimension,
    std::size_t n_roots,
    std::size_t max_iterations = 256,
    double tolerance = 1e-10);

// Block Davidson in the S metric. Applies H/S without factoring the full
// overlap matrix. Only the small Rayleigh-Ritz space is diagonalized.
ComplexGeneralizedEigenResult solve_generalized_hermitian_davidson(
    const std::vector<std::complex<double>>& hamiltonian,
    const std::vector<std::complex<double>>& overlap,
    std::size_t dimension, std::size_t n_roots,
    std::size_t max_iterations = 256, double tolerance = 1e-10,
    const std::vector<std::complex<double>>& initial_vectors = {});

// Matrix-free generalized problem. Diagonals are preconditioners; convergence
// is always checked with the actual H/S callbacks, never their diagonals.
struct HermitianOperator {
  std::size_t dimension{};
  std::vector<double> diagonal;
  std::function<void(const std::vector<std::complex<double>>&,
                     std::vector<std::complex<double>>&)> apply;
  // Contiguous column-major n x columns. exact=true forbids approximate
  // accelerator arithmetic. Scalar apply always remains an FP64 reference.
  std::function<void(const std::vector<std::complex<double>>&,std::size_t,
                     std::vector<std::complex<double>>&,bool)> apply_block;
  // Optional direct contiguous column-major input/output. The caller owns
  // both buffers until the synchronous callback returns; no retained views.
  std::function<void(const std::complex<double>*,std::size_t,
                     std::complex<double>*,bool)> apply_block_into;
  // Optional synchronous joint H/S application. Both operators must have
  // the same non-null identity; never pair unrelated Hamiltonian/overlap.
  // The implementation may share immediate projector amplitudes for this
  // exact input, but must not retain pointers or reuse stale rotated values.
  const void* pair_identity{};
  std::function<void(const std::complex<double>*,std::size_t,
                     std::complex<double>*,std::complex<double>*,bool)> apply_pair_into;
  // Physical T(G+k), in Hartree, independent of H's local/PAW diagonal.
  std::vector<double> kinetic;
  // Optional low-rank state. Identity/rows must agree for H and S. The
  // solver owns amplitudes and their lifetime; callbacks retain no pointers.
  std::size_t projector_rows{};
  std::function<void(const std::complex<double>*,std::size_t,
                     std::complex<double>*,bool)> project_into;
  std::function<void(const std::complex<double>*,std::size_t,
                     const std::complex<double>*,std::complex<double>*,bool)> apply_projected_into;
  // Internal local-solver controls, never a relaxation of residual accuracy.
  std::size_t subspace_capacity{};
  bool band_group_local{};
  // Internal warm search predictor only. Suppresses extra canonical seeds
  // in a bounded local update; the caller MUST then run the original global
  // guarded solve, even if a local original residual happens to vanish.
  bool local_seed_only{};
  // Internal SCF search scheduling, disabled for standalone/NSCF solves.
  // Only the SCF driver may consume a residual-checked trial manifold.
  bool scf_trial_update{};
  // S-normalized completed lower bands, column-major. Search directions
  // are orthogonalized against these without changing H or S.
  std::vector<std::complex<double>> excluded_vectors;
  double metric_identity_scale{1.};
  std::function<void(const std::complex<double>*,std::size_t,std::complex<double>*)> projector_metric_into;
  // Same caller-owned amplitudes for both unchanged original H/S images.
  std::function<void(const std::complex<double>*,std::size_t,const std::complex<double>*,
                     std::complex<double>*,std::complex<double>*,bool)> apply_projected_pair_into;
  // Completed-band images belong to this unchanged operator identity and
  // exactly these excluded_vectors. Owned arrays avoid dangling group views.
  // Empty arrays retain the original fresh-application path.
  const void* excluded_image_identity{};
  std::vector<std::complex<double>> excluded_overlap_images;
  std::vector<std::complex<double>> excluded_projector_images;
  // Optional caller-owned upper search states from a previous Hamiltonian.
  // Geometry/basis order must match. No operator-image reuse is permitted.
  std::vector<std::complex<double>> initial_search_guards;
  // A callback can offer lower-precision search even on a CPU. true only
  // affects its exact=false search calls; acceptance always applies the
  // original exact=true H/S. Default false preserves FP64 CPU search.
  bool approximate_search{};
  // Original PAW overlap may have a negative-signature ghost direction.
  // This selects an unchanged-pencil solve, never a regularized S.
  bool indefinite_metric{};
};
// Positive, bounded kinetic preconditioner. Both arguments are Hartree;
// wavefunction_kinetic is sum_G T_G |psi_G|^2 for S-normalized psi.
double kinetic_preconditioner(double kinetic, double wavefunction_kinetic);
// Internal coarse SCF residual minimization. A residual-checked trial is
// explicitly NONCONVERGED and never claims the lowest spectrum. Strict
// and standalone/NSCF requests are rejected; failed trials use the original
// guarded solver within the same total iteration budget.
ComplexGeneralizedEigenResult solve_generalized_hermitian_rmm_trial(
    const HermitianOperator&,const HermitianOperator&,std::size_t n_roots,
    std::size_t max_iterations,double tolerance,
    const std::vector<std::complex<double>>& initial_vectors);
ComplexGeneralizedEigenResult solve_generalized_hermitian_band_groups(
    const HermitianOperator&,const HermitianOperator&,std::size_t n_roots,
    std::size_t max_iterations,double tolerance,
    const std::vector<std::complex<double>>& initial_vectors={});
ComplexGeneralizedEigenResult solve_generalized_hermitian_davidson(
    const HermitianOperator& hamiltonian, const HermitianOperator& overlap,
    std::size_t n_roots, std::size_t max_iterations = 256,
    double tolerance = 1e-10,
    const std::vector<std::complex<double>>& initial_vectors = {});

ComplexGeneralizedEigenResult solve_generalized_hermitian_indefinite_davidson(
    const HermitianOperator&,const HermitianOperator&,std::size_t n_roots,
    std::size_t max_iterations,double tolerance,
    const std::vector<std::complex<double>>& initial_vectors = {});

void add_paw_low_rank_operator(std::vector<double>& hamiltonian,
                               std::vector<double>& overlap,
                               std::size_t dimension,
                               const PawProjectorOperator& projectors);

}
