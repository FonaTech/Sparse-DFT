#pragma once
#include "pw_core.hpp"
#include "density_functionals.hpp"
#include <memory>
#include <array>
#include <cstdint>
#include <string>

namespace sdft {
struct ComplexPawOperator {
  std::size_t projectors{};
  // Row-major B[projector,G], D and Q. D is in Hartree, Q dimensionless.
  std::vector<std::complex<double>> amplitudes, d, q;
  // Optional real-space geometry for search-only projectors (S6):
  // fractional atom position and the largest PAW projector radius (bohr).
  Vec3 fractional{};
  double radius_bohr{};
  std::array<Vec3,3> lattice_bohr{};
};

// Exact Galerkin application of T + V(r) + B^dagger D B and
// S = (1+regularization) I + B^dagger Q B. FFT padding prevents aliasing;
// no NPW x NPW matrix or overlap Cholesky factor is allocated.
// Each instance owns reusable FFT buffers and must be used by one thread.
class PlaneWaveOperator {
 public:
  PlaneWaveOperator(const std::vector<GVector>& basis, const FourierDensity& potential,
                    std::vector<ComplexPawOperator> paw, double hartree_to_ev,
                    double overlap_regularization = 1e-12,
                    std::uint64_t immutable_projector_identity = 0);
  // A nonzero projector identity may be shared only for identical immutable
  // B arrays (same PAW/positions/G order/units). D, Q and V remain per-instance.
  ~PlaneWaveOperator();
  PlaneWaveOperator(const PlaneWaveOperator&) = delete;
  PlaneWaveOperator& operator=(const PlaneWaveOperator&) = delete;
  HermitianOperator hamiltonian();
  HermitianOperator overlap();
  std::size_t fft_grid_points() const;
  std::size_t storage_bytes() const;
  // Diagnostic only: exact stored Galerkin arrays and band-major warm vectors.
  // This is a solver input, never a converged SCF/checkpoint or physical result.
  void save_solver_snapshot(const std::string& path, std::size_t roots,
                            std::size_t iterations, double tolerance,
                            const std::vector<std::complex<double>>& initial) const;
  static std::unique_ptr<PlaneWaveOperator> load_solver_snapshot(
      const std::string& path, std::size_t& roots, std::size_t& iterations,
      double& tolerance, std::vector<std::complex<double>>& initial);
 private:
  struct Impl;
  explicit PlaneWaveOperator(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}
