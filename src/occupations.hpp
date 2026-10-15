#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sdft {
// Smeared occupation reached the topmost computed band (> 1e-6 at some k).
// Carries the band count so an automatic band rule can grow and restart.
struct InsufficientEmptyBands : std::runtime_error {
  std::size_t bands;
  explicit InsufficientEmptyBands(std::size_t n)
      : std::runtime_error("insufficient empty bands for smearing; increase bands"), bands(n) {}
};
struct OccupationResult {
  std::vector<std::vector<double>> occupations; // spin-summed, 0..2; weights separate
  double chemical_potential_ev{};
  double electron_count{};
  double entropy_energy_ev{}; // -T*S, negative, for Fermi-Dirac free energy
// Spectrum is spin-major [up k..., down k...], weights are physical k weights.
// Reuse the proven Fermi solver with half weights and halve occupations.
OccupationResult fill_collinear_occupations(const std::vector<std::vector<double>>&,
    const std::vector<double>& physical_k_weights,double electrons,double sigma_ev=0.);
};
// One chemical potential for all k-points. Weights must be positive and sum
// to one. sigma_ev = k_B*T; zero selects degeneracy-preserving zero-T filling.
OccupationResult fill_occupations(const std::vector<std::vector<double>>& eigenvalues_ev,
    const std::vector<double>& weights, double electrons, double sigma_ev=0.0);
// Spectrum is spin-major [up k..., down k...], weights are physical k weights.
// Reuse the proven Fermi solver with half weights and halve occupations.
OccupationResult fill_collinear_occupations(const std::vector<std::vector<double>>&,
    const std::vector<double>& physical_k_weights,double electrons,double sigma_ev=0.);
}
