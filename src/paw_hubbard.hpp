#pragma once
#include "sdft_core.hpp"
#include <array>
#include <complex>
#include <vector>

namespace sdft {
struct HubbardSpec { int angular_momentum{-1}; double ueff_ev{}; };
struct HubbardResult {
  std::size_t correlated_dimension{};
  std::array<std::vector<std::complex<double>>,2> occupations;
  std::array<std::vector<std::complex<double>>,2> projector_potential_ev;
  double energy_ev{},potential_expectation_ev{},band_double_counting_ev{};
};
// PAW occupation definition matched to LDAUTYPE=2: contract the complete
// expanded density with AE partial-wave radial overlaps inside RDEP. These
// are neither raw |Bpsi|^2 nor artificially normalized projection weights.
std::vector<double> paw_hubbard_ae_overlap(const PawNumericCache&,
    const std::vector<int>& radial_channel_l,int correlated_l);
// AE-sphere overlap for LORBIT>=10-style sphere weights: the same AE overlap
// inside RDEP for every same-l pair, but with Simpson weights rebuilt on the
// truncated mesh. The DFT+U overlap above keeps the full-mesh weights; the
// two differ only through the edge-point weight.
std::vector<double> paw_sphere_ae_overlap(const PawNumericCache&,
    const std::vector<int>& radial_channel_l);
HubbardResult dudarev_paw(const std::vector<int>& radial_channel_l,
    const std::vector<double>& ae_overlap,HubbardSpec,
    const std::array<std::vector<std::complex<double>>,2>& onsite);
}
