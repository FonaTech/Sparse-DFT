#include "sdft_core.hpp"
#include "paw_augmentation.hpp"
#include "paw_hubbard.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: PAW_CACHE SAVED_ONSITE_ORACLE");
    const auto cache = sdft::read_paw_numeric_cache(argv[1]);
    std::vector<int> channels;
    std::size_t dimension{};
    for (const auto& c : cache.nonlocal_channels)
      for (int p = 0; p < c.nprojectors; ++p) {
        channels.push_back(c.angular_momentum); dimension += 2*c.angular_momentum+1;
      }
    std::ifstream file(argv[2]);
    std::string tag; std::size_t atom_count{}, expected_dimension{};
    double radius{}, limit{}, maximum{};
    if (!(file >> tag >> atom_count >> expected_dimension >> radius >> limit) ||
        tag != "SDFT_HUBBARD_RADIAL_ORACLE_1" || !atom_count || atom_count > 100 ||
        expected_dimension != dimension || !std::isfinite(radius) ||
        !std::isfinite(limit) || limit <= 0 ||
        std::abs(cache.radial_rdep-radius) > 1e-12)
      throw std::runtime_error("Hubbard radius or fixture identity mismatch");
    const auto overlap = sdft::paw_hubbard_ae_overlap(cache, channels, 2);
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      std::array<std::vector<std::complex<double>>,2> density;
      for (auto& spin : density) {
        spin.resize(dimension*dimension);
        for (auto& value : spin) {
          double re{}, im{};
          if (!(file >> re >> im) || !std::isfinite(re) || !std::isfinite(im))
            throw std::runtime_error("invalid saved onsite density");
          value = {re, im};
        }
      }
      const auto result = sdft::dudarev_paw(channels, overlap, {2,4.}, density);
      for (const auto& spin : result.occupations) for (const auto& value : spin) {
        double expected{};
        if (!(file >> expected) || !std::isfinite(expected))
          throw std::runtime_error("invalid independent occupation reference");
        maximum = std::max(maximum, std::abs(value-expected));
      }
      if (!std::isfinite(result.energy_ev)) throw std::runtime_error("invalid Hubbard energy");
    }
    if (file >> tag) throw std::runtime_error("trailing occupation oracle data");
    if (maximum > limit) throw std::runtime_error("independent Hubbard occupation mismatch");
    std::cout << std::setprecision(17) << "PAW_HUBBARD_RADIAL_ORACLE_PASS max_difference="
              << maximum << " radius_angstrom=" << cache.radial_rdep << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
