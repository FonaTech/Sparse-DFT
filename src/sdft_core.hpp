#pragma once
#include <array>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>

namespace sdft {
struct Vec3 { double x{}, y{}, z{}; };
double dot(Vec3 a, Vec3 b);
Vec3 cross(Vec3 a, Vec3 b);
double determinant(const std::array<Vec3,3>& lattice);
struct PawCacheMetadata { std::string record_id, element, family, xc; double zval{}, enmax_ev{}; std::string potcar_sha256, psctr_sha256; };
PawCacheMetadata read_paw_cache_metadata(const std::filesystem::path& path);
// Returns lowercase SHA-256; unreadable files are errors, never an empty identity.
std::string file_sha256(const std::filesystem::path& path);
bool file_sha256_matches(const std::filesystem::path& path, const std::string& expected_hex);

struct PawNumericChannel {
  int angular_momentum{};
  int nprojectors{};
  double cutoff_radius{};
  double reciprocal_cutoff{};
  std::vector<std::vector<double>> d_matrix;
  std::vector<std::vector<double>> reciprocal_projectors;
  std::vector<std::vector<double>> real_projectors;
};

struct PawNumericCache {
  std::uint32_t version{};
  std::uint32_t main_grid_size{};
  std::uint32_t radial_points{};
  // Physical radial-set cutoff from the POTCAR radial header.  This is
  // distinct from radial_grid.back(), which is the end of the logarithmic
  // integration mesh and can be substantially larger than the PAW sphere.
  double radial_rmax{};
  // POTCAR RDEP radius bounding the truncated AE-norm Simpson integral.
  // It is recovered from the adjacent restored POTCAR for v1-v3 caches.
  double radial_rdep{}; // Angstrom, matching radial_grid; exact PSCTR RDEP.
  bool atomic_reference_present{};
  double atomic_reference_energy_ev{};
  double core_xc_outside_ev{};
  std::vector<double> kinetic_error_cutoff_ev, kinetic_error_ev;
  std::vector<double> local_header;
  std::vector<double> local_potential;
  std::vector<double> core_charge_partial;
  std::vector<double> kinetic_energy_partial;
  std::vector<double> atomic_pseudo_charge;
  std::vector<double> atomic_pseudo_charge_tail;
  std::vector<PawNumericChannel> nonlocal_channels;
  std::vector<double> radial_grid;
  std::vector<double> ae_potential;
  std::vector<double> core_charge_density;
  std::vector<double> kinetic_energy_density;
  std::vector<double> kinetic_energy_density_pseudized;
  std::vector<double> local_pseudopotential_core;
  std::vector<double> pseudopotential_valence;
  std::vector<double> core_charge_density_pseudized;
  std::vector<double> augmentation_coefficients;
  std::vector<double> occupancies;
  std::vector<std::vector<double>> pseudo_wavefunctions;
  std::vector<std::vector<double>> ae_wavefunctions;
};

PawNumericCache read_paw_numeric_cache(const std::filesystem::path& path);
double paw_numeric_checksum(const PawNumericCache& cache);
}
