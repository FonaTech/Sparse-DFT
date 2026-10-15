#include "paw_augmentation.hpp"
#include "paw_compensation.hpp"

#include <cmath>
#include <iostream>

namespace {
constexpr double pi = 3.1415926535897932384626433832795;
constexpr int nz = 16;
constexpr int nphi = 64;
constexpr double z[nz] = {
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.0950125098376374,  0.0950125098376374,
     0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,
     0.9894009349916499};
constexpr double wz[nz] = {
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928,
    0.1246289712555339, 0.1495959888165767, 0.1691565193950025,
    0.1826034150449236, 0.1894506104550685, 0.1894506104550685,
    0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479,
    0.0271524594117541};

double direct(int l1, int m1, int l2, int m2, int L, int M) {
  double value = 0.0;
  for (int iz = 0; iz < nz; ++iz) {
    const double zz = z[iz];
    const double s = std::sqrt(std::max(0.0, 1.0 - zz * zz));
    for (int ip = 0; ip < nphi; ++ip) {
      const double phi = 2.0 * pi * static_cast<double>(ip) / nphi;
      const sdft::Vec3 q{s * std::cos(phi), s * std::sin(phi), zz};
      value += wz[iz] * (2.0 * pi / nphi) *
          sdft::paw_real_spherical_harmonic(l1, m1, q) *
          sdft::paw_real_spherical_harmonic(l2, m2, q) *
          sdft::paw_real_spherical_harmonic(L, M, q);
    }
  }
  return value;
}
} // namespace

int main() {
  std::size_t terms = 0;
  double max_error = 0.0;
  double max_inverse = 0.0;
  for (int l1 = 0; l1 <= 3; ++l1) for (int l2 = 0; l2 <= 3; ++l2)
    for (int m1 = -l1; m1 <= l1; ++m1) for (int m2 = -l2; m2 <= l2; ++m2) {
      const auto table = sdft::paw_real_gaunt_terms(l1, l2, m1, m2);
      for (const auto& term : table) {
        if (term.L < std::abs(l1 - l2) || term.L > l1 + l2 ||
            std::abs(term.M) > term.L || !std::isfinite(term.value) ||
            !std::isfinite(term.inverse)) {
          std::cerr << "invalid Gaunt term" << '\n';
          return 2;
        }
        const double reference = direct(l1, m1, l2, m2, term.L, term.M);
        max_error = std::max(max_error, std::abs(reference - term.value));
        max_inverse = std::max(max_inverse, std::abs(term.value * term.inverse));
        ++terms;
      }
    }
  // The largest polynomial degree in this loop is 12; the quadrature above is
  // exact to roundoff. This gate catches sign, sqrt(2), M ordering and parity
  // errors in the closed-form (Wigner 3j) coefficients.
  if (terms == 0 || max_error > 2e-12 || !std::isfinite(max_inverse)) {
    std::cerr << "Gaunt coefficient regression failed terms=" << terms
              << " max_error=" << max_error << " max_inverse=" << max_inverse << '\n';
    return 1;
  }
  const auto table = sdft::paw_real_gaunt_table(3);
  std::size_t lookup = 0;
  if (table.row_offset.empty() || table.row_offset.back() != table.value.size() ||
      table.L.size() != table.value.size() || table.lm_index.size() != table.value.size() ||
      table.inverse.size() != table.value.size()) {
    std::cerr << "Gaunt table dimensions are inconsistent\n";
    return 3;
  }
  for (int l1 = 0; l1 <= 3; ++l1) for (int l2 = 0; l2 <= 3; ++l2)
    for (int m1 = -l1; m1 <= l1; ++m1) for (int m2 = -l2; m2 <= l2; ++m2) {
      const auto terms_here = sdft::paw_real_gaunt_terms(l1, l2, m1, m2);
      if (table.row_offset[lookup] > table.row_offset[lookup + 1] ||
          table.row_offset[lookup + 1] - table.row_offset[lookup] != terms_here.size()) {
        std::cerr << "Gaunt table row span mismatch\n";
        return 4;
      }
      for (std::size_t i = 0; i < terms_here.size(); ++i) {
        const auto pos = table.row_offset[lookup] + i;
        if (table.L[pos] != terms_here[i].L ||
            table.lm_index[pos] != terms_here[i].L * (terms_here[i].L + 1) + terms_here[i].M ||
            std::abs(table.value[pos] - terms_here[i].value) > 1e-15 ||
            std::abs(table.inverse[pos] - terms_here[i].inverse) > 1e-15) {
          std::cerr << "Gaunt table entry mismatch\n";
          return 5;
        }
      }
      ++lookup;
    }
  std::cout << "paw_ylm3_smoke PASS terms=" << terms
            << " max_error=" << max_error
            << " max_inverse_product=" << max_inverse
            << " table_entries=" << table.value.size() << '\n';
  return 0;
}
