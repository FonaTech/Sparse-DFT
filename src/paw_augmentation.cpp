#include "paw_augmentation.hpp"
#include "compute_backend.hpp"
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
#include "pbe_xc.hpp"
#include "paw_hubbard.hpp"
#include "spin_pbe_xc.hpp"
#include "paw_compensation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace sdft {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;

std::vector<double> radial_quadrature_weights(const std::vector<double>& r) {
  const std::size_t n = r.size();
  if (n < 2) return std::vector<double>(n, 0.0);
  // PAW datasets use logarithmic meshes, r_i = r_0 e^{i h}. With x = ln r the
  // integral is int f r dx, evaluated by the composite Simpson rule in x. The
  // chain of Simpson panels ends at the outermost node, so for an even point
  // count the innermost node r_0 lies outside the chain and has zero weight.
  const double h = std::log(r[1] / r[0]);
  bool log_mesh = r[0] > 0.0 && std::isfinite(h);
  for (std::size_t i = 2; i < n && log_mesh; ++i)
    log_mesh = std::abs(std::log(r[i] / r[i - 1]) - h) < 2e-7;
  std::vector<double> weights(n, 0.0);
  if (log_mesh && n >= 3) {
    const std::size_t start = (n % 2 == 1) ? 0 : 1;  // first node of the chain
    for (std::size_t i = start; i < n; ++i) {
      const std::size_t from_outer = n - 1 - i;
      const double simpson = (i == start || from_outer == 0) ? 1.0 : (from_outer % 2 == 1 ? 4.0 : 2.0);
      weights[i] = simpson * r[i] * h / 3.0;
    }
    return weights;
  }
  for (std::size_t i = 0; i + 1 < r.size(); ++i) {
    const double half_dr = 0.5 * (r[i + 1] - r[i]);
    weights[i] += half_dr;
    weights[i + 1] += half_dr;
  }
  return weights;
}

double radial_integral(const std::vector<double>& r, const std::vector<double>& f) {
  if (r.size() != f.size() || r.size() < 2) return 0.0;
  double out = 0.0;
  const auto weights = radial_quadrature_weights(r);
  for (std::size_t i = 0; i < r.size(); ++i) out += weights[i] * f[i];
  return out;
}

double spherical_bessel(int l, double x) {
  if (l < 0) return 0.0;
  if (std::abs(x) < 0.5) {
    double term = 1.0;
    for (int k = 1; k <= l; ++k) term *= x / (2.0 * k + 1.0);
    double value = term;
    for (int k = 1; k < 30; ++k) {
      term *= -x * x / (2.0 * k * (2.0 * l + 2.0 * k + 1.0));
      value += term;
      if (std::abs(term) < 1e-17 * std::max(1.0, std::abs(value))) break;
    }
    return value;
  }
  const double j0 = std::sin(x) / x;
  if (l == 0) return j0;
  const double j1 = std::sin(x) / (x * x) - std::cos(x) / x;
  if (l == 1) return j1;
  double jm1 = j0, j = j1;
  for (int n = 1; n < l; ++n) {
    const double jp1 = (2.0 * n + 1.0) * j / x - jm1;
    jm1 = j;
    j = jp1;
  }
  return j;
}

double spherical_bessel_derivative(int l, double x) {
  if (std::abs(x) < 1e-8) return l == 1 ? 1.0 / 3.0 : 0.0;
  if (l == 0) return -spherical_bessel(1, x);
  return spherical_bessel(l - 1, x) - (l + 1.0) * spherical_bessel(l, x) / x;
}

std::vector<double> first_bessel_roots(int l, int count) {
  std::vector<double> roots;
  // For l>1, j_l(x) is numerically tiny and positive near x=0; starting
  // there causes a spurious sign change to be detected as a root.  Begin
  // beyond the small-x region and bracket only physical nodes.
  double x0 = std::max(0.1, 0.5 * static_cast<double>(l));
  double f0 = spherical_bessel(l, x0);
  for (double x = x0 + 0.01; x < 100.0 && static_cast<int>(roots.size()) < count; x += 0.01) {
    const double f = spherical_bessel(l, x);
    if (f0 * f <= 0.0) {
      double lo = x - 0.01, hi = x;
      for (int it = 0; it < 70; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (spherical_bessel(l, lo) * spherical_bessel(l, mid) <= 0.0) hi = mid;
        else lo = mid;
      }
      roots.push_back(0.5 * (lo + hi));
    }
    f0 = f;
  }
  if (static_cast<int>(roots.size()) != count)
    throw std::runtime_error("failed to construct PAW compensation Bessel roots");
  return roots;
}

struct RealHarmonic {
  double value{}, theta_derivative{}, phi_derivative_over_sin{};
};

RealHarmonic real_harmonic_with_gradient(int l, int m, double z, double phi) {
  if (l < 0 || l > 7 || std::abs(m) > l)
    throw std::invalid_argument("real PAW harmonics support 0<=l<=7 (gradient response extends L=6 to 7)");
  const int order = std::abs(m);
  const double sintheta = std::sqrt(std::max(0.0, 1.0 - z * z));
  double pmm = 1.0;
  for (int i = 1; i <= order; ++i) pmm *= (2.0 * i - 1.0) * sintheta;
  double prev = 0.0, current = pmm;
  if (l > order) {
    prev = pmm;
    current = (2.0 * order + 1.0) * z * pmm;
    for (int degree = order + 2; degree <= l; ++degree) {
      const double next = ((2.0 * degree - 1.0) * z * current -
                           (degree + order - 1.0) * prev) / (degree - order);
      prev = current;
      current = next;
    }
  }
  const double normalization = std::sqrt((2.0 * l + 1.0) / (4.0 * pi) *
      std::exp(std::lgamma(l - order + 1.0) - std::lgamma(l + order + 1.0))) *
      (order ? std::sqrt(2.0) : 1.0);
  const double trig = m < 0 ? std::sin(order * phi) : std::cos(order * phi);
  const double trig_derivative = m < 0 ? order * std::cos(order * phi) : -order * std::sin(order * phi);
  const double ptheta = sintheta > 1e-14 ?
      (l * z * current - (l + order) * prev) / sintheta : 0.0;
  return {normalization * current * trig, normalization * ptheta * trig,
          sintheta > 1e-14 ? normalization * current * trig_derivative / sintheta : 0.0};
}

double real_spherical_harmonic(int l, int m, double x, double y, double z) {
  const double q = std::sqrt(x * x + y * y + z * z);
  if (q < 1e-14) return l == 0 ? 1.0 / (2.0 * std::sqrt(pi)) : 0.0;
  return real_harmonic_with_gradient(l, m, z / q, std::atan2(y, x)).value;
}

double gaunt(int l1, int m1, int l2, int m2, int l3, int m3) {
  // Density-independent coefficients recur for every atom and SCF iteration.
  // Per-thread storage avoids synchronization in parallel atom processing.
  thread_local std::map<std::array<int, 6>, double> cache;
  const std::array<int, 6> key{l1, m1, l2, m2, l3, m3};
  const auto existing = cache.find(key);
  if (existing != cache.end()) return existing->second;
  // Tensor-product Gauss-Legendre x uniform-phi quadrature is exact for these
  // products of real harmonics (projector l<=3, multipole L<=6).
  static const double x[12] = {
      -0.9815606342467191, -0.9041172563704749, -0.7699026741943047,
      -0.5873179542866175, -0.3678314989981802, -0.1252334085114689,
       0.1252334085114689,  0.3678314989981802,  0.5873179542866175,
       0.7699026741943047,  0.9041172563704749,  0.9815606342467191};
  static const double w[12] = {
      0.0471753363865118, 0.1069393259953184, 0.1600783285433462,
      0.2031674267230659, 0.2334925365383548, 0.2491470458134029,
      0.2491470458134029, 0.2334925365383548, 0.2031674267230659,
      0.1600783285433462, 0.1069393259953184, 0.0471753363865118};
  constexpr int nphi = 48;
  double sum = 0.0;
  for (int iz = 0; iz < 12; ++iz) {
    const double z = x[iz], s = std::sqrt(std::max(0.0, 1.0 - z * z));
    for (int ip = 0; ip < nphi; ++ip) {
      const double phi = 2.0 * pi * static_cast<double>(ip) / nphi;
      const double xx = s * std::cos(phi), yy = s * std::sin(phi);
      sum += w[iz] * (2.0 * pi / nphi) *
             real_spherical_harmonic(l1, m1, xx, yy, z) *
             real_spherical_harmonic(l2, m2, xx, yy, z) *
             real_spherical_harmonic(l3, m3, xx, yy, z);
    }
  }
  cache.emplace(key, sum);
  return sum;
}

// Real-harmonic Gaunt coefficients
//
//   G(l1 m1, l2 m2 | L M) = \int Y_{l1 m1} Y_{l2 m2} Y_{L M} dOmega
//
// for the real harmonics of real_harmonic_with_gradient():
//   Y_lm = N_{l|m|} P_l^|m|(cos theta) Phi_m(phi), P without the Condon-Shortley
//   phase, Phi_0 = 1, Phi_{m>0} = sqrt2 cos(m phi), Phi_{m<0} = sqrt2 sin(|m| phi).
// The integral factorises into an azimuthal average of Phi_m1 Phi_m2 Phi_M and a
// polar integral. Writing each Phi as a sum of exp(+-i|m| phi) keeps only the
// exponent triples (e1, e2, e3) with e1 + e2 + e3 = 0. For any such triple the
// polar integral is the complex-harmonic Gaunt integral (DLMF 34.3.22) times the
// Condon-Shortley signs, so only Wigner 3j symbols (DLMF 34.2.4) are needed.
struct GauntEntry { int L{}, M{}; double value{}, inverse{}; };

double parity_sign(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// Binomial coefficient C(n, k). Every partial product is an integer below 2^53
// for n <= 24, the largest argument the Gaunt table needs (l <= 6, L <= 12).
double binomial(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  k = std::min(k, n - k);
  double c = 1.0;
  for (int i = 1; i <= k; ++i) c = c * static_cast<double>(n - k + i) / static_cast<double>(i);
  return std::round(c);
}

// Wigner 3j symbol. Dividing Racah's sum (DLMF 34.2.4) by its triangle and
// magnetic factorials turns every factor into a binomial coefficient:
//   (j1 j2 j3; m1 m2 m3) = (-1)^(j1-j2-m3) sqrt(A / B) sum_k (-1)^k
//       C(J-2j3, k) C(J-2j2, j1-m1-k) C(J-2j1, j2+m2-k),   J = j1+j2+j3,
//   A = C(2j1, J-2j3) C(2j2, J-2j3),
//   B = (J+1) C(J, J-2j3) C(2j1, j1-m1) C(2j2, j2-m2) C(2j3, j3-m3).
// Each binomial and the alternating sum are exact integers in double precision;
// rounding enters only through the products under the square root.
double wigner_3j(int j1, int j2, int j3, int m1, int m2, int m3) {
  const int J = j1 + j2 + j3;
  if (m1 + m2 + m3 != 0 || std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m3) > j3 ||
      J - 2 * j3 < 0 || J - 2 * j2 < 0 || J - 2 * j1 < 0)
    return 0.0;
  const int a = J - 2 * j3, b = J - 2 * j2, c = J - 2 * j1;
  double series = 0.0;
  // binomial() is zero outside 0 <= k <= n, which sets the summation limits.
  for (int k = 0; k <= a; ++k) {
    const double term = binomial(a, k) * binomial(b, j1 - m1 - k) * binomial(c, j2 + m2 - k);
    series += (k % 2 == 0) ? term : -term;
  }
  const double numerator = binomial(2 * j1, a) * binomial(2 * j2, a);
  const double denominator = static_cast<double>(J + 1) * binomial(J, a) * binomial(2 * j1, j1 - m1) *
      binomial(2 * j2, j2 - m2) * binomial(2 * j3, j3 - m3);
  return parity_sign(j1 - j2 - m3) * std::sqrt(numerator / denominator) * series;
}

// Phi_m(phi) as (exponent e, coefficient c) pairs of c exp(i e phi).
std::vector<std::pair<int, std::complex<double>>> azimuthal_exponentials(int m) {
  const double h = 1.0 / std::sqrt(2.0);
  if (m == 0) return {{0, 1.0}};
  if (m > 0) return {{m, h}, {-m, h}};
  return {{-m, {0.0, -h}}, {m, {0.0, h}}};
}

// Complex harmonic Y_le (Condon-Shortley) divided by N P^|e| exp(i e phi).
double condon_shortley_sign(int e) { return e > 0 ? parity_sign(e) : 1.0; }

double real_gaunt(int l1, int m1, int l2, int m2, int L, int M) {
  std::complex<double> azimuthal{};
  double polar = 0.0;
  bool coupled = false;
  for (const auto& [e1, c1] : azimuthal_exponentials(m1))
    for (const auto& [e2, c2] : azimuthal_exponentials(m2))
      for (const auto& [e3, c3] : azimuthal_exponentials(M)) {
        if (e1 + e2 + e3 != 0) continue;
        azimuthal += c1 * c2 * c3;
        if (coupled) continue;
        coupled = true;
        polar = condon_shortley_sign(e1) * condon_shortley_sign(e2) * condon_shortley_sign(e3) *
            std::sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) * (2.0 * L + 1.0) / (4.0 * pi)) *
            wigner_3j(l1, l2, L, 0, 0, 0) * wigner_3j(l1, l2, L, e1, e2, e3);
      }
  return coupled ? azimuthal.real() * polar : 0.0;
}

// Non-zero G(l1 m1, l2 m2 | L M), ordered by L then M. `inverse` is G divided by
// sum_{m1 m2} G^2, which 3j orthogonality fixes at (2l1+1)(2l2+1) w0^2 / (4 pi)
// with w0 = (l1 l2 L; 0 0 0) for every M. It maps an (L, M) moment back onto the
// (m1, m2) occupation entries, so sum_{m1 m2} value * inverse = 1.
std::vector<GauntEntry> real_gaunt_terms(int l1, int l2, int m1, int m2) {
  std::vector<GauntEntry> out;
  for (int L = std::abs(l1 - l2); L <= l1 + l2; L += 2) {
    const double w0 = wigner_3j(l1, l2, L, 0, 0, 0);
    const double norm = (2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) * w0 * w0 / (4.0 * pi);
    for (int M = -L; M <= L; ++M) {
      const double g = real_gaunt(l1, m1, l2, m2, L, M);
      if (std::abs(g) > 1e-15) out.push_back({L, M, g, g / norm});
    }
  }
  return out;
}

struct CompensationBasis {
  std::vector<double> sampled_r2;
  std::array<double, 2> q{}, amplitudes{};
  bool mesh_compensation_norm{false};
};

CompensationBasis compensation_function(const std::vector<double>& r, int l, double rmax) {
  CompensationBasis out;
  out.sampled_r2.assign(r.size(), 0.0);
  if (r.size() < 2 || rmax <= 0.0) return out;
  const auto roots = first_bessel_roots(l, 2);
  const double q1 = roots[0] / rmax, q2 = roots[1] / rmax;
  // SDFT_LCOMPAT is the spelling used by earlier releases.
  const char* norm_flag = std::getenv("SDFT_MESH_COMPENSATION_NORM");
  if (!norm_flag) norm_flag = std::getenv("SDFT_LCOMPAT");
  const bool mesh_compensation_norm = norm_flag && std::string(norm_flag) == "1";
  auto moment = [&](double q) {
    if (mesh_compensation_norm) {
      // Mesh normalisation: Simpson weights of the radial mesh with the
      // boundary point r==rmax excluded.  Keep this discrete functional
      // rather than silently replacing it with the continuous integral.
      const auto weights = radial_quadrature_weights(r);
      double value = 0.0;
      for (std::size_t ir = 0; ir < r.size(); ++ir) {
        if (r[ir] >= rmax) continue;
        value += weights[ir] * spherical_bessel(l, q * r[ir]) *
            std::pow(r[ir], l + 2);
      }
      return value;
    }
    // Default analytic normalisation. The identity
    // d[x^(l+2) j_(l+1)(x)]/dx=x^(l+2)j_l(x) gives the exact integral.
    return std::pow(rmax, l + 2) * spherical_bessel(l + 1, q * rmax) / q;
  };
  const double d1 = q1 * spherical_bessel_derivative(l, roots[0]);
  const double d2 = q2 * spherical_bessel_derivative(l, roots[1]);
  const double m1 = moment(q1), m2 = moment(q2);
  const double det = d1 * m2 - d2 * m1;
  if (std::abs(det) < 1e-14) throw std::runtime_error("singular PAW compensation basis l=" + std::to_string(l));
  const double a1 = -d2 / det, a2 = d1 / det;
  out.q = {q1, q2};
  out.amplitudes = {a1, a2};
  out.mesh_compensation_norm = mesh_compensation_norm;
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (r[i] > rmax) continue;
    const double q = a1 * spherical_bessel(l, q1 * r[i]) + a2 * spherical_bessel(l, q2 * r[i]);
    out.sampled_r2[i] = q * r[i] * r[i];
  }
  return out;
}
} // namespace

std::vector<PawGauntTerm> paw_real_gaunt_terms(int l1, int l2, int m1, int m2) {
  if (l1 < 0 || l2 < 0 || l1 > 6 || l2 > 6 || std::abs(m1) > l1 || std::abs(m2) > l2)
    throw std::invalid_argument("Gaunt coefficient quantum numbers are out of range");
  std::vector<PawGauntTerm> out;
  for (const auto& term : real_gaunt_terms(l1, l2, m1, m2))
    out.push_back({term.L, term.M, term.value, term.inverse});
  return out;
}

PawGauntTable paw_real_gaunt_table(int lmax) {
  if (lmax < 0 || lmax > 6)
    throw std::invalid_argument("Gaunt table lmax must be between 0 and 6");
  PawGauntTable table;
  table.lmax = lmax;
  const auto channels = static_cast<std::size_t>(lmax + 1) * static_cast<std::size_t>(lmax + 1);
  table.row_offset.reserve(channels * channels + 1);
  table.row_offset.push_back(0);
  for (int l1 = 0; l1 <= lmax; ++l1)
    for (int l2 = 0; l2 <= lmax; ++l2)
      for (int m1 = -l1; m1 <= l1; ++m1)
        for (int m2 = -l2; m2 <= l2; ++m2) {
          for (const auto& term : real_gaunt_terms(l1, l2, m1, m2)) {
            table.L.push_back(term.L);
            table.lm_index.push_back(term.L * (term.L + 1) + term.M);
            table.value.push_back(term.value);
            table.inverse.push_back(term.inverse);
          }
          table.row_offset.push_back(table.value.size());
        }
  return table;
}

std::size_t PawAugmentationData::aug_moment_index(std::size_t i, std::size_t j, int l) const {
  const auto n = channel_l.size();
  if (i >= n || j >= n || l < 0 || l > max_l) throw std::out_of_range("augmentation moment index");
  return (i * n + j) * static_cast<std::size_t>(max_l + 1) + static_cast<std::size_t>(l);
}

double PawAugmentationData::moment(std::size_t i, std::size_t j, int l) const {
  return aug_moment.at(aug_moment_index(i, j, l));
}

double PawAugmentationData::overlap_moment(std::size_t i, std::size_t j) const {
  const auto n = channel_l.size();
  if (i >= n || j >= n) throw std::out_of_range("overlap correction index");
  return overlap_correction.at(i * n + j);
}

PawAugmentationData build_paw_augmentation(const PawNumericCache& cache,
                                           const std::vector<int>& channel_l,
                                           int max_l) {
  const std::size_t n = channel_l.size();
  if (n == 0 || cache.ae_wavefunctions.size() != n || cache.pseudo_wavefunctions.size() != n)
    throw std::invalid_argument("augmentation channel count does not match partial waves");
  if (cache.radial_grid.size() < 3 || cache.radial_grid.size() != cache.ae_wavefunctions[0].size())
    throw std::invalid_argument("incomplete PAW radial mesh");
  for (const auto& wave : cache.ae_wavefunctions)
    if (wave.size() != cache.radial_grid.size()) throw std::invalid_argument("AE radial wave size mismatch");
  for (const auto& wave : cache.pseudo_wavefunctions)
    if (wave.size() != cache.radial_grid.size()) throw std::invalid_argument("PS radial wave size mismatch");
  // The radial-set header gives the augmentation radius. It is snapped to
  // the first logarithmic mesh node at or beyond it (relative tolerance
  // 1e-5), and that node bounds the compensation functions. It is independent
  // of the end of the radial integration mesh used for the moments below.
  const double requested_rmax = cache.radial_rmax > 0.0 ? cache.radial_rmax : cache.radial_grid.back();
  const auto edge = std::lower_bound(cache.radial_grid.begin(), cache.radial_grid.end(),
                                     requested_rmax * (1.0 - 1e-5));
  const double rmax = edge == cache.radial_grid.end() ? cache.radial_grid.back() : *edge;
  if (max_l < 0) {
    max_l = 0;
    for (int li : channel_l) for (int lj : channel_l) max_l = std::max(max_l, li + lj);
  }
  PawAugmentationData out;
  out.channel_l = channel_l;
  out.max_l = max_l;
  out.rmax = rmax;
  out.rdep = cache.radial_rdep > 0.0 ? cache.radial_rdep : cache.radial_grid.back();
  out.radial_grid = cache.radial_grid;
  out.aug_moment.assign(n * n * static_cast<std::size_t>(max_l + 1), 0.0);
  out.overlap_correction.assign(n * n, 0.0);
  // Sphere weights truncated at the first point with r - r_dep > -5e-3, with
  // their own end-anchored Simpson panels. The augmentation moments below use
  // the full mesh; the two cutoffs are deliberately not shared.
  std::size_t irdep = cache.radial_grid.size() - 1;
  const double rdep = out.rdep;
  for (std::size_t k = 0; k + 1 < cache.radial_grid.size(); ++k) {
    if (cache.radial_grid[k] - rdep > -5e-3) { irdep = k; break; }
  }
  const auto radial_weights = radial_quadrature_weights(cache.radial_grid);
  std::vector<double> truncated_weights(cache.radial_grid.size(), 0.0);
  if (irdep >= 2) {
    const double h = std::log(cache.radial_grid[1] / cache.radial_grid[0]);
    bool log_mesh = std::isfinite(h) && h > 0.0;
    for (std::size_t k = 2; k <= irdep; ++k)
      log_mesh = log_mesh && std::abs(std::log(cache.radial_grid[k] / cache.radial_grid[k - 1]) - h) < 2e-7;
    if (log_mesh) {
      for (std::size_t k = irdep + 1; k >= 3; k -= 2) {
        truncated_weights[k - 1] += cache.radial_grid[k - 1] * h / 3.0;
        truncated_weights[k - 2] = 4.0 * cache.radial_grid[k - 2] * h / 3.0;
        truncated_weights[k - 3] = cache.radial_grid[k - 3] * h / 3.0;
        if (k == 3) break;
      }
    } else {
      for (std::size_t k = 0; k < irdep; ++k) {
        const double half = 0.5 * (cache.radial_grid[k + 1] - cache.radial_grid[k]);
        truncated_weights[k] += half; truncated_weights[k + 1] += half;
      }
    }
  }
  // Q_ij^L = int (phi_i phi_j - tphi_i tphi_j) r^L dr over the complete
  // radial mesh, stored for every ordered channel pair.
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
    const int lmin = std::abs(channel_l[i] - channel_l[j]);
    const int lmax = std::min(max_l, channel_l[i] + channel_l[j]);
    for (int l = lmin; l <= lmax; l += 2) {
      double moment = 0.0;
      for (std::size_t k = 0; k < cache.radial_grid.size(); ++k) {
        const double f = (cache.ae_wavefunctions[i][k] * cache.ae_wavefunctions[j][k] -
                cache.pseudo_wavefunctions[i][k] * cache.pseudo_wavefunctions[j][k]) *
               std::pow(cache.radial_grid[k], l);
        moment += radial_weights[k] * f;
      }
      out.aug_moment[out.aug_moment_index(i, j, l)] = moment;
    }
  }
  // The PAW overlap S = 1 + sum_ij |p_i> q_ij <p_j| uses the spherical
  // moment q_ij = Q_ij^0 (AE minus PS) for equal-l pairs, on the full mesh
  // (Bloechl, PRB 50, 17953 (1994)). Using the AE norm here double-counts the
  // overlap and breaks the electron-number identity of the eigenproblem.
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
    if (channel_l[i] == channel_l[j]) out.overlap_correction[i * n + j] = out.moment(i, j, 0);
  out.compensation_r2.resize(static_cast<std::size_t>(max_l + 1));
  out.compensation_q.resize(static_cast<std::size_t>(max_l + 1));
  out.compensation_amplitudes.resize(static_cast<std::size_t>(max_l + 1));
  for (int l = 0; l <= max_l; ++l) {
    const auto basis = compensation_function(cache.radial_grid, l, rmax);
    out.mesh_compensation_norm = basis.mesh_compensation_norm;
    out.compensation_r2[static_cast<std::size_t>(l)] = basis.sampled_r2;
    out.compensation_q[static_cast<std::size_t>(l)] = basis.q;
    out.compensation_amplitudes[static_cast<std::size_t>(l)] = basis.amplitudes;
  }
  return out;
}

std::vector<std::complex<double>> paw_lm_moments_from_density(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension) {
  const std::size_t n = a.channel_count();
  std::size_t expanded = 0;
  for (int l : a.channel_l) expanded += static_cast<std::size_t>(2 * l + 1);
  if (expanded_dimension != expanded || density.size() != expanded * expanded)
    throw std::invalid_argument("on-site density matrix dimension mismatch");
  // Compensation multipoles: every ordered channel pair of the complex
  // occupation is contracted with Q_ij^L and the real Gaunt coefficients.
  // These complex lm moments define the compensation charge; the separate
  // real and imaginary transforms below serve the mixer representation.
  std::vector<std::complex<double>> out(
      static_cast<std::size_t>((a.max_l + 1) * (a.max_l + 1)), std::complex<double>{});
  std::vector<std::size_t> offset(n, 0);
  for (std::size_t i = 1; i < n; ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * a.channel_l[i - 1] + 1);
  using TableKey = std::array<int, 4>;
  static thread_local std::map<TableKey, std::vector<GauntEntry>> table_cache;
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
    for (int m = -a.channel_l[i]; m <= a.channel_l[i]; ++m)
      for (int mp = -a.channel_l[j]; mp <= a.channel_l[j]; ++mp) {
        const TableKey key{a.channel_l[i], a.channel_l[j], m, mp};
        auto it = table_cache.find(key);
        if (it == table_cache.end())
          it = table_cache.emplace(key, real_gaunt_terms(a.channel_l[i], a.channel_l[j], m, mp)).first;
        const std::size_t ii = offset[i] + static_cast<std::size_t>(m + a.channel_l[i]);
        const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + a.channel_l[j]);
        const auto value = density[ii * expanded_dimension + jj];
        for (const auto& entry : it->second)
          out[static_cast<std::size_t>(entry.L * entry.L + entry.M + entry.L)] +=
              a.moment(i, j, entry.L) * entry.value * value;
      }
  return out;
}

namespace {
std::vector<double> paw_lm_moment_component(const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density, std::size_t expanded_dimension,
    bool imaginary) {
  const std::size_t n = a.channel_count();
  std::size_t expanded = 0;
  for (int l : a.channel_l) expanded += static_cast<std::size_t>(2 * l + 1);
  if (expanded_dimension != expanded || density.size() != expanded * expanded)
    throw std::invalid_argument("on-site density matrix dimension mismatch");
  for (const auto& z : density)
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
      throw std::invalid_argument("lm-moment transform requires a finite density");
  std::vector<double> out(static_cast<std::size_t>((a.max_l + 1) * (a.max_l + 1)), 0.0);
  using TableKey = std::array<int, 4>;
  static thread_local std::map<TableKey, std::vector<GauntEntry>> table_cache;
  std::vector<std::size_t> offset(n, 0);
  for (std::size_t i = 1; i < n; ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * a.channel_l[i - 1] + 1);

  // Real or imaginary part over the channel upper triangle: off-diagonal
  // channel pairs carry a factor 2, and the imaginary part takes its sign
  // from the order of m and m'.
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) {
    const int lmin = std::abs(a.channel_l[i] - a.channel_l[j]);
    const int lmax = std::min(a.max_l, a.channel_l[i] + a.channel_l[j]);
    const double channel_factor = i == j ? 1.0 : 2.0;
    for (int m = -a.channel_l[i]; m <= a.channel_l[i]; ++m)
      for (int mp = -a.channel_l[j]; mp <= a.channel_l[j]; ++mp) {
        const TableKey key{a.channel_l[i], a.channel_l[j], m, mp};
        auto it = table_cache.find(key);
        if (it == table_cache.end())
          it = table_cache.emplace(key, real_gaunt_terms(a.channel_l[i], a.channel_l[j], m, mp)).first;
        const std::size_t ii = offset[i] + static_cast<std::size_t>(m + a.channel_l[i]);
        const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + a.channel_l[j]);
        const std::complex<double> value = density[ii * expanded + jj];
        const double component = imaginary ? ((m < mp) ? value.imag() : -value.imag()) : value.real();
        for (const auto& entry : it->second) {
          const int l = entry.L;
          if (l < lmin || l > lmax) continue;
          out[static_cast<std::size_t>(l * l + entry.M + l)] +=
              a.moment(i, j, l) * channel_factor * entry.value * component;
        }
      }
  }
  return out;
}
} // namespace

std::vector<double> paw_lm_moments_real(const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density, std::size_t expanded_dimension) {
  return paw_lm_moment_component(a, density, expanded_dimension, false);
}

std::vector<double> paw_lm_moments_imag(const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density, std::size_t expanded_dimension) {
  return paw_lm_moment_component(a, density, expanded_dimension, true);
}

namespace {
std::vector<double> paw_packed_channel_lm_component(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension,
    bool imaginary) {
  const std::size_t n = a.channel_count();
  std::size_t expanded = 0;
  for (int l : a.channel_l) expanded += static_cast<std::size_t>(2 * l + 1);
  if (expanded_dimension != expanded || density.size() != expanded * expanded)
    throw std::invalid_argument("on-site density matrix dimension mismatch");
  for (const auto& z : density)
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
      throw std::invalid_argument("lm-moment transform requires a finite density");

  std::vector<double> out(expanded * expanded, 0.0);
  std::vector<std::size_t> offset(n, 0);
  for (std::size_t i = 1; i < n; ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * a.channel_l[i - 1] + 1);
  using TableKey = std::array<int, 4>;
  static thread_local std::map<TableKey, std::vector<GauntEntry>> table_cache;
  std::size_t ibase = 0;
  // Per-channel-pair (L,M) projection in packed upper-triangle storage, with
  // the off-diagonal factor of two.  It deliberately omits Q_ij^L: these are
  // the unweighted channel coordinates used by the mixer.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const int li = a.channel_l[i], lj = a.channel_l[j];
      const int lmin = std::abs(li - lj);
      const int lmax = li + lj;
      const double channel_factor = i == j ? 1.0 : 2.0;
      const std::size_t block = static_cast<std::size_t>((2 * li + 1) * (2 * lj + 1));
      for (int m = -li; m <= li; ++m) for (int mp = -lj; mp <= lj; ++mp) {
        const TableKey key{li, lj, m, mp};
        auto it = table_cache.find(key);
        if (it == table_cache.end())
          it = table_cache.emplace(key, real_gaunt_terms(li, lj, m, mp)).first;
        const std::size_t ii = offset[i] + static_cast<std::size_t>(m + li);
        const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + lj);
        const std::complex<double> value = density[ii * expanded + jj];
        const double component = imaginary ? ((m < mp) ? value.imag() : -value.imag()) : value.real();
        for (const auto& entry : it->second) {
          if (entry.L < lmin || entry.L > lmax) continue;
          const std::size_t local = static_cast<std::size_t>(entry.L * entry.L + entry.M + entry.L - lmin * lmin);
          if (ibase + local >= out.size())
            throw std::logic_error("packed channel-lm storage exceeds dimension^2");
          out[ibase + local] += entry.value * channel_factor * component;
        }
      }
      ibase += block;
    }
  }
  return out;
}
} // namespace

std::vector<double> paw_packed_channel_lm_real(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension) {
  return paw_packed_channel_lm_component(a, density, expanded_dimension, false);
}

std::vector<double> paw_packed_channel_lm_imag(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension) {
  return paw_packed_channel_lm_component(a, density, expanded_dimension, true);
}

std::vector<std::complex<double>> paw_packed_channel_lm_complex(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension) {
  const auto real = paw_packed_channel_lm_real(a, density, expanded_dimension);
  const auto imag = paw_packed_channel_lm_imag(a, density, expanded_dimension);
  std::vector<std::complex<double>> out(real.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = {real[i], imag[i]};
  return out;
}

std::vector<std::complex<double>> paw_density_from_packed_channel_lm(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& packed_channel_lm,
    std::size_t expanded_dimension) {
  const std::size_t n = a.channel_count();
  std::size_t expanded = 0;
  for (int l : a.channel_l) expanded += static_cast<std::size_t>(2 * l + 1);
  if (expanded_dimension != expanded || packed_channel_lm.size() != expanded * expanded)
    throw std::invalid_argument("packed channel-lm dimension mismatch");
  std::vector<std::complex<double>> out(expanded * expanded, std::complex<double>{});
  std::vector<std::size_t> offset(n, 0);
  for (std::size_t i = 1; i < n; ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * a.channel_l[i - 1] + 1);
  using TableKey = std::array<int, 4>;
  static thread_local std::map<TableKey, std::vector<GauntEntry>> table_cache;
  std::size_t ibase = 0;
  // Inverse of the packed projection, using the Gaunt inverse weights.  The
  // off-diagonal block is divided by two before its Hermitian/anti-Hermitian
  // counterpart is filled, consistent with the upper-triangle storage.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const int li = a.channel_l[i], lj = a.channel_l[j];
      const int lmin = std::abs(li - lj);
      for (int m = -li; m <= li; ++m) for (int mp = -lj; mp <= lj; ++mp) {
        const TableKey key{li, lj, m, mp};
        auto it = table_cache.find(key);
        if (it == table_cache.end())
          it = table_cache.emplace(key, real_gaunt_terms(li, lj, m, mp)).first;
        std::complex<double> value{};
        for (const auto& entry : it->second) {
          const std::size_t local = static_cast<std::size_t>(entry.L * entry.L + entry.M + entry.L - lmin * lmin);
          if (ibase + local >= packed_channel_lm.size())
            throw std::logic_error("packed channel-lm inverse exceeds dimension^2");
          const std::complex<double> d = packed_channel_lm[ibase + local];
          const double sign = (m < mp) ? 1.0 : -1.0;
          value += entry.inverse * std::complex<double>{d.real(), sign * d.imag()};
        }
        if (i != j) value *= 0.5;
        const std::size_t row = offset[i] + static_cast<std::size_t>(m + li);
        const std::size_t col = offset[j] + static_cast<std::size_t>(mp + lj);
        out[row * expanded + col] = value;
        if (i != j) {
          const std::size_t r2 = offset[j] + static_cast<std::size_t>(mp + lj);
          const std::size_t c2 = offset[i] + static_cast<std::size_t>(m + li);
          out[r2 * expanded + c2] = std::conj(value);
        }
      }
      ibase += static_cast<std::size_t>((2 * li + 1) * (2 * lj + 1));
    }
  }
  return out;
}

std::vector<std::vector<std::complex<double>>> paw_compensation_density(
    const PawAugmentationData& a,
    const std::vector<std::complex<double>>& lm_moments) {
  if (lm_moments.size() != static_cast<std::size_t>((a.max_l + 1) * (a.max_l + 1)))
    throw std::invalid_argument("lm-moment dimension mismatch");
  std::vector<std::vector<std::complex<double>>> out(
      static_cast<std::size_t>((a.max_l + 1) * (a.max_l + 1)),
      std::vector<std::complex<double>>(a.radial_grid.size(), std::complex<double>{}));
  for (int l = 0; l <= a.max_l; ++l) for (int M = -l; M <= l; ++M) {
    const auto value = lm_moments[static_cast<std::size_t>(l * l + M + l)];
    for (std::size_t ir = 0; ir < a.radial_grid.size(); ++ir)
      out[static_cast<std::size_t>(l * l + M + l)][ir] = value * a.compensation_r2[static_cast<std::size_t>(l)][ir];
  }
  return out;
}

namespace {
using Complex = std::complex<double>;
constexpr double ev_per_hartree_closure = 27.211386245988;
constexpr double bohr_per_angstrom_closure = 1.8897261246257702;

Complex radial_integral_complex(const std::vector<double>& r, const std::vector<Complex>& f) {
  if (r.size() != f.size() || r.size() < 2) return Complex(0.0, 0.0);
  const auto weights = radial_quadrature_weights(r);
  Complex value(0.0, 0.0);
  for (std::size_t i = 0; i < r.size(); ++i) value += weights[i] * f[i];
  return value;
}

struct MultipoleHartreeResult {
  std::vector<Complex> potential, weighted_derivative;
  double energy{};
};

// Quadratic panels in log(r), including their half-panel primitives. The
// reverse accumulation is the exact transpose of the same linear operator.
std::vector<Complex> log_mesh_prefix(const std::vector<Complex>& f, double h3) {
  std::vector<Complex> prefix(f.size());
  for (std::size_t hi = 2; hi < f.size(); hi += 2) {
    prefix[hi] = prefix[hi - 2] + h3 * (f[hi - 2] + 4.0 * f[hi - 1] + f[hi]);
    prefix[hi - 1] = prefix[hi - 2] + h3 *
        (1.25 * f[hi - 2] + 2.0 * f[hi - 1] - 0.25 * f[hi]);
  }
  if (f.size() % 2 == 0) {
    const std::size_t hi = f.size() - 1;
    prefix[hi] = prefix[hi - 2] + h3 * (f[hi - 2] + 4.0 * f[hi - 1] + f[hi]);
  }
  return prefix;
}

std::vector<Complex> log_mesh_prefix_transpose(std::vector<Complex> response, double h3) {
  std::vector<Complex> derivative(response.size());
  if (response.size() % 2 == 0) {
    const std::size_t hi = response.size() - 1;
    response[hi - 2] += response[hi];
    derivative[hi - 2] += h3 * response[hi];
    derivative[hi - 1] += 4.0 * h3 * response[hi];
    derivative[hi] += h3 * response[hi];
  }
  const std::size_t last = (response.size() - 1) / 2 * 2;
  for (std::size_t hi = last; hi >= 2; hi -= 2) {
    const auto full = response[hi], half = response[hi - 1];
    response[hi - 2] += full + half;
    derivative[hi - 2] += h3 * (full + 1.25 * half);
    derivative[hi - 1] += h3 * (4.0 * full + 2.0 * half);
    derivative[hi] += h3 * (full - 0.25 * half);
  }
  return derivative;
}

MultipoleHartreeResult multipole_hartree(const std::vector<double>& r,
                                       const std::vector<Complex>& rho, int l,
                                       bool direct_mesh_potential = false) {
  const std::size_t n = r.size();
  MultipoleHartreeResult out;
  out.potential.assign(n, Complex(0.0, 0.0));
  out.weighted_derivative.assign(n, Complex(0.0, 0.0));
  if (rho.size() != n || n < 2) return out;
  const auto weights = radial_quadrature_weights(r);
  const double hlog = std::log(r[1] / r[0]);
  bool log_mesh = n >= 3 && r[0] > 0.0 && std::isfinite(hlog);
  for (std::size_t i = 2; i < n && log_mesh; ++i)
    log_mesh = std::abs(std::log(r[i] / r[i - 1]) - hlog) < 2e-7;
  const double factor = 4.0 * pi / (2.0 * l + 1.0);
  if (log_mesh) {
    std::vector<Complex> inner(n), outer(n), inner_response(n), outer_response(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double rl = std::pow(r[i], l);
      inner[n - 1 - i] = rho[i] * rl * r[i];
      outer[n - 1 - i] = rho[i] / rl;
    }
    inner = log_mesh_prefix(inner, hlog / 3.0);
    outer = log_mesh_prefix(outer, hlog / 3.0);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j = n - 1 - i;
      const double rl = std::pow(r[i], l), inverse = 1.0 / (r[i] * rl);
      out.potential[i] = factor * (outer[j] * rl + (inner.back() - inner[j]) * inverse);
      out.energy += 0.5 * (std::conj(rho[i]) * out.potential[i]).real() * weights[i];
      const Complex response = 0.5 * factor * weights[i] * rho[i];
      outer_response[j] += response * rl;
      inner_response.back() += response * inverse;
      inner_response[j] -= response * inverse;
      out.weighted_derivative[i] = 0.5 * weights[i] * out.potential[i];
    }
    if (direct_mesh_potential) {
      // Direct mode: keep the inward-integrated Hartree potential itself and
      // weight it by the Simpson weights before projection (no adjoint).
      for (std::size_t i = 0; i < n; ++i)
        out.weighted_derivative[i] = weights[i] * out.potential[i];
      return out;
    }
    const auto inner_adjoint = log_mesh_prefix_transpose(inner_response, hlog / 3.0);
    const auto outer_adjoint = log_mesh_prefix_transpose(outer_response, hlog / 3.0);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j = n - 1 - i;
      const double rl = std::pow(r[i], l);
      out.weighted_derivative[i] += inner_adjoint[j] * r[i] * rl + outer_adjoint[j] / rl;
      if (weights[i] > 0.0) out.potential[i] = out.weighted_derivative[i] / weights[i];
    }
    return out;
  }
  // Non-logarithmic caller meshes retain the symmetric nodal Coulomb kernel.
  std::vector<Complex> inner(n, Complex(0.0, 0.0)), outer(n, Complex(0.0, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    const double ri = std::max(r[i], 1e-14);
    const Complex term = weights[i] * rho[i] * std::pow(ri, l);
    inner[i] = term + (i ? inner[i - 1] : Complex(0.0, 0.0));
  }
  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    const double ri = std::max(r[i], 1e-14);
    const Complex term = weights[i] * rho[i] / std::pow(ri, l + 1);
    outer[i] = term + (i + 1 < n ? outer[i + 1] : Complex(0.0, 0.0));
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double ri = std::max(r[i], 1e-14);
    const Complex outer_exclusive = i + 1 < n ? outer[i + 1] : Complex(0.0, 0.0);
    out.potential[i] = (inner[i] / std::pow(ri, l + 1) +
                    std::pow(ri, l) * outer_exclusive) * factor;
    out.weighted_derivative[i] = weights[i] * out.potential[i];
    out.energy += 0.5 * (std::conj(rho[i]) * out.weighted_derivative[i]).real();
  }
  return out;
}

struct GradientStencil {
  std::size_t first{}, second{};
  double first_weight{}, second_weight{};
};

std::vector<GradientStencil> radial_gradient_stencils(const std::vector<double>& r) {
  if (r.size() < 2) return std::vector<GradientStencil>(r.size());
  const double hlog = std::log(r[1] / r[0]);
  bool logarithmic = r[0] > 0.0 && std::isfinite(hlog);
  for (std::size_t i = 2; i < r.size() && logarithmic; ++i)
    logarithmic = std::abs(std::log(r[i] / r[i - 1]) - hlog) < 2e-7;
  std::vector<GradientStencil> rows(r.size());
  for (std::size_t i = 0; i < r.size(); ++i) {
    auto& row = rows[i];
    // Radial derivative in x=log(r): first-order forward/backward endpoints,
    // centred three-point interior, then division by r.  The derivative is
    // taken in the logarithmic coordinate although the array is sampled at
    // physical radii.
    if (i == 0) {
      row.first = 0; row.second = 1;
    } else if (i + 1 == r.size()) {
      row.first = i - 1; row.second = i;
    } else {
      row.first = i - 1; row.second = i + 1;
    }
    const double denominator = logarithmic ?
        (i == 0 || i + 1 == r.size() ? hlog : 2.0 * hlog) * r[i] :
        (r[row.second] - r[row.first]);
    row.first_weight = -1.0 / std::max(denominator, 1e-14);
    row.second_weight = -row.first_weight;
  }
  return rows;
}

struct AngularXcResult {
  double energy{};
  std::size_t clipped_density_samples{};
  std::vector<std::vector<double>> weighted_derivative;
};

struct AngularPoint {double weight;std::array<double,3> direction;std::vector<double> y;std::vector<std::array<double,3>> gradient;};
std::vector<AngularPoint> angular_mesh(int lmax,int theta_count,int phi_count){
  std::vector<AngularPoint> points;
  for(int t=0;t<theta_count;++t){
    double z=std::cos(pi*(t+.75)/(theta_count+.5)),derivative=0;
    for(int iteration=0;iteration<32;++iteration){
      double previous=1,current=z;
      for(int l=2;l<=theta_count;++l){const double next=((2*l-1)*z*current-(l-1)*previous)/l;previous=current;current=next;}
      derivative=theta_count*(z*current-previous)/(z*z-1);const double shift=current/derivative;z-=shift;if(std::abs(shift)<2e-16)break;
    }
    const double weight=2/((1-z*z)*derivative*derivative),s=std::sqrt(std::max(0.,1-z*z));
    for(int p=0;p<phi_count;++p){
      const double phi=2*pi*p/phi_count,cp=std::cos(phi),sp=std::sin(phi);
      AngularPoint point{weight*2*pi/phi_count,{s*cp,s*sp,z},{},{}};
      for(int l=0;l<=lmax;++l)for(int m=-l;m<=l;++m){
        const auto h=real_harmonic_with_gradient(l,m,z,phi);point.y.push_back(h.value);
        point.gradient.push_back({z*cp*h.theta_derivative-sp*h.phi_derivative_over_sin,
            z*sp*h.theta_derivative+cp*h.phi_derivative_over_sin,-s*h.theta_derivative});
      }
      points.push_back(std::move(point));
    }
  }
  return points;
}
struct AngularDivergenceCoupling {std::size_t output,input;int axis;double angular,radial;};
struct AsphericalMesh {int theta,phi;std::size_t extended;std::vector<AngularPoint> points;std::vector<AngularDivergenceCoupling> coupling;};
const AsphericalMesh& aspherical_mesh(int max_l){
  static thread_local std::map<int,AsphericalMesh> cache;
  auto found=cache.find(max_l);if(found!=cache.end())return found->second;
  const int extended_l=std::min(6,max_l)+1;
  // The polar rule is the largest Gauss-Legendre order in stored_orders not
  // above the request, so a 9-point request evaluates an 8-point rule and
  // leaves a zero-weight final row. A true 9-point rule changes the nonlinear
  // XC quadrature for d-channel datasets (this matches VASP output). The
  // nodes are generated here by Newton iteration, not from a table.
  constexpr int stored_orders[]{1,2,3,4,5,6,8,10,12,14,16,20,24,32,48,64};
  const int requested_theta=3*(extended_l/2+1);
  int effective_theta=1;
  for(int order:stored_orders)if(order<=requested_theta)effective_theta=order;
  AsphericalMesh mesh{effective_theta,3*(extended_l+1),std::size_t((extended_l+1)*(extended_l+1)),{}, {}};
  mesh.points=angular_mesh(extended_l,mesh.theta,mesh.phi);
  const auto exact=angular_mesh(extended_l,2*(extended_l/2+1),2*(extended_l+1));
  const std::size_t outputs=std::size_t((max_l+1)*(max_l+1));
  for(std::size_t i=0;i<outputs;++i)for(std::size_t j=0;j<mesh.extended;++j)for(int axis=0;axis<3;++axis){
    double angular=0,radial=0;
    for(const auto& point:exact){angular+=point.weight*point.y[i]*point.gradient[j][axis];radial+=point.weight*point.y[i]*point.y[j]*point.direction[axis];}
    if(std::abs(angular)>1e-4||std::abs(radial)>1e-4)mesh.coupling.push_back({i,j,axis,angular,radial});
  }
  return cache.emplace(max_l,std::move(mesh)).first->second;
}

// Aspherical GGA: the Cartesian gradient response is expanded to l=max_l+1
// (at most 7), then a sparse angular-divergence matrix and the
// radial derivative act on it. A spherical lm=0 replacement in this branch
// drops the density-dependent coupling of higher multipoles (notably the
// tetrahedral l=3 term for d-channel PAWs).
AngularXcResult aspherical_pbe_xc(const std::vector<double>& r,
    const std::vector<std::vector<Complex>>& rho,int max_l,const std::vector<double>& weights,
    const std::vector<GradientStencil>& stencils,double length_factor,bool discrete_adjoint=false){
  const auto& mesh=aspherical_mesh(max_l);const std::size_t nr=r.size(),outputs=rho.size(),ext=mesh.extended;
  struct Partial {double energy{};std::size_t clipped{};std::vector<double> mu;};
  std::vector<Partial> partial(mesh.theta);
  for(auto& p:partial)p.mu.assign(4*ext*nr,0.);
  const double gradient_floor=1e-10/std::pow(length_factor,4);
  PerformanceTimer timer("paw_aspherical_pbe");
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
  for(int theta=0;theta<mesh.theta;++theta){
    auto& result=partial[theta];std::vector<double> density(nr);std::vector<std::array<double,3>> tangential(nr);
    for(int phi=0;phi<mesh.phi;++phi){
      const auto& point=mesh.points[theta*mesh.phi+phi];
      for(std::size_t ir=0;ir<nr;++ir){
        double charge=0;std::array<double,3> gradient{};
        for(std::size_t lm=0;lm<outputs;++lm){charge+=rho[lm][ir].real()*point.y[lm];for(int axis=0;axis<3;++axis)gradient[axis]+=rho[lm][ir].real()*point.gradient[lm][axis];}
        density[ir]=charge/(r[ir]*r[ir]);
        for(int axis=0;axis<3;++axis)tangential[ir][axis]=gradient[axis]/(r[ir]*r[ir]*r[ir]);
      }
      for(std::size_t ir=0;ir<nr;++ir){
        const auto& stencil=stencils[ir];const double radial=stencil.first_weight*density[stencil.first]+stencil.second_weight*density[stencil.second];
        std::array<double,3> gradient{};double squared=0;
        for(int axis=0;axis<3;++axis){gradient[axis]=radial*point.direction[axis]+tangential[ir][axis];squared+=gradient[axis]*gradient[axis];}
        const double magnitude=std::sqrt(squared),n=std::max(1e-99,density[ir]);if(density[ir]<=0)++result.clipped;
        const auto xc=pbe_unpolarized_lda_tail(n,magnitude);
        result.energy+=weights[ir]*point.weight*r[ir]*r[ir]*xc.energy_density;
        if(discrete_adjoint) {
          // Differentiate the sampled energy itself, including its radial
          // weights and endpoint stencil. The continuum gradient-divergence
          // is not the transpose of this finite logarithmic-mesh operator.
          const double measure=weights[ir]*point.weight*r[ir]*r[ir];
          const double coefficient=xc.gradient_squared_coefficient;
          double radial_response=0.;
          for(int axis=0;axis<3;++axis)radial_response+=coefficient*gradient[axis]*point.direction[axis];
          const double local=density[ir]>1e-99?xc.exchange_potential+xc.correlation_potential:0.;
          for(std::size_t lm=0;lm<outputs;++lm) {
            result.mu[lm*nr+ir]+=weights[ir]*point.weight*local*point.y[lm];
            for(int axis=0;axis<3;++axis)result.mu[lm*nr+ir]+=
                measure*coefficient*gradient[axis]*point.gradient[lm][axis]/(r[ir]*r[ir]*r[ir]);
            result.mu[lm*nr+stencil.first]+=
                measure*radial_response*stencil.first_weight*point.y[lm]/(r[stencil.first]*r[stencil.first]);
            result.mu[lm*nr+stencil.second]+=
                measure*radial_response*stencil.second_weight*point.y[lm]/(r[stencil.second]*r[stencil.second]);
          }
          continue;
        }
        const double coefficient=xc.gradient_squared_coefficient*(magnitude/std::max(magnitude,gradient_floor));
        for(std::size_t lm=0;lm<ext;++lm){
          const double factor=point.weight*point.y[lm];
          result.mu[lm*nr+ir]+=(xc.exchange_potential+xc.correlation_potential)*factor;
          for(int axis=0;axis<3;++axis)result.mu[((axis+1)*ext+lm)*nr+ir]+=coefficient*gradient[axis]*factor;
        }
      }
    }
  }
  std::vector<double> mu(4*ext*nr,0.);AngularXcResult out;
  for(const auto& p:partial){out.energy+=p.energy;out.clipped_density_samples+=p.clipped;for(std::size_t i=0;i<mu.size();++i)mu[i]+=p.mu[i];}
  out.weighted_derivative.assign(outputs,std::vector<double>(nr));
  for(std::size_t lm=0;lm<outputs;++lm)for(std::size_t ir=0;ir<nr;++ir)out.weighted_derivative[lm][ir]=mu[lm*nr+ir];
  if(discrete_adjoint)return out;
  for(const auto& coupling:mesh.coupling){
    const auto offset=((coupling.axis+1)*ext+coupling.input)*nr;
    for(std::size_t ir=0;ir<nr;++ir){const auto& stencil=stencils[ir];
      const double derivative=stencil.first_weight*mu[offset+stencil.first]+stencil.second_weight*mu[offset+stencil.second];
      out.weighted_derivative[coupling.output][ir]-=coupling.angular*mu[offset+ir]/r[ir]+coupling.radial*derivative;
    }
  }
  for(std::size_t lm=0;lm<outputs;++lm)for(std::size_t ir=0;ir<nr;++ir)out.weighted_derivative[lm][ir]*=weights[ir];
  return out;
}

// Direct spherical GGA potential: the same radial derivative stencil is applied
// to the gradient response, v = df/dn - (d/dr + 2/r) df/dn', instead of its
// transpose. The distinction vanishes in the continuum but matters on the
// logarithmic PAW mesh, especially for the sharply peaked AE core density.
std::vector<double> radial_direct_gga_potential(
    const std::vector<double>& r,
    const std::vector<Complex>& rho_l0,
    const std::vector<double>& radial_weights,
    const std::vector<GradientStencil>& gradient_rows) {
  const double scale = 2.0 * std::sqrt(pi);
  if (r.size() != rho_l0.size() || r.size() != radial_weights.size() ||
      r.size() != gradient_rows.size())
    throw std::invalid_argument("radial GGA mesh dimensions differ");
  const std::size_t n = r.size();
  std::vector<double> density(n, 0.0), gradient(n, 0.0), response(n, 0.0),
      response_gradient(n, 0.0), potential(n, 0.0);
  // Spherical-only GGA (as with LASPH=.FALSE.): only the lm=0 coefficient
  // enters.  The real density is used here; the complex lm moments are kept by
  // the surrounding PAW transform and projected through the real channel.
  for (std::size_t i = 0; i < n; ++i) {
    const double rr = std::max(r[i], 1e-14);
    density[i] = std::max(rho_l0[i].real() / (scale * rr * rr), 1e-99);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto& row = gradient_rows[i];
    const double drho = row.first_weight * density[row.first] +
                        row.second_weight * density[row.second];
    gradient[i] = drho;
    const auto pbe = pbe_unpolarized_lda_tail(density[i], std::abs(drho));
    response[i] = pbe.exchange_potential + pbe.correlation_potential;
    // df/d(drho/dr).  Keeping the signed gradient is required when the
    // radial derivative acts on this response a second time.
    response_gradient[i] = pbe.gradient_squared_coefficient * drho;
  }
  // Second application of the same radial derivative, including its one-sided
  // endpoint stencils; it is intentionally not the variational transpose.
  for (std::size_t i = 0; i < n; ++i) {
    const auto& row = gradient_rows[i];
    const double divergence = row.first_weight * response_gradient[row.first] +
                              row.second_weight * response_gradient[row.second];
    const double vxc = response[i] - divergence -
                       2.0 * response_gradient[i] / std::max(r[i], 1e-14);
    potential[i] = scale * vxc;
  }
  return potential;
}

AngularXcResult angular_pbe_xc(
    const std::vector<double>& r,
    const std::vector<std::vector<Complex>>& rho,
    int max_l,
    const std::vector<double>& radial_weights,
    const std::vector<GradientStencil>& gradient_rows) {
  constexpr double density_floor = 1e-14;
  constexpr int ntheta = 12, nphi = 48;
  constexpr double z[ntheta] = {
      -0.9815606342467191, -0.9041172563704749, -0.7699026741943047,
      -0.5873179542866175, -0.3678314989981802, -0.1252334085114689,
       0.1252334085114689,  0.3678314989981802,  0.5873179542866175,
       0.7699026741943047,  0.9041172563704749,  0.9815606342467191};
  constexpr double theta_weight[ntheta] = {
      0.0471753363865118, 0.1069393259953184, 0.1600783285433462,
      0.2031674267230659, 0.2334925365383548, 0.2491470458134029,
      0.2491470458134029, 0.2334925365383548, 0.2031674267230659,
      0.1600783285433462, 0.1069393259953184, 0.0471753363865118};
  const std::size_t nr = r.size(), nlm = rho.size();
  // Angular basis depends only on lmax, not the atom or SCF density.
  using HarmonicTable=std::vector<std::vector<RealHarmonic>>;
  static thread_local std::map<int,HarmonicTable> harmonic_cache;
  auto found=harmonic_cache.find(max_l);
  if(found==harmonic_cache.end()){
    HarmonicTable table(ntheta*nphi,std::vector<RealHarmonic>(nlm));
    for(int it=0;it<ntheta;++it)for(int ip=0;ip<nphi;++ip)
      for(int l=0;l<=max_l;++l)for(int m=-l;m<=l;++m)
        table[it*nphi+ip][static_cast<std::size_t>(l*l+m+l)]=real_harmonic_with_gradient(l,m,z[it],2.0*pi*ip/nphi);
    found=harmonic_cache.emplace(max_l,std::move(table)).first;
  }
  const auto& table=found->second;
  std::array<AngularXcResult,ntheta> partial;
  std::array<long double,ntheta> energies{};
  for(auto& part:partial)part.weighted_derivative.assign(nlm,std::vector<double>(nr,0.0));
  PerformanceTimer angular_timer("paw_angular_pbe");
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
  for(int it=0;it<ntheta;++it){
    auto& out=partial[it];auto& energy=energies[it];
    std::vector<double> n(nr),angular_theta(nr),angular_phi(nr),radial_derivative(nr);
    std::vector<double> local_response(nr),radial_response(nr);
    for(int ip=0;ip<nphi;++ip){
    const double angular_weight=theta_weight[it]*(2.0*pi/nphi);
    const auto& harmonics=table[it*nphi+ip];
    std::vector<bool> active(nr, false);
    for (std::size_t ir = 0; ir < nr; ++ir) {
      double charge = 0.0, theta_charge = 0.0, phi_charge = 0.0;
      for (std::size_t lm = 0; lm < nlm; ++lm) {
        charge += rho[lm][ir].real() * harmonics[lm].value;
        theta_charge += rho[lm][ir].real() * harmonics[lm].theta_derivative;
        phi_charge += rho[lm][ir].real() * harmonics[lm].phi_derivative_over_sin;
      }
      const double rr = std::max(r[ir], 1e-14);
      const double raw_density = charge / (rr * rr);
      active[ir] = raw_density > density_floor;
      if (!active[ir]) ++out.clipped_density_samples;
      n[ir] = std::max(raw_density, density_floor);
      angular_theta[ir] = active[ir] ? theta_charge / (rr * rr * rr) : 0.0;
      angular_phi[ir] = active[ir] ? phi_charge / (rr * rr * rr) : 0.0;
    }
    for (std::size_t ir = 0; ir < nr; ++ir) {
      const auto& row = gradient_rows[ir];
      radial_derivative[ir] = row.first_weight * n[row.first] +
                              row.second_weight * n[row.second];
      const double grad = std::sqrt(radial_derivative[ir] * radial_derivative[ir] +
                                    angular_theta[ir] * angular_theta[ir] +
                                    angular_phi[ir] * angular_phi[ir]);
      const auto pbe = pbe_unpolarized(n[ir], grad);
      const double measure = radial_weights[ir] * angular_weight * r[ir] * r[ir];
      energy += static_cast<long double>(measure) * pbe.energy_density;
      local_response[ir] = measure * (pbe.exchange_potential + pbe.correlation_potential);
      radial_response[ir] = measure * pbe.gradient_squared_coefficient * radial_derivative[ir];
      const double angular_factor = active[ir] ?
          measure * pbe.gradient_squared_coefficient /
              std::max(r[ir] * r[ir] * r[ir], 1e-42) : 0.0;
      for (std::size_t lm = 0; lm < nlm; ++lm)
        out.weighted_derivative[lm][ir] += angular_factor *
            (angular_theta[ir] * harmonics[lm].theta_derivative +
             angular_phi[ir] * harmonics[lm].phi_derivative_over_sin);
    }
    for (std::size_t ir = 0; ir < nr; ++ir) {
      const auto& row = gradient_rows[ir];
      local_response[row.first] += row.first_weight * radial_response[ir];
      local_response[row.second] += row.second_weight * radial_response[ir];
    }
    for (std::size_t ir = 0; ir < nr; ++ir) {
      if (!active[ir]) continue;
      const double factor = local_response[ir] / std::max(r[ir] * r[ir], 1e-28);
      for (std::size_t lm = 0; lm < nlm; ++lm)
        out.weighted_derivative[lm][ir] += factor * harmonics[lm].value;
    }
  }
  }
  AngularXcResult out;out.weighted_derivative.assign(nlm,std::vector<double>(nr,0.));
  long double energy=0;
  // Fixed 12-chunk reduction independent of the number of CPU threads.
  for(int it=0;it<ntheta;++it){energy+=energies[it];out.clipped_density_samples+=partial[it].clipped_density_samples;
    for(std::size_t lm=0;lm<nlm;++lm)for(std::size_t ir=0;ir<nr;++ir)
      out.weighted_derivative[lm][ir]+=partial[it].weighted_derivative[lm][ir];}
  out.energy = static_cast<double>(energy);
  return out;
}

std::array<AngularXcResult,2> angular_collinear_pbe_xc(const std::vector<double>& r,
    const std::array<std::vector<std::vector<Complex>>,2>& rho,int max_l,
    const std::vector<double>& weights,const std::vector<GradientStencil>& stencils,
    double length_factor,bool aspherical,bool discrete_adjoint=false){
  const auto& mesh=aspherical_mesh(max_l);const auto points=aspherical?mesh.points:angular_mesh(max_l,12,48);
  const std::size_t nr=r.size(),outputs=rho[0].size(),ext=aspherical?mesh.extended:outputs;
  std::array<AngularXcResult,2> out;
  std::array<std::vector<double>,2> mu;
  for(int s=0;s<2;++s){mu[s].assign(4*ext*nr,0.);out[s].weighted_derivative.assign(outputs,std::vector<double>(nr));}
  for(const auto& point:points){
    std::array<std::vector<double>,2> density;std::array<std::vector<std::array<double,3>>,2> tangential;
    for(int s=0;s<2;++s){density[s].resize(nr);tangential[s].resize(nr);
      for(std::size_t i=0;i<nr;++i){double charge=0;std::array<double,3> g{};
        for(std::size_t lm=0;lm<outputs;++lm){charge+=rho[s][lm][i].real()*point.y[lm];
          for(int a=0;a<3;++a)g[a]+=rho[s][lm][i].real()*point.gradient[lm][a];}
        density[s][i]=charge/(r[i]*r[i]);for(int a=0;a<3;++a)tangential[s][i][a]=g[a]/(r[i]*r[i]*r[i]);}}
    for(std::size_t i=0;i<nr;++i){const auto& row=stencils[i];std::array<std::array<double,3>,2> g{};
      for(int s=0;s<2;++s){const double radial=row.first_weight*density[s][row.first]+row.second_weight*density[s][row.second];
        for(int a=0;a<3;++a)g[s][a]=radial*point.direction[a]+tangential[s][i][a];}
      const auto xc=pbe_collinear(std::max(1e-99,density[0][i]),std::max(1e-99,density[1][i]),g[0],g[1],aspherical);
      const double measure=weights[i]*point.weight*r[i]*r[i];
      for(int s=0;s<2;++s){out[s].energy+=.5*measure*xc.energy_density;
        if(density[s][i]<=0)++out[s].clipped_density_samples;
        if(aspherical&&!discrete_adjoint){
          const double magnitude=std::sqrt(g[s][0]*g[s][0]+g[s][1]*g[s][1]+g[s][2]*g[s][2]);
          const double factor=magnitude/std::max(magnitude,.5e-10/std::pow(length_factor,4));
          for(std::size_t lm=0;lm<ext;++lm){const double angular=point.weight*point.y[lm];
            mu[s][lm*nr+i]+=xc.density_potential[s]*angular;
            for(int a=0;a<3;++a)mu[s][((a+1)*ext+lm)*nr+i]+=factor*xc.gradient_response[s][a]*angular;}
        }else{
          double radial_response=0;for(int a=0;a<3;++a)radial_response+=xc.gradient_response[s][a]*point.direction[a];
          for(std::size_t lm=0;lm<outputs;++lm){auto& v=out[s].weighted_derivative[lm];
            if(density[s][i]>1e-99)v[i]+=weights[i]*point.weight*xc.density_potential[s]*point.y[lm];
            for(int a=0;a<3;++a)v[i]+=measure*xc.gradient_response[s][a]*point.gradient[lm][a]/(r[i]*r[i]*r[i]);
            v[row.first]+=measure*radial_response*row.first_weight*point.y[lm]/(r[row.first]*r[row.first]);
            v[row.second]+=measure*radial_response*row.second_weight*point.y[lm]/(r[row.second]*r[row.second]);}
        }
      }
    }
  }
  if(aspherical&&!discrete_adjoint)for(int s=0;s<2;++s){
    for(std::size_t lm=0;lm<outputs;++lm)for(std::size_t i=0;i<nr;++i)out[s].weighted_derivative[lm][i]=mu[s][lm*nr+i];
    for(const auto& c:mesh.coupling){const auto offset=((c.axis+1)*ext+c.input)*nr;
      for(std::size_t i=0;i<nr;++i){const auto& row=stencils[i];
        const double derivative=row.first_weight*mu[s][offset+row.first]+row.second_weight*mu[s][offset+row.second];
        out[s].weighted_derivative[c.output][i]-=c.angular*mu[s][offset+i]/r[i]+c.radial*derivative;}}
    for(std::size_t lm=0;lm<outputs;++lm)for(std::size_t i=0;i<nr;++i)out[s].weighted_derivative[lm][i]*=weights[i];
  }
  return out;
}

}

std::vector<std::complex<double>> paw_atomic_reference_density(
    const PawNumericCache& cache, const std::vector<int>& channel_l) {
  const std::size_t channels = channel_l.size();
  if (cache.occupancies.size() != channels * channels)
    throw std::invalid_argument("PAW atomic reference occupancy tensor is incomplete");
  std::vector<std::size_t> offsets(channels);
  std::size_t dim = 0;
  for (std::size_t i = 0; i < channels; ++i) {
    offsets[i] = dim;
    dim += static_cast<std::size_t>(2 * channel_l[i] + 1);
  }
  std::vector<std::complex<double>> density(dim * dim, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < channels; ++i) for (std::size_t j = 0; j < channels; ++j) {
    const double occupancy = cache.occupancies[j * channels + i];
    if (!std::isfinite(occupancy)) throw std::invalid_argument("nonfinite PAW atomic occupancy");
    if (channel_l[i] != channel_l[j]) {
      if (std::abs(occupancy) > 1e-12)
        throw std::invalid_argument("PAW atomic occupancy couples unequal angular channels");
      continue;
    }
    for (int m = 0; m < 2 * channel_l[i] + 1; ++m)
      density[(offsets[i] + m) * dim + offsets[j] + m] = occupancy;
  }
  return density;
}

PawRadialProjectionResult paw_project_radial_potentials(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    const std::vector<std::vector<double>>& potential_ae_ev,
    const std::vector<std::vector<double>>& potential_ps_ev) {
  const auto augmentation = build_paw_augmentation(cache, channel_l);
  const std::size_t nr = cache.radial_grid.size();
  const std::size_t dim = std::accumulate(channel_l.begin(), channel_l.end(), std::size_t{0},
      [](std::size_t total, int l) { return total + static_cast<std::size_t>(2 * l + 1); });
  if (potential_ae_ev.size() != static_cast<std::size_t>((augmentation.max_l + 1) * (augmentation.max_l + 1)) ||
      potential_ps_ev.size() != potential_ae_ev.size())
    throw std::invalid_argument("radial potential multipole dimensions mismatch");
  for (std::size_t lm = 0; lm < potential_ae_ev.size(); ++lm)
    if (potential_ae_ev[lm].size() != nr || potential_ps_ev[lm].size() != nr)
      throw std::invalid_argument("radial potential mesh dimension mismatch");
  const auto weights = radial_quadrature_weights(cache.radial_grid);
  std::vector<std::size_t> offsets(channel_l.size());
  for (std::size_t i = 1; i < channel_l.size(); ++i)
    offsets[i] = offsets[i - 1] + static_cast<std::size_t>(2 * channel_l[i - 1] + 1);
  PawRadialProjectionResult out;
  const auto matrix_size = dim * dim;
  out.ae_partial_ev.assign(matrix_size, {});
  out.ps_partial_ev.assign(matrix_size, {});
  out.ps_augmentation_ev.assign(matrix_size, {});
  out.total_ev.assign(matrix_size, {});
  for (std::size_t i = 0; i < channel_l.size(); ++i) for (std::size_t j = 0; j < channel_l.size(); ++j)
    for (int m = -channel_l[i]; m <= channel_l[i]; ++m)
      for (int mp = -channel_l[j]; mp <= channel_l[j]; ++mp) {
        const std::size_t ii = offsets[i] + static_cast<std::size_t>(m + channel_l[i]);
        const std::size_t jj = offsets[j] + static_cast<std::size_t>(mp + channel_l[j]);
        const std::size_t index = ii * dim + jj;
        for (int l = std::abs(channel_l[i] - channel_l[j]);
             l <= std::min(augmentation.max_l, channel_l[i] + channel_l[j]); l += 2)
          for (int M = -l; M <= l; ++M) {
            const auto entries = real_gaunt_terms(channel_l[i], channel_l[j], m, mp);
            double coefficient = 0.0;
            for (const auto& entry : entries)
              if (entry.L == l && entry.M == M) { coefficient = entry.value; break; }
            if (std::abs(coefficient) < 1e-13) continue;
            const auto lm = static_cast<std::size_t>(l * l + M + l);
            Complex ae{}, ps{}, aug{};
            for (std::size_t ir = 0; ir < nr; ++ir) {
              const double pair_ae = cache.ae_wavefunctions[i][ir] * cache.ae_wavefunctions[j][ir];
              const double pair_ps = cache.pseudo_wavefunctions[i][ir] * cache.pseudo_wavefunctions[j][ir];
              ae += weights[ir] * pair_ae * potential_ae_ev[lm][ir];
              ps += weights[ir] * pair_ps * potential_ps_ev[lm][ir];
              aug += weights[ir] * augmentation.moment(i, j, l) *
                  augmentation.compensation_r2[static_cast<std::size_t>(l)][ir] * potential_ps_ev[lm][ir];
            }
            out.ae_partial_ev[index] += coefficient * ae;
            out.ps_partial_ev[index] -= coefficient * ps;
            out.ps_augmentation_ev[index] -= coefficient * aug;
          }
      }
  for (std::size_t i = 0; i < matrix_size; ++i)
    out.total_ev[i] = out.ae_partial_ev[i] + out.ps_partial_ev[i] + out.ps_augmentation_ev[i];
  return out;
}

std::vector<std::complex<double>> paw_project_radial_potentials_packed(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    const std::vector<std::vector<double>>& potential_ae_ev,
    const std::vector<std::vector<double>>& potential_ps_ev) {
  const auto augmentation = build_paw_augmentation(cache, channel_l);
  const std::size_t nr = cache.radial_grid.size();
  const auto weights = radial_quadrature_weights(cache.radial_grid);
  const std::size_t packed = std::accumulate(channel_l.begin(), channel_l.end(), std::size_t{0},
      [](std::size_t total, int l) { return total + static_cast<std::size_t>(2 * l + 1); });
  if (potential_ae_ev.size() != static_cast<std::size_t>((augmentation.max_l + 1) * (augmentation.max_l + 1)) ||
      potential_ps_ev.size() != potential_ae_ev.size())
    throw std::invalid_argument("packed radial potential multipole dimensions mismatch");
  for (const auto& row : potential_ae_ev) if (row.size() != nr) throw std::invalid_argument("packed AE potential mesh mismatch");
  for (const auto& row : potential_ps_ev) if (row.size() != nr) throw std::invalid_argument("packed PS potential mesh mismatch");
  std::vector<std::complex<double>> out(packed * packed, std::complex<double>{});
  std::size_t ibase = 0;
  for (std::size_t i = 0; i < channel_l.size(); ++i) {
    for (std::size_t j = i; j < channel_l.size(); ++j) {
      const int li = channel_l[i], lj = channel_l[j];
      const int lmin = std::abs(li - lj);
      const int lmax = std::min(augmentation.max_l, li + lj);
      const std::size_t block = static_cast<std::size_t>((2 * li + 1) * (2 * lj + 1));
      for (int l = lmin; l <= lmax; ++l) for (int M = -l; M <= l; ++M) {
        const std::size_t local = static_cast<std::size_t>(l * l + M + l - lmin * lmin);
        if (((l - lmin) & 1) != 0) {
          out[ibase + local] = std::complex<double>{};
          continue;
        }
        const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
        std::complex<double> value{};
        for (std::size_t ir = 0; ir < nr; ++ir) {
          const double ae = cache.ae_wavefunctions[i][ir] * cache.ae_wavefunctions[j][ir] * potential_ae_ev[lm][ir];
          const double ps = cache.pseudo_wavefunctions[i][ir] * cache.pseudo_wavefunctions[j][ir] * potential_ps_ev[lm][ir];
          const double aug = augmentation.moment(i, j, l) * augmentation.compensation_r2[static_cast<std::size_t>(l)][ir] * potential_ps_ev[lm][ir];
          value += weights[ir] * (ae - ps - aug);
        }
        out[ibase + local] = value;
      }
      ibase += block;
    }
  }
  return out;
}

std::vector<std::complex<double>> paw_project_fourier_potential(
    const PawNumericCache& source_cache, const std::vector<int>& channel_l,
    const std::array<Vec3, 3>& lattice_bohr, Vec3 fractional_position,
    const FourierDensity& potential_hartree) {
  if (potential_hartree.empty())
    throw std::invalid_argument("Dgrid projection requires a non-empty Fourier potential");
  // Grid term of D_ij: integrate the total smooth potential against the
  // compensation shapes g_L(r) Y_LM, then contract with Q_ij^L and the Gaunt
  // coefficients (Kresse & Joubert, PRB 59, 1758 (1999), Eq. 25). The AE-PS
  // radial response belongs only to the one-centre projection; inserting
  // AE-PS-compensation here double counts it and flips the grid term's sign.
  const auto augmentation = build_paw_augmentation(source_cache, channel_l);
  std::vector<std::tuple<int, int, int>> keys;
  keys.reserve(potential_hartree.size());
  for (const auto& [key, value] : potential_hartree) {
    (void)value;
    keys.push_back(key);
  }
  PawCompensationOperator op(augmentation, lattice_bohr, fractional_position, keys);
  auto out = op.project_potential(potential_hartree);
  for (auto& value : out) value *= 27.211386245988;
  return out;
}

PawOneCenterClosureResult paw_one_center_closure(
    const PawNumericCache& source_cache,
    const std::vector<int>& channel_l,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension,
    int spin_multiplicity,
    PawRadialLengthUnit radial_unit,
    const PawOneCenterReferencePotentials* reference,
    PawRadialPotentialMode potential_mode, AtomicUnitConvention constants) {
  // The input density is already summed over spin occupations. A spin factor
  // applied here would double count charge, and polarized PBE needs a separate
  // spin-density functional rather than multiplying this unpolarized result.
  if (spin_multiplicity != 1)
    throw std::invalid_argument("PAW closure expects the total spin-summed density and spin_multiplicity=1");
  // Unless set explicitly, every mode except DiscreteVariational uses the
  // legacy unit constants of matched VASP references (bohr = 0.529177249
  // Angstrom, Rydberg = 13.605826 eV); DiscreteVariational keeps CODATA.
  // Units are fixed before any reference potential is added.
  if(constants==AtomicUnitConvention::MatchRadialMode)
    constants=potential_mode!=PawRadialPotentialMode::DiscreteVariational ?
        AtomicUnitConvention::VaspLegacy : AtomicUnitConvention::Codata;
  const auto units=atomic_units(constants);
  if(reference && reference->constants!=AtomicUnitConvention::MatchRadialMode &&
     reference->constants!=constants)
    throw std::invalid_argument("PAW reference and closure atomic-unit conventions differ");
  const double length_factor=units.bohr_per_angstrom;
  const double energy_factor=units.ev_per_hartree;
  // POTCAR radial arrays are stored on an Angstrom mesh. The Coulomb and PBE
  // routines below are atomic-unit kernels, so convert the entire radial
  // representation, not just the final energy. Reduced waves obey
  // integral u^2 dr = occupancy and rho_lm(r)=r^2 n_lm(r) has units length^-1.
  auto cache = source_cache;
  const double radial_scale = radial_unit == PawRadialLengthUnit::Angstrom ?
      length_factor : 1.0;
  for (double& radius : cache.radial_grid) radius *= radial_scale;
  cache.radial_rmax *= radial_scale;
  const double wave_factor = 1.0 / std::sqrt(radial_scale);
  for (auto& wave : cache.ae_wavefunctions) for (double& x : wave) x *= wave_factor;
  for (auto& wave : cache.pseudo_wavefunctions) for (double& x : wave) x *= wave_factor;
  for (double& x : cache.core_charge_density) x /= radial_scale;
  for (double& x : cache.core_charge_density_pseudized) x /= radial_scale;
  if (channel_l.empty() || density.size() != expanded_dimension * expanded_dimension)
    throw std::invalid_argument("PAW closure density dimensions are invalid");
  for (std::size_t i = 0; i < expanded_dimension; ++i) for (std::size_t j = 0; j < expanded_dimension; ++j) {
    const auto value = density[i * expanded_dimension + j];
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
        std::abs(value - std::conj(density[j * expanded_dimension + i])) > 1e-10)
      throw std::invalid_argument("PAW closure requires a finite Hermitian density matrix");
  }
  std::size_t expanded_expected = 0;
  for (int l : channel_l) expanded_expected += static_cast<std::size_t>(2 * l + 1);
  if (expanded_expected != expanded_dimension)
    throw std::invalid_argument("PAW closure channel expansion mismatch");
  const auto augmentation = build_paw_augmentation(cache, channel_l);
  const auto& r = cache.radial_grid;
  const std::size_t nr = r.size();
  if (reference) {
    for (const auto* potential : {&reference->ae_reference_potential_ev, &reference->ps_reference_potential_ev}) {
      if (potential->size() != nr)
        throw std::invalid_argument("PAW one-centre reference potential grid mismatch");
      for (double value : *potential) if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite PAW one-centre reference potential");
    }
    if (!reference->projected_reference_d_matrix_ev.empty() &&
        reference->projected_reference_d_matrix_ev.size() != density.size())
      throw std::invalid_argument("PAW projected reference matrix dimension mismatch");
    const auto& matrix = reference->projected_reference_d_matrix_ev;
    for (std::size_t i = 0; i < expanded_dimension && !matrix.empty(); ++i)
      for (std::size_t j = 0; j < expanded_dimension; ++j) {
        const auto value = matrix[i * expanded_dimension + j];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
            std::abs(value - std::conj(matrix[j * expanded_dimension + i])) > 1e-10)
          throw std::invalid_argument("PAW projected reference matrix must be finite and Hermitian");
      }
  }
  const std::size_t lm_count = static_cast<std::size_t>((augmentation.max_l + 1) * (augmentation.max_l + 1));
  std::vector<std::vector<Complex>> rho_ae(lm_count, std::vector<Complex>(nr, Complex(0.0, 0.0)));
  std::vector<std::vector<Complex>> rho_ps(lm_count, std::vector<Complex>(nr, Complex(0.0, 0.0)));
  std::vector<std::size_t> offset(channel_l.size(), 0);
  for (std::size_t i = 1; i < channel_l.size(); ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * channel_l[i - 1] + 1);
  for (std::size_t i = 0; i < channel_l.size(); ++i) for (std::size_t j = 0; j < channel_l.size(); ++j) {
    const int lmin = std::abs(channel_l[i] - channel_l[j]);
    const int lmax = std::min(augmentation.max_l, channel_l[i] + channel_l[j]);
      for (int l = lmin; l <= lmax; l += 2) for (int m = -channel_l[i]; m <= channel_l[i]; ++m)
      for (int mp = -channel_l[j]; mp <= channel_l[j]; ++mp) for (int M = -l; M <= l; ++M) {
        static thread_local std::map<std::array<int, 4>, std::vector<GauntEntry>> gaunt_cache;
        const std::array<int, 4> key{channel_l[i], channel_l[j], m, mp};
        auto it = gaunt_cache.find(key);
        if (it == gaunt_cache.end())
          it = gaunt_cache.emplace(key, real_gaunt_terms(channel_l[i], channel_l[j], m, mp)).first;
        double c = 0.0;
        for (const auto& entry : it->second)
          if (entry.L == l && entry.M == M) { c = entry.value; break; }
        if (std::abs(c) < 1e-13) continue;
        const std::size_t ii = offset[i] + static_cast<std::size_t>(m + channel_l[i]);
        const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + channel_l[j]);
        const Complex dij = density[ii * expanded_dimension + jj] * c;
        const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
        for (std::size_t ir = 0; ir < nr; ++ir) {
          rho_ae[lm][ir] += dij * cache.ae_wavefunctions[i][ir] * cache.ae_wavefunctions[j][ir];
          rho_ps[lm][ir] += dij * cache.pseudo_wavefunctions[i][ir] * cache.pseudo_wavefunctions[j][ir];
        }
      }
  }
  // The compensation charge uses the complex lm moments.  The imaginary
  // channel transform is kept separately for mixer diagnostics and complex
  // occupation reconstruction.
  const auto lm_moments = paw_lm_moments_from_density(augmentation, density, expanded_dimension);
  const auto lm_imag = paw_lm_moments_imag(augmentation, density, expanded_dimension);
  const auto compensation = paw_compensation_density(augmentation, lm_moments);
  for (std::size_t lm = 0; lm < lm_count; ++lm)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ps[lm][ir] += compensation[lm][ir];

  // The frozen core is spherical and enters the AE/PS XC branches. The XC
  // density is (rho_00(r) + rho_core(r))/(sqrt(4*pi) r^2); the POTCAR core
  // array is already the Y_00 coefficient times r^2, so it is not rescaled.
  if (rho_ae.empty()) throw std::runtime_error("PAW closure radial arrays are empty");
  if (cache.core_charge_density.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ae[0][ir] += cache.core_charge_density[ir];
  if (cache.core_charge_density_pseudized.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ps[0][ir] += cache.core_charge_density_pseudized[ir];

  std::vector<std::vector<Complex>> rho_ae_h(lm_count, std::vector<Complex>(nr));
  std::vector<std::vector<Complex>> rho_ps_h(lm_count, std::vector<Complex>(nr));
  // Hartree excludes frozen core but includes all valence and compensation
  // multipoles.  Reconstruct the valence arrays by subtracting the core
  // contribution only in the spherical channel.
  rho_ae_h = rho_ae;
  rho_ps_h = rho_ps;
  if (cache.core_charge_density.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ae_h[0][ir] -= cache.core_charge_density[ir];
  if (cache.core_charge_density_pseudized.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ps_h[0][ir] -= cache.core_charge_density_pseudized[ir];
  std::vector<std::vector<Complex>> vae(lm_count, std::vector<Complex>(nr)), vps(lm_count, std::vector<Complex>(nr));
  const auto zero_potential = std::vector<std::vector<Complex>>(lm_count, std::vector<Complex>(nr));
  std::vector<std::vector<Complex>> hartree_vae = zero_potential, hartree_vps = zero_potential;
  std::vector<std::vector<Complex>> weighted_vae(lm_count), weighted_vps(lm_count);
  double ae_h = 0.0, ps_h = 0.0;
  for (int l = 0; l <= augmentation.max_l; ++l) for (int M = -l; M <= l; ++M) {
    const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
    auto ae = multipole_hartree(r, rho_ae_h[lm], l,
        potential_mode != PawRadialPotentialMode::DiscreteVariational &&
        potential_mode != PawRadialPotentialMode::AsphericalAdjoint);
    auto ps = multipole_hartree(r, rho_ps_h[lm], l,
        potential_mode != PawRadialPotentialMode::DiscreteVariational &&
        potential_mode != PawRadialPotentialMode::AsphericalAdjoint);
    vae[lm] = std::move(ae.potential);
    vps[lm] = std::move(ps.potential);
    hartree_vae[lm] = vae[lm];
    hartree_vps[lm] = vps[lm];
    weighted_vae[lm] = std::move(ae.weighted_derivative);
    weighted_vps[lm] = std::move(ps.weighted_derivative);
    ae_h += ae.energy;
    ps_h += ps.energy;
  }
  // The XC energy and potential share radial and angular quadrature and the
  // transpose of the radial finite-difference gradient. Angular derivatives
  // act on real Y_lm and include both tangential gradient components.
  const auto radial_weights = radial_quadrature_weights(r);
  const auto gradient_stencils = radial_gradient_stencils(r);
  const bool discrete_adjoint = potential_mode == PawRadialPotentialMode::AsphericalAdjoint;
  const bool aspherical = potential_mode == PawRadialPotentialMode::AsphericalDirect || discrete_adjoint;
  const auto ae_xc = aspherical ? aspherical_pbe_xc(r, rho_ae, augmentation.max_l,
      radial_weights, gradient_stencils, length_factor, discrete_adjoint) :
      angular_pbe_xc(r, rho_ae, augmentation.max_l, radial_weights, gradient_stencils);
  const auto ps_xc = aspherical ? aspherical_pbe_xc(r, rho_ps, augmentation.max_l,
      radial_weights, gradient_stencils, length_factor, discrete_adjoint) :
      angular_pbe_xc(r, rho_ps, augmentation.max_l, radial_weights, gradient_stencils);
  // Store w*V, the derivative with respect to the radial density coefficient,
  // directly for projection. Simpson weights give the first endpoint zero weight
  // on an even grid, but the transpose gradient can still contribute there.
  // Dividing by w and then integrating would lose that endpoint derivative.
  for (std::size_t lm = 0; lm < lm_count; ++lm) for (std::size_t ir = 0; ir < nr; ++ir) {
    weighted_vae[lm][ir] += ae_xc.weighted_derivative[lm][ir];
    weighted_vps[lm][ir] += ps_xc.weighted_derivative[lm][ir];
    if (radial_weights[ir] > 0.0) {
      vae[lm][ir] += ae_xc.weighted_derivative[lm][ir] / radial_weights[ir];
      vps[lm][ir] += ps_xc.weighted_derivative[lm][ir] / radial_weights[ir];
    }
  }
  // Direct spherical GGA potential (as used for matched VASP references with
  // LASPH=.FALSE.).  The energy remains the same PBE functional; only the
  // mesh-level potential derivative follows radial_direct_gga_potential.
  const bool radial_direct_gga = potential_mode == PawRadialPotentialMode::RadialDirect;
  if (radial_direct_gga) {
    const auto ae_gga = radial_direct_gga_potential(
        r, rho_ae[0], radial_weights, gradient_stencils);
    const auto ps_gga = radial_direct_gga_potential(
        r, rho_ps[0], radial_weights, gradient_stencils);
    for (std::size_t ir = 0; ir < nr; ++ir) {
      // angular_pbe_xc already contains both LDA and GGA in lm=0.  Replace
      // that complete XC contribution with the direct radial result while
      // retaining the independently computed Hartree response.
      const auto ae_hartree = weighted_vae[0][ir] - ae_xc.weighted_derivative[0][ir];
      const auto ps_hartree = weighted_vps[0][ir] - ps_xc.weighted_derivative[0][ir];
      vae[0][ir] += ae_gga[ir] - (radial_weights[ir] > 0.0 ?
          ae_xc.weighted_derivative[0][ir] / radial_weights[ir] : 0.0);
      vps[0][ir] += ps_gga[ir] - (radial_weights[ir] > 0.0 ?
          ps_xc.weighted_derivative[0][ir] / radial_weights[ir] : 0.0);
      weighted_vae[0][ir] = ae_hartree + radial_weights[ir] * ae_gga[ir];
      weighted_vps[0][ir] = ps_hartree + radial_weights[ir] * ps_gga[ir];
    }
  }

  PawOneCenterClosureResult out;
  out.direct_radial_potential = radial_direct_gga || (aspherical && !discrete_adjoint);
  out.lm_moments = std::move(lm_moments);
  // Report lm moments in the archive convention: Q_L is in Angstrom^L.
  for (int l = 0; l <= augmentation.max_l; ++l)
    for (int m = -l; m <= l; ++m)
      out.lm_moments[static_cast<std::size_t>(l * l + m + l)] /=
          std::pow(length_factor, l);
  out.d_matrix_ev.assign(expanded_dimension * expanded_dimension, Complex(0.0, 0.0));
  out.reference_d_matrix_ev.assign(expanded_dimension * expanded_dimension, Complex(0.0, 0.0));
  out.rho_ae_r2.assign(lm_count, std::vector<double>(nr));
  out.rho_ps_r2.assign(lm_count, std::vector<double>(nr));
  out.potential_ae_ev.assign(lm_count, std::vector<double>(nr));
  out.potential_ps_ev.assign(lm_count, std::vector<double>(nr));
  out.hartree_ae_ev.assign(lm_count, std::vector<double>(nr));
  out.hartree_ps_ev.assign(lm_count, std::vector<double>(nr));
  out.xc_potential_ae_ev.assign(lm_count, std::vector<double>(nr));
  out.xc_potential_ps_ev.assign(lm_count, std::vector<double>(nr));
  const double archive_density_scale = length_factor;
  for (std::size_t lm = 0; lm < lm_count; ++lm) for (std::size_t ir = 0; ir < nr; ++ir) {
    out.rho_ae_r2[lm][ir] = rho_ae_h[lm][ir].real() * archive_density_scale;
    out.rho_ps_r2[lm][ir] = rho_ps_h[lm][ir].real() * archive_density_scale;
    out.potential_ae_ev[lm][ir] = vae[lm][ir].real() * energy_factor;
    out.potential_ps_ev[lm][ir] = vps[lm][ir].real() * energy_factor;
    out.hartree_ae_ev[lm][ir] = hartree_vae[lm][ir].real() * energy_factor;
    out.hartree_ps_ev[lm][ir] = hartree_vps[lm][ir].real() * energy_factor;
    out.xc_potential_ae_ev[lm][ir] = out.potential_ae_ev[lm][ir] - out.hartree_ae_ev[lm][ir];
    out.xc_potential_ps_ev[lm][ir] = out.potential_ps_ev[lm][ir] - out.hartree_ps_ev[lm][ir];
    if (reference && lm == 0) {
      out.potential_ae_ev[lm][ir] += std::sqrt(4.0 * pi) * reference->ae_reference_potential_ev[ir];
      out.potential_ps_ev[lm][ir] += std::sqrt(4.0 * pi) * reference->ps_reference_potential_ev[ir];
    }
  }
  double maxv = 0.0;
  for (const auto& v : vae) for (const auto& x : v) maxv = std::max(maxv, std::abs(x) * energy_factor);
  for (const auto& v : vps) for (const auto& x : v) maxv = std::max(maxv, std::abs(x) * energy_factor);
  out.max_potential_ev = maxv;
  for (std::size_t i = 0; i < channel_l.size(); ++i) for (std::size_t j = 0; j < channel_l.size(); ++j) {
    const int lmin = std::abs(channel_l[i] - channel_l[j]);
    const int lmax = std::min(augmentation.max_l, channel_l[i] + channel_l[j]);
    for (int m = -channel_l[i]; m <= channel_l[i]; ++m) for (int mp = -channel_l[j]; mp <= channel_l[j]; ++mp) {
      const std::size_t ii = offset[i] + static_cast<std::size_t>(m + channel_l[i]);
      const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + channel_l[j]);
      Complex value(0.0, 0.0), reference_value(0.0, 0.0);
      Complex ae_value(0.0, 0.0), ps_value(0.0, 0.0);
      for (int l = lmin; l <= lmax; l += 2) for (int M = -l; M <= l; ++M) {
        static thread_local std::map<std::array<int, 4>, std::vector<GauntEntry>> gaunt_cache;
        const std::array<int, 4> key{channel_l[i], channel_l[j], m, mp};
        auto it = gaunt_cache.find(key);
        if (it == gaunt_cache.end())
          it = gaunt_cache.emplace(key, real_gaunt_terms(channel_l[i], channel_l[j], m, mp)).first;
        double c = 0.0;
        for (const auto& entry : it->second)
          if (entry.L == l && entry.M == M) { c = entry.value; break; }
        if (std::abs(c) < 1e-13) continue;
        const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
        Complex radial_response(0.0, 0.0);
        for (std::size_t ir = 0; ir < nr; ++ir) {
          // The PS pair density carries its compensation term Q_ij^L g_L(r),
          // so the radial response is AE - (PS + compensation).
          const double pair_ae = cache.ae_wavefunctions[i][ir] * cache.ae_wavefunctions[j][ir];
          const double pair_ps_comp = cache.pseudo_wavefunctions[i][ir] * cache.pseudo_wavefunctions[j][ir] +
              augmentation.moment(i, j, l) * augmentation.compensation_r2[static_cast<std::size_t>(l)][ir];
          radial_response += pair_ae * weighted_vae[lm][ir] - pair_ps_comp * weighted_vps[lm][ir];
          ae_value += c * pair_ae * weighted_vae[lm][ir];
          ps_value += c * pair_ps_comp * weighted_vps[lm][ir];
          if (reference && l == 0) {
            reference_value += c * radial_weights[ir] * std::sqrt(4.0 * pi) *
                (pair_ae * reference->ae_reference_potential_ev[ir] -
                 pair_ps_comp * reference->ps_reference_potential_ev[ir]);
          }
        }
        value += c * radial_response;
      }
      const auto index = ii * expanded_dimension + jj;
      if (reference && !reference->projected_reference_d_matrix_ev.empty())
        reference_value = reference->projected_reference_d_matrix_ev[index];
      out.reference_d_matrix_ev[index] = reference_value;
      out.d_matrix_ev[index] = value * energy_factor + reference_value;
      out.ae_potential_expectation_ev +=
          (std::conj(density[index]) * ae_value).real() * energy_factor;
      out.ps_potential_expectation_ev +=
          (std::conj(density[index]) * ps_value).real() * energy_factor;
    }
  }
  // Enforce Hermiticity after quadrature roundoff, preserving the variational
  // matrix used in both the Hamiltonian and expectation-value correction.
  for (std::size_t i = 0; i < expanded_dimension; ++i) for (std::size_t j = i + 1; j < expanded_dimension; ++j) {
    const Complex v = 0.5 * (out.d_matrix_ev[i * expanded_dimension + j] +
                             std::conj(out.d_matrix_ev[j * expanded_dimension + i]));
    out.d_matrix_ev[i * expanded_dimension + j] = v;
    out.d_matrix_ev[j * expanded_dimension + i] = std::conj(v);
  }
  // A Hermitian PAW operator has real diagonal entries.  Complex lm moments
  // are kept for the non-Gamma density transform, but roundoff or asymmetric
  // radial contractions must never leak an imaginary diagonal into D_ij.
  for (std::size_t i = 0; i < expanded_dimension; ++i)
    out.d_matrix_ev[i * expanded_dimension + i] =
        Complex(out.d_matrix_ev[i * expanded_dimension + i].real(), 0.0);
  Complex expectation(0.0, 0.0);
  for (std::size_t i = 0; i < expanded_dimension; ++i) for (std::size_t j = 0; j < expanded_dimension; ++j)
    {
      const auto index = i * expanded_dimension + j;
      expectation += std::conj(density[index]) * out.d_matrix_ev[index];
    }
  for (std::size_t i = 0; i < expanded_dimension; ++i)
    for (std::size_t j = 0; j < expanded_dimension; ++j) {
      const auto index = i * expanded_dimension + j;
      out.reference_linear_energy_ev +=
          (std::conj(density[index]) * out.reference_d_matrix_ev[index]).real();
    }
  out.ae_energy_ev = (ae_h + ae_xc.energy) * energy_factor;
  out.ps_energy_ev = (ps_h + ps_xc.energy) * energy_factor;
  out.ae_hartree_ev = ae_h * energy_factor;
  out.ps_hartree_ev = ps_h * energy_factor;
  out.ae_xc_ev = ae_xc.energy * energy_factor;
  out.ps_xc_ev = ps_xc.energy * energy_factor;
  out.ae_double_counting_ev = out.ae_energy_ev - out.ae_potential_expectation_ev;
  out.ps_double_counting_ev = out.ps_energy_ev - out.ps_potential_expectation_ev;
  out.energy_difference_ev = out.ae_energy_ev - out.ps_energy_ev + out.reference_linear_energy_ev;
  // The AE XC energy includes the frozen core, so the AE one-centre energy
  // carries the core-only XC energy of the reference atom as a constant.
  // Attach it to the closure result so the periodic SCF assembler subtracts
  // it exactly once, alongside the AE/PS double-counting difference.
  out.core_xc_energy_ev = reference ? reference->core_xc_energy_ev : 0.0;
  out.potential_expectation_ev = expectation.real();
  // The direct mesh derivative reproduces a reference discretization; it is
  // not the exact adjoint of this code's finite-grid energy quadrature.
  out.spherical_gga_variational = !(radial_direct_gga || (aspherical && !discrete_adjoint));
  out.aspherical_gga_variational = !(radial_direct_gga || (aspherical && !discrete_adjoint));
  out.mesh_compensation_norm = augmentation.mesh_compensation_norm;
  out.lm_moment_imag = lm_imag;
  out.ae_clipped_density_samples = ae_xc.clipped_density_samples;
  out.ps_clipped_density_samples = ps_xc.clipped_density_samples;
  return out;
}

static PawOneCenterClosureResult collinear_closure_channel(
    const PawNumericCache& source_cache,
    const std::vector<int>& channel_l,
    const std::vector<std::complex<double>>& density,
    std::size_t expanded_dimension,
    const std::vector<Complex>& other_density, int spin_channel,
    PawRadialLengthUnit radial_unit,
    const PawOneCenterReferencePotentials* reference,
    PawRadialPotentialMode potential_mode, AtomicUnitConvention constants) {
  if(other_density.size()!=density.size())throw std::invalid_argument("spin onsite dimensions differ");
  for(std::size_t i=0;i<expanded_dimension;++i)for(std::size_t j=0;j<expanded_dimension;++j)
    if(!std::isfinite(other_density[i*expanded_dimension+j].real())||!std::isfinite(other_density[i*expanded_dimension+j].imag())||
       std::abs(other_density[i*expanded_dimension+j]-std::conj(other_density[j*expanded_dimension+i]))>1e-10)
      throw std::invalid_argument("spin onsite requires finite Hermitian matrices");
  if(potential_mode==PawRadialPotentialMode::RadialDirect)throw std::invalid_argument("collinear PAW requires aspherical or variational XC");
  // Unless set explicitly, every mode except DiscreteVariational uses the
  // legacy unit constants of matched VASP references (bohr = 0.529177249
  // Angstrom, Rydberg = 13.605826 eV); DiscreteVariational keeps CODATA.
  // Units are fixed before any reference potential is added.
  if(constants==AtomicUnitConvention::MatchRadialMode)
    constants=potential_mode!=PawRadialPotentialMode::DiscreteVariational ?
        AtomicUnitConvention::VaspLegacy : AtomicUnitConvention::Codata;
  const auto units=atomic_units(constants);
  if(reference && reference->constants!=AtomicUnitConvention::MatchRadialMode &&
     reference->constants!=constants)
    throw std::invalid_argument("PAW reference and closure atomic-unit conventions differ");
  const double length_factor=units.bohr_per_angstrom;
  const double energy_factor=units.ev_per_hartree;
  // POTCAR radial arrays are stored on an Angstrom mesh. The Coulomb and PBE
  // routines below are atomic-unit kernels, so convert the entire radial
  // representation, not just the final energy. Reduced waves obey
  // integral u^2 dr = occupancy and rho_lm(r)=r^2 n_lm(r) has units length^-1.
  auto cache = source_cache;
  const double radial_scale = radial_unit == PawRadialLengthUnit::Angstrom ?
      length_factor : 1.0;
  for (double& radius : cache.radial_grid) radius *= radial_scale;
  cache.radial_rmax *= radial_scale;
  const double wave_factor = 1.0 / std::sqrt(radial_scale);
  for (auto& wave : cache.ae_wavefunctions) for (double& x : wave) x *= wave_factor;
  for (auto& wave : cache.pseudo_wavefunctions) for (double& x : wave) x *= wave_factor;
  for (double& x : cache.core_charge_density) x /= radial_scale;
  for (double& x : cache.core_charge_density_pseudized) x /= radial_scale;
  if (channel_l.empty() || density.size() != expanded_dimension * expanded_dimension)
    throw std::invalid_argument("PAW closure density dimensions are invalid");
  for (std::size_t i = 0; i < expanded_dimension; ++i) for (std::size_t j = 0; j < expanded_dimension; ++j) {
    const auto value = density[i * expanded_dimension + j];
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
        std::abs(value - std::conj(density[j * expanded_dimension + i])) > 1e-10)
      throw std::invalid_argument("PAW closure requires a finite Hermitian density matrix");
  }
  std::size_t expanded_expected = 0;
  for (int l : channel_l) expanded_expected += static_cast<std::size_t>(2 * l + 1);
  if (expanded_expected != expanded_dimension)
    throw std::invalid_argument("PAW closure channel expansion mismatch");
  const auto augmentation = build_paw_augmentation(cache, channel_l);
  const auto& r = cache.radial_grid;
  const std::size_t nr = r.size();
  if (reference) {
    for (const auto* potential : {&reference->ae_reference_potential_ev, &reference->ps_reference_potential_ev}) {
      if (potential->size() != nr)
        throw std::invalid_argument("PAW one-centre reference potential grid mismatch");
      for (double value : *potential) if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite PAW one-centre reference potential");
    }
    if (!reference->projected_reference_d_matrix_ev.empty() &&
        reference->projected_reference_d_matrix_ev.size() != density.size())
      throw std::invalid_argument("PAW projected reference matrix dimension mismatch");
    const auto& matrix = reference->projected_reference_d_matrix_ev;
    for (std::size_t i = 0; i < expanded_dimension && !matrix.empty(); ++i)
      for (std::size_t j = 0; j < expanded_dimension; ++j) {
        const auto value = matrix[i * expanded_dimension + j];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
            std::abs(value - std::conj(matrix[j * expanded_dimension + i])) > 1e-10)
          throw std::invalid_argument("PAW projected reference matrix must be finite and Hermitian");
      }
  }
  const std::size_t lm_count = static_cast<std::size_t>((augmentation.max_l + 1) * (augmentation.max_l + 1));
  std::vector<std::vector<Complex>> rho_ae(lm_count, std::vector<Complex>(nr, Complex(0.0, 0.0)));
  std::vector<std::vector<Complex>> rho_ps(lm_count, std::vector<Complex>(nr, Complex(0.0, 0.0)));
  auto other_ae=rho_ae,other_ps=rho_ps;
  std::vector<std::size_t> offset(channel_l.size(), 0);
  for (std::size_t i = 1; i < channel_l.size(); ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * channel_l[i - 1] + 1);
  for (std::size_t i = 0; i < channel_l.size(); ++i) for (std::size_t j = 0; j < channel_l.size(); ++j) {
    const int lmin = std::abs(channel_l[i] - channel_l[j]);
    const int lmax = std::min(augmentation.max_l, channel_l[i] + channel_l[j]);
      for (int l = lmin; l <= lmax; l += 2) for (int m = -channel_l[i]; m <= channel_l[i]; ++m)
      for (int mp = -channel_l[j]; mp <= channel_l[j]; ++mp) for (int M = -l; M <= l; ++M) {
        static thread_local std::map<std::array<int, 4>, std::vector<GauntEntry>> gaunt_cache;
        const std::array<int, 4> key{channel_l[i], channel_l[j], m, mp};
        auto it = gaunt_cache.find(key);
        if (it == gaunt_cache.end())
          it = gaunt_cache.emplace(key, real_gaunt_terms(channel_l[i], channel_l[j], m, mp)).first;
        double c = 0.0;
        for (const auto& entry : it->second)
          if (entry.L == l && entry.M == M) { c = entry.value; break; }
        if (std::abs(c) < 1e-13) continue;
        const std::size_t ii = offset[i] + static_cast<std::size_t>(m + channel_l[i]);
        const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + channel_l[j]);
        const Complex dij = density[ii * expanded_dimension + jj] * c;
        const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
        for (std::size_t ir = 0; ir < nr; ++ir) {
          rho_ae[lm][ir] += dij * cache.ae_wavefunctions[i][ir] * cache.ae_wavefunctions[j][ir];
          rho_ps[lm][ir] += dij * cache.pseudo_wavefunctions[i][ir] * cache.pseudo_wavefunctions[j][ir];
          other_ae[lm][ir]+=other_density[ii*expanded_dimension+jj]*c*cache.ae_wavefunctions[i][ir]*cache.ae_wavefunctions[j][ir];
          other_ps[lm][ir]+=other_density[ii*expanded_dimension+jj]*c*cache.pseudo_wavefunctions[i][ir]*cache.pseudo_wavefunctions[j][ir];
        }
      }
  }
  // The compensation charge uses the complex lm moments.  The imaginary
  // channel transform is kept separately for mixer diagnostics and complex
  // occupation reconstruction.
  const auto lm_moments = paw_lm_moments_from_density(augmentation, density, expanded_dimension);
  const auto lm_imag = paw_lm_moments_imag(augmentation, density, expanded_dimension);
  const auto compensation = paw_compensation_density(augmentation, lm_moments);
  for (std::size_t lm = 0; lm < lm_count; ++lm)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ps[lm][ir] += compensation[lm][ir];

  const auto other_lm_moments=paw_lm_moments_from_density(augmentation,other_density,expanded_dimension);
  const auto other_comp=paw_compensation_density(augmentation,other_lm_moments);
  for(std::size_t lm=0;lm<lm_count;++lm)for(std::size_t ir=0;ir<nr;++ir)other_ps[lm][ir]+=other_comp[lm][ir];
  // The frozen core is spherical and enters the AE/PS XC branches. The XC
  // density is (rho_00(r) + rho_core(r))/(sqrt(4*pi) r^2); the POTCAR core
  // array is already the Y_00 coefficient times r^2, so it is not rescaled.
  if (rho_ae.empty()) throw std::runtime_error("PAW closure radial arrays are empty");
  if (cache.core_charge_density.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) {rho_ae[0][ir]+=.5*cache.core_charge_density[ir];other_ae[0][ir]+=.5*cache.core_charge_density[ir];}
  if (cache.core_charge_density_pseudized.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) {rho_ps[0][ir]+=.5*cache.core_charge_density_pseudized[ir];other_ps[0][ir]+=.5*cache.core_charge_density_pseudized[ir];}

  std::vector<std::vector<Complex>> rho_ae_h(lm_count, std::vector<Complex>(nr));
  std::vector<std::vector<Complex>> rho_ps_h(lm_count, std::vector<Complex>(nr));
  // Hartree excludes frozen core but includes all valence and compensation
  // multipoles.  Reconstruct the valence arrays by subtracting the core
  // contribution only in the spherical channel.
  rho_ae_h = rho_ae;
  rho_ps_h = rho_ps;
  for(std::size_t lm=0;lm<lm_count;++lm)for(std::size_t ir=0;ir<nr;++ir){rho_ae_h[lm][ir]+=other_ae[lm][ir];rho_ps_h[lm][ir]+=other_ps[lm][ir];}
  if (cache.core_charge_density.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ae_h[0][ir] -= cache.core_charge_density[ir];
  if (cache.core_charge_density_pseudized.size() == nr)
    for (std::size_t ir = 0; ir < nr; ++ir) rho_ps_h[0][ir] -= cache.core_charge_density_pseudized[ir];
  std::vector<std::vector<Complex>> vae(lm_count, std::vector<Complex>(nr)), vps(lm_count, std::vector<Complex>(nr));
  const auto zero_potential = std::vector<std::vector<Complex>>(lm_count, std::vector<Complex>(nr));
  std::vector<std::vector<Complex>> hartree_vae = zero_potential, hartree_vps = zero_potential;
  std::vector<std::vector<Complex>> weighted_vae(lm_count), weighted_vps(lm_count);
  double ae_h = 0.0, ps_h = 0.0;
  for (int l = 0; l <= augmentation.max_l; ++l) for (int M = -l; M <= l; ++M) {
    const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
    auto ae = multipole_hartree(r, rho_ae_h[lm], l,
        potential_mode != PawRadialPotentialMode::DiscreteVariational &&
        potential_mode != PawRadialPotentialMode::AsphericalAdjoint);
    auto ps = multipole_hartree(r, rho_ps_h[lm], l,
        potential_mode != PawRadialPotentialMode::DiscreteVariational &&
        potential_mode != PawRadialPotentialMode::AsphericalAdjoint);
    vae[lm] = std::move(ae.potential);
    vps[lm] = std::move(ps.potential);
    hartree_vae[lm] = vae[lm];
    hartree_vps[lm] = vps[lm];
    weighted_vae[lm] = std::move(ae.weighted_derivative);
    weighted_vps[lm] = std::move(ps.weighted_derivative);
    ae_h += .5*ae.energy;
    ps_h += .5*ps.energy;
  }
  // The XC energy and potential share radial and angular quadrature and the
  // transpose of the radial finite-difference gradient. Angular derivatives
  // act on real Y_lm and include both tangential gradient components.
  const auto radial_weights = radial_quadrature_weights(r);
  const auto gradient_stencils = radial_gradient_stencils(r);
  const bool discrete_adjoint = potential_mode == PawRadialPotentialMode::AsphericalAdjoint;
  const bool aspherical = potential_mode == PawRadialPotentialMode::AsphericalDirect || discrete_adjoint;
  std::array<std::vector<std::vector<Complex>>,2> ae_spin,ps_spin;
  ae_spin[spin_channel]=rho_ae;ae_spin[1-spin_channel]=other_ae;
  ps_spin[spin_channel]=rho_ps;ps_spin[1-spin_channel]=other_ps;
  const auto ae_pair=angular_collinear_pbe_xc(r,ae_spin,augmentation.max_l,radial_weights,gradient_stencils,length_factor,aspherical,discrete_adjoint);
  const auto ps_pair=angular_collinear_pbe_xc(r,ps_spin,augmentation.max_l,radial_weights,gradient_stencils,length_factor,aspherical,discrete_adjoint);
  const auto& ae_xc=ae_pair[spin_channel];const auto& ps_xc=ps_pair[spin_channel];
  // Store w*V, the derivative with respect to the radial density coefficient,
  // directly for projection. Simpson weights give the first endpoint zero weight
  // on an even grid, but the transpose gradient can still contribute there.
  // Dividing by w and then integrating would lose that endpoint derivative.
  for (std::size_t lm = 0; lm < lm_count; ++lm) for (std::size_t ir = 0; ir < nr; ++ir) {
    weighted_vae[lm][ir] += ae_xc.weighted_derivative[lm][ir];
    weighted_vps[lm][ir] += ps_xc.weighted_derivative[lm][ir];
    if (radial_weights[ir] > 0.0) {
      vae[lm][ir] += ae_xc.weighted_derivative[lm][ir] / radial_weights[ir];
      vps[lm][ir] += ps_xc.weighted_derivative[lm][ir] / radial_weights[ir];
    }
  }
  // Direct spherical GGA potential (as used for matched VASP references with
  // LASPH=.FALSE.).  The energy remains the same PBE functional; only the
  // mesh-level potential derivative follows radial_direct_gga_potential.
  const bool radial_direct_gga = potential_mode == PawRadialPotentialMode::RadialDirect;
  if (radial_direct_gga) {
    const auto ae_gga = radial_direct_gga_potential(
        r, rho_ae[0], radial_weights, gradient_stencils);
    const auto ps_gga = radial_direct_gga_potential(
        r, rho_ps[0], radial_weights, gradient_stencils);
    for (std::size_t ir = 0; ir < nr; ++ir) {
      // angular_pbe_xc already contains both LDA and GGA in lm=0.  Replace
      // that complete XC contribution with the direct radial result while
      // retaining the independently computed Hartree response.
      const auto ae_hartree = weighted_vae[0][ir] - ae_xc.weighted_derivative[0][ir];
      const auto ps_hartree = weighted_vps[0][ir] - ps_xc.weighted_derivative[0][ir];
      vae[0][ir] += ae_gga[ir] - (radial_weights[ir] > 0.0 ?
          ae_xc.weighted_derivative[0][ir] / radial_weights[ir] : 0.0);
      vps[0][ir] += ps_gga[ir] - (radial_weights[ir] > 0.0 ?
          ps_xc.weighted_derivative[0][ir] / radial_weights[ir] : 0.0);
      weighted_vae[0][ir] = ae_hartree + radial_weights[ir] * ae_gga[ir];
      weighted_vps[0][ir] = ps_hartree + radial_weights[ir] * ps_gga[ir];
    }
  }

  PawOneCenterClosureResult out;
  out.direct_radial_potential = radial_direct_gga || (aspherical && !discrete_adjoint);
  out.lm_moments = std::move(lm_moments);
  // Report lm moments in the archive convention: Q_L is in Angstrom^L.
  for (int l = 0; l <= augmentation.max_l; ++l)
    for (int m = -l; m <= l; ++m)
      out.lm_moments[static_cast<std::size_t>(l * l + m + l)] /=
          std::pow(length_factor, l);
  out.d_matrix_ev.assign(expanded_dimension * expanded_dimension, Complex(0.0, 0.0));
  out.reference_d_matrix_ev.assign(expanded_dimension * expanded_dimension, Complex(0.0, 0.0));
  out.rho_ae_r2.assign(lm_count, std::vector<double>(nr));
  out.rho_ps_r2.assign(lm_count, std::vector<double>(nr));
  out.potential_ae_ev.assign(lm_count, std::vector<double>(nr));
  out.potential_ps_ev.assign(lm_count, std::vector<double>(nr));
  out.hartree_ae_ev.assign(lm_count, std::vector<double>(nr));
  out.hartree_ps_ev.assign(lm_count, std::vector<double>(nr));
  out.xc_potential_ae_ev.assign(lm_count, std::vector<double>(nr));
  out.xc_potential_ps_ev.assign(lm_count, std::vector<double>(nr));
  const double archive_density_scale = length_factor;
  for (std::size_t lm = 0; lm < lm_count; ++lm) for (std::size_t ir = 0; ir < nr; ++ir) {
    out.rho_ae_r2[lm][ir] = (rho_ae[lm][ir].real()-(lm==0 && cache.core_charge_density.size()==nr?.5*cache.core_charge_density[ir]:0.)) * archive_density_scale;
    out.rho_ps_r2[lm][ir] = (rho_ps[lm][ir].real()-(lm==0 && cache.core_charge_density_pseudized.size()==nr?.5*cache.core_charge_density_pseudized[ir]:0.)) * archive_density_scale;
    out.potential_ae_ev[lm][ir] = vae[lm][ir].real() * energy_factor;
    out.potential_ps_ev[lm][ir] = vps[lm][ir].real() * energy_factor;
    out.hartree_ae_ev[lm][ir] = hartree_vae[lm][ir].real() * energy_factor;
    out.hartree_ps_ev[lm][ir] = hartree_vps[lm][ir].real() * energy_factor;
    out.xc_potential_ae_ev[lm][ir] = out.potential_ae_ev[lm][ir] - out.hartree_ae_ev[lm][ir];
    out.xc_potential_ps_ev[lm][ir] = out.potential_ps_ev[lm][ir] - out.hartree_ps_ev[lm][ir];
    if (reference && lm == 0) {
      out.potential_ae_ev[lm][ir] += std::sqrt(4.0 * pi) * reference->ae_reference_potential_ev[ir];
      out.potential_ps_ev[lm][ir] += std::sqrt(4.0 * pi) * reference->ps_reference_potential_ev[ir];
    }
  }
  double maxv = 0.0;
  for (const auto& v : vae) for (const auto& x : v) maxv = std::max(maxv, std::abs(x) * energy_factor);
  for (const auto& v : vps) for (const auto& x : v) maxv = std::max(maxv, std::abs(x) * energy_factor);
  out.max_potential_ev = maxv;
  for (std::size_t i = 0; i < channel_l.size(); ++i) for (std::size_t j = 0; j < channel_l.size(); ++j) {
    const int lmin = std::abs(channel_l[i] - channel_l[j]);
    const int lmax = std::min(augmentation.max_l, channel_l[i] + channel_l[j]);
    for (int m = -channel_l[i]; m <= channel_l[i]; ++m) for (int mp = -channel_l[j]; mp <= channel_l[j]; ++mp) {
      const std::size_t ii = offset[i] + static_cast<std::size_t>(m + channel_l[i]);
      const std::size_t jj = offset[j] + static_cast<std::size_t>(mp + channel_l[j]);
      Complex value(0.0, 0.0), reference_value(0.0, 0.0);
      Complex ae_value(0.0, 0.0), ps_value(0.0, 0.0);
      for (int l = lmin; l <= lmax; l += 2) for (int M = -l; M <= l; ++M) {
        static thread_local std::map<std::array<int, 4>, std::vector<GauntEntry>> gaunt_cache;
        const std::array<int, 4> key{channel_l[i], channel_l[j], m, mp};
        auto it = gaunt_cache.find(key);
        if (it == gaunt_cache.end())
          it = gaunt_cache.emplace(key, real_gaunt_terms(channel_l[i], channel_l[j], m, mp)).first;
        double c = 0.0;
        for (const auto& entry : it->second)
          if (entry.L == l && entry.M == M) { c = entry.value; break; }
        if (std::abs(c) < 1e-13) continue;
        const std::size_t lm = static_cast<std::size_t>(l * l + M + l);
        Complex radial_response(0.0, 0.0);
        for (std::size_t ir = 0; ir < nr; ++ir) {
          // The PS pair density carries its compensation term Q_ij^L g_L(r),
          // so the radial response is AE - (PS + compensation).
          const double pair_ae = cache.ae_wavefunctions[i][ir] * cache.ae_wavefunctions[j][ir];
          const double pair_ps_comp = cache.pseudo_wavefunctions[i][ir] * cache.pseudo_wavefunctions[j][ir] +
              augmentation.moment(i, j, l) * augmentation.compensation_r2[static_cast<std::size_t>(l)][ir];
          radial_response += pair_ae * weighted_vae[lm][ir] - pair_ps_comp * weighted_vps[lm][ir];
          ae_value += c * pair_ae * weighted_vae[lm][ir];
          ps_value += c * pair_ps_comp * weighted_vps[lm][ir];
          if (reference && l == 0) {
            reference_value += c * radial_weights[ir] * std::sqrt(4.0 * pi) *
                (pair_ae * reference->ae_reference_potential_ev[ir] -
                 pair_ps_comp * reference->ps_reference_potential_ev[ir]);
          }
        }
        value += c * radial_response;
      }
      const auto index = ii * expanded_dimension + jj;
      if (reference && !reference->projected_reference_d_matrix_ev.empty())
        reference_value = reference->projected_reference_d_matrix_ev[index];
      out.reference_d_matrix_ev[index] = reference_value;
      out.d_matrix_ev[index] = value * energy_factor + reference_value;
      out.ae_potential_expectation_ev +=
          (std::conj(density[index]) * ae_value).real() * energy_factor;
      out.ps_potential_expectation_ev +=
          (std::conj(density[index]) * ps_value).real() * energy_factor;
    }
  }
  // Enforce Hermiticity after quadrature roundoff, preserving the variational
  // matrix used in both the Hamiltonian and expectation-value correction.
  for (std::size_t i = 0; i < expanded_dimension; ++i) for (std::size_t j = i + 1; j < expanded_dimension; ++j) {
    const Complex v = 0.5 * (out.d_matrix_ev[i * expanded_dimension + j] +
                             std::conj(out.d_matrix_ev[j * expanded_dimension + i]));
    out.d_matrix_ev[i * expanded_dimension + j] = v;
    out.d_matrix_ev[j * expanded_dimension + i] = std::conj(v);
  }
  // A Hermitian PAW operator has real diagonal entries.  Complex lm moments
  // are kept for the non-Gamma density transform, but roundoff or asymmetric
  // radial contractions must never leak an imaginary diagonal into D_ij.
  for (std::size_t i = 0; i < expanded_dimension; ++i)
    out.d_matrix_ev[i * expanded_dimension + i] =
        Complex(out.d_matrix_ev[i * expanded_dimension + i].real(), 0.0);
  Complex expectation(0.0, 0.0);
  for (std::size_t i = 0; i < expanded_dimension; ++i) for (std::size_t j = 0; j < expanded_dimension; ++j)
    {
      const auto index = i * expanded_dimension + j;
      expectation += std::conj(density[index]) * out.d_matrix_ev[index];
    }
  for (std::size_t i = 0; i < expanded_dimension; ++i)
    for (std::size_t j = 0; j < expanded_dimension; ++j) {
      const auto index = i * expanded_dimension + j;
      out.reference_linear_energy_ev +=
          (std::conj(density[index]) * out.reference_d_matrix_ev[index]).real();
    }
  out.ae_energy_ev = (ae_h + ae_xc.energy) * energy_factor;
  out.ps_energy_ev = (ps_h + ps_xc.energy) * energy_factor;
  out.ae_hartree_ev = ae_h * energy_factor;
  out.ps_hartree_ev = ps_h * energy_factor;
  out.ae_xc_ev = ae_xc.energy * energy_factor;
  out.ps_xc_ev = ps_xc.energy * energy_factor;
  out.ae_double_counting_ev = out.ae_energy_ev - out.ae_potential_expectation_ev;
  out.ps_double_counting_ev = out.ps_energy_ev - out.ps_potential_expectation_ev;
  out.energy_difference_ev = out.ae_energy_ev - out.ps_energy_ev + out.reference_linear_energy_ev;
  // The AE XC energy includes the frozen core, so the AE one-centre energy
  // carries the core-only XC energy of the reference atom as a constant.
  // Attach it to the closure result so the periodic SCF assembler subtracts
  // it exactly once, alongside the AE/PS double-counting difference.
  out.core_xc_energy_ev = reference ? .5*reference->core_xc_energy_ev : 0.0;
  out.potential_expectation_ev = expectation.real();
  // The direct mesh derivative reproduces a reference discretization; it is
  // not the exact adjoint of this code's finite-grid energy quadrature.
  out.spherical_gga_variational = !(radial_direct_gga || (aspherical && !discrete_adjoint));
  out.aspherical_gga_variational = !(radial_direct_gga || (aspherical && !discrete_adjoint));
  out.mesh_compensation_norm = augmentation.mesh_compensation_norm;
  out.lm_moment_imag = lm_imag;
  out.ae_clipped_density_samples = ae_xc.clipped_density_samples;
  out.ps_clipped_density_samples = ps_xc.clipped_density_samples;
  return out;
}

std::array<PawOneCenterClosureResult,2> paw_collinear_one_center_closure(
    const PawNumericCache& cache,const std::vector<int>& channel_l,
    const std::array<std::vector<Complex>,2>& density,std::size_t dimension,
    PawRadialLengthUnit radial_unit,const PawOneCenterReferencePotentials* reference,
    PawRadialPotentialMode potential_mode,AtomicUnitConvention constants){
  return {collinear_closure_channel(cache,channel_l,density[0],dimension,density[1],0,radial_unit,reference,potential_mode,constants),
    collinear_closure_channel(cache,channel_l,density[1],dimension,density[0],1,radial_unit,reference,potential_mode,constants)};
}

PawOneCenterReferencePotentials paw_atomic_reference_potentials(
    const PawNumericCache& cache, const std::vector<int>& channel_l,
    PawRadialLengthUnit radial_unit, PawRadialPotentialMode potential_mode,
    AtomicUnitConvention constants) {
  const auto density = paw_atomic_reference_density(cache, channel_l);
  std::size_t dimension = 0;
  for (int l : channel_l) dimension += static_cast<std::size_t>(2 * l + 1);
  const auto atom = paw_one_center_closure(cache, channel_l, density,
                                          dimension, 1, radial_unit, nullptr, potential_mode, constants);
  PawOneCenterReferencePotentials out;
  const double sphere = std::sqrt(4.0 * pi);
  out.ae_reference_potential_ev.resize(cache.radial_grid.size());
  out.ps_reference_potential_ev.resize(cache.radial_grid.size());
  for (std::size_t ir = 0; ir < cache.radial_grid.size(); ++ir) {
    out.ae_reference_potential_ev[ir] = -atom.potential_ae_ev.at(0).at(ir) / sphere;
    out.ps_reference_potential_ev[ir] = -atom.potential_ps_ev.at(0).at(ir) / sphere;
  }
  out.projected_reference_d_matrix_ev = atom.d_matrix_ev;
  for (auto& value : out.projected_reference_d_matrix_ev) value = -value;
  out.ae_atomic_energy_ev = atom.ae_energy_ev;
  out.ps_atomic_energy_ev = atom.ps_energy_ev;
  auto r = cache.radial_grid;
  // The reference and the current on-site response must use the same
  // coordinate conversion and energy convention. Mixing CODATA core XC
  // with a legacy-unit radial closure introduces meV-scale ledger errors.
  if(constants==AtomicUnitConvention::MatchRadialMode)
    constants=potential_mode!=PawRadialPotentialMode::DiscreteVariational ?
        AtomicUnitConvention::VaspLegacy : AtomicUnitConvention::Codata;
  out.constants=constants;
  const auto units=atomic_units(constants);
  const double length_factor=units.bohr_per_angstrom;
  const double energy_factor=units.ev_per_hartree;
  const double radial_scale = radial_unit == PawRadialLengthUnit::Angstrom ?
      length_factor : 1.0;
  for (double& radius : r) radius *= radial_scale;
  std::vector<std::vector<Complex>> core(1, std::vector<Complex>(r.size()));
  if (!cache.core_charge_density.empty()) {
    if (cache.core_charge_density.size() != r.size())
      throw std::invalid_argument("PAW core reference density grid mismatch");
    for (std::size_t ir = 0; ir < r.size(); ++ir)
      core[0][ir] = cache.core_charge_density[ir] / radial_scale;
    const auto weights = radial_quadrature_weights(r);
    const auto gradients = radial_gradient_stencils(r);
    out.core_xc_energy_ev = ((potential_mode == PawRadialPotentialMode::AsphericalDirect ||
        potential_mode == PawRadialPotentialMode::AsphericalAdjoint) ?
        aspherical_pbe_xc(r, core, 0, weights, gradients, length_factor,
            potential_mode == PawRadialPotentialMode::AsphericalAdjoint) :
        angular_pbe_xc(r, core, 0, weights, gradients)).energy * energy_factor;
  }
  out.core_xc_energy_ev += cache.core_xc_outside_ev;
  return out;
}
std::vector<double> paw_hubbard_ae_overlap(const PawNumericCache& cache,
    const std::vector<int>& channels,int selected_l){
  const auto nr=cache.radial_grid.size(),nc=channels.size();
  if(nr<3||cache.ae_wavefunctions.size()!=nc||selected_l<0||selected_l>3)
    throw std::invalid_argument("incomplete Hubbard AE radial data");
  for(const auto& wave:cache.ae_wavefunctions)
    if(wave.size()!=nr)throw std::invalid_argument("Hubbard AE radial mesh mismatch");
  const auto weights=radial_quadrature_weights(cache.radial_grid);
  std::size_t edge=nr-1;
  if(cache.radial_rdep>0)
    for(std::size_t i=0;i+1<nr;++i)
      if(cache.radial_grid[i]-cache.radial_rdep>-.005){edge=i;break;}
  std::vector<double> out(nc*nc,0.);
  for(std::size_t i=0;i<nc;++i)if(channels[i]==selected_l)
    for(std::size_t j=0;j<nc;++j)if(channels[j]==selected_l)
      for(std::size_t ir=0;ir<=edge;++ir)
        out[i*nc+j]+=weights[ir]*cache.ae_wavefunctions[i][ir]*cache.ae_wavefunctions[j][ir];
  return out;
}
std::vector<double> paw_sphere_ae_overlap(const PawNumericCache& cache,
    const std::vector<int>& channels){
  const auto nr=cache.radial_grid.size(),nc=channels.size();
  if(nr<3||cache.ae_wavefunctions.size()!=nc)
    throw std::invalid_argument("incomplete AE radial data for sphere projections");
  for(const auto& wave:cache.ae_wavefunctions)
    if(wave.size()!=nr)throw std::invalid_argument("AE sphere radial mesh mismatch");
  std::size_t edge=nr-1;
  if(cache.radial_rdep>0)
    for(std::size_t i=0;i+1<nr;++i)
      if(cache.radial_grid[i]-cache.radial_rdep>-.005){edge=i;break;}
  // The sphere integral is truncated at the r_dep edge, and the truncated
  // mesh r[0..edge] gets its own end-anchored Simpson panels.
  const auto weights=radial_quadrature_weights(
      std::vector<double>(cache.radial_grid.begin(),cache.radial_grid.begin()+std::ptrdiff_t(edge)+1));
  std::vector<double> out(nc*nc,0.);
  for(std::size_t i=0;i<nc;++i)
    for(std::size_t j=0;j<nc;++j)if(channels[i]==channels[j])
      for(std::size_t ir=0;ir<=edge;++ir)
        out[i*nc+j]+=weights[ir]*cache.ae_wavefunctions[i][ir]*cache.ae_wavefunctions[j][ir];
  return out;
}
} // namespace sdft
