// S5/S6 operator checks. Mode comes from argv so each CTest entry sets its
// own environment before the operator reads it:
//   blocks       SDFT_PAW_COMPENSATION_SCOPE=blocks equals the dense H/S
//   real-full    real-space search with a sphere covering the cell equals B
//   real-radius  error shrinks monotonically as the sphere grows; exact
//                (certificate) calls are untouched by real-space search
#include "plane_wave_operator.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
using namespace sdft;
using C = std::complex<double>;
using M = Eigen::MatrixXcd;
constexpr double ha = 27.211386245988;

struct Model {
  std::vector<GVector> g;
  FourierDensity v;
  std::vector<ComplexPawOperator> paw;
  M h, s;
};

// Two atoms with Gaussian-like projectors centred at their positions in a
// cubic cell, so a geometric sphere is meaningful.
Model make_model() {
  Model m;
  const double a = 6.0;  // bohr
  const double b = 2 * M_PI / a;
  for (int h = -3; h <= 3; ++h)
    for (int k = -3; k <= 3; ++k)
      for (int l = -3; l <= 3; ++l)
        if (h * h + k * k + l * l <= 9)
          m.g.push_back({h, k, l, {b * h, b * k, b * l}, ha * .5 * b * b * (h * h + k * k + l * l)});
  m.v = {{{0, 0, 0}, .3}, {{1, 0, 0}, C(.05, .02)}, {{-1, 0, 0}, C(.05, -.02)}};
  const std::size_t n = m.g.size();
  const Vec3 tau[2]{{.1, .2, .3}, {.6, .55, .7}};
  const double width[2]{.9, 1.1};
  m.h = M::Zero(n, n);
  m.s = (1 + 1e-12) * M::Identity(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const auto key = std::make_tuple(m.g[i].h - m.g[j].h, m.g[i].k - m.g[j].k, m.g[i].l - m.g[j].l);
      const auto f = m.v.find(key);
      if (f != m.v.end()) m.h(i, j) = f->second;
    }
    m.h(i, i) += m.g[i].kinetic_ev / ha;
  }
  for (int atom = 0; atom < 2; ++atom) {
    const std::size_t np = 3;
    ComplexPawOperator p;
    p.projectors = np;
    p.fractional = tau[atom];
    p.radius_bohr = 1.4;
    p.lattice_bohr = {Vec3{a, 0, 0}, Vec3{0, a, 0}, Vec3{0, 0, a}};
    M bm(np, n), d(np, np), q(np, np);
    for (std::size_t r = 0; r < np; ++r)
      for (std::size_t j = 0; j < n; ++j) {
        const auto& G = m.g[j].cartesian;
        const double g2 = G.x * G.x + G.y * G.y + G.z * G.z;
        const double phase = G.x * a * tau[atom].x + G.y * a * tau[atom].y + G.z * a * tau[atom].z;
        const double angular = r == 0 ? 1. : (r == 1 ? G.x : G.z);
        bm(r, j) = .05 * angular * std::exp(-.5 * width[atom] * width[atom] * g2) * std::exp(C(0, phase));
      }
    d << .4, C(.03, .02), -.02, C(.03, -.02), -.25, .01, -.02, .01, .3;
    q << .05, C(.004, .003), 0, C(.004, -.003), .03, 0, 0, 0, .06;
    for (std::size_t r = 0; r < np; ++r) {
      for (std::size_t j = 0; j < n; ++j) p.amplitudes.push_back(bm(r, j));
      for (std::size_t c = 0; c < np; ++c) { p.d.push_back(d(r, c)); p.q.push_back(q(r, c)); }
    }
    m.h += bm.adjoint() * d * bm;
    m.s += bm.adjoint() * q * bm;
    m.paw.push_back(std::move(p));
  }
  return m;
}

// Max column error of block application with the given exact flag.
double block_error(Model m, bool exact, bool overlap) {
  PlaneWaveOperator op(m.g, m.v, m.paw, ha);
  auto o = overlap ? op.overlap() : op.hamiltonian();
  const std::size_t n = m.g.size();
  std::vector<C> x(n * n, C{}), y;
  for (std::size_t i = 0; i < n; ++i) x[i * n + i] = 1;
  o.apply_block(x, n, y, exact);
  const M& ref = overlap ? m.s : m.h;
  double err = 0;
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t r = 0; r < n; ++r) err = std::max(err, std::abs(y[c * n + r] - ref(r, c)));
  return err;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 9;
  const std::string mode = argv[1];
  const auto m = make_model();
  if (mode == "blocks") {
    // Exact calls with compensation confined to the D/Q blocks.
    const double eh = block_error(m, true, false), es = block_error(m, true, true);
    std::cout << "blocks H " << eh << " S " << es << "\n";
    return eh <= 1e-13 && es <= 1e-13 ? 0 : 1;
  }
  if (mode == "real-full") {
    // A radius factor of 2 on a 1.4*2=2.8 bohr sphere still misses corners
    // of the 6 bohr cell, so enlarge the stored radius to cover the cell.
    auto big = m;
    for (auto& p : big.paw) p.radius_bohr = 3.0 * std::sqrt(3.);
    const double eh = block_error(big, false, false), es = block_error(big, false, true);
    const double xh = block_error(big, true, false);
    std::cout << "real-full search H " << eh << " S " << es << " exact H " << xh << "\n";
    return eh <= 1e-12 && es <= 1e-12 && xh <= 1e-12 ? 0 : 2;
  }
  if (mode == "real-radius") {
    double previous = 1e300, first = 0;
    for (double r : {0.8, 1.4, 2.2, 3.2}) {
      auto mm = m;
      for (auto& p : mm.paw) p.radius_bohr = r;
      const double e = block_error(mm, false, false);
      const double exact = block_error(mm, true, false);
      std::cout << "radius " << r << " search H error " << e << " exact H " << exact << "\n";
      if (exact > 1e-12) return 3;            // certificate path unchanged
      if (e > previous * (1 + 1e-9)) return 4;  // monotone in the radius
      if (!first) first = e;
      previous = e;
    }
    // The model projectors are deliberately broad (Gaussian width ~1 bohr);
    // require a tenfold reduction across the sweep, not a fixed floor.
    return previous < .1 * first ? 0 : 5;
  }
  return 9;
}
