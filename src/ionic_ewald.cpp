#include "ionic_ewald.hpp"

#include <cmath>
#include <stdexcept>

namespace sdft {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;

Vec3 addv(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 subv(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 scalev(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }

Vec3 cart(const std::array<Vec3, 3>& a, Vec3 f) {
  return addv(addv(scalev(a[0], f.x), scalev(a[1], f.y)), scalev(a[2], f.z));
}

std::array<Vec3, 3> reciprocal(const std::array<Vec3, 3>& a) {
  const double det = determinant(a);
  if (!std::isfinite(det) || std::abs(det) < 1e-14) throw std::invalid_argument("singular Ewald lattice");
  return {scalev(cross(a[1], a[2]), 2.0 * pi / det),
          scalev(cross(a[2], a[0]), 2.0 * pi / det),
          scalev(cross(a[0], a[1]), 2.0 * pi / det)};
}

double f_real(double r, double eta) { return std::erfc(eta * r) / r; }
double length(Vec3 r) { return std::sqrt(dot(r, r)); }
double dfdr(double r, double eta) {
  return -std::erfc(eta * r) / (r * r) - 2.0 * eta * std::exp(-eta * eta * r * r) / (std::sqrt(pi) * r);
}
}

IonicEwaldResult evaluate_ionic_ewald(const std::array<Vec3, 3>& lattice,
    const std::vector<IonicEwaldAtom>& atoms, double eta, int real_cutoff, int reciprocal_cutoff) {
  if (!(eta > 0.0) || real_cutoff < 1 || reciprocal_cutoff < 1)
    throw std::invalid_argument("invalid Ewald parameters");
  const double volume = std::abs(determinant(lattice));
  if (!(volume > 1e-14) || !std::isfinite(volume)) throw std::invalid_argument("singular Ewald lattice");
  const auto b = reciprocal(lattice);
  std::vector<Vec3> positions;
  positions.reserve(atoms.size());
  for (const auto& atom : atoms) positions.push_back(cart(lattice, atom.fractional));
  IonicEwaldResult out;
  out.forces_hartree_per_bohr.assign(atoms.size(), Vec3{});

  // The ordered pair sum with one half includes every periodic image. Its
  // derivative is accumulated as a force on the first atom in each ordered
  // pair, which is exactly the derivative of the finite truncated sum.
  for (std::size_t i = 0; i < atoms.size(); ++i) for (std::size_t j = 0; j < atoms.size(); ++j)
    for (int h = -real_cutoff; h <= real_cutoff; ++h)
      for (int k = -real_cutoff; k <= real_cutoff; ++k)
        for (int l = -real_cutoff; l <= real_cutoff; ++l) {
          if (i == j && h == 0 && k == 0 && l == 0) continue;
          const Vec3 image = addv(addv(scalev(lattice[0], h), scalev(lattice[1], k)), scalev(lattice[2], l));
          const Vec3 rij = addv(subv(positions[i], positions[j]), image);
          const double r = length(rij);
          if (r < 1e-13) continue;
          const double qiqj = atoms[i].charge * atoms[j].charge;
          out.energy_hartree += 0.5 * qiqj * f_real(r, eta);
          const double force_factor = -0.5 * qiqj * dfdr(r, eta) / r;
          out.forces_hartree_per_bohr[i] = addv(out.forces_hartree_per_bohr[i], scalev(rij, force_factor));
          out.forces_hartree_per_bohr[j] = subv(out.forces_hartree_per_bohr[j], scalev(rij, force_factor));
          // dE/d epsilon_{alpha,beta}; the ordered-pair one-half is retained.
          const double virial = 0.5 * qiqj * dfdr(r, eta) / r;
          out.stress_hartree_per_bohr3[0].x += virial * rij.x * rij.x;
          out.stress_hartree_per_bohr3[0].y += virial * rij.x * rij.y;
          out.stress_hartree_per_bohr3[0].z += virial * rij.x * rij.z;
          out.stress_hartree_per_bohr3[1].x += virial * rij.y * rij.x;
          out.stress_hartree_per_bohr3[1].y += virial * rij.y * rij.y;
          out.stress_hartree_per_bohr3[1].z += virial * rij.y * rij.z;
          out.stress_hartree_per_bohr3[2].x += virial * rij.z * rij.x;
          out.stress_hartree_per_bohr3[2].y += virial * rij.z * rij.y;
          out.stress_hartree_per_bohr3[2].z += virial * rij.z * rij.z;
        }

  double total_charge = 0.0, self_charge2 = 0.0;
  for (const auto& atom : atoms) { total_charge += atom.charge; self_charge2 += atom.charge * atom.charge; }
  for (int h = -reciprocal_cutoff; h <= reciprocal_cutoff; ++h)
    for (int k = -reciprocal_cutoff; k <= reciprocal_cutoff; ++k)
      for (int l = -reciprocal_cutoff; l <= reciprocal_cutoff; ++l) {
        if (h == 0 && k == 0 && l == 0) continue;
        const Vec3 g = addv(addv(scalev(b[0], h), scalev(b[1], k)), scalev(b[2], l));
        const double g2 = dot(g, g);
        if (g2 < 1e-16 || g2 > 64.0 * eta * eta) continue;
        double sr = 0.0, si = 0.0;
        for (std::size_t i = 0; i < atoms.size(); ++i) {
          const double phase = dot(g, positions[i]);
          sr += atoms[i].charge * std::cos(phase);
          si += atoms[i].charge * std::sin(phase);
        }
        for (std::size_t i = 0; i < atoms.size(); ++i) {
          const double phase = dot(g, positions[i]);
          const double pref = 4.0 * pi / volume * atoms[i].charge * std::exp(-g2 / (4.0 * eta * eta)) / g2;
          const double im = sr * std::sin(phase) - si * std::cos(phase);
          out.forces_hartree_per_bohr[i].x += pref * g.x * im;
          out.forces_hartree_per_bohr[i].y += pref * g.y * im;
          out.forces_hartree_per_bohr[i].z += pref * g.z * im;
        }
        const double s2 = sr * sr + si * si;
        const double a = std::exp(-g2 / (4.0 * eta * eta)) / g2;
        out.energy_hartree += 2.0 * pi / volume * a * s2;
        const double da = a * (-1.0 / (4.0 * eta * eta) - 1.0 / g2);
        // Fixed fractional coordinates keep each g.r phase invariant under
        // homogeneous strain; only V and |g| change in this term.
        const double reciprocal_pref = 2.0 * pi / volume * s2;
        out.stress_hartree_per_bohr3[0].x += reciprocal_pref * (-a - 2.0 * da * g.x * g.x);
        out.stress_hartree_per_bohr3[0].y += reciprocal_pref * (-2.0 * da * g.x * g.y);
        out.stress_hartree_per_bohr3[0].z += reciprocal_pref * (-2.0 * da * g.x * g.z);
        out.stress_hartree_per_bohr3[1].x += reciprocal_pref * (-2.0 * da * g.y * g.x);
        out.stress_hartree_per_bohr3[1].y += reciprocal_pref * (-a - 2.0 * da * g.y * g.y);
        out.stress_hartree_per_bohr3[1].z += reciprocal_pref * (-2.0 * da * g.y * g.z);
        out.stress_hartree_per_bohr3[2].x += reciprocal_pref * (-2.0 * da * g.z * g.x);
        out.stress_hartree_per_bohr3[2].y += reciprocal_pref * (-2.0 * da * g.z * g.y);
        out.stress_hartree_per_bohr3[2].z += reciprocal_pref * (-a - 2.0 * da * g.z * g.z);
      }
  out.energy_hartree += -eta / std::sqrt(pi) * self_charge2;
  out.energy_hartree += -pi * total_charge * total_charge / (2.0 * eta * eta * volume);
  const double bg_stress = pi * total_charge * total_charge / (2.0 * eta * eta * volume * volume);
  out.stress_hartree_per_bohr3[0].x += bg_stress * volume;
  out.stress_hartree_per_bohr3[1].y += bg_stress * volume;
  out.stress_hartree_per_bohr3[2].z += bg_stress * volume;
  for (auto& row : out.stress_hartree_per_bohr3)
    row = scalev(row, 1.0 / volume);
  return out;
}
}
