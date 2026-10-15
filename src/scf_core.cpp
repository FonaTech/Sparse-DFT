#include "fourier_state.hpp"
#include "paw_framework.hpp"
#include "paw_site_model.hpp"
#include "joint_multisecant.hpp"
#include "collinear_mixing.hpp"
#include "scf_inner_accuracy.hpp"
#include "onsite_density.hpp"
#include "scf_core.hpp"
#include "thread_budget.hpp"
#include "parallel_context.hpp"
#include <chrono>
#include <cstdint>
#include "scf_geometry_guess.hpp"
#include "pw_seed_transport.hpp"
#include "ordered_fourier.hpp"
#include "plane_wave_operator.hpp"
#include "solver_search_state.hpp"
#include "compute_backend.hpp"
#include "projector_derivative.hpp"
#include "pbe_xc.hpp"
#include "density_functionals.hpp"
#include "pw_core.hpp"
#include "paw_onecenter.hpp"
#include "paw_augmentation.hpp"
#include "paw_compensation.hpp"
#include "reciprocal_density_table.hpp"
#include "tabulated_spline.hpp"
#include "ewald.hpp"
#include "occupations.hpp"
#include "paw_hubbard.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <set>
#include <optional>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <exception>
#include <filesystem>
#include <atomic>
#include <memory>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#endif

namespace sdft {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
constexpr double twopi = 2.0 * pi;
constexpr double ev_per_hartree = 27.211386245988;
constexpr double bohr_per_angstrom = 1.8897261254578281;
constexpr double felect_ev_angstrom = 14.3996454784255;
// Development toggles; the second name is the spelling used by earlier releases.
const char* env_with_alias(const char* name, const char* legacy) {
  const char* value = std::getenv(name);
  return value ? value : std::getenv(legacy);
}
const FourierDensity& empty_pseudo_core(){
  static const FourierDensity empty;
  return empty;
}

struct AtomPotential {
  std::shared_ptr<const PawNumericCache> cache_owner;
  const PawNumericCache& cache;
  double zval{};
  std::string record_id;
  std::string numeric_sha256;
  TabulatedSpline local_spline;
  TabulatedSpline kinetic_error_spline;
  AtomicUnits units;
  std::map<const double*,TabulatedSpline> projector_splines;
  AtomPotential(std::shared_ptr<const PawNumericCache> raw,double valence,std::string record,std::string checksum,
                TabulatedSpline local,TabulatedSpline kinetic,AtomicUnits constants,bool nlspline)
      : cache_owner(std::move(raw)),cache(*cache_owner),zval(valence),record_id(std::move(record)),numeric_sha256(std::move(checksum)),
        local_spline(std::move(local)),kinetic_error_spline(std::move(kinetic)),units(constants) {
    if(nlspline)for(const auto& ch:cache.nonlocal_channels)for(const auto& values:ch.reciprocal_projectors) {
      const double maximum=ch.reciprocal_cutoff>0?ch.reciprocal_cutoff:twopi/std::max(ch.cutoff_radius,1e-8);
      std::vector<double> grid(values.size()+1),samples(values.size()+1);
      samples[0]=(ch.angular_momentum%2?-1.:1.)*values.at(1);
      for(std::size_t i=0;i<=values.size();++i) {
        grid[i]=(static_cast<double>(i)-1.)*maximum/values.size();
        if(i)samples[i]=values[i-1];
      }
      projector_splines.emplace(values.data(),TabulatedSpline(std::move(grid),std::move(samples)));
    }
  }
  // Immutable PAW storage can be shared; reference/spline addresses remain
  // stable through moves. Per-atom derived views are never mutable aliases.
  AtomPotential(AtomPotential&&)=default;
  AtomPotential& operator=(AtomPotential&&)=delete;
  AtomPotential(const AtomPotential&)=delete;
  AtomPotential& operator=(const AtomPotential&)=delete;
};

struct PawScfClosureContext {
  PawRadialPotentialMode potential_mode{PawRadialPotentialMode::AsphericalDirect};
  std::vector<PawCompensationOperator> compensation;
  std::vector<PawAugmentationData> augmentation;
  std::vector<PawOneCenterReferencePotentials> references;
  FourierDensity pseudo_core;
  const std::vector<PawOneCenterClosureResult>* spin_closures=nullptr;
};

bool share_compensation_geometry() {
  const char* flag=std::getenv("SDFT_SHARE_COMPENSATION");
  if(flag && std::string(flag)!="0" && std::string(flag)!="1")
    throw std::invalid_argument("SDFT_SHARE_COMPENSATION must be 0 or 1");
  return !flag || std::string(flag)=="1";
}

bool paw_closure_enabled() {
  const char* value=std::getenv("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC");
  if(!value || std::string(value)=="1")return true;
  if(std::string(value)=="0")return false;
  throw std::invalid_argument("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC must be 0 or 1");
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot hash PAW numeric cache: " + path.string());
  unsigned char digest[32];
  char block[1 << 16];
#ifdef __APPLE__
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  while (file) {
    file.read(block, sizeof(block));
    const auto count = file.gcount();
    if (count > 0) CC_SHA256_Update(&context, block, static_cast<CC_LONG>(count));
  }
  CC_SHA256_Final(digest, &context);
#else
  SHA256_CTX context;
  SHA256_Init(&context);
  while (file) {
    file.read(block, sizeof(block));
    const auto count = file.gcount();
    if (count > 0) SHA256_Update(&context, block, static_cast<std::size_t>(count));
  }
  SHA256_Final(digest, &context);
#endif
  if (file.bad()) throw std::runtime_error("failed reading PAW numeric cache: " + path.string());
  std::ostringstream hex;
  hex << std::hex << std::setfill('0');
  for (unsigned char byte : digest) hex << std::setw(2) << static_cast<unsigned>(byte);
  return hex.str();
}

Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 scale(Vec3 a, double x) { return {a.x * x, a.y * x, a.z * x}; }
double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

std::array<Vec3, 3> reciprocal(const std::array<Vec3, 3>& a) {
  const double volume = determinant(a);
  if (!std::isfinite(volume) || std::abs(volume) < 1e-12) throw std::invalid_argument("nonfinite or singular lattice");
  return {scale(cross(a[1], a[2]), twopi / volume),
          scale(cross(a[2], a[0]), twopi / volume),
          scale(cross(a[0], a[1]), twopi / volume)};
}

Vec3 cartesian(const std::array<Vec3, 3>& basis, Vec3 f) {
  return add(add(scale(basis[0], f.x), scale(basis[1], f.y)), scale(basis[2], f.z));
}

double interp(const std::vector<double>& values, double x, double xmax) {
  if (values.empty() || x < 0.0) return 0.0;
  if (values.size() == 1 || xmax <= 0.0) return values.front();
  const double t = x * static_cast<double>(values.size() - 1) / xmax;
  if (t >= static_cast<double>(values.size() - 1)) return 0.0;
  const auto i = static_cast<std::size_t>(std::floor(t));
  const double w = t - static_cast<double>(i);
  return values[i] * (1.0 - w) + values[i + 1] * w;
}

// Reciprocal radial tables of a PAW dataset hold N samples at q_k = k*qmax/N,
// k = 0..N-1; the last sample is a guard value, not f(qmax).  This is separate
// from interp(), which is used for the logarithmic density grids.
double uniform_table_linear(const std::vector<double>& values, double q, double qmax) {
  if (values.empty() || q < 0.0 || qmax <= 0.0 || q >= qmax) return 0.0;
  const double n = static_cast<double>(values.size());
  const double u = q * n / qmax;
  if (u <= 0.0) return values.front();
  const std::size_t i = static_cast<std::size_t>(std::floor(u));
  if (i + 1 >= values.size()) return values.back();
  const double f = u - static_cast<double>(i);
  return values[i] * (1.0 - f) + values[i + 1] * f;
}

// Projector form factors p_l(q) = 4 pi \int r^2 j_l(qr) p_l(r) dr have parity
// (-1)^l in q, so the node left of q = 0 is (-1)^l times the node at q_1.  The
// four-node cubic is used only while its stencil stays below the guard sample,
// i.e. for q < (N-3) qmax/N; beyond that the form factor is taken as zero.
// Tables with fewer than four samples, and the SDFT_RECIPROCAL_LINEAR
// diagnostic, use linear interpolation.
FourNodeCubic projector_table_cubic(const std::vector<double>& values, double q, double qmax,
                                    int angular_momentum) {
  const auto n = values.size();
  if (q < 0.0 || qmax <= 0.0 || q >= qmax * (1.0 - 3.0 / static_cast<double>(n))) return {};
  const double u = q * static_cast<double>(n) / qmax;
  const auto k = static_cast<std::size_t>(std::floor(u));
  const double left = k > 0 ? values[k - 1] : (angular_momentum % 2 ? -values[1] : values[1]);
  return four_node_lagrange(left, values[k], values[k + 1], values[k + 2], u - static_cast<double>(k));
}

double projector_table_value(const std::vector<double>& values, double q,
                             double qmax, int angular_momentum) {
  if (std::getenv("SDFT_RECIPROCAL_LINEAR") || values.size() < 4)
    return uniform_table_linear(values, q, qmax);
  return projector_table_cubic(values, q, qmax, angular_momentum).value;
}

double reciprocal_table_derivative(const std::vector<double>& values,double q,double qmax) {
  if(values.size()<2||q<0||q>=qmax||qmax<=0)return 0;
  const auto i=static_cast<std::size_t>(std::floor(q*values.size()/qmax));
  return i+1<values.size() ? (values[i+1]-values[i])*values.size()/qmax : 0.;
}

RadialDensitySample density_table_sample(const std::vector<double>& values,double q,double qmax) {
  if(std::getenv("SDFT_RECIPROCAL_LINEAR"))
    return {uniform_table_linear(values,q,qmax),reciprocal_table_derivative(values,q,qmax)};
  return reciprocal_density_sample(values,q,qmax);
}

double projector_table_derivative(const std::vector<double>& values,double q,double qmax,int l) {
  if(std::getenv("SDFT_RECIPROCAL_LINEAR")||values.size()<4)return reciprocal_table_derivative(values,q,qmax);
  return projector_table_cubic(values,q,qmax,l).slope*static_cast<double>(values.size())/qmax;
}

RadialDensitySample projector_sample(const AtomPotential& pot,const std::vector<double>& values,
                                    double q,double qmax,int l) {
  if(!pot.projector_splines.empty()) {
    if(std::getenv("SDFT_RECIPROCAL_LINEAR"))throw std::invalid_argument("NLSPLINE and legacy linear diagnostic are incompatible");
    // The spline shares the table domain: its last interval ends at the guard sample.
    if(q<0 || qmax<=0 || q>=qmax*(1.-1./values.size()))return {};
    const auto& spline=pot.projector_splines.at(values.data());
    return {spline.value(q),spline.derivative(q)};
  }
  return {projector_table_value(values,q,qmax,l),projector_table_derivative(values,q,qmax,l)};
}

// Snapshot process controls before entering reciprocal-space worker loops.
// A snapshot belongs to one operation, so tests and separate jobs may still
// change controls between operations without stale process-global state.
struct ReciprocalEvaluationOptions {
  bool linear=std::getenv("SDFT_RECIPROCAL_LINEAR")!=nullptr;
  bool local_q_bohr=std::getenv("SDFT_LOCAL_Q_BOHR")!=nullptr;
  bool disable_tail=std::getenv("SDFT_DISABLE_COULOMB_TAIL")!=nullptr;
  double local_scale=[](){const char* text=std::getenv("SDFT_LOCAL_SCALE");return text?std::atof(text):1.;}();
};
bool reciprocal_eval_batch() {
  const char* text=std::getenv("SDFT_RECIPROCAL_EVAL_BATCH");
  if(text&&std::string(text)!="0"&&std::string(text)!="1")
    throw std::invalid_argument("SDFT_RECIPROCAL_EVAL_BATCH must be 0 or 1");
  return text&&std::string(text)=="1";
}

double local_fourier_derivative(const AtomPotential& pot,double q,const ReciprocalEvaluationOptions& options) {
  if(q<1e-14)return 0;
  const auto units=pot.units;const auto& c=pot.cache;
  if(c.local_potential.empty()||c.local_header.empty())return 0;
  const double length=options.local_q_bohr?1.:units.bohr_per_angstrom;
  const double qt=q*length;
  if(qt>=c.local_header.front()*(1.-1./c.local_potential.size()))return 0;
  double derivative=options.linear
      ? reciprocal_table_derivative(c.local_potential,qt,c.local_header.front()) : pot.local_spline.derivative(qt);
  if(!options.disable_tail)derivative+=8*pi*pot.zval*(units.ev_per_hartree/units.bohr_per_angstrom)/(qt*qt*qt);
  const double factor=options.local_scale;
  return factor*length*derivative;
}

double local_fourier_derivative(const AtomPotential& pot,double q) {
  return local_fourier_derivative(pot,q,ReciprocalEvaluationOptions{});
}

double radial_fourier(const std::vector<double>& radius,
                      const std::vector<double>& potential, double q) {
  if (radius.size() != potential.size() || radius.size() < 2) return 0.0;
  double integral = 0.0;
  for (std::size_t i = 0; i + 1 < radius.size(); ++i) {
    const double r0 = radius[i], r1 = radius[i + 1];
    const double f0 = r0 * potential[i] * (q < 1e-12 ? 1.0 : std::sin(q * r0) / q);
    const double f1 = r1 * potential[i + 1] * (q < 1e-12 ? 1.0 : std::sin(q * r1) / q);
    integral += 0.5 * (r1 - r0) * (f0 + f1);
  }
  return 4.0 * pi * integral / ev_per_hartree;
}

struct Basis {
  std::vector<GVector> g;
  std::vector<Vec3> cart;
};

using Complex = std::complex<double>;
using MatrixC = Eigen::MatrixXcd;

// One radial projector is expanded over all m channels.  The old prototype
// kept a single z component for l>0, which breaks Bloch phases and degeneracy.
struct ProjectorRef { int l{}; int radial{}; int m{}; };

// Projectors use real normalized harmonics in the order
// (s; py,pz,px; dxy,dyz,dz2,dxz,dx2-y2; ...), the convention of the POTCAR
// projector tables.  Complex Y_lm would change the relative phase and mix
// degenerate channels.
double spherical_harmonic(int l, int m, Vec3 q) {
  return paw_real_spherical_harmonic(l, m, q);
}

Basis make_basis(const ScfInput& input, const std::array<Vec3, 3>& b, Vec3 kfrac) {
  const double ev = atomic_units(input.unit_convention).ev_per_hartree;
  auto g = input.basis_reference_lattice_bohr
      ? enumerate_g_vectors_fixed_reference(b, reciprocal(*input.basis_reference_lattice_bohr),
                                             input.cutoff_ev + 1e-8, kfrac, ev)
      : enumerate_g_vectors(b, input.cutoff_ev + 1e-8, kfrac, ev);
  Basis out{g, {}};
  out.cart.reserve(g.size());
  const Vec3 k = scale(add(add(scale(b[0], kfrac.x), scale(b[1], kfrac.y)), scale(b[2], kfrac.z)), 1.0);
  for (const auto& gv : g) out.cart.push_back(add(gv.cartesian, k));
  return out;
}

std::string reciprocal_index_sha256(std::vector<std::tuple<int,int,int>> indices) {
  std::sort(indices.begin(), indices.end());
  std::ostringstream serialized;
  for (const auto& [h,k,l] : indices) serialized << h << ' ' << k << ' ' << l << '\n';
  const auto payload = serialized.str(); unsigned char digest[32];
#ifdef __APPLE__
  CC_SHA256(payload.data(), static_cast<CC_LONG>(payload.size()), digest);
#else
  SHA256(reinterpret_cast<const unsigned char*>(payload.data()), payload.size(), digest);
#endif
  std::ostringstream hexadecimal; hexadecimal << std::hex << std::setfill('0');
  for (const auto value : digest) hexadecimal << std::setw(2) << static_cast<unsigned>(value);
  return hexadecimal.str();
}

std::vector<AtomPotential> read_potentials(const ScfInput& input) {
  if(input.projector_interpolation!="cubic" && input.projector_interpolation!="spline")
    throw std::invalid_argument("invalid projector interpolation convention");
  std::vector<AtomPotential> out;
  out.reserve(input.atoms.size());
  struct Prepared {std::shared_ptr<const PawNumericCache> cache;std::string record,hash;};
  std::map<std::filesystem::path,Prepared> prepared;
  const char* flag=std::getenv("SDFT_SHARE_PAW_RADIAL");
  if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("SDFT_SHARE_PAW_RADIAL must be 0 or 1");
  const bool share=!flag||std::string(flag)=="1";
  for (const auto& atom : input.atoms) {
    const auto key=std::filesystem::canonical(atom.paw_cache);
    auto prior=prepared.find(key);
    std::shared_ptr<const PawNumericCache> owner;
    if(share&&prior!=prepared.end())owner=prior->second.cache;
    else owner=std::make_shared<const PawNumericCache>(read_paw_numeric_cache(atom.paw_cache));
    const auto& cache=*owner;
    std::string record_id;
    const auto metadata_path = atom.paw_cache.parent_path() / "metadata.json";
    if(share&&prior!=prepared.end())record_id=prior->second.record;
    else if (std::filesystem::exists(metadata_path))record_id = read_paw_cache_metadata(metadata_path).record_id;
    const auto checksum=share&&prior!=prepared.end()?prior->second.hash:sha256_file(atom.paw_cache);
    if(share&&prior==prepared.end())prepared.emplace(key,Prepared{owner,record_id,checksum});
    record_performance(share&&prior!=prepared.end()?"paw_radial_shared_atom":"paw_radial_unique_load",0);
    std::vector<double> q(cache.local_potential.size());
    for (std::size_t i = 0; i < q.size(); ++i)
      q[i] = static_cast<double>(i) * cache.local_header.front() / q.size();
    TabulatedSpline local(q, cache.local_potential, 0.0);
    TabulatedSpline kinetic;
    if (cache.kinetic_error_cutoff_ev.size() >= 2)
      kinetic = TabulatedSpline(cache.kinetic_error_cutoff_ev, cache.kinetic_error_ev);
    out.push_back({std::move(owner), atom.zval, std::move(record_id),
        checksum, std::move(local), std::move(kinetic), atomic_units(input.unit_convention),
        input.projector_interpolation=="spline"});
  }
  return out;
}

double local_fourier(const AtomPotential& pot, double q_bohr,const ReciprocalEvaluationOptions& options) {
  const double bohr_per_angstrom = pot.units.bohr_per_angstrom;
  const double felect_ev_angstrom = pot.units.ev_per_hartree / bohr_per_angstrom;
  // The POTCAR local-potential table holds only the finite non-Coulomb part.
  // The full local potential adds the ionic tail -4*pi*Z*e^2/G^2 before
  // dividing by the cell volume.  Omitting this tail is a convention error;
  // apparent improvements in an incomplete PAW calculation are accidental
  // cancellation with other missing terms.
  const auto& c = pot.cache;
  if (c.local_potential.empty() || c.local_header.empty()) return 0.0;
  const double scale = options.local_scale;
  if (q_bohr < 1e-12) {
    // The G=0 local ionic term belongs to the neutralizing-background and
    // non-Coulomb (alpha Z) energy bookkeeping and is intentionally excluded
    // from the one-electron Hamiltonian here.
    return 0.0;
  }
  const double q_table = options.local_q_bohr ? q_bohr : q_bohr * bohr_per_angstrom;
  const double table = options.linear
      ? uniform_table_linear(c.local_potential, q_table, c.local_header.front())
      : pot.local_spline.value(q_table);
  if (q_table >= c.local_header.front()*(1.-1./c.local_potential.size())) return 0.0;
  const double coulomb = 4.0 * pi * pot.zval * felect_ev_angstrom / (q_table * q_table);
  if (options.disable_tail) return scale * table;
  return scale * (table - coulomb);
}

double local_fourier(const AtomPotential& pot,double q_bohr) {
  return local_fourier(pot,q_bohr,ReciprocalEvaluationOptions{});
}

double projector_value(const std::vector<double>& table, double q, double rmax) {
  // The 100-point POTCAR reciprocal projector table runs from q=0 to
  // approximately 2*pi/rmax.  This interpolation is exact at the stored
  // nodes and is sufficient for a stable plane-wave matrix construction.
  return interp(table, q, twopi / std::max(rmax, 1e-8));
}

double projector_value_cutoff(const AtomPotential& pot,const std::vector<double>& table, double q, double qmax,
                              int angular_momentum) {
  return pot.projector_splines.empty()?projector_table_value(table,q,qmax,angular_momentum):
      projector_sample(pot,table,q,qmax,angular_momentum).value;
}

void assemble_local(const ScfInput& input, const std::vector<AtomPotential>& pots,
                    const Basis& basis, const std::array<Vec3, 3>& b,
                    const std::array<Vec3, 3>& a, MatrixC& h) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  const double volume_bohr3 = std::abs(determinant(a));
  const double volume_ang3 = volume_bohr3 / std::pow(bohr_per_angstrom, 3);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < basis.g.size(); ++i) {
    for (std::size_t j = 0; j < basis.g.size(); ++j) {
      const Vec3 q = sub(basis.cart[i], basis.cart[j]);
      const double qn = norm(q);
      Complex value(0.0, 0.0);
      for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
        const Vec3 r = cartesian(a, input.atoms[atom].fractional);
        value += Complex(local_fourier(pots[atom], qn) / (volume_ang3 * ev_per_hartree), 0.0) *
                 std::exp(Complex(0.0, -dot(q, r)));
      }
      h(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) += value;
    }
  }
}

MatrixC atom_projector_basis(const ScfInput& input, const AtomPotential& pot,
                             std::size_t atom, const Basis& basis,
                             const std::array<Vec3, 3>& a) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  std::size_t dimension = 0;
  for (const auto& channel : pot.cache.nonlocal_channels)
    dimension += static_cast<std::size_t>(channel.nprojectors * (2 * channel.angular_momentum + 1));
  MatrixC beta(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(basis.g.size()));
  const Vec3 position = cartesian(a, input.atoms[atom].fractional);
  const double volume_ang3 = std::abs(determinant(a)) / std::pow(bohr_per_angstrom, 3);
  const double projector_scale = std::getenv("SDFT_PROJECTOR_SCALE") ? std::atof(std::getenv("SDFT_PROJECTOR_SCALE")) : 1.0;
  const double prefactor = projector_scale / std::sqrt(volume_ang3);
  const char* reuse_flag=std::getenv("SDFT_PROJECTOR_GEOMETRY_REUSE");
  if(reuse_flag&&std::string(reuse_flag)!="0"&&std::string(reuse_flag)!="1")
    throw std::invalid_argument("SDFT_PROJECTOR_GEOMETRY_REUSE must be 0 or 1");
  if(!reuse_flag||std::string(reuse_flag)=="1"){
    struct GeometryTemplate {std::string hash;double volume{},scale{},conversion{};
      bool linear{},spline{};std::vector<Vec3> vectors;MatrixC bare;};
    thread_local std::vector<GeometryTemplate> templates;
    const bool linear=std::getenv("SDFT_RECIPROCAL_LINEAR")!=nullptr;
    const double conversion=std::getenv("SDFT_PROJECTOR_Q_BOHR")?1.:bohr_per_angstrom;
    const bool spline=!pot.projector_splines.empty();
    auto found=templates.end();
    for(auto it=templates.begin();it!=templates.end();++it){
      if(it->hash!=pot.numeric_sha256||it->volume!=volume_ang3||it->scale!=projector_scale||
          it->conversion!=conversion||it->linear!=linear||it->spline!=spline||it->vectors.size()!=basis.cart.size())continue;
      bool same=true;for(std::size_t g=0;g<basis.cart.size();++g){const auto& x=it->vectors[g];const auto& y=basis.cart[g];
        if(x.x!=y.x||x.y!=y.y||x.z!=y.z){same=false;break;}}
      if(same){found=it;break;}
    }
    GeometryTemplate fresh;
    const MatrixC* bare=nullptr;
    if(found==templates.end()){
      PerformanceTimer timer("projector_radial_geometry_prepare");
      fresh.hash=pot.numeric_sha256;fresh.volume=volume_ang3;fresh.scale=projector_scale;
      fresh.conversion=conversion;fresh.linear=linear;fresh.spline=spline;fresh.vectors=basis.cart;
      fresh.bare.resize(dimension,basis.g.size());std::size_t r=0;
      for(const auto& ch:pot.cache.nonlocal_channels)for(int projector=0;projector<ch.nprojectors;++projector)
        for(int m=-ch.angular_momentum;m<=ch.angular_momentum;++m,++r)
          for(std::size_t g=0;g<basis.g.size();++g)fresh.bare(r,g)=prefactor*
              std::pow(Complex(0,1),ch.angular_momentum)*spherical_harmonic(ch.angular_momentum,m,basis.cart[g])*
              projector_value_cutoff(pot,ch.reciprocal_projectors[projector],norm(basis.cart[g])*conversion,
                  ch.reciprocal_cutoff>0?ch.reciprocal_cutoff:twopi/std::max(ch.cutoff_radius,1e-8),ch.angular_momentum);
      const std::size_t bytes=fresh.bare.size()*sizeof(Complex)+fresh.vectors.size()*sizeof(Vec3);
      constexpr std::size_t budget=128*1024*1024;
      std::size_t resident=0;for(const auto& item:templates)resident+=item.bare.size()*sizeof(Complex)+item.vectors.size()*sizeof(Vec3);
      while(!templates.empty()&&(templates.size()>=32||resident+bytes>budget)){
        resident-=templates.front().bare.size()*sizeof(Complex)+templates.front().vectors.size()*sizeof(Vec3);templates.erase(templates.begin());}
      if(bytes<=budget){templates.push_back(std::move(fresh));bare=&templates.back().bare;}else bare=&fresh.bare;
      record_performance("projector_radial_geometry_miss",0);
    }else{bare=&found->bare;record_performance("projector_radial_geometry_hit",0);}
    // Geometry-independent radial/angular data is shared by all equal PAW
    // species. The phase belongs to this atom and is always rebuilt.
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(basis.g.size()>4096 && !omp_in_parallel())
#endif
    for(std::size_t g=0;g<basis.g.size();++g){const auto phase=std::exp(Complex(0,dot(basis.cart[g],position)));
      for(std::size_t r=0;r<dimension;++r)beta(r,g)=(*bare)(r,g)*phase;}
    return beta;
  }
  std::size_t row = 0;
  for (const auto& channel : pot.cache.nonlocal_channels)
    for (int projector = 0; projector < channel.nprojectors; ++projector)
      for (int m = -channel.angular_momentum; m <= channel.angular_momentum; ++m, ++row)
        for (std::size_t ig = 0; ig < basis.g.size(); ++ig) {
          const double q = norm(basis.cart[ig]);
          beta(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(ig)) = prefactor *
              std::pow(Complex(0.0, 1.0), channel.angular_momentum) *
              spherical_harmonic(channel.angular_momentum, m, basis.cart[ig]) *
              projector_value_cutoff(pot,channel.reciprocal_projectors[projector],
              q * (std::getenv("SDFT_PROJECTOR_Q_BOHR") ? 1.0 : bohr_per_angstrom),
                  channel.reciprocal_cutoff > 0.0 ? channel.reciprocal_cutoff : twopi / std::max(channel.cutoff_radius, 1e-8),
                  channel.angular_momentum) * std::exp(Complex(0.0, dot(basis.cart[ig], position)));
        }
  if (std::getenv("SDFT_TRACE_PROJECTOR") && atom == 0) {
    std::cerr << "PROJECTOR_TRACE rows=" << beta.rows() << " cols=" << beta.cols()
              << " max=" << beta.cwiseAbs().maxCoeff() << " norm=" << beta.norm() << "\n";
  }
  return beta;
}

std::vector<Complex> onsite_from_coeff(const MatrixC& projector_basis,
                                     const MatrixC& coefficients, int occupied,
                                     const std::vector<double>& occupations) {
  // Contract through occupied bands, rather than constructing another NPW^2
  // matrix for each atom. The same B is used in H = B^dagger D B.
  MatrixC projected = projector_basis * coefficients.leftCols(occupied);
  MatrixC weighted = projected;
  for (int band = 0; band < occupied; ++band)
    weighted.col(band) *= occupations[static_cast<std::size_t>(band)];
  const MatrixC density = weighted * projected.adjoint();
  const std::size_t dimension = static_cast<std::size_t>(density.rows());
  std::vector<Complex> out(dimension * dimension);
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t col = 0; col < dimension; ++col)
      out[row * dimension + col] = density(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  return out;
}

void assemble_nonlocal(const ScfInput& input, const std::vector<AtomPotential>& pots,
                       const Basis& basis, const std::array<Vec3, 3>& a,
                       MatrixC& h, MatrixC& s,
                       const MatrixC* density_matrix = nullptr,
                       double* paw_spherical_dij_norm_ev = nullptr,
                       double* paw_dc_ae_ev = nullptr,
                       double* paw_dc_ps_ev = nullptr,
                       double* paw_onecenter_energy_ev = nullptr,
                       double* paw_onecenter_potential_expectation_ev = nullptr,
                       double* paw_onecenter_core_xc_ev = nullptr,
                       const std::vector<ScfOneCenterState>* frozen_onecenter = nullptr,
                       std::vector<ScfOneCenterState>* captured_onecenter = nullptr,
                       const std::vector<std::vector<Complex>>* onsite_input = nullptr,
                       const PawScfClosureContext* closure_context = nullptr,
                       const FourierDensity* total_local_potential = nullptr,
                       std::vector<ComplexPawOperator>* operator_blocks = nullptr,
                       const std::vector<MatrixC>* cached_projectors = nullptr) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  if (captured_onecenter) captured_onecenter->clear();
  const double volume_bohr3 = std::abs(determinant(a));
  const double volume_ang3 = volume_bohr3 / std::pow(bohr_per_angstrom, 3);
  const bool has_closure = paw_closure_enabled() &&
      !env_with_alias("SDFT_ENABLE_SPHERICAL_ONECENTER","SDFT_ENABLE_ONECENTER_CHF") &&
      (onsite_input || (density_matrix && density_matrix->rows()==static_cast<Eigen::Index>(basis.g.size()) &&
       density_matrix->cols()==static_cast<Eigen::Index>(basis.g.size())));
  const char* batch_setting=std::getenv("SDFT_BATCH_COMPENSATION_ADJOINT");
  if(batch_setting && std::string(batch_setting)!="0" && std::string(batch_setting)!="1")
    throw std::invalid_argument("SDFT_BATCH_COMPENSATION_ADJOINT must be 0 or 1");
  const bool batch_adjoint=has_closure && !frozen_onecenter && closure_context && total_local_potential &&
      !std::getenv("SDFT_DISABLE_DGRID") && (!batch_setting || std::string(batch_setting)=="1");
  std::vector<std::vector<Complex>> batched_dgrid;
  if(batch_adjoint) {
    if(closure_context->compensation.size()!=input.atoms.size())
      throw std::invalid_argument("PAW compensation atom count mismatch");
    PerformanceTimer timer("paw_compensation_adjoint");
    batched_dgrid=PawCompensationOperator::project_potential_batch(closure_context->compensation,*total_local_potential);
  }
  for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
    const auto& c = pots[atom].cache;
    std::vector<ProjectorRef> refs;
    for (const auto& ch : c.nonlocal_channels)
      for (int np = 0; np < ch.nprojectors; ++np)
        for (int m = -ch.angular_momentum; m <= ch.angular_momentum; ++m)
          refs.push_back({ch.angular_momentum, np, m});
    const std::size_t projector_total = refs.size();
    std::vector<Complex> beta(projector_total * basis.g.size(), Complex(0.0, 0.0));
    // Reciprocal projectors carry 1/sqrt(Omega) with Omega in Angstrom^3,
    // while the bare D_ij strengths are in eV.  The generalized eigenproblem
    // below is in Hartree, so only D_ij is converted; the projector
    // normalization itself is unchanged.
    const char* projector_scale_env = std::getenv("SDFT_PROJECTOR_SCALE");
    const double projector_scale = projector_scale_env ? std::atof(projector_scale_env) : 1.0;
    const MatrixC computed_projector = cached_projectors ? MatrixC{} : atom_projector_basis(input, pots[atom], atom, basis, a);
    const auto& projector_basis = cached_projectors ? cached_projectors->at(atom) : computed_projector;
    for (std::size_t x = 0; x < projector_total; ++x)
      for (std::size_t ig = 0; ig < basis.g.size(); ++ig)
        beta[x * basis.g.size() + ig] = projector_basis(static_cast<Eigen::Index>(x), static_cast<Eigen::Index>(ig));
    std::vector<Complex> d(projector_total * projector_total, Complex(0.0, 0.0));
    std::vector<Complex> qmat(projector_total * projector_total, Complex(0.0, 0.0));
    std::vector<std::size_t> radial_offsets(c.nonlocal_channels.size(), 0);
    std::size_t radial_count = 0;
    for (std::size_t ci = 0; ci < c.nonlocal_channels.size(); ++ci) {
      radial_offsets[ci] = radial_count;
      radial_count += static_cast<std::size_t>(c.nonlocal_channels[ci].nprojectors);
    }
    // Build channel-local D and Q blocks.  The L=0 augmentation moment block is
    // the spherical overlap correction; higher multipoles are handled by the
    // augmentation-density path and are not silently scaled away here.
    std::size_t channel_start = 0;
    std::size_t raw_channel_start = 0;
    for (const auto& ch : c.nonlocal_channels) {
      const std::size_t expanded_start = channel_start;
      const std::size_t radial_start = raw_channel_start;
      for (int i = 0; i < ch.nprojectors; ++i) for (int j = 0; j < ch.nprojectors; ++j) {
        for (int m = -ch.angular_momentum; m <= ch.angular_momentum; ++m) {
          const std::size_t row = expanded_start + static_cast<std::size_t>(i * (2 * ch.angular_momentum + 1) + (m + ch.angular_momentum));
          const std::size_t col = expanded_start + static_cast<std::size_t>(j * (2 * ch.angular_momentum + 1) + (m + ch.angular_momentum));
          // POTCAR bare D_ij strengths are stored in eV; the plane-wave
          // kinetic/local terms are converted to Hartree below, so convert
          // this block only once when the matrix is assembled.
          const char* dscale_env = std::getenv("SDFT_D_SCALE");
          const double dscale = dscale_env ? std::atof(dscale_env) : 1.0;
          d[row * projector_total + col] = dscale * ch.d_matrix[i][j] / ev_per_hartree;
          const std::size_t raw_i = radial_start + static_cast<std::size_t>(i);
          const std::size_t raw_j = radial_start + static_cast<std::size_t>(j);
          // The overlap correction q_ij = <phi_i|phi_j> - <phi~_i|phi~_j> is
          // the L=0 augmentation moment and couples only channels with equal
          // angular momentum (Bloechl, PRB 50, 17953 (1994)).  Cross-l and
          // non-spherical moments belong to the density/compensation path and
          // must not enter the overlap metric.  This loop is channel-local,
          // so the angular-momentum equality holds by construction.
          if (!std::getenv("SDFT_DISABLE_OVERLAP"))
            qmat[row * projector_total + col] = closure_context
                ? closure_context->augmentation.at(atom).overlap_moment(raw_i, raw_j)
                : c.augmentation_coefficients[raw_i * radial_count + raw_j];
        }
      }
      channel_start += static_cast<std::size_t>(ch.nprojectors * (2 * ch.angular_momentum + 1));
      raw_channel_start += static_cast<std::size_t>(ch.nprojectors);
    }
    // Build the on-site occupation and D_ij from the current SCF density, at
    // the point where the Hamiltonian is refreshed.  The first iteration has
    // no wavefunction density yet; use the POTCAR atomic occupancy only as an
    // initial guess.  Later iterations must use the projected density matrix,
    // otherwise the one-centre D_ij is not a functional derivative of the SCF
    // energy.
    const bool closure_diagnostic = has_closure;
    ScfOneCenterState state;
    if (frozen_onecenter) {
      const auto& frozen = frozen_onecenter->at(atom);
      if (frozen.projector_dimension != projector_total ||
          frozen.onsite_dij_ev.size() != projector_total * projector_total)
        throw std::invalid_argument("frozen PAW on-site D_ij projector dimensions do not match the NSCF basis");
      for (std::size_t entry = 0; entry < d.size(); ++entry)
        d[entry] = frozen.onsite_dij_ev[entry] / ev_per_hartree;
    } else if (closure_diagnostic) {
      // assemble_nonlocal applies H = B^dagger D B, where B[x,G] stores
      // the projector amplitude used by the Hamiltonian assembly below. Its matching
      // on-site density is therefore B rho B^dagger. Keeping this operation
      // as matrix products both preserves complex phases and avoids the
      // O(Nproj^2 * Npw^2) four-loop contraction.
      std::vector<Complex> onsite(projector_total * projector_total);
      if (onsite_input) {
        if (onsite_input->size() != input.atoms.size() || onsite_input->at(atom).size() != onsite.size())
          throw std::invalid_argument("mixed PAW onsite density dimensions do not match projectors");
        onsite = onsite_input->at(atom);
      } else {
        const MatrixC onsite_matrix = projector_basis * (*density_matrix) * projector_basis.adjoint();
        for (std::size_t x = 0; x < projector_total; ++x)
          for (std::size_t y = 0; y < projector_total; ++y)
            onsite[x * projector_total + y] = onsite_matrix(static_cast<Eigen::Index>(x), static_cast<Eigen::Index>(y));
      }
      std::vector<int> raw_l;
      for (const auto& channel : c.nonlocal_channels)
        for (int np = 0; np < channel.nprojectors; ++np)
          raw_l.push_back(channel.angular_momentum);
      if (!closure_context || !total_local_potential)
        throw std::invalid_argument("PAW closure requires its augmentation and reference data");
      // Hamiltonian, atomic reference and double counting use the same XC
      // quadrature. LASPH-compatible Cartesian response includes LMAX+1.
      const auto closure = [&](){PerformanceTimer timer("paw_radial_closure");return closure_context->spin_closures ? closure_context->spin_closures->at(atom) : paw_one_center_closure(c, raw_l, onsite, projector_total,
          1, PawRadialLengthUnit::Angstrom, &closure_context->references.at(atom),
          closure_context->potential_mode, input.unit_convention);}();
      // The smooth-grid term integrates the total local potential against the
      // compensation shapes g_L and contracts the result with the augmentation
      // moments and Gaunt coefficients: the positive adjoint of the
      // compensation density (Kresse & Joubert, PRB 59, 1758 (1999)).
      // AE-PS-compensation radial response is already in the closure above;
      // it must never be inserted a second time in the smooth grid term.
      std::vector<Complex> dgrid_ev(projector_total * projector_total, Complex{});
      if (!std::getenv("SDFT_DISABLE_DGRID")) {
        if(batch_adjoint)dgrid_ev=std::move(batched_dgrid.at(atom));
        else {
          PerformanceTimer timer("paw_compensation_adjoint");
          dgrid_ev = closure_context->compensation.at(atom).project_potential(*total_local_potential);
        }
        for (auto& value : dgrid_ev) value *= ev_per_hartree;
      }
      if (paw_onecenter_energy_ev) *paw_onecenter_energy_ev += closure.energy_difference_ev;
      if (paw_onecenter_potential_expectation_ev) *paw_onecenter_potential_expectation_ev += closure.potential_expectation_ev;
      if (paw_onecenter_core_xc_ev) *paw_onecenter_core_xc_ev += closure.core_xc_energy_ev;
      if (paw_dc_ae_ev) *paw_dc_ae_ev += closure.ae_double_counting_ev;
      if (paw_dc_ps_ev) *paw_dc_ps_ev += closure.ps_double_counting_ev;
      double dnorm2 = 0.0;
      for (std::size_t x = 0; x < projector_total; ++x)
        for (std::size_t y = 0; y < projector_total; ++y) {
          const Complex value = closure.d_matrix_ev[x * projector_total + y];
          d[x * projector_total + y] += value / ev_per_hartree +
              dgrid_ev[x * projector_total + y] / ev_per_hartree;
          dnorm2 += std::norm(value);
        }
      // No exact-exchange one-centre term is added: PBE has none.
      if (captured_onecenter) {
        state.atom_index = atom;
        state.projector_dimension = projector_total;
        state.symbol = input.atoms[atom].symbol;
        state.fractional = input.atoms[atom].fractional;
        state.paw_cache = input.atoms[atom].paw_cache;
        state.record_id = pots[atom].record_id;
        state.numeric_cache_version = c.version;
        state.numeric_cache_checksum = paw_numeric_checksum(c);
        state.numeric_cache_sha256 = pots[atom].numeric_sha256;
        state.projector_scale = projector_scale;
        state.projector_q_bohr = std::getenv("SDFT_PROJECTOR_Q_BOHR") != nullptr;
        state.unit_convention = input.unit_convention;
        state.reciprocal_linear = std::getenv("SDFT_RECIPROCAL_LINEAR") != nullptr;
        state.radial_channel_l = std::move(raw_l);
        state.density_matrix = std::move(onsite);
        state.onsite_dij_ev.resize(d.size());
        for (std::size_t entry = 0; entry < d.size(); ++entry)
          state.onsite_dij_ev[entry] = d[entry] * ev_per_hartree;
        state.closure_correction_ev = closure.d_matrix_ev;
        state.compensation_correction_ev = dgrid_ev;
        state.lm_moment_imag = closure.lm_moment_imag;
        state.ae_energy_ev = closure.ae_energy_ev;
        state.ps_energy_ev = closure.ps_energy_ev;
        state.energy_difference_ev = closure.energy_difference_ev;
        state.potential_expectation_ev = closure.potential_expectation_ev;
        state.core_xc_energy_ev = closure.core_xc_energy_ev;
        captured_onecenter->push_back(std::move(state));
      }
      if (std::getenv("SDFT_TRACE_PAW_CLOSURE"))
        std::cerr << "PAW closure diagnostic atom=" << atom
                  << " d_norm_ev=" << std::sqrt(dnorm2)
                  << " ae_ev=" << closure.ae_energy_ev
                  << " ps_ev=" << closure.ps_energy_ev
                  << " delta_ev=" << closure.energy_difference_ev
                  << " expectation_ev=" << closure.potential_expectation_ev
                  << "\n";
    } else if (env_with_alias("SDFT_ENABLE_SPHERICAL_ONECENTER","SDFT_ENABLE_ONECENTER_CHF") && !env_with_alias("SDFT_DISABLE_SPHERICAL_ONECENTER","SDFT_DISABLE_ONECENTER_CHF") &&
        (density_matrix != nullptr || c.occupancies.size() == radial_count * radial_count)) {
      Eigen::MatrixXd onsite = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(projector_total), static_cast<Eigen::Index>(projector_total));
      std::vector<int> ls, nps;
      bool have_density = density_matrix && density_matrix->rows() == static_cast<Eigen::Index>(basis.g.size()) &&
                          density_matrix->cols() == static_cast<Eigen::Index>(basis.g.size()) && density_matrix->norm() > 1e-18;
      if (have_density) {
        std::vector<Complex> projections(projector_total * basis.g.size());
        for (std::size_t x=0; x<projector_total; ++x) {
          // Reconstruct the same beta projector used for H/S.  Keeping this
          // contraction in one routine is less error-prone than storing a
          // second, potentially differently normalized, projector matrix.
          std::size_t pidx=0;
          for (const auto& ch2 : c.nonlocal_channels) for (int np2=0; np2<ch2.nprojectors; ++np2)
            for (int m2=-ch2.angular_momentum; m2<=ch2.angular_momentum; ++m2,++pidx) if (pidx==x) {
              for (std::size_t ig=0; ig<basis.g.size(); ++ig) {
                const double q=norm(basis.cart[ig]);
                const double qmax=ch2.reciprocal_cutoff>0.0 ? ch2.reciprocal_cutoff : twopi/std::max(ch2.cutoff_radius,1e-8);
                projections[x*basis.g.size()+ig] = projector_scale * std::pow(Complex(0.0,1.0), ch2.angular_momentum) *
                  spherical_harmonic(ch2.angular_momentum,m2,basis.cart[ig]) *
                  projector_value_cutoff(pots[atom],ch2.reciprocal_projectors[np2], q*(std::getenv("SDFT_PROJECTOR_Q_BOHR")?1.0:bohr_per_angstrom),qmax,ch2.angular_momentum) *
                  std::exp(Complex(0.0,dot(basis.cart[ig],cartesian(a, input.atoms[atom].fractional)))) / std::sqrt(volume_ang3);
              }
            }
        }
        for (std::size_t x=0;x<projector_total;++x) for (std::size_t y=0;y<projector_total;++y) {
          Complex value(0.0,0.0);
          for (std::size_t i=0;i<basis.g.size();++i) for (std::size_t j=0;j<basis.g.size();++j)
            value += std::conj(projections[x*basis.g.size()+i]) * (*density_matrix)(static_cast<Eigen::Index>(i),static_cast<Eigen::Index>(j)) * projections[y*basis.g.size()+j];
          onsite(static_cast<Eigen::Index>(x),static_cast<Eigen::Index>(y)) = value.real();
        }
      }
      std::size_t es=0, rs=0;
      for (const auto& ch : c.nonlocal_channels) {
        ls.push_back(ch.angular_momentum); nps.push_back(ch.nprojectors);
        const int deg=2*ch.angular_momentum+1;
        if (!have_density) for (int i=0;i<ch.nprojectors;++i) for (int j=0;j<ch.nprojectors;++j)
          for (int m=0;m<deg;++m) onsite(static_cast<Eigen::Index>(es+i*deg+m),static_cast<Eigen::Index>(es+j*deg+m)) = c.occupancies[(rs+i)*radial_count+rs+j];
        es += static_cast<std::size_t>(ch.nprojectors*deg); rs += static_cast<std::size_t>(ch.nprojectors);
      }
      const auto one = paw_one_center_spherical(c, onsite, ls, nps, 1);
      if (paw_dc_ae_ev) *paw_dc_ae_ev += one.double_counting_ae_ev;
      if (paw_dc_ps_ev) *paw_dc_ps_ev += one.double_counting_ps_ev;
      if (paw_spherical_dij_norm_ev) *paw_spherical_dij_norm_ev += one.spherical_dij_ev.norm();
      if (paw_onecenter_energy_ev) *paw_onecenter_energy_ev += one.energy_difference_ev;
      if (paw_onecenter_potential_expectation_ev) *paw_onecenter_potential_expectation_ev += one.potential_expectation_ev;
      es=0; rs=0;
      for (const auto& ch : c.nonlocal_channels) {
        const int deg=2*ch.angular_momentum+1;
        for (int i=0;i<ch.nprojectors;++i) for (int j=0;j<ch.nprojectors;++j) for (int m=0;m<deg;++m) {
          const std::size_t row=es+static_cast<std::size_t>(i*deg+m), col=es+static_cast<std::size_t>(j*deg+m);
          d[row*projector_total+col] += one.spherical_dij_ev(static_cast<Eigen::Index>(rs+i),static_cast<Eigen::Index>(rs+j))/ev_per_hartree;
        }
        es += static_cast<std::size_t>(ch.nprojectors*deg); rs += static_cast<std::size_t>(ch.nprojectors);
      }
      if (std::getenv("SDFT_TRACE_ONECENTER")) std::cerr << "PAW spherical one-centre D_ij norm_eV=" << one.spherical_dij_ev.norm() << " double_ae_eV=" << one.double_counting_ae_ev << " double_ps_eV=" << one.double_counting_ps_ev << "\n";
    }
    MatrixC d_operator(projector_total, projector_total), q_operator(projector_total, projector_total);
    for (std::size_t i = 0; i < projector_total; ++i) for (std::size_t j = 0; j < projector_total; ++j) {
      d_operator(i, j) = d[i * projector_total + j];
      q_operator(i, j) = qmat[i * projector_total + j];
    }
    if(operator_blocks) {
      double radius=0;
      for(const auto& ch:pots[atom].cache.nonlocal_channels)radius=std::max(radius,ch.cutoff_radius);
      operator_blocks->push_back({projector_total,std::move(beta),std::move(d),std::move(qmat),
          input.atoms[atom].fractional,radius*bohr_per_angstrom,a});
    } else {
      h.noalias() += projector_basis.adjoint() * (d_operator * projector_basis);
      s.noalias() += projector_basis.adjoint() * (q_operator * projector_basis);
    }
  }
}

std::vector<Complex> density_from_coeff(const MatrixC& coeff, int occupied,
                                       const std::vector<double>& occupations) {
  const std::size_t n = static_cast<std::size_t>(coeff.rows());
  std::vector<Complex> rho(n * n, Complex(0.0, 0.0));
  for (int band = 0; band < occupied; ++band) {
    const double f = occupations[static_cast<std::size_t>(band)];
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
      rho[i * n + j] += f * coeff(static_cast<Eigen::Index>(i), band) * std::conj(coeff(static_cast<Eigen::Index>(j), band));
  }
  return rho;
}

// Stationary PAW ionic derivative. Projector and overlap contractions use
// the same final on-site D_ij and overlap correction q_ij as H/S. Explicit
// compensation and frozen-core
// translations are separate terms; density-response terms cancel only at a
// converged stationary state. Numerical force validation remains mandatory.
bool density_dataflow_control(const char* name) {
  const char* setting=std::getenv(name);if(!setting)return false;
  if(std::string(setting)!="0"&&std::string(setting)!="1")
    throw std::invalid_argument(std::string(name)+" must be 0 or 1");
  return std::string(setting)=="1";
}
// A verified frozen on-site D_ij does not request another compensation adjoint.
// This control retains the original full-grid construction as an oracle;
// the actual Hartree/PBE local operator and augmentation overlap are unchanged.
bool frozen_nscf_fields() {
  return !std::getenv("SDFT_NSCF_FROZEN_FIELDS") ||
      density_dataflow_control("SDFT_NSCF_FROZEN_FIELDS");
}
bool derivative_band_contraction() {
  const char* setting=std::getenv("SDFT_DERIVATIVE_CONTRACTION");
  if(!setting)return false;
  if(std::string(setting)!="0"&&std::string(setting)!="1")
    throw std::invalid_argument("SDFT_DERIVATIVE_CONTRACTION must be 0 or 1");
  return std::string(setting)=="1";
}
struct StationaryReciprocalData {
  std::vector<Vec3> cell_force_gradients;
  std::vector<Vec3> nlcc_force_gradients;
  std::array<Vec3,3> local_strain{},nlcc_strain{};
  std::vector<PawCompensationOperator::Derivatives> compensation;
};

bool derivative_option(const char* name) {
  const char* value=std::getenv(name);
  if(value&&std::string(value)!="0"&&std::string(value)!="1")
    throw std::invalid_argument(std::string(name)+" must be 0 or 1");
  return value&&std::string(value)=="1";
}

StationaryReciprocalData prepare_stationary_reciprocal(const ScfInput& input,
    const std::vector<AtomPotential>& pots,const PawScfClosureContext& context,
    const std::vector<ScfOneCenterState>& states,const FourierDensity& density,
    const DensityFunctionals& functional,double core_weight=1.) {
  PerformanceTimer timer("stationary_reciprocal_preparation");
  const auto units=atomic_units(input.unit_convention);
  const double volume=std::abs(determinant(input.lattice_bohr));
  const double volume_ang=volume/std::pow(units.bohr_per_angstrom,3),ev=units.ev_per_hartree;
  const bool batch=reciprocal_eval_batch();
  const ReciprocalEvaluationOptions evaluation;
  const bool local_enabled=!std::getenv("SDFT_DISABLE_LOCAL");
  const auto b=reciprocal(input.lattice_bohr);
  // Include XC-only core modes explicitly, retaining the original sparse
  // supports. They do not acquire an ionic potential or valence density.
  std::vector<std::tuple<int,int,int>> keys;
  std::vector<Complex> rho,xc;
  std::vector<unsigned char> density_present,xc_present;
  keys.reserve(std::max(density.size(),functional.xc_potential.size()));
  rho.reserve(keys.capacity());xc.reserve(keys.capacity());
  auto di=density.begin(),xi=functional.xc_potential.begin();
  while(di!=density.end() || xi!=functional.xc_potential.end()) {
    const auto key=xi==functional.xc_potential.end() || (di!=density.end() && di->first<xi->first)
        ? di->first : xi->first;
    const bool has_density=di!=density.end() && di->first==key;
    const bool has_xc=xi!=functional.xc_potential.end() && xi->first==key;
    keys.push_back(key);rho.push_back(has_density?di->second:Complex{});xc.push_back(has_xc?xi->second:Complex{});
    density_present.push_back(has_density);xc_present.push_back(has_xc);
    if(has_density)++di;if(has_xc)++xi;
  }
  std::vector<Vec3> q(keys.size());std::vector<double> radius(keys.size());
  for(std::size_t ig=0;ig<keys.size();++ig) {
    const auto [h,k,l]=keys[ig];q[ig]=cartesian(b,{double(h),double(k),double(l)});radius[ig]=norm(q[ig]);
  }
  struct Radial {std::vector<double> local,local_derivative,core,core_derivative;};
  std::vector<std::size_t> atom_species(pots.size()),representatives;
  std::map<std::string,std::size_t> species;
  for(std::size_t atom=0;atom<pots.size();++atom) {
    auto [found,inserted]=species.try_emplace(pots[atom].numeric_sha256,representatives.size());
    if(inserted)representatives.push_back(atom);
    // Cache identity alone is insufficient for diagnostic valence overrides.
    if(pots[representatives[found->second]].zval!=pots[atom].zval)
      throw std::invalid_argument("stationary reciprocal cache valence mismatch");
    atom_species[atom]=found->second;
  }
  std::vector<Radial> radial(representatives.size());
  std::exception_ptr failure;std::mutex failure_mutex;
  for(std::size_t sp=0;sp<radial.size();++sp) {
    auto& r=radial[sp];const auto& pot=pots[representatives[sp]];
    r.local.resize(keys.size());r.local_derivative.resize(keys.size());
    r.core.resize(keys.size());r.core_derivative.resize(keys.size());
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && keys.size()>=4096)
#endif
    for(std::size_t ig=0;ig<keys.size();++ig) {
      try {
        if(local_enabled&&density_present[ig]) {
          r.local[ig]=batch?local_fourier(pot,radius[ig],evaluation):local_fourier(pot,radius[ig]);
          r.local_derivative[ig]=batch?local_fourier_derivative(pot,radius[ig],evaluation):local_fourier_derivative(pot,radius[ig]);
        }
        if(xc_present[ig]&&!pot.cache.core_charge_partial.empty()) {
          const auto& values=pot.cache.core_charge_partial;
          const double coordinate=radius[ig]*units.bohr_per_angstrom,maximum=pot.cache.local_header.front();
          const auto sample=batch
              ? (evaluation.linear?RadialDensitySample{uniform_table_linear(values,coordinate,maximum),
                    reciprocal_table_derivative(values,coordinate,maximum)}:reciprocal_density_sample(values,coordinate,maximum))
              : density_table_sample(values,coordinate,maximum);
          r.core[ig]=core_weight*sample.value;r.core_derivative[ig]=core_weight*sample.derivative;
        }
      }catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
    }
    if(failure)std::rethrow_exception(failure);
  }
  // Reuse immutable translations from the same current structure and G
  // support. Build an owned array only for a genuine XC-only support union.
  std::vector<std::vector<Complex>> owned_phases(pots.size());
  std::vector<const std::vector<Complex>*> phases(pots.size());
  for(std::size_t atom=0;atom<pots.size();++atom) {
    if(context.compensation[atom].reciprocal_keys()==keys)
      phases[atom]=&context.compensation[atom].translation_phases();
    else {
      auto& phase=owned_phases[atom];phase.resize(keys.size());
      const auto position=cartesian(input.lattice_bohr,input.atoms[atom].fractional);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && keys.size()>=4096)
#endif
      for(std::size_t ig=0;ig<keys.size();++ig)phase[ig]=std::exp(Complex(0,-dot(q[ig],position)));
      phases[atom]=&phase;
    }
  }
  StationaryReciprocalData out;out.cell_force_gradients.resize(pots.size());out.nlcc_force_gradients.resize(pots.size());
  FourierDensity total_potential=functional.potential;
  for(std::size_t ig=0;ig<keys.size();++ig) {
    if(!density_present[ig])continue;
    const auto vector=q[ig];const double qs[]{vector.x,vector.y,vector.z};
    Complex potential{},derivative{};
    if(local_enabled)for(std::size_t atom=0;atom<pots.size();++atom) {
      const auto& r=radial[atom_species[atom]];const auto phase=(*phases[atom])[ig];
      potential+=r.local[ig]*phase/(volume_ang*ev);
      derivative+=r.local_derivative[ig]*phase/(volume_ang*ev);
    }
    total_potential[keys[ig]]+=potential;
    for(int a=0;a<3;++a)for(int c=0;c<3;++c) {
      auto& row=out.local_strain[a];
      (c==0?row.x:c==1?row.y:row.z)+=-volume*(std::conj(rho[ig])*((a==c?potential:Complex{})+
          (radius[ig]>1e-14?qs[a]*qs[c]/radius[ig]*derivative:Complex{}))).real();
    }
  }
  std::vector<std::array<Vec3,3>> nlcc_atoms(pots.size());
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && pots.size()>1 && keys.size()>=128)
#endif
  for(std::size_t atom=0;atom<pots.size();++atom) {
    const auto& r=radial[atom_species[atom]];
    auto& gradient=out.cell_force_gradients[atom];auto& nlcc=nlcc_atoms[atom];
    for(std::size_t ig=0;ig<keys.size();++ig) {
      const auto phase=(*phases[atom])[ig];const auto vector=q[ig];const double qs[]{vector.x,vector.y,vector.z};
      if(local_enabled&&density_present[ig]) {
        const Complex potential=r.local[ig]/(volume_ang*ev)*phase;
        const double response=volume*(std::conj(rho[ig])*potential*Complex(0,-1)).real();
        gradient=add(gradient,scale(vector,response));
      }
    }
    // Keep local, augmentation and NLCC summations separate. Combining them
    // per G changes cancellation in large Coulomb force contributions.
    if(!pots[atom].cache.core_charge_partial.empty())for(std::size_t ig=0;ig<keys.size();++ig) {
      if(!xc_present[ig])continue;
      const auto phase=(*phases[atom])[ig];const auto vector=q[ig];const double qs[]{vector.x,vector.y,vector.z};
      const Complex core=r.core[ig]/volume*phase;
      out.nlcc_force_gradients[atom]=add(out.nlcc_force_gradients[atom],scale(vector,volume*(std::conj(xc[ig])*core*Complex(0,-1)).real()));
      if(radius[ig]<1e-14)continue;
      const Complex derivative=r.core_derivative[ig]*units.bohr_per_angstrom*phase;
      for(int a=0;a<3;++a)for(int c=0;c<3;++c) {
        auto& row=nlcc[a];(c==0?row.x:c==1?row.y:row.z)+=-qs[a]*qs[c]/radius[ig]*(std::conj(xc[ig])*derivative).real();
      }
    }
  }
  for(std::size_t atom=0;atom<pots.size();++atom)for(int a=0;a<3;++a)
    out.nlcc_strain[a]=add(out.nlcc_strain[a],nlcc_atoms[atom][a]);
  std::vector<std::vector<Complex>> onsite;for(const auto& state:states)onsite.push_back(state.density_matrix);
  out.compensation=PawCompensationOperator::energy_derivatives_batch(context.compensation,onsite,total_potential);
  return out;
}

std::vector<Vec3> electronic_hf_forces(const ScfInput& input,
                                       const std::vector<AtomPotential>& pots,
                                       const Basis& basis,
                                       const std::array<Vec3, 3>& a,
                                       const MatrixC& coeff,
                                       const Eigen::VectorXd& eigvals,
                                       const std::vector<double>& occupations,
                                       int occupied,
                                       const std::vector<ScfOneCenterState>* states = nullptr,
                                       const PawScfClosureContext* context = nullptr,
                                       const FourierDensity* full_density = nullptr,
                                       const StationaryReciprocalData* prepared = nullptr,
                                       const std::vector<MatrixC>* cached_projectors = nullptr) {
  PerformanceTimer force_timer("analytic_forces_total");
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  const auto b = reciprocal(a);
  const double volume_bohr3 = std::abs(determinant(a));
  const double volume_ang3 = volume_bohr3 / std::pow(bohr_per_angstrom, 3);
  const bool fuse_derivatives=derivative_option("SDFT_DERIVATIVE_FUSED_CONTRACTION");
  if(cached_projectors&&cached_projectors->size()!=input.atoms.size())
    throw std::invalid_argument("cached force projector atom count mismatch");
  std::vector<Vec3> out(input.atoms.size());
  FourierDensity density;
  if (!prepared && full_density) density = *full_density;
  else if(!prepared) {
    const auto rho = density_from_coeff(coeff, occupied, occupations);
    // charge_coefficients is declared below; form the same Fourier map here.
    for (std::size_t i = 0; i < basis.g.size(); ++i)
      for (std::size_t j = 0; j < basis.g.size(); ++j)
        density[{basis.g[i].h-basis.g[j].h,basis.g[i].k-basis.g[j].k,
                 basis.g[i].l-basis.g[j].l}] += rho[i*basis.g.size()+j] / volume_bohr3;
  }
  FourierDensity total_potential, xc_potential;
  if (!prepared && context && !std::getenv("SDFT_DISABLE_HARTREE_XC")) {
    const auto functional = evaluate_density_functionals(density, a, context->pseudo_core);
    total_potential = functional.potential;
    xc_potential = functional.xc_potential;
  }
  std::vector<std::tuple<int,int,int>> keys;
  for (const auto& [key, value] : density) { (void)value; keys.push_back(key); }
  if (!prepared && context && !std::getenv("SDFT_DISABLE_LOCAL")) {
    for (const auto& key : keys) {
      const auto [h,k,l] = key;
      const Vec3 q = cartesian(b, {double(h),double(k),double(l)});
      for (std::size_t atom = 0; atom < input.atoms.size(); ++atom)
        total_potential[key] += local_fourier(pots[atom], norm(q)) /
            (volume_ang3 * ev_per_hartree) * std::exp(Complex(0,-dot(q,cartesian(a,input.atoms[atom].fractional))));
    }
  }
  const char* atom_setting=std::getenv("SDFT_FORCE_ATOM_PARALLEL");
  if(atom_setting&&std::string(atom_setting)!="0"&&std::string(atom_setting)!="1")
    throw std::invalid_argument("SDFT_FORCE_ATOM_PARALLEL must be 0 or 1");
  const bool atom_parallel=atom_setting&&std::string(atom_setting)=="1";
  std::exception_ptr atom_failure;std::mutex atom_failure_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(dynamic,1) if(atom_parallel && !omp_in_parallel() && input.atoms.size()>1)
#endif
  for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
    try {

    const Vec3 position = cartesian(a, input.atoms[atom].fractional);
    Vec3 gradient=prepared ? prepared->cell_force_gradients.at(atom) : Vec3{};
    // <n_smooth + n_comp, dV_ion/dR>. The compensation contribution must
    // be included here as well as in the explicit d(n_comp)/dR term below.
    if (!prepared && !std::getenv("SDFT_DISABLE_LOCAL"))
      for (const auto& [key, rho] : density) {
        const auto [h,k,l] = key;
        const Vec3 q = cartesian(b, {double(h),double(k),double(l)});
        const Complex v = local_fourier(pots[atom], norm(q)) /
            (volume_ang3 * ev_per_hartree) * std::exp(Complex(0,-dot(q,position)));
        const double response = volume_bohr3 * (std::conj(rho) * v * Complex(0,-1)).real();
        gradient = add(gradient, scale(q,response));
      }
    const auto& cache = pots[atom].cache;
    const MatrixC owned_beta = cached_projectors ? MatrixC{} : atom_projector_basis(input, pots[atom], atom, basis, a);
    const MatrixC& beta = cached_projectors ? cached_projectors->at(atom) : owned_beta;
    const auto nproj = static_cast<std::size_t>(beta.rows());
    MatrixC d = MatrixC::Zero(nproj,nproj), overlap = MatrixC::Zero(nproj,nproj);
    std::size_t expanded = 0, radial = 0, radial_count = 0;
    for (const auto& ch : cache.nonlocal_channels) radial_count += ch.nprojectors;
    for (const auto& ch : cache.nonlocal_channels) {
      const int deg = 2*ch.angular_momentum+1;
      for (int i=0;i<ch.nprojectors;++i) for (int j=0;j<ch.nprojectors;++j)
        for (int m=0;m<deg;++m) {
          const auto row=expanded+i*deg+m, col=expanded+j*deg+m;
          const double dscale = std::getenv("SDFT_D_SCALE") ? std::atof(std::getenv("SDFT_D_SCALE")) : 1.0;
          d(row,col)=dscale*ch.d_matrix[i][j]/ev_per_hartree;
          if (!std::getenv("SDFT_DISABLE_OVERLAP")) overlap(row,col)=context
              ? context->augmentation.at(atom).overlap_moment(radial+i,radial+j)
              : cache.augmentation_coefficients[(radial+i)*radial_count+radial+j];
        }
      expanded+=ch.nprojectors*deg;radial+=ch.nprojectors;
    }
    if (states) {
      const auto& state=states->at(atom);
      if (state.onsite_dij_ev.size()!=nproj*nproj) throw std::runtime_error("force on-site D_ij dimension mismatch");
      for (std::size_t i=0;i<nproj;++i) for (std::size_t j=0;j<nproj;++j)
        d(i,j)=state.onsite_dij_ev[i*nproj+j]/ev_per_hartree;
    }
    if (!std::getenv("SDFT_DISABLE_NONLOCAL")) {
      std::optional<PawDerivativeResponse> derivative_response;
      MatrixC projected;
      if(derivative_band_contraction()&&occupied>0)
        derivative_response=paw_derivative_response(nproj,basis.g.size(),occupied,
            beta.data(),coeff.data(),d.data(),overlap.data(),eigvals.data(),occupations.data());
      else projected=beta*coeff.leftCols(occupied);
      // 2 Re[(dB psi)^dagger (D-epsilon Q) B psi]. Each projection costs
      // O(Npw*Nproj*Nbands), replacing the former Npw^2*Nproj^2 loops.
      if(fuse_derivatives&&derivative_response) {
        gradient=add(gradient,paw_derivative_position_contract(*derivative_response,beta.data(),basis.cart));
      } else for (int axis=0;axis<3;++axis) {
        MatrixC derivative_beta=beta;
        for (std::size_t ig=0;ig<basis.cart.size();++ig) {
          const auto q=basis.cart[ig];const double component=axis==0?q.x:axis==1?q.y:q.z;
          derivative_beta.col(ig)*=Complex(0,component);
        }
        double value=0;
        if(derivative_response)value=paw_derivative_contract(*derivative_response,derivative_beta.data());
        else {
          const MatrixC projected_derivative=derivative_beta*coeff.leftCols(occupied);
          for (int band=0;band<occupied;++band) {
            const auto response=(d-eigvals[band]*overlap)*projected.col(band);
            value+=2*occupations[band]*(projected_derivative.col(band).dot(response)).real();
          }
        }
        if(axis==0)gradient.x+=value;else if(axis==1)gradient.y+=value;else gradient.z+=value;
      }
    }
    if(prepared) {
      gradient=add(gradient,prepared->compensation.at(atom).position);
      gradient=add(gradient,prepared->nlcc_force_gradients.at(atom));
    }
    else if (context && states) {
      gradient=add(gradient,context->compensation.at(atom).position_energy_gradient(
          states->at(atom).density_matrix,total_potential));
      // NLCC contributes only to XC, since the frozen pseudo core does not
      // enter valence Hartree charge or the electron-number constraint.
      for(const auto& [key, vxc]:xc_potential) {
        const auto [h,k,l]=key;
        const Vec3 q=cartesian(b,{double(h),double(k),double(l)});
        if(cache.core_charge_partial.empty())continue;
        const Complex core=density_table_sample(cache.core_charge_partial,norm(q)*bohr_per_angstrom,
            cache.local_header.front()).value/volume_bohr3*std::exp(Complex(0,-dot(q,position)));
        gradient=add(gradient,scale(q,volume_bohr3*(std::conj(vxc)*core*Complex(0,-1)).real()));
      }
    }
    out[atom]=scale(gradient,-ev_per_hartree*bohr_per_angstrom);
    }catch(...){std::lock_guard lock(atom_failure_mutex);if(!atom_failure)atom_failure=std::current_exception();}
  }
  if(atom_failure)std::rethrow_exception(atom_failure);
  return out;
}

std::array<Vec3,3> electronic_paw_stress(const ScfInput& input,const std::vector<AtomPotential>& pots,
    const std::vector<Basis>& bases,const std::vector<MatrixC>& coefficients,
    const std::vector<Eigen::VectorXd>& energies,const std::vector<std::vector<double>>& occupations,
    const std::vector<ScfOneCenterState>& states,const PawScfClosureContext& context,
    const FourierDensity& density,double local_g0_ev,
    std::map<std::string,std::array<Vec3,3>>& components,
    const DensityFunctionals* frozen_functional = nullptr,
    const StationaryReciprocalData* prepared = nullptr,
    const std::vector<std::vector<MatrixC>>* cached_projectors = nullptr) {
  PerformanceTimer stress_timer("analytic_stress_total");
  const bool fuse_derivatives=derivative_option("SDFT_DERIVATIVE_FUSED_CONTRACTION");
  if(cached_projectors) {
    if(cached_projectors->size()!=bases.size())throw std::invalid_argument("cached stress projector k count mismatch");
    for(const auto& k:*cached_projectors)if(k.size()!=input.atoms.size())
      throw std::invalid_argument("cached stress projector atom count mismatch");
  }
  const auto units=atomic_units(input.unit_convention);const double volume=std::abs(determinant(input.lattice_bohr));
  const double volume_ang=volume/std::pow(units.bohr_per_angstrom,3),ev=units.ev_per_hartree;
  const auto b=reciprocal(input.lattice_bohr);
  std::optional<DensityFunctionals> owned_functional;
  if(!frozen_functional)owned_functional=evaluate_density_functionals(density,input.lattice_bohr,context.pseudo_core);
  const auto& functional=frozen_functional ? *frozen_functional : *owned_functional;
  std::array<Vec3,3> gradient{};
  components.clear();
  auto add_component=[&](const char* term,int a,int c,double value){
    auto& row=gradient[a];(c==0?row.x:c==1?row.y:row.z)+=value;
    auto& part=components[term][a];(c==0?part.x:c==1?part.y:part.z)+=value;
  };
  auto component=[](const Vec3& row,int c){return c==0?row.x:c==1?row.y:row.z;};
  for(int a=0;a<3;++a)for(int c=0;c<3;++c)
    {
      add_component("hartree",a,c,volume*component(functional.hartree_stress[a],c));
      add_component("xc",a,c,volume*component(functional.xc_stress[a],c));
    }
  FourierDensity total_potential;
  if(!prepared)total_potential=functional.potential;
  // Ionic V and n both carry 1/volume. The reciprocal-table and compensation
  // shape responses are separate derivatives at fixed fractional atoms.
  if(!prepared)for(const auto& [key,rho]:density) {
    const auto [h,k,l]=key;const Vec3 q=cartesian(b,{double(h),double(k),double(l)});
    const double radius=norm(q),qs[]{q.x,q.y,q.z};Complex potential{},radial{};
    if(!std::getenv("SDFT_DISABLE_LOCAL"))for(std::size_t atom=0;atom<input.atoms.size();++atom) {
      const Complex phase=std::exp(Complex(0,-dot(q,cartesian(input.lattice_bohr,input.atoms[atom].fractional))));
      potential+=local_fourier(pots[atom],radius)*phase/(volume_ang*ev);
      radial+=local_fourier_derivative(pots[atom],radius)*phase/(volume_ang*ev);
    }
    total_potential[key]+=potential;
    for(int a=0;a<3;++a)for(int c=0;c<3;++c)
      add_component("local",a,c,-volume*(std::conj(rho)*((a==c?potential:Complex{})+
          (radius>1e-14?qs[a]*qs[c]/radius*radial:Complex{}))).real());
  }
  if(prepared)for(int a=0;a<3;++a)for(int c=0;c<3;++c){
    add_component("local",a,c,component(prepared->local_strain[a],c));
    add_component("nlcc",a,c,component(prepared->nlcc_strain[a],c));
  }
  for(std::size_t atom=0;atom<input.atoms.size();++atom) {
    const auto extra=prepared ? prepared->compensation.at(atom).shape : context.compensation[atom].shape_strain_energy_gradient(states[atom].density_matrix,total_potential);
    for(int a=0;a<3;++a)for(int c=0;c<3;++c)add_component("augmentation",a,c,component(extra[a],c));
    const auto& cache=pots[atom].cache;
    if(!prepared && !cache.core_charge_partial.empty())for(const auto& [key,vxc]:functional.xc_potential) {
      const auto [h,k,l]=key;const Vec3 q=cartesian(b,{double(h),double(k),double(l)});
      const double radius=norm(q),qs[]{q.x,q.y,q.z};if(radius<1e-14)continue;
      const Complex phase=std::exp(Complex(0,-dot(q,cartesian(input.lattice_bohr,input.atoms[atom].fractional))));
      const Complex radial=density_table_sample(cache.core_charge_partial,radius*units.bohr_per_angstrom,
          cache.local_header.front()).derivative*units.bohr_per_angstrom*phase;
      for(int a=0;a<3;++a)for(int c=0;c<3;++c)
        add_component("nlcc",a,c,-qs[a]*qs[c]/radius*(std::conj(vxc)*radial).real());
    }
  }
  for(std::size_t ik=0;ik<bases.size();++ik) {
    const auto& basis=bases[ik];const auto& coeff=coefficients[ik];const auto& eig=energies[ik];
    const double weight=input.kpoints.empty()?1.:input.kpoints[ik].weight/
        std::accumulate(input.kpoints.begin(),input.kpoints.end(),0.,[](double sum,const auto& k){return sum+k.weight;});
    const int bands=coeff.cols();
    for(std::size_t ig=0;ig<basis.g.size();++ig) {
      const auto q=basis.cart[ig];const double qs[]{q.x,q.y,q.z};double occupation=0;
      for(int band=0;band<bands;++band)occupation+=occupations[ik][band]*std::norm(coeff(ig,band));
      for(int a=0;a<3;++a)for(int c=0;c<3;++c)add_component("kinetic",a,c,-weight*occupation*qs[a]*qs[c]);
    }
    const char* setting=std::getenv("SDFT_FORCE_ATOM_PARALLEL");
    if(setting&&std::string(setting)!="0"&&std::string(setting)!="1")throw std::invalid_argument("SDFT_FORCE_ATOM_PARALLEL must be 0 or 1");
    const bool parallel_atoms=setting&&std::string(setting)=="1";
    std::vector<std::array<Vec3,3>> nonlocal_atoms(input.atoms.size());
    std::exception_ptr stress_failure;std::mutex stress_failure_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(dynamic,1) if(parallel_atoms && !omp_in_parallel() && input.atoms.size()>1)
#endif
    for(std::size_t atom=0;atom<input.atoms.size();++atom) {
      try {

      const auto& cache=pots[atom].cache;
      const MatrixC owned_beta=cached_projectors ? MatrixC{} : atom_projector_basis(input,pots[atom],atom,basis,input.lattice_bohr);
      const MatrixC& beta=cached_projectors ? cached_projectors->at(ik).at(atom) : owned_beta;
      const std::size_t dim=beta.rows();MatrixC d(dim,dim),overlap=MatrixC::Zero(dim,dim);
      for(std::size_t i=0;i<dim;++i)for(std::size_t j=0;j<dim;++j)d(i,j)=states[atom].onsite_dij_ev[i*dim+j]/ev;
      std::size_t expanded=0,radial=0;
      for(const auto& ch:cache.nonlocal_channels) {
        const int deg=2*ch.angular_momentum+1;
        if(!std::getenv("SDFT_DISABLE_OVERLAP"))for(int i=0;i<ch.nprojectors;++i)for(int j=0;j<ch.nprojectors;++j)for(int m=0;m<deg;++m)
          overlap(expanded+i*deg+m,expanded+j*deg+m)=context.augmentation[atom].overlap_moment(radial+i,radial+j);
        expanded+=ch.nprojectors*deg;radial+=ch.nprojectors;
      }
      std::optional<PawDerivativeResponse> derivative_response;
      MatrixC projected;
      if(derivative_band_contraction())
        derivative_response=paw_derivative_response(dim,basis.g.size(),bands,
            beta.data(),coeff.data(),d.data(),overlap.data(),eig.data(),occupations[ik].data());
      else projected=beta*coeff;
      const double factor=(std::getenv("SDFT_PROJECTOR_SCALE")?std::atof(std::getenv("SDFT_PROJECTOR_SCALE")):1.)/std::sqrt(volume_ang);
      const double length=std::getenv("SDFT_PROJECTOR_Q_BOHR")?1.:units.bohr_per_angstrom;
      std::array<MatrixC,3> gradients{MatrixC(dim,basis.g.size()),MatrixC(dim,basis.g.size()),MatrixC(dim,basis.g.size())};
      std::size_t row=0;
      for(const auto& ch:cache.nonlocal_channels)for(int projector=0;projector<ch.nprojectors;++projector)
        for(int m=-ch.angular_momentum;m<=ch.angular_momentum;++m,++row)for(std::size_t ig=0;ig<basis.g.size();++ig) {
          const Vec3 q=basis.cart[ig];const double radius=norm(q),qs[]{q.x,q.y,q.z};
          const double qmax=ch.reciprocal_cutoff>0?ch.reciprocal_cutoff:twopi/std::max(ch.cutoff_radius,1e-8);
          const auto sample=projector_sample(pots[atom],ch.reciprocal_projectors[projector],radius*length,qmax,ch.angular_momentum);
          const double value=sample.value,derivative=sample.derivative*length;
          const auto dy=paw_real_spherical_harmonic_gradient(ch.angular_momentum,m,q);const double dys[]{dy.x,dy.y,dy.z};
          const Complex phase=factor*std::pow(Complex(0,1),ch.angular_momentum)*
              std::exp(Complex(0,dot(q,cartesian(input.lattice_bohr,input.atoms[atom].fractional))));
          for(int a=0;a<3;++a)gradients[a](row,ig)=phase*(value*dys[a]+
              (radius>1e-14?derivative*qs[a]/radius*spherical_harmonic(ch.angular_momentum,m,q):0.));
        }
      if(fuse_derivatives&&derivative_response) {
        const auto contracted=paw_derivative_strain_contract(*derivative_response,beta.data(),
            {gradients[0].data(),gradients[1].data(),gradients[2].data()},basis.cart);
        for(int a=0;a<3;++a)nonlocal_atoms[atom][a]=scale(contracted[a],weight);
      } else for(int a=0;a<3;++a)for(int c=0;c<3;++c) {
        MatrixC derivative_beta=gradients[a];
        for(std::size_t ig=0;ig<basis.g.size();++ig)derivative_beta.col(ig)*=-component(basis.cart[ig],c);
        if(a==c)derivative_beta-=.5*beta;
        double value=0;
        if(derivative_response)value=paw_derivative_contract(*derivative_response,derivative_beta.data());
        else {
          const MatrixC projected_derivative=derivative_beta*coeff;
          for(int band=0;band<bands;++band)value+=2*occupations[ik][band]*
              (projected_derivative.col(band).dot((d-eig[band]*overlap)*projected.col(band))).real();
        }
        auto& r=nonlocal_atoms[atom][a];(c==0?r.x:c==1?r.y:r.z)=weight*value;
      }
      }catch(...){std::lock_guard lock(stress_failure_mutex);if(!stress_failure)stress_failure=std::current_exception();}
    }
    if(stress_failure)std::rethrow_exception(stress_failure);
    // Original atom order for the component ledger, independent of task scheduling.
    for(std::size_t atom=0;atom<input.atoms.size();++atom)for(int a=0;a<3;++a)for(int c=0;c<3;++c)
      add_component("nonlocal",a,c,component(nonlocal_atoms[atom][a],c));
  }
  for(int a=0;a<3;++a)add_component("local_g0",a,a,-local_g0_ev/ev);
  const double conversion=ev*std::pow(units.bohr_per_angstrom,3)/volume;
  for(auto& [name,tensor]:components)for(auto& row:tensor)row=scale(row,conversion);
  // The component ledger is the authoritative stationary derivative. Adding
  // millions of terms in a second interleaved accumulator can drift from
  // its own disjoint ledger even though both expressions are algebraically
  // equal. Sum the nine converted components once, without fitting a stress.
  gradient={};
  for(const auto& [name,tensor]:components)for(int a=0;a<3;++a)gradient[a]=add(gradient[a],tensor[a]);
  return gradient;
}

double density_residual(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double ion_ion_ewald(const ScfInput& input, const std::array<Vec3, 3>& a,
                     const std::array<Vec3, 3>& b,
                     std::vector<Vec3>* forces = nullptr) {
  (void)b;
  std::vector<Vec3> fractional;
  std::vector<double> charges;
  for (const auto& atom : input.atoms) {
    fractional.push_back(atom.fractional);
    charges.push_back(atom.zval);
  }
  const auto ewald = evaluate_ewald(a, fractional, charges);
  if (forces) *forces = ewald.forces;
  return ewald.energy;
}

std::map<std::tuple<int, int, int>, Complex> charge_coefficients(const std::vector<Complex>& rho,
                                                                 const Basis& basis,
                                                                 double volume) {
  const std::size_t n = basis.g.size();
  std::map<std::tuple<int, int, int>, Complex> out;
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
    out[{basis.g[i].h - basis.g[j].h, basis.g[i].k - basis.g[j].k,
         basis.g[i].l - basis.g[j].l}] += rho[i * n + j] / volume;
  }
  return out;
}

double radial_overlap_form_factor(const PawNumericCache& cache, std::size_t i,
                                  std::size_t j, int angular_momentum, double q_angstrom) {
  if (i >= cache.pseudo_wavefunctions.size() || j >= cache.pseudo_wavefunctions.size()) return 0.0;
  if (cache.radial_grid.size() < 2 || cache.radial_grid.size() != cache.ae_wavefunctions[i].size() ||
      cache.radial_grid.size() != cache.pseudo_wavefunctions[i].size()) return 0.0;
  if (q_angstrom < 1e-10) {
    const std::size_t n = static_cast<std::size_t>(std::sqrt(static_cast<double>(cache.augmentation_coefficients.size())));
    if (n == cache.pseudo_wavefunctions.size()) return cache.augmentation_coefficients[i * n + j];
  }
  // POTCAR radial functions are reduced radial functions u_l(r).  The scalar
  // augmentation Fourier component is the radial overlap difference times
  // j_0(qr); the l-dependent angular multipoles are kept for the later
  // compensation-charge implementation.
  double integral = 0.0;
  for (std::size_t n = 0; n + 1 < cache.radial_grid.size(); ++n) {
    const double r0 = cache.radial_grid[n], r1 = std::min(cache.radial_grid[n + 1], cache.radial_rmax > 0.0 ? cache.radial_rmax : cache.radial_grid.back());
    if (r0 >= r1) continue;
    const double d0 = cache.ae_wavefunctions[i][n] * cache.ae_wavefunctions[j][n] -
                      cache.pseudo_wavefunctions[i][n] * cache.pseudo_wavefunctions[j][n];
    const double d1 = cache.ae_wavefunctions[i][n + 1] * cache.ae_wavefunctions[j][n + 1] -
                      cache.pseudo_wavefunctions[i][n + 1] * cache.pseudo_wavefunctions[j][n + 1];
    const double x0 = q_angstrom * r0, x1 = q_angstrom * r1;
    const double j0 = std::abs(x0) < 1e-10 ? 1.0 : std::sin(x0) / x0;
    const double j1 = std::abs(x1) < 1e-10 ? 1.0 : std::sin(x1) / x1;
    integral += 0.5 * (r1 - r0) * (d0 * j0 + d1 * j1);
  }
  (void)angular_momentum;
  return integral;
}

double radial_aug_form_factor(const PawNumericCache& cache, std::size_t i,
                              std::size_t j, int lout, double q_angstrom) {
  if (i >= cache.pseudo_wavefunctions.size() || j >= cache.pseudo_wavefunctions.size()) return 0.0;
  const auto& r = cache.radial_grid;
  if (r.size() < 2 || r.size() != cache.ae_wavefunctions[i].size() ||
      r.size() != cache.ae_wavefunctions[j].size()) return 0.0;
  double integral = 0.0;
  for (std::size_t n = 0; n + 1 < r.size(); ++n) {
    const double rmax = cache.radial_rmax > 0.0 ? cache.radial_rmax : r.back();
    if (r[n] >= rmax) break;
    const auto value = [&](std::size_t k) {
      const double d = cache.ae_wavefunctions[i][k] * cache.ae_wavefunctions[j][k] -
                       cache.pseudo_wavefunctions[i][k] * cache.pseudo_wavefunctions[j][k];
      const double x = q_angstrom * r[k];
      double jl = 0.0;
      if (lout == 0) jl = std::abs(x) < 1e-10 ? 1.0 : std::sin(x) / x;
      else if (lout == 1) jl = std::abs(x) < 1e-8 ? x / 3.0 : std::sin(x) / (x*x) - std::cos(x) / x;
      else if (lout == 2) jl = std::abs(x) < 1e-6 ? x*x / 15.0 :
          (3.0/(x*x*x) - 1.0/x) * std::sin(x) - 3.0*std::cos(x)/(x*x);
      else if (lout == 3) jl = std::abs(x) < 1e-4 ? x*x*x / 105.0 :
          (15.0/(x*x*x*x) - 6.0/(x*x)) * std::sin(x) + (1.0/x - 15.0/(x*x*x)) * std::cos(x);
      else return 0.0;
      return d * jl;
    };
    const double r1 = std::min(r[n + 1], rmax);
    integral += 0.5 * (r1 - r[n]) * (value(n) + value(n + 1));
  }
  return integral;
}

double spherical_bessel(int l, double x) {
  if (l < 0) return 0.0;
  if (std::abs(x) < 1e-8) return l == 0 ? 1.0 : (l == 1 ? x / 3.0 : 0.0);
  const double j0 = std::sin(x) / x;
  if (l == 0) return j0;
  const double j1 = std::sin(x) / (x * x) - std::cos(x) / x;
  if (l == 1) return j1;
  double jm1 = j0, j = j1;
  for (int n = 1; n < l; ++n) {
    const double jp1 = (2.0 * n + 1.0) * j / x - jm1;
    jm1 = j; j = jp1;
  }
  return j;
}

std::array<double, 2> first_bessel_roots(int l) {
  std::array<double, 2> roots{0.0, 0.0};
  double x0 = 1e-5, f0 = spherical_bessel(l, x0);
  int found = 0;
  for (double x = 0.05; x < 80.0 && found < 2; x += 0.05) {
    const double f = spherical_bessel(l, x);
    if (f0 * f <= 0.0) {
      double lo = x - 0.05, hi = x;
      for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (spherical_bessel(l, lo) * spherical_bessel(l, mid) <= 0.0) hi = mid;
        else lo = mid;
      }
      roots[static_cast<std::size_t>(found++)] = 0.5 * (lo + hi);
      f0 = spherical_bessel(l, x);
    } else {
      f0 = f;
    }
    x0 = x;
    (void)x0;
  }
  if (found != 2) throw std::runtime_error("could not construct PAW compensation Bessel roots");
  return roots;
}

double radial_aug_moment(const PawNumericCache& cache, std::size_t i, std::size_t j, int l) {
  if (i >= cache.ae_wavefunctions.size() || j >= cache.ae_wavefunctions.size()) return 0.0;
  if (cache.radial_grid.size() < 2 || cache.radial_grid.size() != cache.ae_wavefunctions[i].size() ||
      cache.radial_grid.size() != cache.pseudo_wavefunctions[i].size()) return 0.0;
  double value = 0.0;
  for (std::size_t n = 0; n + 1 < cache.radial_grid.size(); ++n) {
    const double rmax = cache.radial_rmax > 0.0 ? cache.radial_rmax : cache.radial_grid.back();
    if (cache.radial_grid[n] >= rmax) break;
    const double r0 = cache.radial_grid[n], r1 = std::min(cache.radial_grid[n + 1], rmax);
    if (r1 <= r0) continue;
    const double d0 = cache.ae_wavefunctions[i][n] * cache.ae_wavefunctions[j][n] -
                      cache.pseudo_wavefunctions[i][n] * cache.pseudo_wavefunctions[j][n];
    const double d1 = cache.ae_wavefunctions[i][n + 1] * cache.ae_wavefunctions[j][n + 1] -
                      cache.pseudo_wavefunctions[i][n + 1] * cache.pseudo_wavefunctions[j][n + 1];
    value += 0.5 * (r1 - r0) * (d0 * std::pow(r0, l) + d1 * std::pow(r1, l));
  }
  return value;
}

double compensation_form_factor(const PawNumericCache& cache, std::size_t i, std::size_t j,
                                int channel_l_i, int channel_l_j, int l, double q) {
  if (cache.radial_grid.size() < 2) return 0.0;
  const double rmax = cache.radial_rmax > 0.0 ? cache.radial_rmax : cache.radial_grid.back();
  if (rmax <= 0.0) return 0.0;
  const auto roots = first_bessel_roots(l);
  const double q1 = roots[0] / rmax, q2 = roots[1] / rmax;
  auto derivative = [&](double x) {
    if (l == 0) return -spherical_bessel(1, x);
    return spherical_bessel(l - 1, x) - (l + 1.0) / x * spherical_bessel(l, x);
  };
  const double d1 = q1 * derivative(roots[0]);
  const double d2 = q2 * derivative(roots[1]);
  auto moment = [&](double qq) {
    double sum = 0.0;
    for (std::size_t n = 0; n + 1 < cache.radial_grid.size(); ++n) {
      const double r0 = cache.radial_grid[n], r1 = std::min(cache.radial_grid[n + 1], rmax);
      if (r0 >= rmax || r1 <= r0) continue;
      sum += 0.5 * (r1 - r0) * (spherical_bessel(l, qq * r0) * std::pow(r0, l + 2) +
                                  spherical_bessel(l, qq * r1) * std::pow(r1, l + 2));
    }
    return sum;
  };
  const double m1 = moment(q1), m2 = moment(q2);
  const double det = d1 * m2 - d2 * m1;
  if (std::abs(det) < 1e-14) return 0.0;
  const double a1 = -d2 / det, a2 = d1 / det;
  const double aug_moment = radial_aug_moment(cache, i, j, l);
  if (std::abs(aug_moment) < 1e-16) return 0.0;
  double integral = 0.0;
  for (std::size_t n = 0; n + 1 < cache.radial_grid.size(); ++n) {
    const double r0 = cache.radial_grid[n], r1 = std::min(cache.radial_grid[n + 1], rmax);
    if (r0 >= rmax || r1 <= r0) continue;
    auto aug = [&](double r) {
      return r * r * (a1 * spherical_bessel(l, q1 * r) + a2 * spherical_bessel(l, q2 * r));
    };
    const double f0 = aug(r0) * spherical_bessel(l, q * r0);
    const double f1 = aug(r1) * spherical_bessel(l, q * r1);
    integral += 0.5 * (r1 - r0) * (f0 + f1);
  }
  (void)channel_l_i; (void)channel_l_j;
  return aug_moment * integral;
}

// The diagnostic full augmentation builder is intentionally bounded.  The
// production path must precompute radial Fourier tables; rebuilding Bessel
// roots and radial quadratures inside every G-pair is prohibitively expensive.
bool full_augmentation_allowed(const Basis& basis) {
  const char* enabled = std::getenv("SDFT_ENABLE_FULL_AUGMENTATION");
  return enabled && std::string(enabled) == "1" && basis.g.size() <= 180;
}

double real_gaunt(int l1, int m1, int l2, int m2, int l3, int m3) {
  // Tensor products in PAW are smooth polynomials for the l<=3 channels in
  // PAW_PBE.54.  An 8-point Gauss-Legendre x 24-point azimuth quadrature is
  // exact for the polynomial degree needed by the supported channels and
  // avoids depending on a spherical-harmonics library in the C++ core.
  static const double x[8] = {-0.9602898564975363,-0.7966664774136267,
    -0.5255324099163290,-0.1834346424956498,0.1834346424956498,
    0.5255324099163290,0.7966664774136267,0.9602898564975363};
  static const double w[8] = {0.1012285362903763,0.2223810344533745,
    0.3137066458778873,0.3626837833783620,0.3626837833783620,
    0.3137066458778873,0.2223810344533745,0.1012285362903763};
  double sum = 0.0;
  for (int ix = 0; ix < 8; ++ix) {
    const double z = x[ix], st = std::sqrt(std::max(0.0, 1.0 - z*z));
    for (int ip = 0; ip < 24; ++ip) {
      const double phi = twopi * static_cast<double>(ip) / 24.0;
      const Vec3 u{st * std::cos(phi), st * std::sin(phi), z};
      sum += w[ix] * (twopi / 24.0) * spherical_harmonic(l1,m1,u) *
             spherical_harmonic(l2,m2,u) * spherical_harmonic(l3,m3,u);
    }
  }
  return sum;
}

std::map<std::tuple<int, int, int>, Complex> full_augmentation_density(
    const ScfInput& input, const std::vector<AtomPotential>& pots, const Basis& basis,
    const MatrixC& coeff, int occupied, const std::vector<double>& occupations) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  std::map<std::tuple<int, int, int>, Complex> out;
  const double volume = std::abs(determinant(input.lattice_bohr));
  const double volume_ang3 = volume / std::pow(bohr_per_angstrom, 3);
  for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
    const auto& cache = pots[atom].cache;
    const Vec3 rion = cartesian(input.lattice_bohr, input.atoms[atom].fractional);
    struct RawChannel { int l; int nproj; std::size_t raw; std::size_t expanded; };
    std::vector<RawChannel> channels;
    std::size_t raw = 0, expanded = 0;
    for (const auto& ch : cache.nonlocal_channels) {
      channels.push_back({ch.angular_momentum, ch.nprojectors, raw, expanded});
      raw += static_cast<std::size_t>(ch.nprojectors);
      expanded += static_cast<std::size_t>(ch.nprojectors * (2 * ch.angular_momentum + 1));
    }
    if (expanded == 0 || coeff.cols() < occupied) continue;
    std::vector<Complex> beta(expanded * basis.g.size());
    std::size_t p = 0;
    for (const auto& ch : cache.nonlocal_channels) for (int np = 0; np < ch.nprojectors; ++np)
      for (int m = -ch.angular_momentum; m <= ch.angular_momentum; ++m, ++p)
        for (std::size_t ig = 0; ig < basis.g.size(); ++ig) {
          const double q = norm(basis.cart[ig]);
          const double qmax = ch.reciprocal_cutoff > 0.0 ? ch.reciprocal_cutoff : twopi / std::max(ch.cutoff_radius, 1e-8);
          beta[p * basis.g.size() + ig] = std::pow(Complex(0.0, 1.0), ch.angular_momentum) *
            spherical_harmonic(ch.angular_momentum, m, basis.cart[ig]) *
            projector_value_cutoff(pots[atom],ch.reciprocal_projectors[np], q * (std::getenv("SDFT_PROJECTOR_Q_BOHR") ? 1.0 : bohr_per_angstrom), qmax,
                                   ch.angular_momentum) /
            std::sqrt(volume_ang3) * std::exp(Complex(0.0, dot(basis.cart[ig], rion)));
        }
    std::vector<Complex> onsite(expanded * expanded);
    for (int band = 0; band < occupied; ++band) {
      std::vector<Complex> proj(expanded);
      for (std::size_t x = 0; x < expanded; ++x)
        for (std::size_t ig = 0; ig < basis.g.size(); ++ig)
          proj[x] += std::conj(beta[x * basis.g.size() + ig]) * coeff(static_cast<Eigen::Index>(ig), band);
      for (std::size_t x = 0; x < expanded; ++x) for (std::size_t y = 0; y < expanded; ++y)
        // On-site density matrix is D_ij = sum_n f_n <psi|beta_i>
        // <beta_j|psi> = p_i * conj(p_j).  The previous reversed
        // conjugation transposed the complex density for non-Gamma k points
        // and produced a non-Hermitian augmentation charge.
        onsite[x * expanded + y] += occupations[static_cast<std::size_t>(band)] * proj[x] * std::conj(proj[y]);
    }
    // Full augmentation path: contract the on-site occupation with Q_ij^L and
    // the real Gaunt coefficients to (L,M) moments, as in the radial closure.
    // This branch is opt-in while its Hamiltonian derivative is being
    // validated, but it is no longer the old scalar L=0 overlap approximation.
    std::vector<std::complex<double>> exact_lm_moments;
    std::optional<PawAugmentationData> exact_aug;
    if (env_with_alias("SDFT_ENABLE_EXACT_AUG_MOMENTS","SDFT_ENABLE_EXACT_QPAW")) {
      std::vector<int> raw_l;
      for (const auto& ch : cache.nonlocal_channels)
        for (int np=0; np<ch.nprojectors; ++np) raw_l.push_back(ch.angular_momentum);
      exact_aug = build_paw_augmentation(cache, raw_l);
      exact_lm_moments = paw_lm_moments_from_density(*exact_aug, onsite, expanded);
    }
    // Avoid rebuilding radial Bessel quadratures for every plane-wave pair:
    // cache the compensation form factors for the distinct |G-G'| values in
    // this basis.
    std::vector<double> qnorms;
    for (std::size_t gi=0; gi<basis.g.size(); ++gi) for (std::size_t gj=0; gj<basis.g.size(); ++gj) {
      const double qn=norm(sub(basis.g[gi].cartesian,basis.g[gj].cartesian));
      qnorms.push_back(qn);
    }
    std::sort(qnorms.begin(),qnorms.end());
    qnorms.erase(std::unique(qnorms.begin(),qnorms.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),qnorms.end());
    constexpr int maxL=6; const std::size_t nraw=raw;
    std::vector<std::vector<double>> comp(qnorms.size(), std::vector<double>(nraw*nraw*(maxL+1),0.0));
    for (std::size_t iq=0;iq<qnorms.size();++iq) for (std::size_t ri=0;ri<nraw;++ri) for (std::size_t rj=0;rj<nraw;++rj)
      for (int L=0;L<=maxL;++L) comp[iq][(ri*nraw+rj)*(maxL+1)+static_cast<std::size_t>(L)] = compensation_form_factor(cache,ri,rj,0,0,L,qnorms[iq]);
    // The plane-wave density contraction already sums coefficient pairs with
    // the same reciprocal difference Q.  Evaluate each augmentation form
    // factor once per unique Q; looping over every (G,G') pair would multiply
    // the onsite augmentation charge by the number of pairs sharing Q.
    std::map<std::tuple<int,int,int>, Vec3> unique_q;
    for (std::size_t gi = 0; gi < basis.g.size(); ++gi) for (std::size_t gj = 0; gj < basis.g.size(); ++gj) {
      const auto key = std::make_tuple(basis.g[gi].h - basis.g[gj].h, basis.g[gi].k - basis.g[gj].k, basis.g[gi].l - basis.g[gj].l);
      unique_q.emplace(key, sub(basis.g[gi].cartesian, basis.g[gj].cartesian));
    }
    for (const auto& [key, qvec] : unique_q) {
      // POTCAR radial grids and WAE/WPS are in atomic units.  qvec is
      // already in bohr^-1, so the Bessel argument is qvec*r directly.
      // Multiplying by bohr_per_angstrom here double-converts the argument
      // and was the source of the large non-zero-G augmentation error.
      const double q = norm(qvec), qang = q;
      std::size_t iq=0; while (iq+1<qnorms.size() && std::abs(qnorms[iq]-q)>1e-12) ++iq;
      const Vec3 qhat = q < 1e-12 ? Vec3{0,0,1} : scale(qvec, 1.0/q);
      Complex value(0.0, 0.0);
      if (exact_aug) {
        const double qnorm = norm(qvec);
        const Vec3 qhat = qnorm < 1e-14 ? Vec3{0,0,1} : scale(qvec,1.0/qnorm);
        for (int L=0; L<=exact_aug->max_l; ++L) for (int M=-L; M<=L; ++M) {
          const auto& shape = exact_aug->compensation_r2[static_cast<std::size_t>(L)];
          double radial=0.0;
          for (std::size_t ir=0; ir+1<shape.size(); ++ir) {
            const double r0=exact_aug->radial_grid[ir], r1=exact_aug->radial_grid[ir+1];
            const double f0=shape[ir]*spherical_bessel(L,qnorm*r0), f1=shape[ir+1]*spherical_bessel(L,qnorm*r1);
            radial += 0.5*(r1-r0)*(f0+f1);
          }
          value += exact_lm_moments[static_cast<std::size_t>(L*L+M+L)] *
                   spherical_harmonic(L,M,qhat) * std::pow(Complex(0.0,1.0),L) * 4.0*pi * radial;
        }
      } else {
      for (const auto& ci : channels) for (const auto& cj : channels) {
            const int lmax = std::min(3, ci.l + cj.l);
        for (int ni = 0; ni < ci.nproj; ++ni) for (int nj = 0; nj < cj.nproj; ++nj)
          for (int mi = -ci.l; mi <= ci.l; ++mi) for (int mj = -cj.l; mj <= cj.l; ++mj) {
            const std::size_t ix = ci.expanded + static_cast<std::size_t>(ni * (2*ci.l+1) + (mi + ci.l));
            const std::size_t iy = cj.expanded + static_cast<std::size_t>(nj * (2*cj.l+1) + (mj + cj.l));
            const Complex rhoij = onsite[ix * expanded + iy];
            for (int L = std::abs(ci.l-cj.l); L <= lmax; L += 2) for (int M = -L; M <= L; ++M) {
              const double g = real_gaunt(ci.l,mi,cj.l,mj,L,M);
              if (std::abs(g) < 1e-12) continue;
              const double radial = comp[iq][((ci.raw + static_cast<std::size_t>(ni))*nraw + cj.raw + static_cast<std::size_t>(nj))*(maxL+1)+static_cast<std::size_t>(L)];
              const Complex angular = spherical_harmonic(L,M,qhat) * std::pow(Complex(0.0,1.0), L) * (4.0*pi) * g * radial;
              value += rhoij * angular;
            }
          }
      }
      }
      out[key] += value * std::exp(Complex(0.0, -dot(qvec, rion))) / volume;
    }
  }
  return out;
}

std::map<std::tuple<int, int, int>, Complex> augmentation_density(
    const ScfInput& input, const std::vector<AtomPotential>& pots, const Basis& basis,
    const MatrixC& coeff, int occupied, const std::vector<double>& occupations) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  std::map<std::tuple<int, int, int>, Complex> out;
  // The scalar radial reconstruction is retained as an opt-in diagnostic
  // until the non-spherical compensation-charge tables are available.  A
  // partial augmentation density would violate charge neutrality and make
  // the Hartree term diverge; production runs therefore require the full
  // compensation path explicitly.
  if (!std::getenv("SDFT_ENABLE_AUGMENTATION") || std::getenv("SDFT_DISABLE_AUGMENTATION")) return out;
  const double volume_bohr3 = std::abs(determinant(input.lattice_bohr));
  const auto b = reciprocal(input.lattice_bohr);
  const std::size_t nbasis = basis.g.size();
  for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
    const auto& cache = pots[atom].cache;
    const Vec3 r = cartesian(input.lattice_bohr, input.atoms[atom].fractional);
    std::vector<ProjectorRef> refs;
    std::vector<std::size_t> raw_index;
    std::vector<int> channel_l;
    std::size_t raw_offset = 0;
    for (const auto& ch : cache.nonlocal_channels) {
      channel_l.push_back(ch.angular_momentum);
      for (int np = 0; np < ch.nprojectors; ++np) {
        raw_index.push_back(raw_offset + static_cast<std::size_t>(np));
        for (int m = -ch.angular_momentum; m <= ch.angular_momentum; ++m)
          refs.push_back({ch.angular_momentum, np, m});
      }
      raw_offset += static_cast<std::size_t>(ch.nprojectors);
    }
    const std::size_t nproj = refs.size();
    if (nproj == 0) continue;
    std::vector<Complex> beta(nproj * nbasis, Complex(0.0, 0.0));
    std::size_t p = 0;
    for (const auto& ch : cache.nonlocal_channels) {
      for (int np = 0; np < ch.nprojectors; ++np) {
        for (int m = -ch.angular_momentum; m <= ch.angular_momentum; ++m, ++p) {
          for (std::size_t ig = 0; ig < nbasis; ++ig) {
            const double q = norm(basis.cart[ig]);
            const double qmax = ch.reciprocal_cutoff > 0.0 ? ch.reciprocal_cutoff : twopi / std::max(ch.cutoff_radius, 1e-8);
            const Complex phase = std::exp(Complex(0.0, dot(basis.cart[ig], r)));
            beta[p * nbasis + ig] = std::pow(Complex(0.0, 1.0), ch.angular_momentum) *
              spherical_harmonic(ch.angular_momentum, m, basis.cart[ig]) *
              projector_value_cutoff(pots[atom],ch.reciprocal_projectors[np], q, qmax, ch.angular_momentum) /
              std::sqrt(std::abs(determinant(input.lattice_bohr)) / std::pow(bohr_per_angstrom, 3)) * phase;
          }
        }
      }
    }
    std::vector<Complex> onsite(nproj * nproj, Complex(0.0, 0.0));
    for (int band = 0; band < occupied; ++band) {
      std::vector<Complex> projection(nproj, Complex(0.0, 0.0));
      for (std::size_t x = 0; x < nproj; ++x)
        for (std::size_t ig = 0; ig < nbasis; ++ig)
          projection[x] += std::conj(beta[x * nbasis + ig]) * coeff(static_cast<Eigen::Index>(ig), band);
      for (std::size_t x = 0; x < nproj; ++x) for (std::size_t y = 0; y < nproj; ++y)
        onsite[x * nproj + y] += occupations[static_cast<std::size_t>(band)] *
                                  projection[x] * std::conj(projection[y]);
    }
    if (std::getenv("SDFT_TRACE_AUGMENTATION")) {
      double pmax = 0.0;
      for (const auto& z : onsite) pmax = std::max(pmax, std::abs(z));
      std::cerr << "augmentation atom " << atom << " nproj " << nproj << " onsite_max " << pmax << "\n";
    }
    std::map<std::tuple<int,int,int>, Vec3> unique_q;
    for (std::size_t i = 0; i < basis.g.size(); ++i) for (std::size_t j = 0; j < basis.g.size(); ++j) {
      const auto key = std::make_tuple(basis.g[i].h - basis.g[j].h,
                                       basis.g[i].k - basis.g[j].k,
                                       basis.g[i].l - basis.g[j].l);
      unique_q.emplace(key, sub(basis.g[i].cartesian, basis.g[j].cartesian));
    }
    for (const auto& [key, q] : unique_q) {
      const double qang = norm(q);
      Complex value(0.0, 0.0);
      std::size_t x = 0;
      std::size_t raw_i = 0;
      for (const auto& chi : cache.nonlocal_channels) {
        std::size_t y = 0;
        std::size_t raw_j = 0;
        for (const auto& chj : cache.nonlocal_channels) {
          if (chi.angular_momentum == chj.angular_momentum) {
            for (int ni = 0; ni < chi.nprojectors; ++ni) for (int nj = 0; nj < chj.nprojectors; ++nj)
              for (int m = -chi.angular_momentum; m <= chi.angular_momentum; ++m) {
                const std::size_t ix = x + static_cast<std::size_t>(ni * (2 * chi.angular_momentum + 1) + (m + chi.angular_momentum));
                const std::size_t iy = y + static_cast<std::size_t>(nj * (2 * chj.angular_momentum + 1) + (m + chj.angular_momentum));
                value += onsite[ix * nproj + iy] * radial_overlap_form_factor(cache, raw_i + ni, raw_j + nj, chi.angular_momentum, qang);
              }
          }
          y += static_cast<std::size_t>(chj.nprojectors * (2 * chj.angular_momentum + 1));
          raw_j += static_cast<std::size_t>(chj.nprojectors);
        }
        x += static_cast<std::size_t>(chi.nprojectors * (2 * chi.angular_momentum + 1));
        raw_i += static_cast<std::size_t>(chi.nprojectors);
      }
      out[key] += value * std::exp(Complex(0.0, -dot(q, r))) / volume_bohr3;
    }
  }
  return out;
}

std::map<std::tuple<int, int, int>, Complex> atomic_pseudo_density(
    const ScfInput& input, const std::vector<AtomPotential>& pots, const Basis& basis) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  std::map<std::tuple<int, int, int>, Complex> out;
  const double volume = std::abs(determinant(input.lattice_bohr));
  std::vector<std::tuple<int,int,int>> g;
  for(const auto& vector:basis.g)g.emplace_back(vector.h,vector.k,vector.l);
  const auto support=plane_wave_difference_support(g);
  const auto b=reciprocal(input.lattice_bohr);
  for(const auto& [key,ignored]:support) {
    (void)ignored;const auto [h,k,l]=key;
    const Vec3 q=cartesian(b,{double(h),double(k),double(l)});
    const double qang = norm(q) * bohr_per_angstrom;
    Complex value(0.0, 0.0);
    for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
      const auto& cache = pots[atom].cache;
      if (cache.atomic_pseudo_charge.empty() || cache.local_header.empty()) continue;
      // The POTCAR atomic pseudocharge is a reciprocal table on the same
      // q-grid as the local-potential and projector tables:
      // q_i = i*qmax/N, i=0..N-1.  Using
      // endpoint-inclusive interpolation here shifts every nonzero Fourier
      // component and changes the SCF fixed point even when the residual is
      // numerically small.
      const double rho = density_table_sample(cache.atomic_pseudo_charge,
                                                qang, cache.local_header.front()).value;
      const Vec3 r = cartesian(input.lattice_bohr, input.atoms[atom].fractional);
      value += rho * std::exp(Complex(0.0, -dot(q, r)));
    }
    out[key] = value / volume;
  }
  return out;
}

FourierDensity pseudo_core_density(const ScfInput& input,
    const std::vector<AtomPotential>& pots, const FourierDensity& support) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  FourierDensity out;
  const auto b = reciprocal(input.lattice_bohr);
  const double volume = std::abs(determinant(input.lattice_bohr));
  for (const auto& [key, ignored] : support) {
    (void)ignored;
    const auto [h, k, l] = key;
    const Vec3 q = cartesian(b, {static_cast<double>(h), static_cast<double>(k), static_cast<double>(l)});
    Complex value{};
    for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
      const auto& c = pots[atom].cache;
      if (c.core_charge_partial.empty()) continue;
      value += density_table_sample(c.core_charge_partial, norm(q) * bohr_per_angstrom,
          c.local_header.front()).value * std::exp(Complex(0.0, -dot(q,
              cartesian(input.lattice_bohr, input.atoms[atom].fractional))));
    }
    out[key] = value / volume;
  }
  return out;
}

double charge_residual(const std::vector<Complex>& a, const std::vector<Complex>& b,
                       const Basis& basis, double volume) {
  const auto ca = charge_coefficients(a, basis, volume);
  const auto cb = charge_coefficients(b, basis, volume);
  double residual = 0.0;
  for (const auto& [key, value] : ca) {
    auto it = cb.find(key); residual = std::max(residual, std::abs(value - (it == cb.end() ? 0.0 : it->second)));
  }
  for (const auto& [key, value] : cb) {
    auto it = ca.find(key); residual = std::max(residual, std::abs(value - (it == ca.end() ? 0.0 : it->second)));
  }
  return residual;
}

MatrixC density_potential(const std::map<std::tuple<int, int, int>, Complex>& rho_q, const Basis& basis,
                                  const std::array<Vec3, 3>& a, const std::array<Vec3, 3>& b,
                                  double& eh, double& exc, double& evxc,
                                  FourierDensity* potential = nullptr,
                                  const FourierDensity& pseudo_core = {},
                                  const DensityFunctionals* shared_functional = nullptr) {
  (void)b;
  std::optional<DensityFunctionals> evaluated;
  if(!shared_functional)evaluated=evaluate_density_functionals(rho_q,a,pseudo_core);
  const auto& functional=shared_functional ? *shared_functional : *evaluated;
  eh = functional.hartree_energy;
  exc = functional.xc_energy;
  evxc = functional.vxc_density;
  if (potential) *potential = functional.potential;
  const std::size_t n = basis.g.size();
  int mh=0,mk=0,ml=0;
  for(const auto& [key,value]:functional.potential) {
    (void)value;mh=std::max(mh,std::abs(std::get<0>(key)));
    mk=std::max(mk,std::abs(std::get<1>(key)));ml=std::max(ml,std::abs(std::get<2>(key)));
  }
  const std::size_t ny=2*mk+1,nz=2*ml+1;
  std::vector<Complex> lookup(static_cast<std::size_t>(2*mh+1)*ny*nz);
  const auto index=[&](int h,int k,int l){return (static_cast<std::size_t>(h+mh)*ny+(k+mk))*nz+(l+ml);};
  for(const auto& [key,value]:functional.potential)
    lookup[index(std::get<0>(key),std::get<1>(key),std::get<2>(key))]=value;
  MatrixC v = MatrixC::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
    const int h=basis.g[i].h-basis.g[j].h,k=basis.g[i].k-basis.g[j].k,l=basis.g[i].l-basis.g[j].l;
    if(std::abs(h)<=mh&&std::abs(k)<=mk&&std::abs(l)<=ml)v(i,j)=lookup[index(h,k,l)];
  }
  if (std::getenv("SDFT_TRACE_POTENTIAL"))
    std::cerr << "FFT density minimum " << functional.minimum_density
              << " clipped_points " << functional.clipped_density_points
              << " sampled_electrons " << functional.sampled_electrons << "\n";
  return v;
}

// Fourier coefficients of the same local ionic operator assembled in
// assemble_local().  The smooth-grid D_ij term integrates the complete
// local potential, so the PAW compensation adjoint must see V_ionic_local as well as
// the smooth Hartree/XC potential.  Keep the unit conversion identical to
// assemble_local to avoid introducing a second local-potential convention.
FourierDensity local_ionic_potential(const ScfInput& input,
    const std::vector<AtomPotential>& pots,
    const std::array<Vec3, 3>& a,
    const std::array<Vec3, 3>& b,
    const std::vector<std::tuple<int, int, int>>& keys) {
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  FourierDensity out;
  const double volume_ang3 = std::abs(determinant(a)) / std::pow(bohr_per_angstrom, 3);
  if(reciprocal_eval_batch()) {
    PerformanceTimer timer("local_ionic_batched_reciprocal");
    const ReciprocalEvaluationOptions evaluation;
    std::vector<Vec3> positions;positions.reserve(input.atoms.size());
    std::vector<std::size_t> atom_species(pots.size()),representatives;
    std::map<std::pair<std::string,double>,std::size_t> species;
    for(std::size_t atom=0;atom<pots.size();++atom) {
      positions.push_back(cartesian(a,input.atoms[atom].fractional));
      auto [found,inserted]=species.try_emplace(std::make_pair(pots[atom].numeric_sha256,pots[atom].zval),representatives.size());
      if(inserted)representatives.push_back(atom);
      atom_species[atom]=found->second;
    }
    std::vector<Complex> values(keys.size());
    std::exception_ptr failure;std::mutex failure_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel if(!omp_in_parallel() && keys.size()>=4096)
#endif
    {
    std::vector<double> radial(representatives.size());
#ifdef SDFT_HAS_OPENMP
#pragma omp for schedule(static)
#endif
    for(std::size_t ig=0;ig<keys.size();++ig) {
      try {
        const auto [h,k,l]=keys[ig];
        const Vec3 q=cartesian(b,{double(h),double(k),double(l)});const double qn=norm(q);
        // Sample identical PAW radial tables once per species and mode, then
        // sum atoms in the same order as the reference. No reduction in G.
        for(std::size_t sp=0;sp<representatives.size();++sp)
          radial[sp]=local_fourier(pots[representatives[sp]],qn,evaluation)/(volume_ang3*ev_per_hartree);
        for(std::size_t atom=0;atom<pots.size();++atom)
          values[ig]+=Complex(radial[atom_species[atom]],0)*std::exp(Complex(0,-dot(q,positions[atom])));
      }catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
    }
    }
    if(failure)std::rethrow_exception(failure);
    for(std::size_t ig=0;ig<keys.size();++ig)out.emplace_hint(out.end(),keys[ig],values[ig]);
    return out;
  }
  for (const auto& key : keys) {
    const auto [h, k, l] = key;
    const Vec3 q = cartesian(b, {static_cast<double>(h), static_cast<double>(k), static_cast<double>(l)});
    const double qn = norm(q);
    Complex value{};
    for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
      const Vec3 r = cartesian(a, input.atoms[atom].fractional);
      value += Complex(local_fourier(pots[atom], qn) / (volume_ang3 * ev_per_hartree), 0.0) *
          std::exp(Complex(0.0, -dot(q, r)));
    }
    out[key] = value;
  }
  return out;
}

struct ScfEigenResult {
  Eigen::VectorXd eigenvalues;
  MatrixC eigenvectors;
  bool iterative{false};
  bool converged{true};
  bool trial_manifold{false};
  double max_residual{0.0};
  MatrixC search_guards;
};

bool matrix_free_operator() {
  const char* mode=std::getenv("SDFT_OPERATOR");
  if(mode && std::string(mode)!="fft" && std::string(mode)!="dense")
    throw std::invalid_argument("SDFT_OPERATOR must be fft or dense");
  const char* solver=std::getenv("SDFT_SOLVER");
  const bool requires_dense=solver && (std::string(solver)=="dense"||std::string(solver)=="iterative");
  if(mode && std::string(mode)=="fft" && requires_dense)
    throw std::invalid_argument("FFT operator requires block Davidson; dense/Lanczos are reference operators");
  return mode ? std::string(mode)=="fft" : !requires_dense && !env_with_alias("SDFT_ENABLE_SPHERICAL_ONECENTER","SDFT_ENABLE_ONECENTER_CHF");
}

FourierDensity static_local_fourier(const ScfInput& input,const std::vector<AtomPotential>& pots,
                                   const Basis& basis) {
  if(std::getenv("SDFT_DISABLE_LOCAL"))return {};
  int lo[3]{basis.g[0].h,basis.g[0].k,basis.g[0].l},hi[3]{lo[0],lo[1],lo[2]};
  for(const auto& g:basis.g){const int key[]{g.h,g.k,g.l};for(int a=0;a<3;++a){lo[a]=std::min(lo[a],key[a]);hi[a]=std::max(hi[a],key[a]);}}
  // Every G-G' is in this box. Construct once per k point in O(Ngrid),
  // rather than collecting NPW^2 differences or assembling dense V.
  std::vector<std::tuple<int,int,int>> keys;
  for(int h=lo[0]-hi[0];h<=hi[0]-lo[0];++h)
    for(int k=lo[1]-hi[1];k<=hi[1]-lo[1];++k)
      for(int l=lo[2]-hi[2];l<=hi[2]-lo[2];++l)keys.emplace_back(h,k,l);
  return local_ionic_potential(input,pots,input.lattice_bohr,reciprocal(input.lattice_bohr),keys);
}

ScfEigenResult diagonalize(PlaneWaveOperator& op,int roots,const MatrixC* previous=nullptr,double accuracy_limit=1e-10,
                          const MatrixC* previous_guards=nullptr,bool allow_trial=false) {
  auto h=op.hamiltonian();const auto s=op.overlap();
  if(previous_guards && previous_guards->size()) {
    if(previous_guards->rows()!=static_cast<Eigen::Index>(h.dimension) || previous_guards->cols()>roots)
      throw std::invalid_argument("SCF search guard basis or band count changed");
    h.initial_search_guards.assign(previous_guards->data(),previous_guards->data()+previous_guards->size());
  }
  std::size_t max_iterations=256;
  if(const char* value=std::getenv("SDFT_ITERATIVE_MAX_ITERS")) {
    char* end=nullptr;const long parsed=std::strtol(value,&end,10);
    if(!end||*end!='\0'||parsed<=0)throw std::invalid_argument("invalid SDFT_ITERATIVE_MAX_ITERS");
    max_iterations=parsed;
  }
  // accuracy_limit sets the actual requested error. The larger ceiling
  // permits adaptive coarse searches; default NSCF still requests 1e-10.
  double configured=1e-3;
  if(const char* value=std::getenv("SDFT_ITERATIVE_TOL")) {
    char* end=nullptr;configured=std::strtod(value,&end);
    if(!end||*end!='\0'||!std::isfinite(configured)||configured<=0)throw std::invalid_argument("invalid SDFT_ITERATIVE_TOL");
  }
  const double tolerance=std::min(configured,std::max(5e-14,accuracy_limit));
  std::vector<Complex> initial;
  if(previous&&previous->rows()==static_cast<Eigen::Index>(h.dimension)&&previous->cols()>=roots) {
    initial.resize(h.dimension*roots);
    for(int band=0;band<roots;++band)for(std::size_t g=0;g<h.dimension;++g)initial[band*h.dimension+g]=(*previous)(g,band);
  }
  h.scf_trial_update=allow_trial&&!initial.empty()&&tolerance>1e-9;
  // A search-only operator (S6/S7) has its own discretization error. Below
  // that floor its roots fail the exact certificate and Davidson restarts
  // in FP64, so tight requests go straight to the exact operator.
  if(h.approximate_search) {
    double floor=1e-6;
    if(const char* value=std::getenv("SDFT_SEARCH_OPERATOR_FLOOR")) {
      char* end=nullptr;floor=std::strtod(value,&end);
      if(!end||*end!='\0'||!std::isfinite(floor)||floor<=0||floor>1e-2)
        throw std::invalid_argument("SDFT_SEARCH_OPERATOR_FLOOR must be in (0,1e-2]");
    }
    if(tolerance<floor){h.approximate_search=false;record_performance("search_operator_bypassed_tight_tolerance",0);}
    else record_performance("search_operator_used",0);
  }
  static std::atomic<std::size_t> solver_call{0};
  const auto call=++solver_call;
  const char* snapshot_dir=std::getenv("SDFT_SOLVER_SNAPSHOT_DIR");
  bool snapshot_saved=false;
  const auto save_snapshot=[&](const std::string& reason){
    if(!snapshot_dir||!*snapshot_dir||snapshot_saved)return;
    std::filesystem::create_directories(snapshot_dir);
    std::ostringstream stem;stem<<"call-"<<std::setw(6)<<std::setfill('0')<<call;
    const auto path=(std::filesystem::path(snapshot_dir)/(stem.str()+".pwop")).string();
    op.save_solver_snapshot(path,roots,max_iterations,tolerance,initial);
    std::string guard_path,guard_hash;
    if(!h.initial_search_guards.empty()) {
      guard_path=path+".guards";
      save_search_guards(guard_path,sha256_file(path),h.dimension,h.initial_search_guards);
      guard_hash=sha256_file(guard_path);
    }
    std::ofstream metadata(path+".json");
    metadata<<"{\"schema_version\":\"sparse-dft-solver-diagnostic-1\",\"scientific_result\":false,"
      <<"\"solver_call\":"<<call<<",\"dimension\":"<<h.dimension<<",\"roots\":"<<roots
      <<",\"max_iterations\":"<<max_iterations<<",\"tolerance\":"<<std::setprecision(17)<<tolerance
      <<",\"reason\":\""<<reason<<"\",\"snapshot_sha256\":\""<<sha256_file(path)
      <<"\",\"search_guard_sha256\":\""<<guard_hash<<"\"}\n";
    metadata.close();if(!metadata)throw std::runtime_error("solver snapshot metadata write failed");
    snapshot_saved=true;
  };
  if(const char* target=std::getenv("SDFT_SOLVER_SNAPSHOT_CALL")) {
    char* end=nullptr;const auto parsed=std::strtoull(target,&end,10);
    if(!end||end==target||*end!='\0'||!parsed)throw std::invalid_argument("invalid SDFT_SOLVER_SNAPSHOT_CALL");
    if(call==parsed)save_snapshot("requested-before-solve");
  }
  const auto solved=solve_generalized_hermitian_davidson(h,s,roots,max_iterations,tolerance,initial);
  ScfEigenResult out;out.iterative=true;out.converged=solved.converged;out.trial_manifold=solved.trial_manifold;
  out.eigenvalues=Eigen::Map<const Eigen::VectorXd>(solved.eigenvalues.data(),solved.eigenvalues.size());
  out.eigenvectors.resize(h.dimension,roots);
  for(int band=0;band<roots;++band)for(std::size_t g=0;g<h.dimension;++g)out.eigenvectors(g,band)=solved.eigenvectors[band*h.dimension+g];
  if(!solved.search_guard_vectors.empty())
    out.search_guards=Eigen::Map<const MatrixC>(solved.search_guard_vectors.data(),h.dimension,
                                              solved.search_guard_vectors.size()/h.dimension);
  for(double error:solved.residual_norms)out.max_residual=std::max(out.max_residual,error);
  if(solved.trial_manifold) {
    if(!h.scf_trial_update||solved.converged||out.max_residual>tolerance)
      throw std::runtime_error("invalid or unrequested SCF trial manifold");
  }else if(!solved.converged){save_snapshot("nonconverged");std::ostringstream reason;reason<<"matrix-free Davidson did not converge: residual="<<out.max_residual<<" tolerance="<<tolerance<<" iterations="<<solved.iterations;throw std::runtime_error(reason.str());}
  return out;
}

ScfEigenResult diagonalize(const MatrixC& h, const MatrixC& s, int requested_roots,
                          const MatrixC* previous = nullptr, double accuracy_limit=1e-10) {
  if (h.rows() != h.cols() || s.rows() != s.cols() || h.rows() != s.rows())
    throw std::invalid_argument("generalized eigenproblem matrices have inconsistent dimensions");
  const std::size_t n = static_cast<std::size_t>(h.rows());
  const std::size_t roots = std::min<std::size_t>(n, std::max(1, requested_roots));
  const char* solver_env = std::getenv("SDFT_SOLVER");
  const bool force_dense = solver_env && std::string(solver_env) == "dense";
  const bool force_iterative = solver_env && std::string(solver_env) == "iterative";
  const std::size_t dense_limit = [] {
    const char* value = std::getenv("SDFT_DENSE_LIMIT");
    if (!value) return static_cast<std::size_t>(48);
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return (end && *end == '\0' && parsed >= 1) ? static_cast<std::size_t>(parsed) : static_cast<std::size_t>(48);
  }();
  // Dense diagonalization remains available as a small-matrix/reference path.
  // Production-sized plane-wave matrices use the complex iterative solver by
  // default; SDFT_SOLVER=dense is an explicit diagnostic override.
  const bool use_iterative = force_iterative || (!force_dense && n > dense_limit);
  if (!use_iterative) {
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixC> solver(h, s);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("Sparse-DFT generalized eigenproblem failed; overlap is not positive definite");
    ScfEigenResult out;
    out.eigenvalues = solver.eigenvalues().head(static_cast<Eigen::Index>(roots));
    out.eigenvectors = solver.eigenvectors().leftCols(static_cast<Eigen::Index>(roots));
    for (std::size_t root=0; root<roots; ++root) {
      const auto v=out.eigenvectors.col(root);
      out.max_residual=std::max(out.max_residual, (h*v-out.eigenvalues[root]*(s*v)).norm());
    }
    return out;
  }
  std::vector<std::complex<double>> hf(n * n), sf(n * n);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
    hf[i * n + j] = h(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    sf[i * n + j] = s(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  }
  const std::size_t max_iterations = [] {
    const char* value = std::getenv("SDFT_ITERATIVE_MAX_ITERS");
    if (!value) return static_cast<std::size_t>(256);
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return (end && *end == '\0' && parsed > 0) ? static_cast<std::size_t>(parsed) : static_cast<std::size_t>(256);
  }();
  const double configured_tolerance = [] {
    const char* value = std::getenv("SDFT_ITERATIVE_TOL");
    if (!value) return 1e-10;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    return (end && *end == '\0' && parsed > 0.0 && std::isfinite(parsed)) ? parsed : 1e-10;
  }();
  // A strict coupled SCF cannot resolve a density below the eigensolver
  // noise floor. Tighten the inner solve with its outer request, retaining
  // a machine-precision floor so impossible SCF tolerances report unconverged.
  const double tolerance=std::min(configured_tolerance,std::max(5e-14,accuracy_limit));
  // Lanczos factors the full overlap; block Davidson avoids that cubic
  // factorization. Keep the former only as an explicit reference option.
  const bool use_davidson = !solver_env || std::string(solver_env) != "iterative";
  std::vector<Complex> initial;
  if(use_davidson && previous && previous->rows()==h.rows() && previous->cols()>=static_cast<Eigen::Index>(roots)) {
    initial.resize(n*roots);
    for(std::size_t root=0;root<roots;++root)for(std::size_t i=0;i<n;++i)
      initial[root*n+i]=(*previous)(i,root);
  }
  const auto iterative = use_davidson
      ? solve_generalized_hermitian_davidson(hf, sf, n, roots, max_iterations, tolerance, initial)
      : solve_generalized_hermitian_iterative(hf, sf, n, roots, max_iterations, tolerance);
  if (iterative.eigenvalues.size() != roots || iterative.eigenvectors.size() != roots * n)
    throw std::runtime_error("complex iterative eigensolver returned an incomplete spectrum");
  ScfEigenResult out;
  out.iterative = true;
  out.converged = iterative.converged;
  out.eigenvalues.resize(static_cast<Eigen::Index>(roots));
  out.eigenvectors = MatrixC::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(roots));
  for (std::size_t root = 0; root < roots; ++root) {
    out.eigenvalues(static_cast<Eigen::Index>(root)) = iterative.eigenvalues[root];
    for (std::size_t component = 0; component < n; ++component)
      out.eigenvectors(static_cast<Eigen::Index>(component), static_cast<Eigen::Index>(root)) = iterative.eigenvectors[root * n + component];
    // The transformed Lanczos problem is orthonormal in the Cholesky
    // coordinates, but converting back through L^{-H} can accumulate a
    // noticeable norm error for the ill-conditioned PAW overlap.  SCF
    // occupations and on-site occupations are defined from the original generalized
    // eigenvectors, so enforce the S metric here before constructing either
    // the plane-wave density or the one-centre density.  A residual check
    // alone does not catch this error: an eigenvector with the wrong norm is
    // still an eigenvector of Hx=epsilon Sx.
    const auto vector = out.eigenvectors.col(static_cast<Eigen::Index>(root));
    const std::complex<double> metric = vector.adjoint() * s * vector;
    const double metric_norm = std::sqrt(std::max(0.0, metric.real()));
    if (!(metric_norm > 1e-14) || !std::isfinite(metric_norm))
      throw std::runtime_error("iterative generalized eigenvector has invalid S norm");
    out.eigenvectors.col(static_cast<Eigen::Index>(root)) /= metric_norm;
    if (root < iterative.residual_norms.size()) out.max_residual = std::max(out.max_residual, iterative.residual_norms[root]);
  }
  if (!out.converged)
    {
      std::ostringstream reason;
      reason<<"complex iterative eigensolver did not converge: residual="<<out.max_residual
            <<" tolerance="<<tolerance<<" iterations="<<iterative.iterations;
      throw std::runtime_error(reason.str());
    }
  return out;
}

MatrixC make_hamiltonian(const ScfInput& input, const std::vector<AtomPotential>& pots,
                                 const Basis& basis, const std::array<Vec3, 3>& a,
                                 const std::map<std::tuple<int, int, int>, Complex>& rho_q,
                                 const MatrixC* density_matrix,
                                 MatrixC& overlap,
                                 double& eh, double& exc, double& evxc,
                                 double* paw_spherical_dij_norm_ev = nullptr,
                                 double* paw_dc_ae_ev = nullptr,
                                 double* paw_dc_ps_ev = nullptr,
                                 double* paw_onecenter_energy_ev = nullptr,
                                 double* paw_onecenter_potential_expectation_ev = nullptr,
                                 double* paw_onecenter_core_xc_ev = nullptr,
                                 const std::vector<ScfOneCenterState>* frozen_onecenter = nullptr,
                                 std::vector<ScfOneCenterState>* captured_onecenter = nullptr,
                                 const std::vector<std::vector<Complex>>* onsite_input = nullptr,
                                 const PawScfClosureContext* closure_context = nullptr,
                                 const DensityFunctionals* shared_functional = nullptr,
                                 const MatrixC* static_operator = nullptr,
                                 std::unique_ptr<PlaneWaveOperator>* matrix_free = nullptr,
                                 const FourierDensity* static_fourier = nullptr,
                                 const std::vector<MatrixC>* cached_projectors = nullptr,
                                 const FourierDensity* cached_full_ionic = nullptr,
                                 std::uint64_t immutable_projector_identity = 0) {
  const std::size_t n = basis.g.size();
  MatrixC h;
  if(!matrix_free) {
    h = static_operator ? *static_operator : MatrixC::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    overlap = MatrixC::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  }
  const auto b = reciprocal(a);
  if(!matrix_free && !static_operator) {
    for (std::size_t i = 0; i < n; ++i) h(i,i) = 0.5 * dot(basis.cart[i], basis.cart[i]);
    if (!std::getenv("SDFT_DISABLE_LOCAL")) assemble_local(input, pots, basis, b, a, h);
  }
  eh = exc = evxc = 0.0;
  const bool frozen_grid_unused = frozen_onecenter && frozen_nscf_fields();
  FourierDensity total_local_potential;
  if (!rho_q.empty() && !std::getenv("SDFT_DISABLE_HARTREE_XC")) {
    FourierDensity density_local_potential;
    if(matrix_free) {
      const auto evaluated=shared_functional ? DensityFunctionals{} : evaluate_density_functionals(rho_q,a,
          closure_context ? closure_context->pseudo_core : empty_pseudo_core());
      const auto& functional=shared_functional ? *shared_functional : evaluated;
      eh=functional.hartree_energy;exc=functional.xc_energy;evxc=functional.vxc_density;
      if(!frozen_grid_unused)density_local_potential=functional.potential;
    } else h += density_potential(rho_q, basis, a, b, eh, exc, evxc,
          frozen_grid_unused ? nullptr : &density_local_potential,
          closure_context ? closure_context->pseudo_core : empty_pseudo_core(), shared_functional);
    total_local_potential = std::move(density_local_potential);
  }
  if (!std::getenv("SDFT_DISABLE_LOCAL") && closure_context && !total_local_potential.empty()) {
    std::vector<std::tuple<int, int, int>> keys;
    keys.reserve(total_local_potential.size());
    for (const auto& [key, value] : total_local_potential) { (void)value; keys.push_back(key); }
    const auto computed_ionic = cached_full_ionic ? FourierDensity{} : local_ionic_potential(input, pots, a, b, keys);
    const auto& ionic=cached_full_ionic ? *cached_full_ionic : computed_ionic;
    for (const auto& [key, value] : ionic) total_local_potential[key] += value;
  }
  if(frozen_grid_unused)record_performance("nscf_frozen_dgrid_field_skipped",0);
  std::vector<ComplexPawOperator> blocks;
  if (!std::getenv("SDFT_DISABLE_NONLOCAL")) assemble_nonlocal(input, pots, basis, a, h, overlap, density_matrix, paw_spherical_dij_norm_ev, paw_dc_ae_ev, paw_dc_ps_ev, paw_onecenter_energy_ev, paw_onecenter_potential_expectation_ev, paw_onecenter_core_xc_ev, frozen_onecenter, captured_onecenter, onsite_input, closure_context, &total_local_potential, matrix_free ? &blocks : nullptr,cached_projectors);
  if(matrix_free) {
    FourierDensity local=static_fourier ? *static_fourier : static_local_fourier(input,pots,basis);
    if(!std::getenv("SDFT_DISABLE_HARTREE_XC")) {
      const auto evaluated=shared_functional ? DensityFunctionals{} : evaluate_density_functionals(rho_q,a,
          closure_context ? closure_context->pseudo_core : empty_pseudo_core());
      const auto& functional=shared_functional ? *shared_functional : evaluated;
      // Frequencies outside this G-G' box have exactly zero matrix elements
      // in the wavefunction subspace. Retain them for XC and Dgrid above,
      // but do not enlarge every Davidson FFT with the fine PAW grid.
      int width[3]{};
      for(const auto& [key,value]:local) {
        (void)value;
        const int index[]{std::get<0>(key),std::get<1>(key),std::get<2>(key)};
        for(int axis=0;axis<3;++axis)width[axis]=std::max(width[axis],std::abs(index[axis]));
      }
      // With the ionic operator disabled its support is empty; derive the
      // same bounding box directly from the basis in that case.
      if(local.empty()) {
        int lo[3]{basis.g[0].h,basis.g[0].k,basis.g[0].l},hi[3]{lo[0],lo[1],lo[2]};
        for(const auto& g:basis.g) {
          const int index[]{g.h,g.k,g.l};
          for(int axis=0;axis<3;++axis){lo[axis]=std::min(lo[axis],index[axis]);hi[axis]=std::max(hi[axis],index[axis]);}
        }
        for(int axis=0;axis<3;++axis)width[axis]=hi[axis]-lo[axis];
      }
      for(const auto& [key,value]:functional.potential)
        if(std::abs(std::get<0>(key))<=width[0]&&std::abs(std::get<1>(key))<=width[1]&&
           std::abs(std::get<2>(key))<=width[2])local[key]+=value;
    }
    *matrix_free=std::make_unique<PlaneWaveOperator>(basis.g,local,std::move(blocks),atomic_units(input.unit_convention).ev_per_hartree,
        1e-12,cached_projectors?immutable_projector_identity:0);
  } else overlap += 1e-12 * MatrixC::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  return h;
}

}  // namespace

std::shared_ptr<PawSiteGeometry> prepare_potcar_site_geometry(
    const ScfInput& original,std::size_t atom,const std::vector<std::tuple<int,int,int>>& keys,
    const std::vector<Vec3>& vectors,const std::vector<std::vector<Vec3>>& k_vectors) {
  if(keys.size()!=vectors.size()||atom>=original.atoms.size()||k_vectors.empty())
    throw std::invalid_argument("POTCAR site geometry dimensions invalid");
  ScfInput one=original;one.atoms={original.atoms[atom]};one.atoms[0].fractional={};
  if((one.projector_interpolation=="upf_original_radial_quadrature"||one.projector_interpolation=="per_site_original_paw"))one.projector_interpolation="cubic";
  auto values=read_potentials(one);
  auto pot=std::make_shared<AtomPotential>(std::move(values[0]));
  const auto units=atomic_units(one.unit_convention);const double ev=units.ev_per_hartree;
  const double volume=std::abs(determinant(one.lattice_bohr));
  const double volume_ang=volume/std::pow(units.bohr_per_angstrom,3);
  auto out=std::make_shared<PawSiteGeometry>();out->framework=PawFramework::Potcar;
  out->element=one.atoms[0].symbol;out->record_id=pot->record_id;out->zval=pot->zval;
  out->numeric_cache_version=pot->cache.version;
  for(const auto& ch:pot->cache.nonlocal_channels)for(int j=0;j<ch.nprojectors;++j)
    out->radial_channel_l.push_back(ch.angular_momentum);
  out->atomic_onsite=paw_atomic_reference_density(pot->cache,out->radial_channel_l);
  auto augmentation=std::make_shared<PawAugmentationData>(build_paw_augmentation(pot->cache,out->radial_channel_l));
  auto mode=PawRadialPotentialMode::AsphericalAdjoint;
  if(const char* setting=std::getenv("SDFT_PAW_XC_MODE")) {
    const std::string text=canonical_paw_xc_mode(setting);
    if(text=="radial")mode=PawRadialPotentialMode::RadialDirect;
    else if(text=="aspherical")mode=PawRadialPotentialMode::AsphericalDirect;
    else if(text!="aspherical_adjoint")throw std::invalid_argument("shared SDFT_PAW_XC_MODE must be aspherical_adjoint, aspherical or radial");
  }
  // Keep the original signed atomic linear reference and core-XC ledger.
  // Only the current energy response uses the exact discrete adjoint.
  const auto reference_mode=mode==PawRadialPotentialMode::AsphericalAdjoint ?
      PawRadialPotentialMode::AsphericalDirect : mode;
  auto reference=std::make_shared<PawOneCenterReferencePotentials>(paw_atomic_reference_potentials(
      pot->cache,out->radial_channel_l,PawRadialLengthUnit::Angstrom,reference_mode,one.unit_convention));
  const auto ls=out->radial_channel_l;const auto constants=one.unit_convention;
  const auto dimension=out->atomic_onsite.size();const auto n=std::size_t(std::llround(std::sqrt(double(dimension))));
  if(n*n!=dimension)throw std::invalid_argument("POTCAR atomic density is not square");
  out->closure=[pot,reference,ls,n,mode,constants,ev](const std::array<std::vector<Complex>,2>& p,int spins) {
    PawSiteClosure result;
    std::array<PawOneCenterClosureResult,2> closure;
    if(spins==2)closure=paw_collinear_one_center_closure(pot->cache,ls,p,n,
        PawRadialLengthUnit::Angstrom,reference.get(),mode,constants);
    else if(spins==1)closure[0]=paw_one_center_closure(pot->cache,ls,p[0],n,1,
        PawRadialLengthUnit::Angstrom,reference.get(),mode,constants);
    else throw std::invalid_argument("POTCAR closure requires one or two spin channels");
    for(int spin=0;spin<spins;++spin) {
      result.energy_hartree+=(closure[spin].energy_difference_ev-closure[spin].core_xc_energy_ev)/ev;
      result.derivative_hartree[spin]=std::move(closure[spin].d_matrix_ev);
      for(auto& z:result.derivative_hartree[spin])z/=ev;
    }
    return result;
  };
  out->hubbard_overlap=[pot,ls](int l){return paw_hubbard_ae_overlap(pot->cache,ls,l);};
  if(!pot->cache.atomic_reference_present)throw std::invalid_argument("POTCAR EATOM ledger missing");
  const double error=pot->kinetic_error_spline.empty()?0.:pot->kinetic_error_spline.value(one.cutoff_ev);
  out->reference_energy_hartree=(pot->cache.atomic_reference_energy_ev-error)/ev;
  out->excluded_local_g0_form_hartree=pot->cache.local_potential.empty()?0.:
      pot->cache.local_potential.front()*std::pow(units.bohr_per_angstrom,3)/ev;
  bool compact=true;
  if(const char* setting=std::getenv("SDFT_SHARED_COMPACT_AUGMENTATION")) {
    const std::string text(setting);
    if(text!="0"&&text!="1")throw std::invalid_argument("SDFT_SHARED_COMPACT_AUGMENTATION must be 0 or 1");
    compact=text=="1";
  }
  auto compensation=std::make_shared<PawCompensationOperator>(*augmentation,one.lattice_bohr,Vec3{},keys,one.unit_convention);
  if(compact)out->compact_compensation=[augmentation,compensation,lattice=one.lattice_bohr,keys,constants](Vec3 position) {
    return std::make_shared<PawCompensationOperator>(*augmentation,lattice,position,keys,constants,compensation.get());
  };
  out->local.resize(keys.size());out->local_derivative.resize(keys.size());out->core.resize(keys.size());
  out->core_derivative.resize(keys.size());out->atomic_density.resize(keys.size());
  if(!compact){out->qform.resize(keys.size());out->qgradient.resize(keys.size());}
  for(std::size_t g=0;g<keys.size();++g) {
    const double q=norm(vectors[g]),qang=q*units.bohr_per_angstrom;
    out->local[g]=local_fourier(*pot,q)*std::pow(units.bohr_per_angstrom,3)/ev;
    out->local_derivative[g]=local_fourier_derivative(*pot,q)*std::pow(units.bohr_per_angstrom,3)/ev;
    const auto core=density_table_sample(pot->cache.core_charge_partial,qang,pot->cache.local_header.front());
    out->core[g]=core.value;out->core_derivative[g]=core.derivative*units.bohr_per_angstrom;
    out->atomic_density[g]=density_table_sample(pot->cache.atomic_pseudo_charge,qang,pot->cache.local_header.front()).value;
    if(!compact) {
      out->qform[g]=compensation->reciprocal_matrix(g);
      for(int axis=0;axis<3;++axis)out->qgradient[g][axis]=compensation->reciprocal_matrix(g,axis);
    }
    // The POTCAR atomic reciprocal charge is already the initial total
    // pseudocharge of the reference-atom occupation. Adding its compensation
    // charge here counts compensation twice and can erase a magnetic seed
    // before the first SCF update.
    // The ordinary SCF output still adds compensation to the wave density.
  }
  for(const auto& k:k_vectors) {
    Basis basis;basis.cart=k;basis.g.resize(k.size());MatrixC h,s;
    PawScfClosureContext context;context.augmentation.push_back(*augmentation);
    std::vector<ComplexPawOperator> blocks;
    // Preserve the original assembly and projector conventions.
    auto reread=read_potentials(one);
    assemble_nonlocal(one,reread,basis,one.lattice_bohr,h,s,nullptr,nullptr,nullptr,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&context,nullptr,&blocks);
    if(blocks.size()!=1||blocks[0].projectors!=n)throw std::invalid_argument("POTCAR shared site projector dimension mismatch");
    UpfBetaGeometry beta;beta.dimension=n;beta.npw=k.size();beta.amplitudes=blocks[0].amplitudes;
    for(auto& gradient:beta.gradients)gradient.resize(n*k.size());
    const double factor=(std::getenv("SDFT_PROJECTOR_SCALE")?std::atof(std::getenv("SDFT_PROJECTOR_SCALE")):1.)/std::sqrt(volume_ang);
    const double conversion=std::getenv("SDFT_PROJECTOR_Q_BOHR")?1.:units.bohr_per_angstrom;
    std::size_t row=0;
    for(const auto& ch:pot->cache.nonlocal_channels)for(int p=0;p<ch.nprojectors;++p)
      for(int m=-ch.angular_momentum;m<=ch.angular_momentum;++m,++row)for(std::size_t g=0;g<k.size();++g) {
        const auto v=k[g];const double q=norm(v);
        const double qmax=ch.reciprocal_cutoff>0?ch.reciprocal_cutoff:twopi/std::max(ch.cutoff_radius,1e-8);
        const auto sample=projector_sample(*pot,ch.reciprocal_projectors[p],q*conversion,qmax,ch.angular_momentum);
        const auto phase=factor*std::pow(Complex(0,1),ch.angular_momentum);
        if(q==0) {
          if(ch.angular_momentum==1)for(int axis=0;axis<3;++axis) {
            const auto direction=axis==0?Vec3{1,0,0}:axis==1?Vec3{0,1,0}:Vec3{0,0,1};
            beta.gradients[axis][row*k.size()+g]=phase*sample.derivative*conversion*spherical_harmonic(1,m,direction);
          }
        } else {
          const auto dy=paw_real_spherical_harmonic_gradient(ch.angular_momentum,m,v);
          for(int axis=0;axis<3;++axis)beta.gradients[axis][row*k.size()+g]=phase*(sample.value*upf_component(dy,axis)+
              sample.derivative*conversion*upf_component(v,axis)/q*spherical_harmonic(ch.angular_momentum,m,v));
        }
      }
    out->bare_at_origin.push_back(std::move(blocks[0]));out->beta_geometry.push_back(std::move(beta));
  }
  return out;
}

ComplexPawOperator prepare_potcar_bloch_block(const ScfInput& input,std::size_t atom,
    const std::vector<Vec3>& vectors) {
  ScfInput one=input;one.atoms={input.atoms.at(atom)};
  if(one.projector_interpolation=="per_site_original_paw"||one.projector_interpolation=="upf_original_radial_quadrature")
    one.projector_interpolation="cubic";
  const auto pots=read_potentials(one);std::vector<int> ls;
  for(const auto& ch:pots[0].cache.nonlocal_channels)for(int p=0;p<ch.nprojectors;++p)ls.push_back(ch.angular_momentum);
  PawScfClosureContext context;context.augmentation.push_back(build_paw_augmentation(pots[0].cache,ls));
  Basis basis;basis.cart=vectors;basis.g.resize(vectors.size());MatrixC h,s;
  std::vector<ComplexPawOperator> blocks;
  assemble_nonlocal(one,pots,basis,one.lattice_bohr,h,s,nullptr,nullptr,nullptr,nullptr,
      nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&context,nullptr,&blocks);
  if(blocks.size()!=1)throw std::invalid_argument("POTCAR Bloch block atom count mismatch");
  return std::move(blocks[0]);
}

static ScfResult run_scf_attempt(const ScfInput& input) {
  if(use_shared_site_kernel(input))return run_upf_scf(input);
  PerformanceTimer total_timer("scf_total");
  if(input.spin_channels!=1&&input.spin_channels!=2)throw std::invalid_argument("spin_channels must be 1 or 2");
  const bool polarized=input.spin_channels==2;
  if(polarized&&(std::getenv("SDFT_DISABLE_HARTREE_XC")||std::getenv("SDFT_DISABLE_DGRID")||std::getenv("SDFT_DISABLE_NONLOCAL")||std::getenv("SDFT_DISABLE_OVERLAP")))
    throw std::invalid_argument("spin SCF requires its complete Hartree/XC/PAW operators");
  for(const auto& atom:input.atoms){
    if(!std::isfinite(atom.initial_magnetic_moment)||std::abs(atom.initial_magnetic_moment)>atom.zval)
      throw std::invalid_argument("initial moment exceeds the valence count");
    if(!std::isfinite(atom.hubbard.ueff_ev)||atom.hubbard.ueff_ev<0||atom.hubbard.angular_momentum>3||atom.hubbard.angular_momentum< -1||(atom.hubbard.angular_momentum==-1&&atom.hubbard.ueff_ev!=0))
      throw std::invalid_argument("invalid Dudarev Ueff/l");
    if(!polarized&&(atom.initial_magnetic_moment!=0||atom.hubbard.ueff_ev!=0))
      throw std::invalid_argument("magnetic moments and Hubbard U currently require spin_channels=2");
  }
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  if (input.atoms.empty()) throw std::invalid_argument("SCF requires at least one atom");
  if(input.max_iterations<1 || !std::isfinite(input.cutoff_ev) || input.cutoff_ev<=0 ||
     !std::isfinite(input.mixing) || input.mixing<=0 || input.mixing>1 ||
     !std::isfinite(input.density_tolerance) || input.density_tolerance<=0 ||
     !std::isfinite(input.smearing_ev) || input.smearing_ev<0 || input.bands<0)
    throw std::invalid_argument("invalid SCF cutoff, iterations, mixing, tolerance, bands or smearing");
  for(const auto& atom:input.atoms)
    if(!std::isfinite(atom.zval)||atom.zval<=0||!std::isfinite(atom.fractional.x)||
       !std::isfinite(atom.fractional.y)||!std::isfinite(atom.fractional.z))
      throw std::invalid_argument("SCF requires positive finite valence charges and finite positions");
  if (!std::isfinite(input.onecenter_density_tolerance) || input.onecenter_density_tolerance <= 0.0)
    throw std::invalid_argument("onecenter_density_tolerance must be finite and positive");
  if(!std::isfinite(input.density_cutoff_ev)||input.density_cutoff_ev<0||
     (input.density_cutoff_ev>0&&input.density_cutoff_ev<4*input.cutoff_ev))
    throw std::invalid_argument("density_cutoff_ev must be zero (automatic) or at least 4*cutoff_ev");
  const auto b = reciprocal(input.lattice_bohr);
  auto kpoints = input.kpoints;
  if (kpoints.empty()) kpoints.push_back({input.kpoint_fractional, 1.0});
  double weight_sum = 0.0;
  for (const auto& point : kpoints) {
    if (!std::isfinite(point.weight) || point.weight <= 0.0 ||
        !std::isfinite(point.fractional.x) || !std::isfinite(point.fractional.y) ||
        !std::isfinite(point.fractional.z))
      throw std::invalid_argument("SCF requires finite k-points and positive weights");
    weight_sum += point.weight;
  }
  if (!std::isfinite(weight_sum)) throw std::invalid_argument("SCF k-point weight overflow");
  for (auto& point : kpoints) point.weight /= weight_sum;
  const std::size_t physical_nk=kpoints.size(),natoms=input.atoms.size();
  const auto physical_kpoints=kpoints;
  if(polarized)kpoints.insert(kpoints.end(),physical_kpoints.begin(),physical_kpoints.end());
  std::vector<Basis> bases;
  for (const auto& point : kpoints) bases.push_back(make_basis(input, b, point.fractional));
  const auto& basis = bases.front();
  const auto pots = read_potentials(input);
  const double electrons = std::accumulate(input.atoms.begin(), input.atoms.end(), 0.0, [](double x, const ScfAtom& a) { return x + a.zval; });
  const int occupied = static_cast<int>(std::ceil(electrons / 2.0));
  double absolute_initial_moment=0.;for(const auto& atom:input.atoms)absolute_initial_moment+=std::abs(atom.initial_magnetic_moment);
  const int spin_capacity=polarized?int(std::ceil((electrons+absolute_initial_moment)/2.)):occupied;
  const int bands = input.bands > 0 ? std::max(input.bands, spin_capacity) : spin_capacity + (polarized?8:4);
  for (const auto& kbasis : bases)
    if (kbasis.g.size() < static_cast<std::size_t>(bands))
      throw std::invalid_argument("plane-wave cutoff yields fewer basis states than requested bands; increase cutoff_ev or reduce bands");
  const std::size_t n = basis.g.size();
  MatrixC rho;
  // One reciprocal support shared by all k-points. The atomic starting
  // density is k-independent; merge its keys, never add it once per k.
  FourierDensity density_q;
  for (const auto& kbasis : bases)
    for (const auto& [key, value] : atomic_pseudo_density(input, pots, kbasis))
      density_q[key] = value;
  const FourierDensity smooth_density_support=density_q;
  std::vector<std::vector<double>> occupations(kpoints.size(), std::vector<double>(bands));
  ScfResult result;
  result.spin_channels=input.spin_channels;
  for(const auto& atom:input.atoms)result.hubbard_specs.push_back(atom.hubbard);
  result.projector_interpolation=input.projector_interpolation;
  result.cutoff_ev = input.cutoff_ev;
  result.onecenter_closure_diagnostic = paw_closure_enabled();
  if(polarized&&!result.onecenter_closure_diagnostic)throw std::invalid_argument("spin requires complete PAW closure");
  if (result.onecenter_closure_diagnostic &&
      (env_with_alias("SDFT_ENABLE_SPHERICAL_ONECENTER","SDFT_ENABLE_ONECENTER_CHF") || std::getenv("SDFT_DISABLE_NONLOCAL")))
    throw std::invalid_argument("PAW closure diagnostic requires its nonlocal operator and cannot be combined with the legacy spherical one-centre branch");
  if(result.onecenter_closure_diagnostic) {
    result.density_cutoff_ev=input.density_cutoff_ev>0?input.density_cutoff_ev:16*input.cutoff_ev;
    // The support of |psi|^2 is only the smooth charge support. Localized
    // PAW compensation and frozen-core densities have nonzero frequencies
    // beyond it. Truncating them at G-G' corrupts Dgrid and stationary stress
    // even when their omitted spectral power is tiny (MgO regression).
    const auto selection_b = input.basis_reference_lattice_bohr
        ? reciprocal(*input.basis_reference_lattice_bohr) : b;
    for(const auto& g:enumerate_g_vectors(selection_b,result.density_cutoff_ev,{},ev_per_hartree))
      density_q.try_emplace({g.h,g.k,g.l},Complex{});
  }
  result.density_reciprocal_modes=density_q.size();
  std::vector<std::tuple<int,int,int>> support_indices;
  for (const auto& [key, value] : density_q) { (void)value; support_indices.push_back(key); }
  result.density_support_index_sha256 = reciprocal_index_sha256(std::move(support_indices));
  result.basis_reference_lattice_bohr = input.basis_reference_lattice_bohr;
  // Neither the legacy operator nor the new closure has passed the absolute
  // VASP energy and derivative gates yet. Convergence alone is insufficient.
  result.total_energy_valid = false;
  result.onecenter_density_tolerance = input.onecenter_density_tolerance;
  if(const char* setting=std::getenv("SDFT_SCF_ACCEPTANCE_MARGIN")){
    std::size_t end=0;const std::string value(setting);const double margin=std::stod(value,&end);
    if(end!=value.size()||!std::isfinite(margin)||margin<=0||margin>1.)
      throw std::invalid_argument("SDFT_SCF_ACCEPTANCE_MARGIN must be finite and in (0,1]");
    result.scf_acceptance_margin=margin;
  }
  result.effective_density_tolerance=input.density_tolerance*result.scf_acceptance_margin;
  result.effective_onecenter_density_tolerance=input.onecenter_density_tolerance*result.scf_acceptance_margin;
  result.onecenter_mixing_method = result.onecenter_closure_diagnostic ?
      "shared_damped_pulay_coefficients_and_linear_mixing" : "inactive";
  result.onecenter_lattice_bohr = input.lattice_bohr;
  result.unit_convention = input.unit_convention;
  result.g_vectors = basis.g;
  result.kpoints = kpoints;
  for(std::size_t k=0;k<kpoints.size();++k)result.kpoint_spin_channels.push_back(polarized?int(k/physical_nk):0);
  result.plane_waves = n;
  for (const auto& kbasis : bases) {
    result.plane_waves_per_kpoint.push_back(kbasis.g.size());
    // Count equality alone cannot prove a fixed basis under strain.
    std::vector<std::tuple<int,int,int>> indices;
    for(const auto& g:kbasis.g)indices.emplace_back(g.h,g.k,g.l);
    result.plane_wave_index_sha256.push_back(reciprocal_index_sha256(std::move(indices)));
  }
  const bool use_fft=matrix_free_operator();
  result.solver_method = use_fft ? "matrix_free_fft_block_davidson" :
      (std::getenv("SDFT_SOLVER") ? std::getenv("SDFT_SOLVER") : "block_davidson_above_dense_limit");
  if (kpoints.size() > 1 && env_with_alias("SDFT_ENABLE_SPHERICAL_ONECENTER","SDFT_ENABLE_ONECENTER_CHF"))
    throw std::invalid_argument("multi-k SCF uses the complex PAW closure, not the legacy spherical one-centre branch");
  result.occupied_bands = occupied;
  result.bands = bands;
  result.electron_count = electrons;
  using DensityKey = std::tuple<int, int, int>;
  std::vector<DensityKey> density_keys;
  for (const auto& [key, value] : density_q) { (void)value; density_keys.push_back(key); }
  std::vector<std::vector<Complex>> pulay_residuals;
  std::vector<std::vector<Complex>> pulay_outputs;
  std::vector<std::vector<Complex>> onsite_pulay_residuals;
  std::vector<std::vector<Complex>> onsite_pulay_outputs;
  std::vector<std::vector<MatrixC>> onsite_projectors(kpoints.size());
  std::vector<std::uint64_t> immutable_projector_identities(kpoints.size());
  if(const char* setting=std::getenv("SDFT_SCF_PROJECTOR_RESIDENCY")) {
    if(std::string(setting)!="0"&&std::string(setting)!="1")
      throw std::invalid_argument("SDFT_SCF_PROJECTOR_RESIDENCY must be 0 or 1");
    if(std::string(setting)=="1"&&use_fft&&result.onecenter_closure_diagnostic)
      for(auto& identity:immutable_projector_identities)identity=new_immutable_matrix_identity();
  }
  const FourierDensity cached_full_ionic=local_ionic_potential(input,pots,input.lattice_bohr,b,density_keys);
  std::vector<std::vector<Complex>> onsite_input;
  std::optional<PawScfClosureContext> closure_context;
  if (result.onecenter_closure_diagnostic) {
    PerformanceTimer closure_timer("scf_closure_context_initialization");
    closure_context.emplace();
    result.onecenter_radial_mode="aspherical_gga";
    if(const char* mode=std::getenv("SDFT_PAW_XC_MODE")){
      const std::string value=canonical_paw_xc_mode(mode);
      if(value=="radial"){
        closure_context->potential_mode=PawRadialPotentialMode::RadialDirect;
        result.onecenter_radial_mode="radial_gga";
      }else if(value!="aspherical")throw std::invalid_argument("SDFT_PAW_XC_MODE must be aspherical or radial");
    }
    closure_context->pseudo_core = pseudo_core_density(input, pots, density_q);
    result.frozen_core_q = closure_context->pseudo_core;
    closure_context->compensation.reserve(input.atoms.size());
    for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
      for (std::size_t k = 0; k < kpoints.size(); ++k)
        onsite_projectors[k].push_back(atom_projector_basis(input, pots[atom], atom, bases[k], input.lattice_bohr));
      std::vector<int> raw_l;
      for (const auto& channel : pots[atom].cache.nonlocal_channels)
        for (int projector = 0; projector < channel.nprojectors; ++projector)
          raw_l.push_back(channel.angular_momentum);
      onsite_input.push_back(paw_atomic_reference_density(pots[atom].cache, raw_l));
      std::size_t previous=atom;
      if(share_compensation_geometry())for(std::size_t i=0;i<atom;++i)
        if(pots[i].numeric_sha256==pots[atom].numeric_sha256){previous=i;break;}
      if(previous<atom) {
        closure_context->references.push_back(closure_context->references[previous]);
        closure_context->augmentation.push_back(closure_context->augmentation[previous]);
      } else {
        closure_context->references.push_back(paw_atomic_reference_potentials(
            pots[atom].cache, raw_l, PawRadialLengthUnit::Angstrom,
            closure_context->potential_mode, input.unit_convention));
        closure_context->augmentation.push_back(build_paw_augmentation(pots[atom].cache, raw_l));
      }
      closure_context->compensation.emplace_back(closure_context->augmentation.back(),
          input.lattice_bohr, input.atoms[atom].fractional, density_keys, input.unit_convention,
          previous<atom?&closure_context->compensation[previous]:nullptr);
      const auto& op=closure_context->compensation.back();
      if(previous==atom){++result.compensation_geometry_groups;result.compensation_geometry_bytes+=op.geometry_storage_bytes();}
      result.compensation_atom_bytes+=op.atom_storage_bytes();
      result.compensation_unshared_equivalent_bytes+=op.geometry_storage_bytes()+op.atom_storage_bytes();
    }
  }
  std::array<FourierDensity,2> spin_q;
  std::array<std::vector<std::vector<Complex>>,2> spin_onsite;
  if(polarized){
    const auto atomic_comp=PawCompensationOperator::density_sum(closure_context->compensation,onsite_input);
    for(int channel=0;channel<2;++channel){spin_q[channel]=density_q;
      for(auto& [key,value]:spin_q[channel])value*=.5;
      spin_onsite[channel]=onsite_input;
      for(std::size_t atom=0;atom<natoms;++atom){const double fraction=.5*(1+(channel==0?1.:-1.)*input.atoms[atom].initial_magnetic_moment/input.atoms[atom].zval);
        for(auto& value:spin_onsite[channel][atom])value*=fraction;}
    }
    // Atomic spin differences include smooth and compensation parts at the
    // same Fourier support; AFM seeds retain their spatial magnetization.
    for(std::size_t atom=0;atom<natoms;++atom){
      ScfInput one=input;one.atoms={input.atoms[atom]};std::vector<AtomPotential> one_pot;
      auto selected=read_potentials(one);
      auto smooth=atomic_pseudo_density(one,selected,basis);
      const double fraction=input.atoms[atom].initial_magnetic_moment/input.atoms[atom].zval;
      for(const auto& [key,value]:smooth){spin_q[0][key]+=.5*fraction*value;spin_q[1][key]-=.5*fraction*value;}
    }
    const auto cu=PawCompensationOperator::density_sum(closure_context->compensation,spin_onsite[0]);
    const auto cd=PawCompensationOperator::density_sum(closure_context->compensation,spin_onsite[1]);
    for(const auto& [key,value]:atomic_comp){spin_q[0][key]-=.5*value;spin_q[1][key]-=.5*value;}
    for(const auto& [key,value]:cu)spin_q[0][key]+=value;
    for(const auto& [key,value]:cd)spin_q[1][key]+=value;
    onsite_input=spin_onsite[0];onsite_input.insert(onsite_input.end(),spin_onsite[1].begin(),spin_onsite[1].end());
    density_q=spin_q[0];for(const auto& [key,value]:spin_q[1])density_q[key]+=value;
    // The atomic reference Fourier seed can have a quadrature normalization
    // error. Charge is constrained; the specified initial total moment is not.
    double moment=0;for(const auto& atom:input.atoms)moment+=atom.initial_magnetic_moment;
    const double volume=std::abs(determinant(input.lattice_bohr));
    spin_q[0][{0,0,0}]=(electrons+moment)/(2*volume);spin_q[1][{0,0,0}]=(electrons-moment)/(2*volume);
    density_q[{0,0,0}]=electrons/volume;
  }
  std::vector<MatrixC> static_operators;
  std::vector<FourierDensity> static_fourier;
  for(const auto& kbasis:bases) {
    if(use_fft){static_fourier.push_back(static_local_fourier(input,pots,kbasis));continue;}
    MatrixC h=MatrixC::Zero(kbasis.g.size(),kbasis.g.size());
    for(std::size_t i=0;i<kbasis.g.size();++i)h(i,i)=.5*dot(kbasis.cart[i],kbasis.cart[i]);
    if(!std::getenv("SDFT_DISABLE_LOCAL"))assemble_local(input,pots,kbasis,b,input.lattice_bohr,h);
    static_operators.push_back(std::move(h));
  }
  std::ostringstream geometry_identity;
  geometry_identity<<std::setprecision(17)<<"geometry-guess-1:"<<atomic_unit_name(input.unit_convention)
      <<':'<<input.projector_interpolation<<':'<<input.cutoff_ev<<':'<<result.density_cutoff_ev
      <<':'<<bands<<':'<<input.smearing_ev<<':'<<result.onecenter_radial_mode;
  for(std::size_t atom=0;atom<input.atoms.size();++atom)
    geometry_identity<<':'<<input.atoms[atom].symbol<<':'<<input.atoms[atom].zval<<':'<<pots[atom].numeric_sha256;
  for(const auto& k:kpoints)geometry_identity<<':'<<k.fractional.x<<':'<<k.fractional.y<<':'<<k.fractional.z<<':'<<k.weight;
  if(polarized){geometry_identity<<":spin=2:dudarev";
    for(const auto& a:input.atoms)geometry_identity<<':'<<a.hubbard.angular_momentum<<':'<<a.hubbard.ueff_ev;}
  const auto geometry_signature=geometry_identity.str();
  std::vector<MatrixC> final_coeff(kpoints.size());
  std::vector<Eigen::VectorXd> final_eigenvalues(kpoints.size());
  if(!input.geometry_guess_input.empty()){
    if(!closure_context)throw std::invalid_argument("geometry guesses require the complete PAW closure");
    const auto guess=read_geometry_guess(input.geometry_guess_input,geometry_signature,kpoints.size(),onsite_input.size(),bands,input.spin_channels);
    const char* atomic_flag=std::getenv("SDFT_GEOMETRY_ATOMIC_DIFFERENCE");
    if(atomic_flag&&std::string(atomic_flag)!="0"&&std::string(atomic_flag)!="1")
      throw std::invalid_argument("SDFT_GEOMETRY_ATOMIC_DIFFERENCE must be 0 or 1");
    // The Si cell oracle found 93 versus 76 outer iterations with the atomic
    // difference candidate. Keep the validated affine v56 predictor by
    // default; the extra atomic correction remains an explicit experiment.
    const bool atomic_difference=atomic_flag&&std::string(atomic_flag)=="1";
    const double factor=guess.volume/std::abs(determinant(input.lattice_bohr));
    if(factor<.5||factor>2.)throw std::invalid_argument("geometry guess cell volume changed by more than factor two");
    for(std::size_t atom=0;atom<onsite_input.size();++atom){
      if(guess.onsite[atom].size()!=onsite_input[atom].size())throw std::invalid_argument("geometry guess PAW dimension mismatch");
      const auto dim=std::size_t(std::sqrt(double(guess.onsite[atom].size())));
      for(std::size_t i=0;i<dim;++i)for(std::size_t j=0;j<dim;++j)
        if(std::abs(guess.onsite[atom][i*dim+j]-std::conj(guess.onsite[atom][j*dim+i]))>1e-10)
          throw std::invalid_argument("geometry guess on-site density is not Hermitian");
    }
    onsite_input=guess.onsite;
    if(polarized){
      if(atomic_difference)throw std::invalid_argument("atomic-difference predictor has no spin validation; use affine spin transport");
      for(int c=0;c<2;++c){
        spin_onsite[c].assign(onsite_input.begin()+c*natoms,onsite_input.begin()+(c+1)*natoms);
        const auto aug=PawCompensationOperator::density_sum(closure_context->compensation,spin_onsite[c]);
        for(auto& [key,value]:spin_q[c]){const auto smooth=guess.spin_smooth[c].find(key),comp=aug.find(key);
          value=(smooth==guess.spin_smooth[c].end()?Complex{}:factor*smooth->second)+(comp==aug.end()?Complex{}:comp->second);}
      }
      const double target=electrons/std::abs(determinant(input.lattice_bohr));
      const auto correction=.5*(Complex(target)-spin_q[0][{0,0,0}]-spin_q[1][{0,0,0}]);
      spin_q[0][{0,0,0}]+=correction;spin_q[1][{0,0,0}]+=correction;
      density_q=spin_q[0];for(const auto& [key,value]:spin_q[1])density_q[key]+=value;
    }else{
    const auto compensation=PawCompensationOperator::density_sum(closure_context->compensation,onsite_input);
    for(auto& [key,value]:density_q){const auto smooth=guess.smooth.find(key),aug=compensation.find(key);
      value=(smooth==guess.smooth.end()?Complex{}:factor*smooth->second)+(aug==compensation.end()?Complex{}:aug->second);
      // Transport the bonding difference to the new atomic reference. This
      // is an initial guess only; new geometry H/S, PAW closure and fresh
      // final residual checks remain mandatory.
      if(atomic_difference&&!guess.atomic_smooth.empty()){
        const auto old_atom=guess.atomic_smooth.find(key),new_atom=smooth_density_support.find(key);
        value+=(new_atom==smooth_density_support.end()?Complex{}:new_atom->second)-
            (old_atom==guess.atomic_smooth.end()?Complex{}:factor*old_atom->second);
      }
    }
    if(atomic_difference&&!guess.atomic_smooth.empty())record_performance("geometry_atomic_difference_predictor",0);
    density_q[{0,0,0}]=electrons/std::abs(determinant(input.lattice_bohr));
    }
    result.geometry_guess_retained_fraction=1.;
    for(std::size_t k=0;k<kpoints.size();++k){
      const auto mapped=transport_plane_wave_seeds(guess.g[k],bases[k].g,guess.waves[k],bands);
      if(mapped.minimum_retained_norm_fraction<.5)throw std::invalid_argument("geometry guess lost over half its wavefunction norm");
      result.geometry_guess_retained_fraction=std::min(result.geometry_guess_retained_fraction,mapped.minimum_retained_norm_fraction);
      final_coeff[k]=Eigen::Map<const MatrixC>(mapped.coefficients.data(),bases[k].g.size(),bands);
    }
    result.geometry_guess_used=true;record_performance("scf_geometry_guess_import",0);
  }

  // Per-k search state is local to this fixed-cell/PAW/basis SCF call. Ionic
  // or cell changes start a new cache; no state crosses a geometry boundary.
  std::vector<MatrixC> spectral_guards(kpoints.size());
  const int pulay_depth = 6;
  // Keep the smooth and one-centre state on the same damped input by default.
  // A separate coefficient is intentionally not enabled implicitly: the
  // Pulay metric must be calibrated against the coupled response before
  // changing the relative damping of the on-site occupation and the Fourier
  // density.
  const double onsite_mixing = input.mixing;
  // The Broyden mixer (Johnson, PRB 38, 12807 (1988)) keeps a joint history
  // but uses separate charge/magnetic initial responses and a charge
  // inverse-Kerker metric (Kerker, PRB 23, 3082 (1981)). The default magnetic
  // mixing amplitude is four times the charge amplitude; this is solver
  // preconditioning, not a fixed magnetization or a modification of the PAW
  // functional.
  auto control_number=[](const char* name,double fallback,double upper) {
    if(const char* value=std::getenv(name)){
      std::size_t end=0;const double number=std::stod(value,&end);
      if(end!=std::string(value).size()||!std::isfinite(number)||number<0||number>upper)
        throw std::invalid_argument(std::string(name)+" outside valid range");
      return number;
    }
    return fallback;
  };
  const double magnetic_amix=control_number("SDFT_SPIN_MIXING",std::min(2.,4*input.mixing),2.);
  if(polarized&&magnetic_amix<=0)throw std::invalid_argument("SDFT_SPIN_MIXING must be positive");
  const double charge_bmix=control_number("SDFT_CHARGE_KERKER",1.,100.);
  const double magnetic_bmix=control_number("SDFT_SPIN_KERKER",1.,100.);
  bool full_diis=false, multisecant=true;
  result.mixer_method="multisecant";
  if(const char* value=std::getenv("SDFT_MIXER")){
    const std::string choice(value);
    result.mixer_method=choice;
    if(choice!="full_diis"&&choice!="damped_diis"&&choice!="multisecant")throw std::invalid_argument("SDFT_MIXER must be full_diis, damped_diis or multisecant");
    multisecant=choice=="multisecant";
    full_diis=choice=="full_diis";
  }
  if(full_diis&&result.onecenter_closure_diagnostic)result.onecenter_mixing_method="joint_density_onsite_full_diis";
  if(multisecant)result.onecenter_mixing_method="joint_regularized_multisecant";
  if(polarized&&multisecant)
    result.onecenter_mixing_method="joint_multisecant_charge_kerker_spin_response_packed_lm_metric";
  bool shrink_enabled=multisecant;
  if(const char* v=std::getenv("SDFT_MULTISECANT_BACKTRACK")) {
    if(std::string(v)!="0"&&std::string(v)!="1")throw std::invalid_argument("SDFT_MULTISECANT_BACKTRACK must be 0 or 1");
    shrink_enabled=multisecant&&std::string(v)=="1";
  }
  result.multisecant_backtracking=shrink_enabled;
  AdaptiveSecantStep step_control(shrink_enabled);
  double previous_outer_error=1.;
  const bool adaptive_search=std::getenv("SDFT_ADAPTIVE_SEARCH")&&std::string(std::getenv("SDFT_ADAPTIVE_SEARCH"))=="1";
  result.adaptive_search=adaptive_search;
  const bool trial_search=density_dataflow_control("SDFT_SCF_RMM_TRIAL");
  if(trial_search&&(!adaptive_search||!use_fft))
    throw std::invalid_argument("SCF trial search requires adaptive FFT SCF");
  // The v43 global guarded solver is the fallback; there is no separate
  // local predictor followed by a duplicate complete solve.
  result.trial_search_enabled=trial_search;
  double search_factor=1e-4;
  if(const char* setting=std::getenv("SDFT_SCF_SEARCH_FACTOR")){
    std::size_t end=0;const std::string value(setting);search_factor=std::stod(value,&end);
    if(end!=value.size()||!std::isfinite(search_factor)||search_factor<=0||search_factor>.1)
      throw std::invalid_argument("SDFT_SCF_SEARCH_FACTOR must be in (0,0.1]");
  }
  bool precision_refined=!adaptive_search;
  CollinearInnerAccuracy collinear_accuracy(result.effective_density_tolerance,
      result.effective_onecenter_density_tolerance, adaptive_search);
  const char* gpu_pbe_setting=std::getenv("SDFT_GPU_PBE");
  if(gpu_pbe_setting&&std::string(gpu_pbe_setting)!="0"&&std::string(gpu_pbe_setting)!="1")
    throw std::invalid_argument("SDFT_GPU_PBE must be 0 or 1");
  bool pbe_precision_refined=!(gpu_pbe_setting&&std::string(gpu_pbe_setting)=="1"&&compute_backend_name()!="cpu");
  std::ofstream iteration_log;
  if(!input.iteration_log.empty()) {
    iteration_log.open(input.iteration_log,std::ios::trunc);
    if(!iteration_log)throw std::runtime_error("cannot open SCF iteration log");
  }
  std::optional<DensityFunctionals> stationary_functional;
  std::optional<std::array<DensityFunctionals,2>> stationary_spin_functional;
  std::vector<double> kpoint_cost(kpoints.size(),1.);
  for (int iteration = 1; iteration <= input.max_iterations; ++iteration) {
    const double final_eigen_accuracy=polarized ? collinear_accuracy.final_accuracy() :
        // Even a loose outer request cannot accept roots above the final
        // 1e-12 Ha certificate used by frozen NSCF and analytic derivatives.
        // Strict outer inputs retain their original 5e-14 Ha request.
        std::max(5e-14,std::min(1e-12,std::min(input.density_tolerance,input.onecenter_density_tolerance)/10000.0));
    const double outer_error=iteration==1 ? 1.0 : std::max(result.density_residual,result.onecenter_density_residual);
    if(polarized && iteration>1 && collinear_accuracy.observe(
        result.density_residual, result.onecenter_density_residual)) {
      // Residual histories obtained with a changing inexact solve must not
      // carry its occupation noise into the final FP64 fixed-point solve.
      pulay_residuals.clear();pulay_outputs.clear();
      onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
      record_performance("scf_collinear_final_inner_precision_transition",0);
    }
    if(!pbe_precision_refined&&(outer_error<1e-5||iteration==input.max_iterations)){
      pbe_precision_refined=true;
      pulay_residuals.clear();pulay_outputs.clear();onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
      record_performance("scf_pbe_fp64_transition",0);
    }
    if(!precision_refined&&(outer_error<1e-5||iteration==input.max_iterations)){
      precision_refined=true;result.precision_refinement_iteration=iteration;
      pulay_residuals.clear();pulay_outputs.clear();onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
      record_performance("scf_full_precision_transition",0);
    }
    if(multisecant && iteration>2 && outer_error>3*previous_outer_error) {
      pulay_residuals.clear();pulay_outputs.clear();
      onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
      record_performance("scf_multisecant_response_reset",0);
    }
    previous_outer_error=outer_error;
    double eh = 0.0, exc = 0.0, evxc = 0.0;
    double paw_spherical_dij_norm_ev=0.0, paw_dc_ae_ev=0.0, paw_dc_ps_ev=0.0;
    double paw_onecenter_energy_ev=0.0, paw_onecenter_potential_expectation_ev=0.0;
    double paw_onecenter_core_xc_ev=0.0;
    std::vector<ScfOneCenterState> iteration_onecenter;
    result.hamiltonian_density_q = density_q;
    std::optional<DensityFunctionals> shared_functional;
    std::optional<std::array<DensityFunctionals,2>> spin_functionals;
    std::array<std::vector<ScfOneCenterState>,2> iteration_spin_states;
    std::array<std::vector<PawOneCenterClosureResult>,2> spin_closures;
    if(polarized){
      result.spin_hamiltonian_density_q=spin_q;
      for(int channel=0;channel<2;++channel)spin_onsite[channel].assign(onsite_input.begin()+channel*natoms,onsite_input.begin()+(channel+1)*natoms);
      result.hubbard_energy_ev=0;result.hubbard_potential_expectation_ev=0;result.hubbard_states.clear();
      for(std::size_t atom=0;atom<natoms;++atom){
        std::vector<int> raw_l;for(const auto& c:pots[atom].cache.nonlocal_channels)for(int i=0;i<c.nprojectors;++i)raw_l.push_back(c.angular_momentum);
        const std::array<std::vector<Complex>,2> onsite{spin_onsite[0][atom],spin_onsite[1][atom]};
        const std::size_t dimension=std::size_t(std::sqrt(onsite[0].size()));
        auto pair=paw_collinear_one_center_closure(pots[atom].cache,raw_l,onsite,dimension,PawRadialLengthUnit::Angstrom,
          &closure_context->references[atom],closure_context->potential_mode,input.unit_convention);
        HubbardResult hubbard;
        if(input.atoms[atom].hubbard.angular_momentum>=0){
          const auto ae=paw_hubbard_ae_overlap(pots[atom].cache,raw_l,input.atoms[atom].hubbard.angular_momentum);
          hubbard=dudarev_paw(raw_l,ae,input.atoms[atom].hubbard,onsite);
          for(int channel=0;channel<2;++channel){auto& closure=pair[channel];double expectation=0;
            for(std::size_t i=0;i<onsite[channel].size();++i){const auto v=hubbard.projector_potential_ev[channel][i];
              closure.d_matrix_ev[i]+=v;expectation+=(std::conj(onsite[channel][i])*v).real();}
            closure.energy_difference_ev+=.5*hubbard.energy_ev;closure.potential_expectation_ev+=expectation;
            closure.ae_double_counting_ev+=.5*hubbard.energy_ev-expectation;
          }
        }
        result.hubbard_energy_ev+=hubbard.energy_ev;result.hubbard_potential_expectation_ev+=hubbard.potential_expectation_ev;
        result.hubbard_states.push_back(std::move(hubbard));
        for(int channel=0;channel<2;++channel)spin_closures[channel].push_back(std::move(pair[channel]));
      }
    }
    if(!std::getenv("SDFT_DISABLE_HARTREE_XC")) {
      PerformanceTimer functional_timer("density_functional");
      if(polarized){spin_functionals=evaluate_collinear_density_functionals(spin_q,input.lattice_bohr,&closure_context->pseudo_core,precision_refined?0:1);
        result.max_xc_grid_points=std::max(result.max_xc_grid_points,(*spin_functionals)[0].grid_points);
      }else{
        shared_functional=evaluate_density_functionals(density_q,input.lattice_bohr,
          closure_context ? closure_context->pseudo_core : empty_pseudo_core(), precision_refined ? 0 : 1, !pbe_precision_refined);
        result.max_xc_grid_points=std::max(result.max_xc_grid_points,shared_functional->grid_points);
      }
    }
    std::vector<ScfEigenResult> solvers(kpoints.size());
    std::vector<std::vector<double>> energies_ev(kpoints.size());
    std::vector<std::unique_ptr<PlaneWaveOperator>> fft_operators(kpoints.size());
    std::vector<MatrixC> dense_h(kpoints.size()),dense_s(kpoints.size());
    for (std::size_t k = 0; k < kpoints.size(); ++k) {
      // All operators see the same mixed smooth/on-site input. Scalar
      // functional terms belong to the cell, so only k=0 records them.
      double keh=0.0, kexc=0.0, kevxc=0.0;
      const std::size_t channel=polarized?k/physical_nk:0;
      const bool first=polarized?k%physical_nk==0:k==0;
      auto* captured=polarized?&iteration_spin_states[channel]:&iteration_onecenter;
      if(polarized)closure_context->spin_closures=&spin_closures[channel];
      auto& fft_operator=fft_operators[k];
      auto& overlap=dense_s[k];
      MatrixC h;
      {
      PerformanceTimer assembly_timer("hamiltonian_assembly");
      h = make_hamiltonian(input, pots, bases[k], input.lattice_bohr, polarized?spin_q[channel]:density_q,
          k==0 && rho.size() && rho.norm()>1e-18 ? &rho : nullptr,
          overlap, keh, kexc, kevxc,
          first ? &paw_spherical_dij_norm_ev : nullptr, first ? &paw_dc_ae_ev : nullptr,
          first ? &paw_dc_ps_ev : nullptr, first ? &paw_onecenter_energy_ev : nullptr,
          first ? &paw_onecenter_potential_expectation_ev : nullptr,
          first ? &paw_onecenter_core_xc_ev : nullptr,
          !first && result.onecenter_closure_diagnostic ? captured : nullptr,
          first && result.onecenter_closure_diagnostic ? captured : nullptr,
          polarized ? &spin_onsite[channel] : (onsite_input.empty() ? nullptr : &onsite_input),
          closure_context ? &*closure_context : nullptr,
          polarized ? &(*spin_functionals)[channel] : (shared_functional ? &*shared_functional : nullptr),
          use_fft ? nullptr : &static_operators[k],use_fft ? &fft_operator : nullptr,
          use_fft ? &static_fourier[k] : nullptr,
          result.onecenter_closure_diagnostic ? &onsite_projectors[k] : nullptr,&cached_full_ionic,
          immutable_projector_identities[k]);
      }
      if(first) {eh+=keh;exc+=kexc;evxc+=kevxc;}
      if(fft_operator) {
        result.max_fft_grid_points=std::max(result.max_fft_grid_points,fft_operator->fft_grid_points());
        result.max_operator_storage_bytes=std::max(result.max_operator_storage_bytes,fft_operator->storage_bytes());
      } else {
        result.max_operator_storage_bytes=std::max(result.max_operator_storage_bytes,
            static_cast<std::size_t>(h.size()+overlap.size()+static_operators[k].size())*sizeof(Complex));
      }
      dense_h[k]=std::move(h);
    }
    // The functional's FP64 transition is independent of eigensolver work.
    // While density/on-site D_ij still change, request accuracy proportional to
    // that change. Do not force every subsequent Hamiltonian to final
    // 1e-12 accuracy just because the functional switched at 1e-5. The
    // completion test below still requires the ORIGINAL final residual on
    // freshly applied FP64 H/S, plus both original density thresholds.
    const double eigen_accuracy=polarized ? collinear_accuracy.requested(
        outer_error, search_factor, iteration==input.max_iterations) :
        (adaptive_search && iteration<input.max_iterations
        ? std::clamp(search_factor*outer_error,final_eigen_accuracy,1e-4) : final_eigen_accuracy);
    const bool allow_trial=trial_search&&iteration>1&&(iteration-1)%4!=0&&eigen_accuracy>1e-9;
    int teams=1;
#ifdef SDFT_HAS_OPENMP
    // The accelerator owns a queue and scratch buffers per host thread.
    // Independent k operators can overlap on GPU as well as on CPU. Keep
    // FP64 final certification and density summation in their old order.
    teams=std::min<int>(kpoints.size(),omp_get_max_threads());
    if(compute_backend_name()!="cpu") {
      std::size_t budget_mb=2048;
      if(const char* value=std::getenv("SDFT_GPU_KPOINT_MEMORY_MB")) {
        const std::string text(value);std::size_t consumed=0;
        if(text.empty()||text.front()=='-')throw std::invalid_argument("invalid GPU k-point workspace budget");
        budget_mb=std::stoull(text,&consumed);
        if(consumed!=text.size()||budget_mb<1||budget_mb>1048576)
          throw std::invalid_argument("SDFT_GPU_KPOINT_MEMORY_MB must be in [1,1048576]");
      }
      std::size_t max_npw=0;
      for(const auto& kbasis:bases)max_npw=std::max(max_npw,kbasis.g.size());
      // Conservative extra workspace estimate, not a measurement of RSS:
      // double-float input/output/FFT buffers plus Davidson images/guards.
      const long double subspace=std::min<long double>(max_npw,8.L*bands);
      const long double worker_bytes=std::max<long double>(1,
          8.L*result.max_operator_storage_bytes+16.L*max_npw*subspace*sizeof(Complex));
      const long double slots=static_cast<long double>(budget_mb)*1048576.L/worker_bytes;
      const int memory_teams=static_cast<int>(std::min<long double>(teams,std::max<long double>(1,slots)));
      teams=std::min(teams,memory_teams);
      record_performance("scf_gpu_kpoint_workspace_estimate_bytes",0,static_cast<std::size_t>(worker_bytes));
    }
#endif
    const int scheduling_capacity=teams;
    if(const char* value=std::getenv("SDFT_KPOINT_THREADS")){
      std::size_t consumed=0;const int requested=std::stoi(value,&consumed);
      if(consumed!=std::string(value).size()||requested<1)throw std::invalid_argument("SDFT_KPOINT_THREADS must be positive integer");
      // An explicit request cannot oversubscribe the host team or bypass
      // the GPU workspace scheduling cap.
      teams=std::min(scheduling_capacity,requested);
    }
    if(compute_backend_name()!="cpu")record_performance("scf_gpu_kpoint_worker_capacity",0,teams);
    // Two-level CPU parallelism. With fewer k points than threads, each k
    // team gets an inner budget for its FFT/PAW/GEMM kernels instead of
    // running them serially. Each k result is independent of which thread
    // solves it, so dynamic scheduling only removes the tail imbalance.
    // SDFT_DETERMINISTIC=1 restores the historical static, single-level team.
    int inner_threads=1;
    bool dynamic_k=false;
#ifdef SDFT_HAS_OPENMP
    if(!deterministic_mode()&&teams>1) {
      inner_threads=std::max(1,omp_get_max_threads()/teams);
      dynamic_k=true;
      if(inner_threads>1&&omp_get_max_active_levels()<2)omp_set_max_active_levels(2);
    }
    record_performance("scf_kpoint_teams",0,teams);
    record_performance("scf_kpoint_inner_threads",0,inner_threads);
#endif
    std::exception_ptr failure;std::mutex failure_mutex;
    // MPI: each k group diagonalizes the k points it owns; results are
    // broadcast from their owner below, so every later step (occupations,
    // density, mixing, derivatives) sees the same data on every rank.
    auto& mpi=parallel();
    const bool distributed=mpi.kgroups()>1;
    if(distributed)mpi.assign_kpoints(kpoint_cost);
    const auto solve_k=[&](std::size_t k){
      if(distributed&&!mpi.owns(k))return;
      try {
        InnerThreadBudget budget(inner_threads);
        const auto started=std::chrono::steady_clock::now();
        struct CostRecorder{std::vector<double>& cost;std::size_t k;std::chrono::steady_clock::time_point t0;
          ~CostRecorder(){cost[k]=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}}
          recorder{kpoint_cost,k,started};
        PerformanceTimer solve_timer("scf_diagonalization");
        solvers[k]=use_fft ? diagonalize(*fft_operators[k],bands,&final_coeff[k],eigen_accuracy,&spectral_guards[k],allow_trial)
                          : diagonalize(dense_h[k],dense_s[k],bands,&final_coeff[k],eigen_accuracy);
        final_coeff[k]=solvers[k].eigenvectors;final_eigenvalues[k]=solvers[k].eigenvalues;
        if(!solvers[k].trial_manifold)spectral_guards[k]=std::move(solvers[k].search_guards);
        energies_ev[k].resize(bands);
        for(int band=0;band<bands;++band)energies_ev[k][band]=solvers[k].eigenvalues[band]*ev_per_hartree;
      }catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
    };
    if(dynamic_k) {
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(dynamic,1) num_threads(teams)
#endif
      for(std::size_t k=0;k<kpoints.size();++k)solve_k(k);
    } else {
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(teams) if(teams>1)
#endif
      for(std::size_t k=0;k<kpoints.size();++k)solve_k(k);
    }
    // A failure on any rank must stop all ranks, not leave them waiting.
    if(!mpi.all(!failure)){if(failure)std::rethrow_exception(failure);
      throw std::runtime_error("eigensolver failed on another MPI rank");}
    if(distributed) {
      PerformanceTimer exchange_timer("mpi_kpoint_exchange");
      for(std::size_t k=0;k<kpoints.size();++k) {
        const int owner=mpi.owner_of(k);
        auto& solved=solvers[k];
        struct Flags{std::uint8_t converged,trial;double residual;std::int64_t rows,cols,guard_rows,guard_cols;}
          flags{std::uint8_t(solved.converged),std::uint8_t(solved.trial_manifold),solved.max_residual,
            final_coeff[k].rows(),final_coeff[k].cols(),spectral_guards[k].rows(),spectral_guards[k].cols()};
        mpi.broadcast_from_group(owner,&flags,sizeof flags);
        solved.converged=flags.converged;solved.trial_manifold=flags.trial;solved.max_residual=flags.residual;
        final_coeff[k].resize(flags.rows,flags.cols);
        mpi.broadcast_from_group(owner,final_coeff[k].data(),sizeof(Complex)*final_coeff[k].size());
        final_eigenvalues[k].resize(bands);
        mpi.broadcast_from_group(owner,final_eigenvalues[k].data(),sizeof(double)*bands);
        if(!mpi.owns(k)&&!solved.trial_manifold)spectral_guards[k].resize(flags.guard_rows,flags.guard_cols);
        if(!solved.trial_manifold)
          mpi.broadcast_from_group(owner,spectral_guards[k].data(),sizeof(Complex)*spectral_guards[k].size());
        energies_ev[k].resize(bands);
        for(int band=0;band<bands;++band)energies_ev[k][band]=final_eigenvalues[k][band]*ev_per_hartree;
        mpi.broadcast_from_group(owner,&kpoint_cost[k],sizeof(double));
      }
      record_performance("mpi_kpoint_groups",0,mpi.kgroups());
    }
    result.eigensolver_max_residual=0;
    bool all_guarded=true;
    for(std::size_t k=0;k<kpoints.size();++k){
      all_guarded &= solvers[k].converged&&!solvers[k].trial_manifold;
      if(solvers[k].trial_manifold)++result.trial_manifold_updates;
      else ++result.guarded_manifold_updates;
      result.eigensolver_max_residual=std::max(result.eigensolver_max_residual,solvers[k].max_residual);
      if(fft_operators[k])result.max_operator_storage_bytes=std::max(result.max_operator_storage_bytes,fft_operators[k]->storage_bytes());
    }
    result.final_guarded_search=all_guarded;
    result.eigensolver_history_max_residual=std::max(result.eigensolver_history_max_residual,result.eigensolver_max_residual);
    if(polarized){result.spin_onecenter_states=std::move(iteration_spin_states);result.onecenter_states=result.spin_onecenter_states[0];
      result.onecenter_states.insert(result.onecenter_states.end(),result.spin_onecenter_states[1].begin(),result.spin_onecenter_states[1].end());}
    else result.onecenter_states = std::move(iteration_onecenter);
    std::vector<double> weights;
    for(const auto& point:physical_kpoints)weights.push_back(point.weight);
    const auto filled=polarized?fill_collinear_occupations(energies_ev,weights,electrons,input.smearing_ev):fill_occupations(energies_ev, weights, electrons, input.smearing_ev);
    occupations=filled.occupations;
    result.entropy_energy_ev=filled.entropy_energy_ev;
    result.fermi_energy_ev=filled.chemical_potential_ev;
    result.kpoint_eigenvalues_ev=energies_ev;
    result.kpoint_occupations=occupations;
    FourierDensity new_charge, legacy_augmentation;
    std::array<FourierDensity,2> new_spin_charge;
    const bool ordered_charge=ordered_fourier_enabled();
    {
    PerformanceTimer smooth_density_timer("scf_smooth_density_fft_and_accumulation");
    for(std::size_t k=0;k<kpoints.size();++k) {
      if(k==0 && env_with_alias("SDFT_ENABLE_SPHERICAL_ONECENTER","SDFT_ENABLE_ONECENTER_CHF")) {
        const auto new_rho=density_from_coeff(final_coeff[k],bands,occupations[k]);
        const auto nk=bases[k].g.size();rho.resize(nk,nk);
        for(std::size_t i=0;i<nk;++i)for(std::size_t j=0;j<nk;++j)rho(i,j)=new_rho[i*nk+j];
      }
      std::vector<std::tuple<int,int,int>> kg;
      std::vector<Complex> coefficients(bases[k].g.size()*bands);
      for(const auto& vector:bases[k].g)kg.emplace_back(vector.h,vector.k,vector.l);
      for(int band=0;band<bands;++band)for(std::size_t i=0;i<kg.size();++i)
        coefficients[band*kg.size()+i]=final_coeff[k](i,band);
      auto smooth_charge=plane_wave_density_fft(kg,coefficients,occupations[k],
          std::abs(determinant(input.lattice_bohr)),smooth_density_support);
      if(polarized){for(const auto& [key,value]:smooth_charge){new_charge[key]+=kpoints[k].weight*value;new_spin_charge[k/physical_nk][key]+=kpoints[k].weight*value;}}
      else if(ordered_charge)consume_ordered_fourier(new_charge,std::move(smooth_charge),kpoints[k].weight);
      else for(const auto& [key,value]:smooth_charge)new_charge[key]+=kpoints[k].weight*value;
      if(!closure_context) {
        const auto aug=full_augmentation_allowed(bases[k])
            ? full_augmentation_density(input,pots,bases[k],final_coeff[k],bands,occupations[k])
            : augmentation_density(input,pots,bases[k],final_coeff[k],bands,occupations[k]);
        if(ordered_charge)accumulate_ordered_fourier(legacy_augmentation,aug,kpoints[k].weight);
        else for(const auto& [key,value]:aug)legacy_augmentation[key]+=kpoints[k].weight*value;
      }
    }
    }
    std::vector<MatrixC> fresh_onsite_amplitudes;
    if(closure_context&&use_fft&&density_dataflow_control("SDFT_BATCH_ONSITE_DENSITY")) {
      PerformanceTimer projected_timer("scf_onsite_batch_projection");
      fresh_onsite_amplitudes.resize(kpoints.size());
      for(std::size_t k=0;k<kpoints.size();++k) {
        auto op=fft_operators[k]->hamiltonian();std::size_t expected=0;
        for(const auto& beta:onsite_projectors[k])expected+=beta.rows();
        if(!op.project_into||op.projector_rows!=expected)
          throw std::runtime_error("onsite projection atom ordering/dimension mismatch");
        auto& projected=fresh_onsite_amplitudes[k];projected.resize(expected,bands);
        // Recompute from these final wavefunction values using the original
        // B in FP64, not approximate search amplitudes or a previous SCF.
        op.project_into(final_coeff[k].data(),bands,projected.data(),true);
      }
      record_performance("scf_onsite_batched_columns",0,kpoints.size()*bands);
    }
    std::vector<std::vector<Complex>> onsite_output;
    std::vector<Complex> onsite_prior_vec, onsite_output_vec, onsite_residual_vec;
    result.onecenter_density_residual = 0.0;
    result.onecenter_density_residuals.clear();
    bool onsite_ready = !result.onecenter_closure_diagnostic;
    if (result.onecenter_closure_diagnostic) {
      onsite_ready = result.onecenter_states.size() == natoms*(polarized?2:1);
      PerformanceTimer onsite_timer("scf_onsite_density");
      std::size_t projector_offset=0;
      for (std::size_t atom = 0; atom < natoms*(polarized?2:1); ++atom) {
        const auto atom_rows=onsite_projectors.front()[atom%natoms].rows();
        std::vector<Complex> output(onsite_input[atom].size());
        for(std::size_t k=0;k<kpoints.size();++k) {
          if(polarized&&k/physical_nk!=atom/natoms)continue;
          if(onsite_projectors[k][atom%natoms].rows()!=atom_rows)
            throw std::runtime_error("onsite channel count changed between k points");
          const auto contribution=fresh_onsite_amplitudes.empty()
              ? onsite_from_coeff(onsite_projectors[k][atom%natoms],final_coeff[k],bands,occupations[k])
              : onsite_density_from_amplitudes(atom_rows,bands,
                  fresh_onsite_amplitudes[k].data()+projector_offset,
                  fresh_onsite_amplitudes[k].rows(),occupations[k].data());
          for(std::size_t entry=0;entry<output.size();++entry)output[entry]+=kpoints[k].weight*contribution[entry];
        }
        const auto& prior=onsite_input[atom];
        double error_norm2=0.0,output_norm2=0.0;
        for(std::size_t entry=0;entry<output.size();++entry) {
          const Complex error=output[entry]-prior[entry];
          error_norm2+=std::norm(error);output_norm2+=std::norm(output[entry]);
          onsite_prior_vec.push_back(prior[entry]);onsite_output_vec.push_back(output[entry]);
          onsite_residual_vec.push_back(error);
        }
        const double atom_residual=std::sqrt(error_norm2)/std::max(1.0,std::sqrt(output_norm2));
        result.onecenter_density_residuals.push_back(atom_residual);
        result.onecenter_density_residual=std::max(result.onecenter_density_residual,atom_residual);
        onsite_output.push_back(std::move(output));
        projector_offset+=atom_rows;if((atom+1)%natoms==0)projector_offset=0;
      }
    }
    // Build this iteration's output density from one wavefunction state.  The
    // Fourier and one-centre blocks are kept as separate coordinates of one
    // mixer vector, but both are outputs of the same occupied subspace.
    std::vector<Complex> mixed_onsite(onsite_prior_vec.size());
    for (std::size_t i = 0; i < mixed_onsite.size(); ++i)
      mixed_onsite[i] = onsite_prior_vec[i] + onsite_mixing * onsite_residual_vec[i];
    // The potential depends on the charge Fourier coefficients, not on the
    // arbitrary unitary rotation inside a degenerate occupied subspace.
    FourierDensity aug_charge;
    if (closure_context) {
      std::size_t expected_mixed_size = 0;
      for (const auto& atom_density : onsite_output) expected_mixed_size += atom_density.size();
      if (mixed_onsite.size() != expected_mixed_size)
        throw std::runtime_error("PAW mixed on-site density is not fully formed before augmentation");
      // The output charge belongs to these wavefunctions; mixing follows
      // below. Batch contraction retains atom summation order and avoids a
      // grid-sized tree allocation for every atom at every SCF iteration.
      if(polarized){for(int channel=0;channel<2;++channel){std::vector<std::vector<Complex>> part(onsite_output.begin()+channel*natoms,onsite_output.begin()+(channel+1)*natoms);
        const auto aug=PawCompensationOperator::density_sum(closure_context->compensation,part);
        for(const auto& [key,value]:aug){aug_charge[key]+=value;new_spin_charge[channel][key]+=value;}}}
      else aug_charge=PawCompensationOperator::density_sum(closure_context->compensation,onsite_output);
    } else {
      aug_charge = std::move(legacy_augmentation);
    }
    for (const auto& [key, value] : aug_charge) new_charge[key] += value;
    result.integrated_electron_count = (new_charge[{0, 0, 0}] *
        std::abs(determinant(input.lattice_bohr))).real();
    result.compensation_electron_count = (aug_charge[{0, 0, 0}] *
        std::abs(determinant(input.lattice_bohr))).real();
    // Enforce the electron-number constraint before measuring the residual.
    // The low-rank overlap operator changes the coefficient norm, while the
    // physical density must still integrate to the requested valence count.
    if (!closure_context)
      new_charge[{0, 0, 0}] = electrons / std::abs(determinant(input.lattice_bohr));
    else
      // The smooth plane-wave density and mixed on-site augmentation are
      // separate SCF state variables. Their G=0 components need not agree
      // before mixing, so impose the electron-number constraint on the
      // combined state rather than rejecting a valid damped iterate.
      new_charge[{0, 0, 0}] = electrons / std::abs(determinant(input.lattice_bohr));
    const auto& old_charge = density_q;
    // Sorted-vector residual/mixer state: same values as the tree path
    // (bitwise on Si2, Si8, Ni at CPU8). Default on unless deterministic.
    const bool linear_state=std::getenv("SDFT_LINEAR_FOURIER_MIXER")
        ? density_dataflow_control("SDFT_LINEAR_FOURIER_MIXER") : !deterministic_mode();
    std::optional<AlignedFourierState> aligned;
    if(linear_state) {
      PerformanceTimer mixer_timer("scf_fourier_linear_alignment");
      aligned=align_fourier_states(old_charge,new_charge);
    }
    double residual = aligned ? aligned->maximum_residual : 0.0;
    std::vector<DensityKey> new_keys;
    if(aligned)new_keys=std::move(aligned->keys);
    else {
      for (const auto& [key, value] : new_charge) {
        const auto it=old_charge.find(key);
        residual=std::max(residual,std::abs(value-(it==old_charge.end()?Complex{}:it->second)));
      }
      for (const auto& [key, value] : old_charge) {
        const auto it=new_charge.find(key);
        residual=std::max(residual,std::abs(value-(it==new_charge.end()?Complex{}:it->second)));
      }
      std::set<DensityKey> keys;
      for(const auto& [key,value]:old_charge){(void)value;keys.insert(key);}
      for(const auto& [key,value]:new_charge){(void)value;keys.insert(key);}
      new_keys.assign(keys.begin(),keys.end());
    }
    if(new_keys!=density_keys) {
      pulay_residuals.clear();pulay_outputs.clear();
      onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
      density_keys=std::move(new_keys);
    }
    std::vector<Complex> prior_vec,output_vec,residual_vec;
    if(aligned) {
      prior_vec=std::move(aligned->prior);output_vec=std::move(aligned->output);
      residual_vec=std::move(aligned->residual);
    }else {
      prior_vec.reserve(density_keys.size());output_vec.reserve(density_keys.size());residual_vec.reserve(density_keys.size());
      for(const auto& key:density_keys) {
        const auto old_it=old_charge.find(key);
        const auto new_it=new_charge.find(key);
        const Complex prior=old_it==old_charge.end()?Complex{}:old_it->second;
        const Complex value=new_it==new_charge.end()?Complex{}:new_it->second;
        prior_vec.push_back(prior);output_vec.push_back(value);residual_vec.push_back(value-prior);
      }
    }
    if(polarized){
      // Charge and magnetization form independent coordinates with one
      // coupled history and one residual-feedback step controller.
      for(const auto& key:density_keys){const auto old=spin_q[0][key]-spin_q[1][key],fresh=new_spin_charge[0][key]-new_spin_charge[1][key];
        prior_vec.push_back(old);output_vec.push_back(fresh);residual_vec.push_back(fresh-old);residual=std::max(residual,std::abs(fresh-old));}
    }
    std::vector<double> smooth_response,onsite_response,smooth_metric;
    std::vector<Complex> mixer_onsite_prior,mixer_onsite_output,mixer_onsite_residual;
    if(polarized&&multisecant){
      smooth_response.resize(prior_vec.size());smooth_metric.resize(prior_vec.size());
      std::vector<double> g2(density_keys.size());double minimum_g2=std::numeric_limits<double>::infinity();
      for(std::size_t q=0;q<density_keys.size();++q){
        const auto [h,k,l]=density_keys[q];
        const auto g=add(add(scale(b[0],h),scale(b[1],k)),scale(b[2],l));
        g2[q]=dot(g,g)*bohr_per_angstrom*bohr_per_angstrom;
        if(g2[q]>1e-10)minimum_g2=std::min(minimum_g2,g2[q]);
      }
      if(!std::isfinite(minimum_g2))throw std::runtime_error("spin mixer has no nonzero reciprocal mode");
      const double minimum_response=std::min({.1,input.mixing,magnetic_amix});
      for(std::size_t q=0;q<density_keys.size();++q){
        const auto square=g2[q]>1e-10?g2[q]:minimum_g2;
        smooth_response[q]=kerker_response(square,input.mixing,charge_bmix,minimum_response);
        smooth_response[density_keys.size()+q]=kerker_response(square,magnetic_amix,magnetic_bmix,minimum_response);
        // The charge metric 1+19*G_min^2/G^2 weights the longest nonzero
        // wavelength by 20. Magnetization uses the identity metric, including its G=0.
        smooth_metric[q]=std::sqrt(g2[q]>1e-10?1+19*minimum_g2/g2[q]:1.);
        smooth_metric[density_keys.size()+q]=1.;
      }
      mixer_onsite_prior=collinear_charge_magnetization(onsite_prior_vec);
      mixer_onsite_output=collinear_charge_magnetization(onsite_output_vec);
      mixer_onsite_residual=collinear_charge_magnetization(onsite_residual_vec);
      const auto nsite=mixer_onsite_prior.size()/2;onsite_response.resize(2*nsite);
      std::fill_n(onsite_response.begin(),nsite,std::min(input.mixing,.5));
      std::fill(onsite_response.begin()+nsite,onsite_response.end(),std::min(magnetic_amix,2.));
    }
    pulay_residuals.push_back(residual_vec); pulay_outputs.push_back(output_vec);
    if (result.onecenter_closure_diagnostic) {
      onsite_pulay_residuals.push_back(onsite_residual_vec);
      onsite_pulay_outputs.push_back(onsite_output_vec);
    }
    if (static_cast<int>(pulay_residuals.size()) > pulay_depth) {
      pulay_residuals.erase(pulay_residuals.begin()); pulay_outputs.erase(pulay_outputs.begin());
      if (result.onecenter_closure_diagnostic) {
        onsite_pulay_residuals.erase(onsite_pulay_residuals.begin());
        onsite_pulay_outputs.erase(onsite_pulay_outputs.begin());
      }
    }
    const double feedback=std::max(residual/std::max(result.effective_density_tolerance,1e-30),
        result.onecenter_closure_diagnostic ? result.onecenter_density_residual/
          std::max(result.effective_onecenter_density_tolerance,1e-30) : 0.);
    const double step_scale=step_control.observe(feedback);
    result.final_mixing_step_scale=step_scale;
    std::vector<Complex> mixed_vec(prior_vec.size());
    for (std::size_t i = 0; i < prior_vec.size(); ++i) mixed_vec[i] = prior_vec[i] + input.mixing * residual_vec[i];
    if(polarized&&multisecant){
      for(std::size_t q=0;q<mixed_vec.size();++q)
        mixed_vec[q]=prior_vec[q]+smooth_response[q]*residual_vec[q];
      auto site=mixer_onsite_prior;
      for(std::size_t q=0;q<site.size();++q)site[q]+=onsite_response[q]*mixer_onsite_residual[q];
      mixed_onsite=collinear_spin_blocks(site);
    }
    const auto linear_smooth=mixed_vec,linear_onsite=mixed_onsite;
    double secant_proposal_scale=1., secant_inverse_gain=0.;
    int secant_rank=0;
    if(multisecant && pulay_residuals.size()>=2) {
      PerformanceTimer timer("scf_multisecant_update");
      const double volume_bohr=std::abs(determinant(input.lattice_bohr));
      const double volume_angstrom=volume_bohr/std::pow(bohr_per_angstrom,3);
      // The spin mixer scales integrated Fourier charge by 1/sqrt(Omega[Å^3]),
      // while the packed (L,M) channel moments are already occupations. Sparse-DFT Fourier coefficients
      // are density per Bohr^3: rho_g*Omega_bohr is the integrated charge.
      // Grid-size averages and mean-density normalization are not that
      // metric and change the charge/onsite balance with system size.
      const double smooth_weight=polarized?volume_bohr*volume_bohr/volume_angstrom:
          1./(std::max(1.,static_cast<double>(density_keys.size()))*
          std::pow(electrons/volume_bohr,2));
      const double onsite_weight=polarized?1.:
          1./std::max(1.,static_cast<double>(onsite_residual_vec.size()));
      auto metric_smooth=pulay_residuals,metric_onsite=onsite_pulay_residuals;
      if(polarized){
        for(auto& row:metric_smooth)for(std::size_t q=0;q<row.size();++q)row[q]*=smooth_metric[q];
        for(auto& row:metric_onsite){
          row=collinear_charge_magnetization(row);std::size_t offset=0;
          for(int channel=0;channel<2;++channel)for(std::size_t atom=0;atom<natoms;++atom){
            const auto dim=static_cast<std::size_t>(std::sqrt(double(onsite_input[atom].size())));
            std::vector<Complex> block(row.begin()+offset,row.begin()+offset+dim*dim);
            const auto lm_moments=paw_packed_channel_lm_complex(closure_context->augmentation[atom],block,dim);
            std::copy(lm_moments.begin(),lm_moments.end(),row.begin()+offset);offset+=dim*dim;
          }
        }
      }
      const auto secants=joint_secant_coefficients(metric_smooth,metric_onsite,smooth_weight,onsite_weight);
      secant_rank=secants.rank;
      if(secants.usable) {
        // Once a usable multisecant inverse response exists, extrapolate
        // full output images. Input mixing damps the initial/failed linear
        // step. Damped images are also a valid conservative strategy; the
        // full-image choice requires separate performance validation.
        // The norm bound and residual-feedback
        // contraction below still control the entire coupled step.
        auto proposed=output_vec,proposed_onsite=onsite_output_vec;
        if(polarized){
          proposed=preconditioned_secant_image(prior_vec,pulay_outputs,pulay_residuals,smooth_response,secants);
          auto site_outputs=onsite_pulay_outputs,site_residuals=onsite_pulay_residuals;
          for(auto& row:site_outputs)row=collinear_charge_magnetization(row);
          for(auto& row:site_residuals)row=collinear_charge_magnetization(row);
          proposed_onsite=collinear_spin_blocks(preconditioned_secant_image(
              mixer_onsite_prior,site_outputs,site_residuals,onsite_response,secants));
        }else{
          apply_joint_secants(proposed,pulay_outputs,pulay_residuals,1.,secants);
          if(result.onecenter_closure_diagnostic)
            apply_joint_secants(proposed_onsite,onsite_pulay_outputs,onsite_pulay_residuals,1.,secants);
        }
        double residual_norm=0,step_norm=0,prior_norm=0;
        for(std::size_t q=0;q<proposed.size();++q) {
          residual_norm+=smooth_weight*std::norm(residual_vec[q]);
          step_norm+=smooth_weight*std::norm(proposed[q]-prior_vec[q]);
          prior_norm+=smooth_weight*std::norm(prior_vec[q]);
        }
        for(std::size_t q=0;q<proposed_onsite.size();++q) {
          residual_norm+=onsite_weight*std::norm(onsite_residual_vec[q]);
          step_norm+=onsite_weight*std::norm(proposed_onsite[q]-onsite_prior_vec[q]);
          prior_norm+=onsite_weight*std::norm(onsite_prior_vec[q]);
        }
        const double radius=control_number("SDFT_MULTISECANT_TRUST_RADIUS",.1,1.);
        const double scale=secant_state_trust_scale(prior_norm,step_norm,radius);
        secant_proposal_scale=scale;
        secant_inverse_gain=residual_norm>0 ? std::sqrt(step_norm/residual_norm) : 0.;
        for(std::size_t q=0;q<proposed.size();++q)
          mixed_vec[q]=prior_vec[q]+scale*(proposed[q]-prior_vec[q]);
        for(std::size_t q=0;q<proposed_onsite.size();++q)
          mixed_onsite[q]=onsite_prior_vec[q]+scale*(proposed_onsite[q]-onsite_prior_vec[q]);
        record_performance("scf_multisecant_accepted",0);
      } else record_performance("scf_multisecant_linear_fallback",0);
    }
    if (!multisecant && pulay_residuals.size() >= 2) {
      PerformanceTimer pulay_timer("scf_pulay_metric_and_update");
      const int m = static_cast<int>(pulay_residuals.size());
      Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(m + 1, m + 1);
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1); rhs[m] = 1.0;
      for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) {
        double value = 0.0;
        for (std::size_t q = 0; q < residual_vec.size(); ++q)
          value += (std::conj(pulay_residuals[static_cast<std::size_t>(i)][q]) * pulay_residuals[static_cast<std::size_t>(j)][q]).real();
        // Fourier residuals have density units; normalize by the cell mean
        // density. Onsite blocks carry projected occupancies. Averaging both
        // blocks keeps grid size from silently dominating the shared metric.
        value /= std::max(1.0, static_cast<double>(density_keys.size())) *
                 std::pow(electrons / std::abs(determinant(input.lattice_bohr)), 2);
        if (result.onecenter_closure_diagnostic) {
          double onsite_value = 0.0;
          for (std::size_t entry = 0; entry < onsite_residual_vec.size(); ++entry)
            onsite_value += (std::conj(onsite_pulay_residuals[static_cast<std::size_t>(i)][entry]) *
                             onsite_pulay_residuals[static_cast<std::size_t>(j)][entry]).real();
          value += onsite_value / std::max(1.0, static_cast<double>(onsite_residual_vec.size()));
        }
        augmented(i, j) = value; augmented(i, m) = 1.0; augmented(m, i) = 1.0;
      }
      // The residual Gram block can be 1e-20 while the constraint rows are
      // O(1). Solving that unscaled saddle system declares the physics block
      // rank deficient and silently falls back to linear mixing near a
      // strict fixed point. A uniform rescale preserves the DIIS minimizer.
      const double gram_scale=augmented.topLeftCorner(m,m).cwiseAbs().maxCoeff();
      if(gram_scale>0)augmented.topLeftCorner(m,m)/=gram_scale;
      const Eigen::VectorXd coeffs = augmented.fullPivLu().solve(rhs);
      const bool pulay_solution_valid = coeffs.allFinite() &&
          (augmented*coeffs-rhs).norm()<1e-8 &&
          std::abs(coeffs.head(m).sum() - 1.0) < 1e-8 &&
          coeffs.head(m).cwiseAbs().maxCoeff() < 10.0;
      if (pulay_solution_valid) {
        std::vector<Complex> pulay_vec(prior_vec.size(), Complex(0.0, 0.0));
        std::vector<Complex> pulay_onsite_vec(onsite_prior_vec.size(), Complex(0.0, 0.0));
        for (int i = 0; i < m; ++i) for (std::size_t q = 0; q < pulay_vec.size(); ++q)
          // Store the mixed-image vector X + alpha(F(X)-X).  The current
          // small-matrix DIIS history uses this damped image as its secant
          // approximation; this is the stable form for the complex PAW
          // closure while the full Broyden preconditioner is introduced.
          pulay_vec[q] += coeffs[i] * (pulay_outputs[static_cast<std::size_t>(i)][q] -
              (result.onecenter_closure_diagnostic && !full_diis ? 1.0 - input.mixing : 0.0) *
              pulay_residuals[static_cast<std::size_t>(i)][q]);
        if (result.onecenter_closure_diagnostic) {
          for (int i = 0; i < m; ++i) for (std::size_t q = 0; q < pulay_onsite_vec.size(); ++q)
            pulay_onsite_vec[q] += coeffs[i] *
                (onsite_pulay_outputs[static_cast<std::size_t>(i)][q] -
                 (full_diis ? 0.0 : 1.0 - input.mixing) * onsite_pulay_residuals[static_cast<std::size_t>(i)][q]);
        }
        // A Broyden mixer applies the extrapolated residual in the shared
        // density space.  Keep a conservative default for difficult PAW
        // closures, but make the damping explicit and reproducible so it can
        // be scanned without recompilation.
        double weight = full_diis ? 1.0 : 0.5;
        if (const char* env = std::getenv("SDFT_PULAY_WEIGHT")) {
          std::size_t used=0;
          try { weight = std::stod(env,&used); }
          catch (const std::exception&) { throw std::invalid_argument("SDFT_PULAY_WEIGHT must be a finite number in [0,1]"); }
          if(used!=std::string(env).size()||!std::isfinite(weight)||weight<0||weight>1)
            throw std::invalid_argument("SDFT_PULAY_WEIGHT must be a finite number in [0,1]");
        }
        for (std::size_t q = 0; q < mixed_vec.size(); ++q) mixed_vec[q] = (1.0 - weight) * mixed_vec[q] + weight * pulay_vec[q];
        if (result.onecenter_closure_diagnostic)
          for (std::size_t q = 0; q < mixed_onsite.size(); ++q)
            mixed_onsite[q] = (1.0 - weight) * mixed_onsite[q] + weight * pulay_onsite_vec[q];
      }
    }
    bool spin_feasibility_fallback=false;
    bool spin_convex_fallback=false;
    double positivity_step_scale=1.;
    if(polarized){
      std::vector<std::size_t> dimensions;
      for(const auto& block:onsite_input)dimensions.push_back(std::size_t(std::sqrt(double(block.size()))));
      if(multisecant)spin_feasibility_fallback=finalize_preconditioned_spin_step(
          mixed_vec,mixed_onsite,prior_vec,onsite_prior_vec,linear_smooth,linear_onsite,
          dimensions,step_scale,positivity_step_scale,output_vec,onsite_output_vec,input.mixing,
          &spin_convex_fallback);
      else spin_feasibility_fallback=finalize_collinear_mixing_step(mixed_vec,mixed_onsite,
          prior_vec,onsite_prior_vec,residual_vec,onsite_residual_vec,dimensions,step_scale,input.mixing);
      if(spin_feasibility_fallback){
        pulay_residuals.clear();pulay_outputs.clear();onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
        record_performance("scf_spin_feasibility_linear_reset",0);
      }
    }
    // A multisecant inverse response is allowed to extrapolate the coupled
    // charge/spin state, but it must not jump across a locally resolved spin
    // state in one step.  A Kerker-damped linear mix damps this mode implicitly;
    // the old Sparse-DFT path could take a perfectly positive onsite matrix whose
    // Fe moment had already changed sign, after which the SCF settled in the
    // ferromagnetic basin while reporting a tiny residual.  Reject only that
    // discontinuous proposal and fall back to the same coupled linear step
    // for the magnetic Fourier and onsite coordinates.  This does not impose
    // a target moment and does not alter a continuous converged solution.
    bool spin_basin_guard=false;
    bool spin_basin_guard_enabled=false;
    if(const char* guard=std::getenv("SDFT_SPIN_BASIN_GUARD")) {
      if(std::string(guard)!="0"&&std::string(guard)!="1")
        throw std::invalid_argument("SDFT_SPIN_BASIN_GUARD must be 0 or 1");
      spin_basin_guard_enabled=std::string(guard)=="1";
    }
    if(polarized&&result.onecenter_closure_diagnostic&&spin_basin_guard_enabled&&
       onsite_output.size()==2*natoms&&onsite_input.size()==2*natoms) {
      auto trace_block=[](const std::vector<Complex>& block) {
        const std::size_t n=static_cast<std::size_t>(std::sqrt(double(block.size())));
        if(!n||n*n!=block.size())throw std::runtime_error("invalid onsite block for spin basin guard");
        double trace=0.;for(std::size_t i=0;i<n;++i)trace+=block[i*n+i].real();return trace;
      };
      std::vector<std::size_t> offsets(2*natoms+1,0);
      for(std::size_t i=0;i<2*natoms;++i)offsets[i+1]=offsets[i]+onsite_input[i].size();
      auto mixed_block=[&](std::size_t i)->std::vector<Complex> {
        return std::vector<Complex>(mixed_onsite.begin()+static_cast<std::ptrdiff_t>(offsets[i]),
            mixed_onsite.begin()+static_cast<std::ptrdiff_t>(offsets[i+1]));
      };
      auto reset_spin_coordinates=[&]() {
        const std::size_t smooth_size=density_keys.size();
        if(mixed_vec.size()!=2*smooth_size)
          throw std::runtime_error("spin mixer coordinate size mismatch");
        for(std::size_t i=0;i<smooth_size;++i)
          mixed_vec[smooth_size+i]=prior_vec[smooth_size+i]+step_scale*(linear_smooth[smooth_size+i]-prior_vec[smooth_size+i]);
        for(std::size_t i=0;i<mixed_onsite.size();++i)
          mixed_onsite[i]=onsite_prior_vec[i]+step_scale*(linear_onsite[i]-onsite_prior_vec[i]);
      };
      for(std::size_t atom=0;atom<natoms;++atom) {
        if(std::abs(input.atoms[atom].initial_magnetic_moment)<1e-8)continue;
        const double prior_m=trace_block(onsite_input[atom])-trace_block(onsite_input[natoms+atom]);
        const double output_m=trace_block(onsite_output[atom])-trace_block(onsite_output[natoms+atom]);
        const double candidate_m=trace_block(mixed_block(atom))-trace_block(mixed_block(natoms+atom));
        const double scale=std::max({1.,std::abs(prior_m),std::abs(output_m),std::abs(candidate_m)});
        const bool output_crossed=prior_m*output_m < -1e-8*scale*scale;
        const bool candidate_crossed=output_m*candidate_m < -1e-8*scale*scale;
        if(output_crossed||candidate_crossed) {
          reset_spin_coordinates();spin_basin_guard=true;
          break;
        }
      }
      if(spin_basin_guard) {
        pulay_residuals.clear();pulay_outputs.clear();onsite_pulay_residuals.clear();onsite_pulay_outputs.clear();
        record_performance("scf_spin_basin_linear_reset",0);
      }
    }
    if(shrink_enabled&&!polarized) {
      for(std::size_t q=0;q<mixed_vec.size();++q)
        mixed_vec[q]=prior_vec[q]+step_scale*(mixed_vec[q]-prior_vec[q]);
      for(std::size_t q=0;q<mixed_onsite.size();++q)
        mixed_onsite[q]=onsite_prior_vec[q]+step_scale*(mixed_onsite[q]-onsite_prior_vec[q]);
    }
    density_q.clear();
    for (std::size_t key_index = 0; key_index < density_keys.size(); ++key_index) {
      const auto& key = density_keys[key_index];
      const Complex value = mixed_vec[key_index];
      const auto [dh, dk, dl] = key;
      if(linear_state)density_q.insert_or_assign(density_q.end(),key,value);
      else density_q[key]=value;
    }
    if (!closure_context)
      density_q[{0, 0, 0}] = electrons / std::abs(determinant(input.lattice_bohr));
    else
      density_q[{0, 0, 0}] = electrons / std::abs(determinant(input.lattice_bohr));
    if(polarized){for(std::size_t i=0;i<density_keys.size();++i){const auto& key=density_keys[i];const auto charge=density_q[key],magnetic=mixed_vec[density_keys.size()+i];
      spin_q[0][key]=.5*(charge+magnetic);spin_q[1][key]=.5*(charge-magnetic);}}
    if (result.onecenter_closure_diagnostic) {
      std::size_t offset = 0;
      onsite_input = onsite_output;
      for (auto& atom_density : onsite_input) {
        std::copy_n(mixed_onsite.begin() + static_cast<std::ptrdiff_t>(offset), atom_density.size(), atom_density.begin());
        offset += atom_density.size();
      }
    }
    result.iterations = iteration;
    result.density_residual = residual;
    if(iteration_log.is_open()) {
      iteration_log<<std::setprecision(17)<<"{\"iteration\":"<<iteration
        <<",\"density_residual\":"<<residual<<",\"onecenter_density_residual\":"<<result.onecenter_density_residual
        <<",\"eigensolver_max_residual_hartree\":"<<result.eigensolver_max_residual
        <<",\"eigensolver_requested_tolerance_hartree\":"<<eigen_accuracy
        <<",\"eigensolver_final_tolerance_hartree\":"<<final_eigen_accuracy
        <<",\"eigensolver_final_refinement\":"
        <<(((polarized ? collinear_accuracy.final_refinement() : !adaptive_search)||
            iteration==input.max_iterations)?"true":"false")
        <<",\"mixer\":\""<<result.mixer_method<<"\",\"multisecant_backtracking\":"
        <<(result.multisecant_backtracking?"true":"false")<<",\"mixing_step_scale\":"<<result.final_mixing_step_scale
        <<",\"multisecant_rank\":"<<secant_rank
        <<",\"multisecant_proposal_scale\":"<<secant_proposal_scale
        <<",\"multisecant_inverse_gain\":"<<secant_inverse_gain
        <<",\"spin_feasibility_fallback\":"<<(spin_feasibility_fallback?"true":"false")
        <<",\"positivity_step_scale\":"<<positivity_step_scale
        <<",\"spin_convex_fallback\":"<<(spin_convex_fallback?"true":"false")
        <<",\"spin_basin_guard\":"<<(spin_basin_guard?"true":"false")
        <<",\"charge_amix\":"<<input.mixing<<",\"magnetic_amix\":"<<magnetic_amix
        <<",\"charge_bmix_inverse_angstrom\":"<<charge_bmix
        <<",\"magnetic_bmix_inverse_angstrom\":"<<magnetic_bmix;
      if(polarized){
        const auto volume=std::abs(determinant(input.lattice_bohr));
        double occupation_delta=0.;
        std::vector<std::pair<std::size_t,HubbardResult>> traced_occupations;
        iteration_log<<",\"output_magnetic_moment_muB\":"
            <<(new_spin_charge[0][{0,0,0}]-new_spin_charge[1][{0,0,0}]).real()*volume
            <<",\"mixed_magnetic_moment_muB\":"
            <<(spin_q[0][{0,0,0}]-spin_q[1][{0,0,0}]).real()*volume
            <<",\"output_correlated_moments_muB\":[";
        for(std::size_t atom=0;atom<natoms;++atom){
          if(atom)iteration_log<<',';
          const auto& spec=input.atoms[atom].hubbard;
          if(spec.angular_momentum<0){iteration_log<<"null";continue;}
          std::vector<int> raw_l;for(const auto& ch:pots[atom].cache.nonlocal_channels)
            for(int i=0;i<ch.nprojectors;++i)raw_l.push_back(ch.angular_momentum);
          const auto ae=paw_hubbard_ae_overlap(pots[atom].cache,raw_l,spec.angular_momentum);
          const std::array<std::vector<Complex>,2> blocks{onsite_output[atom],onsite_output[natoms+atom]};
          auto hub=dudarev_paw(raw_l,ae,spec,blocks);double moment=0.;
          for(std::size_t i=0;i<hub.correlated_dimension;++i)
            moment+=(hub.occupations[0][i*hub.correlated_dimension+i]-hub.occupations[1][i*hub.correlated_dimension+i]).real();
          iteration_log<<moment;
          for(int channel=0;channel<2;++channel)
            for(std::size_t q=0;q<hub.occupations[channel].size();++q)
              occupation_delta=std::max(occupation_delta,
                  std::abs(hub.occupations[channel][q]-result.hubbard_states[atom].occupations[channel][q]));
          traced_occupations.emplace_back(atom,std::move(hub));
        }
        iteration_log<<']';
        iteration_log<<",\"output_correlated_states\":[";
        for(std::size_t index=0;index<traced_occupations.size();++index){
          if(index)iteration_log<<',';
          const auto& [atom,hub]=traced_occupations[index];
          iteration_log<<"{\"atom_index\":"<<atom<<",\"dimension\":"<<hub.correlated_dimension
              <<",\"occupations\":[";
          for(int channel=0;channel<2;++channel){
            if(channel)iteration_log<<',';iteration_log<<'[';
            for(std::size_t q=0;q<hub.occupations[channel].size();++q){
              if(q)iteration_log<<',';const auto n=hub.occupations[channel][q];
              iteration_log<<'['<<n.real()<<','<<n.imag()<<']';
            }
            iteration_log<<']';
          }
          iteration_log<<"]}";
        }
        iteration_log<<"],\"hubbard_occupation_residual\":"<<occupation_delta;
      }
      iteration_log
        <<",\"guarded_lowest_spectrum_search\":"<<(all_guarded?"true":"false")
        <<",\"solver\":\""<<result.solver_method<<"\"}\n";
      iteration_log.flush();
      if(!iteration_log)throw std::runtime_error("cannot write SCF iteration log");
    }
    if (std::getenv("SDFT_TRACE_SCF")) std::cerr << "SCF " << iteration << " residual " << residual
        << " onsite_residual " << result.onecenter_density_residual << "\n";
    result.hartree_energy_ev = eh * ev_per_hartree;
    result.xc_energy_ev = exc * ev_per_hartree;
    result.vxc_density_ev = evxc * ev_per_hartree;
    result.paw_spherical_dij_norm_ev = paw_spherical_dij_norm_ev;
    result.paw_double_counting_ae_ev = paw_dc_ae_ev;
    result.paw_double_counting_ps_ev = paw_dc_ps_ev;
    result.paw_onecenter_energy_ev = paw_onecenter_energy_ev;
    result.paw_onecenter_potential_expectation_ev = paw_onecenter_potential_expectation_ev;
    result.paw_onecenter_core_xc_ev = paw_onecenter_core_xc_ev;
    result.eigenvalues_ev = energies_ev.front(); // backward-compatible first-k spectrum
    result.band_energy_ev = 0.0;
    for(std::size_t k=0;k<kpoints.size();++k)for(int band=0;band<bands;++band)
      result.band_energy_ev+=kpoints[k].weight*occupations[k][band]*energies_ev[k][band];
    if (all_guarded && precision_refined && pbe_precision_refined && result.eigensolver_max_residual<=final_eigen_accuracy && residual < result.effective_density_tolerance &&
        onsite_ready && (!result.onecenter_closure_diagnostic ||
        result.onecenter_density_residual < result.effective_onecenter_density_tolerance)) {
      result.converged = true;
      if(polarized)stationary_spin_functional=std::move(spin_functionals);
      if(density_dataflow_control("SDFT_STATIONARY_RECIPROCAL") && shared_functional)
        stationary_functional=std::move(shared_functional);
      break;
    }
  }
  if(polarized){result.spin_density_q=spin_q;const double volume=std::abs(determinant(input.lattice_bohr));
    result.magnetic_moment=(spin_q[0][{0,0,0}]-spin_q[1][{0,0,0}]).real()*volume;}
  result.total_energy_ev = result.band_energy_ev - result.hartree_energy_ev + result.xc_energy_ev - result.vxc_density_ev;
  // The one-centre D_ij is already inside the band-energy sum.  Add the
  // one-centre functional value E^1 - E~^1 and subtract the D_ij expectation
  // once (the PAW double-counting term; Kresse & Joubert, PRB 59, 1758
  // (1999)).  The AE one-centre XC energy includes the frozen core, so that
  // core-only XC energy is removed once as well.
  result.total_energy_ev += result.paw_onecenter_energy_ev -
      result.paw_onecenter_potential_expectation_ev -
      result.paw_onecenter_core_xc_ev;
  result.ion_ion_energy_ev = ion_ion_ewald(input, input.lattice_bohr, b) * ev_per_hartree;
  const double volume_ang3 = std::abs(determinant(input.lattice_bohr)) / std::pow(bohr_per_angstrom, 3);
  for (const auto& pot : pots) {
    if (!pot.cache.local_potential.empty())
      result.local_g0_energy_ev += pot.cache.local_potential.front() * electrons / volume_ang3;
    if (!pot.cache.atomic_reference_present)
      throw std::runtime_error("PAW total-energy ledger requires the original EATOM data");
    const double error = pot.kinetic_error_spline.empty() ? 0.0
        : pot.kinetic_error_spline.value(input.cutoff_ev);
    result.atomic_reference_energy_ev += pot.cache.atomic_reference_energy_ev;
    result.atomic_cutoff_correction_ev -= error;
  }
  result.total_energy_ev += result.ion_ion_energy_ev + result.local_g0_energy_ev +
      result.atomic_reference_energy_ev + result.atomic_cutoff_correction_ev;
  result.internal_energy_ev = result.total_energy_ev;
  result.total_energy_ev += result.entropy_energy_ev; // finite-T free energy
  if (result.converged&&!polarized) {
    PerformanceTimer derivatives_timer("analytic_derivatives");
    const bool reuse_projectors=derivative_option("SDFT_DERIVATIVE_PROJECTOR_REUSE")&&closure_context.has_value();
    if(reuse_projectors)record_performance("analytic_projector_scf_reuse",0,input.atoms.size()*kpoints.size());
    std::optional<StationaryReciprocalData> stationary_reciprocal;
    if(stationary_functional && closure_context)
      stationary_reciprocal=prepare_stationary_reciprocal(input,pots,*closure_context,
          result.onecenter_states,result.hamiltonian_density_q,*stationary_functional);
    result.hf_electronic_forces_ev_per_angstrom.assign(input.atoms.size(), Vec3{});
    for(std::size_t k=0;k<kpoints.size();++k) {
      // Local, compensation and NLCC derivatives are cell terms. Evaluating
      // them once per k then applying weights sums each of them exactly once.
      const auto contribution=electronic_hf_forces(input,pots,bases[k],input.lattice_bohr,
          final_coeff[k],final_eigenvalues[k],occupations[k],bands,
          result.onecenter_closure_diagnostic ? &result.onecenter_states : nullptr,
          closure_context ? &*closure_context : nullptr,
          result.onecenter_closure_diagnostic ? &result.hamiltonian_density_q : nullptr,
          stationary_reciprocal ? &*stationary_reciprocal : nullptr,
          reuse_projectors ? &onsite_projectors[k] : nullptr);
      for(std::size_t atom=0;atom<input.atoms.size();++atom)
        result.hf_electronic_forces_ev_per_angstrom[atom]=add(
            result.hf_electronic_forces_ev_per_angstrom[atom],scale(contribution[atom],kpoints[k].weight));
    }
    std::vector<Vec3> ewald;
    ion_ion_ewald(input, input.lattice_bohr, b, &ewald);
    result.hf_total_forces_ev_per_angstrom = result.hf_electronic_forces_ev_per_angstrom;
    for (std::size_t atom=0;atom<ewald.size();++atom) {
      const auto force = scale(ewald[atom], ev_per_hartree * bohr_per_angstrom);
      result.ewald_forces_ev_per_angstrom.push_back(force);
      result.hf_total_forces_ev_per_angstrom[atom] = add(result.hf_total_forces_ev_per_angstrom[atom], force);
    }
    if(closure_context) {
      result.hf_electronic_stress_ev_per_angstrom3=electronic_paw_stress(input,pots,bases,final_coeff,
          final_eigenvalues,occupations,result.onecenter_states,*closure_context,result.hamiltonian_density_q,
          result.local_g0_energy_ev,result.stress_components_ev_per_angstrom3,
          stationary_functional ? &*stationary_functional : nullptr,
          stationary_reciprocal ? &*stationary_reciprocal : nullptr,
          reuse_projectors ? &onsite_projectors : nullptr);
      std::vector<Vec3> positions;std::vector<double> charges;
      for(const auto& atom:input.atoms){positions.push_back(atom.fractional);charges.push_back(atom.zval);}
      const auto ionic=evaluate_ewald(input.lattice_bohr,positions,charges);
      for(int a=0;a<3;++a){
        result.ewald_stress_ev_per_angstrom3[a]=scale(ionic.stress[a],ev_per_hartree*std::pow(bohr_per_angstrom,3));
        result.hf_total_stress_ev_per_angstrom3[a]=add(result.hf_electronic_stress_ev_per_angstrom3[a],result.ewald_stress_ev_per_angstrom3[a]);
      }
      result.stress_components_ev_per_angstrom3["ewald"]=result.ewald_stress_ev_per_angstrom3;
      result.analytic_stress_available=true;
    }
  }

  if(result.converged&&polarized){
    PerformanceTimer timer("analytic_collinear_derivatives");
    if(!stationary_spin_functional)throw std::runtime_error("missing final FP64 spin functional");
    ScfInput derivative_input=input;derivative_input.kpoints=physical_kpoints;
    result.hf_electronic_forces_ev_per_angstrom.assign(natoms,Vec3{});
    double up_electrons=0;
    for(std::size_t k=0;k<physical_nk;++k)for(double f:occupations[k])up_electrons+=physical_kpoints[k].weight*f;
    for(int channel=0;channel<2;++channel){
      PawScfClosureContext context=*closure_context;context.spin_closures=nullptr;
      for(auto& [key,value]:context.pseudo_core)value*=.5;
      const auto& states=result.spin_onecenter_states[channel];
      const auto& functional=(*stationary_spin_functional)[channel];
      const auto& density=result.spin_hamiltonian_density_q[channel];
      const auto prepared=prepare_stationary_reciprocal(derivative_input,pots,context,states,density,functional,.5);
      std::vector<Basis> cbases(bases.begin()+channel*physical_nk,bases.begin()+(channel+1)*physical_nk);
      std::vector<MatrixC> waves(final_coeff.begin()+channel*physical_nk,final_coeff.begin()+(channel+1)*physical_nk);
      std::vector<Eigen::VectorXd> eigen(final_eigenvalues.begin()+channel*physical_nk,final_eigenvalues.begin()+(channel+1)*physical_nk);
      std::vector<std::vector<double>> occ(occupations.begin()+channel*physical_nk,occupations.begin()+(channel+1)*physical_nk);
      std::vector<std::vector<MatrixC>> projectors(onsite_projectors.begin()+channel*physical_nk,onsite_projectors.begin()+(channel+1)*physical_nk);
      for(std::size_t k=0;k<physical_nk;++k){const auto forces=electronic_hf_forces(derivative_input,pots,cbases[k],input.lattice_bohr,waves[k],eigen[k],occ[k],bands,
        &states,&context,&density,&prepared,&projectors[k]);
        for(std::size_t atom=0;atom<natoms;++atom)result.hf_electronic_forces_ev_per_angstrom[atom]=add(result.hf_electronic_forces_ev_per_angstrom[atom],scale(forces[atom],physical_kpoints[k].weight));
      }
      std::map<std::string,std::array<Vec3,3>> components;
      const double channel_g0=result.local_g0_energy_ev*(channel==0?up_electrons:electrons-up_electrons)/electrons;
      const auto stress=electronic_paw_stress(derivative_input,pots,cbases,waves,eigen,occ,states,context,density,channel_g0,components,&functional,&prepared,&projectors);
      for(int a=0;a<3;++a)result.hf_electronic_stress_ev_per_angstrom3[a]=add(result.hf_electronic_stress_ev_per_angstrom3[a],stress[a]);
      for(const auto& [name,tensor]:components)for(int a=0;a<3;++a)result.stress_components_ev_per_angstrom3[name][a]=add(result.stress_components_ev_per_angstrom3[name][a],tensor[a]);
    }
    std::vector<Vec3> positions;std::vector<double> charges;
    for(const auto& atom:input.atoms){positions.push_back(atom.fractional);charges.push_back(atom.zval);}
    const auto ionic=evaluate_ewald(input.lattice_bohr,positions,charges);
    result.hf_total_forces_ev_per_angstrom=result.hf_electronic_forces_ev_per_angstrom;
    for(std::size_t atom=0;atom<natoms;++atom){const auto f=scale(ionic.forces[atom],ev_per_hartree*bohr_per_angstrom);
      result.ewald_forces_ev_per_angstrom.push_back(f);result.hf_total_forces_ev_per_angstrom[atom]=add(result.hf_total_forces_ev_per_angstrom[atom],f);}
    for(int a=0;a<3;++a){result.ewald_stress_ev_per_angstrom3[a]=scale(ionic.stress[a],ev_per_hartree*std::pow(bohr_per_angstrom,3));
      result.hf_total_stress_ev_per_angstrom3[a]=add(result.hf_electronic_stress_ev_per_angstrom3[a],result.ewald_stress_ev_per_angstrom3[a]);}
    result.stress_components_ev_per_angstrom3["ewald"]=result.ewald_stress_ev_per_angstrom3;result.analytic_stress_available=true;
  }
  if(!input.geometry_guess_output.empty()&&result.converged){
    if(!closure_context)throw std::invalid_argument("geometry guesses require the complete PAW closure");
    ScfGeometryGuess guess;guess.spin_channels=input.spin_channels;guess.signature=geometry_signature;
    guess.volume=std::abs(determinant(input.lattice_bohr));guess.onsite=onsite_input;
    guess.smooth=density_q;guess.atomic_smooth=smooth_density_support;
    if(polarized){
      for(int c=0;c<2;++c){std::vector<std::vector<Complex>> part(onsite_input.begin()+c*natoms,onsite_input.begin()+(c+1)*natoms);
        guess.spin_smooth[c]=spin_q[c];
        const auto comp=PawCompensationOperator::density_sum(closure_context->compensation,part);
        for(const auto& [key,value]:comp)guess.spin_smooth[c][key]-=value;
      }
      guess.smooth=guess.spin_smooth[0];for(const auto& [key,value]:guess.spin_smooth[1])guess.smooth[key]+=value;
    }else{
    const auto compensation=PawCompensationOperator::density_sum(closure_context->compensation,onsite_input);
    for(const auto& [key,value]:compensation)guess.smooth[key]-=value;
    }
    for(std::size_t k=0;k<kpoints.size();++k){guess.g.push_back(bases[k].g);
      guess.waves.emplace_back(final_coeff[k].data(),final_coeff[k].data()+final_coeff[k].size());}
    write_geometry_guess(input.geometry_guess_output,guess);
  }
  result.density_q = density_q;
  support_indices.clear();
  for (const auto& [key, value] : density_q) { (void)value; support_indices.push_back(key); }
  result.density_support_index_sha256 = reciprocal_index_sha256(std::move(support_indices));
  result.density_reciprocal_modes = density_q.size();
  result.density_fourier.clear();
  for (const auto& [key, value] : density_q) result.density_fourier.push_back(value.real());
  return result;
}

// An explicit band count is a hard request and fails loudly. The automatic
// count (occupied + 4, polarized + 8) can be too small for smeared metals and
// slabs, so it grows by max(4, 25 %) and SCF restarts from the atomic density.
// The smearing gate itself (top band empty to 1e-6 at every k) is unchanged.
ScfResult run_scf(const ScfInput& input) {
  if(input.bands>0||input.smearing_ev<=0) {
    auto result=run_scf_attempt(input);result.bands_requested=input.bands;return result;
  }
  constexpr int max_extensions=4;
  ScfInput attempt=input;
  for(int extension=0;;++extension) {
    try {
      auto result=run_scf_attempt(attempt);
      result.bands_requested=0;result.empty_band_extensions=extension;
      return result;
    } catch(const InsufficientEmptyBands& failure) {
      if(extension==max_extensions)throw;
      const int solved=int(failure.bands);
      attempt.bands=solved+std::max(4,int(std::ceil(.25*solved)));
      record_performance("scf_automatic_empty_band_extension",0);
      std::cerr<<"Sparse-DFT: automatic bands "<<solved<<" leave the smeared top band occupied; restarting SCF with "
               <<attempt.bands<<" bands\n";
    }
  }
}


ScfResult read_frozen_scf(const std::filesystem::path& path,const ScfInput& input) {
  std::ifstream digest(path.string()+".sha256");std::string expected;
  if(!(digest>>expected)||expected.size()!=64||sha256_file(path)!=expected)
    throw std::runtime_error("frozen SCF SHA-256 is missing or mismatched");
  std::ifstream f(path);std::string magic,units;double cutoff;
  f>>magic>>units>>cutoff;
  const bool v10 = magic=="SDFT_FROZEN_SCF_V10";
  std::string common_framework;int common_spin=1;
  if(v10) {
    // density cutoff is read below, so retain this extra header until then.
    std::streampos marker=f.tellg();double density_cutoff;f>>density_cutoff>>common_framework>>common_spin;
    if(!f||(common_framework!="mixed_paw"&&common_framework!="potcar_paw")||(common_spin!=1&&common_spin!=2))
      throw std::runtime_error("invalid common PAW frozen header");
    f.seekg(marker);
  }
  const bool v9 = magic=="SDFT_FROZEN_SCF_V9";
  const bool v8 = v9 || magic=="SDFT_FROZEN_SCF_V8";
  const bool v7 = (v10&&common_spin==2) || v9 || magic=="SDFT_FROZEN_SCF_V7";
  const bool v6 = v7 || magic=="SDFT_FROZEN_SCF_V6";
  const bool v5 = v10 || v8 || v6 || magic=="SDFT_FROZEN_SCF_V5";
  const bool v4 = v5 || magic=="SDFT_FROZEN_SCF_V4";
  const bool v3 = v4 || magic=="SDFT_FROZEN_SCF_V3";
  const bool v2 = v3 || magic=="SDFT_FROZEN_SCF_V2";
  if(!f || !std::isfinite(cutoff) || cutoff<=0 ||
     (!v2 && magic!="SDFT_FROZEN_SCF_V1")||units!=atomic_unit_name(input.unit_convention)||
     std::abs(cutoff-input.cutoff_ev)>1e-10)
    throw std::runtime_error("frozen SCF format, atomic units or cutoff differ");
  ScfResult out;out.unit_convention=input.unit_convention;
  if(v8){if(homogeneous_paw_framework(input)!=PawFramework::Upf)throw std::runtime_error("UPF checkpoint requires UPF atom inputs");out.paw_framework="upf_paw";}
  if(v10)out.paw_framework=common_framework;
  out.onecenter_radial_mode="radial_gga"; // Historical V1/V2 closure.
  out.cutoff_ev = cutoff;
  if (v2) {
    f >> out.density_cutoff_ev;
    if (!f || !std::isfinite(out.density_cutoff_ev) || out.density_cutoff_ev<4*cutoff ||
        (input.density_cutoff_ev>0 && std::abs(out.density_cutoff_ev-input.density_cutoff_ev)>1e-10))
      throw std::runtime_error("frozen SCF density_cutoff_ev is invalid or differs from requested NSCF");
  } else if (input.density_cutoff_ev>0 || input.basis_reference_lattice_bohr) {
    throw std::runtime_error("legacy frozen SCF has no density cutoff or reference-basis provenance");
  }
  if(v10) {
    std::string framework;int spins;f>>framework>>spins;
    const auto selected=selected_paw_framework(input);
    if(framework!=common_framework||spins!=input.spin_channels||
       (framework=="mixed_paw"&&selected!=PawFramework::Mixed)||
       (framework=="potcar_paw"&&selected!=PawFramework::Potcar))throw std::runtime_error("frozen per-site PAW model differs");
  }
  if(v3){
    f>>out.onecenter_radial_mode;
    out.onecenter_radial_mode=canonical_onecenter_radial_mode(out.onecenter_radial_mode);
    if(!f || (v10 ? (out.onecenter_radial_mode!="per_site_original_variational_paw_development" && out.onecenter_radial_mode!="per_site_original_energy_adjoint_paw_development") : v8 ? out.onecenter_radial_mode!="UPF_original_QijL_discrete_variational_multik_development" : (out.onecenter_radial_mode!="aspherical_gga" && out.onecenter_radial_mode!="radial_gga")))
      throw std::runtime_error("frozen SCF has an unsupported PAW radial mode");
  }
  if(v5){f>>out.projector_interpolation;out.projector_interpolation=canonical_projector_interpolation(out.projector_interpolation);}
  if(out.projector_interpolation!=input.projector_interpolation)
    throw std::runtime_error("frozen SCF projector interpolation differs from requested calculation");
  if(v6){
    if(v7){std::string convention;f>>convention;
      out.hubbard_occupation_convention=v10?"per_site_original_paw_hubbard_v1":v9?"upf_pseudo_qae_bound_v1":"ae_partial_wave_rdep_angstrom_v1";
      if(!f||convention!=out.hubbard_occupation_convention)
        throw std::runtime_error("frozen Hubbard radial convention differs");}
    std::size_t na=0;f>>out.spin_channels>>na;
    if(!f||out.spin_channels!=2||input.spin_channels!=2||na!=input.atoms.size())
      throw std::runtime_error("frozen spin/model atom count differs");
    for(std::size_t a=0;a<na;++a){HubbardSpec spec;f>>spec.angular_momentum>>spec.ueff_ev;
      if(!f||spec.angular_momentum!=input.atoms[a].hubbard.angular_momentum||
         !std::isfinite(spec.ueff_ev)||spec.ueff_ev!=input.atoms[a].hubbard.ueff_ev)
        throw std::runtime_error("frozen Hubbard model differs from input");
      out.hubbard_specs.push_back(spec);
      if(spec.ueff_ev>0&&!v7)
        throw std::runtime_error("legacy +U checkpoint has an unverified PAW radial convention; recompute SCF");
    }
  }else if(input.spin_channels!=1)throw std::runtime_error("nonmagnetic checkpoint cannot supply a spin Hamiltonian");
  for(auto& a:out.onecenter_lattice_bohr)f>>a.x>>a.y>>a.z;
  for(int a=0;a<3;++a)if(norm(sub(out.onecenter_lattice_bohr[a],input.lattice_bohr[a]))>1e-12)
    throw std::runtime_error("frozen SCF lattice differs from requested NSCF lattice");
  if (v2) {
    int has_reference=-1; f >> has_reference;
    if (!f || (has_reference!=0 && has_reference!=1))
      throw std::runtime_error("invalid frozen SCF reference-basis flag");
    if (has_reference) {
      std::array<Vec3,3> reference{};
      for (auto& v : reference) f >> v.x >> v.y >> v.z;
      reciprocal(reference);
      out.basis_reference_lattice_bohr = reference;
    }
    if (out.basis_reference_lattice_bohr.has_value()!=input.basis_reference_lattice_bohr.has_value())
      throw std::runtime_error("frozen SCF reference-basis policy differs");
    if (out.basis_reference_lattice_bohr) for (int a=0; a<3; ++a)
      if (norm(sub(out.basis_reference_lattice_bohr->at(a), input.basis_reference_lattice_bohr->at(a)))>1e-12)
        throw std::runtime_error("frozen SCF reference-basis lattice differs");
  }
  f>>out.converged>>out.onecenter_closure_diagnostic>>out.total_energy_valid
   >>out.iterations>>out.occupied_bands>>out.electron_count>>out.fermi_energy_ev
   >>out.density_residual>>out.onecenter_density_residual>>out.onecenter_density_tolerance
   >>out.integrated_electron_count>>out.compensation_electron_count;
  if(!out.converged||!out.onecenter_closure_diagnostic||out.total_energy_valid||
     out.iterations<1||out.occupied_bands<1||!std::isfinite(out.fermi_energy_ev)||
     !std::isfinite(out.density_residual)||!std::isfinite(out.onecenter_density_residual)||
     out.density_residual<0||out.onecenter_density_residual<0||
     out.onecenter_density_residual>=out.onecenter_density_tolerance)
    throw std::runtime_error("NSCF requires a converged, finite, correctly labelled PAW state");
  const auto count=[&](std::size_t limit){std::size_t n=0;f>>n;if(!f||n>limit)throw std::runtime_error("invalid frozen SCF count");return n;};
  const auto density=[&](FourierDensity& q) {
    const auto n=count(10000000);
    for(std::size_t i=0;i<n;++i){int h,k,l;double re,im;f>>h>>k>>l>>re>>im;
      if(!f||std::abs(h)>100000||std::abs(k)>100000||std::abs(l)>100000||!std::isfinite(re)||!std::isfinite(im)||!q.emplace(std::make_tuple(h,k,l),Complex(re,im)).second)
        throw std::runtime_error("invalid or duplicate frozen Fourier density");}
  };
  density(out.density_q);density(out.hamiltonian_density_q);density(out.frozen_core_q);
  const auto nk=count(200000);
  for(std::size_t k=0;k<nk;++k){ScfInput::Kpoint point;f>>point.fractional.x>>point.fractional.y>>point.fractional.z>>point.weight;
    if(!std::isfinite(point.weight)||point.weight<=0)throw std::runtime_error("invalid frozen k weight");out.kpoints.push_back(point);}
  if(v4)for(std::size_t k=0;k<nk;++k) {
    const auto bands=count(1000000);if(!bands)throw std::runtime_error("empty frozen spectrum");
    std::vector<double> spectrum(bands),occupations(bands);
    for(auto& value:spectrum){f>>value;if(!f||!std::isfinite(value))throw std::runtime_error("invalid frozen eigenvalue");}
    for(auto& value:occupations){f>>value;if(!f||!std::isfinite(value)||value<0||value>2)throw std::runtime_error("invalid frozen occupation");}
    out.kpoint_eigenvalues_ev.push_back(std::move(spectrum));out.kpoint_occupations.push_back(std::move(occupations));
  }
  const auto read_atoms=[&](std::vector<ScfOneCenterState>& states){
  const auto na=count(100000);
  if(na!=input.atoms.size())throw std::runtime_error("frozen atom count differs");
  for(std::size_t atom=0;atom<na;++atom) {
    ScfOneCenterState state;std::string cache;
    f>>state.atom_index>>state.projector_dimension>>std::quoted(state.symbol)
     >>state.fractional.x>>state.fractional.y>>state.fractional.z>>std::quoted(cache)
     >>std::quoted(state.record_id)>>state.numeric_cache_version>>state.numeric_cache_checksum
     >>state.numeric_cache_sha256>>state.projector_scale>>state.projector_q_bohr>>state.reciprocal_linear;
    state.paw_cache=cache;state.unit_convention=input.unit_convention;
    const auto nc=count(1000);state.radial_channel_l.resize(nc);
    for(auto& l:state.radial_channel_l)f>>l;
    const auto matrix=[&](std::vector<Complex>& values) {
      const auto n=count(1000000);
      if(!state.projector_dimension||n!=state.projector_dimension*state.projector_dimension)
        throw std::runtime_error("invalid frozen PAW matrix dimensions");
      values.resize(n);
      for(auto& value:values){double re,im;f>>re>>im;if(!f||!std::isfinite(re)||!std::isfinite(im))throw std::runtime_error("nonfinite frozen PAW matrix");value={re,im};}
    };
    matrix(state.density_matrix);matrix(state.onsite_dij_ev);matrix(state.closure_correction_ev);matrix(state.compensation_correction_ev);
    const auto ni=count(1000000);state.lm_moment_imag.resize(ni);
    for(auto& value:state.lm_moment_imag)f>>value;
    f>>state.ae_energy_ev>>state.ps_energy_ev>>state.energy_difference_ev
     >>state.potential_expectation_ev>>state.core_xc_energy_ev;
    states.push_back(std::move(state));
  }
  };
  read_atoms(out.onecenter_states);
  if(v6){
    out.spin_onecenter_states[0]=out.onecenter_states;
    for(auto& q:out.spin_density_q)density(q);
    for(auto& q:out.spin_hamiltonian_density_q)density(q);
    read_atoms(out.spin_onecenter_states[1]);
    if(nk==0||nk%2!=0)throw std::runtime_error("spin frozen k count is invalid");
    double population=0.;
    for(int c=0;c<2;++c){double weight=0.;
      for(std::size_t k=c*nk/2;k<(c+1)*nk/2;++k){
        const auto& point=out.kpoints[k];weight+=point.weight;
        out.kpoint_spin_channels.push_back(c);
        for(double occ:out.kpoint_occupations[k]){
          if(occ>1.)throw std::runtime_error("spin frozen occupations exceed capacity one");
          population+=point.weight*occ;
        }
        if(c==1){const auto& up=out.kpoints[k-nk/2];
          if(norm(sub(up.fractional,point.fractional))>1e-12||std::abs(up.weight-point.weight)>1e-12)
            throw std::runtime_error("spin frozen sampling differs between channels");}
      }
      if(std::abs(weight-1.)>1e-12)throw std::runtime_error("spin frozen k weights must sum to one per channel");
    }
    if(std::abs(population-out.electron_count)>1e-6)throw std::runtime_error("spin frozen occupations do not conserve charge");
    const auto sum_matches=[](const std::array<FourierDensity,2>& pair,const FourierDensity& total){
      FourierDensity sum=pair[0];for(const auto& [key,value]:pair[1])sum[key]+=value;
      if(sum.size()!=total.size())throw std::runtime_error("spin frozen density supports differ");
      for(const auto& [key,value]:total){const auto it=sum.find(key);
        if(it==sum.end()||std::abs(it->second-value)>1e-12*std::max(1.,std::abs(value)))
          throw std::runtime_error("spin frozen densities do not sum to total charge");}
    };
    sum_matches(out.spin_density_q,out.density_q);
    sum_matches(out.spin_hamiltonian_density_q,out.hamiltonian_density_q);
    out.magnetic_moment=(out.spin_density_q[0][{0,0,0}]-out.spin_density_q[1][{0,0,0}]).real()*std::abs(determinant(input.lattice_bohr));
  }
  std::string end;f>>end;
  if(!f||end!="END"||(f>>end))throw std::runtime_error("truncated or trailing frozen SCF payload");
  // Older minimal JSON did not export frozen_core_q. Rebuild the identical
  // core from verified PAW files on the frozen density support in that case.
  if(out.frozen_core_q.empty()&&!v8&&!v10)out.frozen_core_q=pseudo_core_density(input,read_potentials(input),out.hamiltonian_density_q);
  const double electrons=std::accumulate(input.atoms.begin(),input.atoms.end(),0.,[](double n,const ScfAtom& a){return n+a.zval;});
  if(!std::isfinite(out.electron_count)||std::abs(out.electron_count-electrons)>1e-10||
     !std::isfinite(out.integrated_electron_count)||std::abs(out.integrated_electron_count-electrons)>1e-6)
    throw std::runtime_error("frozen SCF electron count differs from input valence");
  out.onecenter_mixing_method="frozen_converged_state";
  return out;
}

void write_projector_stress_diagnostic(const ScfInput& input,const ScfResult& scf,
                                     const std::filesystem::path& output) {
  if(scf.spin_channels!=1)throw std::invalid_argument("projector-only stress diagnostic is nonmagnetic; use full spin derivative validation");
  if(!scf.converged || !scf.onecenter_closure_diagnostic || scf.onecenter_states.size()!=input.atoms.size() ||
     scf.kpoints.empty() || scf.kpoint_eigenvalues_ev.size()!=scf.kpoints.size() ||
     scf.kpoint_occupations.size()!=scf.kpoints.size())
    throw std::invalid_argument("projector stress diagnostic requires audited converged PAW state");
  if(std::filesystem::exists(output))throw std::runtime_error("refusing to overwrite projector stress evidence");
  const auto units=atomic_units(input.unit_convention);
  const double volume_ang=std::abs(determinant(input.lattice_bohr))/std::pow(units.bohr_per_angstrom,3);
  const auto pots=read_potentials(input);
  const auto reciprocal_basis=reciprocal(input.lattice_bohr);
  PawScfClosureContext context;
  context.pseudo_core=scf.frozen_core_q;
  // Frozen on-site D_ij already contains the compensation adjoint. Reconstruct only
  // the overlap moments needed by H/S; no per-atom grid-shape cache is needed
  // for a nonlocal-projector derivative at fixed converged density.
  for(const auto& pot:pots) {
    std::vector<int> raw_l;
    for(const auto& ch:pot.cache.nonlocal_channels)for(int p=0;p<ch.nprojectors;++p)raw_l.push_back(ch.angular_momentum);
    context.augmentation.push_back(build_paw_augmentation(pot.cache,raw_l));
  }
  const auto functional=evaluate_density_functionals(scf.hamiltonian_density_q,input.lattice_bohr,context.pseudo_core);
  const std::array<double,4> steps{1e-4,1e-5,1e-6,1e-7};
  std::array<double,4> finite_cell_force{};
  double analytic_cell_force=0.,max_eigen_error=0.,max_residual=0.;
  const auto mesh=scf.kpoints;
  const auto bands=input.bands>0?input.bands:static_cast<int>(scf.kpoint_eigenvalues_ev.at(0).size());
  std::vector<std::string> warm_hashes;
  for(std::size_t ik=0;ik<mesh.size();++ik) {
    const auto basis=make_basis(input,reciprocal_basis,mesh[ik].fractional);
    std::vector<MatrixC> betas;
    for(std::size_t atom=0;atom<input.atoms.size();++atom)
      betas.push_back(atom_projector_basis(input,pots[atom],atom,basis,input.lattice_bohr));
    MatrixC overlap;
    std::unique_ptr<PlaneWaveOperator> op;
    double eh=0.,exc=0.,evxc=0.;
    auto unused_h=make_hamiltonian(input,pots,basis,input.lattice_bohr,scf.hamiltonian_density_q,
        nullptr,overlap,eh,exc,evxc,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        &scf.onecenter_states,nullptr,nullptr,&context,&functional,nullptr,&op,nullptr,&betas);
    if(!op)throw std::invalid_argument("projector stress diagnostic requires the production FFT operator");
    const auto solver=diagonalize(*op,bands,nullptr,1e-12);
    max_residual=std::max(max_residual,solver.max_residual);
    for(int band=0;band<bands;++band)max_eigen_error=std::max(max_eigen_error,
        std::abs(solver.eigenvalues[band]*units.ev_per_hartree-scf.kpoint_eigenvalues_ev.at(ik).at(band)));
    const auto snapshot=output.string()+".k"+std::to_string(ik)+".warm.pwop";
    std::vector<Complex> coefficients(basis.g.size()*bands);
    for(int band=0;band<bands;++band)for(std::size_t g=0;g<basis.g.size();++g)
      coefficients[band*basis.g.size()+g]=solver.eigenvectors(g,band);
    op->save_solver_snapshot(snapshot,bands,256,1e-12,coefficients);
    warm_hashes.push_back(sha256_file(snapshot));
    const auto& occupation=scf.kpoint_occupations.at(ik);
    const double weight=mesh[ik].weight/std::accumulate(mesh.begin(),mesh.end(),0.,[](double s,const auto& k){return s+k.weight;});
    for(std::size_t atom=0;atom<input.atoms.size();++atom) {
      const auto& cache=pots[atom].cache;const auto& beta=betas[atom];const std::size_t dim=beta.rows();
      MatrixC d(dim,dim),q=MatrixC::Zero(dim,dim);
      for(std::size_t i=0;i<dim;++i)for(std::size_t j=0;j<dim;++j)d(i,j)=scf.onecenter_states[atom].onsite_dij_ev[i*dim+j]/units.ev_per_hartree;
      std::size_t expanded=0,radial=0;
      for(const auto& ch:cache.nonlocal_channels) {
        const auto deg=2*ch.angular_momentum+1;
        for(int i=0;i<ch.nprojectors;++i)for(int j=0;j<ch.nprojectors;++j)for(int m=0;m<deg;++m)
          q(expanded+i*deg+m,expanded+j*deg+m)=context.augmentation[atom].overlap_moment(radial+i,radial+j);
        expanded+=ch.nprojectors*deg;radial+=ch.nprojectors;
      }
      const MatrixC projected=beta*solver.eigenvectors;
      const auto contract=[&](const MatrixC& derivative) {
        const MatrixC response=derivative*solver.eigenvectors;double value=0.;
        for(int band=0;band<bands;++band)value+=2*occupation[band]*
            response.col(band).dot((d-solver.eigenvalues[band]*q)*projected.col(band)).real();
        return -weight*units.ev_per_hartree*value;
      };
      MatrixC analytic=beta*(-.5);
      const double factor=(std::getenv("SDFT_PROJECTOR_SCALE")?std::atof(std::getenv("SDFT_PROJECTOR_SCALE")):1.)/std::sqrt(volume_ang);
      const double length=std::getenv("SDFT_PROJECTOR_Q_BOHR")?1.:units.bohr_per_angstrom;
      std::size_t row=0;
      for(const auto& ch:cache.nonlocal_channels)for(int p=0;p<ch.nprojectors;++p)
        for(int m=-ch.angular_momentum;m<=ch.angular_momentum;++m,++row)for(std::size_t g=0;g<basis.g.size();++g) {
          const auto wave=basis.cart[g];const double radius=norm(wave);
          const double qmax=ch.reciprocal_cutoff>0?ch.reciprocal_cutoff:twopi/std::max(ch.cutoff_radius,1e-8);
          const auto sample=projector_sample(pots[atom],ch.reciprocal_projectors[p],radius*length,qmax,ch.angular_momentum);
          const auto value=sample.value,derivative=sample.derivative*length;
          const auto dy=paw_real_spherical_harmonic_gradient(ch.angular_momentum,m,wave);
          const auto phase=factor*std::pow(Complex(0,1),ch.angular_momentum)*
              std::exp(Complex(0,dot(wave,cartesian(input.lattice_bohr,input.atoms[atom].fractional))));
          analytic(row,g)-=wave.y*phase*(value*dy.y+(radius>1e-14?
              derivative*wave.y/radius*spherical_harmonic(ch.angular_momentum,m,wave):0.));
        }
      analytic_cell_force+=contract(analytic);
      for(std::size_t step=0;step<steps.size();++step) {
        std::array<MatrixC,2> strained;
        for(int side=0;side<2;++side) {
          const auto strain=(side? -1.:1.)*steps[step];auto lattice=input.lattice_bohr;
          for(auto& v:lattice)v.y*=1.+strain;
          Basis strained_basis=basis;
          for(auto& v:strained_basis.cart)v.y/=1.+strain;
          strained[side]=atom_projector_basis(input,pots[atom],atom,strained_basis,lattice);
        }
        finite_cell_force[step]+=contract((strained[0]-strained[1])/(2*steps[step]));
      }
    }
  }
  std::ofstream file(output.string()+".tmp");file<<std::setprecision(17);
  file<<"{\"schema_version\":\"sparse-dft-projector-stress-diagnostic-1\",\"scientific_acceptance\":false,"
      <<"\"scope\":\"yy nonlocal projector response at fixed converged H/S, occupations and G indices\","
      <<"\"volume_angstrom3\":"<<volume_ang<<",\"maximum_eigenvalue_difference_ev\":"<<max_eigen_error
      <<",\"fresh_residual_hartree\":"<<max_residual<<",\"analytic_cell_force_ev\":"<<analytic_cell_force
      <<",\"finite_displacement\":[";
  for(std::size_t i=0;i<steps.size();++i)file<<(i?",":"")<<"{\"step\":"<<steps[i]
      <<",\"cell_force_ev\":"<<finite_cell_force[i]<<",\"difference_from_analytic_gpa\":"
      <<-(finite_cell_force[i]-analytic_cell_force)/volume_ang*160.2176634<<"}";
  file<<"],\"warm_snapshot_sha256\":[";
  for(std::size_t i=0;i<warm_hashes.size();++i)file<<(i?",":"")<<'"'<<warm_hashes[i]<<'"';
  file<<"]}\n";file.close();if(!file)throw std::runtime_error("projector stress diagnostic write failed");
  std::filesystem::rename(output.string()+".tmp",output);
}

BandPathResult run_nscf(const ScfInput& input, const ScfResult& scf,
                        const std::vector<Vec3>& kpoints_fractional, int bands) {
  if(input.spin_channels!=scf.spin_channels)throw std::invalid_argument("NSCF spin model differs from frozen SCF");
  if(scf.spin_channels==2){
    PerformanceTimer timer("collinear_nscf_total");
    if(!scf.converged||!scf.onecenter_closure_diagnostic||scf.hubbard_specs.size()!=input.atoms.size())
      throw std::invalid_argument("spin NSCF requires complete converged spin PAW/model state");
    for(std::size_t a=0;a<input.atoms.size();++a)
      if(input.atoms[a].hubbard.angular_momentum!=scf.hubbard_specs[a].angular_momentum||
         input.atoms[a].hubbard.ueff_ev!=scf.hubbard_specs[a].ueff_ev)
        throw std::invalid_argument("NSCF Hubbard model differs from frozen SCF");
    const auto functional=evaluate_collinear_density_functionals(scf.spin_hamiltonian_density_q,
        input.lattice_bohr,&scf.frozen_core_q,0);
    ScfInput channel_input=input;channel_input.spin_channels=1;
    std::array<BandPathResult,2> channels;
    for(int c=0;c<2;++c){
      ScfResult view=scf;view.spin_channels=1;view.onecenter_states=scf.spin_onecenter_states[c];
      view.hamiltonian_density_q=scf.spin_hamiltonian_density_q[c];view.density_q=scf.spin_density_q[c];
      view.frozen_channel_functional=functional[c];
      for(auto& [key,value]:view.frozen_core_q)value*=.5;
      channels[c]=run_nscf(channel_input,view,kpoints_fractional,bands);
    }
    auto out=std::move(channels[0]);out.spin_channels=2;out.spin_projection_available=true;
    out.spin_eigenvalues_ev[0]=out.eigenvalues_ev;out.spin_eigenvalues_ev[1]=std::move(channels[1].eigenvalues_ev);
    out.spin_projector_weights[0]=out.projector_weights;out.spin_projector_weights[1]=std::move(channels[1].projector_weights);
    out.spin_projector_lm_weights[0]=out.projector_lm_weights;out.spin_projector_lm_weights[1]=std::move(channels[1].projector_lm_weights);
    out.spin_residuals[0]=out.eigensolver_max_residuals_hartree;out.spin_residuals[1]=std::move(channels[1].eigensolver_max_residuals_hartree);
    out.spin_onecenter_states=scf.spin_onecenter_states;
    for(std::size_t k=0;k<out.eigensolver_max_residuals_hartree.size();++k)
      out.eigensolver_max_residuals_hartree[k]=std::max(out.spin_residuals[0][k],out.spin_residuals[1][k]);
    out.onecenter_dij_applications+=channels[1].onecenter_dij_applications;
    out.warm_start_kpoints+=channels[1].warm_start_kpoints;out.identical_k_spectrum_reuses+=channels[1].identical_k_spectrum_reuses;
    out.max_xc_grid_points=std::max(out.max_xc_grid_points,channels[1].max_xc_grid_points);
    return out;
  }
  if(use_shared_site_kernel(input)||(scf.onecenter_radial_mode=="per_site_original_variational_paw_development" || scf.onecenter_radial_mode=="per_site_original_energy_adjoint_paw_development"))return run_upf_nscf(input,scf,kpoints_fractional,bands);
  PerformanceTimer total_timer("nscf_total");
  const auto units = atomic_units(input.unit_convention);
  const double bohr_per_angstrom = units.bohr_per_angstrom;
  const double ev_per_hartree = units.ev_per_hartree;
  if (!scf.converged) throw std::invalid_argument("NSCF requires a converged SCF result");
  if(scf.projector_interpolation!=input.projector_interpolation)
    throw std::invalid_argument("NSCF projector interpolation differs from frozen SCF");
  if(scf.unit_convention!=input.unit_convention)throw std::invalid_argument("NSCF atomic units differ from frozen SCF");
  if (!std::isfinite(input.density_cutoff_ev) || input.density_cutoff_ev<0 ||
      (input.density_cutoff_ev>0 && std::abs(input.density_cutoff_ev-scf.density_cutoff_ev)>1e-10))
    throw std::invalid_argument("NSCF density_cutoff_ev differs from frozen SCF");
  BandPathResult out;
  out.kpoints = kpoints_fractional;
  out.nkpoints = kpoints_fractional.size();
  out.bands = bands > 0 ? bands : scf.occupied_bands + 4;
  out.fermi_energy_ev = scf.fermi_energy_ev;
  const auto b = reciprocal(input.lattice_bohr);
  const auto pots = read_potentials(input);
  std::set<int> angular_momenta;
  for(const auto& pot:pots)for(const auto& ch:pot.cache.nonlocal_channels)angular_momenta.insert(ch.angular_momentum);
  // paw_real_spherical_harmonic runs m=-l..l with sin(|m|phi) for m<0 and
  // cos(m phi) for m>0, which is VASP's PROCAR lm order for l<=3.
  static const std::map<int,std::vector<std::string>> lm_labels{
      {0,{"s"}},{1,{"py","pz","px"}},{2,{"dxy","dyz","dz2","dxz","dx2-y2"}},
      {3,{"fy3x2","fxyz","fyz2","fz3","fxz2","fzx2","fx3"}}};
  std::map<int,std::size_t> orbital_index,lm_offset;
  for(int l:angular_momenta) {
    orbital_index[l]=out.projection_orbitals.size();
    out.projection_orbitals.push_back(l==0?"s":l==1?"p":l==2?"d":l==3?"f":"l="+std::to_string(l));
    lm_offset[l]=out.projection_lm_orbitals.size();
    for(int m=-l;m<=l;++m)
      out.projection_lm_orbitals.push_back(l<=3?lm_labels.at(l)[std::size_t(m+l)]:"l="+std::to_string(l)+",m="+std::to_string(m));
  }
  for(const auto& atom:input.atoms)out.projection_atom_symbols.push_back(atom.symbol);
  // lm weights are the AE-sphere charge reported by VASP PROCAR for
  // LORBIT=11: w_lm = sum_ij conj(c_i,lm) Q_ij c_j,lm with
  // Q_ij = int_0^r_dep phi^AE_i phi^AE_j over same-l channels, integrated by
  // Simpson's rule on the mesh truncated at r_dep (see paw_sphere_ae_overlap).
  std::vector<std::vector<double>> sphere_overlap(input.atoms.size());
  try {
    for(std::size_t atom=0;atom<input.atoms.size();++atom) {
      std::vector<int> raw_l;
      for(const auto& ch:pots[atom].cache.nonlocal_channels)
        for(int i=0;i<ch.nprojectors;++i)raw_l.push_back(ch.angular_momentum);
      sphere_overlap[atom]=paw_sphere_ae_overlap(pots[atom].cache,raw_l);
    }
    out.projection_lm_weight_definition="vasp_lorbit11_ae_sphere_rdep_sum_ij_conj_ci_Qij_cj";
  } catch(const std::invalid_argument&) {
    // Incomplete AE radial data: no physical lm weights rather than raw ones.
    sphere_overlap.clear();out.projection_lm_orbitals.clear();lm_offset.clear();
  }
  const std::size_t stride=std::size_t(out.bands)*input.atoms.size()*out.projection_orbitals.size();
  const std::size_t lm_stride=std::size_t(out.bands)*input.atoms.size()*out.projection_lm_orbitals.size();
  PawScfClosureContext frozen_context;
  frozen_context.pseudo_core = scf.frozen_core_q;
  const std::vector<ScfOneCenterState>* frozen_onecenter = nullptr;
  if (scf.onecenter_closure_diagnostic) {
    if (std::getenv("SDFT_DISABLE_NONLOCAL"))
      throw std::invalid_argument("diagnostic NSCF cannot disable the frozen PAW nonlocal operator");
    if (scf.total_energy_valid)
      throw std::invalid_argument("diagnostic PAW on-site D_ij cannot be marked as a validated total energy");
    if (scf.onecenter_states.size() != input.atoms.size() || scf.hamiltonian_density_q.empty())
      throw std::invalid_argument("diagnostic NSCF requires the frozen SCF atom on-site D_ij and Hamiltonian density");
    for (std::size_t axis = 0; axis < 3; ++axis)
      if (norm(sub(input.lattice_bohr[axis], scf.onecenter_lattice_bohr[axis])) > 1e-12)
        throw std::invalid_argument("NSCF lattice differs from frozen PAW SCF lattice");
    const double projector_scale = std::getenv("SDFT_PROJECTOR_SCALE") ? std::atof(std::getenv("SDFT_PROJECTOR_SCALE")) : 1.0;
    frozen_context.compensation.reserve(input.atoms.size());
    for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
      const auto& state = scf.onecenter_states[atom];
      if (state.atom_index != atom || state.symbol != input.atoms[atom].symbol ||
          norm(sub(state.fractional, input.atoms[atom].fractional)) > 1e-12 ||
          state.paw_cache.lexically_normal() != input.atoms[atom].paw_cache.lexically_normal() ||
          state.numeric_cache_version != pots[atom].cache.version ||
          state.numeric_cache_checksum != paw_numeric_checksum(pots[atom].cache) ||
          state.numeric_cache_sha256 != pots[atom].numeric_sha256 ||
          state.unit_convention != input.unit_convention ||
          state.projector_scale != projector_scale ||
          state.projector_q_bohr != (std::getenv("SDFT_PROJECTOR_Q_BOHR") != nullptr) ||
          state.reciprocal_linear != (std::getenv("SDFT_RECIPROCAL_LINEAR") != nullptr))
        throw std::invalid_argument("NSCF atom, PAW cache, or projector convention differs from frozen SCF on-site D_ij");
      const auto dim = state.projector_dimension;
      if (!dim || state.onsite_dij_ev.size() != dim * dim || state.density_matrix.size() != dim * dim ||
          state.closure_correction_ev.size() != dim * dim)
        throw std::invalid_argument("frozen PAW on-site D_ij state has invalid dimensions");
      for (std::size_t i = 0; i < dim; ++i) for (std::size_t j = 0; j < dim; ++j) {
        const auto value = state.onsite_dij_ev[i * dim + j];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
            std::abs(value - std::conj(state.onsite_dij_ev[j * dim + i])) > 1e-10 * std::max(1.0, std::abs(value)))
          throw std::invalid_argument("frozen PAW on-site D_ij is non-finite or non-Hermitian");
      }
    }
    frozen_onecenter = &scf.onecenter_states;
    out.onecenter_states = scf.onecenter_states;
    out.onecenter_dij_reused = true;

    // The frozen NSCF operator uses the same PAW closure as the converged SCF:
    // on-site D_ij is frozen, but the compensation adjoint still needs the
    // exact augmentation moments and compensation shapes and the atomic
    // reference potentials that were used when D_ij was formed.  Reconstruct
    // these density-independent objects from
    // the verified numeric cache instead of carrying a partial context with
    // only pseudo-core data (which used to cause vector::at() in
    // assemble_nonlocal()).
    const auto density_keys = [&]() {
      std::vector<std::tuple<int, int, int>> keys;
      keys.reserve(scf.hamiltonian_density_q.size());
      for (const auto& [key, value] : scf.hamiltonian_density_q) {
        (void)value;
        keys.push_back(key);
      }
      return keys;
    }();
    for (std::size_t atom = 0; atom < input.atoms.size(); ++atom) {
      std::vector<int> raw_l;
      for (const auto& channel : pots[atom].cache.nonlocal_channels)
        for (int projector = 0; projector < channel.nprojectors; ++projector)
          raw_l.push_back(channel.angular_momentum);
      auto augmentation = build_paw_augmentation(pots[atom].cache, raw_l);
      frozen_context.references.push_back(
          paw_atomic_reference_potentials(pots[atom].cache, raw_l,
              PawRadialLengthUnit::Angstrom, scf.onecenter_radial_mode=="aspherical_gga" ?
                PawRadialPotentialMode::AsphericalDirect : PawRadialPotentialMode::RadialDirect, input.unit_convention));
      frozen_context.augmentation.push_back(augmentation);
      frozen_context.compensation.emplace_back(
          frozen_context.augmentation.back(), input.lattice_bohr,
          input.atoms[atom].fractional, density_keys, input.unit_convention,
          [&]() -> const PawCompensationOperator* {
            if(share_compensation_geometry())for(std::size_t i=0;i<atom;++i)
              if(pots[i].numeric_sha256==pots[atom].numeric_sha256)return &frozen_context.compensation[i];
            return nullptr;
          }());
    }
  }
  std::optional<DensityFunctionals> path_functional;
  if(scf.frozen_channel_functional)path_functional=scf.frozen_channel_functional;
  const bool use_fft=matrix_free_operator();
  out.solver_method=use_fft ? "matrix_free_fft_block_davidson" :
      (std::getenv("SDFT_SOLVER") ? std::getenv("SDFT_SOLVER") : "block_davidson_above_dense_limit");
  const auto& path_density=frozen_onecenter ? scf.hamiltonian_density_q : scf.density_q;
  if(!path_functional&&!std::getenv("SDFT_DISABLE_HARTREE_XC"))
    path_functional=evaluate_density_functionals(path_density,input.lattice_bohr,frozen_context.pseudo_core);
  if(path_functional)out.max_xc_grid_points=path_functional->grid_points;
  const bool warm_path=density_dataflow_control("SDFT_NSCF_WARM_START");
  const bool cache_path=density_dataflow_control("SDFT_NSCF_CACHE_IDENTICAL_K");
  const bool reuse_setup=density_dataflow_control("SDFT_NSCF_FROZEN_SETUP");
  FourierDensity path_ionic;
  if(reuse_setup && frozen_onecenter && path_functional && !frozen_nscf_fields() &&
      !std::getenv("SDFT_DISABLE_LOCAL")) {
    PerformanceTimer timer("nscf_frozen_ionic_setup");
    std::vector<std::tuple<int,int,int>> keys;
    keys.reserve(path_functional->potential.size());
    for(const auto& [key,value]:path_functional->potential) {
      (void)value;keys.push_back(key);
    }
    // Positions, cell, potentials and Fourier support are unchanged across
    // this one path. These values are independent of Bloch k; on-site D_ij is still
    // the identical frozen SCF state and each k gets new projectors and H/S.
    path_ionic=local_ionic_potential(input,pots,input.lattice_bohr,b,keys);
  }
  std::map<std::array<int,3>,FourierDensity> path_static_ionic;
  MatrixC previous,previous_guards;
  std::vector<GVector> previous_g;
  struct Spectrum {std::vector<double> ev,projections,lm_projections;double residual;};
  std::map<std::tuple<double,double,double>,Spectrum> spectrum_cache;
  for (const auto& k : kpoints_fractional) {
    if(!std::isfinite(k.x)||!std::isfinite(k.y)||!std::isfinite(k.z))
      throw std::invalid_argument("NSCF k-point must be finite");
    const auto cache_key=std::tuple{k.x,k.y,k.z};
    if(cache_path)if(const auto cached=spectrum_cache.find(cache_key);cached!=spectrum_cache.end()) {
      out.eigenvalues_ev.insert(out.eigenvalues_ev.end(),cached->second.ev.begin(),cached->second.ev.end());
      out.projector_weights.insert(out.projector_weights.end(),cached->second.projections.begin(),cached->second.projections.end());
      out.projector_lm_weights.insert(out.projector_lm_weights.end(),cached->second.lm_projections.begin(),cached->second.lm_projections.end());
      out.eigensolver_max_residuals_hartree.push_back(cached->second.residual);
      out.seed_transport_retained_fraction.push_back(0.);
      ++out.identical_k_spectrum_reuses;record_performance("nscf_identical_k_spectrum_reuse",0);
      continue;
    }
    const auto basis = make_basis(input, b, k);
    if (basis.g.size() < static_cast<std::size_t>(out.bands))
      throw std::invalid_argument("NSCF k-point has fewer plane waves than requested bands; increase cutoff or reduce bands");
    MatrixC overlap;
    std::unique_ptr<PlaneWaveOperator> fft_operator;
    double eh = 0.0, exc = 0.0, evxc = 0.0;
    const auto& frozen_charge = frozen_onecenter ? scf.hamiltonian_density_q : scf.density_q;
    const FourierDensity* static_ionic=nullptr;
    if(reuse_setup && use_fft) {
      int lo[3]{basis.g[0].h,basis.g[0].k,basis.g[0].l},hi[3]{lo[0],lo[1],lo[2]};
      for(const auto& g:basis.g) {
        const int indices[]{g.h,g.k,g.l};
        for(int axis=0;axis<3;++axis){lo[axis]=std::min(lo[axis],indices[axis]);hi[axis]=std::max(hi[axis],indices[axis]);}
      }
      const std::array<int,3> support{hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]};
      auto found=path_static_ionic.find(support);
      if(found==path_static_ionic.end()) {
        PerformanceTimer timer("nscf_static_ionic_setup");
        found=path_static_ionic.emplace(support,static_local_fourier(input,pots,basis)).first;
      } else record_performance("nscf_static_ionic_cache_hit",0);
      static_ionic=&found->second;
    }
    MatrixC h;
    {
    PerformanceTimer timer("nscf_operator_setup");
    h = make_hamiltonian(input, pots, basis, input.lattice_bohr, frozen_charge,
                              nullptr, overlap, eh, exc, evxc, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, frozen_onecenter, nullptr, nullptr,
                              frozen_onecenter ? &frozen_context : nullptr, path_functional ? &*path_functional : nullptr,
                              nullptr,use_fft ? &fft_operator : nullptr,static_ionic,nullptr,
                              !path_ionic.empty() ? &path_ionic : nullptr);
    }
    if (frozen_onecenter) out.onecenter_dij_applications += frozen_onecenter->size();
    if(fft_operator) {
      out.max_fft_grid_points=std::max(out.max_fft_grid_points,fft_operator->fft_grid_points());
      out.max_operator_storage_bytes=std::max(out.max_operator_storage_bytes,fft_operator->storage_bytes());
    } else {
      out.max_operator_storage_bytes=std::max(out.max_operator_storage_bytes,
          static_cast<std::size_t>(h.size()+overlap.size())*sizeof(Complex));
    }
    MatrixC guess,guards;double retained=0.;
    if(warm_path&&previous.size()) {
      PerformanceTimer transport_timer("nscf_seed_transport");
      const auto transported=transport_plane_wave_seeds(previous_g,basis.g,
          std::vector<Complex>(previous.data(),previous.data()+previous.size()),out.bands);
      retained=transported.minimum_retained_norm_fraction;
      // Very different cutoffs/k-points can lose the useful manifold. The
      // original cold global search remains available without changing H/S.
      if(retained>=.5) {
        guess=Eigen::Map<const MatrixC>(transported.coefficients.data(),basis.g.size(),out.bands);
        if(previous_guards.size()) {
          const auto mapped=transport_plane_wave_seeds(previous_g,basis.g,
              std::vector<Complex>(previous_guards.data(),previous_guards.data()+previous_guards.size()),previous_guards.cols());
          if(mapped.minimum_retained_norm_fraction>=.5)
            guards=Eigen::Map<const MatrixC>(mapped.coefficients.data(),basis.g.size(),previous_guards.cols());
        }
        ++out.warm_start_kpoints;record_performance("nscf_warm_start_kpoint",0);
      }
    }
    const MatrixC* seed=guess.size()?&guess:nullptr;
    // Same strict original H/S gate for warm and cold paths. A seed never
    // bypasses canonical guards, spectral completeness, or FP64 acceptance.
    auto solver = use_fft ? diagonalize(*fft_operator,out.bands,seed,1e-12,&guards)
                          : diagonalize(h,overlap,out.bands,seed,1e-12);
    if(solver.max_residual>1e-12||solver.trial_manifold||!solver.converged)
      throw std::runtime_error("NSCF strict original residual or lowest-spectrum gate failed");
    out.seed_transport_retained_fraction.push_back(retained);
    if(warm_path) {previous=solver.eigenvectors;previous_guards=solver.search_guards;previous_g=basis.g;}
    std::vector<double> weights(stride,0.),lm_weights(lm_stride,0.);
    {
      PerformanceTimer timer("nscf_projection_output");
      // Reuse the immutable B arrays already stored by the operator. A
      // fresh FP64 B*psi is still required for the accepted final vectors;
      // this avoids rebuilding every atomic B solely for output.
      MatrixC all_projected;
      if(fft_operator) {
        auto op=fft_operator->hamiltonian();
        if(op.project_into) {
          all_projected.resize(op.projector_rows,out.bands);
          op.project_into(solver.eigenvectors.data(),out.bands,all_projected.data(),true);
          record_performance("nscf_projection_reuses_operator_geometry",0);
        }
      }
      std::size_t atom_offset=0;
      for(std::size_t atom=0;atom<input.atoms.size();++atom) {
        std::size_t atom_rows=0;
        for(const auto& ch:pots[atom].cache.nonlocal_channels)
          atom_rows+=ch.nprojectors*(2*ch.angular_momentum+1);
        MatrixC projected;
        if(all_projected.size()) {
          if(atom_offset+atom_rows>std::size_t(all_projected.rows()))throw std::runtime_error("NSCF projector row layout mismatch");
          projected=all_projected.middleRows(atom_offset,atom_rows);
        } else {
          const MatrixC beta=atom_projector_basis(input,pots[atom],atom,basis,input.lattice_bohr);
          projected=beta*solver.eigenvectors;
        }
        atom_offset+=atom_rows;
        // Projector rows run channel -> radial -> m; record each radial
        // projector's (l, first row) for the same-l cross terms below.
        std::vector<std::pair<int,std::size_t>> radial_rows;
        std::size_t row=0;
        for(const auto& ch:pots[atom].cache.nonlocal_channels) {
          const auto orbital=orbital_index.at(ch.angular_momentum);
          for(int radial=0;radial<ch.nprojectors;++radial) {
            radial_rows.emplace_back(ch.angular_momentum,row);
            for(int m=-ch.angular_momentum;m<=ch.angular_momentum;++m,++row)
              for(int band=0;band<out.bands;++band)
                weights[(std::size_t(band)*input.atoms.size()+atom)*out.projection_orbitals.size()+orbital]+=std::norm(projected(row,band));
          }
        }
        if(!sphere_overlap.empty()) {
          const auto& q=sphere_overlap[atom];const std::size_t nr=radial_rows.size();
          for(std::size_t i=0;i<nr;++i)for(std::size_t j=0;j<nr;++j) {
            const auto [l,row_i]=radial_rows[i];
            if(radial_rows[j].first!=l||q[i*nr+j]==0.)continue;
            const std::size_t row_j=radial_rows[j].second,lm_base=lm_offset.at(l);
            for(int m=0;m<=2*l;++m)
              for(int band=0;band<out.bands;++band)
                lm_weights[(std::size_t(band)*input.atoms.size()+atom)*out.projection_lm_orbitals.size()+lm_base+std::size_t(m)]+=
                    q[i*nr+j]*(std::conj(projected(row_i+m,band))*projected(row_j+m,band)).real();
          }
        }
      }
    }
    out.projector_weights.insert(out.projector_weights.end(),weights.begin(),weights.end());
    out.projector_lm_weights.insert(out.projector_lm_weights.end(),lm_weights.begin(),lm_weights.end());
    if(cache_path) {
      Spectrum cached;cached.projections=std::move(weights);cached.lm_projections=std::move(lm_weights);cached.residual=solver.max_residual;
      for(int band=0;band<out.bands;++band)cached.ev.push_back(solver.eigenvalues[band]*ev_per_hartree);
      spectrum_cache.emplace(cache_key,std::move(cached));
    }
    out.eigensolver_max_residuals_hartree.push_back(solver.max_residual);
    if (std::getenv("SDFT_TRACE_BANDS")) {
      std::cerr << "BAND k " << k.x << " " << k.y << " " << k.z
                << " G0 " << basis.cart.front().x << " " << basis.cart.front().y << " " << basis.cart.front().z
                << " E0 " << solver.eigenvalues[0] * ev_per_hartree
                << " E4 " << solver.eigenvalues[std::min<int>(4, solver.eigenvalues.size()-1)] * ev_per_hartree << "\n";
    }
    const int available = static_cast<int>(basis.g.size());
    const int returned = std::min(out.bands, available);
    for (int band = 0; band < returned; ++band) out.eigenvalues_ev.push_back(solver.eigenvalues[band] * ev_per_hartree);
    for (int band = returned; band < out.bands; ++band) out.eigenvalues_ev.push_back(std::numeric_limits<double>::quiet_NaN());
  }
  return out;
}


std::string density_spectrum_probe_json(const ScfInput& input) {
  // A-priori spectral tails of the atom-centred hard densities (partial
  // pseudo-core and compensation shapes g_L) that set the density support.
  // Diagnostic only: it reads the same PAW tables as the SCF but changes
  // nothing. Tails are fractions of the power up to probe_factor*ENCUT.
  const auto units=atomic_units(input.unit_convention);
  const auto pots=read_potentials(input);
  const auto q_of=[&](double ev){return std::sqrt(2.*ev/units.ev_per_hartree)*units.bohr_per_angstrom;};
  // Tail factors (multiples of ENCUT). SDFT_SPARSITY_PROBE_FACTORS="4,5,7"
  // replaces the default list; each must lie in [1, probe_factor).
  const double probe_factor=24.;
  std::vector<double> factors{4,5,6,8,10,12,16};
  if(const char* setting=std::getenv("SDFT_SPARSITY_PROBE_FACTORS")) {
    factors.clear();std::stringstream list(setting);std::string item;
    while(std::getline(list,item,',')) {
      std::size_t end=0;double f=std::numeric_limits<double>::quiet_NaN();
      try{f=std::stod(item,&end);}catch(const std::exception&){}
      if(end!=item.size()||!std::isfinite(f)||f<1.||f>=probe_factor)
        throw std::invalid_argument("SDFT_SPARSITY_PROBE_FACTORS entries must be numbers in [1,24)");
      factors.push_back(f);
    }
    std::sort(factors.begin(),factors.end());
    factors.erase(std::unique(factors.begin(),factors.end()),factors.end());
    if(factors.empty()||factors.size()>32)throw std::invalid_argument("SDFT_SPARSITY_PROBE_FACTORS needs 1..32 entries");
  }
  const double qtop=q_of(probe_factor*input.cutoff_ev);
  constexpr int nq=3072;const double dq=qtop/nq;
  std::ostringstream out;out<<std::setprecision(10);
  out<<"{\"schema_version\":\"sparse-dft-density-spectrum-probe-1\",\"cutoff_ev\":"<<input.cutoff_ev
     <<",\"probe_factor\":"<<probe_factor<<",\"factors\":[";
  for(std::size_t i=0;i<factors.size();++i)out<<(i?",":"")<<factors[i];
  out<<"],\"species\":[";
  std::set<std::string> seen;bool first=true;
  for(std::size_t atom=0;atom<pots.size();++atom) {
    const auto& pot=pots[atom];
    if(!seen.insert(pot.numeric_sha256).second)continue;
    const auto& c=pot.cache;
    std::vector<int> raw_l;
    for(const auto& channel:c.nonlocal_channels)for(int i=0;i<channel.nprojectors;++i)raw_l.push_back(channel.angular_momentum);
    const auto aug=build_paw_augmentation(c,raw_l);
    const std::size_t nl=static_cast<std::size_t>(aug.max_l+1);
    // Absolute weights: |rho_LM| of the free-atom reference occupation for
    // each L (compensation) and ZVAL-normalised core charge. The unweighted
    // tails compare shapes; the weighted ones compare actual charge.
    std::size_t expanded=0;for(int l:raw_l)expanded+=static_cast<std::size_t>(2*l+1);
    const auto lm_moments=paw_lm_moments_from_density(aug,paw_atomic_reference_density(c,raw_l),expanded);
    std::vector<double> weight(nl+1,0.);
    for(std::size_t L=0;L<nl;++L){double w=0;for(int M=-int(L);M<=int(L);++M){const auto i=L*L+std::size_t(M+int(L));
      if(i<lm_moments.size())w+=std::norm(lm_moments[i]);}weight[L+1]=std::sqrt(w);}
    // Cumulative power: [component][q] for L2 (|f|^2 q^2) and Hartree (|f|^2) weights.
    std::vector<double> absolute_tail(nl+1,0.);
    std::vector<std::vector<double>> l2(nl+1,std::vector<double>(nq+1,0.)),hartree=l2,amplitude=l2;
    for(int i=0;i<nq;++i) {
      const double q=(i+.5)*dq;
      std::vector<double> f(nl+1,0.);
      if(!c.core_charge_partial.empty()&&!c.local_header.empty())
        f[0]=density_table_sample(c.core_charge_partial,q,c.local_header.front()).value;
      const auto g=paw_compensation_radial_form(aug,q);
      for(std::size_t L=0;L<nl;++L)f[L+1]=g[L];
      for(std::size_t m=0;m<=nl;++m) {
        l2[m][i+1]=l2[m][i]+f[m]*f[m]*q*q*dq;
        hartree[m][i+1]=hartree[m][i]+f[m]*f[m]*dq;
        amplitude[m][i]=std::abs(f[m]);
      }
    }
    // Suffix maxima of |f| for the amplitude criterion.
    for(std::size_t m=0;m<=nl;++m)for(int i=nq-1;i>0;--i)amplitude[m][i-1]=std::max(amplitude[m][i-1],amplitude[m][i]);
    const auto absolute=[&](const std::vector<double>& cumulative,double factor) {
      const int i=std::min(nq,static_cast<int>(std::floor(q_of(factor*input.cutoff_ev)/dq)));
      return cumulative[nq]-cumulative[i];
    };
    const auto tail=[&](const std::vector<double>& cumulative,double factor) {
      const int i=std::min(nq,static_cast<int>(std::floor(q_of(factor*input.cutoff_ev)/dq)));
      return cumulative[nq]>0 ? (cumulative[nq]-cumulative[i])/cumulative[nq] : 0.;
    };
    const auto amp=[&](const std::vector<double>& a,double factor) {
      const int i=std::min(nq-1,static_cast<int>(std::floor(q_of(factor*input.cutoff_ev)/dq)));
      return a[0]>0 ? a[i]/a[0] : 0.;
    };
    out<<(first?"":",")<<"{\"record_id\":"<<std::quoted(pot.record_id)<<",\"zval\":"<<pot.zval
       <<",\"has_partial_core\":"<<(c.core_charge_partial.empty()?"false":"true")
       <<",\"max_l\":"<<aug.max_l<<",\"reference_multipoles\":[";
    for(std::size_t L=0;L<nl;++L)out<<(L?",":"")<<weight[L+1];
    out<<"],\"tails\":[";
    first=false;
    for(std::size_t k=0;k<factors.size();++k) {
      double comp_l2=0,comp_h=0,comp_amp=0,weighted_h=0,weighted_l2=0;
      for(std::size_t L=1;L<=nl;++L){comp_l2=std::max(comp_l2,tail(l2[L],factors[k]));
        comp_h=std::max(comp_h,tail(hartree[L],factors[k]));comp_amp=std::max(comp_amp,amp(amplitude[L],factors[k]));
        weighted_h+=weight[L]*weight[L]*absolute(hartree[L],factors[k]);
        weighted_l2+=weight[L]*weight[L]*absolute(l2[L],factors[k]);}
      out<<(k?",":"")<<"{\"factor\":"<<factors[k]
         <<",\"core_l2\":"<<tail(l2[0],factors[k])<<",\"core_hartree\":"<<tail(hartree[0],factors[k])
         <<",\"core_amplitude\":"<<amp(amplitude[0],factors[k])
         <<",\"compensation_l2\":"<<comp_l2<<",\"compensation_hartree\":"<<comp_h
         <<",\"compensation_amplitude\":"<<comp_amp
         <<",\"weighted_compensation_hartree\":"<<weighted_h
         <<",\"weighted_compensation_l2\":"<<weighted_l2
         <<",\"absolute_core_hartree\":"<<absolute(hartree[0],factors[k])
         <<",\"absolute_core_l2\":"<<absolute(l2[0],factors[k])<<"}";
    }
    out<<"]}";
  }
  out<<"]}\n";
  return out.str();
}

}  // namespace sdft
