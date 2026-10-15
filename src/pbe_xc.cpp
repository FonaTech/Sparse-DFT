#include "pbe_xc.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

namespace sdft {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
constexpr double gamma = 0.031090690869654895;
constexpr double beta = 0.06672455060314922;
constexpr double mu = 0.2195149727645171;
constexpr double kappa = 0.804;

// Forward-mode derivatives with respect to density and |grad(n)|^2 keep the
// GGA potential consistent with the energy density without finite-difference
// noise in the SCF loop.
struct Dual {
  double value{};
  double dn{};
  double dgrad2{};
};

Dual operator+(Dual a, Dual b) { return {a.value + b.value, a.dn + b.dn, a.dgrad2 + b.dgrad2}; }
Dual operator-(Dual a, Dual b) { return {a.value - b.value, a.dn - b.dn, a.dgrad2 - b.dgrad2}; }
Dual operator-(Dual a) { return {-a.value, -a.dn, -a.dgrad2}; }
Dual operator*(Dual a, Dual b) {
  return {a.value * b.value, a.dn * b.value + a.value * b.dn,
          a.dgrad2 * b.value + a.value * b.dgrad2};
}
Dual operator/(Dual a, Dual b) {
  const double inv = 1.0 / b.value;
  const double inv2 = inv * inv;
  return {a.value * inv, (a.dn * b.value - a.value * b.dn) * inv2,
          (a.dgrad2 * b.value - a.value * b.dgrad2) * inv2};
}
Dual operator+(Dual a, double b) { return a + Dual{b, 0.0, 0.0}; }
Dual operator+(double a, Dual b) { return Dual{a, 0.0, 0.0} + b; }
Dual operator-(Dual a, double b) { return a - Dual{b, 0.0, 0.0}; }
Dual operator-(double a, Dual b) { return Dual{a, 0.0, 0.0} - b; }
Dual operator*(Dual a, double b) { return {a.value * b, a.dn * b, a.dgrad2 * b}; }
Dual operator*(double a, Dual b) { return b * a; }
Dual operator/(Dual a, double b) { return a * (1.0 / b); }
Dual operator/(double a, Dual b) { return Dual{a, 0.0, 0.0} / b; }

Dual pow(Dual x, double p) {
  // The derivative is p*x^p/x; a second transcendental pow is redundant.
  const double value = p==0.5 ? std::sqrt(x.value) : std::pow(x.value, p);
  const double deriv = p * value / x.value;
  return {value, deriv * x.dn, deriv * x.dgrad2};
}
Dual log(Dual x) { return {std::log(x.value), x.dn / x.value, x.dgrad2 / x.value}; }
Dual log1p(Dual x) { return {std::log1p(x.value), x.dn / (1.0+x.value), x.dgrad2 / (1.0+x.value)}; }
Dual exp(Dual x) {
  const double value = std::exp(x.value);
  return {value, value * x.dn, value * x.dgrad2};
}
Dual expm1(Dual x) {
  const double value = std::expm1(x.value);
  return {value, (1.0+value)*x.dn, (1.0+value)*x.dgrad2};
}

Dual pw92_unpolarized(Dual rs) {
  constexpr double A = 0.0310907;
  constexpr double alpha1 = 0.21370;
  constexpr double b1 = 7.5957;
  constexpr double b2 = 3.5876;
  constexpr double b3 = 1.6382;
  constexpr double b4 = 0.49294;
  const Dual root = pow(rs, 0.5);
  const Dual poly = b1 * root + b2 * rs + b3 * rs * root + b4 * rs * rs;
  return -2.0 * A * (1.0 + alpha1 * rs) * log1p(1.0 / (2.0 * A * poly));
}

Dual pbe_energy_density(Dual density, Dual grad2, Dual* exchange_out = nullptr,
                        Dual* correlation_out = nullptr, double* gradient_response = nullptr) {
  const Dual root_n = pow(density, 1.0 / 3.0);
  const Dual rs = std::cbrt(3.0/(4.0*pi)) / root_n;
  const Dual kf = std::cbrt(3.0*pi*pi)*root_n;
  const Dual eps_x_lda = (-0.75*std::cbrt(3.0/pi))*root_n;
  const Dual s2 = grad2 / (4.0 * kf * kf * density * density);
  const Dual fx = 1.0 + kappa - kappa / (1.0 + (mu / kappa) * s2);
  const Dual exchange = density * eps_x_lda * fx;

  const Dual eps_c_lda = pw92_unpolarized(rs);
  const Dual ks2 = 4.0 * kf / pi;
  const Dual t2 = grad2 / (4.0 * ks2 * density * density);
  const Dual exp_arg = -eps_c_lda / gamma;
  const Dual exp_minus_one = expm1(exp_arg);
  const Dual A = (beta / gamma) / exp_minus_one;
  const Dual at2 = A * t2;
  const Dual t4a2 = at2 * at2;
  const Dual h = gamma * log1p((beta / gamma) * t2 * (1.0 + at2) /
                             (1.0 + at2 + t4a2));
  const Dual correlation = density * (eps_c_lda + h);
  if(gradient_response){
    // PBE chooses mu=beta*pi^2/3. Exchange and correlation sigma
    // derivatives therefore cancel exactly at sigma=0. Adding two huge
    // opposite terms in a vacuum loses accuracy even in FP64. Evaluate the
    // difference of their rational enhancement factors without subtracting
    // their common constant 1. Energy/density derivatives still use AD.
    const double u=(mu/kappa)*s2.value,x=at2.value,den=1+x+x*x;
    const double exchange_loss=u*(2+u)/((1+u)*(1+u));
    const double correlation_loss=(x*x*(3+x*(2+x))+(beta/gamma)*t2.value*(1+x)*den)/
        (den*(den+(beta/gamma)*t2.value*(1+x)));
    const double difference=u>.25||x>.25 ? (1+2*x)/(den*(den+(beta/gamma)*t2.value*(1+x)))-1/((1+u)*(1+u))
        : exchange_loss-correlation_loss;
    *gradient_response=beta/(2*ks2.value*density.value)*difference;
  }
  if (exchange_out) *exchange_out = exchange;
  if (correlation_out) *correlation_out = correlation;
  return exchange + correlation;
}
}

PbeXcPoint pbe_unpolarized(double density, double gradient_magnitude) {
  if (!(density > 0.0) || !std::isfinite(density) || gradient_magnitude < 0.0 ||
      !std::isfinite(gradient_magnitude))
    throw std::invalid_argument("PBE density must be positive and gradient finite");

  const Dual n{density, 1.0, 0.0};
  const Dual grad2{gradient_magnitude * gradient_magnitude, 0.0, 1.0};
  Dual exchange, correlation;
  double gradient_response=0.;
  const Dual total = pbe_energy_density(n, grad2, &exchange, &correlation, &gradient_response);
  PbeXcPoint point;
  point.energy_density = total.value;
  point.energy_per_particle = total.value / density;
  point.exchange_energy_density = exchange.value;
  point.correlation_energy_density = correlation.value;
  point.exchange_potential = exchange.dn;
  point.correlation_potential = correlation.dn;
  point.gradient_squared_coefficient = gradient_response;
  // getenv can acquire a process-wide lock on macOS. Never query it for
  // every valid XC point inside an OpenMP loop; tracing only matters after
  // a nonfinite result. This keeps runtime diagnostic toggles functional.
  if((!std::isfinite(point.energy_density)||!std::isfinite(point.exchange_potential)||!std::isfinite(point.correlation_potential)||!std::isfinite(point.gradient_squared_coefficient))&&std::getenv("SDFT_TRACE_NONFINITE_PBE"))
    std::cerr<<"NONFINITE_PBE density="<<density<<" gradient="<<gradient_magnitude<<" energy="<<point.energy_density<<" exchange_v="<<point.exchange_potential<<" correlation_v="<<point.correlation_potential<<" grad2_response="<<point.gradient_squared_coefficient<<'\n';
  return point;
}
PbeXcPoint pbe_unpolarized_lda_tail(double density,double gradient_magnitude){
  if(density>1e-10)return pbe_unpolarized(density,gradient_magnitude);
  if(!(density>0)||!std::isfinite(density)||gradient_magnitude<0||!std::isfinite(gradient_magnitude))
    throw std::invalid_argument("invalid radial PBE input");
  const Dual n{density,1,0},root=pow(n,1./3.);
  const Dual exchange=(-.75*std::cbrt(3./pi))*n*root;
  const Dual rs=std::cbrt(3./(4*pi))/root;
  // PW92 correlation is taken as zero once sqrt(rs)>1e10 (vanishing density).
  const Dual correlation=std::sqrt(rs.value)>1e10 ? Dual{} : n*pw92_unpolarized(rs);
  PbeXcPoint out;out.energy_density=exchange.value+correlation.value;
  out.energy_per_particle=out.energy_density/density;out.exchange_energy_density=exchange.value;
  out.correlation_energy_density=correlation.value;out.exchange_potential=exchange.dn;out.correlation_potential=correlation.dn;
  out.gradient_squared_coefficient=0.;return out;
}
}
