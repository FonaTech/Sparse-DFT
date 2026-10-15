#include "paw_onecenter.hpp"
#include "pbe_xc.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace sdft {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
constexpr double ev_per_hartree = 27.211386245988;

double trapz(const std::vector<double>& r, const std::vector<double>& f) {
  if (r.size() != f.size() || r.size() < 2) return 0.0;
  double s = 0.0;
  for (std::size_t i=0;i+1<r.size();++i) s += 0.5*(r[i+1]-r[i])*(f[i]+f[i+1]);
  return s;
}

std::vector<double> radial_potential(const std::vector<double>& r,
                                     const std::vector<double>& rho_r) {
  const std::size_t n=r.size();
  std::vector<double> out(n,0.0), qint(n,0.0), tail(n,0.0);
  if (n<2 || rho_r.size()!=n) return out;
  // rho_r is the spherical 3-D density.  Hartree in atomic units.
  for (std::size_t i=1;i<n;++i) {
    const double f0=4.0*pi*rho_r[i-1]*r[i-1]*r[i-1];
    const double f1=4.0*pi*rho_r[i]*r[i]*r[i];
    qint[i]=qint[i-1]+0.5*(r[i]-r[i-1])*(f0+f1);
  }
  for (std::size_t ii=0;ii<n-1;++ii) {
    const std::size_t i=n-1-ii;
    const double f0=rho_r[i]*r[i];
    const double f1=rho_r[i-1]*r[i-1];
    tail[i-1]=tail[i]+0.5*(r[i]-r[i-1])*(f0+f1);
  }
  for (std::size_t i=0;i<n;++i) {
    const double r0=std::max(r[i],1e-12);
    out[i]=qint[i]/r0 + 4.0*pi*tail[i];
  }
  return out;
}

double interp_grad(const std::vector<double>& r, const std::vector<double>& y, std::size_t i) {
  if (i==0) return (y[1]-y[0])/(r[1]-r[0]);
  if (i+1==r.size()) return (y[i]-y[i-1])/(r[i]-r[i-1]);
  return (y[i+1]-y[i-1])/(r[i+1]-r[i-1]);
}
}

PawOneCenterResult paw_one_center_spherical(const PawNumericCache& cache,
                                            const Eigen::MatrixXd& onsite,
                                            const std::vector<int>& channel_l,
                                            const std::vector<int>& channel_nproj,
                                            int spin_multiplicity) {
  if (cache.radial_grid.size()<2 || cache.radial_grid.size()!=cache.pseudo_wavefunctions.front().size())
    throw std::invalid_argument("PAW one-centre radial data are incomplete");
  if (channel_l.size()!=channel_nproj.size()) throw std::invalid_argument("PAW channel metadata mismatch");
  std::size_t expanded=0, raw=0;
  for (std::size_t c=0;c<channel_l.size();++c) { expanded += static_cast<std::size_t>(channel_nproj[c]*(2*channel_l[c]+1)); raw += static_cast<std::size_t>(channel_nproj[c]); }
  if (onsite.rows()!=static_cast<Eigen::Index>(expanded) || onsite.cols()!=onsite.rows())
    throw std::invalid_argument("PAW onsite matrix dimension mismatch");
  if (cache.ae_wavefunctions.size()<raw || cache.pseudo_wavefunctions.size()<raw)
    throw std::invalid_argument("PAW partial waves do not cover all projectors");
  const auto& r=cache.radial_grid; const std::size_t nr=r.size();
  std::vector<double> rho_ae(nr,0.0),rho_ps(nr,0.0);
  // Isotropic (L=0) density.  The m sum removes all angular dependence;
  // radial partial waves are reduced u_l(r), hence rho=u_i u_j/(4 pi r^2).
  std::size_t eoff=0; std::size_t roff=0;
  for (std::size_t c=0;c<channel_l.size();++c) {
    const int deg=2*channel_l[c]+1;
    for (int i=0;i<channel_nproj[c];++i) for (int j=0;j<channel_nproj[c];++j) {
      double occ=0.0;
      for (int m=0;m<deg;++m) occ += onsite(static_cast<Eigen::Index>(eoff+i*deg+m), static_cast<Eigen::Index>(eoff+j*deg+m));
      occ*=static_cast<double>(spin_multiplicity);
      for (std::size_t k=0;k<nr;++k) {
        const double den=4.0*pi*std::max(r[k]*r[k],1e-18);
        rho_ae[k]+=occ*cache.ae_wavefunctions[roff+i][k]*cache.ae_wavefunctions[roff+j][k]/den;
        rho_ps[k]+=occ*cache.pseudo_wavefunctions[roff+i][k]*cache.pseudo_wavefunctions[roff+j][k]/den;
      }
    }
    eoff += static_cast<std::size_t>(channel_nproj[c]*deg); roff += static_cast<std::size_t>(channel_nproj[c]);
  }
  const std::vector<double> rho_val_ae = rho_ae;
  const std::vector<double> rho_val_ps = rho_ps;
  // POTCAR stores the frozen core and its pseudized counterpart on the same
  // radial mesh.  The AE XC branch is evaluated with rho_val+rho_core and the
  // PS branch with rho_val+rho_core^PS (nonlinear core correction).  Omitting
  // these arrays makes the spherical D_ij correction a reference-atom
  // diagnostic rather than the variational PAW term.
  if (cache.core_charge_density.size() == nr) {
    for (std::size_t k=0;k<nr;++k) {
      const double rr=std::max(r[k],1e-14);
      // The POTCAR core arrays store r^2*n(r)/sqrt(4*pi), not a
      // Cartesian density.  Convert before adding them to the valence
      // density used by the radial Hartree/XC evaluator.
      rho_ae[k] += cache.core_charge_density[k] /(std::sqrt(4.0*pi)*rr*rr);
    }
  }
  if (cache.core_charge_density_pseudized.size() == nr) {
    for (std::size_t k=0;k<nr;++k) {
      const double rr=std::max(r[k],1e-14);
      rho_ps[k] += cache.core_charge_density_pseudized[k] /(std::sqrt(4.0*pi)*rr*rr);
    }
  }
  // Hartree is formed from the valence augmentation density.  The core is
  // frozen and enters only the XC branch as rho_core, rather than
  // being re-counted as a valence Hartree charge here.
  auto vh_ae=radial_potential(r,rho_val_ae), vh_ps=radial_potential(r,rho_val_ps);
  std::vector<double> v_ae(nr),v_ps(nr);
  double eae=0.0,eps=0.0,maxv=0.0;
  double max_xc_ae=0.0, min_xc_ae=0.0;
  for (std::size_t k=0;k<nr;++k) {
    const double ga=std::abs(interp_grad(r,rho_ae,k)), gp=std::abs(interp_grad(r,rho_ps,k));
    const double na=std::max(rho_ae[k],1e-14), np=std::max(rho_ps[k],1e-14);
    const auto xa=pbe_unpolarized(na,ga), xp=pbe_unpolarized(np,gp);
    max_xc_ae=std::max(max_xc_ae, std::abs(xa.energy_density));
    min_xc_ae=std::min(min_xc_ae, xa.energy_density);
    v_ae[k]=vh_ae[k]+xa.exchange_potential+xa.correlation_potential;
    v_ps[k]=vh_ps[k]+xp.exchange_potential+xp.correlation_potential;
    maxv=std::max(maxv,std::abs(v_ae[k]-v_ps[k])*ev_per_hartree);
    // pbe_unpolarized returns an energy density per volume.  The radial
    // integral carries 4*pi*r^2 dr; use the same trapezoidal measure as the
    // Hartree helper.  Omitting dr here creates an O(N_r) energy blow-up.
    if (k + 1 < nr) {
      const double dr = r[k + 1] - r[k];
      const double ea = r[k]*r[k]*(0.5*rho_val_ae[k]*vh_ae[k] + xa.energy_density);
      const double ep = r[k]*r[k]*(0.5*rho_val_ps[k]*vh_ps[k] + xp.energy_density);
      const double na = std::max(rho_ae[k + 1],1e-14), np1 = std::max(rho_ps[k + 1],1e-14);
      const double ga1 = std::abs(interp_grad(r,rho_ae,k+1)), gp1 = std::abs(interp_grad(r,rho_ps,k+1));
      const auto xa1 = pbe_unpolarized(na,ga1), xp1 = pbe_unpolarized(np1,gp1);
      const double ea1 = r[k+1]*r[k+1]*(0.5*rho_val_ae[k+1]*vh_ae[k+1] + xa1.energy_density);
      const double ep1 = r[k+1]*r[k+1]*(0.5*rho_val_ps[k+1]*vh_ps[k+1] + xp1.energy_density);
      eae += 0.5*dr*(ea+ea1);
      eps += 0.5*dr*(ep+ep1);
    }
  }
  eae*=4.0*pi; eps*=4.0*pi;
  if (std::getenv("SDFT_TRACE_ONECENTER_DETAIL")) {
    std::cerr << "onecenter eae_ha=" << eae << " eps_ha=" << eps
              << " rho0_ae=" << rho_ae.front() << " rho0_ps=" << rho_ps.front()
              << " vh0_ae=" << vh_ae.front() << " vh0_ps=" << vh_ps.front()
              << " xc_abs_max=" << max_xc_ae << " xc_min=" << min_xc_ae << "\n";
  }
  PawOneCenterResult out; out.spherical_dij_ev=Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(raw),static_cast<Eigen::Index>(raw));
  roff=0;
  for (std::size_t c=0;c<channel_l.size();++c) {
    const int deg=2*channel_l[c]+1;
    for (int i=0;i<channel_nproj[c];++i) for (int j=0;j<channel_nproj[c];++j) {
      std::vector<double> f(nr);
      for (std::size_t k=0;k<nr;++k) f[k]=(cache.ae_wavefunctions[roff+i][k]*cache.ae_wavefunctions[roff+j][k]*v_ae[k]-cache.pseudo_wavefunctions[roff+i][k]*cache.pseudo_wavefunctions[roff+j][k]*v_ps[k]);
      out.spherical_dij_ev(static_cast<Eigen::Index>(roff+i),static_cast<Eigen::Index>(roff+j))=trapz(r,f)*ev_per_hartree;
    }
    roff+=static_cast<std::size_t>(channel_nproj[c]);
  }
  out.double_counting_ae_ev=eae*ev_per_hartree; out.double_counting_ps_ev=eps*ev_per_hartree;
  out.energy_difference_ev = (eae - eps) * ev_per_hartree;
  // The on-site potential inserted into D must be subtracted from the band
  // sum once.  Report its expectation explicitly so the caller can assemble
  // the PAW double-counting contribution without inventing a separate scale.
  double vexp = 0.0;
  roff=0; eoff=0;
  for (std::size_t c=0;c<channel_l.size();++c) {
    const int deg=2*channel_l[c]+1;
    for (int i=0;i<channel_nproj[c];++i) for (int j=0;j<channel_nproj[c];++j) {
      double occ=0.0;
      for (int m=0;m<deg;++m)
        occ += onsite(static_cast<Eigen::Index>(eoff+i*deg+m), static_cast<Eigen::Index>(eoff+j*deg+m));
      vexp += occ * out.spherical_dij_ev(static_cast<Eigen::Index>(roff+i), static_cast<Eigen::Index>(roff+j));
    }
    eoff += static_cast<std::size_t>(channel_nproj[c]*deg);
    roff += static_cast<std::size_t>(channel_nproj[c]);
  }
  out.potential_expectation_ev = vexp;
  std::vector<double> charge(nr);
  for (std::size_t k=0;k<nr;++k) charge[k]=4.0*pi*rho_val_ae[k]*r[k]*r[k];
  out.electron_count=trapz(r,charge);
  out.max_potential_ev=maxv;
  return out;
}
}
