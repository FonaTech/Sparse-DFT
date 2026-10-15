#include "paw_compensation.hpp"
#include "compute_backend.hpp"
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <exception>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace sdft {
namespace {
using Complex = std::complex<double>;
constexpr double pi = 3.14159265358979323846;
constexpr double bohr_per_angstrom = 1.8897261246257702;
constexpr std::size_t compensation_tile_size=8192;
constexpr std::size_t compensation_tile_window=16;

bool blas_compensation() {
  const char* value=std::getenv("SDFT_COMPENSATION_BLAS");
  if(value&&std::string(value)!="0"&&std::string(value)!="1")
    throw std::invalid_argument("SDFT_COMPENSATION_BLAS must be 0 or 1");
  return value&&std::string(value)=="1";
}

bool compact_geometry() {
  const char* value=std::getenv("SDFT_COMPACT_COMPENSATION");
  if(value && std::string(value)!="0" && std::string(value)!="1")
    throw std::invalid_argument("SDFT_COMPACT_COMPENSATION must be 0 or 1");
  return value && std::string(value)=="1";
}

bool batched_geometry() {
  const char* value=std::getenv("SDFT_BATCH_COMPENSATION_GEOMETRY");
  if(value && std::string(value)!="0" && std::string(value)!="1")
    throw std::invalid_argument("SDFT_BATCH_COMPENSATION_GEOMETRY must be 0 or 1");
  return value && std::string(value)=="1";
}

Vec3 scaled(Vec3 v, double s) { return {v.x * s, v.y * s, v.z * s}; }
Vec3 sum(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 cartesian(const std::array<Vec3, 3>& a, Vec3 f) {
  return sum(sum(scaled(a[0], f.x), scaled(a[1], f.y)), scaled(a[2], f.z));
}

std::vector<double> radial_weights(const std::vector<double>& r) {
  if (r.size() < 3 || r.front() <= 0.0)
    throw std::invalid_argument("PAW compensation requires a positive radial mesh");
  const double h = std::log(r[1] / r[0]);
  bool logarithmic = std::isfinite(h) && h > 0.0;
  for (std::size_t i = 1; i < r.size(); ++i) {
    if (!std::isfinite(r[i]) || r[i] <= r[i - 1])
      throw std::invalid_argument("PAW radial mesh must be finite and increasing");
    logarithmic = logarithmic && std::abs(std::log(r[i] / r[i - 1]) - h) < 2e-7;
  }
  std::vector<double> weights(r.size(), 0.0);
  if (logarithmic) {
    for (std::size_t k = r.size(); k >= 3; k -= 2) {
      weights[k - 1] += r[k - 1] * h / 3.0;
      weights[k - 2] = 4.0 * r[k - 2] * h / 3.0;
      weights[k - 3] = r[k - 3] * h / 3.0;
      if (k == 3) break;
    }
  } else {
    for (std::size_t i = 0; i + 1 < r.size(); ++i) {
      weights[i] += (r[i + 1] - r[i]) / 2.0;
      weights[i + 1] += (r[i + 1] - r[i]) / 2.0;
    }
  }
  return weights;
}

double bessel(int l, double x) {
  if (std::abs(x) < 0.5) {
    // The series avoids catastrophic upward-recurrence cancellation for L>1.
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
  if (x > l + 1.0) {
    double jm1 = std::sin(x) / x;
    if (l == 0) return jm1;
    double j = std::sin(x) / (x * x) - std::cos(x) / x;
    for (int n = 1; n < l; ++n) {
      const double next = (2.0 * n + 1.0) * j / x - jm1;
      jm1 = j;
      j = next;
    }
    return j;
  }
  // Miller downward recurrence is stable when x is below the target order.
  const int top = l + 32;
  // Constructor validation bounds L by 6 (derivatives use at most L=6).
  // Keep the same recurrence on the stack rather than allocating for each
  // radial quadrature point and reciprocal shell.
  std::array<double, 6 + 32 + 2> j{};
  j[static_cast<std::size_t>(top)] = 1.0;
  for (int n = top; n >= 1; --n) {
    j[static_cast<std::size_t>(n - 1)] =
        (2.0 * n + 1.0) * j[static_cast<std::size_t>(n)] / x - j[static_cast<std::size_t>(n + 1)];
    if (std::abs(j[static_cast<std::size_t>(n - 1)]) > 1e100) {
      for (int k = n - 1; k <= top + 1; ++k) j[static_cast<std::size_t>(k)] *= 1e-100;
    }
  }
  const double normalization = (std::sin(x) / x) / j[0];
  return j[static_cast<std::size_t>(l)] * normalization;
}
} // namespace

double paw_real_spherical_harmonic(int l, int m, Vec3 q) {
  if (l < 0 || l > 6 || std::abs(m) > l)
    throw std::invalid_argument("PAW real harmonics require 0<=L<=6 and |M|<=L");
  const double radius = std::sqrt(dot(q, q));
  if (radius < 1e-14) return l == 0 ? 1.0 / std::sqrt(4.0 * pi) : 0.0;
  const double z = std::clamp(q.z / radius, -1.0, 1.0);
  const double s = std::sqrt(std::max(0.0, 1.0 - z * z));
  const int order = std::abs(m);
  double pmm = 1.0;
  for (int k = 1; k <= order; ++k) pmm *= (2.0 * k - 1.0) * s;
  double previous = 0.0, value = pmm;
  if (l > order) {
    previous = pmm;
    value = (2.0 * order + 1.0) * z * pmm;
    for (int degree = order + 2; degree <= l; ++degree) {
      const double next = ((2.0 * degree - 1.0) * z * value -
          (degree + order - 1.0) * previous) / (degree - order);
      previous = value;
      value = next;
    }
  }
  const double factor = std::sqrt((2.0 * l + 1.0) / (4.0 * pi) *
      std::exp(std::lgamma(l - order + 1.0) - std::lgamma(l + order + 1.0))) *
      (order ? std::sqrt(2.0) : 1.0);
  const double angle = order * std::atan2(q.y, q.x);
  return factor * value * (m < 0 ? std::sin(angle) : std::cos(angle));
}

Vec3 paw_real_spherical_harmonic_gradient(int l,int m,Vec3 q) {
  if(l<0||l>6||std::abs(m)>l)throw std::invalid_argument("unsupported real harmonic gradient");
  const double r2=dot(q,q),r=std::sqrt(r2);
  if(r<1e-14||l==0)return {};
  // Homogeneous solid-harmonic recurrence is regular on the polar axis,
  // unlike derivatives through atan2 or acos. Values have no CS phase.
  using A=std::array<Complex,3>;
  const int order=std::abs(m);const Complex zxy(q.x,q.y);
  Complex value=1;A derivative{};
  for(int degree=1;degree<=order;++degree) {
    const double factor=2*degree-1.;
    for(int a=0;a<3;++a)derivative[a]=factor*(derivative[a]*zxy+value*(a==0?Complex(1):a==1?Complex(0,1):Complex(0)));
    value*=factor*zxy;
  }
  Complex prev{};A prev_derivative{};
  if(l>order) {
    prev=value;prev_derivative=derivative;
    const double factor=2*order+1.;
    for(int a=0;a<3;++a)derivative[a]=factor*(q.z*derivative[a]+(a==2?value:Complex{}));
    value*=factor*q.z;
    const double qs[]{q.x,q.y,q.z};
    for(int degree=order+2;degree<=l;++degree) {
      const Complex next=((2*degree-1.)*q.z*value-(degree+order-1.)*r2*prev)/double(degree-order);
      A next_derivative;
      for(int a=0;a<3;++a)next_derivative[a]=((2*degree-1.)*(q.z*derivative[a]+(a==2?value:Complex{}))-
          (degree+order-1.)*(r2*prev_derivative[a]+2*qs[a]*prev))/double(degree-order);
      prev=value;prev_derivative=derivative;value=next;derivative=next_derivative;
    }
  }
  const double factor=std::sqrt((2*l+1.)/(4*pi)*std::exp(std::lgamma(l-order+1.)-std::lgamma(l+order+1.)))*(order?std::sqrt(2.):1.)/std::pow(r,l);
  const double qs[]{q.x,q.y,q.z};double out[3];
  for(int a=0;a<3;++a){const Complex d=factor*(derivative[a]-double(l)*value*qs[a]/r2);out[a]=m<0?d.imag():d.real();}
  return {out[0],out[1],out[2]};
}

PawCompensationOperator::PawCompensationOperator(const PawAugmentationData& a,
    const std::array<Vec3, 3>& lattice, Vec3 fractional,
    const std::vector<std::tuple<int, int, int>>& keys, AtomicUnitConvention constants,
    const PawCompensationOperator* shared)
    : augmentation_(a) {
  const double bohr_per_angstrom=atomic_units(constants).bohr_per_angstrom;
  const double signed_volume = determinant(lattice);
  volume_ = std::abs(signed_volume);
  if (!std::isfinite(volume_) || volume_ < 1e-12)
    throw std::invalid_argument("singular PAW compensation lattice");
  if (a.max_l < 0 || a.max_l > 6 || a.compensation_r2.size() != static_cast<std::size_t>(a.max_l + 1))
    throw std::invalid_argument("incomplete PAW compensation multipoles");
  for (const auto& shape : a.compensation_r2)
    if (shape.size() != a.radial_grid.size())
      throw std::invalid_argument("PAW compensation radial shape mismatch");
  for (int l : a.channel_l) dimension_ += static_cast<std::size_t>(2 * l + 1);
  multipole_count_ = static_cast<std::size_t>((a.max_l + 1) * (a.max_l + 1));
  const Vec3 position = cartesian(lattice, fractional);
  phases_.resize(keys.size());
  if(shared) {
    const auto& other=shared->augmentation_;const auto& geometry=*shared->geometry_;
    bool same_cell=true;
    for(int axis=0;axis<3;++axis)same_cell &= lattice[axis].x==geometry.lattice[axis].x &&
        lattice[axis].y==geometry.lattice[axis].y && lattice[axis].z==geometry.lattice[axis].z;
    if(!same_cell || constants!=geometry.constants || keys!=geometry.keys ||
       a.channel_l!=other.channel_l || a.max_l!=other.max_l || a.rmax!=other.rmax || a.rdep!=other.rdep ||
       a.mesh_compensation_norm!=other.mesh_compensation_norm || a.radial_grid!=other.radial_grid || a.aug_moment!=other.aug_moment || a.overlap_correction!=other.overlap_correction ||
       a.compensation_r2!=other.compensation_r2 || a.compensation_q!=other.compensation_q ||
       a.compensation_amplitudes!=other.compensation_amplitudes)
      throw std::invalid_argument("shared PAW compensation geometry differs in cell, units, G support or augmentation");
    geometry_=shared->geometry_;
    density_transform_=shared->density_transform_;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(batched_geometry() && !omp_in_parallel() && keys.size()>=4096)
#endif
    for(std::size_t ik=0;ik<keys.size();++ik)
      phases_[ik]=std::exp(Complex(0.,-dot(geometry.reciprocal_vectors[ik],position)));
    return;
  }
  auto geometry=std::make_shared<Geometry>();
  geometry->lattice=lattice;geometry->constants=constants;geometry->keys=keys;
  geometry_=geometry;
  const std::size_t entries = dimension_ * dimension_;
  density_transform_.assign(multipole_count_ * entries, Complex{});
  std::vector<Complex> unit(entries, Complex{});
  for (std::size_t entry = 0; entry < entries; ++entry) {
    unit[entry] = 1.0;
    const auto transformed = paw_lm_moments_from_density(a, unit, dimension_);
    unit[entry] = 0.0;
    for (std::size_t lm = 0; lm < multipole_count_; ++lm)
      density_transform_[lm * entries + entry] = transformed[lm];
  }
  const std::array<Vec3, 3> b{{scaled(cross(lattice[1], lattice[2]), 2.0 * pi / signed_volume),
      scaled(cross(lattice[2], lattice[0]), 2.0 * pi / signed_volume),
      scaled(cross(lattice[0], lattice[1]), 2.0 * pi / signed_volume)}};
  const auto weights = radial_weights(a.radial_grid);
  PerformanceTimer geometry_timer("paw_compensation_geometry_initialization");
  const bool compact=compact_geometry();
  if(compact) {
    geometry->compact_form.resize(keys.size()*multipole_count_);
    for(auto& gradient:geometry->compact_gradients)gradient.resize(geometry->compact_form.size());
    geometry->imaginary_channel.resize(multipole_count_);
    for(int L=0;L<=a.max_l;++L)for(int M=-L;M<=L;++M)
      geometry->imaginary_channel[static_cast<std::size_t>(L*L+M+L)]=L%2;
  }else {
    geometry->form_factors.resize(keys.size() * multipole_count_);
    for(auto& gradient:geometry->shape_gradients)gradient.resize(geometry->form_factors.size());
  }
  const auto integrate=[&](double q_angstrom) {
    std::pair<std::vector<double>,std::vector<double>> result;
    auto& [integrals,derivatives]=result;
    integrals.resize(static_cast<std::size_t>(a.max_l+1));
    derivatives.resize(integrals.size());
    for(int L=0;L<=a.max_l;++L)for(std::size_t ir=0;ir<weights.size();++ir) {
      const double x=q_angstrom*a.radial_grid[ir];
      integrals[L]+=weights[ir]*a.compensation_r2[L][ir]*bessel(L,x);
      const double derivative=std::abs(x)<1e-14 ? (L==1?1./3.:0.) :
          (L==0?-bessel(1,x):bessel(L-1,x)-(L+1.)*bessel(L,x)/x);
      derivatives[L]+=weights[ir]*a.compensation_r2[L][ir]*a.radial_grid[ir]*derivative*bohr_per_angstrom;
    }
    if(q_angstrom<1e-14)integrals[0]=1.;
    return result;
  };
  const auto fill=[&](std::size_t ik,const std::vector<double>& integrals,const std::vector<double>& derivatives) {
    const auto q=geometry->reciprocal_vectors[ik];
    phases_[ik]=std::exp(Complex(0,-dot(q,position)));
    const double radius=std::sqrt(dot(q,q)),qs[]{q.x,q.y,q.z};
    for(int L=0;L<=a.max_l;++L) {
      // Integer Fourier angular phase is an exact quarter turn. The
      // complex pow overload uses log/exp and leaves spurious 1e-16 parts.
      // Use the same exact phase in both representations, never truncate
      // a computed complex component to make compact storage appear valid.
      static constexpr Complex angular_phase[]{Complex(1,0),Complex(0,-1),
                                                Complex(-1,0),Complex(0,1)};
      const Complex factor=4.*pi*angular_phase[L%4];
      for(int M=-L;M<=L;++M) {
        const auto lm=static_cast<std::size_t>(L*L+M+L);
        const double harmonic=paw_real_spherical_harmonic(L,M,q);
        const Complex form=factor*harmonic*integrals[L];
        const auto index=ik*multipole_count_+lm;
        if(compact) {
          if((L%2 ? form.real() : form.imag())!=0.)throw std::runtime_error("compensation form has an unexpected angular phase");
          geometry->compact_form[index]=L%2 ? form.imag() : form.real();
        }else geometry->form_factors[index]=form;
        const auto dy=paw_real_spherical_harmonic_gradient(L,M,q);const double ys[]{dy.x,dy.y,dy.z};
        for(int axis=0;axis<3;++axis) {
          const Complex derivative=factor*(ys[axis]*integrals[L]+(radius>1e-14?harmonic*derivatives[L]*qs[axis]/radius:0.));
          if(compact) {
            if((L%2 ? derivative.real() : derivative.imag())!=0.)throw std::runtime_error("compensation derivative has an unexpected angular phase");
            geometry->compact_gradients[axis][index]=L%2 ? derivative.imag() : derivative.real();
          }else geometry->shape_gradients[axis][index]=derivative;
        }
      }
    }
  };
  geometry->reciprocal_vectors.reserve(keys.size());
  for(const auto& [h,k,l]:keys)
    geometry->reciprocal_vectors.push_back(cartesian(b,{double(h),double(k),double(l)}));
  if(!batched_geometry()) {
    std::map<double,std::pair<std::vector<double>,std::vector<double>>> cache;
    for(std::size_t ik=0;ik<keys.size();++ik) {
      const double radius=std::sqrt(dot(geometry->reciprocal_vectors[ik],geometry->reciprocal_vectors[ik]))*bohr_per_angstrom;
      auto found=cache.find(radius);
      if(found==cache.end())found=cache.emplace(radius,integrate(radius)).first;
      fill(ik,found->second.first,found->second.second);
    }
    return;
  }
  // Exact double-radius identity matches the legacy cache. No interpolation,
  // shell quantization, discarded multipoles or quadrature changes.
  std::map<double,std::size_t> shell_ids;
  std::vector<double> radii;std::vector<std::size_t> shell(keys.size());
  for(std::size_t ik=0;ik<keys.size();++ik) {
    const double radius=std::sqrt(dot(geometry->reciprocal_vectors[ik],geometry->reciprocal_vectors[ik]))*bohr_per_angstrom;
    auto [found,inserted]=shell_ids.try_emplace(radius,radii.size());
    if(inserted)radii.push_back(radius);
    shell[ik]=found->second;
  }
  std::vector<std::pair<std::vector<double>,std::vector<double>>> radial(radii.size());
  std::exception_ptr failure;std::mutex failure_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && radii.size()>=64)
#endif
  for(std::size_t i=0;i<radii.size();++i) {
    try { radial[i]=integrate(radii[i]); }
    catch(...) {std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
  }
  if(failure)std::rethrow_exception(failure);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && keys.size()>=4096)
#endif
  for(std::size_t ik=0;ik<keys.size();++ik) {
    try { fill(ik,radial[shell[ik]].first,radial[shell[ik]].second); }
    catch(...) {std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
  }
  if(failure)std::rethrow_exception(failure);

}

std::size_t PawCompensationOperator::form_factor_count()const{return geometry_->compact_form.empty()?geometry_->form_factors.size():geometry_->compact_form.size();}
std::vector<Complex> PawCompensationOperator::Geometry::contract_density_tile(
    std::size_t first,std::size_t count,const std::vector<Complex>& coefficients,
    std::size_t columns,int axis)const {
  const auto lm=imaginary_channel.size();
  if(compact_form.empty()||!lm||!columns||!count||first>keys.size()||count>keys.size()-first||
      coefficients.size()!=lm*columns||axis<-1||axis>2)
    throw std::invalid_argument("compensation BLAS density tile requires exact compact geometry and matching dimensions");
  const auto& values=axis<0?compact_form:compact_gradients[axis];
  std::vector<double> cr(coefficients.size()),ci(coefficients.size()),re(count*columns),im(re.size());
  for(std::size_t col=0;col<columns;++col)for(std::size_t l=0;l<lm;++l) {
    const auto index=col*lm+l;const auto c=coefficients[index];
    // F_lm is either a signed real number or i times that number. This
    // rotates BOTH components of the occupation and its (L,M) moments,
    // without assuming Gamma.
    cr[index]=imaginary_channel[l]?-c.imag():c.real();
    ci[index]=imaginary_channel[l]? c.real():c.imag();
  }
  cpu_real_gemm(MatrixOp::Adjoint,MatrixOp::Normal,count,columns,lm,
      values.data()+first*lm,lm,cr.data(),lm,re.data(),count);
  cpu_real_gemm(MatrixOp::Adjoint,MatrixOp::Normal,count,columns,lm,
      values.data()+first*lm,lm,ci.data(),lm,im.data(),count);
  std::vector<Complex> result(re.size());
  for(std::size_t i=0;i<result.size();++i)result[i]={re[i],im[i]};
  return result;
}

std::vector<Complex> PawCompensationOperator::Geometry::contract_potential_tile(
    std::size_t first,std::size_t count,const std::vector<Complex>& phased,
    std::size_t columns)const {
  const auto lm=imaginary_channel.size();
  if(compact_form.empty()||!lm||!columns||!count||first>keys.size()||count>keys.size()-first||phased.size()!=count*columns)
    throw std::invalid_argument("compensation BLAS potential tile requires exact compact geometry and matching dimensions");
  std::vector<double> vr(phased.size()),vi(phased.size()),re(lm*columns),im(re.size());
  for(std::size_t i=0;i<phased.size();++i){vr[i]=phased[i].real();vi[i]=phased[i].imag();}
  cpu_real_gemm(MatrixOp::Normal,MatrixOp::Normal,lm,columns,count,
      compact_form.data()+first*lm,lm,vr.data(),count,re.data(),lm);
  cpu_real_gemm(MatrixOp::Normal,MatrixOp::Normal,lm,columns,count,
      compact_form.data()+first*lm,lm,vi.data(),count,im.data(),lm);
  std::vector<Complex> result(re.size());
  for(std::size_t col=0;col<columns;++col)for(std::size_t l=0;l<lm;++l) {
    const auto i=col*lm+l;
    result[i]=imaginary_channel[l]?Complex(-im[i],re[i]):Complex(re[i],im[i]);
  }
  return result;
}

std::vector<std::vector<std::size_t>> PawCompensationOperator::geometry_groups(
    const std::vector<PawCompensationOperator>& atoms) {
  std::vector<std::vector<std::size_t>> groups;
  for(std::size_t i=0;i<atoms.size();++i) {
    const auto* geometry=atoms[i].geometry_.get();
    if(geometry->compact_form.empty())
      throw std::invalid_argument("SDFT_COMPENSATION_BLAS=1 requires SDFT_COMPACT_COMPENSATION=1");
    // Cap atom columns as well as G rows: a larger system must not allocate
    // a whole G x Natoms temporary on every OpenMP worker.
    auto it=std::find_if(groups.begin(),groups.end(),[&](const auto& group){
      return group.size()<16&&atoms[group.front()].geometry_.get()==geometry;
    });
    if(it==groups.end())groups.push_back({i});else it->push_back(i);
  }
  return groups;
}
std::size_t PawCompensationOperator::geometry_storage_bytes()const {
  std::size_t bytes=geometry_->keys.capacity()*sizeof(geometry_->keys[0])+
      geometry_->reciprocal_vectors.capacity()*sizeof(Vec3)+geometry_->form_factors.capacity()*sizeof(Complex);
  for(const auto& gradient:geometry_->shape_gradients)bytes+=gradient.capacity()*sizeof(Complex);
  bytes+=geometry_->compact_form.capacity()*sizeof(double)+geometry_->imaginary_channel.capacity();
  for(const auto& gradient:geometry_->compact_gradients)bytes+=gradient.capacity()*sizeof(double);
  return bytes;
}
std::size_t PawCompensationOperator::atom_storage_bytes()const {
  return phases_.capacity()*sizeof(Complex)+density_transform_.capacity()*sizeof(Complex);
}
bool PawCompensationOperator::shares_geometry_with(const PawCompensationOperator& other)const {
  return geometry_.get()==other.geometry_.get();
}

std::vector<Complex> PawCompensationOperator::multipoles(const std::vector<Complex>& p) const {
  const std::size_t entries = dimension_ * dimension_;
  if (p.size() != entries) throw std::invalid_argument("PAW compensation density dimension mismatch");
  for (std::size_t i = 0; i < dimension_; ++i) for (std::size_t j = 0; j < dimension_; ++j) {
    if (!std::isfinite(p[i * dimension_ + j].real()) || !std::isfinite(p[i * dimension_ + j].imag()) ||
        std::abs(p[i * dimension_ + j] - std::conj(p[j * dimension_ + i])) > 1e-9)
      throw std::invalid_argument("PAW compensation density must be finite and Hermitian");
  }
  // The separate real/imaginary moment transforms serve the mixer and
  // diagnostics; the physical compensation charge contracts every ordered
  // pair of the Hermitian occupation matrix.  Using the upper-triangle
  // imaginary transform as a charge coefficient creates
  // an unphysical complex monopole and breaks the generalized-overlap charge
  // identity for a single complex k point.
  return paw_lm_moments_from_density(augmentation_, p, dimension_);
}

std::vector<Complex> PawCompensationOperator::reciprocal_matrix(std::size_t g,int axis) const {
  if(g>=geometry_->keys.size()||axis< -1||axis>2)
    throw std::invalid_argument("compensation reciprocal matrix index/axis invalid");
  const auto entries=dimension_*dimension_;
  std::vector<Complex> result(entries);
  for(std::size_t i=0;i<dimension_;++i)for(std::size_t j=0;j<dimension_;++j)
    for(std::size_t lm=0;lm<multipole_count_;++lm) {
      const auto index=g*multipole_count_+lm;
      const auto form=axis<0?geometry_->form(index,lm):geometry_->gradient(axis,index,lm);
      result[j*dimension_+i]+=form*density_transform_[lm*entries+i*dimension_+j];
    }
  return result;
}

FourierDensity PawCompensationOperator::density(const std::vector<Complex>& p) const {
  const auto lm_moments = multipoles(p);
  FourierDensity out;
  for (std::size_t ik = 0; ik < geometry_->keys.size(); ++ik) {
    Complex value{};
    for (std::size_t lm = 0; lm < multipole_count_; ++lm)
      value += geometry_->form(ik * multipole_count_ + lm,lm) * lm_moments[lm];
    out[geometry_->keys[ik]] = phases_[ik] * value / volume_;
  }
  return out;
}

FourierDensity PawCompensationOperator::density_sum(
    const std::vector<PawCompensationOperator>& atoms,
    const std::vector<std::vector<Complex>>& onsite) {
  if(atoms.size()!=onsite.size())throw std::invalid_argument("compensation atom/density count mismatch");
  if(atoms.empty())return {};
  PerformanceTimer timer("paw_compensation_density_batch");
  const auto& keys=atoms.front().geometry_->keys;
  std::vector<std::vector<Complex>> lm_moments;
  lm_moments.reserve(atoms.size());
  for(std::size_t atom=0;atom<atoms.size();++atom) {
    if(atoms[atom].geometry_->keys!=keys)
      throw std::invalid_argument("batched compensation reciprocal support mismatch");
    lm_moments.push_back(atoms[atom].multipoles(onsite[atom]));
  }
  std::vector<Complex> charge(keys.size());
  if(blas_compensation()&&!keys.empty()) {
    PerformanceTimer contraction_timer("paw_compensation_blas_density");
    const auto groups=geometry_groups(atoms);
    std::vector<std::vector<Complex>> coefficients;
    for(const auto& group:groups) {
      coefficients.emplace_back();auto& c=coefficients.back();
      for(auto atom:group)c.insert(c.end(),lm_moments[atom].begin(),lm_moments[atom].end());
    }
    const auto tiles=(keys.size()+compensation_tile_size-1)/compensation_tile_size;
    std::exception_ptr failure;std::mutex failure_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel()&&tiles>1)
#endif
    for(std::size_t tile=0;tile<tiles;++tile) {
      try {
        const auto first=tile*compensation_tile_size,count=std::min(compensation_tile_size,keys.size()-first);
        for(std::size_t group=0;group<groups.size();++group) {
          const auto& members=groups[group];
          const auto values=atoms[members[0]].geometry_->contract_density_tile(first,count,coefficients[group],members.size());
          for(std::size_t col=0;col<members.size();++col) {
            const auto& atom=atoms[members[col]];
            for(std::size_t i=0;i<count;++i)charge[first+i]+=atom.phases_[first+i]*values[col*count+i]/atom.volume_;
          }
        }
      }catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
    }
    if(failure)std::rethrow_exception(failure);
    FourierDensity out;
    for(std::size_t ig=0;ig<keys.size();++ig)out.insert_or_assign(out.end(),keys[ig],charge[ig]);
    return out;
  }
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && keys.size()>=4096)
#endif
  for(std::size_t ig=0;ig<keys.size();++ig) {
    Complex sum{};
    for(std::size_t atom=0;atom<atoms.size();++atom) {
      const auto& op=atoms[atom];Complex value{};
      for(std::size_t lm=0;lm<op.multipole_count_;++lm)value+=op.geometry_->form(ig*op.multipole_count_+lm,lm)*lm_moments[atom][lm];
      sum+=op.phases_[ig]*value/op.volume_;
    }
    charge[ig]=sum;
  }
  FourierDensity out;
  for(std::size_t ig=0;ig<keys.size();++ig)out.insert_or_assign(out.end(),keys[ig],charge[ig]);
  return out;
}

std::vector<Complex> PawCompensationOperator::project_potential(const FourierDensity& v) const {
  std::vector<Complex> responses(multipole_count_, Complex{});
  for (std::size_t ik = 0; ik < geometry_->keys.size(); ++ik) {
    const auto [h, k, l] = geometry_->keys[ik];
    const auto opposite = v.find({-h, -k, -l});
    if (opposite == v.end()) continue;
    for (std::size_t lm = 0; lm < multipole_count_; ++lm)
      // This is the adjoint of density(): keep both moment components. For a
      // Hermitian onsite density the resulting projector matrix is Hermitian;
      // taking .real() here would silently implement only the Gamma path.
      // density() divides the Fourier coefficient by the cell volume, while
      // the variational pairing carries the compensating volume factor. The
      // adjoint matrix therefore uses V[-G] F_G without another volume
      // division.
      responses[lm] += opposite->second * phases_[ik] * geometry_->form(ik * multipole_count_ + lm,lm);
  }
  const std::size_t entries = dimension_ * dimension_;
  std::vector<Complex> out(entries, Complex{});
  for (std::size_t lm = 0; lm < multipole_count_; ++lm)
    for (std::size_t entry = 0; entry < entries; ++entry)
      out[entry] += responses[lm] * density_transform_[lm * entries + entry];
  return out;
}

std::vector<std::vector<Complex>> PawCompensationOperator::project_potential_batch(
    const std::vector<PawCompensationOperator>& atoms,const FourierDensity& v) {
  if(atoms.empty())return {};
  PerformanceTimer timer("paw_compensation_adjoint_batch");
  const auto& keys=atoms.front().geometry_->keys;
  std::vector<std::vector<Complex>> results(atoms.size()),responses(atoms.size());
  for(std::size_t atom=0;atom<atoms.size();++atom) {
    const auto& op=atoms[atom];
    if(op.geometry_->keys!=keys)
      throw std::invalid_argument("batched compensation potential reciprocal support mismatch");
    results[atom].resize(op.dimension_*op.dimension_);
    responses[atom].resize(op.multipole_count_);
  }
  // A missing mode remains absent, as in the serial adjoint. The sampled
  // array is temporary, not retained once this Hamiltonian has been built.
  std::vector<Complex> sampled(keys.size());
  std::vector<unsigned char> present(keys.size());
  for(std::size_t ig=0;ig<keys.size();++ig) {
    const auto [h,k,l]=keys[ig];
    if(const auto found=v.find({-h,-k,-l});found!=v.end()) {
      sampled[ig]=found->second;present[ig]=1;
    }
  }
  if(blas_compensation()&&!keys.empty()) {
    PerformanceTimer contraction_timer("paw_compensation_blas_adjoint");
    const auto groups=geometry_groups(atoms);
    const auto tiles=(keys.size()+compensation_tile_size-1)/compensation_tile_size;
    std::exception_ptr failure;std::mutex failure_mutex;
    for(std::size_t begin=0;begin<tiles;begin+=compensation_tile_window) {
    const auto end=std::min(tiles,begin+compensation_tile_window);
    std::vector<std::vector<std::vector<Complex>>> partial(end-begin);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel()&&end-begin>1)
#endif
    for(std::size_t tile=begin;tile<end;++tile) {
      try {
        const auto first=tile*compensation_tile_size,count=std::min(compensation_tile_size,keys.size()-first);
        partial[tile-begin].resize(atoms.size());
        for(const auto& members:groups) {
          std::vector<Complex> phased(count*members.size());
          for(std::size_t col=0;col<members.size();++col)for(std::size_t i=0;i<count;++i)
            phased[col*count+i]=sampled[first+i]*atoms[members[col]].phases_[first+i];
          const auto& geometry=*atoms[members[0]].geometry_;const auto lm=geometry.imaginary_channel.size();
          const auto values=geometry.contract_potential_tile(first,count,phased,members.size());
          for(std::size_t col=0;col<members.size();++col)
            partial[tile-begin][members[col]].assign(values.begin()+col*lm,values.begin()+(col+1)*lm);
        }
      }catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
    }
    if(failure)std::rethrow_exception(failure);
    // Fixed tile order, independent of the OpenMP schedule. Only the BLAS
    // reduction order changes; no Fourier modes or multipoles are omitted.
    for(std::size_t tile=begin;tile<end;++tile)for(std::size_t atom=0;atom<atoms.size();++atom)
      for(std::size_t lm=0;lm<responses[atom].size();++lm)responses[atom][lm]+=partial[tile-begin][atom][lm];
    }
    for(std::size_t atom=0;atom<atoms.size();++atom) {
      const auto& op=atoms[atom];const auto entries=op.dimension_*op.dimension_;
      for(std::size_t lm=0;lm<op.multipole_count_;++lm)for(std::size_t entry=0;entry<entries;++entry)
        results[atom][entry]+=responses[atom][lm]*op.density_transform_[lm*entries+entry];
    }
    return results;
  }
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && atoms.size()>1 && keys.size()>=128)
#endif
  for(std::size_t atom=0;atom<atoms.size();++atom) {
    const auto& op=atoms[atom];auto& response=responses[atom];
    for(std::size_t ig=0;ig<keys.size();++ig) {
      if(!present[ig])continue;
      const Complex phased=sampled[ig]*op.phases_[ig];
      for(std::size_t lm=0;lm<op.multipole_count_;++lm)
        response[lm]+=phased*op.geometry_->form(ig*op.multipole_count_+lm,lm);
    }
    const auto entries=op.dimension_*op.dimension_;auto& out=results[atom];
    for(std::size_t lm=0;lm<op.multipole_count_;++lm)
      for(std::size_t entry=0;entry<entries;++entry)
        out[entry]+=response[lm]*op.density_transform_[lm*entries+entry];
  }
  return results;
}

Vec3 PawCompensationOperator::position_energy_gradient(const std::vector<Complex>& p,
    const FourierDensity& v) const {
  PerformanceTimer timer("paw_compensation_position_derivative");
  const auto lm_moments = multipoles(p);
  Vec3 out{};
  for (std::size_t ik = 0; ik < geometry_->keys.size(); ++ik) {
    const auto [h, k, l] = geometry_->keys[ik];
    const auto opposite = v.find({-h, -k, -l});
    if (opposite == v.end()) continue;
    Complex value{};
    for (std::size_t lm = 0; lm < multipole_count_; ++lm)
      value += geometry_->form(ik * multipole_count_ + lm,lm) * lm_moments[lm];
    const double response = (Complex(0.0, -1.0) * opposite->second * phases_[ik] * value).real();
    out = sum(out, scaled(geometry_->reciprocal_vectors[ik], response));
  }
  return out;
}

std::array<Vec3,3> PawCompensationOperator::shape_strain_energy_gradient(
    const std::vector<Complex>& p,const FourierDensity& v)const {
  PerformanceTimer timer("paw_compensation_shape_derivative");
  const auto lm_moments=multipoles(p);std::array<Vec3,3> out{};
  for(std::size_t ik=0;ik<geometry_->keys.size();++ik) {
    const auto [h,k,l]=geometry_->keys[ik];const auto potential=v.find({-h,-k,-l});if(potential==v.end())continue;
    const auto q=geometry_->reciprocal_vectors[ik];const double qs[]{q.x,q.y,q.z};
    for(int a=0;a<3;++a){
      Complex response{};
      for(std::size_t lm=0;lm<multipole_count_;++lm)response+=geometry_->gradient(a,ik*multipole_count_+lm,lm)*lm_moments[lm];
      for(int c=0;c<3;++c){auto& row=out[a];(c==0?row.x:c==1?row.y:row.z)-=qs[c]*(potential->second*phases_[ik]*response).real();}
    }
  }
  return out;
}

std::vector<PawCompensationOperator::Derivatives> PawCompensationOperator::energy_derivatives_batch(
    const std::vector<PawCompensationOperator>& atoms,
    const std::vector<std::vector<Complex>>& onsite,const FourierDensity& v) {
  if(atoms.size()!=onsite.size())throw std::invalid_argument("compensation derivative atom/density count mismatch");
  if(atoms.empty())return {};
  PerformanceTimer timer("paw_compensation_derivatives_batch");
  const auto& keys=atoms.front().geometry_->keys;
  std::vector<std::vector<Complex>> lm_moments;
  for(std::size_t atom=0;atom<atoms.size();++atom) {
    if(atoms[atom].geometry_->keys!=keys)
      throw std::invalid_argument("compensation derivative reciprocal support mismatch");
    lm_moments.push_back(atoms[atom].multipoles(onsite[atom]));
  }
  std::vector<Complex> sampled(keys.size());std::vector<unsigned char> present(keys.size());
  for(std::size_t ig=0;ig<keys.size();++ig) {
    const auto [h,k,l]=keys[ig];
    if(const auto found=v.find({-h,-k,-l});found!=v.end()) {
      sampled[ig]=found->second;present[ig]=1;
    }
  }
  std::vector<Derivatives> result(atoms.size());
  if(blas_compensation()&&!keys.empty()) {
    PerformanceTimer contraction_timer("paw_compensation_blas_derivatives");
    const auto groups=geometry_groups(atoms);
    std::vector<std::vector<Complex>> coefficients;
    for(const auto& members:groups) {
      coefficients.emplace_back();auto& c=coefficients.back();
      for(auto atom:members)c.insert(c.end(),lm_moments[atom].begin(),lm_moments[atom].end());
    }
    const auto tiles=(keys.size()+compensation_tile_size-1)/compensation_tile_size;
    std::exception_ptr failure;std::mutex failure_mutex;
    for(std::size_t begin=0;begin<tiles;begin+=compensation_tile_window) {
    const auto end=std::min(tiles,begin+compensation_tile_window);
    std::vector<std::vector<Derivatives>> partial(end-begin,std::vector<Derivatives>(atoms.size()));
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel()&&end-begin>1)
#endif
    for(std::size_t tile=begin;tile<end;++tile) {
      try {
        const auto first=tile*compensation_tile_size,count=std::min(compensation_tile_size,keys.size()-first);
        for(std::size_t group=0;group<groups.size();++group) {
          const auto& members=groups[group];const auto& geometry=*atoms[members[0]].geometry_;
          const auto values=geometry.contract_density_tile(first,count,coefficients[group],members.size());
          std::array<std::vector<Complex>,3> shapes;
          for(int axis=0;axis<3;++axis)shapes[axis]=geometry.contract_density_tile(first,count,coefficients[group],members.size(),axis);
          for(std::size_t col=0;col<members.size();++col) {
            const auto atom=members[col];auto& out=partial[tile-begin][atom];
            for(std::size_t i=0;i<count;++i) {
              const auto ig=first+i;if(!present[ig])continue;
              const auto q=geometry.reciprocal_vectors[ig];const double qs[]{q.x,q.y,q.z};
              const Complex phased=sampled[ig]*atoms[atom].phases_[ig];
              out.position=sum(out.position,scaled(q,(Complex(0,-1)*phased*values[col*count+i]).real()));
              for(int axis=0;axis<3;++axis)for(int c=0;c<3;++c) {
                auto& row=out.shape[axis];
                (c==0?row.x:c==1?row.y:row.z)-=qs[c]*(phased*shapes[axis][col*count+i]).real();
              }
            }
          }
        }
      }catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();}
    }
    if(failure)std::rethrow_exception(failure);
    for(std::size_t tile=begin;tile<end;++tile)for(std::size_t atom=0;atom<atoms.size();++atom) {
      result[atom].position=sum(result[atom].position,partial[tile-begin][atom].position);
      for(int axis=0;axis<3;++axis)result[atom].shape[axis]=sum(result[atom].shape[axis],partial[tile-begin][atom].shape[axis]);
    }
    }
    return result;
  }
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel() && keys.size()>=128 && atoms.size()>1)
#endif
  for(std::size_t atom=0;atom<atoms.size();++atom) {
    const auto& op=atoms[atom];auto& out=result[atom];
    for(std::size_t ig=0;ig<keys.size();++ig) {
      if(!present[ig])continue;
      const auto q=op.geometry_->reciprocal_vectors[ig];const double qs[]{q.x,q.y,q.z};
      Complex value{};
      for(std::size_t lm=0;lm<op.multipole_count_;++lm)
        value+=op.geometry_->form(ig*op.multipole_count_+lm,lm)*lm_moments[atom][lm];
      const double response=(Complex(0,-1)*sampled[ig]*op.phases_[ig]*value).real();
      out.position=sum(out.position,scaled(q,response));
      for(int a=0;a<3;++a) {
        Complex shape{};
        for(std::size_t lm=0;lm<op.multipole_count_;++lm)
          shape+=op.geometry_->gradient(a,ig*op.multipole_count_+lm,lm)*lm_moments[atom][lm];
        for(int c=0;c<3;++c) {
          auto& row=out.shape[a];
          (c==0?row.x:c==1?row.y:row.z)-=qs[c]*(sampled[ig]*op.phases_[ig]*shape).real();
        }
      }
    }
  }
  return result;
}

std::vector<double> paw_compensation_radial_form(const PawAugmentationData& a,double q_angstrom) {
  // Same radial quadrature and Bessel recurrence as the operator geometry;
  // used only by the density-spectrum probe, never by the SCF operator.
  if(a.max_l<0||a.compensation_r2.size()!=static_cast<std::size_t>(a.max_l+1))
    throw std::invalid_argument("incomplete PAW compensation multipoles");
  const auto weights=radial_weights(a.radial_grid);
  std::vector<double> integrals(static_cast<std::size_t>(a.max_l+1),0.);
  for(int L=0;L<=a.max_l;++L)for(std::size_t ir=0;ir<weights.size();++ir)
    integrals[L]+=weights[ir]*a.compensation_r2[L][ir]*bessel(L,q_angstrom*a.radial_grid[ir]);
  if(q_angstrom<1e-14)integrals[0]=1.;
  return integrals;
}
} // namespace sdft
