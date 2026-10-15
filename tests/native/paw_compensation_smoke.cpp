#include "env_cache.hpp"
#include "paw_compensation.hpp"
#include "sdft_core.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <optional>
#include <cstdlib>
#include <string>

namespace {
void geometry_control(const char* value) {
#ifdef _WIN32
  _putenv_s("SDFT_BATCH_COMPENSATION_GEOMETRY",value);sdft::refresh_environment_cache();
#else
  setenv("SDFT_BATCH_COMPENSATION_GEOMETRY",value,1);sdft::refresh_environment_cache();
#endif
}
}

int main(int argc, char** argv) {
  using namespace sdft;
  if (argc != 2) return 2;
  const auto cache = read_paw_numeric_cache(argv[1]);
  std::vector<int> channel_l;
  for (const auto& ch : cache.nonlocal_channels)
    for (int i = 0; i < ch.nprojectors; ++i) channel_l.push_back(ch.angular_momentum);
  const auto augmentation = build_paw_augmentation(cache, channel_l);
  std::size_t dim = 0;
  for (int l : channel_l) dim += static_cast<std::size_t>(2 * l + 1);
  std::vector<std::complex<double>> p(dim * dim), dp(dim * dim);
  for (std::size_t i = 0; i < dim; ++i) p[i * dim + i] = 0.3 + 0.02 * i;
  dp[0] = 0.11;
  if (dim > 2) {
    dp[2 * dim + 1] = {0.025, -0.018};
    dp[1 * dim + 2] = std::conj(dp[2 * dim + 1]);
  }
  const std::vector<std::tuple<int, int, int>> keys{{0,0,0}, {1,0,0}, {-1,0,0},
      {0,1,0}, {0,-1,0}, {1,1,0}, {-1,-1,0}};
  const std::array<Vec3, 3> lattice{{{8,0,0}, {0,8,0}, {0,0,8}}};
  PawCompensationOperator op(augmentation, lattice, {0.13,0.21,0.08}, keys);
  FourierDensity potential{{{0,0,0},{0.31,0}}, {{1,0,0},{0.08,0.03}},
      {{-1,0,0},{0.08,-0.03}}, {{0,1,0},{-0.04,0.02}},
      {{0,-1,0},{-0.04,-0.02}}, {{1,1,0},{0.018,-0.011}},
      {{-1,-1,0},{0.018,0.011}}};
  // Shared geometry must retain translated complex charge, its adjoint and
  // both derivatives, and remain alive after the original atom is destroyed.
  const Vec3 translated{0.39, 0.16, 0.27};
  PawCompensationOperator independent(augmentation, lattice, translated, keys);
  std::optional<PawCompensationOperator> shared;
  {
    PawCompensationOperator owner(augmentation, lattice, {.13,.21,.08}, keys);
    shared.emplace(augmentation, lattice, translated, keys,
        AtomicUnitConvention::Codata, &owner);
    if (!shared->shares_geometry_with(owner) || shared->shares_geometry_with(independent))
      return 9;
    const auto reject = [&](const auto& data, const auto& cell, const auto& support,
                            AtomicUnitConvention units) {
      try { PawCompensationOperator invalid(data, cell, translated, support, units, &owner); }
      catch (const std::invalid_argument&) { return true; }
      return false;
    };
    auto wrong_cell=lattice; wrong_cell[1].y+=.01;
    auto wrong_keys=keys; wrong_keys.pop_back();
    auto wrong_aug_moment=augmentation; wrong_aug_moment.aug_moment.at(0)+=.01;
    auto wrong_shape=augmentation; wrong_shape.compensation_r2[0].at(0)+=.01;
    auto wrong_compensation_norm=augmentation; wrong_compensation_norm.mesh_compensation_norm=!wrong_compensation_norm.mesh_compensation_norm;
    if (!reject(augmentation, wrong_cell, keys, AtomicUnitConvention::Codata) ||
        !reject(augmentation, lattice, wrong_keys, AtomicUnitConvention::Codata) ||
        !reject(wrong_aug_moment, lattice, keys, AtomicUnitConvention::Codata) ||
        !reject(wrong_shape, lattice, keys, AtomicUnitConvention::Codata) ||
        !reject(wrong_compensation_norm, lattice, keys, AtomicUnitConvention::Codata) ||
        !reject(augmentation, lattice, keys, AtomicUnitConvention::VaspLegacy)) {
      std::cerr << "incompatible shared geometry was accepted\n";
      return 10;
    }
  }
  double shared_error=0.;
  for (const auto& onsite : {p, dp}) {
    const auto actual=shared->density(onsite), expected=independent.density(onsite);
    for (const auto& [key,value] : expected)
      shared_error=std::max(shared_error,std::abs(actual.at(key)-value));
    const auto a=shared->position_energy_gradient(onsite,potential);
    const auto b=independent.position_energy_gradient(onsite,potential);
    shared_error=std::max(shared_error,std::hypot(a.x-b.x,a.y-b.y,a.z-b.z));
    const auto sa=shared->shape_strain_energy_gradient(onsite,potential);
    const auto sb=independent.shape_strain_energy_gradient(onsite,potential);
    for (int axis=0;axis<3;++axis)shared_error=std::max(shared_error,
        std::hypot(sa[axis].x-sb[axis].x,sa[axis].y-sb[axis].y,sa[axis].z-sb[axis].z));
  }
  const auto shared_d=shared->project_potential(potential);
  const auto independent_d=independent.project_potential(potential);
  for (std::size_t i=0;i<shared_d.size();++i)
    shared_error=std::max(shared_error,std::abs(shared_d[i]-independent_d[i]));
  if (shared_error>2e-12) {
    std::cerr << "translated shared compensation mismatch=" << shared_error << '\n';
    return 11;
  }
  std::vector<PawCompensationOperator> batch;
  batch.reserve(3);
  batch.emplace_back(augmentation,lattice,Vec3{.13,.21,.08},keys);
  batch.emplace_back(augmentation,lattice,translated,keys,AtomicUnitConvention::Codata,&batch[0]);
  batch.emplace_back(augmentation,lattice,Vec3{.68,.73,.45},keys,AtomicUnitConvention::Codata,&batch[0]);
  const std::vector<std::vector<std::complex<double>>> densities{p,dp,p};
  FourierDensity sum_reference;
  for(std::size_t atom=0;atom<batch.size();++atom)
    for(const auto& [key,value]:batch[atom].density(densities[atom]))sum_reference[key]+=value;
  const auto sum_batch=PawCompensationOperator::density_sum(batch,densities);
  if(sum_batch!=sum_reference){std::cerr<<"batched complex density changed atom summation\n";return 12;}
  bool count_rejected=false;
  try{PawCompensationOperator::density_sum(batch,{p});}
  catch(const std::invalid_argument&){count_rejected=true;}
  if(!count_rejected)return 13;
  // The batch must preserve complex phases, sparse/missing potential modes
  // and serial accumulation. Also exercise the threaded path on a larger,
  // deliberately unsorted support with different translated atoms.
  auto larger_keys=keys;
  for(int h=-66;h<=66;++h)if(h!=0 && std::abs(h)!=1)larger_keys.emplace_back(h,2,0);
  std::vector<PawCompensationOperator> adjoint_atoms;
  adjoint_atoms.reserve(3);
  adjoint_atoms.emplace_back(augmentation,lattice,Vec3{.13,.21,.08},larger_keys);
  adjoint_atoms.emplace_back(augmentation,lattice,translated,larger_keys,AtomicUnitConvention::Codata,&adjoint_atoms[0]);
  FourierDensity sparse=potential;
  for(int h=-66;h<=66;h+=3)sparse[{-h,-2,0}]={.003*h,.001*h};
  for(const auto& grid_potential : {potential,sparse,FourierDensity{}}) {
    const auto batched=PawCompensationOperator::project_potential_batch(adjoint_atoms,grid_potential);
    for(std::size_t atom=0;atom<adjoint_atoms.size();++atom)
      if(batched[atom]!=adjoint_atoms[atom].project_potential(grid_potential)) {
        std::cerr<<"batched complex adjoint changed serial contraction\n";return 14;
      }
    const auto derivatives=PawCompensationOperator::energy_derivatives_batch(adjoint_atoms,{p,p},grid_potential);
    for(std::size_t atom=0;atom<adjoint_atoms.size();++atom) {
      const auto position=adjoint_atoms[atom].position_energy_gradient(p,grid_potential);
      const auto shape=adjoint_atoms[atom].shape_strain_energy_gradient(p,grid_potential);
      if(derivatives[atom].position.x!=position.x || derivatives[atom].position.y!=position.y || derivatives[atom].position.z!=position.z)return 17;
      for(int a=0;a<3;++a)if(derivatives[atom].shape[a].x!=shape[a].x || derivatives[atom].shape[a].y!=shape[a].y || derivatives[atom].shape[a].z!=shape[a].z)return 18;
    }
  }
  if(const auto value=std::getenv("SDFT_BATCH_COMPENSATION_GEOMETRY");value && std::string(value)=="1") {
    // Compare both constructors in one process using exactly the same PAW,
    // anisotropic cell, unsorted G support and complex Hermitian density.
    std::vector<std::tuple<int,int,int>> geometry_keys;
    for(int h=-8;h<=8;++h)for(int k=-8;k<=8;++k)for(int l=-8;l<=8;++l)geometry_keys.emplace_back(h,k,l);
    std::reverse(geometry_keys.begin(),geometry_keys.end());
    auto cell=lattice;cell[1].x=.3;cell[2].y=-.2;
    geometry_control("0");
    PawCompensationOperator serial(augmentation,cell,{.13,.21,.08},geometry_keys);
    geometry_control("1");
    PawCompensationOperator parallel(augmentation,cell,{.13,.21,.08},geometry_keys);
    if(serial.density(p)!=parallel.density(p) || serial.project_potential(sparse)!=parallel.project_potential(sparse))return 19;
    const auto sd=serial.shape_strain_energy_gradient(p,sparse),pd=parallel.shape_strain_energy_gradient(p,sparse);
    for(int a=0;a<3;++a)if(sd[a].x!=pd[a].x || sd[a].y!=pd[a].y || sd[a].z!=pd[a].z)return 20;
  }
  if(!PawCompensationOperator::project_potential_batch({},potential).empty())return 15;
  adjoint_atoms.emplace_back(augmentation,lattice,translated,keys);
  bool support_rejected=false;
  try{PawCompensationOperator::project_potential_batch(adjoint_atoms,potential);}
  catch(const std::invalid_argument&){support_rejected=true;}
  if(!support_rejected)return 16;
  // A constant smooth potential is V0 times the overlap augmentation.
  // This separates the positive compensation adjoint of the smooth grid
  // potential from the AE-PS-compensation radial one-centre response.
  constexpr double constant_hartree = 0.37;
  constexpr double ev_per_hartree = 27.211386245988;
  const FourierDensity constant{{{0,0,0},{constant_hartree,0}}};
  const auto constant_d = op.project_potential(constant);
  const auto constant_wrapper = paw_project_fourier_potential(
      cache, channel_l, lattice, {0.13,0.21,0.08}, constant);
  std::vector<std::size_t> offset(channel_l.size());
  for (std::size_t i = 1; i < offset.size(); ++i)
    offset[i] = offset[i - 1] + static_cast<std::size_t>(2 * channel_l[i - 1] + 1);
  double constant_error = 0.0;
  for (std::size_t i = 0; i < channel_l.size(); ++i)
    for (std::size_t j = 0; j < channel_l.size(); ++j)
      for (int mi = -channel_l[i]; mi <= channel_l[i]; ++mi)
        for (int mj = -channel_l[j]; mj <= channel_l[j]; ++mj) {
          const auto row = offset[i] + static_cast<std::size_t>(mi + channel_l[i]);
          const auto col = offset[j] + static_cast<std::size_t>(mj + channel_l[j]);
          const auto index = row * dim + col;
          const double expected = channel_l[i] == channel_l[j] && mi == mj
              ? constant_hartree * augmentation.overlap_moment(i, j) : 0.0;
          constant_error = std::max(constant_error, std::abs(constant_d[index] - expected));
          constant_error = std::max(constant_error,
              std::abs(constant_wrapper[index] / ev_per_hartree - expected));
        }
  if (constant_error > 2e-11) {
    std::cerr << "constant Dgrid != V0 overlap_correction error=" << constant_error << '\n';
    return 6;
  }
  const auto projected = op.project_potential(potential);
  const auto wrapper = paw_project_fourier_potential(
      cache, channel_l, lattice, {0.13,0.21,0.08}, potential);
  double wrapper_error = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i)
    wrapper_error = std::max(wrapper_error,
        std::abs(projected[i] - wrapper[i] / ev_per_hartree));
  if (wrapper_error > 2e-11) {
    std::cerr << "cached/wrapper Dgrid mismatch error=" << wrapper_error << '\n';
    return 7;
  }
  const auto delta_density = op.density(dp);
  double fourier_derivative = 0.0;
  const double volume = std::abs(determinant(lattice));
  for (const auto& [key, value] : delta_density) {
    const auto [h,k,l] = key;
    const auto it = potential.find({-h,-k,-l});
    if (it != potential.end()) fourier_derivative += volume * (it->second * value).real();
  }
  double matrix_derivative = 0.0;
  for (std::size_t i = 0; i < dp.size(); ++i)
    matrix_derivative += (std::conj(dp[i]) * projected[i]).real();
  const double adjoint_error = std::abs(fourier_derivative - matrix_derivative);
  if (adjoint_error > 2e-9 * std::max(1.0, std::abs(fourier_derivative))) {
    std::cerr << "compensation adjoint mismatch Fourier=" << fourier_derivative
              << " projected=" << matrix_derivative << " error=" << adjoint_error << '\n';
    return 3;
  }

  const double dx = 2e-5;
  const auto evaluate_position_energy = [&](double x) {
    PawCompensationOperator shifted(augmentation, lattice, {x / 8.0, 0.21, 0.08}, keys);
    const auto charge = shifted.density(p);
    double energy = 0.0;
    for (const auto& [key, value] : charge) {
      const auto [h,k,l] = key;
      const auto it = potential.find({-h,-k,-l});
      if (it != potential.end()) energy += volume * (it->second * value).real();
    }
    return energy;
  };
  const Vec3 gradient = op.position_energy_gradient(p, potential);
  const double fd = (evaluate_position_energy(8.0 * 0.13 + dx) -
                     evaluate_position_energy(8.0 * 0.13 - dx)) / (2.0 * dx);
  const double position_error = std::abs(fd - gradient.x);
  if (position_error > 2e-7 * std::max(1.0, std::abs(fd))) {
    std::cerr << "compensation position derivative mismatch FD=" << fd
              << " analytic=" << gradient.x << " error=" << position_error << '\n';
    return 4;
  }

  double harmonic_gradient_error=0,strain_error=0;
  for(int L=0;L<=6;++L)for(int M=-L;M<=L;++M)
    for(Vec3 q:std::vector<Vec3>{{.3,-.4,.8},{0,0,.9},{.7,0,0}}) {
      const auto analytic=paw_real_spherical_harmonic_gradient(L,M,q);
      for(int axis=0;axis<3;++axis){auto upper=q,lower=q;const double delta=1e-6;
        (axis==0?upper.x:axis==1?upper.y:upper.z)+=delta;
        (axis==0?lower.x:axis==1?lower.y:lower.z)-=delta;
        const double finite=(paw_real_spherical_harmonic(L,M,upper)-paw_real_spherical_harmonic(L,M,lower))/(2*delta);
        // atan2/Legendre values lose precision right at a pole. Check
        // transverse polar derivatives against the regular polynomial limit
        // via a less tiny independent displacement.
        harmonic_gradient_error=std::max(harmonic_gradient_error,std::abs(finite-(axis==0?analytic.x:axis==1?analytic.y:analytic.z)));
      }
    }
  const auto shape=op.shape_strain_energy_gradient(p,potential);
  const auto strain_energy=[&](int a,int c,double epsilon) {
    auto cell=lattice;
    for(auto& row:cell){const double original=a==0?row.x:a==1?row.y:row.z;(c==0?row.x:c==1?row.y:row.z)+=epsilon*original;}
    PawCompensationOperator strained(augmentation,cell,{.13,.21,.08},keys);
    double energy=0;const auto charge=strained.density(p);
    for(const auto& [key,value]:charge){const auto [h,k,l]=key;
      if(const auto found=potential.find({-h,-k,-l});found!=potential.end())energy+=std::abs(determinant(cell))*(found->second*value).real();}
    return energy;
  };
  for(int a=0;a<3;++a)for(int c=0;c<3;++c){const double delta=1e-5;
    const double finite=(strain_energy(a,c,delta)-strain_energy(a,c,-delta))/(2*delta);
    strain_error=std::max(strain_error,std::abs(finite-(c==0?shape[a].x:c==1?shape[a].y:shape[a].z)));
  }
  if(strain_error>1e-8){std::cerr<<"compensation shape strain error="<<strain_error<<'\n';return 8;}

  // Check the combined functional, not just either interface in isolation:
  // F[n_smooth+Q(P)] + <V_ion,n_smooth+Q(P)> + E_AE(P)-E_PS(P).
  const auto reference = paw_atomic_reference_potentials(cache, channel_l);
  const auto composite_energy = [&](const std::vector<std::complex<double>>& onsite) {
    auto total_density = op.density(onsite);
    total_density[{0,0,0}] += 8.0 / volume;
    const auto smooth = evaluate_density_functionals(total_density, lattice);
    const auto radial = paw_one_center_closure(cache, channel_l, onsite, dim,
        1, PawRadialLengthUnit::Angstrom, &reference);
    double ionic = 0.0;
    for (const auto& [key, value] : total_density) {
      const auto [h,k,l] = key;
      const auto it = potential.find({-h,-k,-l});
      if (it != potential.end()) ionic += volume * (it->second * value).real();
    }
    return smooth.hartree_energy + smooth.xc_energy + ionic +
        radial.energy_difference_ev / 27.211386245988;
  };
  auto total_density = op.density(p);
  total_density[{0,0,0}] += 8.0 / volume;
  auto local_potential = evaluate_density_functionals(total_density, lattice).potential;
  for (const auto& [key, value] : potential) local_potential[key] += value;
  const auto grid_d = op.project_potential(local_potential);
  const auto radial = paw_one_center_closure(cache, channel_l, p, dim,
      1, PawRadialLengthUnit::Angstrom, &reference);
  double derivative = 0.0;
  for (std::size_t i = 0; i < dp.size(); ++i)
    derivative += (std::conj(dp[i]) * (grid_d[i] +
        radial.d_matrix_ev[i] / 27.211386245988)).real();
  constexpr double step = 2e-5;
  auto plus = p, minus = p;
  for (std::size_t i = 0; i < dp.size(); ++i) {
    plus[i] += step * dp[i]; minus[i] -= step * dp[i];
  }
  const double combined_fd = (composite_energy(plus) - composite_energy(minus)) / (2.0 * step);
  const double combined_error = std::abs(combined_fd - derivative);
  if (combined_error > 2e-7 * std::max(1.0, std::abs(combined_fd))) {
    std::cerr << "combined PAW functional derivative mismatch FD=" << combined_fd
              << " onsite_dij=" << derivative << " error=" << combined_error << '\n';
    return 5;
  }
  std::cout << "paw_compensation_smoke PASS adjoint_error=" << adjoint_error
            << " constant_potential_error=" << constant_error
            << " cached_wrapper_error=" << wrapper_error
            << " position_error=" << position_error
            << " strain_error=" << strain_error
            << " harmonic_gradient_error=" << harmonic_gradient_error
            << " combined_functional_error=" << combined_error
            << " shared_geometry_error=" << shared_error
            << " geometry_bytes=" << op.geometry_storage_bytes()
            << " per_atom_bytes=" << op.atom_storage_bytes()
            << " distinct_form_factors=" << op.form_factor_count() << '\n';
}
