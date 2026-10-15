#include "env_cache.hpp"
#include "density_functionals.hpp"
#include "pbe_xc.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  using namespace sdft;
  // Independent O(Npw^2) contraction tests complex phases, off-axis modes,
  // fractional occupations, normalization and absence of circular aliasing.
  const std::vector<std::tuple<int,int,int>> waves{{-2,1,0},{1,-1,2},{0,0,0},{2,2,-1}};
  const std::vector<std::complex<double>> coefficients{{.2,.1},{-.4,.3},{.7,-.2},{.1,.4},
      {-.1,.3},{.2,.5},{.4,.2},{-.3,-.2}};
  const std::vector<double> occupations{2.,.7};
  FourierDensity exact;
  for(std::size_t band=0;band<2;++band)for(std::size_t i=0;i<waves.size();++i)for(std::size_t j=0;j<waves.size();++j) {
    const auto key=std::make_tuple(std::get<0>(waves[i])-std::get<0>(waves[j]),
        std::get<1>(waves[i])-std::get<1>(waves[j]),std::get<2>(waves[i])-std::get<2>(waves[j]));
    exact[key]+=occupations[band]*coefficients[band*waves.size()+i]*
        std::conj(coefficients[band*waves.size()+j])/64.;
  }
  const auto transformed=plane_wave_density_fft(waves,coefficients,occupations,64.,exact);
  const auto support=plane_wave_difference_support(waves);
  assert(support.size()==exact.size());
  for(const auto& [key,value]:exact)assert(support.count(key)==1);
  for(const auto& [key,value]:exact)assert(std::abs(transformed.at(key)-value)<1e-14);
  for(const char* threads:{"1","2","4","8"}){
    setenv("SDFT_DENSITY_BAND_THREADS",threads,1);sdft::refresh_environment_cache();
    const auto parallel=plane_wave_density_fft(waves,coefficients,occupations,64.,exact);
    for(const auto& [key,value]:exact)if(std::abs(parallel.at(key)-value)>1e-14)return 10;
    auto fractional=occupations;fractional[0]=0.;fractional[1]=.37;
    const auto zero_band=plane_wave_density_fft(waves,coefficients,fractional,64.,exact);
    FourierDensity separate;
    for(std::size_t i=0;i<waves.size();++i)for(std::size_t j=0;j<waves.size();++j){
      const auto key=std::make_tuple(std::get<0>(waves[i])-std::get<0>(waves[j]),std::get<1>(waves[i])-std::get<1>(waves[j]),std::get<2>(waves[i])-std::get<2>(waves[j]));
      separate[key]+=.37*coefficients[waves.size()+i]*std::conj(coefficients[waves.size()+j])/64.;
    }
    for(const auto& [key,value]:separate)if(std::abs(zero_band.at(key)-value)>1e-14)return 11;
  }
  unsetenv("SDFT_DENSITY_BAND_THREADS");sdft::refresh_environment_cache();
  const std::array<Vec3, 3> lattice{{{4.0,0,0},{0,4.0,0},{0,0,4.0}}};
  FourierDensity density{{{0,0,0}, {0.5,0.0}}, {{1,0,0}, {0.01,0.0}}, {{-1,0,0}, {0.01,0.0}}};
  const auto result = evaluate_density_functionals(density, lattice);
  const double volume = 64.0;
  const double q = 2.0 * 3.14159265358979323846 / 4.0;
  const double expected_hartree = 4.0 * 3.14159265358979323846 * volume * 0.01 * 0.01 / (q*q);
  assert(std::abs(result.sampled_electrons - 32.0) < 1e-11);
  assert(std::abs(result.hartree_energy - expected_hartree) < 1e-11);
  assert(result.minimum_density > 0.0 && result.clipped_density_points == 0);
  assert(std::isfinite(result.xc_energy) && std::isfinite(result.vxc_density));
  const auto uniform = evaluate_density_functionals({{{0,0,0}, {0.5,0.0}}}, lattice);
  const auto expected_xc = pbe_unpolarized(0.5, 0.0);
  assert(std::abs(uniform.sampled_electrons - 32.0) < 1e-11);
  assert(std::abs(uniform.xc_energy - volume * expected_xc.energy_density) < 1e-11);
  assert(std::abs(uniform.vxc_density - volume * 0.5 *
      (expected_xc.exchange_potential + expected_xc.correlation_potential)) < 1e-11);
  // Test the full periodic GGA functional derivative, including the spectral
  // divergence, on a complex mode and a constant-density perturbation.
  density[{1,0,0}] = {0.02,0.006}; density[{-1,0,0}] = {0.02,-0.006};
  density[{0,1,0}] = {0.003,-0.005}; density[{0,-1,0}] = {0.003,0.005};
  const FourierDensity delta{{{0,0,0}, {0.01,0.0}}, {{1,0,0}, {0.01,0.02}},
                            {{-1,0,0}, {0.01,-0.02}}, {{0,1,0}, {-0.01,0.01}},
                            {{0,-1,0}, {-0.01,-0.01}}};
  const auto full = evaluate_density_functionals(density, lattice);
  auto plus = density, minus = density;
  const double step = 1e-4;
  double potential_derivative = 0.0;
  for (const auto& [key, value] : delta) {
    plus[key] += step * value; minus[key] -= step * value;
    const auto opposite = std::make_tuple(-std::get<0>(key), -std::get<1>(key), -std::get<2>(key));
    potential_derivative += volume * (value * full.potential.at(opposite)).real();
  }
  const auto upper = evaluate_density_functionals(plus, lattice);
  const auto lower = evaluate_density_functionals(minus, lattice);
  const double energy_derivative = (upper.hartree_energy + upper.xc_energy - lower.hartree_energy - lower.xc_energy) / (2.0 * step);
  assert(std::abs(energy_derivative - potential_derivative) < 1e-8);

  // Nonlinear core correction: XC sees valence+core, while Hartree and
  // the XC double-counting contraction remain valence-only.
  const FourierDensity uniform_core{{{0,0,0}, {0.2,0.0}}};
  const auto uniform_with_core = evaluate_density_functionals({{{0,0,0}, {0.5,0.0}}}, lattice, &uniform_core);
  const auto combined_xc = pbe_unpolarized(0.7, 0.0);
  const double combined_potential = combined_xc.exchange_potential + combined_xc.correlation_potential;
  assert(std::abs(uniform_with_core.sampled_electrons - 32.0) < 1e-11);
  assert(uniform_with_core.hartree_energy == 0.0);
  assert(std::abs(uniform_with_core.xc_energy - volume * combined_xc.energy_density) < 1e-11);
  assert(std::abs(uniform_with_core.vxc_density - volume * 0.5 * combined_potential) < 1e-11);
  assert(std::abs(uniform_with_core.xc_potential.at({0,0,0}).real() - combined_potential) < 1e-11);

  FourierDensity core{{{0,0,0}, {0.2,0.0}}, {{1,0,0}, {0.012,0.006}},
                      {{-1,0,0}, {0.012,-0.006}}, {{2,0,0}, {0.004,0.003}},
                      {{-2,0,0}, {0.004,-0.003}}};
  const auto with_core = evaluate_density_functionals(density, lattice, &core);
  // Compare the packed two-real-field transform against independent complex
  // transforms, including core modes absent from the valence support.
  setenv("SDFT_PAIRED_REAL_FFT","0",1);sdft::refresh_environment_cache();
  const auto separate=evaluate_density_functionals(density,lattice,&core);
  unsetenv("SDFT_PAIRED_REAL_FFT");sdft::refresh_environment_cache();
  assert(std::abs(separate.xc_energy-with_core.xc_energy)<1e-12);
  assert(std::abs(separate.vxc_density-with_core.vxc_density)<1e-12);
  for(const auto& [key,value]:with_core.potential)assert(std::abs(value-separate.potential.at(key))<1e-12);
  for(int a=0;a<3;++a){
    assert(std::abs(with_core.xc_stress[a].x-separate.xc_stress[a].x)<1e-12);
    assert(std::abs(with_core.xc_stress[a].y-separate.xc_stress[a].y)<1e-12);
    assert(std::abs(with_core.xc_stress[a].z-separate.xc_stress[a].z)<1e-12);
  }
  for(int changed=0;changed<3;++changed){
    auto changed_core=core;auto changed_lattice=lattice;
    if(changed==1){changed_core[{1,0,0}]+=std::complex<double>(.002,.003);changed_core[{-1,0,0}]=std::conj(changed_core[{1,0,0}]);}
    if(changed==2)changed_lattice[0].x*=1.01;
    setenv("SDFT_FROZEN_CORE_FFT_REUSE","1",1);sdft::refresh_environment_cache();setenv("SDFT_DENSITY_TILED_REDUCTION","1",1);sdft::refresh_environment_cache();
    const auto cached=evaluate_density_functionals(density,changed_lattice,&changed_core);
    setenv("SDFT_FROZEN_CORE_FFT_REUSE","0",1);sdft::refresh_environment_cache();setenv("SDFT_DENSITY_TILED_REDUCTION","0",1);sdft::refresh_environment_cache();
    const auto reference=evaluate_density_functionals(density,changed_lattice,&changed_core);
    if(std::abs(cached.xc_energy-reference.xc_energy)>1e-11||std::abs(cached.vxc_density-reference.vxc_density)>1e-11)return 17;
    for(const auto& [key,value]:cached.potential)if(std::abs(value-reference.potential.at(key))>1e-12)return 18;
  }
  unsetenv("SDFT_FROZEN_CORE_FFT_REUSE");sdft::refresh_environment_cache();unsetenv("SDFT_DENSITY_TILED_REDUCTION");sdft::refresh_environment_cache();
  assert(std::abs(with_core.hartree_energy - full.hartree_energy) < 1e-13);
  assert(std::abs(with_core.sampled_electrons - full.sampled_electrons) < 1e-11);
  assert(std::abs(with_core.xc_energy - full.xc_energy) > 1e-3);
  assert(with_core.xc_potential.count({2,0,0}) == 1);
  double valence_xc_expectation = 0.0;
  double valence_derivative = 0.0;
  for (const auto& [key, rho] : density) {
    const auto opposite = std::make_tuple(-std::get<0>(key), -std::get<1>(key), -std::get<2>(key));
    valence_xc_expectation += volume * (rho * with_core.xc_potential.at(opposite)).real();
  }
  for (const auto& [key, value] : delta) {
    const auto opposite = std::make_tuple(-std::get<0>(key), -std::get<1>(key), -std::get<2>(key));
    valence_derivative += volume * (value * with_core.potential.at(opposite)).real();
  }
  assert(std::abs(with_core.vxc_density - valence_xc_expectation) < 1e-11);
  const auto core_upper = evaluate_density_functionals(plus, lattice, &core);
  const auto core_lower = evaluate_density_functionals(minus, lattice, &core);
  const double core_energy_derivative = (core_upper.hartree_energy + core_upper.xc_energy -
      core_lower.hartree_energy - core_lower.xc_energy) / (2.0 * step);
  assert(std::abs(core_energy_derivative - valence_derivative) < 1e-8);

  // The derivative of a translated frozen core contracts XC only. The
  // h=2 mode is absent in valence and still needs a returned XC potential.
  auto translated_plus = core, translated_minus = core;
  double core_position_derivative = 0.0;
  for (const auto& [key, rho] : core) {
    const double qx = std::get<0>(key) * q;
    translated_plus[key] *= std::exp(std::complex<double>(0.0, -qx * step));
    translated_minus[key] *= std::exp(std::complex<double>(0.0, qx * step));
    const auto opposite = std::make_tuple(-std::get<0>(key), -std::get<1>(key), -std::get<2>(key));
    core_position_derivative += volume *
        (std::complex<double>(0.0, -qx) * rho * with_core.xc_potential.at(opposite)).real();
  }
  const auto shifted_upper = evaluate_density_functionals(density, lattice, &translated_plus);
  const auto shifted_lower = evaluate_density_functionals(density, lattice, &translated_minus);
  assert(std::abs(shifted_upper.hartree_energy - with_core.hartree_energy) < 1e-13);
  assert(std::abs(shifted_lower.hartree_energy - with_core.hartree_energy) < 1e-13);
  const double translated_derivative = (shifted_upper.xc_energy - shifted_lower.xc_energy) / (2.0 * step);
  assert(std::abs(translated_derivative - core_position_derivative) < 1e-8);

  // Homogeneous strain at fixed fractional Fourier amplitudes. All nine
  // components and a frozen core are checked against the same energy.
  const auto strain_energy=[&](int a,int c,double epsilon) {
    auto cell=lattice;for(auto& row:cell){const double original=c==0?row.x:c==1?row.y:row.z;
      (a==0?row.x:a==1?row.y:row.z)+=epsilon*original;}
    const double factor=volume/std::abs(determinant(cell));auto rho=density,ncore=core;
    for(auto& [key,value]:rho)value*=factor;
    for(auto& [key,value]:ncore)value*=factor;
    const auto evaluated=evaluate_density_functionals(rho,cell,ncore);
    return evaluated.hartree_energy+evaluated.xc_energy;
  };
  double strain_error=0;
  for(int a=0;a<3;++a)for(int c=0;c<3;++c) {
    const double delta=1e-5;
    const double finite=(strain_energy(a,c,delta)-strain_energy(a,c,-delta))/(2*delta*volume);
    const auto row=Vec3{with_core.hartree_stress[a].x+with_core.xc_stress[a].x,
      with_core.hartree_stress[a].y+with_core.xc_stress[a].y,
      with_core.hartree_stress[a].z+with_core.xc_stress[a].z};
    const double analytic=c==0?row.x:c==1?row.y:row.z;
    strain_error=std::max(strain_error,std::abs(finite-analytic));
  }
  if(strain_error>1e-8){std::cerr<<"GGA/Hartree strain error="<<strain_error<<'\n';return 2;}
  if(const char* fast=std::getenv("SDFT_FAST_DENSITY_GRID");fast&&std::string(fast)=="1") {
    // Exercise actual large-prime axes rather than only grids that already
    // have smooth dimensions. Small Hermitian amplitudes permit a tight
    // independent quadrature comparison without clipping the density.
    FourierDensity high{{{0,0,0},.5},{{11,0,0},std::complex<double>(1e-5,2e-6)},
      {{-11,0,0},std::complex<double>(1e-5,-2e-6)},
      {{0,15,0},std::complex<double>(2e-6,-3e-6)},{{0,-15,0},std::complex<double>(2e-6,3e-6)},
      {{0,0,8},std::complex<double>(-3e-6,1e-6)},{{0,0,-8},std::complex<double>(-3e-6,-1e-6)}};
    setenv("SDFT_FAST_DENSITY_GRID","0",1);sdft::refresh_environment_cache();
    const auto original=evaluate_density_functionals(high,lattice,uniform_core);
    setenv("SDFT_FAST_DENSITY_GRID","1",1);sdft::refresh_environment_cache();
    const auto improved=evaluate_density_functionals(high,lattice,uniform_core);
    setenv("SDFT_DENSITY_MEMORY_PARALLEL","0",1);sdft::refresh_environment_cache();
    const auto memory_reference=evaluate_density_functionals(high,lattice,uniform_core);
    setenv("SDFT_DENSITY_MEMORY_PARALLEL","1",1);sdft::refresh_environment_cache();
    const auto memory_parallel=evaluate_density_functionals(high,lattice,uniform_core);
    unsetenv("SDFT_DENSITY_MEMORY_PARALLEL");sdft::refresh_environment_cache();
    setenv("SDFT_DENSITY_TILED_REDUCTION","1",1);sdft::refresh_environment_cache();
    setenv("SDFT_DENSITY_FUSE_PBE_TILES","0",1);sdft::refresh_environment_cache();
    const auto unfused=evaluate_density_functionals(high,lattice,uniform_core);
    setenv("SDFT_DENSITY_FUSE_PBE_TILES","1",1);sdft::refresh_environment_cache();
    const auto fused=evaluate_density_functionals(high,lattice,uniform_core);
    unsetenv("SDFT_DENSITY_FUSE_PBE_TILES");sdft::refresh_environment_cache();
    if(std::abs(unfused.xc_energy-fused.xc_energy)>1e-12||
       std::abs(unfused.vxc_density-fused.vxc_density)>1e-12||
       std::abs(unfused.sampled_electrons-fused.sampled_electrons)>1e-12)return 24;
    for(const auto& [key,value]:unfused.potential)
      if(std::abs(value-fused.potential.at(key))>1e-12)return 25;
    for(int a=0;a<3;++a){const auto u=unfused.xc_stress[a],v=fused.xc_stress[a];
      if(std::max({std::abs(u.x-v.x),std::abs(u.y-v.y),std::abs(u.z-v.z)})>1e-12)return 26;}
    if(memory_reference.grid_points!=memory_parallel.grid_points||
       std::abs(memory_reference.xc_energy-memory_parallel.xc_energy)>1e-12||
       std::abs(memory_reference.vxc_density-memory_parallel.vxc_density)>1e-12||
       std::abs(memory_reference.sampled_electrons-memory_parallel.sampled_electrons)>1e-12)return 21;
    for(const auto& [key,value]:memory_reference.potential)
      if(std::abs(value-memory_parallel.potential.at(key))>1e-12)return 22;
    for(int axis=0;axis<3;++axis){const auto a=memory_reference.xc_stress[axis],b=memory_parallel.xc_stress[axis];
      if(std::max({std::abs(a.x-b.x),std::abs(a.y-b.y),std::abs(a.z-b.z)})>1e-12)return 23;}
    if(improved.grid_points<=original.grid_points||
       std::abs(original.xc_energy-improved.xc_energy)>1e-10||
       std::abs(original.sampled_electrons-improved.sampled_electrons)>1e-9)return 12;
    for(const auto& [key,value]:original.potential)
      if(std::abs(value-improved.potential.at(key))>1e-10)return 13;
    const double epsilon=1e-4;
    auto hi=high,lo=high;hi[{11,0,0}]+=epsilon*1e-3;hi[{-11,0,0}]+=epsilon*1e-3;
    lo[{11,0,0}]-=epsilon*1e-3;lo[{-11,0,0}]-=epsilon*1e-3;
    const auto upper=evaluate_density_functionals(hi,lattice,uniform_core);
    const auto lower=evaluate_density_functionals(lo,lattice,uniform_core);
    const double finite=(upper.hartree_energy+upper.xc_energy-lower.hartree_energy-lower.xc_energy)/(2*epsilon);
    const double response=volume*1e-3*(improved.potential.at({11,0,0})+improved.potential.at({-11,0,0})).real();
    if(std::abs(finite-response)>1e-8)return 14;
  }
  std::cout << "sdft_density_functionals_smoke PASS\n";
}
