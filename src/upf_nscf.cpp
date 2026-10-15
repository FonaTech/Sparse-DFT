#include "paw_framework.hpp"
#include "paw_site_model.hpp"
#include "upf_paw_operator.hpp"
#include "pw_seed_transport.hpp"
#include "compute_backend.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <iomanip>

namespace sdft {
BandPathResult run_upf_nscf(const ScfInput& input,const ScfResult& scf,
                           const std::vector<Vec3>& path,int bands) {
  PerformanceTimer timer("nscf_upf_total");
  constexpr double pi=3.1415926535897932384626433832795;
  const double ev=shared_paw_atomic_units(input).ev_per_hartree;
  using C=std::complex<double>;
  if(!scf.converged||(scf.paw_framework!="upf_paw"&&(scf.onecenter_radial_mode!="per_site_original_variational_paw_development" && scf.onecenter_radial_mode!="per_site_original_energy_adjoint_paw_development"))||input.spin_channels!=1||scf.spin_channels!=1||
     scf.unit_convention!=input.unit_convention||scf.onecenter_states.size()!=input.atoms.size()||
     scf.hamiltonian_density_q.empty()||path.empty())
    throw std::invalid_argument("UPF NSCF requires a complete converged same-model frozen PAW state");
  for(int axis=0;axis<3;++axis){const auto a=input.lattice_bohr[axis],b=scf.onecenter_lattice_bohr[axis];
    if(std::abs(a.x-b.x)>1e-12||std::abs(a.y-b.y)>1e-12||std::abs(a.z-b.z)>1e-12)
      throw std::invalid_argument("UPF NSCF lattice differs from frozen SCF");}
  if(std::abs(input.cutoff_ev-scf.cutoff_ev)>1e-10||
     (input.density_cutoff_ev>0&&std::abs(input.density_cutoff_ev-scf.density_cutoff_ev)>1e-10))
    throw std::invalid_argument("UPF NSCF cutoff differs from frozen SCF");
  const auto& a=input.lattice_bohr;const double det=determinant(a),volume=std::abs(det);
  auto scale=[](Vec3 v,double s){return Vec3{v.x*s,v.y*s,v.z*s};};
  const std::array<Vec3,3> reciprocal{scale(cross(a[1],a[2]),2*pi/det),
    scale(cross(a[2],a[0]),2*pi/det),scale(cross(a[0],a[1]),2*pi/det)};
  auto cart=[](const std::array<Vec3,3>& a,Vec3 f){return Vec3{a[0].x*f.x+a[1].x*f.y+a[2].x*f.z,
    a[0].y*f.x+a[1].y*f.y+a[2].y*f.z,a[0].z*f.x+a[1].z*f.y+a[2].z*f.z};};
  const auto func=scf.frozen_channel_functional?*scf.frozen_channel_functional:evaluate_density_functionals(scf.hamiltonian_density_q,a,scf.frozen_core_q);
  auto potential=func.potential;
  std::vector<std::shared_ptr<UpfPawData>> data;
  std::vector<std::shared_ptr<UpfPawOperator>> atomic;
  std::map<std::string,std::size_t> identities;
  std::set<int> angular;
  std::vector<std::vector<int>> site_l;
  for(std::size_t ia=0;ia<input.atoms.size();++ia){const auto& atom=input.atoms[ia];const auto& state=scf.onecenter_states[ia];
    const auto hash=file_sha256(atom.paw_cache);
    if(state.atom_index!=ia||state.symbol!=atom.symbol||state.numeric_cache_sha256!=hash||
       std::abs(state.fractional.x-atom.fractional.x)>1e-12||std::abs(state.fractional.y-atom.fractional.y)>1e-12||std::abs(state.fractional.z-atom.fractional.z)>1e-12)
      throw std::invalid_argument("UPF frozen atom, PAW SHA or magnetic model differs");
    if(paw_framework(atom.paw_cache)==PawFramework::Upf) {
    auto found=identities.find(hash);
    if(found==identities.end()){
      auto parsed=std::make_shared<UpfPawData>(read_native_upf_paw(atom.paw_cache));
      identities[hash]=data.size();data.push_back(parsed);atomic.push_back(std::make_shared<UpfPawOperator>(parsed));
    }else {data.push_back(data[found->second]);atomic.push_back(atomic[found->second]);}
    if(data.back()->element!=atom.symbol||data.back()->original_sha256!=state.record_id||
       data.back()->z_valence!=atom.zval||atomic.back()->dimension()!=state.projector_dimension||
       state.onsite_dij_ev.size()!=state.projector_dimension*state.projector_dimension)
      throw std::invalid_argument("UPF frozen identity or matrix dimensions differ");
      site_l.push_back(data.back()->projector_l);
    } else {
      std::vector<std::tuple<int,int,int>> keys;std::vector<Vec3> gv;
      for(const auto& [key,value]:scf.hamiltonian_density_q) {
        auto[h,k,l]=key;keys.push_back(key);gv.push_back(cart(reciprocal,{double(h),double(k),double(l)}));
      }
      const auto site=prepare_potcar_site_geometry(input,ia,keys,gv,{{}});
      if(site->record_id!=state.record_id||site->dimension()!=state.projector_dimension||
          site->numeric_cache_version!=state.numeric_cache_version)
        throw std::invalid_argument("POTCAR frozen site identity differs");
      for(std::size_t g=0;g<keys.size();++g)potential[keys[g]]+=site->local[g]*
          std::exp(C(0,-dot(gv[g],cart(a,atom.fractional))))/volume;
      site_l.push_back(site->radial_channel_l);data.push_back(nullptr);atomic.push_back(nullptr);
    }
    for(std::size_t i=0;i<state.projector_dimension;++i)for(std::size_t j=0;j<state.projector_dimension;++j){
      const auto z=state.onsite_dij_ev[i*state.projector_dimension+j];
      if(!std::isfinite(z.real())||!std::isfinite(z.imag())||std::abs(z-std::conj(state.onsite_dij_ev[j*state.projector_dimension+i]))>1e-10)
        throw std::invalid_argument("UPF frozen onsite D_ij must be finite Hermitian");}
    angular.insert(site_l.back().begin(),site_l.back().end());
    if(!data.back())continue;
    std::map<double,double> radial;
    for(const auto& [key,value]:scf.hamiltonian_density_q){auto[h,k,l]=key;
      const auto g=cart(reciprocal,{double(h),double(k),double(l)});const double q=std::sqrt(dot(g,g));
      auto [it,new_q]=radial.try_emplace(q,0.);
      if(new_q)it->second=.5*upf_local_form_ry(*data.back(),q);
      potential[key]+=it->second*std::exp(C(0,-dot(g,cart(a,atom.fractional))))/volume;
    }
  }
  BandPathResult out;out.kpoints=path;out.nkpoints=path.size();out.bands=bands>0?bands:scf.occupied_bands+4;
  out.fermi_energy_ev=scf.fermi_energy_ev;out.onecenter_dij_reused=true;out.onecenter_states=scf.onecenter_states;
  out.solver_method="matrix_free_fft_block_davidson";out.max_xc_grid_points=func.grid_points;
  std::map<int,std::size_t> orbital;
  for(int l:angular){orbital[l]=out.projection_orbitals.size();out.projection_orbitals.push_back(l==0?"s":l==1?"p":l==2?"d":l==3?"f":"l="+std::to_string(l));}
  for(const auto& atom:input.atoms)out.projection_atom_symbols.push_back(atom.symbol);
  struct Spectrum {std::vector<double> values,projections;double residual;};
  std::map<std::tuple<double,double,double>,Spectrum> cache;
  std::vector<GVector> previous_basis;std::vector<C> previous;
  for(const auto& point:path){
    if(!std::isfinite(point.x)||!std::isfinite(point.y)||!std::isfinite(point.z))throw std::invalid_argument("nonfinite UPF NSCF k point");
    const auto key=std::make_tuple(point.x,point.y,point.z);auto found=cache.find(key);
    if(found!=cache.end()){
      out.eigenvalues_ev.insert(out.eigenvalues_ev.end(),found->second.values.begin(),found->second.values.end());
      out.projector_weights.insert(out.projector_weights.end(),found->second.projections.begin(),found->second.projections.end());
      out.eigensolver_max_residuals_hartree.push_back(found->second.residual);out.seed_transport_retained_fraction.push_back(0.);
      ++out.identical_k_spectrum_reuses;continue;
    }
    auto reference_reciprocal=reciprocal;
    if(input.basis_reference_lattice_bohr){const auto& reference=*input.basis_reference_lattice_bohr;const double d=determinant(reference);
      reference_reciprocal={scale(cross(reference[1],reference[2]),2*pi/d),scale(cross(reference[2],reference[0]),2*pi/d),scale(cross(reference[0],reference[1]),2*pi/d)};}
    auto basis=enumerate_g_vectors_fixed_reference(reciprocal,reference_reciprocal,input.cutoff_ev,point,ev);
    if(std::size_t(out.bands)>basis.size())throw std::invalid_argument("UPF NSCF roots exceed basis");
    const auto shift=cart(reciprocal,point);std::vector<Vec3> vectors;
    for(const auto& g:basis)vectors.push_back({g.cartesian.x+shift.x,g.cartesian.y+shift.y,g.cartesian.z+shift.z});
    std::vector<ComplexPawOperator> blocks;
    for(std::size_t ia=0;ia<input.atoms.size();++ia){auto block=atomic[ia]?atomic[ia]->nonlocal_block(vectors,volume,cart(a,input.atoms[ia].fractional)):
          prepare_potcar_bloch_block(input,ia,vectors);
      block.d=scf.onecenter_states[ia].onsite_dij_ev;for(auto& z:block.d)z/=ev;blocks.push_back(std::move(block));}
    PlaneWaveOperator op(basis,potential,std::move(blocks),ev,0.);
    out.onecenter_dij_applications+=input.atoms.size();out.max_fft_grid_points=std::max(out.max_fft_grid_points,op.fft_grid_points());
    out.max_operator_storage_bytes=std::max(out.max_operator_storage_bytes,op.storage_bytes());
    std::vector<C> seed;double retained=0.;
    if(!previous.empty()) {auto transported=transport_plane_wave_seeds(previous_basis,basis,previous,out.bands);
      retained=transported.minimum_retained_norm_fraction;if(retained>=.5){seed=std::move(transported.coefficients);++out.warm_start_kpoints;}}
    auto h=op.hamiltonian();const auto solved=solve_generalized_hermitian_davidson(h,op.overlap(),out.bands,256,1e-12,seed);
    const auto residual=*std::max_element(solved.residual_norms.begin(),solved.residual_norms.end());
    if(!solved.converged||residual>1e-12) {
      const auto snapshot=input.iteration_log.string()+".nscf-k"+std::to_string(cache.size())+".failed.pwop";
      op.save_solver_snapshot(snapshot,out.bands,256,1e-12,seed);
      std::ostringstream message;message<<std::setprecision(17)<<"UPF NSCF final fresh residual gate failed k="
        <<point.x<<","<<point.y<<","<<point.z<<" roots="<<out.bands<<" iterations="<<solved.iterations
        <<" fresh="<<residual<<" requested=1e-12 snapshot="<<snapshot;
      throw std::runtime_error(message.str());
    }
    previous_basis=basis;previous=solved.eigenvectors;out.seed_transport_retained_fraction.push_back(retained);
    std::vector<C> projected(h.projector_rows*out.bands);h.project_into(previous.data(),out.bands,projected.data(),true);
    Spectrum spectrum;spectrum.residual=residual;for(auto value:solved.eigenvalues)spectrum.values.push_back(value*ev);
    spectrum.projections.resize(std::size_t(out.bands)*input.atoms.size()*orbital.size());std::size_t row=0;
    for(std::size_t ia=0;ia<input.atoms.size();++ia)for(int l:site_l[ia])for(int m=-l;m<=l;++m,++row)
      for(int band=0;band<out.bands;++band)spectrum.projections[(std::size_t(band)*input.atoms.size()+ia)*orbital.size()+orbital.at(l)]+=std::norm(projected[std::size_t(band)*h.projector_rows+row]);
    out.eigenvalues_ev.insert(out.eigenvalues_ev.end(),spectrum.values.begin(),spectrum.values.end());
    out.projector_weights.insert(out.projector_weights.end(),spectrum.projections.begin(),spectrum.projections.end());
    out.eigensolver_max_residuals_hartree.push_back(residual);cache.emplace(key,std::move(spectrum));
  }
  return out;
}
}
