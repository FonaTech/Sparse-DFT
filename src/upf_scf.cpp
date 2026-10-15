#include "paw_framework.hpp"
#include "paw_site_model.hpp"
#include "upf_hubbard.hpp"
#include "joint_paw_mixer.hpp"
#include "scf_inner_accuracy.hpp"
#include "scf_geometry_guess.hpp"
#include "pw_seed_transport.hpp"
#include "paw_compensation.hpp"
#include "compute_backend.hpp"
#include <chrono>
#include <exception>
#include <mutex>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
#include "upf_onecenter_fast.hpp"
#include "upf_beta_cache.hpp"
#include "upf_geometry_derivatives_v2.hpp"
#include "ewald.hpp"
#include "occupations.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace sdft {
namespace {
constexpr double pi=3.1415926535897932384626433832795;
using C=std::complex<double>;
using Key=std::tuple<int,int,int>;
Vec3 scale(Vec3 v,double x){return {v.x*x,v.y*x,v.z*x};}
Vec3 cart(const std::array<Vec3,3>& a,Vec3 f){return {
 a[0].x*f.x+a[1].x*f.y+a[2].x*f.z,
 a[0].y*f.x+a[1].y*f.y+a[2].y*f.z,
 a[0].z*f.x+a[1].z*f.y+a[2].z*f.z};}
double length(Vec3 v){return std::sqrt(dot(v,v));}
struct KState {
 ScfInput::Kpoint point;
 int spin{};
 std::vector<GVector> basis;
 std::vector<Vec3> vectors;
 std::vector<Key> indices;
 std::vector<C> wave;
 std::vector<double> energies;
};
struct Atom {
 std::shared_ptr<PawSiteGeometry> geometry;
 std::vector<ComplexPawOperator> bare;
 std::vector<C> p;
 std::vector<std::vector<C>> qform;
 std::shared_ptr<PawCompensationOperator> compensation;
};
std::vector<C> augmentation_density(const Atom& atom,const std::vector<C>& p) {
 PerformanceTimer timer("shared_paw_augmentation_density");
 if(atom.compensation) {
   const auto density=atom.compensation->density(p);
   std::vector<C> result;result.reserve(density.size());
   for(const auto& key:atom.compensation->reciprocal_keys())result.push_back(density.at(key));
   return result;
 }
 const auto n=atom.geometry->dimension();std::vector<C> result(atom.qform.size());
 for(std::size_t g=0;g<result.size();++g)for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)
   result[g]+=p[i*n+j]*atom.qform[g][j*n+i];
 return result;
}
std::vector<C> channel_block(const Atom& atom,int spin) {
 const auto n=atom.geometry->dimension();
 return {atom.p.begin()+spin*n*n,atom.p.begin()+(spin+1)*n*n};
}
std::vector<C> compact_project_potential(const Atom& atom,const FourierDensity& potential) {
 const auto coefficient=atom.compensation->project_potential(potential);
 const auto n=atom.geometry->dimension();std::vector<C> result(n*n);
 // The common Hamiltonian contracts Tr(P D). The compensation API returns
 // coefficients of P_ij; transpose them into D_ji, as reciprocal_matrix does.
 for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)result[j*n+i]=coefficient[i*n+j];
 return result;
}
}

// Native UPF model in the main Sparse-DFT library. Bloch states use the shared
// operator, eigensolver, density functional and joint multisecant primitives.
ScfResult run_upf_scf(const ScfInput& input){
 PerformanceTimer total_timer("scf_upf_total");
 if(input.spin_channels!=1&&input.spin_channels!=2)throw std::invalid_argument("UPF spin_channels must be 1 or 2");
 const int ns=input.spin_channels;const bool polarized=ns==2;const auto na=input.atoms.size();
 for(const auto& atom:input.atoms) {
   if(!std::isfinite(atom.initial_magnetic_moment)||std::abs(atom.initial_magnetic_moment)>atom.zval||
      !std::isfinite(atom.hubbard.ueff_ev)||atom.hubbard.ueff_ev<0||atom.hubbard.angular_momentum< -1||
      atom.hubbard.angular_momentum>3||(atom.hubbard.angular_momentum<0&&atom.hubbard.ueff_ev!=0))
     throw std::invalid_argument("invalid UPF magnetic moment/Hubbard model");
   if(!polarized&&(atom.initial_magnetic_moment!=0||atom.hubbard.ueff_ev!=0))
     throw std::invalid_argument("UPF moments/nonzero U require two physical spin channels");
 }
 const auto units=shared_paw_atomic_units(input);
 const double ev=units.ev_per_hartree,bohr_a=1./units.bohr_per_angstrom;

 if(!std::isfinite(input.cutoff_ev)||input.cutoff_ev<=0||input.max_iterations<1||
    !std::isfinite(input.density_tolerance)||input.density_tolerance<=0||
    !std::isfinite(input.onecenter_density_tolerance)||input.onecenter_density_tolerance<=0)
   throw std::invalid_argument("invalid UPF SCF controls");
 const auto started=std::chrono::steady_clock::now();
 const auto& a=input.lattice_bohr;
 const double det=determinant(a),volume=std::abs(det);
 if(!std::isfinite(volume)||volume<1e-10||input.atoms.empty())
  throw std::invalid_argument("invalid UPF pilot cell or atom count");
 if(!std::isfinite(input.mixing)||input.mixing<=0||input.mixing>1)
  throw std::invalid_argument("invalid coupled linear mixing");
 const std::array<Vec3,3> rec{scale(cross(a[1],a[2]),2*pi/det),
  scale(cross(a[2],a[0]),2*pi/det),scale(cross(a[0],a[1]),2*pi/det)};
 auto points=input.kpoints;
 if(points.empty())points.push_back({input.kpoint_fractional,1.});
 double sum=0;
 for(const auto& point:points){
  if(!std::isfinite(point.weight)||point.weight<=0||
     !std::isfinite(point.fractional.x)||!std::isfinite(point.fractional.y)||!std::isfinite(point.fractional.z))
   throw std::invalid_argument("invalid UPF k point");
  sum+=point.weight;
 }
 if(!std::isfinite(sum)||sum<=0)throw std::invalid_argument("UPF k weight sum invalid");
 for(auto& point:points)point.weight/=sum;
 auto reference_rec=rec;
 if(input.basis_reference_lattice_bohr){const auto& reference=*input.basis_reference_lattice_bohr;
   const double determinant_reference=determinant(reference);
   if(!std::isfinite(determinant_reference)||std::abs(determinant_reference)<1e-10)throw std::invalid_argument("invalid fixed basis reference");
   reference_rec={scale(cross(reference[1],reference[2]),2*pi/determinant_reference),
     scale(cross(reference[2],reference[0]),2*pi/determinant_reference),scale(cross(reference[0],reference[1]),2*pi/determinant_reference)};
 }
 const auto physical_points=points;const std::size_t physical_nk=points.size();
 if(polarized)points.insert(points.end(),physical_points.begin(),physical_points.end());
 std::vector<KState> ks;
 FourierDensity support;
 for(const auto& point:points){
  KState k;k.point=point;k.spin=polarized?int(ks.size()/physical_nk):0;k.basis=enumerate_g_vectors_fixed_reference(rec,reference_rec,input.cutoff_ev,point.fractional,ev);
  // GVector::cartesian stores G; only kinetic_ev includes the Bloch shift.
  // Beta projectors and kinetic strain derivatives require actual G+k.
  const auto shift=cart(rec,point.fractional);
  for(const auto& g:k.basis){k.vectors.push_back({g.cartesian.x+shift.x,g.cartesian.y+shift.y,g.cartesian.z+shift.z});k.indices.emplace_back(g.h,g.k,g.l);}
  for(const auto& [key,value]:plane_wave_difference_support(k.indices))support.try_emplace(key,0.);
  ks.push_back(std::move(k));
 }
 // Match the common SCF automatic fine-support contract. Four ENCUT
 // only covers smooth wavefunction products, not general PAW augmentation.
 const double density_cutoff=input.density_cutoff_ev>0?input.density_cutoff_ev:16*input.cutoff_ev;
 if(density_cutoff<4*input.cutoff_ev)throw std::invalid_argument("UPF charge support requires >=4 wave cutoff");
 for(const auto& g:enumerate_g_vectors_fixed_reference(rec,reference_rec,density_cutoff,{},ev))support.try_emplace(Key{g.h,g.k,g.l},0.);
 std::vector<Key> keys;std::vector<Vec3> gv;
 for(const auto& [key,value]:support){auto[h,j,l]=key;keys.push_back(key);gv.push_back(cart(rec,{double(h),double(j),double(l)}));}
 struct Geometry : PawSiteGeometry {
   std::shared_ptr<UpfPawData> data;std::shared_ptr<UpfPawOperator> op;
 };
 std::map<std::string,std::shared_ptr<PawSiteGeometry>> geometry;
 std::vector<std::shared_ptr<PawSiteGeometry>> atom_geometry;
 std::vector<Atom> atoms;double electrons=0,core_energy=0;
 FourierDensity local,core;
 for(const auto& in:input.atoms){
  const auto identity=file_sha256(in.paw_cache);
  auto found_geometry=geometry.find(identity);
  std::shared_ptr<PawSiteGeometry> common;
  std::shared_ptr<Geometry> prepared;
  if(found_geometry!=geometry.end())common=found_geometry->second;
  else if(paw_framework(in.paw_cache)==PawFramework::Potcar) {
    std::vector<std::vector<Vec3>> vectors;for(const auto& k:ks)vectors.push_back(k.vectors);
    common=prepare_potcar_site_geometry(input,atoms.size(),keys,gv,vectors);
    geometry.emplace(identity,common);
  } else {
    prepared=std::make_shared<Geometry>();prepared->data=std::make_shared<UpfPawData>(read_native_upf_paw(in.paw_cache));
    prepared->op=std::make_shared<UpfPawOperator>(prepared->data);
    prepared->local.resize(keys.size());prepared->local_derivative.resize(keys.size());
    prepared->core.resize(keys.size());prepared->core_derivative.resize(keys.size());
    prepared->atomic_density.resize(keys.size());
    prepared->qform.resize(keys.size());prepared->qgradient.resize(keys.size());
    const auto& r=prepared->data->array("UPF/PP_MESH/PP_R");
    const auto w=upf_simpson_weights(prepared->data->array("UPF/PP_MESH/PP_RAB"),r.size());
    const auto core_found=prepared->data->arrays.find("UPF/PP_NLCC");
    const auto& atomic=prepared->data->array("UPF/PP_RHOATOM");
    double atomic_charge=0.;for(std::size_t t=0;t<r.size();++t)atomic_charge+=w[t]*atomic[t];
    if(!std::isfinite(atomic_charge)||atomic_charge<=0)throw std::invalid_argument("invalid UPF atomic charge seed");
    struct RadiusGroup {double magnitude;std::vector<std::size_t> indices;};
    std::map<double,std::vector<std::size_t>> grouped;
    for(std::size_t j=0;j<gv.size();++j)grouped[length(gv[j])].push_back(j);
    std::vector<RadiusGroup> groups;for(auto& [q,indices]:grouped)groups.push_back({q,std::move(indices)});
    record_performance("upf_geometry_exact_radii",0.,groups.size());
    record_performance("upf_geometry_direction_count",0.,gv.size());
    PerformanceTimer geometry_timer("upf_geometry_radial_and_angular_setup");
    std::exception_ptr error;std::mutex error_mutex;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(std::ptrdiff_t group=0;group<static_cast<std::ptrdiff_t>(groups.size());++group){try {
      const auto& entry=groups[group];const double q=entry.magnitude;
      const double vl=.5*upf_local_form_ry(*prepared->data,q);
      const double dvl=q>0?.5*upf_local_form_ry(*prepared->data,q,true):0.;
      double cr=0.,dcr=0.,atomic_form=0.;
      for(std::size_t t=0;t<r.size();++t)atomic_form+=w[t]*atomic[t]*upf_spherical_bessel(0,q*r[t]);
      atomic_form*=prepared->data->z_valence/atomic_charge;
      if(core_found!=prepared->data->arrays.end())for(std::size_t t=0;t<r.size();++t){
        cr+=4*pi*w[t]*r[t]*r[t]*core_found->second[t]*upf_spherical_bessel(0,q*r[t]);
        if(q>0)dcr+=4*pi*w[t]*r[t]*r[t]*r[t]*core_found->second[t]*upf_spherical_bessel_derivative(0,q*r[t]);
      }
      const auto radial=prepared->op->radial_forms(q);
      for(const auto j:entry.indices){
        prepared->local[j]=vl;prepared->local_derivative[j]=dvl;
        prepared->core[j]=cr;prepared->core_derivative[j]=dcr;
        prepared->atomic_density[j]=atomic_form;
        prepared->qform[j]=prepared->op->augmentation_form(gv[j],radial);
        prepared->qgradient[j]=prepared->op->augmentation_form_gradient(gv[j],radial);
      }
    } catch(...) {std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=std::current_exception();}}
    if(error)std::rethrow_exception(error);
    for(const auto& k:ks){
      // D/Q are independent of atom phase. Evaluate them once at G=0;
      // the full original beta geometry is assembled by the exact cache.
      auto bare=prepared->op->nonlocal_block({},volume,{});
      auto beta=prepare_upf_beta_geometry(*prepared->data,k.vectors,volume);
      bare.amplitudes=beta.amplitudes;
      prepared->bare_at_origin.push_back(std::move(bare));prepared->beta_geometry.push_back(std::move(beta));
    }
    prepared->framework=PawFramework::Upf;prepared->element=prepared->data->element;
    prepared->zval=prepared->data->z_valence;prepared->frozen_core_energy_hartree=.5*prepared->data->core_energy_ry;
    prepared->record_id=prepared->data->original_sha256;prepared->numeric_cache_version=2;
    prepared->projector_q_bohr=true;prepared->radial_channel_l=prepared->data->projector_l;
    const auto data=prepared->data;const auto op=prepared->op;
    const auto n=op->dimension();prepared->atomic_onsite.assign(n*n,0.);
    const auto& occupations=data->array("UPF/PP_PAW/PP_OCCUPATIONS");
    if(occupations.size()!=data->projector_count())throw std::invalid_argument("UPF atomic occupation/beta mismatch");
    std::size_t offset=0;
    for(std::size_t radial=0;radial<occupations.size();++radial) {
      if(!std::isfinite(occupations[radial])||occupations[radial]<0)throw std::invalid_argument("UPF atomic occupations must be nonnegative");
      const int count=2*data->projector_l[radial]+1;
      for(int m=0;m<count;++m,++offset)prepared->atomic_onsite[offset*n+offset]=occupations[radial]/count;
    }
    prepared->closure=[data,op](const std::array<std::vector<C>,2>& p,int spins) {
      PawSiteClosure result;
      if(spins==2) {const auto closure=upf_onecenter_collinear(*data,*op,p,12);
        result.energy_hartree=closure.energy;result.derivative_hartree=closure.derivative;
      } else {const auto closure=upf_onecenter_fast(*data,*op,p[0],12);
        result.energy_hartree=closure.energy;result.derivative_hartree[0]=closure.derivative;
      }
      return result;
    };
    prepared->hubbard_overlap=[data](int l){return upf_pseudo_hubbard_overlap(*data,l);};
    common=prepared;geometry.emplace(identity,common);
  }
  Atom atom;atom.geometry=common;
  if(common->compact_compensation)atom.compensation=common->compact_compensation(in.fractional);
  if(common->element!=in.symbol||std::abs(common->zval-in.zval)>1e-12)throw std::invalid_argument("PAW atom element/valence mismatch");
  electrons+=in.zval;
  core_energy+=common->frozen_core_energy_hartree;
  for(std::size_t ik=0;ik<ks.size();++ik){auto bare=common->bare_at_origin[ik];
    bare.amplitudes=phased_upf_beta(common->beta_geometry[ik],ks[ik].vectors,cart(a,in.fractional));
    atom.bare.push_back(std::move(bare));}
  atom.p.assign(ns*common->dimension()*common->dimension(),0.);
  for(std::size_t j=0;j<keys.size();++j){
    const auto phase=std::exp(C(0,-dot(gv[j],cart(a,in.fractional))));
    local[keys[j]]+=common->local[j]*phase/volume;
    core[keys[j]]+=common->core[j]*phase/volume;
    if(!atom.compensation) {
      auto form=common->qform[j];for(auto& z:form)z*=phase/volume;
      atom.qform.push_back(std::move(form));
    }
  }
  atom_geometry.push_back(common);
  atoms.push_back(std::move(atom));
 }
 double absolute_moment=0.;for(const auto& atom:input.atoms)absolute_moment+=std::abs(atom.initial_magnetic_moment);
 const auto capacity=std::size_t(std::ceil((electrons+absolute_moment)/2));
 const std::size_t roots=input.bands>0?std::max(std::size_t(input.bands),capacity):capacity+4;
 for(const auto& k:ks)if(roots>k.basis.size())throw std::invalid_argument("UPF roots exceed k basis");
 FourierDensity density=support;
 for(auto& [key,value]:density)value=0.;density[{0,0,0}]=electrons/volume;
 bool atomic_guess=true;
 if(const char* value=std::getenv("SDFT_UPF_ATOMIC_GUESS")){
   if(std::string(value)!="0"&&std::string(value)!="1")throw std::invalid_argument("SDFT_UPF_ATOMIC_GUESS must be 0 or 1");
   atomic_guess=std::string(value)=="1";
 }
 if(atomic_guess){
   // PP_RHOATOM stores 4*pi*r^2*rho (UPF format), so its
   // radial transform has neither a second r^2 nor another 4*pi.
   // PP_OCCUPATIONS corresponds to beta channels in original PAW data.
   for(std::size_t ia=0;ia<atoms.size();++ia){auto& atom=atoms[ia];
     const auto n=atom.geometry->dimension();
     for(int spin=0;spin<ns;++spin) {
       const double fraction=polarized?.5*(1+(spin==0?1.:-1.)*input.atoms[ia].initial_magnetic_moment/input.atoms[ia].zval):1.;
       for(std::size_t x=0;x<n*n;++x)atom.p[spin*n*n+x]=fraction*atom.geometry->atomic_onsite[x];
     }
     const auto position=cart(a,input.atoms[ia].fractional);
     for(std::size_t g=0;g<keys.size();++g)density.at(keys[g])+=atom_geometry[ia]->atomic_density[g]*std::exp(C(0,-dot(gv[g],position)))/volume;
   }
   density[{0,0,0}]=electrons/volume;
   record_performance("upf_original_atomic_density_and_occupations_guess",0.);
 }
 std::array<FourierDensity,2> spin_density;
 if(polarized) {
   for(auto& q:spin_density){q=support;for(auto& [key,value]:q)value=0.;}
   double moment=0.;for(const auto& atom:input.atoms)moment+=atom.initial_magnetic_moment;
   for(int spin=0;spin<2;++spin) {
     if(atomic_guess)for(std::size_t ia=0;ia<na;++ia) {
       const double fraction=.5*(1+(spin==0?1.:-1.)*input.atoms[ia].initial_magnetic_moment/input.atoms[ia].zval);
       const auto position=cart(a,input.atoms[ia].fractional);
       for(std::size_t g=0;g<keys.size();++g)spin_density[spin].at(keys[g])+=fraction*atom_geometry[ia]->atomic_density[g]*std::exp(C(0,-dot(gv[g],position)))/volume;
     }
     spin_density[spin][{0,0,0}]=.5*(electrons+(spin==0?moment:-moment))/volume;
   }
 }
 std::vector<std::vector<double>> hubbard_overlap(na);
 for(std::size_t ia=0;ia<na;++ia)if(polarized&&input.atoms[ia].hubbard.angular_momentum>=0)
   hubbard_overlap[ia]=atoms[ia].geometry->hubbard_overlap(input.atoms[ia].hubbard.angular_momentum);
 const auto site_block=[](const Atom& atom,int spin) {
   const auto n=atom.geometry->dimension();return std::vector<C>(atom.p.begin()+spin*n*n,atom.p.begin()+(spin+1)*n*n);
 };
 const auto selected_framework=selected_paw_framework(input);
 const bool all_upf=selected_framework==PawFramework::Upf;
 ScfResult result;result.paw_framework=all_upf?"upf_paw":selected_framework==PawFramework::Mixed?"mixed_paw":"potcar_paw";
 result.spin_channels=ns;result.hubbard_occupation_convention=all_upf?"upf_pseudo_qae_bound_v1":"per_site_original_paw_hubbard_v1";for(const auto& atom:input.atoms)result.hubbard_specs.push_back(atom.hubbard);
 result.basis_reference_lattice_bohr=input.basis_reference_lattice_bohr;
 result.projector_interpolation=all_upf?"upf_original_radial_quadrature":selected_framework==PawFramework::Mixed?"per_site_original_paw":input.projector_interpolation;
 result.onecenter_closure_diagnostic=true;result.onecenter_density_tolerance=input.onecenter_density_tolerance;
 result.onecenter_residual_definition="max_atom_max_complex_element_absolute_output_minus_input";
 if(const char* value=std::getenv("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC")) {
   if(std::string(value)!="0"&&std::string(value)!="1")throw std::invalid_argument("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC must be 0 or 1");
   result.onecenter_diagnostic_output_enabled=std::string(value)=="1";
 }
 result.effective_density_tolerance=input.density_tolerance;result.effective_onecenter_density_tolerance=input.onecenter_density_tolerance;
 std::vector<std::size_t> dimensions;for(const auto& atom:atoms)for(int spin=0;spin<ns;++spin)dimensions.push_back(atom.geometry->dimension());
 JointPawMixer mixer(input.mixing,ns*keys.size(),electrons/volume,dimensions);
 result.mixer_method=mixer.name();result.onecenter_mixing_method="shared_joint_multisecant";
 result.multisecant_backtracking=mixer.adaptive_enabled();
 std::ostringstream guess_identity;guess_identity<<std::setprecision(17)<<"upf-geometry-guess-1:codata:"
     <<input.cutoff_ev<<':'<<density_cutoff<<':'<<roots<<':'<<input.smearing_ev<<':'<<ns;
 for(std::size_t ia=0;ia<atoms.size();++ia)guess_identity<<':'<<input.atoms[ia].symbol<<':'<<input.atoms[ia].zval<<':'<<file_sha256(input.atoms[ia].paw_cache);
 for(const auto& atom:input.atoms)guess_identity<<':'<<atom.hubbard.angular_momentum<<':'<<atom.hubbard.ueff_ev;
 for(const auto& point:points)guess_identity<<':'<<point.fractional.x<<':'<<point.fractional.y<<':'<<point.fractional.z<<':'<<point.weight;
 const auto guess_signature=guess_identity.str();
 if(!input.geometry_guess_input.empty()){
   const auto guess=read_geometry_guess(input.geometry_guess_input,guess_signature,ks.size(),atoms.size(),roots,ns);
   const double factor=guess.volume/volume;
   if(factor<.5||factor>2.)throw std::invalid_argument("UPF geometry volume changed by more than factor two");
   std::vector<C> sites;
   for(std::size_t ia=0;ia<atoms.size();++ia){if(guess.onsite[ia].size()!=atoms[ia].p.size())throw std::invalid_argument("UPF geometry on-site dimensions differ");
     sites.insert(sites.end(),guess.onsite[ia].begin(),guess.onsite[ia].end());}
   if(!onsite_positive_semidefinite(sites,dimensions))throw std::invalid_argument("UPF geometry on-site density is not positive semidefinite");
   for(std::size_t ia=0;ia<atoms.size();++ia)atoms[ia].p=guess.onsite[ia];
   std::vector<std::vector<C>> transported_augmentation;
   for(int spin=0;spin<ns;++spin) {
     std::vector<C> sum(keys.size());
     for(const auto& atom:atoms){const auto charge=augmentation_density(atom,channel_block(atom,spin));
       for(std::size_t g=0;g<keys.size();++g)sum[g]+=charge[g];}
     transported_augmentation.push_back(std::move(sum));
   }
   for(std::size_t g=0;g<keys.size();++g) {
     C total=0.;
     for(int spin=0;spin<ns;++spin) {
       const auto& smooth=polarized?guess.spin_smooth[spin]:guess.smooth;
       const auto old=smooth.find(keys[g]);C value=old==smooth.end()?C{}:factor*old->second;
       value+=transported_augmentation[spin][g];
       if(polarized)spin_density[spin].at(keys[g])=value;
       total+=value;
     }
     density.at(keys[g])=total;
   }
   if(polarized) {
     // Correct only total charge. The transported magnetic density is an
     // unconstrained initial guess, never a fixed-magnetization boundary.
     const C correction=.5*(C(electrons/volume)-density.at({0,0,0}));
     for(auto& channel:spin_density)channel.at({0,0,0})+=correction;
   }
   density[{0,0,0}]=electrons/volume;result.geometry_guess_retained_fraction=1.;
   for(std::size_t ik=0;ik<ks.size();++ik){const auto mapped=transport_plane_wave_seeds(guess.g[ik],ks[ik].basis,guess.waves[ik],roots);
     if(mapped.minimum_retained_norm_fraction<.5)throw std::invalid_argument("UPF geometry wavefunction lost over half its norm");
     ks[ik].wave=mapped.coefficients;result.geometry_guess_retained_fraction=std::min(result.geometry_guess_retained_fraction,mapped.minimum_retained_norm_fraction);}
   result.geometry_guess_used=true;record_performance("scf_geometry_guess_import",0.);
 }

 std::ofstream log(input.iteration_log);
 if(!log)throw std::runtime_error("cannot open UPF iteration log");log<<std::setprecision(17);
 std::vector<Vec3> positions;std::vector<double> charges,weights;
 for(const auto& in:input.atoms){positions.push_back(in.fractional);charges.push_back(in.zval);}
 for(const auto& point:physical_points)weights.push_back(point.weight);
 const auto ewald=evaluate_ewald(a,positions,charges);
 double reference_energy=0.,excluded_local_g0_energy=0.;
 for(const auto& site:atom_geometry) {
   reference_energy+=site->reference_energy_hartree;
   excluded_local_g0_energy+=site->excluded_local_g0_form_hartree*electrons/volume;
 }
 result.atomic_reference_energy_ev=reference_energy*ev;
 result.local_g0_energy_ev=excluded_local_g0_energy*ev;

 // Reuse the same inexact-search/final-certificate controller as the
 // POTCAR collinear kernel. Coarse search never satisfies the final gate.
 bool adaptive_inner=false;
 if(const char* flag=std::getenv("SDFT_ADAPTIVE_SEARCH")) {
   if(std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("SDFT_ADAPTIVE_SEARCH must be 0 or 1");
   adaptive_inner=std::string(flag)=="1";
 }
 double search_factor=1e-4;
 if(const char* flag=std::getenv("SDFT_SCF_SEARCH_FACTOR")) {
   std::size_t end=0;const std::string setting(flag);search_factor=std::stod(setting,&end);
   if(end!=setting.size()||!std::isfinite(search_factor)||search_factor<=0||search_factor>.1)
     throw std::invalid_argument("SDFT_SCF_SEARCH_FACTOR must be in (0,0.1]");
 }
 result.adaptive_search=adaptive_inner;
 CollinearInnerAccuracy inner_accuracy(input.density_tolerance,input.onecenter_density_tolerance,true);
 OccupationResult occupation;bool final_precision=false;
 for(int iteration=1;iteration<=input.max_iterations;++iteration){
  PerformanceTimer iteration_timer("scf_upf_iteration");
  std::array<DensityFunctionals,2> functions;
  if(polarized)functions=evaluate_collinear_density_functionals(spin_density,a,&core);
  else functions[0]=evaluate_density_functionals(density,a,core);
  std::array<FourierDensity,2> potentials;
  double hartree_energy=0.,xc_energy=0.,vxc_density=0.;std::size_t xc_grid=0;
  for(int spin=0;spin<ns;++spin) {
    const auto& f=functions[spin];potentials[spin]=f.potential;
    for(const auto& [key,value]:local)potentials[spin][key]+=value;
    hartree_energy+=f.hartree_energy;xc_energy+=f.xc_energy;vxc_density+=f.vxc_density;
    xc_grid=std::max(xc_grid,f.grid_points);
  }
  result.max_xc_grid_points=std::max(result.max_xc_grid_points,xc_grid);
  double correction=0,onecenter_energy=0;
  std::vector<std::vector<C>> effective(ns*na),closure_derivatives(ns*na);
  std::vector<double> closure_energies(ns*na);
  result.hubbard_states.clear();result.hubbard_energy_ev=0.;result.hubbard_potential_expectation_ev=0.;
  for(std::size_t ia=0;ia<na;++ia){const auto& atom=atoms[ia];
   PerformanceTimer closure_timer("scf_upf_onecenter");const std::size_t n=atom.geometry->dimension();
   const auto closure=atom.geometry->closure({site_block(atom,0),polarized?site_block(atom,1):std::vector<C>{}},ns);
   HubbardResult hubbard;
   if(polarized&&input.atoms[ia].hubbard.angular_momentum>=0)
     hubbard=dudarev_paw(atom.geometry->radial_channel_l,hubbard_overlap[ia],input.atoms[ia].hubbard,{site_block(atom,0),site_block(atom,1)});
   result.hubbard_energy_ev+=hubbard.energy_ev;result.hubbard_potential_expectation_ev+=hubbard.potential_expectation_ev;
   result.hubbard_states.push_back(hubbard);onecenter_energy+=closure.energy_hartree+hubbard.energy_ev/ev;
   for(int spin=0;spin<ns;++spin) {
     const auto index=spin*na+ia;auto d=atom.bare[0].d;auto derivative=closure.derivative_hartree[spin];
     if(hubbard.correlated_dimension)for(std::size_t x=0;x<n*n;++x)derivative[x]+=hubbard.projector_potential_ev[spin][x]/ev;
     closure_derivatives[index]=derivative;closure_energies[index]=(closure.energy_hartree+hubbard.energy_ev/ev)/ns;
     for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j) {
       d[i*n+j]+=derivative[i*n+j];correction+=(atom.p[spin*n*n+i*n+j]*derivative[j*n+i]).real();
     }
     if(atom.compensation) {
       PerformanceTimer timer("shared_paw_compact_augmentation_adjoint");
       const auto response=compact_project_potential(atom,potentials[spin]);
       for(std::size_t x=0;x<n*n;++x)d[x]+=response[x];
     } else for(std::size_t g=0;g<keys.size();++g){const auto[h,j,l]=keys[g];
       const auto v=potentials[spin].find({-h,-j,-l});if(v==potentials[spin].end())continue;
       for(std::size_t x=0;x<n*n;++x)d[x]+=volume*v->second*atom.qform[g][x];
     }
     effective[index]=std::move(d);
   }
  }
  result.onecenter_states.clear();for(auto& states:result.spin_onecenter_states)states.clear();
  for(int spin=0;spin<ns;++spin)for(std::size_t ia=0;ia<na;++ia){const auto& atom=atoms[ia];const auto n=atom.geometry->dimension();
    const auto index=spin*na+ia;ScfOneCenterState state;state.atom_index=ia;state.projector_dimension=n;
    state.symbol=input.atoms[ia].symbol;state.fractional=input.atoms[ia].fractional;
    state.paw_cache=input.atoms[ia].paw_cache;state.record_id=atom.geometry->record_id;
    state.numeric_cache_version=atom.geometry->numeric_cache_version;state.numeric_cache_sha256=file_sha256(state.paw_cache);
    state.unit_convention=input.unit_convention;state.projector_q_bohr=atom.geometry->projector_q_bohr;
    state.radial_channel_l=atom.geometry->radial_channel_l;state.density_matrix=site_block(atom,spin);
    state.onsite_dij_ev=effective[index];for(auto& z:state.onsite_dij_ev)z*=ev;
    state.closure_correction_ev=closure_derivatives[index];for(auto& z:state.closure_correction_ev)z*=ev;
    state.compensation_correction_ev.resize(n*n);
    for(std::size_t x=0;x<n*n;++x)state.compensation_correction_ev[x]=state.onsite_dij_ev[x]-atom.bare[0].d[x]*ev-state.closure_correction_ev[x];
    state.energy_difference_ev=closure_energies[index]*ev;
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)state.potential_expectation_ev+=(atom.p[spin*n*n+i*n+j]*closure_derivatives[index][j*n+i]).real()*ev;
    if(polarized)result.spin_onecenter_states[spin].push_back(state);
    result.onecenter_states.push_back(std::move(state));
  }
  bool precision_transition=false;
  if(adaptive_inner) {
    if(iteration>1)precision_transition=inner_accuracy.observe(result.density_residual,
      std::max(result.onecenter_density_residual,result.hubbard_occupation_residual));
    final_precision=inner_accuracy.final_refinement()||iteration==input.max_iterations;
  } else {
    precision_transition=!final_precision&&iteration>1&&
      result.density_residual<1e-8&&result.onecenter_density_residual<1e-8;
    final_precision=final_precision||precision_transition;
  }
  if(precision_transition)mixer.clear_inexact_history();
  const bool final_inner=final_precision;
  const double outer_error=iteration==1?1.:std::max({result.density_residual,
    result.onecenter_density_residual,result.hubbard_occupation_residual});
  // Keep this UPF lane's existing strict final request unchanged, including
  // user tolerances looser than the validation inputs. No coarse acceptance.
  const double tolerance=final_inner?5e-14:(adaptive_inner?
    inner_accuracy.requested(outer_error,search_factor,false):1e-10);
  double fresh=0;std::vector<std::vector<double>> energies;
  for(std::size_t ik=0;ik<ks.size();++ik){auto& k=ks[ik];std::vector<ComplexPawOperator> blocks;
   for(std::size_t ia=0;ia<atoms.size();++ia){auto block=atoms[ia].bare[ik];block.d=effective[k.spin*na+ia];blocks.push_back(std::move(block));}
   PlaneWaveOperator h(k.basis,potentials[k.spin],std::move(blocks),ev,0.);
   result.max_fft_grid_points=std::max(result.max_fft_grid_points,h.fft_grid_points());
   result.max_operator_storage_bytes=std::max(result.max_operator_storage_bytes,h.storage_bytes());
   PerformanceTimer solver_timer("scf_upf_eigensolver");
   const auto solved=[&]() {
     try{return solve_generalized_hermitian_davidson(h.hamiltonian(),h.overlap(),roots,256,tolerance,k.wave);}
     catch(const std::exception& failure) {
       const auto snapshot=input.iteration_log.string()+".k"+std::to_string(ik)+".exception.pwop";
       h.save_solver_snapshot(snapshot,roots,256,tolerance,k.wave);
       throw std::runtime_error("shared PAW solver exception SCF="+std::to_string(iteration)+
           " k="+std::to_string(ik)+": "+failure.what()+"; original H/S snapshot="+snapshot);
     }
   }();
   const double residual=*std::max_element(solved.residual_norms.begin(),solved.residual_norms.end());
   if(!solved.converged){
    h.save_solver_snapshot(input.iteration_log.string()+".k"+std::to_string(ik)+".failed.pwop",roots,256,tolerance,k.wave);
    std::ostringstream error;error<<std::setprecision(17)<<"UPF Bloch Davidson failure SCF="<<iteration<<" k="<<ik<<" fresh="<<residual<<" requested="<<tolerance;
    throw std::runtime_error(error.str());
   }
   if(const char* audit=std::getenv("SDFT_SHARED_SNAPSHOT_ITERATION")) {
     std::size_t end=0;const std::string setting(audit);const auto chosen=std::stoull(setting,&end);
     if(end!=setting.size()||chosen<1||chosen>std::size_t(input.max_iterations))
       throw std::invalid_argument("invalid declared shared snapshot iteration");
     if(chosen==std::size_t(iteration))
       h.save_solver_snapshot(input.iteration_log.string()+".k"+std::to_string(ik)+".audit.pwop",roots,256,tolerance,solved.eigenvectors);
   }
   fresh=std::max(fresh,residual);k.wave=solved.eigenvectors;k.energies.clear();
   for(double e:solved.eigenvalues)k.energies.push_back(e*ev);energies.push_back(k.energies);
  }
  occupation=polarized?fill_collinear_occupations(energies,weights,electrons,input.smearing_ev):fill_occupations(energies,weights,electrons,input.smearing_ev);
  FourierDensity next=support;for(auto& [key,value]:next)value=0.;
  std::array<FourierDensity,2> next_spin{next,next};
  double band_energy=0;
  std::vector<std::vector<C>> onsites;for(const auto& atom:atoms)onsites.emplace_back(atom.p.size(),0.);
  for(std::size_t ik=0;ik<ks.size();++ik){const auto& k=ks[ik];const auto& occ=occupation.occupations[ik];
   const auto smooth=plane_wave_density_fft(k.indices,k.wave,occ,volume,support);
   for(const auto& [key,value]:smooth){next[key]+=k.point.weight*value;if(polarized)next_spin[k.spin][key]+=k.point.weight*value;}
   for(std::size_t b=0;b<roots;++b)band_energy+=k.point.weight*occ[b]*k.energies[b]/ev;
   for(std::size_t ia=0;ia<atoms.size();++ia){const auto& bare=atoms[ia].bare[ik];const std::size_t n=bare.projectors;
    for(std::size_t b=0;b<roots;++b){std::vector<C> projection(n);
     for(std::size_t i=0;i<n;++i)for(std::size_t g=0;g<k.basis.size();++g)
      projection[i]+=bare.amplitudes[i*k.basis.size()+g]*k.wave[b*k.basis.size()+g];
     for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)
      onsites[ia][k.spin*n*n+i*n+j]+=k.point.weight*occ[b]*projection[i]*std::conj(projection[j]);
    }
   }
  }
  double onsite_residual=0;result.onecenter_density_residuals.clear();
  for(std::size_t ia=0;ia<atoms.size();++ia){const auto& atom=atoms[ia];const std::size_t n=atom.geometry->dimension();
   double atom_residual=0.;
   for(std::size_t x=0;x<ns*n*n;++x)atom_residual=std::max(atom_residual,std::abs(onsites[ia][x]-atom.p[x]));
   result.onecenter_density_residuals.push_back(atom_residual);onsite_residual=std::max(onsite_residual,atom_residual);
   for(int spin=0;spin<ns;++spin){
     const std::vector<C> p(onsites[ia].begin()+spin*n*n,onsites[ia].begin()+(spin+1)*n*n);
     const auto charge=augmentation_density(atom,p);
     for(std::size_t g=0;g<keys.size();++g){next[keys[g]]+=charge[g];if(polarized)next_spin[spin][keys[g]]+=charge[g];}
   }
  }
  result.hubbard_occupation_residual=0.;
  if(polarized)for(std::size_t ia=0;ia<na;++ia)if(input.atoms[ia].hubbard.angular_momentum>=0) {
    const auto n=atoms[ia].geometry->dimension();std::array<std::vector<C>,2> output;
    for(int spin=0;spin<2;++spin)output[spin]=std::vector<C>(onsites[ia].begin()+spin*n*n,onsites[ia].begin()+(spin+1)*n*n);
    const auto hub=dudarev_paw(atoms[ia].geometry->radial_channel_l,hubbard_overlap[ia],input.atoms[ia].hubbard,output);
    for(int spin=0;spin<2;++spin)for(std::size_t x=0;x<hub.occupations[spin].size();++x)
      result.hubbard_occupation_residual=std::max(result.hubbard_occupation_residual,std::abs(hub.occupations[spin][x]-result.hubbard_states[ia].occupations[spin][x]));
  }
  double residual=0;
  if(polarized)for(int spin=0;spin<2;++spin)for(const auto& [key,value]:next_spin[spin])residual=std::max(residual,std::abs(value-spin_density[spin].at(key)));
  else for(const auto& [key,value]:next)residual=std::max(residual,std::abs(value-density.at(key)));
  result.total_energy_ev=(band_energy-hartree_energy-vxc_density+xc_energy+onecenter_energy-correction+ewald.energy+reference_energy+excluded_local_g0_energy)*ev+occupation.entropy_energy_ev;
  result.band_energy_ev=band_energy*ev;result.hartree_energy_ev=hartree_energy*ev;
  result.xc_energy_ev=xc_energy*ev;result.vxc_density_ev=vxc_density*ev;
  result.paw_onecenter_energy_ev=onecenter_energy*ev;result.paw_onecenter_potential_expectation_ev=correction*ev;
  result.ion_ion_energy_ev=ewald.energy*ev;
  result.internal_energy_ev=result.total_energy_ev-occupation.entropy_energy_ev;
  result.entropy_energy_ev=occupation.entropy_energy_ev;result.fermi_energy_ev=occupation.chemical_potential_ev;
  result.iterations=iteration;result.density_residual=residual;result.onecenter_density_residual=onsite_residual;
  result.integrated_electron_count=next.at({0,0,0}).real()*volume;result.eigensolver_max_residual=fresh;
  log<<"{\"iteration\":"<<iteration<<",\"density_residual\":"<<residual<<",\"onecenter_residual\":"<<onsite_residual
     <<",\"energy_ev\":"<<result.total_energy_ev<<",\"electrons\":"<<result.integrated_electron_count
     <<",\"fresh_ha\":"<<fresh<<",\"nkpoints\":"<<ks.size()<<",\"xc_grid_points\":"<<xc_grid
     <<",\"mixer\":\""<<mixer.name()<<"\",\"multisecant_rank\":"<<mixer.rank<<",\"mixing_step_scale\":"<<mixer.step_scale
     <<",\"prior_joint_feedback_l2\":"<<mixer.feedback_norm<<",\"prior_mixer_history_restart\":"<<(mixer.history_restarted?"true":"false")
     <<",\"prior_mixer_trial_rejected\":"<<(mixer.trial_rejected?"true":"false")<<",\"prior_mixer_accepted_feedback_l2\":"<<mixer.accepted_feedback
     <<",\"prior_mixer_feedback_bound_l2\":"<<mixer.feedback_bound
     <<",\"prior_mixer_proposal_scale\":"<<mixer.proposal_scale<<",\"prior_positivity_scale\":"<<mixer.positivity_scale
     <<",\"prior_positivity_convex_fallback\":"<<(mixer.positivity_fallback?"true":"false")<<",\"prior_mixer_history_size\":"<<mixer.history_size()
     <<",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
     <<",\"inner_requested_ha\":"<<tolerance<<",\"inner_final_refinement\":"<<(final_inner?"true":"false")
     <<",\"hubbard_occupation_residual\":"<<result.hubbard_occupation_residual
     <<",\"magnetic_moment_muB\":"<<(polarized?(next_spin[0].at({0,0,0})-next_spin[1].at({0,0,0})).real()*volume:0.)
     <<",\"frozen_core_energy_ha\":"<<core_energy<<"}\n";log.flush();
  if(std::abs(result.integrated_electron_count-electrons)>1e-7)throw std::runtime_error("UPF generalized charge conservation failed");
  if(final_inner&&residual<input.density_tolerance&&onsite_residual<input.onecenter_density_tolerance&&result.hubbard_occupation_residual<input.onecenter_density_tolerance){result.converged=true;break;}
  std::vector<C> smooth_values,next_values,sites,next_sites;
  if(polarized)for(int spin=0;spin<2;++spin)for(const auto& key:keys){smooth_values.push_back(spin_density[spin].at(key));next_values.push_back(next_spin[spin].at(key));}
  else for(const auto& key:keys){smooth_values.push_back(density.at(key));next_values.push_back(next.at(key));}
  for(std::size_t ia=0;ia<atoms.size();++ia){sites.insert(sites.end(),atoms[ia].p.begin(),atoms[ia].p.end());next_sites.insert(next_sites.end(),onsites[ia].begin(),onsites[ia].end());}
  mixer.update(smooth_values,sites,next_values,next_sites);
  if(polarized)for(std::size_t g=0;g<keys.size();++g){spin_density[0].at(keys[g])=smooth_values[g];spin_density[1].at(keys[g])=smooth_values[keys.size()+g];density.at(keys[g])=spin_density[0].at(keys[g])+spin_density[1].at(keys[g]);}
  else for(std::size_t g=0;g<keys.size();++g)density.at(keys[g])=smooth_values[g];
  std::size_t offset=0;for(auto& atom:atoms)for(auto& value:atom.p)value=sites[offset++];
  if(polarized){const auto correction=(electrons/volume-density.at({0,0,0}))/2.;
    for(auto& q:spin_density)q.at({0,0,0})+=correction;density.at({0,0,0})=electrons/volume;}
  result.final_mixing_step_scale=mixer.step_scale;

 }
 if(result.converged){
  std::array<DensityFunctionals,2> functions;
  if(polarized)functions=evaluate_collinear_density_functionals(spin_density,a,&core);
  else functions[0]=evaluate_density_functionals(density,a,core);
  std::vector<Vec3> forces(atoms.size());double stress[3][3]{};
  for(int axis=0;axis<3;++axis)stress[axis][axis]-=excluded_local_g0_energy;
  auto addforce=[](Vec3& f,int axis,double x){if(axis==0)f.x+=x;else if(axis==1)f.y+=x;else f.z+=x;};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
   for(int spin=0;spin<ns;++spin)stress[i][j]+=volume*(upf_component(functions[spin].hartree_stress[i],j)+upf_component(functions[spin].xc_stress[i],j));
  for(std::size_t ik=0;ik<ks.size();++ik){const auto& k=ks[ik];
   for(std::size_t b=0;b<roots;++b)for(std::size_t g=0;g<k.basis.size();++g)
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
     stress[i][j]-=k.point.weight*occupation.occupations[ik][b]*std::norm(k.wave[b*k.basis.size()+g])*upf_component(k.vectors[g],i)*upf_component(k.vectors[g],j);
  }
  for(int spin=0;spin<ns;++spin)for(std::size_t ia=0;ia<atoms.size();++ia){const auto& atom=atoms[ia];const std::size_t n=atom.geometry->dimension();
   const auto& func=functions[spin];auto potential=func.potential;
   for(const auto& [key,value]:local)potential[key]+=value;
   const auto channel=site_block(atom,spin);const auto& channel_density=polarized?spin_density[spin]:density;
   const auto position=cart(a,input.atoms[ia].fractional);auto d=atom.bare[0].d;
   const auto& frozen=result.onecenter_states[spin*na+ia];
   for(std::size_t x=0;x<n*n;++x)d[x]+=frozen.closure_correction_ev[x]/ev;
   const auto& prepared=*atom_geometry[ia];
   const auto augmentation_charge=augmentation_density(atom,channel);
   if(atom.compensation) {
     const auto response=compact_project_potential(atom,potential);
     for(std::size_t x=0;x<n*n;++x)d[x]+=response[x];
     const auto shape=atom.compensation->shape_strain_energy_gradient(channel,potential);
     for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress[i][j]+=upf_component(shape[j],i);
   }
   for(std::size_t g=0;g<keys.size();++g){const double q=length(gv[g]);const auto phase=std::exp(C(0,-dot(gv[g],position)));
    const auto v=potential.at(keys[g]),xc=func.xc_potential.at(keys[g]),rho=channel_density.at(keys[g]);
    const auto vl=prepared.local[g]*phase/volume;
    const double cr=prepared.core[g]/ns,dc=prepared.core_derivative[g]/ns;
    const auto nc=cr*phase/volume;const C augmentation=augmentation_charge[g];
    if(!atom.compensation)for(std::size_t x=0;x<n*n;++x)d[x]+=volume*std::conj(v)*atom.qform[g][x];
    for(int axis=0;axis<3;++axis){const auto factor=C(0,-upf_component(gv[g],axis));
     addforce(forces[ia],axis,-volume*(std::conj(rho)*factor*vl+std::conj(xc)*factor*nc+std::conj(v)*factor*augmentation).real());}
    const double local_energy=volume*(std::conj(rho)*vl).real();for(int i=0;i<3;++i)stress[i][i]-=local_energy;
    if(q>0){const double dv=prepared.local_derivative[g];
     for(int i=0;i<3;++i)for(int j=0;j<3;++j){const double gi=upf_component(gv[g],i),gj=upf_component(gv[g],j);C response=0;
      if(!atom.compensation)for(std::size_t x=0;x<n;++x)for(std::size_t y=0;y<n;++y)response+=channel[x*n+y]*prepared.qgradient[g][j][y*n+x];
      stress[i][j]-=(std::conj(rho)*dv*phase*gi*gj/q+std::conj(xc)*dc*phase*gi*gj/q+std::conj(v)*response*phase*gi).real();
     }
    }
   }
   for(std::size_t ik=0;ik<ks.size();++ik){const auto& k=ks[ik];if(k.spin!=spin)continue;const auto& bare=atom.bare[ik];
    PerformanceTimer derivative_timer("upf_cached_beta_force_stress_contraction");
    const auto& unphased_gradient=prepared.beta_geometry[ik].gradients;
    auto gradient=unphased_gradient;
    for(std::size_t g=0;g<k.vectors.size();++g){const auto phase=std::exp(C(0,dot(k.vectors[g],position)));
      for(int axis=0;axis<3;++axis)for(std::size_t x=0;x<n;++x)gradient[axis][x*k.vectors.size()+g]*=phase;
    }
    for(std::size_t b=0;b<roots;++b){std::vector<C> c(n),response(n);std::array<std::vector<C>,3> dc;
     for(auto& row:dc)row.resize(n);
     for(std::size_t x=0;x<n;++x)for(std::size_t g=0;g<k.basis.size();++g){const auto projected=bare.amplitudes[x*k.basis.size()+g]*k.wave[b*k.basis.size()+g];c[x]+=projected;
      for(int axis=0;axis<3;++axis)dc[axis][x]+=C(0,upf_component(k.vectors[g],axis))*projected;}
     for(std::size_t x=0;x<n;++x)for(std::size_t y=0;y<n;++y)response[x]+=(d[x*n+y]-k.energies[b]/ev*bare.q[x*n+y])*c[y];
     const double count=k.point.weight*occupation.occupations[ik][b];
     for(int axis=0;axis<3;++axis){C de=0;for(std::size_t x=0;x<n;++x)de+=std::conj(dc[axis][x])*response[x];addforce(forces[ia],axis,-2*count*de.real());}
     for(int i=0;i<3;++i)for(int j=0;j<3;++j){C de=0;
      for(std::size_t x=0;x<n;++x){C db=0;for(std::size_t g=0;g<k.basis.size();++g)db-=upf_component(k.vectors[g],i)*gradient[j][x*k.basis.size()+g]*k.wave[b*k.basis.size()+g];
       if(i==j)db-=.5*c[x];de+=std::conj(db)*response[x];}
      stress[i][j]+=2*count*de.real();
     }
    }
   }
  }
  for(std::size_t ia=0;ia<forces.size();++ia){result.hf_electronic_forces_ev_per_angstrom.push_back(scale(forces[ia],ev/bohr_a));
   result.ewald_forces_ev_per_angstrom.push_back(scale(ewald.forces[ia],ev/bohr_a));
   result.hf_total_forces_ev_per_angstrom.push_back(scale({forces[ia].x+ewald.forces[ia].x,forces[ia].y+ewald.forces[ia].y,forces[ia].z+ewald.forces[ia].z},ev/bohr_a));}
  for(int i=0;i<3;++i){const Vec3 row{stress[i][0]/volume,stress[i][1]/volume,stress[i][2]/volume};
   result.hf_electronic_stress_ev_per_angstrom3[i]=scale(row,ev/std::pow(bohr_a,3));
   result.ewald_stress_ev_per_angstrom3[i]=scale(ewald.stress[i],ev/std::pow(bohr_a,3));
   result.hf_total_stress_ev_per_angstrom3[i]=scale({row.x+ewald.stress[i].x,row.y+ewald.stress[i].y,row.z+ewald.stress[i].z},ev/std::pow(bohr_a,3));}
 }
 if(result.converged&&!input.geometry_guess_output.empty()){
   ScfGeometryGuess guess;guess.signature=guess_signature;guess.volume=volume;
   guess.spin_channels=ns;guess.smooth=density;
   if(polarized)guess.spin_smooth=spin_density;
   for(int spin=0;spin<ns;++spin)for(const auto& atom:atoms) {
     const auto charge=augmentation_density(atom,channel_block(atom,spin));
     for(std::size_t g=0;g<keys.size();++g){guess.smooth.at(keys[g])-=charge[g];
       if(polarized)guess.spin_smooth[spin].at(keys[g])-=charge[g];}
   }
   for(const auto& atom:atoms)guess.onsite.push_back(atom.p);
   for(const auto& k:ks){guess.g.push_back(k.basis);guess.waves.push_back(k.wave);}
   write_geometry_guess(input.geometry_guess_output,guess);
 }
 if(polarized){result.spin_density_q=spin_density;result.spin_hamiltonian_density_q=spin_density;
   result.magnetic_moment=(spin_density[0].at({0,0,0})-spin_density[1].at({0,0,0})).real()*volume;}
 result.analytic_stress_available=result.converged;
 result.density_q=density;result.hamiltonian_density_q=density;result.frozen_core_q=core;
 result.g_vectors=ks.front().basis;result.occupied_bands=std::ceil(electrons/2);result.bands=int(roots);
 result.solver_method="matrix_free_fft_block_davidson";result.unit_convention=input.unit_convention;
 result.onecenter_lattice_bohr=a;
 auto index_hash=[&](const std::string& suffix,std::vector<Key> list){std::sort(list.begin(),list.end());const auto path=input.iteration_log.string()+suffix;
   {std::ofstream stream(path);if(!stream)throw std::runtime_error("cannot write UPF basis evidence");for(auto key:list){auto[h,k,l]=key;stream<<h<<" "<<k<<" "<<l<<"\n";}}
   return file_sha256(path);
 };
 result.density_support_index_sha256=index_hash(".density-indices",keys);
 for(std::size_t k=0;k<ks.size();++k){result.plane_wave_index_sha256.push_back(index_hash(".k"+std::to_string(k)+".wave-indices",ks[k].indices));result.kpoint_spin_channels.push_back(ks[k].spin);}
 result.kpoints=points;result.kpoint_occupations=occupation.occupations;
 for(const auto& k:ks){result.plane_waves_per_kpoint.push_back(k.basis.size());result.plane_waves=std::max(result.plane_waves,k.basis.size());result.kpoint_eigenvalues_ev.push_back(k.energies);}
 result.eigenvalues_ev=ks.front().energies;result.electron_count=electrons;
 result.cutoff_ev=input.cutoff_ev;result.density_cutoff_ev=density_cutoff;result.density_reciprocal_modes=keys.size();
 result.onecenter_radial_mode=all_upf?"UPF_original_QijL_discrete_variational_multik_development":"per_site_original_energy_adjoint_paw_development";
 result.total_energy_valid=false;
 return result;
}
}
