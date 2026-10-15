#include "upf_paw_operator.hpp"
#include "paw_compensation.hpp"
#include "upf_geometry_derivatives_v2.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace sdft {
namespace {
using Complex=std::complex<double>;
constexpr double pi=3.1415926535897932384626433832795;
double length(Vec3 q){return std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);}
double product(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
// Same compensated accumulation and nodal order as the independent scalar
// transform. Only immutable input preparation and Bessel calls are shared.
struct ExactSum {
  double total{},correction{};
  void add(double x){const double t=total+x;
    correction+=std::abs(total)>=std::abs(x)?(total-t)+x:(x-t)+total;total=t;}
  double value()const{return total+correction;}
};
void onsite_check(const std::vector<Complex>& p,std::size_t n) {
  if(p.size()!=n*n)throw std::invalid_argument("UPF onsite dimension mismatch");
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){
    const auto z=p[i*n+j];
    if(!std::isfinite(z.real())||!std::isfinite(z.imag())||std::abs(z-std::conj(p[j*n+i]))>1e-10)
      throw std::invalid_argument("UPF onsite density must be finite Hermitian");
  }
}
}

UpfPawOperator::UpfPawOperator(std::shared_ptr<const UpfPawData> data):data_(std::move(data)) {
  if(!data_||data_->functional!="PBE"||data_->relativistic!="scalar")
    throw std::invalid_argument("UPF operator requires scalar PBE PAW data");
  for(std::size_t i=0;i<data_->projector_count();++i){const int l=data_->projector_l[i];
    // The shared real harmonics and Cartesian shape derivatives currently
    // support L<=6. Reject larger density multipoles instead of truncating.
    if(l>3)throw std::invalid_argument("UPF operator angular support requires beta l<=3");
    maximum_l_=std::max(maximum_l_,l);
    for(int m=-l;m<=l;++m)expanded_.push_back({i,l,m});}
  for(std::size_t a=0;a<dimension();++a)for(std::size_t b=0;b<dimension();++b){
    const auto x=expanded_[a],y=expanded_[b];
    for(const auto& c:paw_real_gaunt_terms(x.l,y.l,x.m,y.m))
      if(std::abs(c.value)>1e-14)coupling_.push_back({a,b,x.radial,y.radial,c.L,c.M,c.value});
  }
  radial_weights_=upf_simpson_weights(data_->array("UPF/PP_MESH/PP_RAB"),data_->integration_count);
  const auto mesh=data_->array("UPF/PP_MESH/PP_R").size();
  for(const auto& c:coupling_){const auto key=std::make_tuple(std::min(c.i,c.j),std::max(c.i,c.j),c.l);
    if(radial_pairs_.count(key))continue;
    RadialPair pair;pair.augmentation=upf_augmentation_radial(*data_,c.i,c.j,c.l);
    pair.all_electron.resize(mesh);pair.pseudo.resize(mesh);
    const auto& ae_i=data_->array("UPF/PP_FULL_WFC/PP_AEWFC."+std::to_string(c.i+1));
    const auto& ae_j=data_->array("UPF/PP_FULL_WFC/PP_AEWFC."+std::to_string(c.j+1));
    const auto& ps_i=data_->array("UPF/PP_FULL_WFC/PP_PSWFC."+std::to_string(c.i+1));
    const auto& ps_j=data_->array("UPF/PP_FULL_WFC/PP_PSWFC."+std::to_string(c.j+1));
    for(std::size_t k=0;k<mesh;++k){
      pair.all_electron[k]=ae_i[k]*ae_j[k]+0.;
      pair.pseudo[k]=ps_i[k]*ps_j[k]+pair.augmentation[k];
    }
    radial_pairs_.emplace(key,std::move(pair));
  }
}

ComplexPawOperator UpfPawOperator::nonlocal_block(const std::vector<Vec3>& vectors,
    double volume,Vec3 position) const {
  if(!std::isfinite(volume)||volume<=0)throw std::invalid_argument("invalid UPF cell volume");
  ComplexPawOperator out;const auto dim=dimension(),npw=vectors.size(),n=data_->projector_count();
  out.projectors=dim;out.amplitudes.resize(dim*npw);out.d.assign(dim*dim,0.);out.q.assign(dim*dim,0.);
  const auto& d=data_->array("UPF/PP_NONLOCAL/PP_DIJ");
  // The overlap q_ij is the G=0 limit of the augmentation form factor (as in
  // QE), not the separately tabulated PP_Q. This preserves the discrete
  // equality between integrated augmentation charge and the S operator.
  const auto integrated_q=augmentation_form({0,0,0});
  std::map<std::pair<std::size_t,double>,double> radial;
  for(std::size_t a=0;a<dim;++a){const auto channel=expanded_[a];
    for(std::size_t g=0;g<npw;++g){const double magnitude=length(vectors[g]);const auto key=std::make_pair(channel.radial,magnitude);
      auto it=radial.find(key);if(it==radial.end())it=radial.emplace(key,upf_projector_form(*data_,channel.radial,magnitude)).first;
      // B=<beta|G+k>. The real-harmonic Fourier transform of the bra has
      // i^l and exp(+i(G+k).R); the 4*pi/sqrt(Omega) factor is explicit.
      out.amplitudes[a*npw+g]=4*pi/std::sqrt(volume)*std::pow(Complex(0,1),channel.l)*
          upf_real_spherical_harmonic(channel.l,channel.m,vectors[g])*it->second*
          std::exp(Complex(0,product(vectors[g],position)));}
    for(std::size_t b=0;b<dim;++b){const auto other=expanded_[b];
      if(channel.l==other.l&&channel.m==other.m){
        out.d[a*dim+b]=.5*d[channel.radial+n*other.radial]; // Ry -> Ha once
        out.q[a*dim+b]=integrated_q[a*dim+b];}}
  }
  return out;
}

UpfPawOperator::RadialForms UpfPawOperator::radial_forms(double magnitude) const {
  if(!std::isfinite(magnitude)||magnitude<0)throw std::invalid_argument("invalid UPF radial magnitude");
  RadialForms forms;forms.magnitude=magnitude;
  const auto& r=data_->array("UPF/PP_MESH/PP_R");
  std::map<int,std::pair<std::vector<double>,std::vector<double>>> bessel;
  for(const auto& [key,pair]:radial_pairs_){const int l=std::get<2>(key);
    auto [it,inserted]=bessel.try_emplace(l);
    if(inserted){auto& [values,derivatives]=it->second;
      values.resize(data_->integration_count);derivatives.resize(data_->integration_count);
      for(std::size_t k=0;k<data_->integration_count;++k){
        values[k]=upf_spherical_bessel(l,magnitude*r[k]);
        derivatives[k]=r[k]*upf_spherical_bessel_derivative(l,magnitude*r[k]);
      }
    }
    const auto& [values,derivatives]=it->second;ExactSum value,derivative;
    for(std::size_t k=0;k<data_->integration_count;++k){
      value.add(radial_weights_[k]*pair.augmentation[k]*values[k]);
      derivative.add(radial_weights_[k]*pair.augmentation[k]*derivatives[k]);
    }
    forms.pairs.emplace(key,std::make_pair(value.value(),derivative.value()));
  }
  return forms;
}

std::vector<Complex> UpfPawOperator::augmentation_form(Vec3 g) const {
  std::vector<Complex> result(dimension()*dimension(),0.);const double magnitude=length(g);
  std::map<std::tuple<std::size_t,std::size_t,int>,double> radial;
  for(const auto& c:coupling_){const auto key=std::make_tuple(std::min(c.i,c.j),std::max(c.i,c.j),c.l);
    auto it=radial.find(key);if(it==radial.end())it=radial.emplace(key,upf_augmentation_form(*data_,c.i,c.j,c.l,magnitude)).first;
    result[c.a*dimension()+c.b]+=4*pi*std::pow(Complex(0,-1),c.l)*c.gaunt*
        upf_real_spherical_harmonic(c.l,c.m,g)*it->second;}
  return result;
}

std::array<std::vector<Complex>,3> UpfPawOperator::augmentation_form_gradient(Vec3 g) const {
  std::array<std::vector<Complex>,3> result;for(auto& row:result)row.assign(dimension()*dimension(),0.);
  const double magnitude=length(g);const double components[3]={g.x,g.y,g.z};
  std::map<std::tuple<std::size_t,std::size_t,int>,std::pair<double,double>> radial;
  for(const auto& c:coupling_){const auto key=std::make_tuple(std::min(c.i,c.j),std::max(c.i,c.j),c.l);
    auto it=radial.find(key);if(it==radial.end())it=radial.emplace(key,std::make_pair(
        upf_augmentation_form(*data_,c.i,c.j,c.l,magnitude),upf_augmentation_form(*data_,c.i,c.j,c.l,magnitude,true))).first;
    const auto factor=4*pi*std::pow(Complex(0,-1),c.l)*c.gaunt;
    if(magnitude<1e-12){
      if(c.l==1){ // d[j1(|G|r)Y1m(Ghat)]/dGa at the origin.
        const double radial_derivative=it->second.second;
        const double axis=std::sqrt(3/(4*pi));
        if(c.m==1)result[0][c.a*dimension()+c.b]+=factor*radial_derivative*axis;
        if(c.m==-1)result[1][c.a*dimension()+c.b]+=factor*radial_derivative*axis;
        if(c.m==0)result[2][c.a*dimension()+c.b]+=factor*radial_derivative*axis;
      }
      continue;
    }
    const double y=upf_real_spherical_harmonic(c.l,c.m,g);
    const auto dy=paw_real_spherical_harmonic_gradient(c.l,c.m,g);const double dyc[3]={dy.x,dy.y,dy.z};
    for(int a=0;a<3;++a)result[a][c.a*dimension()+c.b]+=factor*(
        it->second.second*components[a]/magnitude*y+it->second.first*dyc[a]);
  }
  return result;
}

std::vector<Complex> UpfPawOperator::augmentation_form(Vec3 g,const RadialForms& forms) const {
  std::vector<Complex> result(dimension()*dimension(),0.);const double magnitude=length(g);
  if(magnitude!=forms.magnitude)throw std::invalid_argument("UPF exact-radius cache mismatch");
  std::vector<double> harmonics((density_lmax()+1)*(density_lmax()+1));
  for(int l=0;l<=density_lmax();++l)for(int m=-l;m<=l;++m)
    harmonics[l*l+m+l]=upf_real_spherical_harmonic(l,m,g);
  for(const auto& c:coupling_){const auto key=std::make_tuple(std::min(c.i,c.j),std::max(c.i,c.j),c.l);
    const auto it=forms.pairs.find(key);if(it==forms.pairs.end())throw std::invalid_argument("incomplete UPF radial forms");
    result[c.a*dimension()+c.b]+=4*pi*std::pow(Complex(0,-1),c.l)*c.gaunt*
        harmonics[c.l*c.l+c.m+c.l]*it->second.first;}
  return result;
}

std::array<std::vector<Complex>,3> UpfPawOperator::augmentation_form_gradient(Vec3 g,const RadialForms& forms) const {
  std::array<std::vector<Complex>,3> result;for(auto& row:result)row.assign(dimension()*dimension(),0.);
  const double magnitude=length(g);const double components[3]={g.x,g.y,g.z};
  if(magnitude!=forms.magnitude)throw std::invalid_argument("UPF exact-radius cache mismatch");
  std::vector<double> harmonics((density_lmax()+1)*(density_lmax()+1));
  std::vector<Vec3> gradients(harmonics.size());
  if(magnitude>=1e-12)for(int l=0;l<=density_lmax();++l)for(int m=-l;m<=l;++m){
    harmonics[l*l+m+l]=upf_real_spherical_harmonic(l,m,g);
    gradients[l*l+m+l]=paw_real_spherical_harmonic_gradient(l,m,g);
  }
  for(const auto& c:coupling_){const auto key=std::make_tuple(std::min(c.i,c.j),std::max(c.i,c.j),c.l);
    const auto it=forms.pairs.find(key);if(it==forms.pairs.end())throw std::invalid_argument("incomplete UPF radial forms");
    const auto factor=4*pi*std::pow(Complex(0,-1),c.l)*c.gaunt;
    if(magnitude<1e-12){
      if(c.l==1){ // d[j1(|G|r)Y1m(Ghat)]/dGa at the origin.
        const double radial_derivative=it->second.second;
        const double axis=std::sqrt(3/(4*pi));
        if(c.m==1)result[0][c.a*dimension()+c.b]+=factor*radial_derivative*axis;
        if(c.m==-1)result[1][c.a*dimension()+c.b]+=factor*radial_derivative*axis;
        if(c.m==0)result[2][c.a*dimension()+c.b]+=factor*radial_derivative*axis;
      }
      continue;
    }
    const double y=harmonics[c.l*c.l+c.m+c.l];
    const auto dy=gradients[c.l*c.l+c.m+c.l];const double dyc[3]={dy.x,dy.y,dy.z};
    for(int a=0;a<3;++a)result[a][c.a*dimension()+c.b]+=factor*(
        it->second.second*components[a]/magnitude*y+it->second.first*dyc[a]);
  }
  return result;
}

std::vector<std::vector<double>> UpfPawOperator::radial_density(const std::vector<Complex>& onsite,bool ae) const {
  onsite_check(onsite,dimension());const auto mesh=data_->array("UPF/PP_MESH/PP_R").size();
  std::vector<std::vector<double>> result((density_lmax()+1)*(density_lmax()+1),std::vector<double>(mesh,0.));
  for(const auto& c:coupling_){const double coefficient=c.gaunt*onsite[c.a*dimension()+c.b].real();
    if(coefficient==0)continue;
    const auto& pair=radial_pairs_.at({std::min(c.i,c.j),std::max(c.i,c.j),c.l});
    const auto& product=ae?pair.all_electron:pair.pseudo;
    auto& row=result[c.l*c.l+c.m+c.l];
    for(std::size_t k=0;k<data_->augmentation_count;++k)row[k]+=coefficient*product[k];}
  return result;
}

std::vector<Complex> UpfPawOperator::project_radial_potential(
    const std::vector<std::vector<double>>& potential,bool ae) const {
  const auto mesh=data_->array("UPF/PP_MESH/PP_R").size();
  if(potential.size()!=std::size_t((density_lmax()+1)*(density_lmax()+1)))
    throw std::invalid_argument("UPF radial potential angular dimensions invalid");
  for(const auto& row:potential)if(row.size()!=mesh)throw std::invalid_argument("UPF radial potential mesh dimensions invalid");
  std::vector<Complex> result(dimension()*dimension(),0.);
  for(const auto& c:coupling_){
    const auto& pair=radial_pairs_.at({std::min(c.i,c.j),std::max(c.i,c.j),c.l});
    const auto& product=ae?pair.all_electron:pair.pseudo;
    const auto& row=potential[c.l*c.l+c.m+c.l];double value=0.;
    for(std::size_t k=0;k<data_->augmentation_count;++k)value+=product[k]*row[k];
    result[c.a*dimension()+c.b]+=c.gaunt*value;
  }
  return result;
}
} // namespace sdft
