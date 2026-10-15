#include "thread_budget.hpp"
#include "density_functionals.hpp"
#include "spin_pbe_xc.hpp"
#include "ordered_fourier.hpp"
#include "pbe_xc.hpp"
#include "compute_backend.hpp"
#include "fft_planning.hpp"
#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <memory>
#include <map>

namespace sdft {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
using Complex = std::complex<double>;
bool parallel_grid_memory(std::size_t count){
  const char* flag=std::getenv("SDFT_DENSITY_MEMORY_PARALLEL");
  if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")
    throw std::invalid_argument("SDFT_DENSITY_MEMORY_PARALLEL must be 0 or 1");
#ifdef SDFT_HAS_OPENMP
  return (!flag||std::string(flag)=="1")&&count>=65536&&!omp_in_parallel();
#else
  (void)count;return false;
#endif
}

Vec3 scale(Vec3 v, double t) { return {v.x * t, v.y * t, v.z * t}; }
Vec3 reciprocal_vector(const std::array<Vec3, 3>& b, const std::tuple<int, int, int>& key) {
  const auto [h, k, l] = key;
  return {h*b[0].x + k*b[1].x + l*b[2].x,
          h*b[0].y + k*b[1].y + l*b[2].y,
          h*b[0].z + k*b[1].z + l*b[2].z};
}

class FourierGrid {
 public:
  FourierGrid(int nx, int ny, int nz,int threads): nx_(nx), ny_(ny), nz_(nz),
      count_(static_cast<std::size_t>(nx) * ny * nz) {
    // FFTW_MEASURE timed ~3.5 s of planning per grid at 8 threads on Si8
    // (6 SCF iterations: density_functional 4.28 s vs 0.77 s with ESTIMATE)
    // with equal transform time afterwards, and its plan choice depends on
    // timing noise. ESTIMATE is the default; SDFT_DETERMINISTIC=1 keeps the
    // historical MEASURE planner.
    unsigned flags=deterministic_mode()?FFTW_MEASURE:FFTW_ESTIMATE;
    if(const char* mode=std::getenv("SDFT_DENSITY_FFT_PLAN")){
      const std::string choice(mode);
      if(choice!="estimate"&&choice!="measure")throw std::invalid_argument("SDFT_DENSITY_FFT_PLAN must be estimate or measure");
      flags=choice=="estimate"?FFTW_ESTIMATE:FFTW_MEASURE;
    }
    planning_flags_=flags;
    data_ = fftw_alloc_complex(count_);
    if (!data_) throw std::bad_alloc();
    std::lock_guard lock(fft_planner_mutex());
    set_fft_planner_threads(threads);
    forward_ = fftw_plan_dft_3d(nx, ny, nz, data_, data_, FFTW_FORWARD, flags);
    backward_ = fftw_plan_dft_3d(nx, ny, nz, data_, data_, FFTW_BACKWARD, flags);
    set_fft_planner_threads(1);
    if (!forward_ || !backward_) {
      if (forward_) fftw_destroy_plan(forward_);
      if (backward_) fftw_destroy_plan(backward_);
      fftw_free(data_);
      throw std::runtime_error("FFTW density-grid plan failed");
    }
  }
  void add_wavefunction_density(const std::vector<std::size_t>& indices,const Complex* coefficients,
      double weight,std::vector<double>& density){
    std::fill_n(&data_[0][0],2*count_,0.);
    for(std::size_t g=0;g<indices.size();++g){data_[indices[g]][0]=coefficients[g].real();data_[indices[g]][1]=coefficients[g].imag();}
    fftw_execute(backward_);
    for(std::size_t i=0;i<count_;++i)density[i]+=weight*(data_[i][0]*data_[i][0]+data_[i][1]*data_[i][1]);
  }
  void add_density_bands(const std::vector<std::size_t>& indices,const std::vector<Complex>& coefficients,
      const std::vector<double>& occupations,double volume,std::vector<double>& density,std::size_t workers){
    if(!band_backward_){std::lock_guard lock(fft_planner_mutex());set_fft_planner_threads(1);
      band_backward_=fftw_plan_dft_3d(nx_,ny_,nz_,data_,data_,FFTW_BACKWARD,FFTW_ESTIMATE);
      if(!band_backward_)throw std::runtime_error("band-parallel density plan failed");}
    while(band_buffers_.size()<workers){auto* b=fftw_alloc_complex(count_);if(!b)throw std::bad_alloc();band_buffers_.push_back(b);}
    // Parallel transforms, then deterministic band-order accumulation for
    // each grid point. No thread-dependent reduction of electron density.
    for(std::size_t first=0;first<occupations.size();first+=workers){
      const auto width=std::min(workers,occupations.size()-first);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1)
#endif
      for(std::size_t slot=0;slot<width;++slot){auto* work=band_buffers_[slot];std::fill_n(&work[0][0],2*count_,0.);
        if(occupations[first+slot]==0)continue;
        for(std::size_t g=0;g<indices.size();++g){const auto c=coefficients[(first+slot)*indices.size()+g];work[indices[g]][0]=c.real();work[indices[g]][1]=c.imag();}
        fftw_execute_dft(band_backward_,work,work);
      }
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1 && count_>=16384)
#endif
      for(std::size_t i=0;i<count_;++i)for(std::size_t slot=0;slot<width;++slot){
        if(occupations[first+slot]==0)continue;const auto* w=band_buffers_[slot];
        density[i]+=occupations[first+slot]/volume*(w[i][0]*w[i][0]+w[i][1]*w[i][1]);
      }
    }
    record_performance("density_parallel_band_transforms",0,occupations.size());
  }
  ~FourierGrid() {
    std::lock_guard lock(fft_planner_mutex());
    fftw_destroy_plan(forward_);
    fftw_destroy_plan(backward_);
    if(band_backward_)fftw_destroy_plan(band_backward_);
    if(fields_forward_)fftw_destroy_plan(fields_forward_);
    if(fields_backward_)fftw_destroy_plan(fields_backward_);
    if(fields_buffer_)fftw_free(fields_buffer_);
    for(auto* b:band_buffers_)fftw_free(b);
    fftw_free(data_);
  }
  FourierGrid(const FourierGrid&) = delete;
  FourierGrid& operator=(const FourierGrid&) = delete;

  std::size_t index(const std::tuple<int, int, int>& key) const {
    const auto [h, k, l] = key;
    const auto wrap = [](int p, int dim) { const int value = p % dim; return value < 0 ? value + dim : value; };
    return (static_cast<std::size_t>(wrap(h, nx_)) * ny_ + wrap(k, ny_)) * nz_ + wrap(l, nz_);
  }
  std::size_t count() const { return count_; }
  // Two complex transforms carry four real fields. The original pair
  // packing arithmetic is unchanged; FFTW can partition both transforms
  // together instead of repeatedly launching workers for each small grid.
  std::array<std::vector<double>,4> real_fields_batched(
      const FourierDensity& density,const std::array<Vec3,3>& reciprocal) {
    prepare_field_plans();
    PerformanceTimer timer("density_batched_inverse_fft");
    std::fill_n(&fields_buffer_[0][0],4*count_,0.);
    const auto fill=[&](int field,int axis,bool imaginary){
      auto* work=fields_buffer_+field*count_;
      for(const auto& [key,value]:density){
        Complex packed=value;
        if(axis>=0){const auto q=reciprocal_vector(reciprocal,key);
          packed*=Complex(0,axis==0?q.x:axis==1?q.y:q.z);}
        if(imaginary)packed*=Complex(0,1);
        const auto i=index(key);work[i][0]+=packed.real();work[i][1]+=packed.imag();
      }
    };
    fill(0,-1,false);fill(0,2,true);fill(1,0,false);fill(1,1,true);
    fftw_execute(fields_backward_);
    std::array<std::vector<double>,4> out;
    for(auto& v:out)v.resize(count_);
    for(std::size_t i=0;i<count_;++i){
      out[0][i]=fields_buffer_[i][0];out[3][i]=fields_buffer_[i][1];
      out[1][i]=fields_buffer_[count_+i][0];out[2][i]=fields_buffer_[count_+i][1];
    }
    record_performance("density_batched_inverse_transforms",0,2);
    return out;
  }
  std::array<std::vector<Complex>,4> fourier_fields_batched(
      const std::vector<double>& a,const std::vector<double>& b,
      const std::vector<double>& c,const std::vector<double>& d,
      const auto& support) {
    if(a.size()!=count_||b.size()!=count_||c.size()!=count_||d.size()!=count_)
      throw std::invalid_argument("batched density FFT real grid size mismatch");
    prepare_field_plans();
    PerformanceTimer timer("density_batched_forward_fft");
    for(std::size_t i=0;i<count_;++i){
      fields_buffer_[i][0]=a[i];fields_buffer_[i][1]=b[i];
      fields_buffer_[count_+i][0]=c[i];fields_buffer_[count_+i][1]=d[i];
    }
    fftw_execute(fields_forward_);
    std::array<std::vector<Complex>,4> out;
    for(auto& v:out)v.reserve(support.size());
    for(const auto& [key,ignored]:support){
      (void)ignored;const auto [h,k,l]=key;
      const auto i=index(key),j=index({-h,-k,-l});
      for(std::size_t field=0;field<2;++field){const auto* work=fields_buffer_+field*count_;
        const Complex direct(work[i][0],work[i][1]),opposite(work[j][0],-work[j][1]);
        out[2*field].push_back((direct+opposite)/(2.*double(count_)));
        out[2*field+1].push_back(Complex(0,-1)*(direct-opposite)/(2.*double(count_)));
      }
    }
    record_performance("density_batched_forward_transforms",0,2);
    return out;
  }
  std::pair<std::vector<double>,std::vector<double>> real_components(
      const auto& first,const auto& second,
      const std::array<Vec3,3>& reciprocal,int axis_a,int axis_b) {
    std::fill_n(&data_[0][0],2*count_,0.);
    const auto fill=[&](const auto& field,int axis,bool imaginary){
      for(const auto& [key,value]:field){
        Complex packed=value;
        if(axis>=0){const auto q=reciprocal_vector(reciprocal,key);
          packed*=Complex(0,axis==0?q.x:axis==1?q.y:q.z);}
        if(imaginary)packed*=Complex(0,1);
        const auto i=index(key);data_[i][0]+=packed.real();data_[i][1]+=packed.imag();
      }
    };
    fill(first,axis_a,false);fill(second,axis_b,true);fftw_execute(backward_);
    std::vector<double> a(count_),b(count_);
    const bool parallel=parallel_grid_memory(count_);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for(std::size_t i=0;i<count_;++i){a[i]=data_[i][0];b[i]=data_[i][1];}
    return {std::move(a),std::move(b)};
  }
  const std::array<std::vector<double>,4>& frozen_core_fields(
      const FourierDensity& core,const std::array<Vec3,3>& reciprocal) {
    bool same=core==core_coefficients_;
    for(int a=0;a<3;++a)same=same&&reciprocal[a].x==core_reciprocal_[a].x&&
        reciprocal[a].y==core_reciprocal_[a].y&&reciprocal[a].z==core_reciprocal_[a].z;
    if(!same||core_fields_[0].empty()){
      PerformanceTimer timer("density_frozen_core_fft");
      auto nz=real_components(core,core,reciprocal,-1,2);
      auto xy=real_components(core,core,reciprocal,0,1);
      core_fields_={std::move(nz.first),std::move(xy.first),std::move(xy.second),std::move(nz.second)};
      core_coefficients_=core;core_reciprocal_=reciprocal;
      record_performance("density_frozen_core_cache_miss",0);
    }else record_performance("density_frozen_core_cache_hit",0);
    return core_fields_;
  }
  void add_wavefunction_density(const FourierDensity& coefficients,
      double weight, std::vector<double>& density) {
    std::fill_n(&data_[0][0],2*count_,0.0);
    for(const auto& [key,value]:coefficients) {
      const auto i=index(key);data_[i][0]=value.real();data_[i][1]=value.imag();
    }
    fftw_execute(backward_);
    for(std::size_t i=0;i<count_;++i)
      density[i]+=weight*(data_[i][0]*data_[i][0]+data_[i][1]*data_[i][1]);
  }
  std::vector<double> to_real(const auto& coefficients,
      const std::array<Vec3,3>* reciprocal=nullptr,int component=0) {
    std::fill_n(&data_[0][0], 2 * count_, 0.0);
    for (const auto& [key, value] : coefficients) {
      const auto i = index(key);
      const Vec3 q=reciprocal ? reciprocal_vector(*reciprocal,key) : Vec3{};
      const double scalar=component==0?q.x:component==1?q.y:q.z;
      const Complex packed=reciprocal ? Complex(0,scalar)*value : value;
      data_[i][0] = packed.real(); data_[i][1] = packed.imag();
    }
    fftw_execute(backward_);
    std::vector<double> result(count_);
    const bool parallel=parallel_grid_memory(count_);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (std::size_t i = 0; i < count_; ++i) result[i] = data_[i][0];
    return result;
  }
  std::pair<std::vector<double>,std::vector<double>> to_real_pair(
      const auto& first,const auto& second,
      const std::array<Vec3,3>* reciprocal=nullptr,int first_axis=0,int second_axis=1){
    std::fill_n(&data_[0][0],2*count_,0.);
    const auto fill=[&](const auto& field,int axis,bool imaginary){
      for(const auto& [key,value]:field){
        Complex packed=value;
        if(reciprocal){const auto q=reciprocal_vector(*reciprocal,key);packed*=Complex(0,axis==0?q.x:axis==1?q.y:q.z);}
        if(imaginary)packed*=Complex(0,1);
        const auto i=index(key);data_[i][0]+=packed.real();data_[i][1]+=packed.imag();
      }
    };
    fill(first,first_axis,false);fill(second,second_axis,true);fftw_execute(backward_);
    std::vector<double> a(count_),b(count_);
    const bool parallel=parallel_grid_memory(count_);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for(std::size_t i=0;i<count_;++i){a[i]=data_[i][0];b[i]=data_[i][1];}
    return {std::move(a),std::move(b)};
  }
  FourierDensity to_fourier(const std::vector<double>& values, const auto& support) {
    if (values.size() != count_) throw std::invalid_argument("FFT real grid size mismatch");
    const bool parallel=parallel_grid_memory(count_);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (std::size_t i = 0; i < count_; ++i) { data_[i][0] = values[i]; data_[i][1] = 0.0; }
    fftw_execute(forward_);
    FourierDensity result;
    for (const auto& [key, ignored] : support) {
      (void)ignored;
      const auto i = index(key);
      result.emplace_hint(result.end(),key,Complex(data_[i][0], data_[i][1]) / static_cast<double>(count_));
    }
    return result;
  }
  std::vector<Complex> to_fourier_values(const std::vector<double>& values,const auto& support){
    if(values.size()!=count_)throw std::invalid_argument("FFT real grid size mismatch");
    const bool parallel=parallel_grid_memory(count_);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for(std::size_t i=0;i<count_;++i){data_[i][0]=values[i];data_[i][1]=0.;}
    fftw_execute(forward_);std::vector<Complex> result;result.reserve(support.size());
    for(const auto& [key,ignored]:support){(void)ignored;const auto i=index(key);result.emplace_back(data_[i][0]/double(count_),data_[i][1]/double(count_));}
    return result;
  }
  std::pair<std::vector<Complex>,std::vector<Complex>> to_fourier_pair(
      const std::vector<double>& a,const std::vector<double>& b,const auto& support){
    if(a.size()!=count_||b.size()!=count_)throw std::invalid_argument("paired FFT real grid size mismatch");
    const bool parallel=parallel_grid_memory(count_);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for(std::size_t i=0;i<count_;++i){data_[i][0]=a[i];data_[i][1]=b[i];}
    fftw_execute(forward_);std::vector<Complex> first,second;first.reserve(support.size());second.reserve(support.size());
    for(const auto& [key,ignored]:support){
      (void)ignored;const auto [h,k,l]=key;
      const auto i=index(key),j=index({-h,-k,-l});
      const Complex direct(data_[i][0],data_[i][1]),opposite(data_[j][0],-data_[j][1]);
      first.push_back((direct+opposite)/(2.*double(count_)));
      second.push_back(Complex(0,-1)*(direct-opposite)/(2.*double(count_)));
    }
    return {std::move(first),std::move(second)};
  }

 private:
  void prepare_field_plans(){
    const int threads=density_fft_threads(2*count_);
    if(fields_forward_&&fields_backward_&&threads==fields_threads_){
      record_performance("density_field_plan_cache_hit",0);return;
    }
    if(count_>std::size_t(std::numeric_limits<int>::max())||
       count_>std::numeric_limits<std::size_t>::max()/(2*sizeof(fftw_complex)))
      throw std::overflow_error("batched density FFT grid exceeds FFTW limits");
    std::lock_guard lock(fft_planner_mutex());
    if(fields_forward_)fftw_destroy_plan(fields_forward_);
    if(fields_backward_)fftw_destroy_plan(fields_backward_);
    fields_forward_=nullptr;fields_backward_=nullptr;
    if(!fields_buffer_)fields_buffer_=fftw_alloc_complex(2*count_);
    if(!fields_buffer_)throw std::bad_alloc();
    const int dims[]{nx_,ny_,nz_};
    set_fft_planner_threads(threads);
    fields_forward_=fftw_plan_many_dft(3,dims,2,fields_buffer_,nullptr,1,int(count_),
        fields_buffer_,nullptr,1,int(count_),FFTW_FORWARD,planning_flags_);
    fields_backward_=fftw_plan_many_dft(3,dims,2,fields_buffer_,nullptr,1,int(count_),
        fields_buffer_,nullptr,1,int(count_),FFTW_BACKWARD,planning_flags_);
    set_fft_planner_threads(1);
    if(!fields_forward_||!fields_backward_)throw std::runtime_error("batched density FFT planning failed");
    fields_threads_=threads;
    record_performance("density_field_plan_cache_miss",0);
  }
  int nx_, ny_, nz_;
  std::size_t count_;
  fftw_complex* data_{};
  fftw_plan forward_{}, backward_{};
  fftw_plan band_backward_{};
  unsigned planning_flags_{};
  fftw_complex* fields_buffer_{};
  fftw_plan fields_forward_{},fields_backward_{};
  int fields_threads_{};
  std::vector<fftw_complex*> band_buffers_;
  FourierDensity core_coefficients_;
  std::array<Vec3,3> core_reciprocal_{};
  std::array<std::vector<double>,4> core_fields_;
};
FourierGrid& cached_grid(int nx,int ny,int nz){
  if(const char* setting=std::getenv("SDFT_FAST_DENSITY_GRID")) {
    if(std::string(setting)!="0"&&std::string(setting)!="1")
      throw std::invalid_argument("SDFT_FAST_DENSITY_GRID must be 0 or 1");
    if(std::string(setting)=="1") {
      // Only enlarge the existing alias-free/oversampled grid. Retain every
      // Fourier mode and the original padding; avoid large-prime FFT axes.
      nx=fast_fft_length(nx);ny=fast_fft_length(ny);nz=fast_fft_length(nz);
      record_performance("density_smooth_grid_requests",0);
    }
  }
  using Key=std::tuple<int,int,int,int>;
  thread_local std::map<Key,std::unique_ptr<FourierGrid>> cache;
  const int threads=density_fft_threads(std::size_t(nx)*ny*nz);
  const Key key{nx,ny,nz,threads};auto found=cache.find(key);
  if(found==cache.end()){
    // Bounded per-worker working set; geometrically changing jobs cannot
    // accumulate an unbounded set of grid plans and arrays.
    if(cache.size()>=4)cache.erase(cache.begin());
    found=cache.emplace(key,std::make_unique<FourierGrid>(nx,ny,nz,threads)).first;
    record_performance("density_fft_plan_cache_miss",0);
  }else record_performance("density_fft_plan_cache_hit",0);
  return *found->second;
}
}

FourierDensity plane_wave_difference_support(const std::vector<std::tuple<int,int,int>>& g) {
  if(g.empty())throw std::invalid_argument("empty plane-wave support");
  int low[3]{std::get<0>(g.front()),std::get<1>(g.front()),std::get<2>(g.front())},high[3]{low[0],low[1],low[2]};
  FourierDensity mask;
  for(const auto& key:g) {
    if(!mask.emplace(key,1.).second)throw std::invalid_argument("duplicate plane-wave support index");
    const int k[]{std::get<0>(key),std::get<1>(key),std::get<2>(key)};
    for(int a=0;a<3;++a){low[a]=std::min(low[a],k[a]);high[a]=std::max(high[a],k[a]);}
  }
  int width[3]{high[0]-low[0],high[1]-low[1],high[2]-low[2]};
  auto& fft=cached_grid(std::max(3,2*width[0]+1),std::max(3,2*width[1]+1),std::max(3,2*width[2]+1));
  FourierDensity box;
  for(int h=-width[0];h<=width[0];++h)for(int k=-width[1];k<=width[1];++k)for(int l=-width[2];l<=width[2];++l)box[{h,k,l}]=0.;
  std::vector<double> intensity(fft.count());fft.add_wavefunction_density(mask,1.,intensity);
  const auto counts=fft.to_fourier(intensity,box);FourierDensity out;
  const double roundoff=std::max(1e-7,64*std::numeric_limits<double>::epsilon()*double(g.size())*double(g.size()));
  if(roundoff>=.1)throw std::runtime_error("basis-mask autocorrelation exceeds safe integer resolution");
  for(const auto& [key,count]:counts) {
    const double integer=std::round(count.real());
    if(std::abs(count.imag())>roundoff||std::abs(count.real()-integer)>roundoff||integer<0||integer>double(g.size()))
      throw std::runtime_error("basis-mask FFT did not recover integer difference counts");
    if(integer>.5)out[key]=1.;
  }
  return out;
}

FourierDensity plane_wave_density_fft(
    const std::vector<std::tuple<int,int,int>>& g,
    const std::vector<Complex>& coefficients,const std::vector<double>& occupations,
    double volume,const FourierDensity& support) {
  PerformanceTimer density_timer("wavefunction_density_fft");
  if(g.empty()||support.empty()||!std::isfinite(volume)||volume<=0||
     coefficients.size()!=g.size()*occupations.size())
    throw std::invalid_argument("invalid plane-wave FFT density dimensions");
  int mh=0,mk=0,ml=0;
  for(const auto& [key,value]:support) {
    (void)value;const auto [h,k,l]=key;
    mh=std::max(mh,std::abs(h));mk=std::max(mk,std::abs(k));ml=std::max(ml,std::abs(l));
  }
  // Each density frequency is a G-G' difference. N>2*max|G-G'| avoids
  // circular convolution folding any product into another retained mode.
  int min_g[3]{},max_g[3]{};
  for(const auto& key:g) {
    const int c[]{std::get<0>(key),std::get<1>(key),std::get<2>(key)};
    for(int a=0;a<3;++a){min_g[a]=std::min(min_g[a],c[a]);max_g[a]=std::max(max_g[a],c[a]);}
  }
  mh=std::max(mh,max_g[0]-min_g[0]);mk=std::max(mk,max_g[1]-min_g[1]);ml=std::max(ml,max_g[2]-min_g[2]);
  auto& fft=cached_grid(std::max(3,2*mh+1),std::max(3,2*mk+1),std::max(3,2*ml+1));
  std::vector<double> density(fft.count());
  std::vector<std::size_t> indices;indices.reserve(g.size());
  for(const auto& key:g)indices.push_back(fft.index(key));
  std::size_t band_threads=0;
  if(const char* value=std::getenv("SDFT_DENSITY_BAND_THREADS")){
    const std::string text(value);std::size_t end=0;
    if(text.empty()||text[0]=='-')throw std::invalid_argument("invalid density band threads");
    band_threads=std::stoull(text,&end);
    if(end!=text.size()||band_threads>256)throw std::invalid_argument("density band threads must be in [0,256]");
  }
#ifdef SDFT_HAS_OPENMP
  band_threads=omp_in_parallel()?0:std::min<std::size_t>(band_threads,omp_get_max_threads());
#else
  band_threads=std::min<std::size_t>(band_threads,1);
#endif
  // At most 256 MiB of aligned extra transform buffers per grid context.
  band_threads=std::min({band_threads,occupations.size(),std::max<std::size_t>(1,(256*1024*1024)/(sizeof(Complex)*fft.count()))});
  for(std::size_t band=0;band<occupations.size();++band) {
    const double occupation=occupations[band];
    if(!std::isfinite(occupation)||occupation<0||occupation>2)
      throw std::invalid_argument("invalid FFT spin-summed occupation");
    if(occupation==0)continue;
    for(std::size_t i=0;i<g.size();++i) {
      const auto value=coefficients[band*g.size()+i];
      if(!std::isfinite(value.real())||!std::isfinite(value.imag()))
        throw std::invalid_argument("nonfinite wavefunction coefficient");
    }
    if(!band_threads)fft.add_wavefunction_density(indices,coefficients.data()+band*g.size(),occupation/volume,density);
  }
  if(band_threads)fft.add_density_bands(indices,coefficients,occupations,volume,density,band_threads);
  return fft.to_fourier(density,support);
}

static DensityFunctionals evaluate_density_functionals_impl(const FourierDensity& density,
    const std::array<Vec3, 3>& lattice_bohr,
    const FourierDensity* frozen_pseudo_core, int quadrature_padding, bool allow_accelerator,
    const auto& xc_density) {
  const double det = determinant(lattice_bohr);
  const double volume = std::abs(det);
  if (volume < 1e-12 || !std::isfinite(volume)) throw std::invalid_argument("singular density-grid lattice");
  if (density.empty()) throw std::invalid_argument("density Fourier support is empty");
  const std::array<Vec3, 3> b{{scale(cross(lattice_bohr[1], lattice_bohr[2]), 2.0*pi/det),
                             scale(cross(lattice_bohr[2], lattice_bohr[0]), 2.0*pi/det),
                             scale(cross(lattice_bohr[0], lattice_bohr[1]), 2.0*pi/det)}};
  int mh = 0, mk = 0, ml = 0;
  const bool ordered=ordered_fourier_enabled();
  DensityFunctionals result;
  for (const auto& [key, rho] : density) {
    if (!std::isfinite(rho.real()) || !std::isfinite(rho.imag()))
      throw std::invalid_argument("non-finite valence density Fourier coefficient");
    const Vec3 q = reciprocal_vector(b, key);
    const double q2 = dot(q, q);
    if (q2 > 1e-12) {
      result.hartree_energy += 2.0 * pi * volume * std::norm(rho) / q2;
      const double qs[]{q.x,q.y,q.z};
      for(int a=0;a<3;++a) {
        for(int c=0;c<3;++c) {
          auto& row=result.hartree_stress[a];
          (c==0?row.x:c==1?row.y:row.z)+=4*pi*std::norm(rho)*qs[a]*qs[c]/(q2*q2);
        }
      }
      const auto potential=4.0*pi*rho/q2;
      if(ordered)result.potential.emplace_hint(result.potential.end(),key,potential);
      else result.potential[key]=potential;
    } else {
      if(ordered)result.potential.emplace_hint(result.potential.end(),key,0.);
      else result.potential[key]=0.;
    }
  }
  for (const auto& [key, rho] : xc_density) {
    if (!std::isfinite(rho.real()) || !std::isfinite(rho.imag()))
      throw std::invalid_argument("non-finite valence plus pseudo-core Fourier coefficient");
    const auto [h, k, l] = key;
    mh = std::max(mh, std::abs(h)); mk = std::max(mk, std::abs(k)); ml = std::max(ml, std::abs(l));
  }
  // Sampling the nonlinear GGA on the minimal Fourier-support grid aliases
  // products back into retained modes and can break translation invariance
  // of the discrete XC energy. Zero padding changes only quadrature, not the
  // plane-wave cutoff or density support. Retain a diagnostic control.
  int padding = 2;
  if (const char* env = std::getenv("SDFT_FFT_OVERSAMPLING")) {
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (!end || *end != '\0' || parsed < 1 || parsed > 8)
      throw std::invalid_argument("SDFT_FFT_OVERSAMPLING must be an integer in [1,8]");
    padding = static_cast<int>(parsed);
  }
  if(quadrature_padding){
    if(quadrature_padding<1||quadrature_padding>8)throw std::invalid_argument("invalid XC quadrature override");
    padding=quadrature_padding;
  }
  auto& fft=cached_grid(padding*std::max(3, 2*mh+1), padding*std::max(3, 2*mk+1), padding*std::max(3, 2*ml+1));
  result.grid_points = fft.count();
  const double weight = volume / static_cast<double>(fft.count());
  const bool paired=!(std::getenv("SDFT_PAIRED_REAL_FFT")&&std::string(std::getenv("SDFT_PAIRED_REAL_FFT"))=="0");
  const auto enabled=[](const char* name){const char* p=std::getenv(name);
    if(p&&std::string(p)!="0"&&std::string(p)!="1")throw std::invalid_argument(std::string(name)+" must be 0 or 1");
    return !p||std::string(p)=="1";};
  const bool core_reuse=paired&&frozen_pseudo_core&&enabled("SDFT_FROZEN_CORE_FFT_REUSE");
  const char* batched_control=std::getenv("SDFT_DENSITY_BATCHED_FFT");
  if(batched_control&&std::string(batched_control)!="0"&&std::string(batched_control)!="1")
    throw std::invalid_argument("SDFT_DENSITY_BATCHED_FFT must be 0 or 1");
  // Experimental until the complete workflow and scale gates pass.
  const bool batched=batched_control&&std::string(batched_control)=="1";
  std::pair<std::vector<double>,std::vector<double>> densities,gradients;
  std::vector<double> gz;
  if(core_reuse){
    const auto& core=fft.frozen_core_fields(*frozen_pseudo_core,b);
    if(batched){
      auto fields=fft.real_fields_batched(density,b);
      densities.second=std::move(fields[0]);gradients.first=std::move(fields[1]);
      gradients.second=std::move(fields[2]);gz=std::move(fields[3]);
    }else{
      auto nz=fft.real_components(density,density,b,-1,2);
      gradients=fft.real_components(density,density,b,0,1);
      densities.second=std::move(nz.first);gz=std::move(nz.second);
    }
    densities.first.resize(fft.count());
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(fft.count()>16384)
#endif
    for(std::size_t i=0;i<fft.count();++i){
      densities.first[i]=densities.second[i]+core[0][i];
      gradients.first[i]+=core[1][i];gradients.second[i]+=core[2][i];gz[i]+=core[3][i];
    }
  }else{
    densities=paired&&frozen_pseudo_core ? fft.to_real_pair(xc_density,density) :
        std::make_pair(fft.to_real(xc_density),frozen_pseudo_core?fft.to_real(density):std::vector<double>{});
    gradients=paired ? fft.to_real_pair(xc_density,xc_density,&b,0,1) :
        std::make_pair(fft.to_real(xc_density,&b,0),fft.to_real(xc_density,&b,1));
    gz=fft.to_real(xc_density,&b,2);
  }
  const auto& real_density=densities.first;
  const auto& valence_density=frozen_pseudo_core?densities.second:densities.first;
  const auto& gx=gradients.first;
  const auto& gy=gradients.second;
  const bool tiled=enabled("SDFT_DENSITY_TILED_REDUCTION");
  const bool fuse_control=enabled("SDFT_DENSITY_FUSE_PBE_TILES");
  const bool fused=tiled&&fuse_control;
  std::vector<double> local(fft.count()), dx(fft.count()), dy(fft.count()), dz(fft.count()), energy(fused?0:fft.count());
  std::vector<PbeXcPoint> accelerated_points;
  bool gpu_pbe=false;
  const char* pbe_flag=std::getenv("SDFT_GPU_PBE");
  if(allow_accelerator&&pbe_flag&&std::string(pbe_flag)!="0"&&std::string(pbe_flag)!="1")
    throw std::invalid_argument("SDFT_GPU_PBE must be 0 or 1");
  const char* backend=std::getenv("SDFT_COMPUTE_BACKEND");
  // The exact final iteration and CPU paths need no GPU input arrays. Do
  // not allocate/scan two full grids only to have dispatch return false.
  if(allow_accelerator&&pbe_flag&&std::string(pbe_flag)=="1"&&backend&&std::string(backend)!="cpu"){
    std::vector<double> n(fft.count()),gradient(fft.count());
    const bool parallel=parallel_grid_memory(fft.count());
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for(std::size_t i=0;i<fft.count();++i){n[i]=std::max(1e-10,real_density[i]);
      gradient[i]=std::sqrt(gx[i]*gx[i]+gy[i]*gy[i]+gz[i]*gz[i]);}
    gpu_pbe=accelerated_pbe_points(n,gradient,accelerated_points);
  }
  if(!fused){
  PerformanceTimer timer("pbe_grid_evaluation");
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(fft.count()>16384)
#endif
  for (std::size_t i = 0; i < fft.count(); ++i) {
    const double n=std::max(1e-10,real_density[i]);
    const auto xc=gpu_pbe?accelerated_points[i]:pbe_unpolarized(n,std::sqrt(gx[i]*gx[i]+gy[i]*gy[i]+gz[i]*gz[i]));
    energy[i]=xc.energy_density;
    local[i]=xc.exchange_potential+xc.correlation_potential;
    dx[i]=xc.gradient_squared_coefficient*gx[i];
    dy[i]=xc.gradient_squared_coefficient*gy[i];
    dz[i]=xc.gradient_squared_coefficient*gz[i];
  }
  }
  result.minimum_density = std::numeric_limits<double>::infinity();
  if(tiled){
    PerformanceTimer timer(fused?"density_fused_pbe_reduction":"density_tiled_reduction");
    struct Tile {long double electrons{},energy{},stress[9]{};
      double minimum=std::numeric_limits<double>::infinity();std::size_t clipped{};};
    constexpr std::size_t width=2048;
    std::vector<Tile> tiles((fft.count()+width-1)/width);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(fft.count()>16384)
#endif
    for(std::size_t t=0;t<tiles.size();++t){auto& tile=tiles[t];
      for(std::size_t i=t*width;i<std::min(fft.count(),(t+1)*width);++i){
        // Identical point arithmetic and fixed-tile summation order; avoid
        // writing and rereading a global energy grid between parallel teams.
        double e{};
        if(fused){
          const double n=std::max(1e-10,real_density[i]);
          const auto xc=gpu_pbe?accelerated_points[i]:pbe_unpolarized(n,std::sqrt(gx[i]*gx[i]+gy[i]*gy[i]+gz[i]*gz[i]));
          e=xc.energy_density;
          local[i]=xc.exchange_potential+xc.correlation_potential;
          dx[i]=xc.gradient_squared_coefficient*gx[i];
          dy[i]=xc.gradient_squared_coefficient*gy[i];
          dz[i]=xc.gradient_squared_coefficient*gz[i];
        }else e=energy[i];
        const double raw=real_density[i];tile.minimum=std::min(tile.minimum,raw);
        tile.clipped+=raw<1e-10;tile.electrons+=valence_density[i];tile.energy+=e;
        const double g[]{gx[i],gy[i],gz[i]},v[]{dx[i],dy[i],dz[i]};
        const double pressure=e-local[i]*raw-dx[i]*gx[i]-dy[i]*gy[i]-dz[i]*gz[i];
        for(int a=0;a<3;++a)for(int c=0;c<3;++c)tile.stress[3*a+c]+=(a==c?pressure:0.)-v[a]*g[c];
      }
    }
    Tile total;
    // Fixed tiles and ordered extended-precision merge, independent of the
    // thread count. Retain reference serial arithmetic for variational A/B.
    for(const auto& tile:tiles){total.electrons+=tile.electrons;total.energy+=tile.energy;
      total.minimum=std::min(total.minimum,tile.minimum);total.clipped+=tile.clipped;
      for(int j=0;j<9;++j)total.stress[j]+=tile.stress[j];}
    result.sampled_electrons=double(total.electrons*weight);result.xc_energy=double(total.energy*weight);
    result.minimum_density=total.minimum;result.clipped_density_points=total.clipped;
    for(int a=0;a<3;++a){auto& row=result.xc_stress[a];row={double(total.stress[3*a]/fft.count()),
      double(total.stress[3*a+1]/fft.count()),double(total.stress[3*a+2]/fft.count())};}
  }else{
  for (std::size_t i = 0; i < fft.count(); ++i) {
    const double raw = real_density[i];
    result.sampled_electrons += valence_density[i] * weight;
    result.minimum_density = std::min(result.minimum_density, raw);
    if (raw < 1e-10) ++result.clipped_density_points;
    // Ordered FP64 reduction preserves the serial summation convention
    // regardless of thread count. Only independent XC point work is parallel.
    result.xc_energy += energy[i] * weight;
    const double gradients[]{gx[i],gy[i],gz[i]},response[]{dx[i],dy[i],dz[i]};
    const double pressure=(energy[i]-local[i]*raw-
        dx[i]*gx[i]-dy[i]*gy[i]-dz[i]*gz[i])/double(fft.count());
    for(int a=0;a<3;++a) {
      for(int c=0;c<3;++c) {
        auto& row=result.xc_stress[a];
        (c==0?row.x:c==1?row.y:row.z)+=(a==c?pressure:0.)-response[a]*gradients[c]/double(fft.count());
      }
    }
  }
  }
  std::array<std::vector<Complex>,4> transforms;
  if(paired&&batched)transforms=fft.fourier_fields_batched(local,dx,dy,dz,xc_density);
  else{
    auto first=paired?fft.to_fourier_pair(local,dx,xc_density):
        std::make_pair(fft.to_fourier_values(local,xc_density),fft.to_fourier_values(dx,xc_density));
    auto second=paired?fft.to_fourier_pair(dy,dz,xc_density):
        std::make_pair(fft.to_fourier_values(dy,xc_density),fft.to_fourier_values(dz,xc_density));
    transforms={std::move(first.first),std::move(first.second),std::move(second.first),std::move(second.second)};
  }
  const auto& lv=transforms[0];const auto& dxq=transforms[1];
  const auto& dyq=transforms[2];const auto& dzq=transforms[3];
  FourierDensity vxc;std::size_t mode_index=0;
  for (const auto& [key, ignored] : xc_density) {
    (void)ignored;
    const auto q = reciprocal_vector(b, key);
    const Complex value=lv[mode_index]-Complex(0.,1.)*(q.x*dxq[mode_index]+q.y*dyq[mode_index]+q.z*dzq[mode_index]);
    ++mode_index;vxc.emplace_hint(vxc.end(),key,value);
    if(!ordered)result.potential[key] += value;
  }
  if(ordered) {
    PerformanceTimer timer("density_ordered_fourier_potential");
    accumulate_ordered_fourier(result.potential,vxc);
    result.vxc_density=opposite_fourier_contract(density,vxc,volume);
  }else for (const auto& [key, rho] : density) {
      const auto opposite = std::make_tuple(-std::get<0>(key), -std::get<1>(key), -std::get<2>(key));
      const auto it = vxc.find(opposite);
      if (it != vxc.end()) result.vxc_density += volume * (rho * it->second).real();
  }
  result.xc_potential = std::move(vxc);
  for(int a=0;a<3;++a) {auto& row=result.hartree_stress[a];(a==0?row.x:a==1?row.y:row.z)-=result.hartree_energy/volume;}
  return result;
}

DensityFunctionals evaluate_density_functionals(const FourierDensity& density,
    const std::array<Vec3,3>& lattice_bohr,const FourierDensity* core,
    int quadrature_padding,bool allow_accelerator) {
  const char* setting=std::getenv("SDFT_DENSITY_FLAT_SUPPORT");
  if(setting&&std::string(setting)!="0"&&std::string(setting)!="1")
    throw std::invalid_argument("SDFT_DENSITY_FLAT_SUPPORT must be 0 or 1");
  if(setting&&std::string(setting)=="1") {
    if(!core)return evaluate_density_functionals_impl(density,lattice_bohr,core,
        quadrature_padding,allow_accelerator,density);
    using Entry=std::pair<std::tuple<int,int,int>,Complex>;
    std::vector<Entry> xc;
    xc.reserve(density.size()+core->size());
    auto valence=density.begin(),frozen=core->begin();
    while(valence!=density.end()||frozen!=core->end()) {
      if(frozen==core->end()||(valence!=density.end()&&valence->first<frozen->first)) {
        xc.emplace_back(valence->first,valence->second);++valence;
      }else if(valence==density.end()||frozen->first<valence->first) {
        Complex sum{};sum+=frozen->second;
        xc.emplace_back(frozen->first,sum);++frozen;
      }else {
        Complex sum=valence->second;sum+=frozen->second;
        xc.emplace_back(valence->first,sum);++valence;++frozen;
      }
    }
    record_performance("density_flat_support_modes",0,xc.size());
    return evaluate_density_functionals_impl(density,lattice_bohr,core,
        quadrature_padding,allow_accelerator,xc);
  }
  FourierDensity xc=density;
  if(core) {
    if(ordered_fourier_enabled())accumulate_ordered_fourier(xc,*core);
    else for(const auto& [key,rho]:*core)xc[key]+=rho;
  }
  return evaluate_density_functionals_impl(density,lattice_bohr,core,
      quadrature_padding,allow_accelerator,xc);
}
std::array<DensityFunctionals,2> evaluate_collinear_density_functionals(
    const std::array<FourierDensity,2>& density,const std::array<Vec3,3>& lattice,
    const FourierDensity* core,int padding) {
  PerformanceTimer timer("collinear_density_functionals");
  const double det=determinant(lattice),volume=std::abs(det);
  if(!std::isfinite(volume)||volume<1e-12||density[0].empty()||density[1].empty())
    throw std::invalid_argument("invalid collinear density/lattice");
  std::array<Vec3,3> b{scale(cross(lattice[1],lattice[2]),2*pi/det),
    scale(cross(lattice[2],lattice[0]),2*pi/det),scale(cross(lattice[0],lattice[1]),2*pi/det)};
  FourierDensity total=density[0],support=density[0];
  for(const auto& [key,value]:density[1]){total[key]+=value;support.try_emplace(key,0.);}
  if(core)for(const auto& [key,value]:*core)support.try_emplace(key,0.);
  int maxima[3]{};
  for(const auto& [key,value]:support){const auto [h,k,l]=key;
    maxima[0]=std::max(maxima[0],std::abs(h));maxima[1]=std::max(maxima[1],std::abs(k));maxima[2]=std::max(maxima[2],std::abs(l));}
  if(!padding){padding=2;if(const char* v=std::getenv("SDFT_FFT_OVERSAMPLING")){
    char* end=nullptr;auto n=std::strtol(v,&end,10);
    if(!end||*end||n<1||n>8)throw std::invalid_argument("invalid FFT oversampling");padding=int(n);}}
  if(padding<1||padding>8)throw std::invalid_argument("invalid collinear XC padding");
  auto& fft=cached_grid(padding*std::max(3,2*maxima[0]+1),padding*std::max(3,2*maxima[1]+1),padding*std::max(3,2*maxima[2]+1));
  std::array<DensityFunctionals,2> out;
  double eh=0.;std::array<Vec3,3> sh{};FourierDensity vh;
  for(const auto& [key,rho]:total){
    if(!std::isfinite(rho.real())||!std::isfinite(rho.imag()))throw std::invalid_argument("nonfinite spin density");
    const auto q=reciprocal_vector(b,key);const double q2=dot(q,q);
    if(q2<1e-12){vh[key]=0.;continue;}
    eh+=2*pi*volume*std::norm(rho)/q2;vh[key]=4*pi*rho/q2;
    const double x[]{q.x,q.y,q.z};
    for(int a=0;a<3;++a)for(int c=0;c<3;++c){auto& row=sh[a];
      (c==0?row.x:c==1?row.y:row.z)+=4*pi*std::norm(rho)*x[a]*x[c]/(q2*q2);}
  }
  for(int a=0;a<3;++a){auto& row=sh[a];(a==0?row.x:a==1?row.y:row.z)-=eh/volume;}
  std::array<std::array<std::vector<double>,4>,2> fields,response;
  for(int s=0;s<2;++s){FourierDensity xc=density[s];
    if(core)for(const auto& [key,value]:*core)xc[key]+=.5*value;
    fields[s]=fft.real_fields_batched(xc,b);
    for(auto& v:response[s])v.resize(fft.count());
    out[s].potential=vh;out[s].hartree_energy=.5*eh;out[s].hartree_stress=sh;
    for(auto& row:out[s].hartree_stress)row=scale(row,.5);
    out[s].grid_points=fft.count();out[s].minimum_density=std::numeric_limits<double>::infinity();
    auto it=density[s].find({0,0,0});if(it!=density[s].end())out[s].sampled_electrons=volume*it->second.real();
  }
  struct Tile{long double energy{},stress[9]{};double minimum[2]{INFINITY,INFINITY};std::size_t clipped[2]{};};
  constexpr std::size_t width=2048;std::vector<Tile> tiles((fft.count()+width-1)/width);
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) if(fft.count()>16384)
#endif
  for(std::size_t t=0;t<tiles.size();++t){auto& tile=tiles[t];
    for(std::size_t i=t*width;i<std::min(fft.count(),(t+1)*width);++i){
      std::array<std::array<double,3>,2> g{{{fields[0][1][i],fields[0][2][i],fields[0][3][i]},
        {fields[1][1][i],fields[1][2][i],fields[1][3][i]}}};
      // Preserve the legacy total-density floor in the nonmagnetic limit.
      double n[2]{fields[0][0][i],fields[1][0][i]};
      if(n[0]+n[1]<1e-10){n[0]=std::max(n[0],5e-11);n[1]=std::max(n[1],5e-11);}
      const auto xc=pbe_collinear(std::max(5e-11,n[0]),std::max(5e-11,n[1]),g[0],g[1]);tile.energy+=xc.energy_density;
      double pressure=xc.energy_density;
      for(int s=0;s<2;++s){tile.minimum[s]=std::min(tile.minimum[s],fields[s][0][i]);tile.clipped[s]+=fields[s][0][i]<0;
        response[s][0][i]=xc.density_potential[s];pressure-=xc.density_potential[s]*fields[s][0][i];
        for(int a=0;a<3;++a){response[s][a+1][i]=xc.gradient_response[s][a];pressure-=xc.gradient_response[s][a]*g[s][a];}}
      for(int a=0;a<3;++a)for(int c=0;c<3;++c){tile.stress[3*a+c]+=(a==c?pressure:0.);
        for(int s=0;s<2;++s)tile.stress[3*a+c]-=xc.gradient_response[s][a]*g[s][c];}
    }
  }
  Tile sum;for(const auto& t:tiles){sum.energy+=t.energy;for(int j=0;j<9;++j)sum.stress[j]+=t.stress[j];
    for(int s=0;s<2;++s){sum.minimum[s]=std::min(sum.minimum[s],t.minimum[s]);sum.clipped[s]+=t.clipped[s];}}
  for(int s=0;s<2;++s){out[s].xc_energy=.5*double(sum.energy)*volume/fft.count();
    out[s].minimum_density=sum.minimum[s];out[s].clipped_density_points=sum.clipped[s];
    for(int a=0;a<3;++a)out[s].xc_stress[a]={.5*double(sum.stress[3*a])/fft.count(),.5*double(sum.stress[3*a+1])/fft.count(),.5*double(sum.stress[3*a+2])/fft.count()};
    auto fourier=fft.fourier_fields_batched(response[s][0],response[s][1],response[s][2],response[s][3],support);
    std::size_t j=0;for(const auto& [key,unused]:support){const auto q=reciprocal_vector(b,key);
      const Complex v=fourier[0][j]-Complex(0.,1.)*(q.x*fourier[1][j]+q.y*fourier[2][j]+q.z*fourier[3][j]);
      out[s].xc_potential.emplace_hint(out[s].xc_potential.end(),key,v);out[s].potential[key]+=v;++j;}
    out[s].vxc_density=opposite_fourier_contract(density[s],out[s].xc_potential,volume);
  }
  return out;
}

}
