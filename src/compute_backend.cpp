#include "compute_backend.hpp"
#include "env_cache.hpp"
#include "thread_budget.hpp"
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
#include "apple_modern_blas.hpp"
#endif
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <stdexcept>
#ifdef SDFT_HAS_CBLAS
#ifdef __APPLE__
#include <vecLib/cblas.h>
#else
#include <cblas.h>
#endif
#endif
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
namespace sdft {
namespace {
double clock_seconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Counter{double seconds{};std::size_t calls{};};
// Each thread accumulates into its own map; a registry merges them for the
// report. The hot path takes only the thread's own (uncontended) lock, so
// k-point teams no longer serialize on one global mutex per timer.
struct CounterShard{std::mutex mutex;std::map<std::string,Counter> counters;};
std::mutex counter_mutex;  // guards the shard registry and device_name
std::vector<std::shared_ptr<CounterShard>>& counter_shards(){static std::vector<std::shared_ptr<CounterShard>> shards;return shards;}
CounterShard& local_counters(){
  thread_local std::shared_ptr<CounterShard> shard=[]{
    auto created=std::make_shared<CounterShard>();
    std::lock_guard lock(counter_mutex);counter_shards().push_back(created);return created;}();
  return *shard;
}
std::map<std::string,Counter> merged_counters(){
  std::map<std::string,Counter> merged;
  std::lock_guard lock(counter_mutex);
  for(const auto& shard:counter_shards()){
    std::lock_guard shard_lock(shard->mutex);
    for(const auto& [key,c]:shard->counters){auto& m=merged[key];m.seconds+=c.seconds;m.calls+=c.calls;}
  }
  return merged;
}
std::string device_name;
std::string mode(){const char* value=cached_getenv("SDFT_COMPUTE_BACKEND");return value?value:"cpu";}
std::size_t min_work(){
  const char* value=cached_getenv("SDFT_GPU_MIN_WORK");if(!value)return 2000000;
  std::size_t end=0;const std::string s(value);
  if(s.empty()||s[0]=='-')throw std::invalid_argument("SDFT_GPU_MIN_WORK must be a nonnegative integer");
  const auto result=std::stoull(s,&end);if(end!=s.size())throw std::invalid_argument("invalid SDFT_GPU_MIN_WORK");return result;
}
Accelerator* accelerator(){
  // One queue/context per host thread. No shared mutable command buffers.
  thread_local std::unique_ptr<Accelerator> instance;
  if(!instance){
    const auto selected=mode();
    if(selected=="mps")instance=make_metal_accelerator();
    else if(selected=="cuda")instance=make_cuda_accelerator();
    else if(selected!="cpu")throw std::invalid_argument("SDFT_COMPUTE_BACKEND must be cpu, mps or cuda");
    if(instance){std::lock_guard lock(counter_mutex);device_name=instance->device();}
  }
  return instance.get();
}
}
PerformanceTimer::PerformanceTimer(std::string name):name_(std::move(name)),start_(clock_seconds()){}
PerformanceTimer::~PerformanceTimer(){record_performance(name_,clock_seconds()-start_);}
void record_performance(const std::string& name,double seconds,std::size_t calls){
  auto& shard=local_counters();std::lock_guard lock(shard.mutex);
  auto& c=shard.counters[name];c.seconds+=seconds;c.calls+=calls;}
std::string compute_backend_name(){accelerator();return mode();}
std::uint64_t new_immutable_matrix_identity(){
  static std::atomic<std::uint64_t> next{1};
  const auto id=next.fetch_add(1,std::memory_order_relaxed);
  if(!id)throw std::overflow_error("immutable matrix identity exhausted");
  return id;
}
void complex_gemm_immutable_left(std::uint64_t identity,MatrixOp ao,std::size_t m,std::size_t n,std::size_t k,
    const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool exact,bool add){
  if(!identity||!m||!n||!k||!a||!b||!c||lda<(ao==MatrixOp::Normal?m:k)||ldb<k||ldc<m)
    throw std::invalid_argument("invalid immutable GEMM operand");
  if(std::max({m,n,k,lda,ldb,ldc})>std::size_t(std::numeric_limits<int>::max()))throw std::overflow_error("immutable BLAS dimension overflow");
  const char* setting=cached_getenv("SDFT_GPU_RESIDENT_PROJECTORS");
  if(setting&&std::string(setting)!="0"&&std::string(setting)!="1")throw std::invalid_argument("SDFT_GPU_RESIDENT_PROJECTORS must be 0 or 1");
  auto* gpu=accelerator();
  if(gpu&&!exact&&(!setting||std::string(setting)=="1")&&double(m)*n*k>=double(min_work())) {
    PerformanceTimer timer("gpu_resident_gemm_synchronized");
    gpu->gemm_immutable_left(identity,ao,m,n,k,a,lda,b,ldb,c,ldc,add);return;
  }
  complex_gemm(ao,MatrixOp::Normal,m,n,k,a,lda,b,ldb,c,ldc,exact,add);
}
bool accelerated_paw_pair(std::uint64_t identity,std::size_t npw,std::size_t projectors,
    std::size_t columns,const Complex64* b,const Complex64* d,const Complex64* q,
    const Complex64* input,Complex64* h,Complex64* s,bool exact,std::uint64_t projector_identity){
  const char* flag=cached_getenv("SDFT_GPU_PAW_CHAIN");
  if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("SDFT_GPU_PAW_CHAIN must be 0 or 1");
  if(exact||!flag||std::string(flag)!="1"||double(npw)*projectors*columns<double(min_work()))return false;
  auto* gpu=accelerator();
  return gpu&&gpu->paw_pair(identity,npw,projectors,columns,b,d,q,input,h,s,projector_identity?projector_identity:identity);
}
std::string compiled_compute_capabilities(){
  return std::string("{\"schema_version\":\"sparse-dft-compute-capabilities-1\",\"cpu_fp64\":true,\"metal_compiled\":")+
#ifdef SDFT_HAS_MPS
      "true"
#else
      "false"
#endif
      +",\"cuda_compiled\":"+
#ifdef SDFT_HAS_CUDA
      "true"
#else
      "false"
#endif
      +",\"gpu_active\":false,\"gpu_execution_requires_numerical_dispatch\":true}";
}
void cpu_real_gemm(MatrixOp ao,MatrixOp bo,std::size_t m,std::size_t n,std::size_t k,
    const double* a,std::size_t lda,const double* b,std::size_t ldb,
    double* c,std::size_t ldc,bool add) {
  if(!m||!n||!k||!a||!b||!c||lda<(ao==MatrixOp::Normal?m:k)||
      ldb<(bo==MatrixOp::Normal?k:n)||ldc<m)
    throw std::invalid_argument("invalid real FP64 GEMM shape/leading dimension");
  if(std::max({m,n,k,lda,ldb,ldc})>std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("real FP64 BLAS dimension overflow");
  PerformanceTimer timer("cpu_real_gemm_fp64");
  const std::string interface=cached_getenv("SDFT_CPU_BLAS_INTERFACE")?
      cached_getenv("SDFT_CPU_BLAS_INTERFACE"):"legacy";
  if(interface!="legacy"&&interface!="modern")
    throw std::invalid_argument("SDFT_CPU_BLAS_INTERFACE must be legacy or modern");
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
  if(interface=="modern") {
    apple_modern_dgemm(ao,bo,int(m),int(n),int(k),a,int(lda),b,int(ldb),c,int(ldc),add);
    return;
  }
#else
  if(interface=="modern")throw std::runtime_error("modern Apple BLAS was not compiled for this platform");
#endif
#ifdef SDFT_HAS_CBLAS
  cblas_dgemm(CblasColMajor,ao==MatrixOp::Normal?CblasNoTrans:CblasTrans,
      bo==MatrixOp::Normal?CblasNoTrans:CblasTrans,int(m),int(n),int(k),1.,
      a,int(lda),b,int(ldb),add?1.:0.,c,int(ldc));
#else
  using M=Eigen::MatrixXd;using Stride=Eigen::OuterStride<>;
  const Eigen::Map<const M,0,Stride> am(a,ao==MatrixOp::Normal?m:k,ao==MatrixOp::Normal?k:m,Stride(lda));
  const Eigen::Map<const M,0,Stride> bm(b,bo==MatrixOp::Normal?k:n,bo==MatrixOp::Normal?n:k,Stride(ldb));
  Eigen::Map<M,0,Stride> cm(c,m,n,Stride(ldc));
  const M av=ao==MatrixOp::Normal?M(am):M(am.transpose()),bv=bo==MatrixOp::Normal?M(bm):M(bm.transpose());
  if(add)cm.noalias()+=av*bv;else cm.noalias()=av*bv;
#endif
}
void complex_gemm(MatrixOp ao,MatrixOp bo,std::size_t m,std::size_t n,std::size_t k,
    const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,
    Complex64* c,std::size_t ldc,bool exact,bool add){
  if(!m||!n||!k||!a||!b||!c||lda<(ao==MatrixOp::Normal?m:k)||ldb<(bo==MatrixOp::Normal?k:n)||ldc<m)
    throw std::invalid_argument("invalid complex GEMM shape/leading dimension");
  if(std::max({m,n,k,lda,ldb,ldc})>std::size_t(std::numeric_limits<int>::max()))throw std::overflow_error("BLAS dimension overflow");
  auto* gpu=accelerator();
  const bool enough=double(m)*double(n)*double(k)>=double(min_work());
  if(gpu&&!exact&&enough){PerformanceTimer timer("gpu_gemm_synchronized");gpu->gemm(ao,bo,m,n,k,a,lda,b,ldb,c,ldc,add);return;}
  PerformanceTimer timer(gpu?(exact?"cpu_exact_gemm":"cpu_small_gemm"):"cpu_gemm");
#ifdef SDFT_HAS_CBLAS
  const Complex64 one(1),beta(add?1.:0.);
  const std::string interface=cached_getenv("SDFT_CPU_BLAS_INTERFACE")?cached_getenv("SDFT_CPU_BLAS_INTERFACE"):"legacy";
  if(interface!="legacy"&&interface!="modern")throw std::invalid_argument("SDFT_CPU_BLAS_INTERFACE must be legacy or modern");
#ifndef SDFT_HAS_APPLE_MODERN_BLAS
  if(interface=="modern")throw std::runtime_error("modern Apple BLAS was not compiled for this platform");
#endif
  const auto native_gemm=[&](int mm,int nn,const Complex64* aa,const Complex64* bb,Complex64* cc) {
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
    if(interface=="modern") {
      apple_modern_zgemm(ao,bo,mm,nn,int(k),aa,int(lda),bb,int(ldb),cc,int(ldc),add);return;
    }
#endif
    cblas_zgemm(CblasColMajor,ao==MatrixOp::Normal?CblasNoTrans:CblasConjTrans,
        bo==MatrixOp::Normal?CblasNoTrans:CblasConjTrans,mm,nn,int(k),&one,
        aa,int(lda),bb,int(ldb),&beta,cc,int(ldc));
  };
#ifdef SDFT_HAS_OPENMP
  // Independent output-column tiles let a single-k SCF use CPU cores even
  // when BLAS itself is single-threaded. Never create a nested team inside
  // k-point/band parallelism. Each tile uses the original leading strides.
  // auto (2026-10-15 default): row tiles when A dominates and one worker
  // per 4e6 MACs. Each tile is an independent BLAS call on the original
  // strides; the contracted dimension is never split. Si64 Davidson replay
  // 81 s -> 54 s; converged Si2/Ni/Si8 unchanged within 2.4e-12 eV.
  const char* partition_setting=cached_getenv("SDFT_CPU_GEMM_PARTITION");
  const std::string partition=partition_setting?partition_setting:(deterministic_mode()?"legacy":"auto");
  if(partition!="legacy"&&partition!="auto"&&partition!="memory"&&partition!="rows"&&partition!="columns"&&partition!="serial")
    throw std::invalid_argument("SDFT_CPU_GEMM_PARTITION must be legacy, auto, memory, rows, columns or serial");
  const bool adaptive_partition=partition=="auto"||partition=="memory";
  const bool row_partition=partition=="rows"||(adaptive_partition&&m>n);
  int team=std::min<std::size_t>(kernel_threads(),std::max(m,n));
  if(partition=="serial")team=1;
  if(adaptive_partition) {
    // Do not spawn eight workers for one worker's worth of work. Row tiles
    // reuse the small right operand when A dominates the memory traffic;
    // column tiles reuse the small left operand in the converse shape.
    team=std::min<int>(team,std::max(1.,std::floor(double(m)*n*k/4000000.)));
    team=std::min<std::size_t>(team,row_partition?m:n);
    // Identical FP64 matrix trials show negative scaling when eight row
    // tiles repeatedly pack a long contracted dimension into small outputs.
    // Cap only this shape class; retain all workers for PAW backprojection
    // and rotations. This never splits/reduces the contracted dimension.
    if(partition=="memory" && k>=4096 && k>=4*std::max(m,n) && std::max(m,n)<=1024)
      team=std::min(team,2);
  }
  if(team>1 && double(m)*n*k>=4000000.){
    if(row_partition||(n<std::size_t(team)&&m>=std::size_t(team))) {
      // Narrow blocks (including CGS2 GEMV fallbacks) have too few output
      // columns to occupy a CPU team. Split independent output rows instead;
      // the contracted dimension is never split or reduced across threads.
      PerformanceTimer parallel_timer("cpu_gemm_parallel_rows");
      const std::size_t width=(m+team-1)/team;
#pragma omp parallel for schedule(static) num_threads(team)
      for(int tile=0;tile<team;++tile) {
        const std::size_t first=std::size_t(tile)*width;
        if(first>=m)continue;
        const std::size_t rows=std::min(width,m-first);
        const Complex64* at=a+(ao==MatrixOp::Normal?first:first*lda);
        native_gemm(int(rows),int(n),at,b,c+first);
      }
      return;
    }
    PerformanceTimer parallel_timer("cpu_gemm_parallel_columns");
    const std::size_t width=(n+team-1)/team;
#pragma omp parallel for schedule(static) num_threads(team)
    for(int tile=0;tile<team;++tile){
      const std::size_t first=std::size_t(tile)*width;
      if(first>=n)continue;
      const std::size_t columns=std::min(width,n-first);
      const Complex64* bt=b+(bo==MatrixOp::Normal?first*ldb:first);
      native_gemm(int(m),int(columns),a,bt,c+first*ldc);
    }
    return;
  }
#endif
  native_gemm(int(m),int(n),a,b,c);
#else
  using M=Eigen::MatrixXcd;using Stride=Eigen::OuterStride<>;
  const Eigen::Map<const M,0,Stride> am(a,ao==MatrixOp::Normal?m:k,ao==MatrixOp::Normal?k:m,Stride(lda));
  const Eigen::Map<const M,0,Stride> bm(b,bo==MatrixOp::Normal?k:n,bo==MatrixOp::Normal?n:k,Stride(ldb));
  Eigen::Map<M,0,Stride> cm(c,m,n,Stride(ldc));
  const M av=ao==MatrixOp::Normal?M(am):M(am.adjoint()),bv=bo==MatrixOp::Normal?M(bm):M(bm.adjoint());
  if(add)cm.noalias()+=av*bv;else cm.noalias()=av*bv;
#endif
}
void complex_gemm_triple_right(std::size_t m,std::size_t n,std::size_t k,
    const std::array<const Complex64*,3>& a,std::size_t lda,
    const Complex64* b,std::size_t ldb,const std::array<Complex64*,3>& c,
    std::size_t ldc,bool exact) {
  if(!m||!n||!k||!b||lda<m||ldb<k||ldc<m||
      std::any_of(a.begin(),a.end(),[](auto p){return !p;})||
      std::any_of(c.begin(),c.end(),[](auto p){return !p;}))
    throw std::invalid_argument("invalid joint Ritz rotation dimensions");
  if(std::max({m,n,k,lda,ldb,ldc})>std::size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("joint Ritz rotation BLAS dimensions overflow");
  const auto interval=[](const Complex64* p,std::size_t rows,std::size_t columns,std::size_t ld) {
    const auto limit=std::numeric_limits<std::uintptr_t>::max();
    if(columns-1>(limit/sizeof(Complex64)-rows)/ld)
      throw std::overflow_error("joint Ritz rotation operand span overflow");
    const auto bytes=((columns-1)*ld+rows)*sizeof(Complex64),first=reinterpret_cast<std::uintptr_t>(p);
    if(first>limit-bytes)throw std::overflow_error("joint Ritz rotation address span overflow");
    return std::pair{first,first+bytes};
  };
  const auto overlaps=[](auto x,auto y){return x.first<y.second&&y.first<x.second;};
  const auto right=interval(b,k,n,ldb);
  for(std::size_t i=0;i<3;++i) {
    const auto out=interval(c[i],m,n,ldc);
    if(overlaps(out,right))throw std::invalid_argument("joint Ritz output aliases right operand");
    for(std::size_t j=0;j<3;++j)
      if(overlaps(out,interval(a[j],m,k,lda))||
          (i!=j&&overlaps(out,interval(c[j],m,n,ldc))))
        throw std::invalid_argument("joint Ritz outputs/inputs must be disjoint");
  }
  auto* gpu=accelerator();
  // Preserve the per-product threshold; never force three tiny products
  // onto an accelerator simply because they share a submission.
  if(gpu&&!exact&&double(m)*n*k>=double(min_work())) {
    PerformanceTimer timer("gpu_triple_rotation_synchronized");
    gpu->gemm_triple_right(m,n,k,a,lda,b,ldb,c,ldc);return;
  }
  PerformanceTimer timer("cpu_triple_rotation");
#ifdef SDFT_HAS_CBLAS
  const std::string interface=cached_getenv("SDFT_CPU_BLAS_INTERFACE")?cached_getenv("SDFT_CPU_BLAS_INTERFACE"):"legacy";
  if(interface!="legacy"&&interface!="modern")throw std::invalid_argument("SDFT_CPU_BLAS_INTERFACE must be legacy or modern");
#ifndef SDFT_HAS_APPLE_MODERN_BLAS
  if(interface=="modern")throw std::runtime_error("modern Apple BLAS was not compiled for this platform");
#endif
  const auto native=[&](std::size_t block,int rows,int columns,std::size_t row,std::size_t column) {
    const auto* left=a[block]+row;const auto* right=b+column*ldb;auto* output=c[block]+row+column*ldc;
#ifdef SDFT_HAS_APPLE_MODERN_BLAS
    if(interface=="modern") {
      apple_modern_zgemm(MatrixOp::Normal,MatrixOp::Normal,rows,columns,int(k),
          left,int(lda),right,int(ldb),output,int(ldc),false);return;
    }
#endif
    const Complex64 one(1.),zero;
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,rows,columns,int(k),&one,
        left,int(lda),right,int(ldb),&zero,output,int(ldc));
  };
#ifdef SDFT_HAS_OPENMP
  // A single CPU team spans all three disjoint outputs. Contracted K stays
  // unsplit, preserving FP64 arithmetic without thread-order reductions.
  int team=1;
  team=int(std::min(double(kernel_threads()),
      std::max(1.,std::floor(3.*double(m)*n*k/4000000.))));
  const bool rows=m>=n;const std::size_t extent=rows?m:n;
  const std::size_t tiles=std::min<std::size_t>(extent,(team+2)/3),width=(extent+tiles-1)/tiles;
#pragma omp parallel for schedule(static) num_threads(team) if(team>1)
  for(std::size_t task=0;task<3*tiles;++task) {
    const auto first=(task%tiles)*width;if(first>=extent)continue;
    const auto length=std::min(width,extent-first),block=task/tiles;
    native(block,int(rows?length:m),int(rows?n:length),rows?first:0,rows?0:first);
  }
  record_performance("cpu_triple_rotation_teams",0,team>1?1:0);
#else
  for(std::size_t i=0;i<3;++i)native(i,int(m),int(n),0,0);
#endif
#else
  for(std::size_t i=0;i<3;++i)
    complex_gemm(MatrixOp::Normal,MatrixOp::Normal,m,n,k,a[i],lda,b,ldb,c[i],ldc,true);
#endif
  record_performance("cpu_triple_rotation_products",0,3);
}
bool accelerated_local_hamiltonian(const std::array<int,3>& shape,
    const std::vector<std::size_t>& indices,const std::vector<double>& kinetic,
    const std::vector<double>& potential,const Complex64* input,
    std::size_t columns,Complex64* output,bool exact,std::uint64_t operator_identity){
  const char* option=cached_getenv("SDFT_GPU_FFT");
  if(option&&std::string(option)!="0"&&std::string(option)!="1")throw std::invalid_argument("SDFT_GPU_FFT must be 0 or 1");
  if(exact||!option||std::string(option)=="0"||mode()=="cpu")return false;
  if(!columns||indices.empty()||indices.size()!=kinetic.size()||!input||!output||
      std::any_of(shape.begin(),shape.end(),[](int s){return s<=0;}))throw std::invalid_argument("invalid accelerated FFT dimensions");
  const std::size_t count=std::size_t(shape[0])*shape[1]*shape[2];
  if(potential.size()!=count)throw std::invalid_argument("accelerated FFT potential grid mismatch");
  for(const auto i:indices)if(i>=count)throw std::invalid_argument("accelerated FFT plane-wave index out of range");
  // Small FFT blocks are dominated by command encoding, 3D stage dispatches
  // and a host wait. Keep those blocks on CPU without changing their grid,
  // operator or precision. A zero threshold explicitly forces real GPU work.
  std::size_t minimum=262144;
  if(const char* setting=cached_getenv("SDFT_GPU_FFT_MIN_POINTS")){
    const std::string text(setting);std::size_t end=0;
    if(text.empty()||text[0]=='-')throw std::invalid_argument("SDFT_GPU_FFT_MIN_POINTS must be a nonnegative integer");
    minimum=std::stoull(text,&end);
    if(end!=text.size())throw std::invalid_argument("invalid SDFT_GPU_FFT_MIN_POINTS");
  }
  if(double(count)*double(columns)<double(minimum)){
    record_performance("gpu_fft_small_block_cpu",0);
    return false;
  }
  PerformanceTimer timer("gpu_fft_local_synchronized");
  const char* reuse=cached_getenv("SDFT_GPU_FFT_IMMUTABLE_CACHE");
  if(reuse&&std::string(reuse)!="0"&&std::string(reuse)!="1")
    throw std::invalid_argument("SDFT_GPU_FFT_IMMUTABLE_CACHE must be 0 or 1");
  accelerator()->local_hamiltonian(shape,indices,kinetic,potential,input,columns,output,
      reuse&&std::string(reuse)=="0"?0:operator_identity);return true;
}
bool accelerated_pbe_points(const std::vector<double>& density,
    const std::vector<double>& gradient,std::vector<PbeXcPoint>& output){
  const char* option=cached_getenv("SDFT_GPU_PBE");
  if(option&&std::string(option)!="0"&&std::string(option)!="1")throw std::invalid_argument("SDFT_GPU_PBE must be 0 or 1");
  if(!option||std::string(option)=="0"||mode()=="cpu")return false;
  if(density.empty()||density.size()!=gradient.size())throw std::invalid_argument("invalid GPU PBE dimensions");
  for(std::size_t i=0;i<density.size();++i)
    if(!(density[i]>0)||!std::isfinite(density[i])||gradient[i]<0||!std::isfinite(gradient[i]))throw std::invalid_argument("invalid GPU PBE point");
  output.resize(density.size());PerformanceTimer timer("gpu_pbe_synchronized");
  accelerator()->pbe_points(density.data(),gradient.data(),density.size(),output.data());return true;
}
void write_performance_report(const std::string& filename){
  std::ofstream f(filename);if(!f)throw std::runtime_error("cannot write performance report");
  const auto counters=merged_counters();
  std::string device;{std::lock_guard lock(counter_mutex);device=device_name;}
  f<<std::setprecision(17)<<"{\n\"schema_version\":\"sparse-dft-performance-1\",\n\"backend\":\""<<mode()<<"\",\n\"device\":\""<<device<<"\",\n\"openmp_max_threads\":";
#ifdef SDFT_HAS_OPENMP
  f<<omp_get_max_threads();
#else
  f<<1;
#endif
  f<<",\n\"timings_overlap\":true,\n\"counters\":{";bool comma=false;
  for(const auto& [key,c]:counters){f<<(comma?",":"")<<"\n\""<<key<<"\":{\"seconds\":"<<c.seconds<<",\"calls\":"<<c.calls<<"}";comma=true;}
  f<<"\n}}\n";
  if(!f)throw std::runtime_error("performance report write failed");
}
#ifndef SDFT_HAS_MPS
std::unique_ptr<Accelerator> make_metal_accelerator(){throw std::runtime_error("native Metal backend was not compiled; configure SDFT_ENABLE_MPS=ON on Apple");}
#endif
#ifndef SDFT_HAS_CUDA
std::unique_ptr<Accelerator> make_cuda_accelerator(){throw std::runtime_error("CUDA backend was not compiled; configure SDFT_ENABLE_CUDA=ON with a CUDA toolkit");}
#endif
}
