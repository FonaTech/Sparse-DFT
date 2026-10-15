#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "compute_backend.hpp"
#include "thread_budget.hpp"
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
// Native Metal backend. The two-float (hi+lo FP32) GEMM, the FFT and the PBE
// kernels live in metal_compensated.hpp, metal_fft.hpp and metal_pbe.hpp; every
// GPU search result is certified by the FP64 CPU operator before acceptance.
#include "metal_compensated.hpp"
#include "metal_fft.hpp"
#include "metal_pbe.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <map>
#include <tuple>
namespace sdft {
namespace {
struct PairComplex {float rh,rl,ih,il;};
struct Shape {std::uint32_t m,n,k,sa,sb,sc,is_complex;};
PairComplex pack(Complex64 v){
  const float re=float(v.real()),im=float(v.imag());
  if(!std::isfinite(re)||!std::isfinite(im))throw std::overflow_error("Metal two-float exponent range exceeded; FP64 required");
  return {re,float(v.real()-double(re)),im,float(v.imag()-double(im))};
}
class MetalAccelerator final:public Accelerator {
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> pipeline_;
  id<MTLComputePipelineState> pack_pipeline_;
  id<MTLComputePipelineState> pbe_pipeline_;
  id<MTLBuffer> pbe_input_,pbe_output_;
  id<MTLBuffer> a_,b_,c_;
  id<MTLBuffer> source_a_,source_b_;
  struct ResidentLeft {std::uint64_t identity{};std::size_t rows{},cols{},ld{};id<MTLBuffer> values;};
  ResidentLeft resident_left_[2];
  struct PawResident {std::uint64_t identity{},projector_identity{};std::size_t npw{},np{};
    id<MTLBuffer> b,bh,d,q,input,source,amplitudes,dp,qp,h,s;};
  PawResident paw_;
  id<MTLComputePipelineState> fft_scatter_,fft_permute_,fft_butterfly_,fft_first_,fft_multiply_,fft_gather_;
  id<MTLBuffer> fft_input_,fft_work0_,fft_work1_,fft_output_,fft_lookup_,fft_indices_,fft_potential_;
  struct FftStage {unsigned radix,previous;id<MTLBuffer> twiddles;};
  struct FftAxis {unsigned length,stride;id<MTLBuffer> permutation;std::vector<FftStage> stages;};
  std::map<std::array<int,3>,std::array<FftAxis,3>> fft_plans_;
  std::vector<std::size_t> fft_cached_indices_;
  std::vector<double> fft_cached_potential_;
  std::array<int,3> fft_cached_shape_{};
  std::uint64_t fft_cached_operator_identity_{};
  bool register_tile_{};
  bool gpu_pack_{};
  bool fft_fuse_permute_{};
  void reserve(id<MTLBuffer> __strong& buffer,std::size_t bytes){
    if(!buffer||buffer.length<bytes){buffer=[device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];record_performance("gpu_buffer_allocation",0);}
    if(!buffer)throw std::bad_alloc();
  }
 public:
  MetalAccelerator(){@autoreleasepool {
    device_=MTLCreateSystemDefaultDevice();
    if(!device_)throw std::runtime_error("Metal device unavailable to process (check host GPU permissions)");
    queue_=[device_ newCommandQueue];
    NSError* error=nil;MTLCompileOptions* opts=[MTLCompileOptions new];
    if(@available(macOS 15.0,*))opts.mathMode=MTLMathModeSafe;else opts.fastMathEnabled=NO;
    NSString* source=[NSString stringWithFormat:@"%s\n%s\n%s",compensatedSource,metalFftSource,metalPbeSource];
    id<MTLLibrary> lib=[device_ newLibraryWithSource:source options:opts error:&error];
    if(!lib)throw std::runtime_error(std::string("Metal shader compile: ")+[[error description] UTF8String]);
    const char* precision=std::getenv("SDFT_METAL_PRECISION");
    if(precision&&std::string(precision)!="fp32x2"&&std::string(precision)!="fp32")throw std::invalid_argument("SDFT_METAL_PRECISION must be fp32x2 or fp32");
    const char* tile=std::getenv("SDFT_METAL_TILE");
    if(tile&&std::string(tile)!="16"&&std::string(tile)!="16x2")throw std::invalid_argument("SDFT_METAL_TILE must be 16 or 16x2");
    // 2x2 register tile is the default (Si64 replay: gpu_kernel 14.5 s ->
    // 13.3 s, identical certificate); SDFT_METAL_TILE=16 keeps the old one.
    register_tile_=(tile?std::string(tile)=="16x2":!deterministic_mode())&&!(precision&&std::string(precision)=="fp32");
    pipeline_=[device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:precision&&std::string(precision)=="fp32"?@"fp32_gemm":register_tile_?@"compensated_gemm_2x2":@"compensated_gemm"] error:&error];
    if(!pipeline_||!queue_)throw std::runtime_error("Metal compute pipeline creation failed");
    const char* packing=std::getenv("SDFT_METAL_PACK");
    if(packing&&std::string(packing)!="cpu"&&std::string(packing)!="gpu")throw std::invalid_argument("SDFT_METAL_PACK must be cpu or gpu");
    // GPU-side IEEE->double-float conversion is the default (Si64 Davidson
    // replay: pack+submit 10.0 s -> 5.5 s, eigenvalues within 5e-15 Ha);
    // SDFT_METAL_PACK=cpu or SDFT_DETERMINISTIC=1 keeps the CPU packer.
    gpu_pack_=packing?std::string(packing)=="gpu":!deterministic_mode();
    const char* fuse=std::getenv("SDFT_METAL_FFT_FUSE_PERMUTE");
    if(fuse&&std::string(fuse)!="0"&&std::string(fuse)!="1")throw std::invalid_argument("SDFT_METAL_FFT_FUSE_PERMUTE must be 0 or 1");
    fft_fuse_permute_=!fuse||std::string(fuse)=="1";
    pack_pipeline_=[device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ieee64_pack"] error:&error];
    if(!pack_pipeline_)throw std::runtime_error("Metal IEEE storage conversion pipeline creation failed");
    pbe_pipeline_=[device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pbe_points"] error:&error];
    if(!pbe_pipeline_)throw std::runtime_error("Metal PBE pipeline creation failed");
    const auto fft_pipeline=[&](NSString* name){
      auto result=[device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];
      if(!result)throw std::runtime_error("Metal FFT pipeline creation failed");return result;
    };
    fft_scatter_=fft_pipeline(@"fft_scatter");fft_permute_=fft_pipeline(@"fft_permute");
    fft_butterfly_=fft_pipeline(@"fft_butterfly");fft_first_=fft_pipeline(@"fft_permute_first_stage");
    fft_multiply_=fft_pipeline(@"fft_multiply_potential");fft_gather_=fft_pipeline(@"fft_gather");
  }}
  std::string device()const override {return [device_.name UTF8String];}
  void pbe_points(const double* density,const double* gradient,std::size_t count,PbeXcPoint* output)override {@autoreleasepool {
    if(count>std::numeric_limits<std::uint32_t>::max()||count>device_.maxBufferLength/(6*2*sizeof(float)))
      throw std::overflow_error("Metal PBE points exceed device/index limits");
    reserve(pbe_input_,count*2*sizeof(double));reserve(pbe_output_,count*6*2*sizeof(float));
    {PerformanceTimer timer("gpu_pbe_source_copy");
      std::memcpy(pbe_input_.contents,density,count*sizeof(double));
      std::memcpy(static_cast<double*>(pbe_input_.contents)+count,gradient,count*sizeof(double));}
    constexpr double pi=3.1415926535897932384626433832795;
    const double constants[]{1./3.,std::cbrt(3./(4*pi)),std::cbrt(3*pi*pi),.2195149727645171,.804,
      -.75*std::cbrt(3./pi),.0310907,.21370,7.5957,3.5876,1.6382,.49294,4./pi,.06672455060314922,
      .031090690869654895,std::log(2.)};
    struct Pair {float hi,lo;};Pair c[16];
    for(int i=0;i<16;++i){auto pair=pack({constants[i],0});c[i]={pair.rh,pair.rl};}
    const auto size=std::uint32_t(count);
    id<MTLCommandBuffer> cmd=[queue_ commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:pbe_pipeline_];[enc setBuffer:pbe_input_ offset:0 atIndex:0];[enc setBuffer:pbe_output_ offset:0 atIndex:1];
    [enc setBytes:c length:sizeof(c) atIndex:2];[enc setBytes:&size length:sizeof(size) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,pbe_pipeline_.maxTotalThreadsPerThreadgroup),1,1)];[enc endEncoding];
    {PerformanceTimer timer("gpu_pbe_submit_wait");[cmd commit];[cmd waitUntilCompleted];}
    if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal PBE command failed");
    record_performance("gpu_pbe_kernel",cmd.GPUEndTime-cmd.GPUStartTime);
    {PerformanceTimer timer("gpu_pbe_unpack");const auto* values=static_cast<const Pair*>(pbe_output_.contents);
      for(std::size_t i=0;i<count;++i){double fields[6];
        for(int j=0;j<6;++j){const auto p=values[i*6+j];fields[j]=double(p.hi)+p.lo;
          if(!std::isfinite(fields[j]))throw std::runtime_error("Metal PBE nonfinite response: point="+std::to_string(i)+" field="+std::to_string(j)+" density="+std::to_string(density[i]));}
        auto& point=output[i];point.energy_density=fields[0];point.energy_per_particle=fields[0]/density[i];
        point.exchange_energy_density=fields[1];point.correlation_energy_density=fields[2];
        point.exchange_potential=fields[3];point.correlation_potential=fields[4];point.gradient_squared_coefficient=fields[5];}
    }
  }}
  void local_hamiltonian(const std::array<int,3>& shape,const std::vector<std::size_t>& indices,
      const std::vector<double>& kinetic,const std::vector<double>& potential,
      const Complex64* input,std::size_t columns,Complex64* output,std::uint64_t operator_identity)override {@autoreleasepool {
    const std::size_t count=potential.size(),npw=indices.size(),total=count*columns;
    if(columns>std::numeric_limits<std::uint32_t>::max()/count||npw>std::size_t(std::numeric_limits<int>::max())||
       total>device_.maxBufferLength/sizeof(PairComplex))throw std::overflow_error("Metal batched FFT exceeds device/index limits");
    const bool immutable_hit=operator_identity&&operator_identity==fft_cached_operator_identity_;
    if(immutable_hit&&(shape!=fft_cached_shape_||indices.size()!=fft_cached_indices_.size()||
        potential.size()!=fft_cached_potential_.size()))
      throw std::invalid_argument("immutable FFT operator token changed dimensions");
    // Invalidate before touching static buffers, including a failed upload.
    if(!immutable_hit)fft_cached_operator_identity_=0;
    auto plan=fft_plans_.find(shape);
    if(plan==fft_plans_.end()){
      if(fft_plans_.size()>=4)fft_plans_.erase(fft_plans_.begin());
      std::array<FftAxis,3> axes;
      for(int axis=0;axis<3;++axis){
        auto& data=axes[axis];data.length=shape[axis];data.stride=1;
        for(int j=axis+1;j<3;++j)data.stride*=shape[j];
        unsigned remaining=data.length,previous=1;std::vector<unsigned> radices;
        while(remaining>1){unsigned radix=0;for(unsigned r:{2u,3u,5u,7u})if(remaining%r==0){radix=r;break;}
          if(!radix)throw std::invalid_argument("Metal FFT requires smooth 2/3/5/7 grid lengths; select SDFT_FFT_GRID=fast");
          radices.push_back(radix);remaining/=radix;
          const unsigned stage=previous*radix;std::vector<PairComplex> twiddle(stage);
          for(unsigned t=0;t<stage;++t){const double angle=-6.283185307179586476925286766559*double(t)/stage;twiddle[t]=pack({std::cos(angle),std::sin(angle)});}
          id<MTLBuffer> buffer=[device_ newBufferWithBytes:twiddle.data() length:twiddle.size()*sizeof(PairComplex) options:MTLResourceStorageModeShared];
          if(!buffer)throw std::bad_alloc();data.stages.push_back({radix,previous,buffer});previous=stage;
        }
        std::vector<unsigned> permutation(data.length);
        for(unsigned i=0;i<data.length;++i){unsigned value=i,multiplier=data.length,target=0;
          for(auto radix:radices){multiplier/=radix;target+=(value%radix)*multiplier;value/=radix;}
          permutation[i]=target;
        }
        data.permutation=[device_ newBufferWithBytes:permutation.data() length:permutation.size()*sizeof(unsigned) options:MTLResourceStorageModeShared];
        if(!data.permutation)throw std::bad_alloc();
      }
      plan=fft_plans_.emplace(shape,std::move(axes)).first;record_performance("gpu_fft_plan_cache_miss",0);
    }else record_performance("gpu_fft_plan_cache_hit",0);
    reserve(fft_input_,npw*columns*sizeof(Complex64));reserve(fft_work0_,total*sizeof(PairComplex));
    reserve(fft_work1_,total*sizeof(PairComplex));reserve(fft_output_,npw*columns*sizeof(PairComplex));
    reserve(fft_lookup_,count*sizeof(int));reserve(fft_indices_,npw*sizeof(unsigned));reserve(fft_potential_,count*2*sizeof(float));
    // Only immutable local grids and indices carry a nonzero lifetime token.
    // Zero uses content checks; dynamic wavefunctions are always uploaded.
    if(!immutable_hit&&(indices!=fft_cached_indices_||shape!=fft_cached_shape_)){
      auto* lookup=static_cast<int*>(fft_lookup_.contents);std::fill_n(lookup,count,-1);auto* packed=static_cast<unsigned*>(fft_indices_.contents);
      for(std::size_t g=0;g<npw;++g){lookup[indices[g]]=int(g);packed[g]=unsigned(indices[g]);}
      fft_cached_indices_=indices;fft_cached_shape_=shape;
    }
    if(!immutable_hit&&potential!=fft_cached_potential_){
      auto* values=static_cast<float*>(fft_potential_.contents);
      for(std::size_t i=0;i<count;++i){const auto p=pack({potential[i],0});values[2*i]=p.rh;values[2*i+1]=p.rl;}
      fft_cached_potential_=potential;
    }
    record_performance(immutable_hit?"gpu_fft_immutable_operator_hit":"gpu_fft_immutable_operator_miss",0);
    {PerformanceTimer timer("gpu_fft_source_copy");std::memcpy(fft_input_.contents,input,npw*columns*sizeof(Complex64));}
    struct FFTShape {std::uint32_t count,columns,length,stride,radix,previous,backward,npw;};
    FFTShape info{unsigned(count),unsigned(columns),0,0,0,0,0,unsigned(npw)};
    id<MTLCommandBuffer> cmd=[queue_ commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    const auto dispatch=[&](id<MTLComputePipelineState> pipeline,std::size_t elements){
      [enc setComputePipelineState:pipeline];[enc setBytes:&info length:sizeof(info) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(elements,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];record_performance("gpu_fft_kernel_dispatch",0);
    };
    [enc setBuffer:fft_input_ offset:0 atIndex:0];[enc setBuffer:fft_lookup_ offset:0 atIndex:1];[enc setBuffer:fft_work0_ offset:0 atIndex:2];
    dispatch(fft_scatter_,total);id<MTLBuffer> current=fft_work0_,next=fft_work1_;
    const auto transform=[&](bool backward){
      info.backward=backward;
      for(const auto& axis:plan->second){
        info.length=axis.length;info.stride=axis.stride;
        std::size_t first=0;
        if(fft_fuse_permute_&&!axis.stages.empty()){
          const auto& stage=axis.stages.front();info.radix=stage.radix;info.previous=stage.previous;
          [enc setBuffer:current offset:0 atIndex:0];[enc setBuffer:next offset:0 atIndex:1];
          [enc setBuffer:stage.twiddles offset:0 atIndex:2];[enc setBuffer:axis.permutation offset:0 atIndex:4];
          dispatch(fft_first_,total);std::swap(current,next);first=1;
          record_performance("gpu_fft_permutation_pass_saved",0);
        }else{
          [enc setBuffer:current offset:0 atIndex:0];[enc setBuffer:next offset:0 atIndex:1];[enc setBuffer:axis.permutation offset:0 atIndex:2];dispatch(fft_permute_,total);std::swap(current,next);
        }
        for(std::size_t i=first;i<axis.stages.size();++i){const auto& stage=axis.stages[i];info.radix=stage.radix;info.previous=stage.previous;
          [enc setBuffer:current offset:0 atIndex:0];[enc setBuffer:next offset:0 atIndex:1];[enc setBuffer:stage.twiddles offset:0 atIndex:2];dispatch(fft_butterfly_,total);std::swap(current,next);}
      }
    };
    transform(true);
    [enc setBuffer:current offset:0 atIndex:0];[enc setBuffer:fft_potential_ offset:0 atIndex:1];dispatch(fft_multiply_,total);
    transform(false);
    [enc setBuffer:current offset:0 atIndex:0];[enc setBuffer:fft_indices_ offset:0 atIndex:1];[enc setBuffer:fft_output_ offset:0 atIndex:2];dispatch(fft_gather_,npw*columns);[enc endEncoding];
    {PerformanceTimer timer("gpu_fft_submit_wait");[cmd commit];[cmd waitUntilCompleted];}
    if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal batched FFT command failed");
    fft_cached_operator_identity_=operator_identity;
    record_performance("gpu_fft_kernel",cmd.GPUEndTime-cmd.GPUStartTime);
    {PerformanceTimer timer("gpu_fft_unpack");const auto* values=static_cast<const PairComplex*>(fft_output_.contents);
      for(std::size_t i=0;i<npw*columns;++i){const auto p=values[i];const Complex64 local((double(p.rh)+p.rl)/count,(double(p.ih)+p.il)/count);
        if(!std::isfinite(local.real())||!std::isfinite(local.imag()))throw std::runtime_error("nonfinite Metal FFT result");
        output[i]=kinetic[i%npw]*input[i]+local;}
    }
  }}
  bool paw_pair(std::uint64_t identity,std::size_t npw,std::size_t np,std::size_t columns,
      const Complex64* b,const Complex64* d,const Complex64* q,const Complex64* input,
      Complex64* h,Complex64* s,std::uint64_t projector_identity)override {@autoreleasepool {
    if(!identity||!npw||!np||!columns||std::max({npw,np,columns})>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid Metal PAW chain dimensions");
    for(auto elements:{npw*np,np*np,npw*columns,np*columns})
      if(elements>device_.maxBufferLength/sizeof(PairComplex))throw std::overflow_error("Metal PAW chain buffer limit");
    const bool reuse_b=paw_.projector_identity==projector_identity&&paw_.npw==npw&&paw_.np==np;
    const bool reuse_d=reuse_b&&paw_.identity==identity;
    if(!reuse_b){
      paw_.projector_identity=0;reserve(paw_.b,np*npw*sizeof(PairComplex));reserve(paw_.bh,npw*np*sizeof(PairComplex));
      PerformanceTimer timer("gpu_paw_chain_projector_pack");
      auto* pb=static_cast<PairComplex*>(paw_.b.contents);auto* ph=static_cast<PairComplex*>(paw_.bh.contents);
      for(std::size_t i=0;i<np;++i){
        for(std::size_t j=0;j<npw;++j){pb[i*npw+j]=pack(b[i+j*np]);ph[j*np+i]=pack(std::conj(b[i+j*np]));}
      }
      record_performance("gpu_paw_chain_static_miss",0);
    }else record_performance("gpu_paw_chain_static_hit",0);
    if(!reuse_d){
      // Projector identity guarantees B only. The operator token covers D
      // AND Q, including a new overlap at the same geometry. Upload these
      // small matrices together; never reuse Q merely because B survived.
      paw_.identity=0;reserve(paw_.d,np*np*sizeof(PairComplex));reserve(paw_.q,np*np*sizeof(PairComplex));
      PerformanceTimer timer("gpu_paw_chain_density_pack");auto* pd=static_cast<PairComplex*>(paw_.d.contents);
      auto* pq=static_cast<PairComplex*>(paw_.q.contents);
      for(std::size_t i=0;i<np;++i)for(std::size_t j=0;j<np;++j){pd[i*np+j]=pack(d[i+j*np]);pq[i*np+j]=pack(q[i+j*np]);}
      record_performance("gpu_paw_chain_density_miss",0);
    }else record_performance("gpu_paw_chain_density_hit",0);
    reserve(paw_.input,npw*columns*sizeof(PairComplex));reserve(paw_.source,npw*columns*sizeof(Complex64));
    reserve(paw_.amplitudes,np*columns*sizeof(PairComplex));reserve(paw_.dp,np*columns*sizeof(PairComplex));
    reserve(paw_.qp,np*columns*sizeof(PairComplex));reserve(paw_.h,npw*columns*sizeof(PairComplex));reserve(paw_.s,npw*columns*sizeof(PairComplex));
    {PerformanceTimer timer("gpu_paw_chain_input_copy");std::memcpy(paw_.source.contents,input,npw*columns*sizeof(Complex64));}
    id<MTLCommandBuffer> cmd=[queue_ commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    struct PackShape{std::uint32_t rows,cols,ld,adjoint;};const PackShape shape{std::uint32_t(npw),std::uint32_t(columns),std::uint32_t(npw),0};
    [enc setComputePipelineState:pack_pipeline_];[enc setBuffer:paw_.source offset:0 atIndex:0];[enc setBuffer:paw_.input offset:0 atIndex:1];[enc setBytes:&shape length:sizeof(shape) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(npw*columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    const auto dispatch=[&](id<MTLBuffer> a,id<MTLBuffer> bb,id<MTLBuffer> c,std::size_t m,std::size_t k){
      const Shape shape{std::uint32_t(m),std::uint32_t(columns),std::uint32_t(k),std::uint32_t(k),std::uint32_t(columns),std::uint32_t(columns),1};
      [enc setComputePipelineState:pipeline_];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:bb offset:0 atIndex:1];[enc setBuffer:c offset:0 atIndex:2];[enc setBytes:&shape length:sizeof(shape) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((columns+15)/16,(m+15)/16,1) threadsPerThreadgroup:MTLSizeMake(register_tile_?8:16,register_tile_?8:16,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    dispatch(paw_.b,paw_.input,paw_.amplitudes,np,npw);
    dispatch(paw_.d,paw_.amplitudes,paw_.dp,np,np);dispatch(paw_.q,paw_.amplitudes,paw_.qp,np,np);
    dispatch(paw_.bh,paw_.dp,paw_.h,npw,np);dispatch(paw_.bh,paw_.qp,paw_.s,npw,np);[enc endEncoding];
    {PerformanceTimer timer("gpu_paw_chain_submit_wait");[cmd commit];[cmd waitUntilCompleted];}
    if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal PAW chain command failed");
    paw_.identity=identity;paw_.projector_identity=projector_identity;paw_.npw=npw;paw_.np=np;
    const auto* ph=static_cast<const PairComplex*>(paw_.h.contents);const auto* ps=static_cast<const PairComplex*>(paw_.s.contents);
    {PerformanceTimer timer("gpu_paw_chain_unpack");
      for(std::size_t i=0;i<npw;++i)for(std::size_t j=0;j<columns;++j){
        const auto a=ph[i*columns+j],b=ps[i*columns+j];
        const Complex64 hv(double(a.rh)+a.rl,double(a.ih)+a.il),sv(double(b.rh)+b.rl,double(b.ih)+b.il);
        if(!std::isfinite(hv.real())||!std::isfinite(hv.imag())||!std::isfinite(sv.real())||!std::isfinite(sv.imag()))throw std::runtime_error("nonfinite Metal PAW chain result");
        h[i+j*npw]=hv;s[i+j*npw]=sv;
      }
    }
    record_performance("gpu_paw_chain_dispatch",0);record_performance("gpu_paw_chain_kernels",0,5);
    record_performance("gpu_paw_chain_saved_sync",0,4);record_performance("gpu_paw_chain_saved_intermediate_values",0,3*np*columns);
    return true;
  }}
  void gemm(MatrixOp ao,MatrixOp bo,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,
      Complex64* c,std::size_t ldc,bool add) override {
    gemm_impl(0,ao,bo,m,n,k,a,lda,b,ldb,c,ldc,add);
  }
  void gemm_triple_right(std::size_t m,std::size_t n,std::size_t k,
      const std::array<const Complex64*,3>& a,std::size_t lda,
      const Complex64* b,std::size_t ldb,const std::array<Complex64*,3>& c,std::size_t ldc)override {@autoreleasepool {
    const auto max_values=device_.maxBufferLength/sizeof(PairComplex);
    if(std::max({m,n,k,lda,ldb})>std::numeric_limits<std::uint32_t>::max()||
        m>max_values/3/k||m>max_values/3/n||k>max_values/n||
        m>std::numeric_limits<std::uint32_t>::max()/k||
        m>std::numeric_limits<std::uint32_t>::max()/n||k>std::numeric_limits<std::uint32_t>::max()/n)
      throw std::overflow_error("joint Metal Ritz rotation exceeds buffer/index limits");
    const std::size_t aspan=(k-1)*lda+m,bspan=(n-1)*ldb+k;
    if(aspan>device_.maxBufferLength/sizeof(Complex64)/3||
        bspan>device_.maxBufferLength/sizeof(Complex64))
      throw std::overflow_error("joint Metal Ritz source span exceeds buffer limits");
    reserve(a_,3*m*k*sizeof(PairComplex));reserve(b_,k*n*sizeof(PairComplex));
    reserve(c_,3*m*n*sizeof(PairComplex));
    if(gpu_pack_) {
      PerformanceTimer timer("gpu_triple_source_copy");
      reserve(source_a_,3*aspan*sizeof(Complex64));reserve(source_b_,bspan*sizeof(Complex64));
      for(std::size_t image=0;image<3;++image)
        std::memcpy(static_cast<char*>(source_a_.contents)+image*aspan*sizeof(Complex64),a[image],aspan*sizeof(Complex64));
      std::memcpy(source_b_.contents,b,bspan*sizeof(Complex64));
    }else {
      PerformanceTimer timer("gpu_triple_cpu_pack");
      auto* left=static_cast<PairComplex*>(a_.contents);auto* right=static_cast<PairComplex*>(b_.contents);
      for(std::size_t image=0;image<3;++image)for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<k;++j)
        left[image*m*k+i*k+j]=pack(a[image][i+j*lda]);
      for(std::size_t i=0;i<k;++i)for(std::size_t j=0;j<n;++j)right[i*n+j]=pack(b[i+j*ldb]);
    }
    id<MTLCommandBuffer> cmd=[queue_ commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    if(gpu_pack_) {
      struct PackShape{std::uint32_t rows,cols,ld,adjoint;};
      const PackShape ap{std::uint32_t(m),std::uint32_t(k),std::uint32_t(lda),0},bp{std::uint32_t(k),std::uint32_t(n),std::uint32_t(ldb),0};
      [enc setComputePipelineState:pack_pipeline_];
      for(std::size_t image=0;image<3;++image) {
        [enc setBuffer:source_a_ offset:image*aspan*sizeof(Complex64) atIndex:0];
        [enc setBuffer:a_ offset:image*m*k*sizeof(PairComplex) atIndex:1];
        [enc setBytes:&ap length:sizeof(ap) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(m*k,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      }
      [enc setBuffer:source_b_ offset:0 atIndex:0];[enc setBuffer:b_ offset:0 atIndex:1];
      [enc setBytes:&bp length:sizeof(bp) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(k*n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      record_performance("gpu_triple_right_pack_saved",0,2);
    }
    const Shape shape{std::uint32_t(m),std::uint32_t(n),std::uint32_t(k),std::uint32_t(k),std::uint32_t(n),std::uint32_t(n),1};
    [enc setComputePipelineState:pipeline_];[enc setBuffer:b_ offset:0 atIndex:1];
    [enc setBytes:&shape length:sizeof(shape) atIndex:3];
    for(std::size_t image=0;image<3;++image) {
      [enc setBuffer:a_ offset:image*m*k*sizeof(PairComplex) atIndex:0];
      [enc setBuffer:c_ offset:image*m*n*sizeof(PairComplex) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake((n+15)/16,(m+15)/16,1)
          threadsPerThreadgroup:MTLSizeMake(register_tile_?8:16,register_tile_?8:16,1)];
    }
    [enc endEncoding];
    {PerformanceTimer timer("gpu_triple_submit_wait");[cmd commit];[cmd waitUntilCompleted];}
    if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("joint Metal Ritz command failed");
    record_performance("gpu_kernel",cmd.GPUEndTime-cmd.GPUStartTime);
    {PerformanceTimer timer("gpu_triple_unpack");
      const auto* packed=static_cast<const PairComplex*>(c_.contents);bool nonfinite=false;
      const std::size_t row_tiles=(m+31)/32,column_tiles=(n+31)/32;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for collapse(3) schedule(static) reduction(||:nonfinite) if(!omp_in_parallel() && 3*m*n>=32768)
#endif
      for(std::size_t image=0;image<3;++image)for(std::size_t ti=0;ti<row_tiles;++ti)for(std::size_t tj=0;tj<column_tiles;++tj)
        for(std::size_t i=ti*32;i<std::min(m,(ti+1)*32);++i)for(std::size_t j=tj*32;j<std::min(n,(tj+1)*32);++j) {
          const auto v=packed[image*m*n+i*n+j];const Complex64 value(double(v.rh)+v.rl,double(v.ih)+v.il);
          nonfinite|=!std::isfinite(value.real())||!std::isfinite(value.imag());c[image][i+j*ldc]=value;
        }
      if(nonfinite)throw std::runtime_error("nonfinite joint Metal Ritz result");
    }
    record_performance("gpu_triple_rotation_products",0,3);record_performance("gpu_triple_sync_saved",0,2);
  }}
  void gemm_immutable_left(std::uint64_t identity,MatrixOp ao,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add) override {
    gemm_impl(identity,ao,MatrixOp::Normal,m,n,k,a,lda,b,ldb,c,ldc,add);
  }
  void gemm_impl(std::uint64_t identity,MatrixOp ao,MatrixOp bo,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add) {@autoreleasepool {
    if(std::max({m,n,k})>std::numeric_limits<std::uint32_t>::max()||
       m>device_.maxBufferLength/sizeof(PairComplex)/k||k>device_.maxBufferLength/sizeof(PairComplex)/n||
       m>device_.maxBufferLength/sizeof(PairComplex)/n)throw std::overflow_error("Metal GEMM exceeds device buffer limits");
    auto& resident=resident_left_[ao==MatrixOp::Normal?0:1];
    const bool reuse=identity&&resident.identity==identity&&resident.rows==m&&resident.cols==k&&resident.ld==lda;
    id<MTLBuffer> left;
    if(identity){if(!reuse){resident.identity=0;reserve(resident.values,m*k*sizeof(PairComplex));}left=resident.values;}
    else {reserve(a_,m*k*sizeof(PairComplex));left=a_;}
    reserve(b_,k*n*sizeof(PairComplex));reserve(c_,m*n*sizeof(PairComplex));
    if(reuse){record_performance("gpu_resident_left_hit",0);record_performance("gpu_resident_left_saved_values",0,m*k);}
    else if(identity)record_performance("gpu_resident_left_miss",0);
    if(gpu_pack_){
      PerformanceTimer timer("gpu_source_copy");
      const auto ac=ao==MatrixOp::Normal?k:m,bc=bo==MatrixOp::Normal?n:k;
      const auto ar=ao==MatrixOp::Normal?m:k,br=bo==MatrixOp::Normal?k:n;
      const auto asize=(ac-1)*lda+ar,bsize=(bc-1)*ldb+br;
      if(asize>device_.maxBufferLength/sizeof(Complex64)||bsize>device_.maxBufferLength/sizeof(Complex64))throw std::overflow_error("Metal packed input exceeds buffer limits");
      if(!reuse){reserve(source_a_,asize*sizeof(Complex64));std::memcpy(source_a_.contents,a,asize*sizeof(Complex64));}
      reserve(source_b_,bsize*sizeof(Complex64));std::memcpy(source_b_.contents,b,bsize*sizeof(Complex64));
    }else {PerformanceTimer timer("gpu_pack");
      auto* pa=(PairComplex*)left.contents;auto* pb=(PairComplex*)b_.contents;
      for(std::size_t i=0;i<(reuse?0:m);++i)for(std::size_t j=0;j<k;++j)
        pa[i*k+j]=pack(ao==MatrixOp::Normal?a[i+j*lda]:std::conj(a[j+i*lda]));
      for(std::size_t i=0;i<k;++i)for(std::size_t j=0;j<n;++j)
        pb[i*n+j]=pack(bo==MatrixOp::Normal?b[i+j*ldb]:std::conj(b[j+i*ldb]));
    }
    const Shape shape{std::uint32_t(m),std::uint32_t(n),std::uint32_t(k),std::uint32_t(k),std::uint32_t(n),std::uint32_t(n),1};
    id<MTLCommandBuffer> cmd=[queue_ commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    if(gpu_pack_){
      struct PackShape{std::uint32_t rows,cols,ld,adjoint;};
      const PackShape ap{std::uint32_t(m),std::uint32_t(k),std::uint32_t(lda),ao==MatrixOp::Adjoint?1u:0u},
          bp{std::uint32_t(k),std::uint32_t(n),std::uint32_t(ldb),bo==MatrixOp::Adjoint?1u:0u};
      [enc setComputePipelineState:pack_pipeline_];
      if(!reuse){
        [enc setBuffer:source_a_ offset:0 atIndex:0];[enc setBuffer:left offset:0 atIndex:1];
        [enc setBytes:&ap length:sizeof(ap) atIndex:2];[enc dispatchThreads:MTLSizeMake(m*k,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [enc setBuffer:source_b_ offset:0 atIndex:0];[enc setBuffer:b_ offset:0 atIndex:1];[enc setBytes:&bp length:sizeof(bp) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(k*n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];record_performance("gpu_ieee64_pack_dispatch",0,reuse?1:2);
    }
    [enc setComputePipelineState:pipeline_];[enc setBuffer:left offset:0 atIndex:0];[enc setBuffer:b_ offset:0 atIndex:1];
    [enc setBuffer:c_ offset:0 atIndex:2];[enc setBytes:&shape length:sizeof(shape) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n+15)/16,(m+15)/16,1) threadsPerThreadgroup:MTLSizeMake(register_tile_?8:16,register_tile_?8:16,1)];[enc endEncoding];
    {PerformanceTimer timer("gpu_submit_wait");[cmd commit];[cmd waitUntilCompleted];}
    if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(std::string("Metal command failed: ")+[[cmd.error description] UTF8String]);
    if(identity){resident.identity=identity;resident.rows=m;resident.cols=k;resident.ld=lda;}
    record_performance("gpu_kernel",cmd.GPUEndTime-cmd.GPUStartTime);
    {PerformanceTimer timer("gpu_unpack");
      const auto* pc=(const PairComplex*)c_.contents;
      bool nonfinite=false;
      // Tiled transpose of Metal's row-major compensated output to BLAS
      // column-major FP64. Output tiles do not overlap; no cross-thread sums
      // or Objective-C command queues are used inside the CPU team.
      const std::size_t row_tiles=(m+31)/32,column_tiles=(n+31)/32;
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for collapse(2) schedule(static) reduction(||:nonfinite) if(!omp_in_parallel() && m*n>=32768)
#endif
      for(std::size_t ti=0;ti<row_tiles;++ti)for(std::size_t tj=0;tj<column_tiles;++tj) {
        for(std::size_t i=ti*32;i<std::min(m,(ti+1)*32);++i)
          for(std::size_t j=tj*32;j<std::min(n,(tj+1)*32);++j) {
            const auto v=pc[i*n+j];
            const Complex64 result(double(v.rh)+double(v.rl),double(v.ih)+double(v.il));
            nonfinite |= !std::isfinite(result.real())||!std::isfinite(result.imag());
            if(add)c[i+j*ldc]+=result;else c[i+j*ldc]=result;
          }
      }
      if(nonfinite)throw std::runtime_error("nonfinite Metal GEMM output");
    }
  }}
};
}
std::unique_ptr<Accelerator> make_metal_accelerator(){PerformanceTimer timer("gpu_initialization");return std::make_unique<MetalAccelerator>();}
}
