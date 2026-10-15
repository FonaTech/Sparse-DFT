#include "compute_backend.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cuda_pbe.cuh"
#include <cufft.h>
#include <limits>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
namespace sdft {
namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(code));}
void checked(cublasStatus_t code){if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS error "+std::to_string(int(code)));}
void checked(cufftResult code){if(code!=CUFFT_SUCCESS)throw std::runtime_error("cuFFT error "+std::to_string(int(code)));}
__global__ void scatter_fft(const cuDoubleComplex* input,const std::size_t* indices,
    cuDoubleComplex* grid,std::size_t npw,std::size_t count,std::size_t elements){
  const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<elements)grid[(i/npw)*count+indices[i%npw]]=input[i];
}
__global__ void multiply_fft(cuDoubleComplex* grid,const double* potential,std::size_t count,std::size_t elements){
  const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<elements){grid[i].x*=potential[i%count];grid[i].y*=potential[i%count];}
}
__global__ void gather_fft(const cuDoubleComplex* input,const cuDoubleComplex* grid,
    const std::size_t* indices,const double* kinetic,cuDoubleComplex* output,
    std::size_t npw,std::size_t count,std::size_t elements){
  const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<elements){const auto local=grid[(i/npw)*count+indices[i%npw]];
    output[i]=make_cuDoubleComplex(kinetic[i%npw]*input[i].x+local.x/double(count),kinetic[i%npw]*input[i].y+local.y/double(count));}
}
class CudaAccelerator final:public Accelerator {
  cublasHandle_t handle_{};
  cudaStream_t stream_{};
  cuDoubleComplex *a_{},*b_{},*c_{};
  std::size_t na_{},nb_{},nc_{};
  struct ResidentLeft {std::uint64_t identity{};std::size_t rows{},cols{},ld{},capacity{};cuDoubleComplex* values{};};
  ResidentLeft resident_left_[2];
  std::uint64_t paw_identity_{},paw_projector_identity_{};std::size_t paw_npw_{},paw_np_{};
  cuDoubleComplex *paw_b_{},*paw_d_{},*paw_q_{},*paw_input_{},*paw_p_{},*paw_dp_{},*paw_qp_{},*paw_h_{},*paw_s_{};
  std::size_t paw_bc_{},paw_dc_{},paw_qc_{},paw_ic_{},paw_pc_{},paw_dpc_{},paw_qpc_{},paw_hc_{},paw_sc_{};
  std::string device_;
  // Davidson alternates block widths (new directions, restarts, final
  // certification), so one plan slot re-planned on almost every call.
  // Keep a small LRU of plans keyed by (shape, columns).
  struct FftPlan{std::array<int,3> shape{};std::size_t columns{};cufftHandle plan{};std::uint64_t used{};};
  std::array<FftPlan,6> fft_plans_{};std::uint64_t fft_clock_{};
  // Potential/kinetic/indices are immutable for a nonzero operator identity.
  std::uint64_t fft_identity_{};
  cuDoubleComplex *fft_grid_{},*fft_input_{},*fft_output_{};
  double *fft_potential_{},*fft_kinetic_{};std::size_t* fft_indices_{};
  std::size_t grid_capacity_{},input_capacity_{},output_capacity_{},potential_capacity_{},kinetic_capacity_{},indices_capacity_{};
  double* pbe_input_{};PbeXcPoint* pbe_output_{};
  std::size_t pbe_input_capacity_{},pbe_output_capacity_{};
  // Page-lock a host range for one transfer so the async copies really
  // overlap (pageable memory forces a staged, synchronous copy). Ranges
  // below 4 MiB are cheaper to copy pageable than to register.
  struct PinnedRange {
    void* base{};bool registered{};
    PinnedRange(const void* data,std::size_t bytes){
      if(bytes<(std::size_t(4)<<20))return;
      base=const_cast<void*>(data);
      const auto status=cudaHostRegister(base,bytes,cudaHostRegisterDefault);
      if(status==cudaSuccess){registered=true;record_performance("gpu_pinned_registrations",0);}
      else{cudaGetLastError();record_performance("gpu_pinned_registration_failed",0);}  // already pinned or unsupported
    }
    ~PinnedRange(){if(registered)cudaHostUnregister(base);}
  };
  void reserve(cuDoubleComplex*& p,std::size_t& capacity,std::size_t count){
    if(capacity<count){if(p)checked(cudaFree(p));p=nullptr;capacity=0;
      checked(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(cuDoubleComplex)));capacity=count;record_performance("gpu_buffer_allocation",0);}
  }
  template<class T>void reserve_array(T*& p,std::size_t& capacity,std::size_t count){
    if(capacity<count){if(p)checked(cudaFree(p));p=nullptr;capacity=0;
      checked(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));capacity=count;record_performance("gpu_buffer_allocation",0);}
  }
 public:
  CudaAccelerator(){int index=0;checked(cudaGetDevice(&index));cudaDeviceProp props{};checked(cudaGetDeviceProperties(&props,index));device_=props.name;
    checked(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
    try{checked(cublasCreate(&handle_));checked(cublasSetStream(handle_,stream_));
      // No TF32/FP32 conversions: true complex FP64 kernels.
      checked(cublasSetMathMode(handle_,CUBLAS_DEFAULT_MATH));
    }catch(...){cudaStreamDestroy(stream_);throw;}}
  ~CudaAccelerator(){if(a_)cudaFree(a_);if(b_)cudaFree(b_);if(c_)cudaFree(c_);
    for(auto& r:resident_left_)if(r.values)cudaFree(r.values);
    for(auto p:{paw_b_,paw_d_,paw_q_,paw_input_,paw_p_,paw_dp_,paw_qp_,paw_h_,paw_s_})if(p)cudaFree(p);
    for(auto& entry:fft_plans_)if(entry.plan)cufftDestroy(entry.plan);if(fft_grid_)cudaFree(fft_grid_);if(fft_input_)cudaFree(fft_input_);if(fft_output_)cudaFree(fft_output_);
    if(fft_potential_)cudaFree(fft_potential_);if(fft_kinetic_)cudaFree(fft_kinetic_);if(fft_indices_)cudaFree(fft_indices_);
    if(pbe_input_)cudaFree(pbe_input_);if(pbe_output_)cudaFree(pbe_output_);
    if(handle_)cublasDestroy(handle_);if(stream_)cudaStreamDestroy(stream_);}
  std::string device()const override{return device_;}
  bool paw_pair(std::uint64_t identity,std::size_t npw,std::size_t np,std::size_t columns,
      const Complex64* b,const Complex64* d,const Complex64* q,const Complex64* input,
      Complex64* h,Complex64* s,std::uint64_t projector_identity)override {
    if(!identity||!npw||!np||!columns||std::max({npw,np,columns})>std::size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid CUDA PAW chain shape");
    const bool reuse_b=paw_projector_identity_==projector_identity&&paw_npw_==npw&&paw_np_==np;
    const bool reuse_d=reuse_b&&paw_identity_==identity;
    if(!reuse_b){paw_projector_identity_=0;reserve(paw_b_,paw_bc_,np*npw);
      checked(cudaMemcpyAsync(paw_b_,b,np*npw*sizeof(Complex64),cudaMemcpyHostToDevice,stream_));
      record_performance("gpu_paw_chain_static_miss",0);
    }else record_performance("gpu_paw_chain_static_hit",0);
    if(!reuse_d){paw_identity_=0;reserve(paw_d_,paw_dc_,np*np);reserve(paw_q_,paw_qc_,np*np);
      checked(cudaMemcpyAsync(paw_d_,d,np*np*sizeof(Complex64),cudaMemcpyHostToDevice,stream_));
      checked(cudaMemcpyAsync(paw_q_,q,np*np*sizeof(Complex64),cudaMemcpyHostToDevice,stream_));record_performance("gpu_paw_chain_density_miss",0);
    }else record_performance("gpu_paw_chain_density_hit",0);
    reserve(paw_input_,paw_ic_,npw*columns);reserve(paw_p_,paw_pc_,np*columns);reserve(paw_dp_,paw_dpc_,np*columns);
    reserve(paw_qp_,paw_qpc_,np*columns);reserve(paw_h_,paw_hc_,npw*columns);reserve(paw_s_,paw_sc_,npw*columns);
    checked(cudaMemcpyAsync(paw_input_,input,npw*columns*sizeof(Complex64),cudaMemcpyHostToDevice,stream_));
    const cuDoubleComplex one=make_cuDoubleComplex(1,0),zero=make_cuDoubleComplex(0,0);
    const auto multiply=[&](cublasOperation_t op,int m,int k,const cuDoubleComplex* a,int lda,const cuDoubleComplex* b,cuDoubleComplex* c){
      checked(cublasZgemm(handle_,op,CUBLAS_OP_N,m,int(columns),k,&one,a,lda,b,k,&zero,c,m));
    };
    multiply(CUBLAS_OP_N,int(np),int(npw),paw_b_,int(np),paw_input_,paw_p_);
    multiply(CUBLAS_OP_N,int(np),int(np),paw_d_,int(np),paw_p_,paw_dp_);
    multiply(CUBLAS_OP_N,int(np),int(np),paw_q_,int(np),paw_p_,paw_qp_);
    multiply(CUBLAS_OP_C,int(npw),int(np),paw_b_,int(np),paw_dp_,paw_h_);
    multiply(CUBLAS_OP_C,int(npw),int(np),paw_b_,int(np),paw_qp_,paw_s_);
    checked(cudaMemcpyAsync(h,paw_h_,npw*columns*sizeof(Complex64),cudaMemcpyDeviceToHost,stream_));
    checked(cudaMemcpyAsync(s,paw_s_,npw*columns*sizeof(Complex64),cudaMemcpyDeviceToHost,stream_));checked(cudaStreamSynchronize(stream_));
    paw_identity_=identity;paw_projector_identity_=projector_identity;paw_npw_=npw;paw_np_=np;
    record_performance("gpu_paw_chain_dispatch",0);record_performance("gpu_paw_chain_kernels",0,5);
    record_performance("gpu_paw_chain_saved_sync",0,4);record_performance("gpu_paw_chain_saved_intermediate_values",0,3*np*columns);
    return true;
  }
  void pbe_points(const double* density,const double* gradient,std::size_t count,PbeXcPoint* output)override{
    if(count>std::numeric_limits<std::size_t>::max()/(2*sizeof(double)))throw std::overflow_error("CUDA PBE buffer overflow");
    reserve_array(pbe_input_,pbe_input_capacity_,2*count);reserve_array(pbe_output_,pbe_output_capacity_,count);
    checked(cudaMemcpyAsync(pbe_input_,density,count*sizeof(double),cudaMemcpyHostToDevice,stream_));
    checked(cudaMemcpyAsync(pbe_input_+count,gradient,count*sizeof(double),cudaMemcpyHostToDevice,stream_));
    cuda_pbe_points<<<unsigned((count+255)/256),256,0,stream_>>>(pbe_input_,pbe_input_+count,count,pbe_output_);checked(cudaGetLastError());
    checked(cudaMemcpyAsync(output,pbe_output_,count*sizeof(PbeXcPoint),cudaMemcpyDeviceToHost,stream_));checked(cudaStreamSynchronize(stream_));
    for(std::size_t i=0;i<count;++i){const auto& p=output[i];
      if(!std::isfinite(p.energy_density)||!std::isfinite(p.exchange_potential)||!std::isfinite(p.correlation_potential)||!std::isfinite(p.gradient_squared_coefficient))
        throw std::runtime_error("nonfinite CUDA PBE output");}
    record_performance("cuda_fp64_pbe_dispatch",0);
  }
  void local_hamiltonian(const std::array<int,3>& shape,const std::vector<std::size_t>& indices,
      const std::vector<double>& kinetic,const std::vector<double>& potential,
      const Complex64* input,std::size_t columns,Complex64* output,std::uint64_t operator_identity)override{
    const std::size_t count=potential.size(),npw=indices.size(),elements=npw*columns,total=count*columns;
    if(count>std::size_t(std::numeric_limits<int>::max())||columns>std::size_t(std::numeric_limits<int>::max())||
       total/std::max<std::size_t>(1,columns)!=count)throw std::overflow_error("CUDA batched FFT dimensions exceed plan limits");
    FftPlan* slot=nullptr;
    for(auto& entry:fft_plans_)if(entry.plan&&entry.shape==shape&&entry.columns==columns){slot=&entry;break;}
    if(!slot){
      slot=&*std::min_element(fft_plans_.begin(),fft_plans_.end(),[](const FftPlan& a,const FftPlan& b){return a.used<b.used;});
      if(slot->plan)checked(cufftDestroy(slot->plan));slot->plan=0;
      int dims[]{shape[0],shape[1],shape[2]};
      checked(cufftPlanMany(&slot->plan,3,dims,nullptr,1,int(count),nullptr,1,int(count),CUFFT_Z2Z,int(columns)));
      checked(cufftSetStream(slot->plan,stream_));slot->shape=shape;slot->columns=columns;record_performance("gpu_fft_plan_cache_miss",0);
    }else record_performance("gpu_fft_plan_cache_hit",0);
    slot->used=++fft_clock_;
    const cufftHandle plan=slot->plan;
    reserve_array(fft_grid_,grid_capacity_,total);reserve_array(fft_input_,input_capacity_,elements);reserve_array(fft_output_,output_capacity_,elements);
    const bool fresh_operator=!operator_identity||operator_identity!=fft_identity_||potential_capacity_<count||kinetic_capacity_<npw||indices_capacity_<npw;
    reserve_array(fft_potential_,potential_capacity_,count);reserve_array(fft_kinetic_,kinetic_capacity_,npw);reserve_array(fft_indices_,indices_capacity_,npw);
    PinnedRange pin_in(input,elements*sizeof(Complex64)),pin_out(output,elements*sizeof(Complex64));
    checked(cudaMemcpyAsync(fft_input_,input,elements*sizeof(Complex64),cudaMemcpyHostToDevice,stream_));
    if(fresh_operator){
      checked(cudaMemcpyAsync(fft_potential_,potential.data(),count*sizeof(double),cudaMemcpyHostToDevice,stream_));
      checked(cudaMemcpyAsync(fft_kinetic_,kinetic.data(),npw*sizeof(double),cudaMemcpyHostToDevice,stream_));
      checked(cudaMemcpyAsync(fft_indices_,indices.data(),npw*sizeof(std::size_t),cudaMemcpyHostToDevice,stream_));
      fft_identity_=operator_identity;
    }else record_performance("gpu_fft_operator_upload_saved",0,count+npw);
    checked(cudaMemsetAsync(fft_grid_,0,total*sizeof(cuDoubleComplex),stream_));
    scatter_fft<<<unsigned((elements+255)/256),256,0,stream_>>>(fft_input_,fft_indices_,fft_grid_,npw,count,elements);checked(cudaGetLastError());
    checked(cufftExecZ2Z(plan,fft_grid_,fft_grid_,CUFFT_INVERSE));
    multiply_fft<<<unsigned((total+255)/256),256,0,stream_>>>(fft_grid_,fft_potential_,count,total);checked(cudaGetLastError());
    checked(cufftExecZ2Z(plan,fft_grid_,fft_grid_,CUFFT_FORWARD));
    gather_fft<<<unsigned((elements+255)/256),256,0,stream_>>>(fft_input_,fft_grid_,fft_indices_,fft_kinetic_,fft_output_,npw,count,elements);checked(cudaGetLastError());
    checked(cudaMemcpyAsync(output,fft_output_,elements*sizeof(Complex64),cudaMemcpyDeviceToHost,stream_));checked(cudaStreamSynchronize(stream_));
    record_performance("cuda_fp64_fft_dispatch",0);
  }
  void gemm(MatrixOp ao,MatrixOp bo,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add)override {
    gemm_impl(0,ao,bo,m,n,k,a,lda,b,ldb,c,ldc,add);
  }
  void gemm_immutable_left(std::uint64_t identity,MatrixOp ao,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add)override {
    gemm_impl(identity,ao,MatrixOp::Normal,m,n,k,a,lda,b,ldb,c,ldc,add);
  }
  void gemm_impl(std::uint64_t identity,MatrixOp ao,MatrixOp bo,std::size_t m,std::size_t n,std::size_t k,
      const Complex64* a,std::size_t lda,const Complex64* b,std::size_t ldb,Complex64* c,std::size_t ldc,bool add) {
    const std::size_t ar=ao==MatrixOp::Normal?m:k,ac=ao==MatrixOp::Normal?k:m;
    const std::size_t br=bo==MatrixOp::Normal?k:n,bc=bo==MatrixOp::Normal?n:k;
    auto& resident=resident_left_[ao==MatrixOp::Normal?0:1];
    const bool reuse=identity&&resident.identity==identity&&resident.rows==ar&&resident.cols==ac&&resident.ld==lda;
    cuDoubleComplex* left;
    if(identity){if(!reuse){resident.identity=0;reserve(resident.values,resident.capacity,ar*ac);}left=resident.values;}
    else {reserve(a_,na_,ar*ac);left=a_;}
    reserve(b_,nb_,br*bc);reserve(c_,nc_,m*n);
    PinnedRange pin_a(reuse?nullptr:a,reuse?0:lda*ac*sizeof(Complex64));
    PinnedRange pin_b(b,ldb*bc*sizeof(Complex64)),pin_c(c,ldc*n*sizeof(Complex64));
    if(!reuse)checked(cudaMemcpy2DAsync(left,ar*sizeof(Complex64),a,lda*sizeof(Complex64),ar*sizeof(Complex64),ac,cudaMemcpyHostToDevice,stream_));
    if(reuse){record_performance("gpu_resident_left_hit",0);record_performance("gpu_resident_left_saved_values",0,ar*ac);}
    else if(identity)record_performance("gpu_resident_left_miss",0);
    checked(cudaMemcpy2DAsync(b_,br*sizeof(Complex64),b,ldb*sizeof(Complex64),br*sizeof(Complex64),bc,cudaMemcpyHostToDevice,stream_));
    if(add)checked(cudaMemcpy2DAsync(c_,m*sizeof(Complex64),c,ldc*sizeof(Complex64),m*sizeof(Complex64),n,cudaMemcpyHostToDevice,stream_));
    const cuDoubleComplex one=make_cuDoubleComplex(1,0),beta=make_cuDoubleComplex(add?1:0,0);
    checked(cublasZgemm(handle_,ao==MatrixOp::Normal?CUBLAS_OP_N:CUBLAS_OP_C,
        bo==MatrixOp::Normal?CUBLAS_OP_N:CUBLAS_OP_C,int(m),int(n),int(k),&one,left,int(ar),b_,int(br),&beta,c_,int(m)));
    checked(cudaMemcpy2DAsync(c,ldc*sizeof(Complex64),c_,m*sizeof(Complex64),m*sizeof(Complex64),n,cudaMemcpyDeviceToHost,stream_));
    checked(cudaStreamSynchronize(stream_));record_performance("cuda_fp64_dispatch",0);
    if(identity){resident.identity=identity;resident.rows=ar;resident.cols=ac;resident.ld=lda;}
  }
};
}
std::unique_ptr<Accelerator> make_cuda_accelerator(){PerformanceTimer timer("gpu_initialization");return std::make_unique<CudaAccelerator>();}
}
