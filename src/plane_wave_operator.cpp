#include "plane_wave_operator.hpp"
#include "env_cache.hpp"
#include "thread_budget.hpp"
#include "parallel_context.hpp"
#include "compute_backend.hpp"
#include "paw_compensated_product.hpp"
#include "fft_planning.hpp"
#include <Eigen/Dense>
#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <bit>
#include <cstdint>
#include <climits>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif

namespace sdft {
using C=std::complex<double>;
using M=Eigen::MatrixXcd;
struct PlaneWaveOperator::Impl {
  std::size_t n{},count{};
  int shape[3]{};
  double regularization{};
  std::vector<std::size_t> indices;
  std::vector<double> kinetic, local_real, h_diagonal, s_diagonal;
  struct Paw {M b,d,q;};
  std::vector<Paw> paw;
  // S6: per-atom real-space projector values on the FFT points where the
  // atom's projectors are non-negligible. Used only by inexact search
  // (exact=false); certification, density and derivatives keep B.
  struct RealSpacePaw {std::vector<std::size_t> points;M values;};
  struct AtomGeometry {Vec3 fractional{};double radius{};};
  std::vector<AtomGeometry> atom_geometry;
  std::array<Vec3,3> lattice{};
  double real_paw_cost_ratio{1.};
  std::vector<RealSpacePaw> real_paw;
  std::size_t real_paw_points{};
  double real_paw_retained{1.};
  bool real_space_search{};
  M all_b,all_db,all_qb,all_d,all_q;
  std::uint64_t b_identity=new_immutable_matrix_identity();
  const std::uint64_t operator_identity=new_immutable_matrix_identity();
  M chain_d,chain_q;
  std::vector<fftw_complex*> block_buffers;
  fftw_complex* buffer{};
  fftw_plan forward{},backward{};
  static bool compensated(bool exact) {
    const char* option=cached_getenv("SDFT_PAW_COMPENSATED_ACCUMULATION");
    if(option&&std::string(option)!="0"&&std::string(option)!="1")
      throw std::invalid_argument("SDFT_PAW_COMPENSATED_ACCUMULATION must be 0 or 1");
    return exact&&(!option||std::string(option)=="1");
  }
  static void paw_gemm(MatrixOp ao,MatrixOp bo,std::size_t rows,std::size_t columns,std::size_t inner,
      const C* a,std::size_t lda,const C* b,std::size_t ldb,C* c,std::size_t ldc,bool exact,bool add=false) {
    if(compensated(exact)&&bo==MatrixOp::Normal) {
      PerformanceTimer timer("paw_compensated_factor_product");
      compensated_paw_product(ao,rows,columns,inner,a,lda,b,ldb,c,ldc,add);
    } else complex_gemm(ao,bo,rows,columns,inner,a,lda,b,ldb,c,ldc,exact,add);
  }
  // full: every PAW factor product is compensated (historical default).
  // blocks: only the per-atom D/Q products, where radial channels cancel,
  // stay compensated; the long B*X and B^H*P contractions use BLAS.
  // Default (2026-10-15): blocks. On the Si64 Davidson replay the full scope
  // spent 273 of 320 s in scalar compensated B*X / B^H*P; blocks finished
  // in 81 s with eigenvalues within 3.4e-15 Ha and the same 1e-12 Ha
  // certificate. Converged Si2/MgO/Ni/Si8 SCF: |dE| <= 2.4e-12 eV,
  // |dF| <= 1.5e-9 eV/A against full. SDFT_DETERMINISTIC=1 or
  // SDFT_PAW_COMPENSATION_SCOPE=full restores the historical scope.
  static bool compensate_projections(bool exact) {
    static const bool blocks=[]{
      const char* scope=std::getenv("SDFT_PAW_COMPENSATION_SCOPE");
      if(scope&&std::string(scope)!="full"&&std::string(scope)!="blocks")
        throw std::invalid_argument("SDFT_PAW_COMPENSATION_SCOPE must be full or blocks");
      if(!scope)return !deterministic_mode();
      return std::string(scope)=="blocks";
    }();
    return compensated(exact)&&!blocks;
  }
  static void paw_immutable_left(std::uint64_t identity,MatrixOp ao,std::size_t rows,std::size_t columns,std::size_t inner,
      const C* a,std::size_t lda,const C* b,std::size_t ldb,C* c,std::size_t ldc,bool exact,bool add=false) {
    if(compensate_projections(exact))paw_gemm(ao,MatrixOp::Normal,rows,columns,inner,a,lda,b,ldb,c,ldc,exact,add);
    else complex_gemm_immutable_left(identity,ao,rows,columns,inner,a,lda,b,ldb,c,ldc,exact,add);
  }
  ~Impl(){std::lock_guard lock(fft_planner_mutex());if(forward)fftw_destroy_plan(forward);if(backward)fftw_destroy_plan(backward);if(buffer)fftw_free(buffer);for(auto* b:block_buffers)fftw_free(b);}
  std::size_t index(int h,int k,int l)const {
    auto wrap=[](int x,int size){int r=x%size;return r<0?r+size:r;};
    return (static_cast<std::size_t>(wrap(h,shape[0]))*shape[1]+wrap(k,shape[1]))*shape[2]+wrap(l,shape[2]);
  }
  // B is the bra (A=B*X). w_i(r) = (1/N) sum_G B_iG e^{-iG.r} (forward FFT)
  // gives sum_r w_i(r) psi(r) = sum_G B_iG psi_G for psi(r) the backward
  // FFT of psi_G, exactly when all points are kept. Keep
  // points where max_i |w_i(r)| exceeds threshold*max |w|, and record the
  // retained fraction of sum |w|^2 for the ledger. Threshold is set from
  // SDFT_PROJECTOR_RADIUS_FACTOR: a larger factor keeps more points.
  void build_real_space_projectors() {
    PerformanceTimer timer("real_space_projector_setup");
    double factor=1.3;
    if(const char* v=std::getenv("SDFT_PROJECTOR_RADIUS_FACTOR")) {
      char* end=nullptr;factor=std::strtod(v,&end);
      if(!end||*end||!std::isfinite(factor)||factor<1.||factor>2.)
        throw std::invalid_argument("SDFT_PROJECTOR_RADIUS_FACTOR must be in [1,2]");
    }
    // factor 1.0 -> 1e-2, 1.3 -> 1e-3.5, 2.0 -> 1e-7 relative amplitude.
    const double threshold=std::pow(10.,-2.-5.*(factor-1.));
    std::vector<C> grid(count);
    double kept=0,total=0;
    Eigen::Index offset=0;
    for(auto& atom:paw) {
      const Eigen::Index rows=atom.d.rows();
      M full(static_cast<Eigen::Index>(count),rows);
      for(Eigen::Index i=0;i<rows;++i) {
        std::fill_n(&buffer[0][0],2*count,0.);
        for(std::size_t g=0;g<n;++g){const C v=all_b(offset+i,g);buffer[indices[g]][0]=v.real();buffer[indices[g]][1]=v.imag();}
        fftw_execute_dft(forward,buffer,buffer);
        for(std::size_t r=0;r<count;++r)full(r,i)=C(buffer[r][0],buffer[r][1])/double(count);
      }
      Eigen::VectorXd peak=full.rowwise().norm();
      const double top=peak.maxCoeff();
      RealSpacePaw rs;
      const std::size_t atom_index=real_paw.size();
      const bool sphere=atom_index<atom_geometry.size()&&atom_geometry[atom_index].radius>0;
      // Geometric sphere (VASP LREAL-like, minimum image) when the atom
      // geometry is known; the amplitude threshold is the fallback for
      // snapshots, whose band-limited projectors ring across the cell.
      const double cut=sphere?factor*atom_geometry[atom_index].radius:0.;
      for(std::size_t r=0;r<count;++r){
        total+=peak[r]*peak[r];
        bool keep;
        if(sphere) {
          const std::size_t i0=r/(std::size_t(shape[1])*shape[2]),i1=(r/shape[2])%shape[1],i2=r%shape[2];
          double f[3]{double(i0)/shape[0],double(i1)/shape[1],double(i2)/shape[2]};
          const Vec3 tau=atom_geometry[atom_index].fractional;const double t[3]{tau.x,tau.y,tau.z};
          for(int a=0;a<3;++a){f[a]-=t[a];f[a]-=std::round(f[a]);}
          const double x=f[0]*lattice[0].x+f[1]*lattice[1].x+f[2]*lattice[2].x;
          const double y=f[0]*lattice[0].y+f[1]*lattice[1].y+f[2]*lattice[2].y;
          const double z=f[0]*lattice[0].z+f[1]*lattice[1].z+f[2]*lattice[2].z;
          keep=x*x+y*y+z*z<=cut*cut;
        } else keep=peak[r]>threshold*top;
        if(keep){rs.points.push_back(r);kept+=peak[r]*peak[r];}
      }
      rs.values.resize(rows,static_cast<Eigen::Index>(rs.points.size()));
      for(std::size_t j=0;j<rs.points.size();++j)rs.values.col(j)=full.row(rs.points[j]).transpose();
      real_paw_points+=rs.points.size();
      real_paw.push_back(std::move(rs));
      offset+=rows;
    }
    real_paw_retained=total>0?kept/total:1.;
    // Gather/scatter work relative to the dense B*X it replaces.
    real_paw_cost_ratio=double(real_paw_points)/(double(paw.size())*double(n));
    real_space_search=true;
    record_performance("real_space_projector_retained_power_ppm",0,std::size_t(1e6*real_paw_retained));
    record_performance("real_space_projector_cost_ratio_ppm",0,std::size_t(1e6*real_paw_cost_ratio));
    record_performance("real_space_projector_points",0,real_paw_points);
  }
  // Search-only H or S: one backward FFT per column gives psi(r); each
  // atom projects on its own points, applies D or Q, and adds back before
  // the forward FFT. S needs both FFTs because its PAW term lives on r.
  void apply_real_space_search(const C* x,std::size_t columns,C* y,bool is_overlap) {
    PerformanceTimer timer(is_overlap?"real_space_search_overlap":"real_space_search_hamiltonian");
    std::size_t workers=1;
#ifdef SDFT_HAS_OPENMP
    if(count*columns>=32768)workers=std::min<std::size_t>(columns,std::size_t(kernel_threads()));
#endif
    while(block_buffers.size()<workers){auto* b=fftw_alloc_complex(count);if(!b)throw std::bad_alloc();block_buffers.push_back(b);}
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1)
#endif
    for(std::size_t column=0;column<columns;++column) {
      const auto* in=x+column*n;auto* out=y+column*n;
      std::size_t worker=0;
#ifdef SDFT_HAS_OPENMP
      worker=omp_get_thread_num();
#endif
      auto* work=block_buffers[worker];
      std::fill_n(&work[0][0],2*count,0.);
      for(std::size_t i=0;i<n;++i){work[indices[i]][0]=in[i].real();work[indices[i]][1]=in[i].imag();}
      fftw_execute_dft(backward,work,work);
      // Projections from psi(r) before the local potential is applied.
      thread_local std::vector<Eigen::VectorXcd> amplitudes;
      amplitudes.resize(real_paw.size());
      for(std::size_t a=0;a<real_paw.size();++a) {
        const auto& rs=real_paw[a];
        Eigen::VectorXcd psi(static_cast<Eigen::Index>(rs.points.size()));
        for(std::size_t j=0;j<rs.points.size();++j)psi[j]=C(work[rs.points[j]][0],work[rs.points[j]][1]);
        const Eigen::VectorXcd p=rs.values*psi;
        amplitudes[a]=(is_overlap?paw[a].q:paw[a].d)*p;
      }
      if(is_overlap) for(std::size_t i=0;i<count;++i){work[i][0]=0;work[i][1]=0;}
      else for(std::size_t i=0;i<count;++i){work[i][0]*=local_real[i];work[i][1]*=local_real[i];}
      // Adjoint: f(r)=N sum_i conj(w_i(r)) c_i; the forward FFT and 1/N then
      // give sum_i conj(B_iG) c_i, i.e. B^H c.
      for(std::size_t a=0;a<real_paw.size();++a) {
        const auto& rs=real_paw[a];
        const Eigen::VectorXcd image=rs.values.adjoint()*amplitudes[a];
        for(std::size_t j=0;j<rs.points.size();++j){
          const C v=image[j]*double(count);work[rs.points[j]][0]+=v.real();work[rs.points[j]][1]+=v.imag();}
      }
      fftw_execute_dft(forward,work,work);
      if(is_overlap)for(std::size_t i=0;i<n;++i)out[i]=(1+regularization)*in[i]+C(work[indices[i]][0],work[indices[i]][1])/double(count);
      else for(std::size_t i=0;i<n;++i)out[i]=kinetic[i]*in[i]+C(work[indices[i]][0],work[indices[i]][1])/double(count);
    }
    record_performance("real_space_search_columns",0,columns);
  }
  void apply_block(const std::vector<C>& x,std::size_t columns,std::vector<C>& y,bool is_overlap,bool exact) {
    if(!columns||x.size()!=n*columns)throw std::invalid_argument("plane-wave block operator input dimension mismatch");
    y.resize(x.size());
    apply_block_into(x.data(),columns,y.data(),is_overlap,exact);
  }
  void apply_block_into(const C* x,std::size_t columns,C* y,bool is_overlap,bool exact,
                        const C* shared_amplitudes=nullptr,bool skip_paw=false) {
    if(!columns||!x||!y||x==y)throw std::invalid_argument("invalid direct plane-wave input/output buffers");
    PerformanceTimer timer(is_overlap?"operator_overlap":"operator_hamiltonian");
    if(real_space_search&&!exact&&!shared_amplitudes&&!skip_paw) {
      apply_real_space_search(x,columns,y,is_overlap);return;
    }
    const bool gpu_local=!is_overlap&&accelerated_local_hamiltonian(
        {shape[0],shape[1],shape[2]},indices,kinetic,local_real,x,columns,y,exact,operator_identity);
    // A completed GPU operation needs neither a CPU FFT workspace nor an
    // empty OpenMP team. Overlap identity also needs no FFT allocation.
    if(!gpu_local) {
    std::size_t workers=1;
#ifdef SDFT_HAS_OPENMP
    if(count*columns>=32768)workers=std::min<std::size_t>(columns,std::size_t(kernel_threads()));
#endif
    // FFTW plan execution on distinct aligned arrays is thread-safe. Planning
    // and buffer allocation occur before the team; no mutable buffer is shared.
    if(!is_overlap)while(block_buffers.size()<workers){auto* b=fftw_alloc_complex(count);if(!b)throw std::bad_alloc();block_buffers.push_back(b);}
    {
      PerformanceTimer fft_timer(is_overlap?"overlap_identity":"fft_local_block");
#ifdef SDFT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1)
#endif
      for(std::size_t column=0;column<columns;++column){
        const auto* in=x+column*n;auto* out=y+column*n;
        if(is_overlap){for(std::size_t i=0;i<n;++i)out[i]=(1+regularization)*in[i];continue;}
        std::size_t worker=0;
#ifdef SDFT_HAS_OPENMP
        worker=omp_get_thread_num();
#endif
        auto* work=block_buffers[worker];
        std::fill_n(&work[0][0],2*count,0.);
        for(std::size_t i=0;i<n;++i){work[indices[i]][0]=in[i].real();work[indices[i]][1]=in[i].imag();}
        fftw_execute_dft(backward,work,work);
        for(std::size_t i=0;i<count;++i){work[i][0]*=local_real[i];work[i][1]*=local_real[i];}
        fftw_execute_dft(forward,work,work);
        for(std::size_t i=0;i<n;++i)out[i]=kinetic[i]*in[i]+C(work[indices[i]][0],work[indices[i]][1])/double(count);
      }
    }
    }
    if(all_b.rows()&&!skip_paw){
      PerformanceTimer paw_timer("paw_block_projectors");
      M projected(all_b.rows(),columns);
      if(!paw.empty() || all_d.rows()){
        // Preserve Hermitian factorization B^H D B (and B^H Q B). Folding
        // D*B over NPW can amplify roundoff in almost cancelling PAW radial
        // channels into a non-Hermitian operator at strict SCF precision.
        // Only the small projector matrix multiplication is additional;
        // storing two projector x NPW transformed blocks is avoided.
        M owned_amplitudes;
        if(!shared_amplitudes) {
          owned_amplitudes.resize(all_b.rows(),columns);
          paw_immutable_left(b_identity,MatrixOp::Normal,all_b.rows(),columns,n,
              all_b.data(),all_b.outerStride(),x,n,owned_amplitudes.data(),owned_amplitudes.outerStride(),exact);
          record_performance("paw_amplitude_columns",0,columns);
          shared_amplitudes=owned_amplitudes.data();
        }
        const Eigen::Map<const M> amplitudes(shared_amplitudes,all_b.rows(),columns);
        if(!paw.empty()) {
          PerformanceTimer onsite_timer("paw_atom_block_onsite");
          std::size_t offset=0;
          // D/Q couples only channels on the same atom. Retain the original
          // Hermitian factors, but never multiply their off-atom zero blocks.
          // Small blocks stay in FP64; large B and B^H use the selected backend.
          for(const auto& atom:paw) {
            const auto& onsite=is_overlap?atom.q:atom.d;
            paw_gemm(MatrixOp::Normal,MatrixOp::Normal,onsite.rows(),columns,onsite.cols(),
                onsite.data(),onsite.outerStride(),amplitudes.data()+offset,amplitudes.outerStride(),
                projected.data()+offset,projected.outerStride(),true);
            offset+=onsite.rows();
          }
        } else {
          // PWO2 retains its actual historical all-projector dense operation.
          const auto& onsite=is_overlap?all_q:all_d;
          paw_gemm(MatrixOp::Normal,MatrixOp::Normal,all_b.rows(),columns,all_b.rows(),
              onsite.data(),onsite.outerStride(),amplitudes.data(),amplitudes.outerStride(),
              projected.data(),projected.outerStride(),exact);
        }
      }else{
        // Version1 snapshots replay the actual historical folded operator.
        const auto& transformed=is_overlap?all_qb:all_db;
        paw_gemm(MatrixOp::Normal,MatrixOp::Normal,all_b.rows(),columns,n,
            transformed.data(),transformed.outerStride(),x,n,projected.data(),projected.outerStride(),exact);
      }
      paw_immutable_left(b_identity,MatrixOp::Adjoint,n,columns,all_b.rows(),
          all_b.data(),all_b.outerStride(),projected.data(),projected.outerStride(),y,n,exact,true);
    }
  }
  void apply_pair_into(const C* x,std::size_t columns,C* hy,C* sy,bool exact,const C* provided=nullptr) {
    if(!columns||!x||!hy||!sy||x==hy||x==sy||hy==sy)
      throw std::invalid_argument("invalid joint plane-wave input/output buffers");
    PerformanceTimer timer("operator_joint_hs");
    if(real_space_search&&!exact&&!provided) {
      apply_real_space_search(x,columns,hy,false);apply_real_space_search(x,columns,sy,true);return;
    }
    if(!provided&&all_b.rows()&&(!paw.empty()||all_d.rows())&&!exact&&all_b.rows()<=1024) {
      const char* chain=cached_getenv("SDFT_GPU_PAW_CHAIN");
      if(chain&&std::string(chain)=="1") {
        if(!chain_d.rows()) {
          chain_d=M::Zero(all_b.rows(),all_b.rows());chain_q=chain_d;
          if(paw.empty()){chain_d=all_d;chain_q=all_q;}
          else {Eigen::Index offset=0;for(const auto& atom:paw){chain_d.block(offset,offset,atom.d.rows(),atom.d.cols())=atom.d;chain_q.block(offset,offset,atom.q.rows(),atom.q.cols())=atom.q;offset+=atom.d.rows();}}
        }
        M nh(n,columns),ns(n,columns);
        if(accelerated_paw_pair(operator_identity,n,all_b.rows(),columns,all_b.data(),chain_d.data(),chain_q.data(),x,nh.data(),ns.data(),false,b_identity)) {
          apply_block_into(x,columns,hy,false,exact,nullptr,true);
          apply_block_into(x,columns,sy,true,exact,nullptr,true);
          Eigen::Map<M>(hy,n,columns)+=nh;Eigen::Map<M>(sy,n,columns)+=ns;
          record_performance("paw_amplitude_columns",0,columns);
          record_performance("paw_joint_saved_amplitude_columns",0,columns);return;
        }
      }
    }
    M amplitudes;
    const C* shared=provided;
    if(!shared&&all_b.rows()&&(!paw.empty()||all_d.rows())) {
      // Fresh B*X computed once, with exactly the original factorization.
      // No DB folding, rotation propagation or pointer-keyed cache. Legacy
      // PWO1 has distinct folded DB/QB and must replay both separately.
      amplitudes.resize(all_b.rows(),columns);
      {
        PerformanceTimer projection_timer("paw_joint_amplitudes");
        paw_immutable_left(b_identity,MatrixOp::Normal,all_b.rows(),columns,n,
            all_b.data(),all_b.outerStride(),x,n,amplitudes.data(),amplitudes.outerStride(),exact);
      }
      record_performance("paw_amplitude_columns",0,columns);
      record_performance("paw_joint_saved_amplitude_columns",0,columns);
      shared=amplitudes.data();
    }
    const char* option=std::getenv("SDFT_PAW_PAIR_BACKPROJECT");
    if(option&&std::string(option)!="0"&&std::string(option)!="1")
      throw std::invalid_argument("SDFT_PAW_PAIR_BACKPROJECT must be 0 or 1");
    // Strict compensated H/S uses the same final addition as separate
    // application. Batched image-then-add has a different rounding order.
    if(shared&&option&&std::string(option)=="1"&&!compensate_projections(exact)) {
      PerformanceTimer batch_timer("paw_pair_backprojection");
      const Eigen::Index np=all_b.rows();M dq(np,2*columns),images(n,2*columns);
      const auto onsite=[&](const M& d,const M& q,Eigen::Index offset){
        for(int part=0;part<2;++part){const M& factor=part?q:d;
          paw_gemm(MatrixOp::Normal,MatrixOp::Normal,factor.rows(),columns,factor.cols(),factor.data(),factor.outerStride(),
              shared+offset,np,dq.data()+offset+part*columns*np,np,true);}
      };
      if(paw.empty())onsite(all_d,all_q,0);
      else {Eigen::Index offset=0;for(const auto& atom:paw){onsite(atom.d,atom.q,offset);offset+=atom.d.rows();}}
      paw_immutable_left(b_identity,MatrixOp::Adjoint,n,2*columns,np,
          all_b.data(),all_b.outerStride(),dq.data(),np,images.data(),n,exact);
      apply_block_into(x,columns,hy,false,exact,nullptr,true);
      apply_block_into(x,columns,sy,true,exact,nullptr,true);
      Eigen::Map<M>(hy,n,columns)+=images.leftCols(columns);
      Eigen::Map<M>(sy,n,columns)+=images.rightCols(columns);
      record_performance("paw_pair_saved_backprojection_calls",0);
      return;
    }
    apply_block_into(x,columns,hy,false,exact,shared);
    apply_block_into(x,columns,sy,true,exact,shared);
  }
  void project_into(const C* x,std::size_t columns,C* p,bool exact) {
    if(!columns||!x||!p||x==p||!all_b.rows())throw std::invalid_argument("invalid PAW projection buffers");
    paw_immutable_left(b_identity,MatrixOp::Normal,all_b.rows(),columns,n,
        all_b.data(),all_b.outerStride(),x,n,p,all_b.rows(),exact);
    record_performance("paw_amplitude_columns",0,columns);
  }
  void projector_metric_into(const C* p,std::size_t columns,C* qp) {
    if(!columns||!p||!qp||p==qp)throw std::invalid_argument("invalid projector metric buffers");
    if(!paw.empty()) {
      std::size_t offset=0;
      for(const auto& atom:paw){paw_gemm(MatrixOp::Normal,MatrixOp::Normal,atom.q.rows(),columns,atom.q.cols(),
          atom.q.data(),atom.q.rows(),p+offset,all_b.rows(),qp+offset,all_b.rows(),true);offset+=atom.q.rows();}
    }else paw_gemm(MatrixOp::Normal,MatrixOp::Normal,all_q.rows(),columns,all_q.cols(),
        all_q.data(),all_q.rows(),p,all_b.rows(),qp,all_b.rows(),true);
  }
  void apply(const std::vector<C>& x,std::vector<C>& y,bool is_overlap){apply_block(x,1,y,is_overlap,true);}
};

PlaneWaveOperator::PlaneWaveOperator(const std::vector<GVector>& basis,const FourierDensity& potential,
    std::vector<ComplexPawOperator> projectors,double hartree_to_ev,double regularization,
    std::uint64_t immutable_projector_identity):impl_(std::make_unique<Impl>()) {
  auto& op=*impl_;op.n=basis.size();op.regularization=regularization;
  if(immutable_projector_identity)op.b_identity=immutable_projector_identity;
  if(!op.n||!std::isfinite(hartree_to_ev)||hartree_to_ev<=0||!std::isfinite(regularization)||regularization<0)
    throw std::invalid_argument("invalid plane-wave operator dimensions or units");
  int low[3]{basis[0].h,basis[0].k,basis[0].l},high[3]{low[0],low[1],low[2]},radius[3]{};
  for(const auto& g:basis){const int key[]{g.h,g.k,g.l};for(int a=0;a<3;++a){low[a]=std::min(low[a],key[a]);high[a]=std::max(high[a],key[a]);}}
  double max_potential=0;
  for(const auto& [key,v]:potential){
    if(!std::isfinite(v.real())||!std::isfinite(v.imag()))throw std::invalid_argument("nonfinite local potential");
    const int k[]{std::get<0>(key),std::get<1>(key),std::get<2>(key)};
    // A local matrix element uses only V(G_i-G_j). Fine compensation/core
    // support can be much larger than this difference set. Excluding modes
    // outside its bounding box is an exact Galerkin projection, not a
    // density-cutoff approximation; it also prevents irrelevant high modes
    // from aliasing after we select the smaller wavefunction FFT grid.
    bool retained=true;
    for(int a=0;a<3;++a)retained &= std::abs(k[a])<=high[a]-low[a];
    if(retained)for(int a=0;a<3;++a)radius[a]=std::max(radius[a],std::abs(k[a]));
    max_potential=std::max(max_potential,std::abs(v));
  }
  const char* linear_flag=std::getenv("SDFT_LINEAR_OPERATOR_SETUP");
  if(linear_flag&&std::string(linear_flag)!="0"&&std::string(linear_flag)!="1")
    throw std::invalid_argument("SDFT_LINEAR_OPERATOR_SETUP must be 0 or 1");
  if(linear_flag&&std::string(linear_flag)=="1") {
    // Negation reverses lexicographic G order. Merge the normal map with its
    // reverse view instead of doing a tree lookup for every coefficient.
    // Missing opposite modes still contribute zero, including outside the
    // retained Galerkin box; finite checks and the original norm stay above.
    PerformanceTimer timer("operator_linear_hermiticity_check");
    const auto negated=[](const auto& key) {
      const auto [h,k,l]=key;
      if(h==INT_MIN||k==INT_MIN||l==INT_MIN)
        throw std::invalid_argument("Fourier reciprocal index cannot be negated");
      return std::tuple{-h,-k,-l};
    };
    auto reverse=potential.rbegin();
    for(const auto& [key,v]:potential) {
      while(reverse!=potential.rend() && negated(reverse->first)<key)++reverse;
      const C partner=reverse!=potential.rend() && negated(reverse->first)==key ? reverse->second : C{};
      if(std::abs(v-std::conj(partner))>1e-10*std::max(1.,max_potential))
        throw std::invalid_argument("local Fourier potential is not Hermitian");
    }
  }else {
    for(const auto& [key,v]:potential){
      const auto other=potential.find({-std::get<0>(key),-std::get<1>(key),-std::get<2>(key)});
      if(std::abs(v-std::conj(other==potential.end()?C{}:other->second))>1e-10*std::max(1.,max_potential))
        throw std::invalid_argument("local Fourier potential is not Hermitian");
    }
  }
  op.count=1;
  for(int a=0;a<3;++a){
    // N > max|V frequency| + max|Gi-Gj| excludes every cyclic alias that
    // could enter a retained coefficient. Even padded grids are also safe:
    // all retained frequencies lie strictly below their Nyquist frequency.
    const int minimum=std::max(3,2*std::max(radius[a],high[a]-low[a])+1);
    const char* mode=std::getenv("SDFT_FFT_GRID");
    if(mode&&std::string(mode)!="minimal"&&std::string(mode)!="fast")
      throw std::invalid_argument("SDFT_FFT_GRID must be minimal or fast");
    op.shape[a]=mode&&std::string(mode)=="minimal"?minimum:fast_fft_length(minimum);
    if(op.count>std::numeric_limits<std::size_t>::max()/op.shape[a])throw std::overflow_error("plane-wave FFT grid overflow");
    op.count*=op.shape[a];
  }
  op.buffer=fftw_alloc_complex(op.count);if(!op.buffer)throw std::bad_alloc();
  {std::lock_guard lock(fft_planner_mutex());
    op.forward=fftw_plan_dft_3d(op.shape[0],op.shape[1],op.shape[2],op.buffer,op.buffer,FFTW_FORWARD,FFTW_ESTIMATE);
    op.backward=fftw_plan_dft_3d(op.shape[0],op.shape[1],op.shape[2],op.buffer,op.buffer,FFTW_BACKWARD,FFTW_ESTIMATE);
  }
  if(!op.forward||!op.backward)throw std::runtime_error("plane-wave FFT planning failed");
  std::fill_n(&op.buffer[0][0],2*op.count,0.);
  for(const auto& [key,v]:potential){
    const int k[]{std::get<0>(key),std::get<1>(key),std::get<2>(key)};
    bool retained=true;for(int a=0;a<3;++a)retained &= std::abs(k[a])<=high[a]-low[a];
    if(!retained)continue;
    const auto i=op.index(k[0],k[1],k[2]);op.buffer[i][0]=v.real();op.buffer[i][1]=v.imag();
  }
  fftw_execute(op.backward);op.local_real.resize(op.count);
  for(std::size_t i=0;i<op.count;++i){
    if(std::abs(op.buffer[i][1])>1e-9*std::max(1.,std::abs(op.buffer[i][0])))throw std::runtime_error("complex real-space local potential");
    op.local_real[i]=op.buffer[i][0];
  }
  const auto zero=potential.find({0,0,0});const double v0=zero==potential.end()?0.:zero->second.real();
  for(const auto& g:basis){
    if(!std::isfinite(g.kinetic_ev))throw std::invalid_argument("nonfinite kinetic energy");
    op.indices.push_back(op.index(g.h,g.k,g.l));op.kinetic.push_back(g.kinetic_ev/hartree_to_ev);
    op.h_diagonal.push_back(g.kinetic_ev/hartree_to_ev+v0);op.s_diagonal.push_back(1+regularization);
  }
  auto sorted=op.indices;std::sort(sorted.begin(),sorted.end());
  if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())throw std::invalid_argument("duplicate plane-wave index");
  for(const auto& raw:projectors){
    const std::size_t nproj=raw.projectors;
    if(!nproj||raw.amplitudes.size()!=nproj*op.n||raw.d.size()!=nproj*nproj||raw.q.size()!=nproj*nproj)
      throw std::invalid_argument("invalid PAW low-rank operator dimensions");
    Impl::Paw p{M(nproj,op.n),M(nproj,nproj),M(nproj,nproj)};
    for(std::size_t i=0;i<nproj;++i){
      for(std::size_t j=0;j<op.n;++j)p.b(i,j)=raw.amplitudes[i*op.n+j];
      for(std::size_t j=0;j<nproj;++j){p.d(i,j)=raw.d[i*nproj+j];p.q(i,j)=raw.q[i*nproj+j];}
    }
    if(!p.b.allFinite()||!p.d.allFinite()||!p.q.allFinite())throw std::invalid_argument("nonfinite PAW projector/D/Q data");
    const double de=(p.d-p.d.adjoint()).norm(),qe=(p.q-p.q.adjoint()).norm();
    if(de>1e-10*std::max(1.,p.d.norm())||qe>1e-10*std::max(1.,p.q.norm())){
      std::ostringstream reason;reason<<"invalid Hermitian PAW blocks: atom="<<op.paw.size()<<" D_norm="<<p.d.norm()<<" D_asymmetry="<<de<<" Q_norm="<<p.q.norm()<<" Q_asymmetry="<<qe;
      throw std::invalid_argument(reason.str());
    }
    p.d=(.5*(p.d+p.d.adjoint())).eval();p.q=(.5*(p.q+p.q.adjoint())).eval();
    op.atom_geometry.push_back({raw.fractional,raw.radius_bohr});op.lattice=raw.lattice_bohr;
    const M db=p.d*p.b,qb=p.q*p.b;
    for(std::size_t i=0;i<op.n;++i){op.h_diagonal[i]+=(p.b.col(i).dot(db.col(i))).real();op.s_diagonal[i]+=(p.b.col(i).dot(qb.col(i))).real();}
    op.paw.push_back(std::move(p));
  }
  std::size_t total=0;for(const auto& p:op.paw)total+=p.b.rows();
  op.all_b.resize(total,op.n);
  std::size_t offset=0;
  for(const auto& p:op.paw){const auto rows=p.b.rows();
    op.all_b.middleRows(offset,rows)=p.b;offset+=rows;}
  if(const char* space=std::getenv("SDFT_PROJECTOR_SPACE")) {
    if(std::string(space)!="reciprocal"&&std::string(space)!="real-search")
      throw std::invalid_argument("SDFT_PROJECTOR_SPACE must be reciprocal or real-search");
    if(std::string(space)=="real-search"&&!op.paw.empty())op.build_real_space_projectors();
  }
  for(auto& p:op.paw)p.b.resize(0,0);
}
PlaneWaveOperator::~PlaneWaveOperator()=default;
PlaneWaveOperator::PlaneWaveOperator(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}

namespace {
// PWO1 stores folded DB/QB; PWO2 stores full Hermitian D/Q; PWO3 stores
// atom-block D/Q and boundaries, retaining each version's application order.
// Both use a little-endian uint64 header; IEEE754 FP64 arrays; complex
// arrays are interleaved real/imaginary in column-major order. No FFT plans,
// backend pointers or object ABI are serialized. Reject foreign endianness
// rather than silently interpreting corrupt values.
void snapshot_platform() {
  static_assert(sizeof(C)==2*sizeof(double));
  static_assert(std::numeric_limits<double>::is_iec559);
  if(std::endian::native!=std::endian::little)
    throw std::runtime_error("solver snapshot currently requires little-endian IEEE754 host");
}
template<class T> void snapshot_write(std::ofstream& f,const T* values,std::size_t count) {
  if(count)f.write(reinterpret_cast<const char*>(values),count*sizeof(T));
  if(!f)throw std::runtime_error("solver snapshot write failed");
}
template<class T> void snapshot_read(std::ifstream& f,T* values,std::size_t count) {
  if(count)f.read(reinterpret_cast<char*>(values),count*sizeof(T));
  if(!f)throw std::runtime_error("truncated solver snapshot");
}
}
void PlaneWaveOperator::save_solver_snapshot(const std::string& path,std::size_t roots,
    std::size_t iterations,double tolerance,const std::vector<C>& initial)const {
  snapshot_platform();const auto& op=*impl_;
  if(!roots||roots>op.n||!iterations||!std::isfinite(tolerance)||tolerance<=0||
      (!initial.empty()&&initial.size()!=roots*op.n))throw std::invalid_argument("invalid solver snapshot settings");
  for(auto z:initial)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::invalid_argument("nonfinite snapshot initial vector");
  if(std::filesystem::exists(path)||std::filesystem::exists(path+".tmp"))
    throw std::runtime_error("refusing to overwrite solver snapshot: "+path);
  std::ofstream file(path+".tmp",std::ios::binary);if(!file)throw std::runtime_error("cannot open solver snapshot: "+path);
  const bool blocked=!op.paw.empty();
  const bool factored=blocked||op.all_d.rows()||!op.all_b.rows();
  file.write(blocked?"SDFTPWO3":factored?"SDFTPWO2":"SDFTPWO1",8);
  const std::uint64_t header[]{op.n,op.count,std::uint64_t(op.shape[0]),std::uint64_t(op.shape[1]),
      std::uint64_t(op.shape[2]),std::uint64_t(op.all_b.rows()),roots,iterations,initial.size()};
  snapshot_write(file,header,9);const double settings[]{op.regularization,tolerance};snapshot_write(file,settings,2);
  if(blocked) {
    const std::uint64_t atoms=op.paw.size();snapshot_write(file,&atoms,1);
    for(const auto& p:op.paw){const std::uint64_t channels=p.d.rows();snapshot_write(file,&channels,1);}
  }
  std::vector<std::uint64_t> indices(op.indices.begin(),op.indices.end());snapshot_write(file,indices.data(),op.n);
  for(const auto* array:{&op.kinetic,&op.h_diagonal,&op.s_diagonal,&op.local_real})snapshot_write(file,array->data(),array->size());
  snapshot_write(file,op.all_b.data(),op.all_b.size());
  if(blocked){for(const auto& p:op.paw)for(const auto* matrix:{&p.d,&p.q})snapshot_write(file,matrix->data(),matrix->size());}
  else if(factored){for(const auto* matrix:{&op.all_d,&op.all_q})snapshot_write(file,matrix->data(),matrix->size());}
  else {for(const auto* matrix:{&op.all_db,&op.all_qb})snapshot_write(file,matrix->data(),matrix->size());}
  snapshot_write(file,initial.data(),initial.size());file.close();if(!file)throw std::runtime_error("cannot finalize solver snapshot");
  std::filesystem::rename(path+".tmp",path);
}
std::unique_ptr<PlaneWaveOperator> PlaneWaveOperator::load_solver_snapshot(const std::string& path,
    std::size_t& roots,std::size_t& iterations,double& tolerance,std::vector<C>& initial) {
  snapshot_platform();std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file)throw std::runtime_error("cannot open solver snapshot: "+path);
  const auto length=file.tellg();file.seekg(0);char magic[8];snapshot_read(file,magic,8);
  const std::string format(magic,8);
  if(format!="SDFTPWO1"&&format!="SDFTPWO2"&&format!="SDFTPWO3")throw std::runtime_error("unsupported solver snapshot format");
  const bool blocked=format=="SDFTPWO3",factored=format!="SDFTPWO1";
  std::uint64_t header[9];snapshot_read(file,header,9);
  const auto [n,count,s0,s1,s2,projectors,r,it,ni]=std::tuple{header[0],header[1],header[2],header[3],header[4],header[5],header[6],header[7],header[8]};
  // Limits prevent malformed snapshots from allocating unbounded arrays.
  // An exact file-length check precedes every allocation and FFT plan.
  constexpr std::uint64_t limit=std::uint64_t(1)<<40;
  auto product=[&](std::uint64_t a,std::uint64_t b){if(b&&a>limit/b)throw std::runtime_error("solver snapshot dimension overflow");return a*b;};
  if(!n||!count||n>count||s0<3||s1<3||s2<3||s0>INT_MAX||s1>INT_MAX||s2>INT_MAX||
      product(product(s0,s1),s2)!=count||!r||r>n||!it||it>INT_MAX||
      (ni&&ni!=product(n,r)))throw std::runtime_error("invalid solver snapshot dimensions");
  double settings[2];snapshot_read(file,settings,2);
  std::vector<std::uint64_t> blocks;
  std::uint64_t block_square=0,extra_header=0;
  if(blocked) {
    std::uint64_t atoms;snapshot_read(file,&atoms,1);
    if(!atoms||atoms>projectors||length<0||atoms>std::uint64_t(length)/8)
      throw std::runtime_error("invalid solver snapshot atom count");
    blocks.resize(atoms);snapshot_read(file,blocks.data(),atoms);
    std::uint64_t sum=0;
    for(auto channels:blocks) {
      if(!channels||channels>projectors-sum)throw std::runtime_error("invalid solver snapshot atom boundaries");
      sum+=channels;const auto square=product(channels,channels);
      if(square>limit-block_square)throw std::runtime_error("solver snapshot atom storage overflow");
      block_square+=square;
    }
    if(sum!=projectors)throw std::runtime_error("solver snapshot atom boundaries do not span projectors");
    extra_header=8*(1+atoms);
  }
  const auto paw_doubles=factored?product(2,product(n,projectors))+product(4,blocked?block_square:product(projectors,projectors)):
      product(6,product(n,projectors));
  const auto doubles=product(4,n)+count+paw_doubles+product(2,ni);
  if(doubles>limit/8||length<0||std::uint64_t(length)!=96+extra_header+product(8,doubles))
    throw std::runtime_error("solver snapshot size mismatch");
  auto op=std::make_unique<Impl>();op->n=n;op->count=count;op->shape[0]=s0;op->shape[1]=s1;op->shape[2]=s2;
  op->regularization=settings[0];tolerance=settings[1];
  if(!std::isfinite(op->regularization)||op->regularization<0||!std::isfinite(tolerance)||tolerance<=0)
    throw std::runtime_error("invalid solver snapshot scalars");
  std::vector<std::uint64_t> indices(n);snapshot_read(file,indices.data(),n);
  op->indices.assign(indices.begin(),indices.end());auto sorted=indices;std::sort(sorted.begin(),sorted.end());
  if(sorted.back()>=count||std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
    throw std::runtime_error("invalid solver snapshot FFT indices");
  op->kinetic.resize(n);op->h_diagonal.resize(n);op->s_diagonal.resize(n);op->local_real.resize(count);
  for(auto* array:{&op->kinetic,&op->h_diagonal,&op->s_diagonal,&op->local_real}) {
    snapshot_read(file,array->data(),array->size());
    for(double value:*array)if(!std::isfinite(value))throw std::runtime_error("nonfinite solver snapshot array");
  }
  for(double value:op->s_diagonal)if(value<=0)throw std::runtime_error("invalid solver snapshot overlap diagonal");
  op->all_b.resize(projectors,n);snapshot_read(file,op->all_b.data(),op->all_b.size());
  if(!op->all_b.allFinite())throw std::runtime_error("nonfinite solver snapshot projector array");
  const auto read_onsite=[&](M& matrix,std::size_t rows,std::size_t cols,bool hermitian) {
    matrix.resize(rows,cols);snapshot_read(file,matrix.data(),matrix.size());
    if(!matrix.allFinite())throw std::runtime_error("nonfinite solver snapshot PAW array");
    if(hermitian&&(matrix-matrix.adjoint()).norm()>1e-12*std::max(1.,matrix.norm()))
      throw std::runtime_error("non-Hermitian solver snapshot onsite matrix");
  };
  if(blocked)for(auto channels:blocks) {
    Impl::Paw p;read_onsite(p.d,channels,channels,true);read_onsite(p.q,channels,channels,true);op->paw.push_back(std::move(p));
  }
  else for(auto* matrix:factored?std::vector<M*>{&op->all_d,&op->all_q}:std::vector<M*>{&op->all_db,&op->all_qb}) {
    matrix->resize(projectors,factored?projectors:n);snapshot_read(file,matrix->data(),matrix->size());
    if(!matrix->allFinite())throw std::runtime_error("nonfinite solver snapshot PAW array");
    if(factored&&(*matrix-matrix->adjoint()).norm()>1e-12*std::max(1.,matrix->norm()))
      throw std::runtime_error("non-Hermitian solver snapshot onsite matrix");
  }
  initial.resize(ni);snapshot_read(file,initial.data(),ni);
  for(auto z:initial)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite solver snapshot warm vector");
  op->buffer=fftw_alloc_complex(count);if(!op->buffer)throw std::bad_alloc();
  {std::lock_guard lock(fft_planner_mutex());
    op->forward=fftw_plan_dft_3d(s0,s1,s2,op->buffer,op->buffer,FFTW_FORWARD,FFTW_ESTIMATE);
    op->backward=fftw_plan_dft_3d(s0,s1,s2,op->buffer,op->buffer,FFTW_BACKWARD,FFTW_ESTIMATE);
  }
  if(!op->forward||!op->backward)throw std::runtime_error("snapshot FFT planning failed");
  if(const char* space=std::getenv("SDFT_PROJECTOR_SPACE"))
    if(std::string(space)=="real-search"&&!op->paw.empty())op->build_real_space_projectors();
  roots=r;iterations=it;
  return std::unique_ptr<PlaneWaveOperator>(new PlaneWaveOperator(std::move(op)));
}
namespace {
// MPI band groups: every rank of a k group holds the same Davidson block.
// Each rank applies the operator to its contiguous share of columns, then
// the block is reassembled by an allgather. Columns are independent, so the
// result equals the single-rank one; the per-column arithmetic is unchanged.
constexpr std::size_t band_split_minimum_columns=4;
template<class Apply>
void band_split(std::size_t columns,std::size_t in_rows,std::size_t out_rows,
                const std::complex<double>* x,std::complex<double>* y,Apply&& apply) {
  auto& mpi=parallel();
  if(mpi.bandgroups()<=1||columns<band_split_minimum_columns){apply(x,columns,y);return;}
  const auto first=mpi.band_first_column(columns,mpi.bandgroup_rank());
  const auto last=mpi.band_first_column(columns,mpi.bandgroup_rank()+1);
  if(last>first)apply(x+first*in_rows,last-first,y+first*out_rows);
  mpi.band_allgather_columns(y,out_rows,columns);
  record_performance("mpi_band_split_columns",0,last-first);
}
}  // namespace
HermitianOperator PlaneWaveOperator::hamiltonian(){
  HermitianOperator op{impl_->n,impl_->h_diagonal,[this](const auto& x,auto& y){impl_->apply(x,y,false);},[this](const auto& x,std::size_t c,auto& y,bool exact){impl_->apply_block(x,c,y,false,exact);},[this](const C* x,std::size_t c,C* y,bool exact){band_split(c,impl_->n,impl_->n,x,y,[&](const C* a,std::size_t m,C* b){impl_->apply_block_into(a,m,b,false,exact);});},impl_.get(),[this](const C* x,std::size_t c,C* hy,C* sy,bool exact){
    auto& mpi=parallel();
    if(mpi.bandgroups()<=1||c<band_split_minimum_columns){impl_->apply_pair_into(x,c,hy,sy,exact);return;}
    const auto first=mpi.band_first_column(c,mpi.bandgroup_rank()),last=mpi.band_first_column(c,mpi.bandgroup_rank()+1);
    if(last>first)impl_->apply_pair_into(x+first*impl_->n,last-first,hy+first*impl_->n,sy+first*impl_->n,exact);
    mpi.band_allgather_columns(hy,impl_->n,c);mpi.band_allgather_columns(sy,impl_->n,c);}};
  op.kinetic=impl_->kinetic;
  // Real-space projectors are a search accelerator only: Davidson then
  // searches with exact=false and refreshes/certifies with exact=true.
  op.approximate_search=impl_->real_space_search;
  if(impl_->all_b.rows()&&(!impl_->paw.empty()||impl_->all_d.rows())) {
    op.projector_rows=impl_->all_b.rows();
    op.project_into=[this](const C* x,std::size_t c,C* p,bool exact){impl_->project_into(x,c,p,exact);};
    op.apply_projected_into=[this](const C* x,std::size_t c,const C* p,C* y,bool exact){impl_->apply_block_into(x,c,y,false,exact,p);};
    op.apply_projected_pair_into=[this](const C* x,std::size_t c,const C* p,C* h,C* s,bool exact){impl_->apply_pair_into(x,c,h,s,exact,p);};
  }
  return op;
}
HermitianOperator PlaneWaveOperator::overlap(){
  HermitianOperator op{impl_->n,impl_->s_diagonal,[this](const auto& x,auto& y){impl_->apply(x,y,true);},[this](const auto& x,std::size_t c,auto& y,bool exact){impl_->apply_block(x,c,y,true,exact);},[this](const C* x,std::size_t c,C* y,bool exact){band_split(c,impl_->n,impl_->n,x,y,[&](const C* a,std::size_t m,C* b){impl_->apply_block_into(a,m,b,true,exact);});},impl_.get(),{}};
  if(impl_->all_b.rows()&&(!impl_->paw.empty()||impl_->all_d.rows())) {
    op.projector_rows=impl_->all_b.rows();
    op.apply_projected_into=[this](const C* x,std::size_t c,const C* p,C* y,bool exact){impl_->apply_block_into(x,c,y,true,exact,p);};
    op.metric_identity_scale=1.+impl_->regularization;
    op.projector_metric_into=[this](const C* p,std::size_t c,C* qp){impl_->projector_metric_into(p,c,qp);};
  }
  // Determine the full nontrivial S signature from the original low-rank
  // block; the complementary plane-wave directions have norm 1+reg.
  // No dense NPW-square matrix and no modification of Q or beta is made.
  if(impl_->all_b.rows()&&op.projector_metric_into) {
    const auto p=impl_->all_b.rows();
    M gram=impl_->all_b*impl_->all_b.adjoint();
    Eigen::SelfAdjointEigenSolver<M> ge((.5*(gram+gram.adjoint())).eval());
    if(ge.info()!=Eigen::Success)throw std::runtime_error("original PAW Gram decomposition failed");
    const M root=ge.eigenvectors()*ge.eigenvalues().cwiseMax(0.).cwiseSqrt().asDiagonal()*ge.eigenvectors().adjoint();
    M q(p,p);const M identity=M::Identity(p,p);
    impl_->projector_metric_into(identity.data(),p,q.data());
    M small=(1.+impl_->regularization)*identity+root*q*root;
    Eigen::SelfAdjointEigenSolver<M> se((.5*(small+small.adjoint())).eval());
    if(se.info()!=Eigen::Success)throw std::runtime_error("original PAW overlap signature failed");
    const double minimum=std::min(1.+impl_->regularization,se.eigenvalues().minCoeff());
    op.indefinite_metric=minimum < -1e-10;
    if(op.indefinite_metric)record_performance("original_paw_negative_metric_detected",0.,1);
    if(std::abs(minimum)<=1e-10)throw std::runtime_error("original PAW overlap is numerically singular");
  }
  return op;
}
std::size_t PlaneWaveOperator::fft_grid_points()const{return impl_->count;}
std::size_t PlaneWaveOperator::storage_bytes()const{
  std::size_t bytes=impl_->count*(2*sizeof(double)+sizeof(double))+impl_->n*(3*sizeof(double)+sizeof(std::size_t));
  bytes+=(impl_->all_b.size()+impl_->all_db.size()+impl_->all_qb.size()+impl_->all_d.size()+impl_->all_q.size())*sizeof(C);
  for(const auto& p:impl_->paw)bytes+=(p.d.size()+p.q.size())*sizeof(C);
  bytes+=impl_->block_buffers.size()*impl_->count*sizeof(C);
  return bytes;
}
}
