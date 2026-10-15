#include "env_cache.hpp"
#include "plane_wave_operator.hpp"
#include "compute_backend.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>

int main() {
  using namespace sdft;using C=std::complex<double>;using M=Eigen::MatrixXcd;
  std::vector<GVector> g;
  for(int h=-2;h<=2;++h)for(int k=-1;k<=1;++k)for(int l=-1;l<=1;++l)
    g.push_back({h,k,l,{double(h)+.13,double(k)+.21,double(l)-.11},27.211386245988*(.8+h*h*.6+k*k*.9+l*l*.7)});
  FourierDensity v{{{0,0,0},.4},{{1,-1,0},C(.21,.13)},{{-1,1,0},C(.21,-.13)},
    {{4,2,-2},C(.09,-.06)},{{-4,-2,2},C(.09,.06)},
    // Outside all basis differences; must never alias back into H.
    {{7,0,0},C(.7,.2)},{{-7,0,0},C(.7,-.2)}};
  const std::size_t n=g.size(),np=3;
  ComplexPawOperator p;p.projectors=np;p.amplitudes.resize(np*n);p.d.resize(np*np);p.q.resize(np*np);
  M b(np,n),d=M::Zero(np,np),q=M::Zero(np,np);
  for(std::size_t i=0;i<np;++i)for(std::size_t j=0;j<n;++j)
    p.amplitudes[i*n+j]=b(i,j)=C(.04*std::sin(.3*(i+1)*(j+2)),.03*std::cos(.7*(i+3)*(j+1)));
  d<<.3,C(.04,.06),-.03,C(.04,-.06),-.2,C(.02,.05),-.03,C(.02,-.05),.5;
  q<<.06,C(.01,-.01),0,C(.01,.01),.04,0,0,0,.08;
  for(std::size_t i=0;i<np;++i)for(std::size_t j=0;j<np;++j){p.d[i*np+j]=d(i,j);p.q[i*np+j]=q(i,j);}
  ComplexPawOperator p2;p2.projectors=2;
  M b2=.7*b.topRows(2);b2.row(1)*=C(.6,.8);
  const M d2=d.topLeftCorner(2,2),q2=q.topLeftCorner(2,2);
  for(int i=0;i<2;++i) {
    for(std::size_t j=0;j<n;++j)p2.amplitudes.push_back(b2(i,j));
    for(int j=0;j<2;++j){p2.d.push_back(d2(i,j));p2.q.push_back(q2(i,j));}
  }
  PlaneWaveOperator op(g,v,{p,p2},27.211386245988);
  auto retained_v=v;retained_v.erase({7,0,0});retained_v.erase({-7,0,0});
  PlaneWaveOperator retained(g,retained_v,{p,p2},27.211386245988);
  // Irrelevant fine-grid frequencies must change neither H nor its FFT
  // allocation. Include a far-away term to detect accidental grid inflation.
  auto fine_v=v;fine_v[{91,-37,63}]=C(.1,.2);fine_v[{-91,37,-63}]=C(.1,-.2);
  PlaneWaveOperator fine(g,fine_v,{p,p2},27.211386245988);
  if(op.fft_grid_points()!=retained.fft_grid_points()||op.fft_grid_points()!=fine.fft_grid_points())return 7;
  M h=M::Zero(n,n),s=(1+1e-12)*M::Identity(n,n);
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){
    const auto key=std::make_tuple(g[i].h-g[j].h,g[i].k-g[j].k,g[i].l-g[j].l);
    const auto found=v.find(key);if(found!=v.end())h(i,j)=found->second;
  }
  for(std::size_t i=0;i<n;++i)h(i,i)+=g[i].kinetic_ev/27.211386245988;
  h+=b.adjoint()*d*b;s+=b.adjoint()*q*b;
  h+=b2.adjoint()*d2*b2;s+=b2.adjoint()*q2*b2;
  // Check every column, not just one convenient vector.
  double application_error=0;
  for(std::size_t i=0;i<n;++i){
    std::vector<C> x(n),hy,sy,fy;x[i]=1;op.hamiltonian().apply(x,hy);op.overlap().apply(x,sy);fine.hamiltonian().apply(x,fy);
    for(std::size_t j=0;j<n;++j)application_error=std::max({application_error,std::abs(hy[j]-h(j,i)),std::abs(sy[j]-s(j,i)),std::abs(fy[j]-h(j,i))});
  }
  if(application_error>1e-12)return 1;
  // Nontrivial band block, conjugate PAW channels, and FFT thread buffers.
  M block(n,13),expected_h,expected_s;
  for(std::size_t i=0;i<n;++i)for(int j=0;j<13;++j)block(i,j)=C(std::sin(.27*(i+1)*(j+2)),std::cos(.34*(i+3)*(j+1)));
  expected_h=h*block;expected_s=s*block;
  std::vector<C> input(block.data(),block.data()+block.size()),bh,bs;
  op.hamiltonian().apply_block(input,13,bh,true);op.overlap().apply_block(input,13,bs,true);
  const double block_error=std::max((Eigen::Map<const M>(bh.data(),n,13)-expected_h).norm(),(Eigen::Map<const M>(bs.data(),n,13)-expected_s).norm());
  if(block_error>1e-10)return 5;
  M direct_h(n,13),direct_s(n,13);
  op.hamiltonian().apply_block_into(block.data(),13,direct_h.data(),true);
  op.overlap().apply_block_into(block.data(),13,direct_s.data(),true);
  if(!std::equal(bh.begin(),bh.end(),direct_h.data()) ||
      !std::equal(bs.begin(),bs.end(),direct_s.data()))return 16;
  // Native PAW chain must match dense H/S for new values and repeated
  // static projectors; no intermediate FP64 host roundtrip is required.
  for(int rep=0;rep<3;++rep){
    const M changed=block*C(1.+.1*rep,.07*rep);
    op.hamiltonian().apply_pair_into(changed.data(),13,direct_h.data(),direct_s.data(),false);
    if((direct_h-h*changed).norm()>1e-10||(direct_s-s*changed).norm()>1e-10)return 32;
  }
  // Projector state belongs to these values, not to their addresses. Check
  // a complex rotation, changed values in the same storage, and S as well as H.
  auto state_h=op.hamiltonian(),state_s=op.overlap();
  M amplitudes(state_h.projector_rows,13),projected_h(n,13),projected_s(n,13);
  state_h.project_into(block.data(),13,amplitudes.data(),true);
  M qp(amplitudes.rows(),amplitudes.cols());
  state_s.projector_metric_into(amplitudes.data(),13,qp.data());
  if(!state_h.apply_projected_pair_into)return 48;
  state_h.apply_projected_pair_into(block.data(),13,amplitudes.data(),projected_h.data(),projected_s.data(),true);
  if((projected_h-expected_h).norm()>1e-10||(projected_s-expected_s).norm()>1e-10)return 49;
  const M factored_gram=state_s.metric_identity_scale*(block.adjoint()*block)+amplitudes.adjoint()*qp;
  if((factored_gram-block.adjoint()*s*block).cwiseAbs().maxCoeff()>1e-10)return 33;
  state_h.apply_projected_into(block.data(),13,amplitudes.data(),projected_h.data(),true);
  state_s.apply_projected_into(block.data(),13,amplitudes.data(),projected_s.data(),true);
  if((projected_h-expected_h).norm()>1e-10||(projected_s-expected_s).norm()>1e-10)return 28;
  M rotation(13,7);
  for(int i=0;i<13;++i)for(int j=0;j<7;++j)rotation(i,j)=C(std::sin(i+2.*j),std::cos(2.*i-j))/13.;
  M rotated=block*rotation,rotated_p=amplitudes*rotation,rotated_h(n,7),rotated_s(n,7);
  state_h.apply_projected_into(rotated.data(),7,rotated_p.data(),rotated_h.data(),true);
  state_s.apply_projected_into(rotated.data(),7,rotated_p.data(),rotated_s.data(),true);
  if((rotated_h-h*rotated).norm()>1e-10||(rotated_s-s*rotated).norm()>1e-10)return 29;
  for(const char* metric:{"0","1"})for(const char* reuse:{"0","1"})for(const char* preconditioner:{"diagonal","kinetic"}) {
    setenv("SDFT_PROJECTOR_METRIC",metric,1);sdft::refresh_environment_cache();
    setenv("SDFT_PROJECTOR_STATE",reuse,1);sdft::refresh_environment_cache();setenv("SDFT_PRECONDITIONER",preconditioner,1);sdft::refresh_environment_cache();
    const auto solved=solve_generalized_hermitian_davidson(state_h,state_s,8,256,1e-12);
    Eigen::GeneralizedSelfAdjointEigenSolver<M> independent(h,s);
    if(!solved.converged)return 30;
    for(int j=0;j<8;++j)if(std::abs(solved.eigenvalues[j]-independent.eigenvalues()[j])>1e-11)return 31;
  }
  unsetenv("SDFT_PROJECTOR_STATE");sdft::refresh_environment_cache();unsetenv("SDFT_PRECONDITIONER");sdft::refresh_environment_cache();
  unsetenv("SDFT_PROJECTOR_METRIC");sdft::refresh_environment_cache();
  bool alias_rejected=false;
  try{op.hamiltonian().apply_block_into(block.data(),13,block.data(),true);}
  catch(const std::invalid_argument&){alias_rejected=true;}
  if(!alias_rejected)return 17;
  const auto pair_h=op.hamiltonian(),pair_s=op.overlap();
  if(!pair_h.pair_identity||pair_h.pair_identity!=pair_s.pair_identity||!pair_h.apply_pair_into)return 18;
  pair_h.apply_pair_into(block.data(),13,direct_h.data(),direct_s.data(),true);
  if(!std::equal(bh.begin(),bh.end(),direct_h.data())||!std::equal(bs.begin(),bs.end(),direct_s.data()))return 19;
  // Same buffer address, different values: projector reuse is synchronous
  // and input-bound, never a pointer cache that can silently become stale.
  M changed_block=block;changed_block*=C(.7,.3);
  pair_h.apply_pair_into(changed_block.data(),13,direct_h.data(),direct_s.data(),true);
  if((direct_h-h*changed_block).norm()>1e-10||(direct_s-s*changed_block).norm()>1e-10)return 20;
  bool joint_alias_rejected=false;
  try{pair_h.apply_pair_into(block.data(),13,direct_h.data(),direct_h.data(),true);}
  catch(const std::invalid_argument&){joint_alias_rejected=true;}
  if(!joint_alias_rejected)return 21;
  // A fixed geometry keeps B resident across SCF operators; density-dependent
  // D/Q must NEVER share the PAW-chain cache identity. Independently alter
  // both onsite matrices while retaining the same immutable projector token.
  const auto geometry_identity=new_immutable_matrix_identity();
  auto updated_p=p;updated_p.d[0]+=.17;updated_p.q[0]+=.025;
  PlaneWaveOperator fixed_geometry(g,v,{p,p2},27.211386245988,1e-12,geometry_identity);
  PlaneWaveOperator changed_onsite(g,v,{updated_p,p2},27.211386245988,1e-12,geometry_identity);
  M changed_h=h+.17*b.row(0).adjoint()*b.row(0);
  M changed_s=s+.025*b.row(0).adjoint()*b.row(0);
  for(int rep=0;rep<3;++rep) {
    auto& instance=rep==1?changed_onsite:fixed_geometry;
    instance.hamiltonian().apply_pair_into(block.data(),13,direct_h.data(),direct_s.data(),false);
    const M& expected_h=rep==1?changed_h:h;const M& expected_s=rep==1?changed_s:s;
    if((direct_h-expected_h*block).norm()>1e-10||(direct_s-expected_s*block).norm()>1e-10)return 50;
  }
  // Fresh residual and independent lowest spectrum in both candidate/control.
  Eigen::GeneralizedSelfAdjointEigenSolver<M> reference(h,s);
  if(reference.info()!=Eigen::Success)return 22;
  for(const char* setting:{"0","1"}) {
    setenv("SDFT_JOINT_HS",setting,1);sdft::refresh_environment_cache();
    const auto roots=solve_generalized_hermitian_davidson(pair_h,pair_s,8,256,1e-12);
    if(!roots.converged)return 23;
    for(std::size_t j=0;j<8;++j)if(std::abs(roots.eigenvalues[j]-reference.eigenvalues()[j])>1e-11)return 24;
  }
  // Mismatched overlap identity must never invoke the joint H/S callback.
  auto mismatched_s=pair_s;mismatched_s.pair_identity=&retained;
  auto poisoned_h=pair_h;poisoned_h.apply_pair_into=[](const C*,std::size_t,C*,C*,bool){throw std::runtime_error("invalid pairing");};
  if(!solve_generalized_hermitian_davidson(poisoned_h,mismatched_s,8,256,1e-12).converged)return 25;
  setenv("SDFT_JOINT_HS","invalid",1);sdft::refresh_environment_cache();
  bool joint_setting_rejected=false;
  try{solve_generalized_hermitian_davidson(pair_h,pair_s,8,256,1e-12);}
  catch(const std::invalid_argument&){joint_setting_rejected=true;}
  unsetenv("SDFT_JOINT_HS");sdft::refresh_environment_cache();
  if(!joint_setting_rejected)return 26;
  // Snapshot must preserve actual atom-block PAW tensors and FFT grid bitwise,
  // including complex warm vectors. Test corruption before allocating plans.
  const auto snapshot=(std::filesystem::temp_directory_path()/
      ("sparse-dft-pwop-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".pwop")).string();
  struct Cleanup {std::string path;~Cleanup(){std::filesystem::remove(path);std::filesystem::remove(path+".bad");}} cleanup{snapshot};
  op.save_solver_snapshot(snapshot,13,256,1e-12,input);
  std::size_t restored_roots,restored_iters;double restored_tolerance;std::vector<C> restored_initial;
  auto restored=PlaneWaveOperator::load_solver_snapshot(snapshot,restored_roots,restored_iters,restored_tolerance,restored_initial);
  std::vector<C> restored_h,restored_s;
  restored->hamiltonian().apply_block(restored_initial,13,restored_h,true);
  restored->overlap().apply_block(restored_initial,13,restored_s,true);
  if(restored_roots!=13||restored_iters!=256||restored_tolerance!=1e-12||restored_initial!=input||restored_h!=bh||restored_s!=bs)return 11;
  bool overwrite_rejected=false;
  try{op.save_solver_snapshot(snapshot,13,256,1e-12,input);}catch(const std::runtime_error&){overwrite_rejected=true;}
  if(!overwrite_rejected)return 12;
  std::ifstream saved(snapshot,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(saved)),{});saved.close();
  if(bytes.substr(0,8)!="SDFTPWO3")return 14;
  // PWO1 and PWO2 remain their historical folded/full-dense operations. Build
  // independent legacy fixtures from the same arrays, never replace their
  // actual operator with the new atom-block application on import.
  const std::size_t total=np+2,extra_header=8*3;
  const std::size_t begin_b=96+extra_header+(4*n+op.fft_grid_points())*8;
  const std::size_t end_b=begin_b+total*n*sizeof(C);
  const std::size_t end_small=end_b+2*(np*np+4)*sizeof(C);
  M all_b(total,n),all_d=M::Zero(total,total),all_q=M::Zero(total,total);
  all_b.topRows(np)=b;all_b.bottomRows(2)=b2;
  all_d.topLeftCorner(np,np)=d;all_d.bottomRightCorner(2,2)=d2;
  all_q.topLeftCorner(np,np)=q;all_q.bottomRightCorner(2,2)=q2;
  for(int version : {1,2}) {
    std::string legacy="SDFTPWO"+std::to_string(version);
    legacy+=bytes.substr(8,88);legacy+=bytes.substr(96+extra_header,end_b-96-extra_header);
    const M hd=version==1?M(all_d*all_b):all_d,qs=version==1?M(all_q*all_b):all_q;
    legacy.append(reinterpret_cast<const char*>(hd.data()),hd.size()*sizeof(C));
    legacy.append(reinterpret_cast<const char*>(qs.data()),qs.size()*sizeof(C));
    legacy+=bytes.substr(end_small);
    {std::ofstream file(snapshot+".bad",std::ios::binary);file.write(legacy.data(),legacy.size());}
    auto old=PlaneWaveOperator::load_solver_snapshot(snapshot+".bad",restored_roots,restored_iters,restored_tolerance,restored_initial);
    old->hamiltonian().apply_block(input,13,restored_h,true);old->overlap().apply_block(input,13,restored_s,true);
    if((Eigen::Map<const M>(restored_h.data(),n,13)-expected_h).norm()>1e-10 ||
       (Eigen::Map<const M>(restored_s.data(),n,13)-expected_s).norm()>1e-10)return 15;
    old->hamiltonian().apply_pair_into(input.data(),13,direct_h.data(),direct_s.data(),true);
    if(!std::equal(restored_h.begin(),restored_h.end(),direct_h.data())||
        !std::equal(restored_s.begin(),restored_s.end(),direct_s.data()))return 27;
  }
  for(int damage=0;damage<5;++damage){auto damaged=bytes;
    if(damage==0)damaged.resize(damaged.size()-1);
    if(damage==1)damaged.push_back('x');
    if(damage==2)damaged[0]='X';
    if(damage==3)for(int i=8;i<16;++i)damaged[i]=char(0xff);
    if(damage==4){const double nan=std::numeric_limits<double>::quiet_NaN();std::memcpy(damaged.data()+80,&nan,8);}
    {std::ofstream bad(snapshot+".bad",std::ios::binary);bad.write(damaged.data(),damaged.size());}
    bool corrupt_rejected=false;
    try{auto bad=PlaneWaveOperator::load_solver_snapshot(snapshot+".bad",restored_roots,restored_iters,restored_tolerance,restored_initial);}
    catch(const std::runtime_error&){corrupt_rejected=true;}
    if(!corrupt_rejected)return 13;
  }
  // Accelerator search path is checked against the independently assembled
  // dense operator, not against a second invocation of its own FFT code.
  op.hamiltonian().apply_block(input,13,bh,false);op.overlap().apply_block(input,13,bs,false);
  const double search_block_error=std::max((Eigen::Map<const M>(bh.data(),n,13)-expected_h).norm(),(Eigen::Map<const M>(bs.data(),n,13)-expected_s).norm());
  if(search_block_error>1e-10){std::cerr<<"search operator error="<<search_block_error<<'\n';return 6;}
  Eigen::GeneralizedSelfAdjointEigenSolver<M> dense(h,s);
  const auto solved=solve_generalized_hermitian_davidson(op.hamiltonian(),op.overlap(),8,256,1e-12);
  if(!solved.converged)return 2;
  double eigen_error=0,metric_error=0;
  M vectors(n,8);
  for(std::size_t i=0;i<8;++i){eigen_error=std::max(eigen_error,std::abs(solved.eigenvalues[i]-dense.eigenvalues()[i]));
    for(std::size_t j=0;j<n;++j)vectors(j,i)=solved.eigenvectors[i*n+j];}
  metric_error=(vectors.adjoint()*s*vectors-M::Identity(8,8)).norm();
  if(eigen_error>1e-10||metric_error>1e-10)return 3;
  // Incremental slabs must recover the same spectrum as full Galerkin
  // recomputation, including fresh original H/S residual certification.
  const char* setting=std::getenv("SDFT_INCREMENTAL_RITZ");
  const std::string previous=setting?setting:"";
  const auto set_ritz_mode=[](const std::string& value) {
#ifdef _WIN32
    _putenv_s("SDFT_INCREMENTAL_RITZ",value.c_str());sdft::refresh_environment_cache();
#else
    if(value.empty())unsetenv("SDFT_INCREMENTAL_RITZ");else setenv("SDFT_INCREMENTAL_RITZ",value.c_str(),1);sdft::refresh_environment_cache();
#endif
  };
  set_ritz_mode("0");
  const auto full_ritz=solve_generalized_hermitian_davidson(op.hamiltonian(),op.overlap(),8,256,1e-12);
  set_ritz_mode(previous);
  if(!full_ritz.converged)return 18;
  for(std::size_t root=0;root<8;++root)
    if(std::abs(full_ritz.eigenvalues[root]-solved.eigenvalues[root])>1e-12)return 19;
  // Check returned roots against an independent dense H/S application,
  // including an iteration-limited solve that must not report convergence.
  // This also tests fresh FP64 certification of an uncertified final block.
  const auto limited=solve_generalized_hermitian_davidson(op.hamiltonian(),op.overlap(),8,1,1e-12);
  if(limited.converged)return 8;
  double certified_residual_error=0;
  for(const auto* result : {&solved,&limited}){
    bool converged=true;
    for(std::size_t root=0;root<8;++root){
      const Eigen::Map<const Eigen::VectorXcd> vector(result->eigenvectors.data()+root*n,n);
      const double norm=(h*vector-result->eigenvalues[root]*(s*vector)).norm();
      certified_residual_error=std::max(certified_residual_error,std::abs(norm-result->residual_norms[root]));
      if(norm>1.01e-12)converged=false;
      if(std::abs((vector.adjoint()*s*vector)(0,0)-C(1))>1e-12)return 9;
    }
    if(converged!=result->converged||certified_residual_error>1e-13)return 10;
  }
  // Invalid Fourier data must fail before a non-Hermitian H reaches SCF.
  if(std::getenv("SDFT_LINEAR_OPERATOR_SETUP")&&std::string(std::getenv("SDFT_LINEAR_OPERATOR_SETUP"))=="1") {
    const auto mode=[](const char* value) {
#ifdef _WIN32
      _putenv_s("SDFT_LINEAR_OPERATOR_SETUP",value);sdft::refresh_environment_cache();
#else
      setenv("SDFT_LINEAR_OPERATOR_SETUP",value,1);sdft::refresh_environment_cache();
#endif
    };
    auto missing_large=v;missing_large.erase({-1,1,0});
    auto missing_small=FourierDensity{{{3,-2,1},C(2e-12,-3e-12)}};
    auto nonfinite=v;nonfinite[{90,0,0}]=C(std::numeric_limits<double>::quiet_NaN(),0);
    for(const auto& sample:{v,fine_v,retained_v,missing_large,missing_small,nonfinite,FourierDensity{},
        FourierDensity{{{0,0,0},C(0,.5)}},FourierDensity{{{0,0,0},C(.37,0)}}}) {
      bool previous_rejected=false,current_rejected=false;
      std::vector<C> previous_h,previous_s,current_h,current_s;
      std::size_t previous_grid=0,current_grid=0;
      for(const char* setting:{"0","1"}) {
        mode(setting);
        try {
          PlaneWaveOperator trial(g,sample,{p,p2},27.211386245988);
          auto& hy=setting[0]=='0'?previous_h:current_h;
          auto& sy=setting[0]=='0'?previous_s:current_s;
          (setting[0]=='0'?previous_grid:current_grid)=trial.fft_grid_points();
          trial.hamiltonian().apply_block(input,13,hy,true);
          trial.overlap().apply_block(input,13,sy,true);
        }catch(const std::invalid_argument&) {
          (setting[0]=='0'?previous_rejected:current_rejected)=true;
        }
      }
      if(previous_rejected!=current_rejected || previous_grid!=current_grid ||
          previous_h!=current_h || previous_s!=current_s)return 27;
    }
    mode("1");
  }
  bool rejected=false;v[{1,-1,0}]+=C(0,.5);
  try{PlaneWaveOperator bad(g,v,{p},27.211386245988);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)return 4;
  std::cout<<"plane_wave_operator PASS application_error="<<application_error<<" search_block_error="<<search_block_error<<" eigen_error="<<eigen_error
    <<" S_orthogonality_error="<<metric_error<<" certified_residual_error="<<certified_residual_error
    <<" grid="<<op.fft_grid_points()<<" storage_bytes="<<op.storage_bytes()<<'\n';
  write_performance_report("plane_wave_operator.performance.json");
}
