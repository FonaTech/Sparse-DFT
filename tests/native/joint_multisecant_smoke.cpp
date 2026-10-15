#include "env_cache.hpp"
#include "joint_paw_mixer.hpp"
#include <iostream>
int main() {
  sdft::AdaptiveSecantStep adaptive(true),disabled(false);
  if(adaptive.observe(1.)!=1. || adaptive.observe(2.)!=.5 || adaptive.observe(.5)!=.625)return 10;
  if(disabled.observe(1.)!=1. || disabled.observe(2.)!=1.)return 11;
  try {adaptive.observe(NAN);return 12;}catch(const std::invalid_argument&){}
  // Independent exact-inverse model: damping an exact root proposal
  // contracts the residual by 1-scale, not necessarily by more than 30%.
  sdft::AdaptiveSecantStep recovering(true);
  double feedback=1., recovered=recovering.observe(feedback);
  for(int i=0;i<3;++i){feedback*=2.;recovered=recovering.observe(feedback);}
  if(recovered!=.125)return 16;
  for(int i=0;i<50&&recovered<1.;++i){feedback*=1.-recovered;recovered=recovering.observe(feedback);}
  if(recovered!=1.)return 17;
  // Alternating growth/decline and a near-flat residual cannot justify
  // recovery. This is residual feedback, not an evaluated line search.
  for(bool oscillating:{false,true}){
    sdft::AdaptiveSecantStep conservative(true);feedback=1.;conservative.observe(feedback);
    for(int i=0;i<3;++i){feedback*=2.;conservative.observe(feedback);}
    for(int i=0;i<30;++i){feedback*=oscillating?(i%2?1.1:.9):.999;
      if(conservative.observe(feedback)!=.125)return 18;}
  }
  using C=std::complex<double>;
  using H=sdft::JointSecants::History;
  {
    // Analytic cone boundary: .2-alpha>=0 gives alpha<=.2. The largest
    // allowed dyadic step is 1/8. A useful extrapolation exists even though
    // the full matrix is negative. The legacy guard discards this direction.
    const std::vector<C> prior_s{2.,.4},prior_p{.2,0.,0.,.4};
    const std::vector<C> full_s{3.,.8},full_p{-.8,0.,0.,.6};
    const std::vector<C> output_s{2.5,.6},output_p{.3,0.,0.,.45};
    auto s=full_s,p=full_p;
    const auto cone=sdft::retract_joint_paw_cone(s,p,prior_s,prior_p,output_s,output_p,{2},.2);
    if(cone.convex_fallback||cone.scale!=.125||std::abs(p[0]-.075)>1e-14||
       std::abs(p[3]-.425)>1e-14||std::abs(s[1]-.45)>1e-14)return 43;
    auto negative_s=full_s,negative_p=full_p,rs=output_s,rp=output_p;
    for(std::size_t i=0;i<rs.size();++i)rs[i]-=prior_s[i];
    for(std::size_t i=0;i<rp.size();++i)rp[i]-=prior_p[i];
    if(!sdft::finalize_collinear_mixing_step(negative_s,negative_p,prior_s,prior_p,
       rs,rp,{2},1.,.2)||std::abs(negative_p[0]-p[0])<.1)return 44;
    // The same eigenvalue bound after a complex unitary orbital rotation.
    Eigen::Matrix2cd u;u<<C(1,0),C(0,1),C(0,1),C(1,0);u/=std::sqrt(2.);
    const auto rotate=[&](const std::vector<C>& v) {
      Eigen::Matrix2cd m;for(int i=0;i<2;++i)for(int j=0;j<2;++j)m(i,j)=v[2*i+j];
      const Eigen::Matrix2cd r=u*m*u.adjoint();std::vector<C> out(4);
      for(int i=0;i<2;++i)for(int j=0;j<2;++j)out[2*i+j]=r(i,j);
      return out;
    };
    auto rotated_s=full_s,rotated_p=rotate(full_p);
    const auto rotated=sdft::retract_joint_paw_cone(rotated_s,rotated_p,prior_s,
       rotate(prior_p),output_s,rotate(output_p),{2},.2);
    const auto expected=rotate(p);
    if(rotated.scale!=cone.scale||rotated.convex_fallback)return 45;
    for(std::size_t i=0;i<p.size();++i)if(std::abs(rotated_p[i]-expected[i])>1e-14)return 46;
    // Global spin exchange must retain the same scalar step and both blocks.
    auto spin_s=full_s;std::vector<C> spin_p{.6,-.8};
    const auto swapped=sdft::retract_joint_paw_cone(spin_s,spin_p,prior_s,
       {.4,.2},output_s,{.45,.3},{1,1},.2);
    if(swapped.scale!=.125||swapped.convex_fallback||std::abs(spin_p[1]-.075)>1e-14)return 47;
    // A singular prior and outward direction has no useful finite cone
    // segment. Physical convex fallback is required, never eigenvalue clips.
    s={2.};p={-1.};
    const auto singular=sdft::retract_joint_paw_cone(s,p,{1.},{0.},{3.},{1.},{1},.2);
    if(!singular.convex_fallback||std::abs(p[0]-.2)>1e-14||std::abs(s[0]-1.4)>1e-14)return 48;
    s={NAN};p={1.};
    try{sdft::retract_joint_paw_cone(s,p,{1.},{1.},{1.},{1.},{1},.2);return 49;}
    catch(const std::invalid_argument&){}
    std::cout<<"analytic PAW cone boundary, complex orbital covariance, global spin swap, singular fallback PASS\n";
  }
  // Independent constant-norm rotation: the L-infinity feedback doubles
  // although the weighted residual geometry used by the fit is unchanged.
  // A genuine norm increase must still contract the default adaptive step.
  {
    unsetenv("SDFT_MIXER");sdft::refresh_environment_cache();unsetenv("SDFT_MULTISECANT_BACKTRACK");sdft::refresh_environment_cache();
    sdft::JointPawMixer mixer(.2,4,1.,{1});
    std::vector<C> smooth(4,2.),site{1.};
    auto next=smooth;for(auto& v:next)v+=.5;
    mixer.update(smooth,site,next,site);
    const double first=mixer.feedback_norm;
    next=smooth;next[0]+=1.;mixer.update(smooth,site,next,site);
    if(std::abs(mixer.feedback_norm-first)>1e-14||mixer.step_scale!=1.||mixer.history_restarted)return 28;
    next=smooth;next[0]+=2.;mixer.update(smooth,site,next,site);
    if(mixer.step_scale!=.5||!mixer.trial_rejected||mixer.history_restarted)return 29;
    next=smooth;next[0]+=7.;mixer.update(smooth,site,next,site);
    if(mixer.step_scale!=.25||!mixer.trial_rejected||mixer.history_restarted)return 30;
    sdft::AdaptiveSecantStep old_maximum(true);old_maximum.observe(.5);
    if(old_maximum.observe(1.)!=.5)return 31;
    std::cout<<"constant-L2 rotation preserves step; evaluated L2 growth rejects actual trial\n";
  }
  {
    // Independent two-mode response with a transient rotation. The small
    // sample following the initial one is not an envelope for every later
    // orthogonal mode. A larger residual still inside accepted history must
    // be available to learn a secant; a value beyond that history is rejected.
    sdft::JointPawMixer window(.2,2,1.,{1});
    std::vector<C> x{2.,2.},site{1.},image=x;
    image[0]+=1.;window.update(x,site,image,site);
    image=x;image[0]+=.1;window.update(x,site,image,site);
    image=x;image[1]+=.5;window.update(x,site,image,site);
    if(window.trial_rejected||!(.5>1.25*.1))return 40;
    image=x;image[1]+=2.;window.update(x,site,image,site);
    if(!window.trial_rejected)return 41;
    for(int i=0;i<8;++i){image=x;image[0]+=.1;window.update(x,site,image,site);}
    image=x;image[1]+=.5;window.update(x,site,image,site);
    if(!window.trial_rejected)return 42;
    std::cout<<"finite accepted residual window permits response rotation; large trial rejected\n";
  }
  // Independent soft-orbital susceptibility model. Its analytic root is
  // known without evaluating the implementation's inverse approximation.
  // Both sample points are positive occupations; the root also is physical.
  // The old residual-proportional bound is a negative control: it cannot
  // resolve this mode even if the inverse response is learned exactly.
  {
    constexpr double target=1.01, susceptibility=1e-5;
    H res,images;
    for(double point:{1.,1.00001}) {
      const double r=-susceptibility*(point-target);
      res.push_back({C(r,0)});images.push_back({C(point+r,0)});
    }
    const auto secant=sdft::joint_secant_coefficients(res,{},1.,0.);
    auto proposal=images.back();
    sdft::apply_joint_secants(proposal,images,res,1.,secant);
    const double prior=1.00001,step=proposal[0].real()-prior;
    const double scale=sdft::secant_state_trust_scale(prior*prior,step*step);
    const double actual=prior+scale*step;
    const double old_scale=std::min(1.,2*std::abs(res.back()[0])/std::abs(step));
    const double negative=prior+old_scale*step;
    if(!secant.usable||std::abs(susceptibility*(actual-target))>1e-12||
       std::abs(susceptibility*(negative-target))<9e-8)return 25;
    // A large extrapolation must still obey the displacement trust bound.
    const double bounded=sdft::secant_state_trust_scale(4.,100.);
    if(std::abs(bounded-.02)>1e-14)return 26;
    for(double invalid:std::vector<double>{-1.,NAN,INFINITY}) {
      try {sdft::secant_state_trust_scale(1.,invalid);return 27;}
      catch(const std::invalid_argument&){}
    }
    std::cout<<"soft-response analytic error="<<std::abs(actual-target)
             <<" old-bound residual="<<std::abs(susceptibility*(negative-target))<<'\n';
  }
  // Independent partial-history oracle: one resolved soft mode and a
  // stiff mode absent from the secant span. Learned response must retain
  // its exact inverse; the complement must retain the requested B0=.2.
  // Unit B0 (full DIIS) grows the stiff residual fivefold in this model.
  {
    const std::vector<C> target{1.02,1.};
    const auto image=[&](const std::vector<C>& x) {
      return std::vector<C>{x[0]-1e-5*(x[0]-target[0]),
                            x[1]-6.*(x[1]-target[1])};
    };
    const auto run=[&](const char* method) {
      setenv("SDFT_MIXER",method,1);sdft::refresh_environment_cache();
      // Prescribed sampling is not an actual consecutive SCF trajectory.
      setenv("SDFT_MULTISECANT_BACKTRACK","0",1);sdft::refresh_environment_cache();
      sdft::JointPawMixer mixer(.2,2,1.,{1});
      std::vector<C> x{1.,1.01},site{1.};
      mixer.update(x,site,image(x),site);
      // Prescribe a second independently sampled point; both samples have
      // the same stiff coordinate, hence no stiff secant was learned.
      x={1.01,1.01};mixer.update(x,site,image(x),site);
      if(mixer.rank!=1||mixer.step_scale!=1.)throw std::runtime_error("partial-history rank or scale differs");
      return x;
    };
    const auto actual=run("multisecant"),negative=run("full_diis");
    unsetenv("SDFT_MIXER");sdft::refresh_environment_cache();
    unsetenv("SDFT_MULTISECANT_BACKTRACK");sdft::refresh_environment_cache();
    if(std::abs(actual[0]-target[0])>1e-10||std::abs(actual[1]-.998)>1e-12||
       std::abs(negative[1]-.95)>1e-12)return 33;
    if(std::abs(actual[1]-target[1])>=.01||std::abs(negative[1]-target[1])<=.04)return 34;
    std::cout<<"partial-history learned soft root preserved; B0 stiff error="
             <<std::abs(actual[1]-target[1])<<" unit-B0 negative="<<std::abs(negative[1]-target[1])<<'\n';
  }
  // Independent nonlinear root x*=1 of R=1-x^2. The evaluated Anderson
  // overshoot increases |R|; the next state must be half the actual segment
  // from the previous accepted point, not a proposal from the bad endpoint.
  {
    setenv("SDFT_MULTISECANT_TRUST_RADIUS","1",1);sdft::refresh_environment_cache();
    const auto image=[](const std::vector<C>& x){return std::vector<C>{x[0]+1.-x[0]*x[0]};};
    sdft::JointPawMixer mixer(.2,1,10.,{1});
    std::vector<C> x{.2},site{1.};
    mixer.update(x,site,image(x),site);
    if(std::abs(x[0]-.392)>1e-14)return 35;
    const double anchor=x[0].real();
    mixer.update(x,site,image(x),site);
    const double overshoot=x[0].real();
    if(std::abs(1.-overshoot*overshoot)<=1.25*std::abs(1.-anchor*anchor))return 36;
    mixer.update(x,site,image(x),site);
    if(!mixer.trial_rejected||std::abs(x[0]-.5*(anchor+overshoot))>1e-14||
        std::abs(site[0]-1.)>1e-14)return 37;
    for(int it=0;it<80&&std::abs(1.-x[0]*x[0])>1e-11;++it)mixer.update(x,site,image(x),site);
    if(std::abs(x[0]-1.)>1e-10)return 38;
    // Explicit user disable and full DIIS keep their selected semantics.
    for(const char* mode:{"multisecant","full_diis"}) {
      setenv("SDFT_MIXER",mode,1);sdft::refresh_environment_cache();setenv("SDFT_MULTISECANT_BACKTRACK","0",1);sdft::refresh_environment_cache();
      sdft::JointPawMixer unguarded(.2,1,1.,{1});x={.2};
      for(int i=0;i<3;++i)unguarded.update(x,site,image(x),site);
      if(unguarded.trial_rejected||unguarded.adaptive_enabled())return 39;
    }
    unsetenv("SDFT_MIXER");sdft::refresh_environment_cache();unsetenv("SDFT_MULTISECANT_BACKTRACK");sdft::refresh_environment_cache();unsetenv("SDFT_MULTISECANT_TRUST_RADIUS");sdft::refresh_environment_cache();
    std::cout<<"nonlinear evaluated overshoot retracted from accepted anchor; root and user overrides PASS\n";
  }
  // Independently known two-mode root with nearly parallel secants.
  // A resolved 1e-5 orbital response must not be removed by squaring its
  // condition number. The old 1e-8 Gram cutoff is a negative control.
  {
    const std::vector<C> target{.2,1.01};H residuals,outputs;
    for(const std::vector<double>& point:std::vector<std::vector<double>>{{.4,1.04},{1.4,2.04},{2.4,1.04}}) {
      const std::vector<C> r{-(point[0]-target[0].real()),-1e-5*(point[1]-target[1].real())};
      residuals.push_back(r);outputs.push_back({point[0]+r[0],point[1]+r[1]});
    }
    const auto fitted=sdft::joint_secant_coefficients(residuals,{},1.,0.);
    auto proposal=outputs.back();sdft::apply_joint_secants(proposal,outputs,residuals,1.,fitted);
    double exact_error=0.;for(int i=0;i<2;++i)exact_error=std::max(exact_error,std::abs(proposal[i]-target[i]));
    Eigen::Matrix2d independent_directions;
    for(int j=0;j<2;++j)for(int i=0;i<2;++i)independent_directions(i,j)=(residuals[j+1][i]-residuals[j][i]).real();
    const auto independent_coefficients=independent_directions.fullPivLu().solve(
        Eigen::Vector2d(residuals.back()[0].real(),residuals.back()[1].real())).eval();
    auto oracle=outputs.back();
    for(int j=0;j<2;++j)for(int i=0;i<2;++i)oracle[i]-=independent_coefficients[j]*(outputs[j+1][i]-outputs[j][i]);
    double oracle_error=0.;for(int i=0;i<2;++i)oracle_error=std::max(oracle_error,std::abs(oracle[i]-target[i]));
    const Eigen::Matrix2d normalized=independent_directions/std::sqrt(1.+1e-10);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> gram(normalized.transpose()*normalized);
    const int old_rank=(gram.eigenvalues().array()>gram.eigenvalues().maxCoeff()*1e-8).count();
    if(fitted.rank!=2||old_rank!=1||exact_error>1e-10||oracle_error>1e-10)return 32;
    std::cout<<"resolved weak response SVD rank="<<fitted.rank<<" old Gram rank="<<old_rank
             <<" analytic error="<<exact_error<<" independent LU error="<<oracle_error<<'\n';
  }
  // A full extrapolation diag(-.8,.6) is indefinite, but its one-eighth
  // step from diag(.2,.4) is diag(.075,.425), a legitimate density.
  // The corresponding smooth charge/magnetization must use the same step.
  {
    const std::vector<C> prior_s{{2.,.4},{-.2,.1}},prior_p{.2,0.,0.,.4};
    const std::vector<C> residual_s{{.5,-.2},{.3,.6}},residual_p{.1,0.,0.,-.1};
    auto smooth=std::vector<C>{{6.,-.4},{1.4,.9}};
    auto onsite=std::vector<C>{-.8,0.,0.,.6};
    if(sdft::finalize_collinear_mixing_step(smooth,onsite,prior_s,prior_p,
        residual_s,residual_p,{2},.125,.2))return 19;
    if(std::abs(onsite[0]-.075)>1e-14||std::abs(onsite[3]-.425)>1e-14||
        std::abs(smooth[0]-C(2.5,.3))>1e-14||std::abs(smooth[1]-C(0.,.2))>1e-14)return 20;
    // A still-invalid contracted proposal triggers one coupled convex
    // fallback, rather than clipping only the onsite density.
    smooth={{6.,-.4},{1.4,.9}};onsite={-3.,0.,0.,.6};
    if(!sdft::finalize_collinear_mixing_step(smooth,onsite,prior_s,prior_p,
        residual_s,residual_p,{2},.125,.2))return 21;
    for(std::size_t q=0;q<smooth.size();++q)
      if(std::abs(smooth[q]-(prior_s[q]+.025*residual_s[q]))>1e-14)return 22;
    if(std::abs(onsite[0]-.2025)>1e-14||std::abs(onsite[3]-.3975)>1e-14)return 23;
    try {sdft::finalize_collinear_mixing_step(smooth,onsite,prior_s,prior_p,
        residual_s,residual_p,{2},0.,.2);return 24;}catch(const std::invalid_argument&){}
  }
  // Coupled affine response with an unstable undamped fixed-point map,
  // complex density components, and a second onsite block.
  std::vector<C> x{{1.2,.6},{-.8,.2},{.5,-.7},{1.1,.9}};
  const std::vector<C> root{{.1,.3},{-.4,.7},{.2,-.5},{.4,-.3}};
  Eigen::Matrix4d a;
  a<<-8.,.7,.2,.1,.7,-.3,.15,.05,.2,.15,-2.,.4,.1,.05,.4,-.7;
  H sr,so,pr,po;
  constexpr double damping=.05;
  double residual=1.;
  for(int it=0;it<120;++it) {
    Eigen::Vector4cd delta;
    for(int i=0;i<4;++i)delta[i]=x[i]-root[i];
    const auto r=(a*delta).eval();residual=r.norm();
    if(residual<1e-11)break;
    sr.push_back({r[0],r[1]});pr.push_back({r[2],r[3]});
    so.push_back({x[0]+r[0],x[1]+r[1]});po.push_back({x[2]+r[2],x[3]+r[3]});
    if(sr.size()>6)for(auto* h:{&sr,&so,&pr,&po})h->erase(h->begin());
    std::vector<C> s{x[0]+damping*r[0],x[1]+damping*r[1]},
                   p{x[2]+damping*r[2],x[3]+damping*r[3]};
    const auto update=sdft::joint_secant_coefficients(sr,pr,.5,.5);
    sdft::apply_joint_secants(s,so,sr,damping,update);
    sdft::apply_joint_secants(p,po,pr,damping,update);
    x={s[0],s[1],p[0],p[1]};
  }
  if(residual>1e-11){std::cerr<<"failed residual="<<residual<<"\n";return 1;}
  // Coefficients are real to preserve physical density constraints. Four
  // complex components therefore require eight real-independent secants.
  // Retain the old four-direction history as a negative span test.
  H full_r,full_out;
  Eigen::Vector4cd initial;
  for(int i=0;i<4;++i)initial[i]=root[i]+C(.7*(i+1),-.4*(i+2));
  for(int sample=0;sample<5;++sample) {
    Eigen::Vector4cd point=initial;
    if(sample<4)point[sample]+=C(.3,.2);
    const Eigen::Vector4cd res=(a*(point-Eigen::Map<const Eigen::Vector4cd>(root.data()))).eval();
    std::vector<C> rv(4),ov(4);
    for(int i=0;i<4;++i){rv[i]=res[i];ov[i]=point[i]+res[i];}
    full_r.push_back(rv);full_out.push_back(ov);
  }
  const auto full_update=sdft::joint_secant_coefficients(full_r,{},1.,0.);
  auto extrapolated=full_out.back();
  sdft::apply_joint_secants(extrapolated,full_out,full_r,1.,full_update);
  double incomplete_error=0.;
  for(int i=0;i<4;++i)incomplete_error=std::max(incomplete_error,std::abs(extrapolated[i]-root[i]));
  if(full_update.rank!=4||incomplete_error<1e-2)return 13;

  full_r.clear();full_out.clear();
  for(int sample=0;sample<9;++sample) {
    Eigen::Vector4cd point=initial;
    if(sample<8)point[sample/2]+=sample%2 ? C(0.,.3) : C(.3,0.);
    const Eigen::Vector4cd res=(a*(point-Eigen::Map<const Eigen::Vector4cd>(root.data()))).eval();
    std::vector<C> rv(4),ov(4);
    for(int i=0;i<4;++i){rv[i]=res[i];ov[i]=point[i]+res[i];}
    full_r.push_back(rv);full_out.push_back(ov);
  }
  // Independent unregularized realified QR oracle. It uses neither the
  // implementation's Gram matrix nor its eigensolver/filtering.
  Eigen::Matrix<double,8,8> directions;
  Eigen::Matrix<double,8,1> last_residual;
  for(int q=0;q<4;++q) {
    last_residual[2*q]=full_r.back()[q].real();
    last_residual[2*q+1]=full_r.back()[q].imag();
    for(int j=0;j<8;++j) {
      const C d=full_r[j+1][q]-full_r[j][q];
      directions(2*q,j)=d.real();directions(2*q+1,j)=d.imag();
    }
  }
  const auto qr=directions.colPivHouseholderQr();
  if(qr.rank()!=8)return 14;
  const Eigen::Matrix<double,8,1> oracle_coefficients=qr.solve(last_residual);
  auto oracle=full_out.back();
  for(int j=0;j<8;++j)for(int q=0;q<4;++q)
    oracle[q]-=oracle_coefficients[j]*(full_out[j+1][q]-full_out[j][q]);
  const auto spanning_update=sdft::joint_secant_coefficients(full_r,{},1.,0.);
  extrapolated=full_out.back();
  sdft::apply_joint_secants(extrapolated,full_out,full_r,1.,spanning_update);
  double error=0.,oracle_error=0.,agreement=0.;
  for(int i=0;i<4;++i) {
    error=std::max(error,std::abs(extrapolated[i]-root[i]));
    oracle_error=std::max(oracle_error,std::abs(oracle[i]-root[i]));
    agreement=std::max(agreement,std::abs(extrapolated[i]-oracle[i]));
  }
  if(spanning_update.rank!=8||oracle_error>1e-11||error>1e-10||agreement>1e-10) {
    std::cerr<<"full-image real-span error="<<error<<" QR error="<<oracle_error
             <<" agreement="<<agreement<<" rank="<<spanning_update.rank<<'\n';return 15;
  }
  std::cout<<"full-image real-span error="<<error<<" QR error="<<oracle_error
           <<" insufficient-span error="<<incomplete_error<<'\n';
  H duplicate{{C{1,2}}, {C{1,2}}};
  if(sdft::joint_secant_coefficients(duplicate,{},1,0).usable)return 2;
  try {sdft::joint_secant_coefficients(sr,H{{C{1,0}}},1,1);return 3;}
  catch(const std::invalid_argument&) {}
  H nonfinite{{C{1,0}},{C{NAN,0}}};
  try {sdft::joint_secant_coefficients(nonfinite,{},1,0);return 4;}
  catch(const std::invalid_argument&) {}
  std::cout<<"joint complex smooth/onsite affine response residual="<<residual<<'\n';
}
