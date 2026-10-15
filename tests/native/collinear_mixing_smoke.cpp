#include "collinear_mixing.hpp"
#include "scf_inner_accuracy.hpp"
#include <iostream>

int main(){
  using C=std::complex<double>;using H=sdft::JointSecants::History;
  // A spin label swap leaves charge invariant and reverses magnetization.
  // Use heterogeneous flattened atom blocks (1x1 and 2x2) and imaginary
  // coherences, rather than identical scalar spin densities.
  const std::vector<C> spin{{.8,0},{.4,0},{.1,.2},{.1,-.2},{.6,0},
                            {.2,0},{.3,0},{.04,-.1},{.04,.1},{.2,0}};
  auto cm=sdft::collinear_charge_magnetization(spin);
  auto back=sdft::collinear_spin_blocks(cm);
  for(std::size_t q=0;q<spin.size();++q)if(std::abs(back[q]-spin[q])>1e-14)return 1;
  auto swap=spin;std::rotate(swap.begin(),swap.begin()+5,swap.end());
  const auto swapped=sdft::collinear_charge_magnetization(swap);
  for(std::size_t q=0;q<5;++q)
    if(std::abs(swapped[q]-cm[q])>1e-14||std::abs(swapped[q+5]+cm[q+5])>1e-14)return 2;
  if(std::abs(sdft::kerker_response(4.,.4,1.,.1)-.32)>1e-14||
     std::abs(sdft::kerker_response(.01,.4,1.,.1)-.1)>1e-14||
     std::abs(sdft::kerker_response(4.,1.6,1.,.1)-1.28)>1e-14)return 3;
  // Independent affine-response oracle. Four independent real secants
  // span a coupled charge/magnetic/onsite model; a diagonal B0 with distinct
  // responses must reach the same exact root. Compute coefficients by QR
  // without using the implementation's Gram/eigensolver filter.
  Eigen::Matrix4d response;
  response<<-3.,.2,.1,0., .2,-.8,.1,.1, .1,.1,-2.,.3, 0.,.1,.3,-.5;
  const Eigen::Vector4d root(.2,-.7,.4,.9),initial(1.,.5,-.3,.1);
  H residuals,outputs;
  for(int sample=0;sample<5;++sample){
    Eigen::Vector4d x=initial;if(sample<4)x[sample]+=.3;
    const Eigen::Vector4d r=response*(x-root);
    std::vector<C> rv(4),ov(4);for(int q=0;q<4;++q){rv[q]=r[q];ov[q]=x[q]+r[q];}
    residuals.push_back(rv);outputs.push_back(ov);
  }
  Eigen::Matrix4d differences;Eigen::Vector4d last;
  for(int q=0;q<4;++q){last[q]=residuals.back()[q].real();
    for(int i=0;i<4;++i)differences(q,i)=(residuals[i+1][q]-residuals[i][q]).real();}
  const Eigen::Vector4d c=differences.colPivHouseholderQr().solve(last);
  sdft::JointSecants exact;exact.usable=true;exact.rank=4;
  exact.coefficients.assign(c.data(),c.data()+4);
  std::vector<C> prior(4);for(int q=0;q<4;++q)prior[q]=initial[q];
  const std::vector<double> b0{.1,.8,.2,1.2};
  const auto image=sdft::preconditioned_secant_image(prior,outputs,residuals,b0,exact);
  double error=0.;for(int q=0;q<4;++q)error=std::max(error,std::abs(image[q]-C(root[q])));
  if(error>1e-12){std::cerr<<"B0 affine root error "<<error<<'\n';return 4;}
  auto metric=residuals;for(auto& row:metric)for(int q=0;q<4;++q)row[q]*=std::sqrt(q==0?20.:1.);
  const auto secants=sdft::joint_secant_coefficients(metric,{},1.,0.);
  const auto regularized=sdft::preconditioned_secant_image(prior,outputs,residuals,b0,secants);
  double agreement=0.;for(int q=0;q<4;++q)agreement=std::max(agreement,std::abs(regularized[q]-image[q]));
  if(secants.rank!=4||agreement>1e-5)return 5;
  // No usable history still has the actual coordinate response, including
  // magnetic AMIX>1. Rejecting all such responses would break the plain
  // Kerker-preconditioned linear step.
  const auto linear=sdft::preconditioned_secant_image(prior,outputs,residuals,b0,{});
  for(int q=0;q<4;++q)
    if(std::abs(linear[q]-(prior[q]+b0[q]*residuals.back()[q]))>1e-14)return 6;
  try{sdft::collinear_spin_blocks({C(1.)});return 7;}catch(const std::invalid_argument&){}
  try{sdft::kerker_response(1.,.4,-1.,.1);return 8;}catch(const std::invalid_argument&){}
  {
    const std::vector<C> ps{2.,.4},pp{.2,0.,0.,.4},ls{3.,.8},lp{-.8,0.,0.,.6};
    auto s=ls,p=lp;double positivity=0.;
    if(!sdft::finalize_preconditioned_spin_step(s,p,ps,pp,ls,lp,{2},1.,positivity)||
       positivity!=.125||std::abs(p[0]-.075)>1e-14||std::abs(s[1]-.45)>1e-14)return 9;
    auto other_s=ls,other_p=lp;
    if(sdft::finalize_preconditioned_spin_step(other_s,other_p,ps,pp,ls,lp,{2},.125,positivity)||
       positivity!=1.||std::abs(other_p[0]-.075)>1e-14)return 10;
  }
  {
    // Independently derived nullspace counterexample: both prior spin
    // matrices are zero in this orbital; occupied output is up=1,down=0.
    // charge AMIX=.2, magnetic AMIX=.8 gives du=.5,dd=-.3. Every positive
    // scalar step still has a negative down eigenvalue. A convex update of
    // the physical endpoint gives du=.2,dd=0 and preserves spin covariance.
    const std::vector<C> ps{2.,0.},pp{0.,0.},os{3.,1.},op{1.,0.};
    const std::vector<C> ls{2.2,.8},lp{.5,-.3};
    auto s=ls,p=lp;double positivity=0.;bool convex=false;
    if(!sdft::finalize_preconditioned_spin_step(s,p,ps,pp,ls,lp,{1,1},1.,positivity,
        os,op,.2,&convex)||!convex||std::abs(p[0]-.2)>1e-14||std::abs(p[1])>1e-14||
        std::abs(s[1]-.2)>1e-14)return 11;
    auto swapped_p=std::vector<C>{-.3,.5},swapped_s=std::vector<C>{2.2,-.8};
    const std::vector<C> swapped_o{0.,1.},swapped_os{3.,-1.};
    const auto swapped_lp=swapped_p,swapped_ls=swapped_s;
    if(!sdft::finalize_preconditioned_spin_step(swapped_s,swapped_p,ps,pp,
        swapped_ls,swapped_lp,{1,1},1.,positivity,swapped_os,swapped_o,.2,&convex)||!convex||
        std::abs(swapped_p[1]-.2)>1e-14||std::abs(swapped_s[1]+.2)>1e-14)return 12;
  }
  {
    // An independent inexact fixed-point model: F(x)=x/2, with alternating
    // eigensolver error amplified into an occupation coordinate. A request
    // derived from the loose user threshold cannot resolve the strict outer
    // margin; final refinement must remove that floor without weakening it.
    auto solve=[](bool use_refinement) {
      sdft::CollinearInnerAccuracy accuracy(1e-10,1e-10,true);
      double x=.01, residual=1.;bool converged=false;
      for(int iteration=0;iteration<200;++iteration) {
        accuracy.observe(residual,residual);
        const double tolerance=use_refinement ? accuracy.requested(residual,1e-4,false) :
            std::clamp(1e-4*residual,1e-12,1e-4);
        const double output=.5*x+(iteration%2?1.:-1.)*400.*tolerance;
        residual=std::abs(output-x);x=output;
        if(residual<1e-10 && std::abs(x)<1e-10){converged=true;break;}
      }
      return converged;
    };
    if(solve(false)||!solve(true))return 13;
    sdft::CollinearInnerAccuracy accuracy(1e-10,1e-10,true);
    if(accuracy.observe(1e-9,1e-5)||accuracy.final_refinement())return 14;
    if(!accuracy.observe(1e-9,1e-8)||!accuracy.final_refinement())return 15;
    accuracy.observe(1.,1.); // Hysteresis: a late excursion never coarsens H/S.
    if(accuracy.requested(1.,1e-4,false)!=5e-14)return 16;
    try{accuracy.observe(NAN,0);return 17;}catch(const std::invalid_argument&){}
  }
  std::cout<<"spin swap covariance, Kerker response, inexact-SCF precision and independent B0 QR oracle PASS; error="
      <<error<<" regularized agreement="<<agreement<<'\n';
}
