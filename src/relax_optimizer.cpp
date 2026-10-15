#include "relax_optimizer.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace sdft {
namespace {
double dot(const std::vector<double>& a,const std::vector<double>& b) {
  return std::inner_product(a.begin(),a.end(),b.begin(),0.);
}
bool finite(const std::vector<double>& a) {
  return std::all_of(a.begin(),a.end(),[](double x){return std::isfinite(x);});
}
}
RelaxDirection lbfgs_direction(const std::vector<double>& g,const std::vector<LbfgsPair>& history,
                              double initial_scale,double max_step) {
  if(g.empty()||!finite(g)||!std::isfinite(initial_scale)||initial_scale<=0||
      !std::isfinite(max_step)||max_step<=0)
    throw std::invalid_argument("invalid L-BFGS gradient or step controls");
  RelaxDirection out;
  std::vector<const LbfgsPair*> pairs;
  std::vector<double> rho;
  for(const auto& pair:history) {
    const auto& s=pair.displacement;const auto& y=pair.gradient_change;
    if(s.size()!=g.size()||y.size()!=g.size()||!finite(s)||!finite(y))
      throw std::invalid_argument("invalid L-BFGS curvature pair");
    const double sy=dot(s,y), norm=std::sqrt(dot(s,s))*std::sqrt(dot(y,y));
    if(!std::isfinite(sy)||!std::isfinite(norm)||sy<=1e-10*norm||sy<=0) {
      ++out.rejected_pairs;continue;
    }
    pairs.push_back(&pair);rho.push_back(1./sy);
  }
  out.used_pairs=pairs.size();
  std::vector<double> q=g,alpha(pairs.size());
  for(std::size_t i=pairs.size();i-->0;) {
    alpha[i]=rho[i]*dot(pairs[i]->displacement,q);
    for(std::size_t j=0;j<g.size();++j)q[j]-=alpha[i]*pairs[i]->gradient_change[j];
  }
  double scale=initial_scale;
  if(!pairs.empty()) {
    const auto& last=*pairs.back();scale=dot(last.displacement,last.gradient_change)/dot(last.gradient_change,last.gradient_change);
    scale=std::clamp(scale,1e-6,1e6);
  }
  for(auto& x:q)x*=scale;
  for(std::size_t i=0;i<pairs.size();++i) {
    const double beta=rho[i]*dot(pairs[i]->gradient_change,q);
    for(std::size_t j=0;j<g.size();++j)q[j]+=pairs[i]->displacement[j]*(alpha[i]-beta);
  }
  out.step=q;for(auto& x:out.step)x=-x;
  if(!finite(out.step)||dot(g,out.step)>=0) {
    for(std::size_t j=0;j<g.size();++j)out.step[j]=-initial_scale*g[j];
    out.steepest_descent=true;
  }
  const double norm=std::sqrt(dot(out.step,out.step));
  if(!std::isfinite(norm))throw std::runtime_error("L-BFGS step overflow");
  if(norm>max_step)for(auto& x:out.step)x*=max_step/norm;
  out.directional_derivative=dot(g,out.step);
  return out;
}
}
