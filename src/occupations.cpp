#include "occupations.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>

namespace sdft {
OccupationResult fill_occupations(const std::vector<std::vector<double>>& energies,
    const std::vector<double>& weights,double electrons,double sigma) {
  if(energies.empty()||weights.size()!=energies.size()||!std::isfinite(electrons)||electrons<=0||
     !std::isfinite(sigma)||sigma<0)throw std::invalid_argument("invalid global occupations input");
  struct Level {double energy;std::size_t k,b;};
  std::vector<Level> levels;double capacity=0,weight_sum=0;
  OccupationResult out;out.occupations.resize(energies.size());
  for(std::size_t k=0;k<energies.size();++k) {
    if(!std::isfinite(weights[k])||weights[k]<=0||energies[k].empty())
      throw std::invalid_argument("invalid k-point weight or empty spectrum");
    weight_sum+=weights[k];capacity+=2*weights[k]*energies[k].size();
    out.occupations[k].resize(energies[k].size());
    for(std::size_t b=0;b<energies[k].size();++b) {
      if(!std::isfinite(energies[k][b]))throw std::invalid_argument("nonfinite occupation eigenvalue");
      levels.push_back({energies[k][b],k,b});
    }
  }
  if(std::abs(weight_sum-1)>1e-12||electrons>capacity+1e-12)
    throw std::invalid_argument("k weights must sum to one and bands must hold all electrons");
  std::sort(levels.begin(),levels.end(),[](const auto& a,const auto& b){return a.energy<b.energy;});
  if(sigma==0) {
    double remaining=electrons;
    for(std::size_t first=0;first<levels.size()&&remaining>1e-13;) {
      std::size_t last=first+1;double group_capacity=2*weights[levels[first].k];
      while(last<levels.size()&&std::abs(levels[last].energy-levels[first].energy)<1e-10) {
        group_capacity+=2*weights[levels[last].k];++last;
      }
      const double fraction=std::min(1.,remaining/group_capacity);
      for(std::size_t i=first;i<last;++i)out.occupations[levels[i].k][levels[i].b]=2*fraction;
      out.chemical_potential_ev=levels[first].energy;
      remaining-=fraction*group_capacity;first=last;
    }
  } else {
    if(capacity-electrons<1e-8)throw std::invalid_argument("finite-temperature occupations require empty bands");
    const auto fermi=[&](double e,double mu) {
      const double x=(e-mu)/sigma;
      return x>=0?std::exp(-x)/(1+std::exp(-x)):1/(1+std::exp(x));
    };
    double lo=levels.front().energy-80*sigma-1,hi=levels.back().energy+80*sigma+1;
    // Integer filling: the lowest `filled` levels hold exactly N_e and the next
    // level lies higher, so the spectrum has a gap at this electron count.
    std::size_t filled=0;double below=0;
    while(filled<levels.size()&&below<electrons-1e-10)below+=2*weights[levels[filled++].k];
    const bool gapped=filled>0&&filled<levels.size()&&std::abs(below-electrons)<=1e-10&&
        levels[filled].energy-levels[filled-1].energy>1e-10;
    if(gapped) {
      // N(mu)-N_e = (electrons above the gap) - (holes below it), exactly. In the gap
      // both tails fall below 1e-16 N_e, so a direct count rounds to N_e on a wide
      // plateau and bisection stops at its edge. Summing each tail in log space keeps
      // the balance resolved for any gap and width; ln f(x) = -softplus(x).
      const double minus_inf=-std::numeric_limits<double>::infinity();
      const auto log_fermi=[](double x){return -(std::max(x,0.)+std::log1p(std::exp(-std::abs(x))));};
      const auto log_add=[&](double a,double b){
        if(a<b)std::swap(a,b);
        return b==minus_inf?a:a+std::log1p(std::exp(b-a));
      };
      std::vector<double> log_capacity(levels.size());
      for(std::size_t i=0;i<levels.size();++i)log_capacity[i]=std::log(2*weights[levels[i].k]);
      const auto log_balance=[&](double mu) {  // ln(electrons) - ln(holes), increasing in mu
        double electrons_above=minus_inf,holes_below=minus_inf;
        for(std::size_t i=0;i<levels.size();++i) {
          const double x=(levels[i].energy-mu)/sigma;
          if(i<filled)holes_below=log_add(holes_below,log_capacity[i]+log_fermi(-x));
          else electrons_above=log_add(electrons_above,log_capacity[i]+log_fermi(x));
        }
        return electrons_above-holes_below;
      };
      for(int i=0;i<200;++i) {
        const double mu=(lo+hi)*.5;
        if(mu<=lo||mu>=hi)break;  // interval at machine resolution
        if(log_balance(mu)<0)lo=mu;else hi=mu;
      }
    } else {
      for(int i=0;i<200;++i) {
        const double mu=(lo+hi)*.5;double count=0;
        for(const auto& level:levels)count+=2*weights[level.k]*fermi(level.energy,mu);
        if(count<electrons)lo=mu;else hi=mu;
      }
    }
    out.chemical_potential_ev=(lo+hi)*.5;
    for(const auto& level:levels) {
      const double f=fermi(level.energy,out.chemical_potential_ev);
      out.occupations[level.k][level.b]=2*f;
      if(f>0&&f<1)out.entropy_energy_ev+=2*sigma*weights[level.k]*(f*std::log(f)+(1-f)*std::log1p(-f));
    }
    for(std::size_t k=0;k<energies.size();++k)
      if(out.occupations[k].back()>1e-6)
        throw InsufficientEmptyBands(out.occupations[k].size());
  }
  for(std::size_t k=0;k<energies.size();++k)
    out.electron_count+=weights[k]*std::accumulate(out.occupations[k].begin(),out.occupations[k].end(),0.);
  if(std::abs(out.electron_count-electrons)>1e-10)
    throw std::runtime_error("global occupation electron conservation failed");
  return out;
}
OccupationResult fill_collinear_occupations(const std::vector<std::vector<double>>& energies,
    const std::vector<double>& weights,double electrons,double sigma){
  if(energies.size()!=2*weights.size())throw std::invalid_argument("collinear spectra need two channels");
  std::vector<double> expanded;for(int s=0;s<2;++s)for(double w:weights)expanded.push_back(.5*w);
  auto out=fill_occupations(energies,expanded,electrons,sigma);
  for(auto& k:out.occupations)for(auto& f:k)f*=.5;
  return out;
}

}
