#include "compute_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
int main(){
  using namespace sdft;
  try{
    constexpr std::size_t count=131072;
    std::vector<double> n(count),g(count);
    std::vector<PbeXcPoint> ref(count),candidate;
    for(std::size_t i=0;i<count;++i){n[i]=std::exp(-23.025850929940457+27.630*double(i%10007)/10006.);
      g[i]=i%7==0?std::pow(10.,-9.+11.*double(i%31)/30.):n[i]*double(i%101)/3.;}
    std::vector<double> cpu_times,gpu_times;bool accelerated=false;
    for(int rep=0;rep<3;++rep){
      auto start=std::chrono::steady_clock::now();
      for(std::size_t i=0;i<count;++i)ref[i]=pbe_unpolarized(n[i],g[i]);
      cpu_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
      start=std::chrono::steady_clock::now();accelerated=accelerated_pbe_points(n,g,candidate);
      if(!accelerated){candidate.resize(count);for(std::size_t i=0;i<count;++i)candidate[i]=pbe_unpolarized(n[i],g[i]);}
      gpu_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    }
    double errors[6]{};std::size_t worst_gradient=0;
    for(std::size_t i=0;i<count;++i){const auto& a=ref[i];const auto& b=candidate[i];
      const double expected[]{a.energy_density,a.exchange_energy_density,a.correlation_energy_density,a.exchange_potential,a.correlation_potential,a.gradient_squared_coefficient};
      const double actual[]{b.energy_density,b.exchange_energy_density,b.correlation_energy_density,b.exchange_potential,b.correlation_potential,b.gradient_squared_coefficient};
      for(int field=0;field<6;++field){const auto error=std::abs(expected[field]-actual[field])/std::max(1.,std::abs(expected[field]));
        if(!std::isfinite(error))throw std::runtime_error("nonfinite PBE backend result");
        if(field==5&&error>errors[field])worst_gradient=i;errors[field]=std::max(errors[field],error);}
    }
    double fd_error=0;
    if(accelerated){
      std::vector<double> pn,pg;const double delta=1e-5;
      for(double density:{.01,.1,.5,2.})for(double gradient:{.01,.1,1.}){
        for(auto point:{std::pair{density,gradient},std::pair{density+delta,gradient},std::pair{density-delta,gradient},
                       std::pair{density,gradient+delta},std::pair{density,gradient-delta}}){pn.push_back(point.first);pg.push_back(point.second);}}
      std::vector<PbeXcPoint> samples;accelerated_pbe_points(pn,pg,samples);
      for(std::size_t i=0;i<samples.size();i+=5){
        const double dn=(samples[i+1].energy_density-samples[i+2].energy_density)/(2*delta);
        const double dg=(samples[i+3].energy_density-samples[i+4].energy_density)/(2*delta);
        fd_error=std::max({fd_error,std::abs(dn-samples[i].exchange_potential-samples[i].correlation_potential),
                          std::abs(dg-samples[i].gradient_squared_coefficient*pg[i])});}
    }
    std::sort(cpu_times.begin(),cpu_times.end());std::sort(gpu_times.begin(),gpu_times.end());
    const double maximum=*std::max_element(errors,errors+6);const bool passed=maximum<2e-11&&fd_error<1e-7;
    std::cout<<std::setprecision(17)<<"{\"scope\":\"PBE point kernel only; not full SCF\",\"backend\":\""<<compute_backend_name()
      <<"\",\"accelerated\":"<<(accelerated?"true":"false")<<",\"points\":"<<count
      <<",\"cpu_serial_seconds\":"<<cpu_times[1]<<",\"candidate_seconds\":"<<gpu_times[1]<<",\"speedup\":";
    if(accelerated)std::cout<<cpu_times[1]/gpu_times[1];else std::cout<<"null";
    std::cout<<",\"scaled_errors\":[";for(int i=0;i<6;++i)std::cout<<(i?",":"")<<errors[i];
    std::cout<<"],\"worst_gradient_point\":{\"density\":"<<n[worst_gradient]<<",\"gradient\":"<<g[worst_gradient]
      <<",\"reference\":"<<ref[worst_gradient].gradient_squared_coefficient<<",\"candidate\":"<<candidate[worst_gradient].gradient_squared_coefficient<<"},\"maximum_scaled_error\":"<<maximum<<",\"finite_difference_error\":"<<fd_error
      <<",\"passed\":"<<(passed?"true":"false")<<"}\n";
    write_performance_report("pbe_backend.performance.json");return passed?0:3;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
