#include "reciprocal_density_table.hpp"
#include <iostream>
#include <limits>

int main() {
  const double qmax=8.;const int n=32;
  auto f=[](double q){return 3.+.2*q-.4*q*q+.03*q*q*q;};
  auto df=[](double q){return .2-.8*q+.09*q*q;};
  std::vector<double> table(n);
  for(int i=0;i<n;++i)table[i]=f(i*qmax/n);
  double value_error=0,derivative_error=0,fd_error=0;
  for(double q:{.001,.07,.249,.251,.811,2.19,5.03,6.97}) {
    const auto sample=sdft::reciprocal_density_sample(table,q,qmax);
    value_error=std::max(value_error,std::abs(sample.value-f(q)));
    derivative_error=std::max(derivative_error,std::abs(sample.derivative-df(q)));
    const double step=1e-6;
    const double fd=(sdft::reciprocal_density_sample(table,q+step,qmax).value-
        sdft::reciprocal_density_sample(table,q-step,qmax).value)/(2*step);
    fd_error=std::max(fd_error,std::abs(fd-sample.derivative));
  }
  // Derivatives also need to differentiate the actual non-polynomial table,
  // rather than an independently smoothed or resampled approximation.
  for(int i=0;i<n;++i)table[i]=std::exp(-.3*i*qmax/n);
  for(double q:{.01,.19,.26,1.07,4.21,7.01}) {
    const double step=1e-6;
    const auto sample=sdft::reciprocal_density_sample(table,q,qmax);
    const double fd=(sdft::reciprocal_density_sample(table,q+step,qmax).value-
        sdft::reciprocal_density_sample(table,q-step,qmax).value)/(2*step);
    fd_error=std::max(fd_error,std::abs(fd-sample.derivative));
  }
  const auto zero=sdft::reciprocal_density_sample(table,0,qmax);
  const auto guard=sdft::reciprocal_density_sample(table,qmax*(1.-3./n),qmax);
  bool rejects_nan=false;
  try {sdft::reciprocal_density_sample(table,std::numeric_limits<double>::quiet_NaN(),qmax);}
  catch(const std::invalid_argument&) {rejects_nan=true;}
  std::cout<<"reciprocal_density value_error="<<value_error<<" derivative_error="<<derivative_error
      <<" fd_error="<<fd_error<<'\n';
  return value_error<1e-12 && derivative_error<1e-12 && fd_error<5e-9 &&
      zero.value==table.front() && zero.derivative==0 && guard.value==0 && guard.derivative==0 && rejects_nan ? 0:1;
}
