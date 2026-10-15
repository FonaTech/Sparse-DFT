#include "projector_derivative.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

int main() {
  using C=std::complex<double>;using M=Eigen::MatrixXcd;
  constexpr int p=7,n=43,bands=11;
  std::mt19937_64 random(543127);std::normal_distribution<double> normal;
  auto fill=[&](M& matrix){for(Eigen::Index i=0;i<matrix.size();++i)matrix.data()[i]=C(normal(random),normal(random))*.1;};
  M beta(p,n),coeff(n,bands),source(p,p),derivative(p,n);fill(beta);fill(coeff);fill(source);fill(derivative);
  M d=(source+source.adjoint()).eval(),q=(source*source.adjoint()).eval();
  Eigen::VectorXd eigen(bands),occupation(bands);
  for(int i=0;i<bands;++i){eigen[i]=-.8+.13*i;occupation[i]=i%3?.37*(i%5):0.;}
  auto energy=[&](const M& b){const M amplitude=b*coeff;double value=0;
    for(int i=0;i<bands;++i)value+=occupation[i]*std::real(amplitude.col(i).dot((d-eigen[i]*q)*amplitude.col(i)));
    return value;};
  for(int rep=0;rep<3;++rep) {
    if(rep){coeff*=C(.91,.13);d(0,0)+=.19;occupation[0]=.5;}
    const auto state=sdft::paw_derivative_response(p,n,bands,beta.data(),coeff.data(),d.data(),q.data(),eigen.data(),occupation.data());
    const M amplitudes=beta*coeff;
    for(int direction=0;direction<12;++direction) {
      fill(derivative);const M direct_derivative=derivative*coeff;double direct=0;
      for(int i=0;i<bands;++i)direct+=2*occupation[i]*std::real(direct_derivative.col(i).dot((d-eigen[i]*q)*amplitudes.col(i)));
      const double contracted=sdft::paw_derivative_contract(state,derivative.data());
      if(std::abs(contracted-direct)>1e-11){std::cerr<<"complex PAW derivative mismatch "<<contracted-direct<<'\n';return 1;}
      const double step=1e-4,finite=(energy(beta+step*derivative)-energy(beta-step*derivative))/(2*step);
      if(std::abs(finite-contracted)>1e-8){std::cerr<<"fixed-wavefunction finite-difference mismatch "<<finite-contracted<<'\n';return 2;}
    }
    std::vector<sdft::Vec3> wavevectors(n);
    std::array<M,3> momentum_gradients{M(p,n),M(p,n),M(p,n)};
    for(int g=0;g<n;++g)wavevectors[g]={.07*g-.9,.3*std::sin(.2*g),g%3?-.04*g:0.};
    for(auto& gradient:momentum_gradients)fill(gradient);
    const auto position=sdft::paw_derivative_position_contract(state,beta.data(),wavevectors);
    const auto strain=sdft::paw_derivative_strain_contract(state,beta.data(),
        {momentum_gradients[0].data(),momentum_gradients[1].data(),momentum_gradients[2].data()},wavevectors);
    for(int axis=0;axis<3;++axis) {
      M plus=beta,minus=beta;const double step=1e-4;
      for(int g=0;g<n;++g) {
        const auto v=wavevectors[g];const double q=axis==0?v.x:axis==1?v.y:v.z;
        plus.col(g)*=std::exp(C(0,q*step));minus.col(g)*=std::exp(C(0,-q*step));
      }
      const double actual=axis==0?position.x:axis==1?position.y:position.z;
      if(std::abs((energy(plus)-energy(minus))/(2*step)-actual)>1e-8) {
        std::cerr<<"fused position independent phase finite difference mismatch\n";return 4;
      }
      for(int component=0;component<3;++component) {
        M db=momentum_gradients[axis];
        for(int g=0;g<n;++g) {
          const auto v=wavevectors[g];const double q=component==0?v.x:component==1?v.y:v.z;
          db.col(g)*=-q;
        }
        if(axis==component)db-=.5*beta;
        const auto& row=strain[axis];const double value=component==0?row.x:component==1?row.y:row.z;
        if(std::abs(value-sdft::paw_derivative_contract(state,db.data()))>1e-11||
           std::abs((energy(beta+step*db)-energy(beta-step*db))/(2*step)-value)>1e-8) {
          std::cerr<<"fused strain contraction/finite difference mismatch\n";return 5;
        }
      }
    }
    bool rejected=false;
    try{sdft::paw_derivative_position_contract(state,nullptr,wavevectors);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)return 6;
    rejected=false;
    try{sdft::paw_derivative_strain_contract(state,beta.data(),{momentum_gradients[0].data(),nullptr,momentum_gradients[2].data()},wavevectors);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)return 7;
    wavevectors.pop_back();rejected=false;
    try{sdft::paw_derivative_position_contract(state,beta.data(),wavevectors);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)return 8;
  }
  occupation[0]=-1.;
  try{sdft::paw_derivative_response(p,n,bands,beta.data(),coeff.data(),d.data(),q.data(),eigen.data(),occupation.data());return 3;}
  catch(const std::invalid_argument&){}
  std::cout<<"PAW derivatives: complex D/Q, fractional/zero occupations, changing state, fused positions/strains and independent finite differences PASS\n";
}
