#include "fourier_state.hpp"
#include "onsite_density.hpp"
#include "fft_planning.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <limits>

int main() {
  using namespace sdft;using C=std::complex<double>;using M=Eigen::MatrixXcd;
  FourierDensity old{{{-2,0,0},C(.2,.3)},{{0,0,0},.5},{{4,1,-1},C(-.1,.02)}};
  FourierDensity next{{{-2,0,0},C(.21,.28)},{{-1,1,2},C(.1,-.2)},{{0,0,0},.5}};
  const auto merged=align_fourier_states(old,next);
  FourierDensity independent=old;for(const auto& [key,value]:next)independent.try_emplace(key,0.);
  if(merged.keys.size()!=independent.size())return 1;
  std::size_t index=0;double residual=0;
  for(const auto& [key,ignored]:independent) {
    const auto prior=old.count(key)?old.at(key):C{},out=next.count(key)?next.at(key):C{};
    if(merged.keys[index]!=key||merged.prior[index]!=prior||merged.output[index]!=out||
       merged.residual[index]!=out-prior)return 2;
    residual=std::max(residual,std::abs(out-prior));++index;
  }
  if(merged.maximum_residual!=residual||!align_fourier_states({},{}).keys.empty())return 3;
  constexpr int n=43,bands=11,rows=11;
  M b(rows,n),wave(n,bands);
  for(int i=0;i<b.size();++i)b.data()[i]=C(std::sin(.11*i),std::cos(.23*i))*.03;
  for(int i=0;i<wave.size();++i)wave.data()[i]=C(std::cos(.31*i),std::sin(.43*i))*.07;
  std::vector<double> occupations(bands);for(int i=0;i<bands;++i)occupations[i]=i%3?.19*(i%7):0.;
  for(int trial=0;trial<3;++trial) {
    if(trial){wave*=C(.93,.11);occupations[0]=.6;}
    M projected(rows,bands);
    complex_gemm(MatrixOp::Normal,MatrixOp::Normal,rows,bands,n,b.data(),rows,
        wave.data(),n,projected.data(),rows,true);
    int offset=0;
    for(int atom_rows:{7,4}) {
      const auto density=onsite_density_from_amplitudes(atom_rows,bands,
          projected.data()+offset,rows,occupations.data());
      const M direct=b.middleRows(offset,atom_rows)*wave;
      for(int i=0;i<atom_rows;++i)for(int j=0;j<atom_rows;++j) {
        C expected{};for(int band=0;band<bands;++band)
          expected+=occupations[band]*direct(i,band)*std::conj(direct(j,band));
        if(std::abs(density[i*atom_rows+j]-expected)>1e-12)return 4;
        if(std::abs(density[i*atom_rows+j]-std::conj(density[j*atom_rows+i]))>1e-12)return 5;
      }
      offset+=atom_rows;
    }
  }
  occupations[0]=-1;
  try{onsite_density_from_amplitudes(7,bands,b.data(),rows,occupations.data());return 6;}
  catch(const std::invalid_argument&){}
  for(int minimum:{3,46,62,89,117,157,178,234,314}) {
    const auto selected=fast_fft_length(minimum);if(selected<minimum)return 7;
    int remainder=selected;for(int factor:{2,3,5,7})while(remainder%factor==0)remainder/=factor;
    if(remainder!=1)return 8;
  }
  std::cout<<"Fourier linear merge and FP64 batched PAW onsite density PASS\n";
}
