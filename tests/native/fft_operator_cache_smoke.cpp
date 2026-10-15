#include "env_cache.hpp"
#include "compute_backend.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  using namespace sdft;
  try {
    const bool required=argc>1&&std::string(argv[1])=="--require-gpu";
    if(compute_backend_name()=="cpu") {
      if(required)throw std::runtime_error("real GPU immutable FFT test required");
      std::cout<<"{\"executed_gpu\":false,\"reason\":\"CPU-only regression; real GPU test is separate\"}\n";
      return 0;
    }
    setenv("SDFT_GPU_FFT","1",1);sdft::refresh_environment_cache();setenv("SDFT_GPU_FFT_MIN_POINTS","0",1);sdft::refresh_environment_cache();
    setenv("SDFT_GPU_FFT_IMMUTABLE_CACHE","1",1);sdft::refresh_environment_cache();
    const std::array<int,3> shape{8,15,14};
    const std::size_t count=8*15*14,npw=73,columns=5;
    const auto index=[&](int x,int y,int z) {
      auto wrap=[](int a,int b){return (a%b+b)%b;};
      return (std::size_t(wrap(x,shape[0]))*shape[1]+wrap(y,shape[1]))*shape[2]+wrap(z,shape[2]);
    };
    std::vector<std::size_t> indices(npw);for(std::size_t i=0;i<npw;++i)indices[i]=i;
    std::vector<double> kinetic(npw),potential(count);
    std::vector<Complex64> input(npw*columns),output(input.size());
    double maximum=0.;std::uint64_t token=0;
    for(int variant=0;variant<8;++variant) {
      // Repeat immutable geometry with a new dynamic vector, then switch
      // potential/indices at identical dimensions using a fresh lifetime.
      if(variant%2==0)token=new_immutable_matrix_identity();
      if(variant==4)for(auto& i:indices)i=(i+19)%count;
      const double average=.4+.07*(variant/2);
      std::vector<int> lookup(count,-1);for(std::size_t g=0;g<npw;++g)lookup[indices[g]]=int(g);
      for(int x=0;x<shape[0];++x)for(int y=0;y<shape[1];++y)for(int z=0;z<shape[2];++z)
        potential[index(x,y,z)]=average+.13*std::cos(6.2831853071795864769*(double(x)/shape[0]+2.*y/shape[1]));
      for(std::size_t g=0;g<npw;++g)kinetic[g]=.01*(1+g%13);
      for(std::size_t i=0;i<input.size();++i)input[i]={std::sin(.17*(i+1)*(variant+1)),std::cos(.13*(i+2))};
      if(!accelerated_local_hamiltonian(shape,indices,kinetic,potential,input.data(),columns,output.data(),false,token))
        throw std::runtime_error("immutable FFT GPU operation not executed");
      for(std::size_t b=0;b<columns;++b)for(std::size_t g=0;g<npw;++g) {
        const int z=indices[g]%shape[2],y=(indices[g]/shape[2])%shape[1],x=indices[g]/shape[2]/shape[1];
        const auto at=[&](int xx,int yy){const int j=lookup[index(xx,yy,z)];return j<0?Complex64{}:input[b*npw+j];};
        const auto expected=(kinetic[g]+average)*input[b*npw+g]+.065*(at(x-1,y-2)+at(x+1,y+2));
        maximum=std::max(maximum,std::abs(output[b*npw+g]-expected)/std::max(1.,std::abs(expected)));
      }
    }
    // A token-less caller is still content-checked when it changes V(r).
    for(auto& v:potential)v+=.125;
    auto previous=output;
    if(!accelerated_local_hamiltonian(shape,indices,kinetic,potential,input.data(),columns,output.data(),false))
      throw std::runtime_error("legacy content-check FFT GPU operation not executed");
    for(std::size_t i=0;i<input.size();++i)
      maximum=std::max(maximum,std::abs(output[i]-previous[i]-.125*input[i]));
    if(maximum>2e-11)throw std::runtime_error("immutable operator cache failed independent Fourier convolution oracle");
    write_performance_report("fft_operator_cache.performance.json");
    std::cout<<std::setprecision(17)<<"{\"executed_gpu\":true,\"operations\":9,\"maximum_scaled_error\":"<<maximum
      <<",\"limit\":2e-11,\"passed\":true}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
