#include "compute_backend.hpp"
#include "fft_planning.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc,char** argv){
  using namespace sdft;
  try{
    const bool benchmark=argc>1&&std::string(argv[1])=="--benchmark";
    const bool require_gpu=argc>1&&std::string(argv[1])=="--require-gpu";
    const int extent=benchmark&&argc>2?std::stoi(argv[2]):0;
    const std::size_t columns=benchmark&&argc>3?std::stoull(argv[3]):5;
    if(benchmark&&(extent<4||!columns))throw std::invalid_argument("invalid FFT benchmark extent/columns");
    const std::vector<std::array<int,3>> shapes=benchmark?std::vector<std::array<int,3>>{{extent,extent,extent}}:
        std::vector<std::array<int,3>>{{8,15,14},{27,25,16},{4,6,7}};
    double maximum=0;std::cout<<"{\"scope\":\"batched_local_operator_only\",\"backend\":\""<<compute_backend_name()<<"\",\"cases\":[";bool comma=false;
    for(const auto& shape:shapes){
      const std::size_t count=std::size_t(shape[0])*shape[1]*shape[2],npw=benchmark?std::min<std::size_t>(8192,count/4):73;
      const auto index=[&](int x,int y,int z){
        auto wrap=[](int a,int b){return (a%b+b)%b;};
        return (std::size_t(wrap(x,shape[0]))*shape[1]+wrap(y,shape[1]))*shape[2]+wrap(z,shape[2]);
      };
      std::vector<std::size_t> indices(npw);std::vector<int> lookup(count,-1);std::vector<double> potential(count),kinetic(npw);
      for(std::size_t g=0;g<npw;++g){indices[g]=g;lookup[g]=int(g);kinetic[g]=.013*(1+g%19);}
      for(int x=0;x<shape[0];++x)for(int y=0;y<shape[1];++y)for(int z=0;z<shape[2];++z){
        const double phase=6.2831853071795864769*(double(x)/shape[0]+2.*y/shape[1]);
        potential[index(x,y,z)]=.4+.13*std::cos(phase)+.07*std::sin(12.5663706143591729538*z/shape[2]);
      }
      std::vector<Complex64> input(npw*columns),expected(input.size()),candidate(input.size()),cpu(input.size());
      for(std::size_t b=0;b<columns;++b)for(std::size_t g=0;g<npw;++g)
        input[b*npw+g]={std::sin(.17*(g+1)*(b+2)),std::cos(.13*(g+2)*(b+1))};
      for(std::size_t b=0;b<columns;++b)for(std::size_t g=0;g<npw;++g){
        const int z=indices[g]%shape[2],y=(indices[g]/shape[2])%shape[1],x=indices[g]/shape[2]/shape[1];
        const auto at=[&](int a,int c,int d){const int j=lookup[index(a,c,d)];return j<0?Complex64{}:input[b*npw+j];};
        expected[b*npw+g]=(kinetic[g]+.4)*input[b*npw+g]+.065*(at(x-1,y-2,z)+at(x+1,y+2,z))+
            Complex64(0,-.035)*at(x,y,z-2)+Complex64(0,.035)*at(x,y,z+2);
      }
      auto* work=fftw_alloc_complex(count*columns);if(!work)throw std::bad_alloc();
      fftw_plan forward{},backward{};
      {std::lock_guard lock(fft_planner_mutex());set_fft_planner_threads(density_fft_threads(count*columns));int dims[]{shape[0],shape[1],shape[2]};
        forward=fftw_plan_many_dft(3,dims,int(columns),work,nullptr,1,int(count),work,nullptr,1,int(count),FFTW_FORWARD,FFTW_ESTIMATE);
        backward=fftw_plan_many_dft(3,dims,int(columns),work,nullptr,1,int(count),work,nullptr,1,int(count),FFTW_BACKWARD,FFTW_ESTIMATE);set_fft_planner_threads(1);}
      if(!forward||!backward)throw std::runtime_error("FFT test planning failed");
      const auto cpu_apply=[&](){
        std::fill_n(&work[0][0],2*count*columns,0.);
        for(std::size_t b=0;b<columns;++b)for(std::size_t g=0;g<npw;++g){const auto i=b*count+indices[g];work[i][0]=input[b*npw+g].real();work[i][1]=input[b*npw+g].imag();}
        fftw_execute(backward);
        for(std::size_t i=0;i<count*columns;++i){work[i][0]*=potential[i%count];work[i][1]*=potential[i%count];}
        fftw_execute(forward);
        for(std::size_t b=0;b<columns;++b)for(std::size_t g=0;g<npw;++g){const auto i=b*count+indices[g];cpu[b*npw+g]=kinetic[g]*input[b*npw+g]+Complex64(work[i][0],work[i][1])/double(count);}
      };
      bool accelerated = false;
      const auto gpu_apply=[&](){
        accelerated = accelerated_local_hamiltonian(shape,indices,kinetic,potential,input.data(),columns,candidate.data(),false);
        if(!accelerated){cpu_apply();candidate=cpu;}
      };
      cpu_apply();gpu_apply();
      if(require_gpu&&!accelerated)throw std::runtime_error("real GPU FFT dispatch was required but not performed");
      const auto measure=[&](auto action){std::vector<double> elapsed;for(int repeat=0;repeat<(benchmark?3:1);++repeat){const auto start=std::chrono::steady_clock::now();action();elapsed.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());}std::sort(elapsed.begin(),elapsed.end());return elapsed[elapsed.size()/2];};
      const double cpu_seconds=measure(cpu_apply),gpu_seconds=measure(gpu_apply);
      double cpu_error=0,gpu_error=0,scale=1;
      for(std::size_t i=0;i<input.size();++i){cpu_error=std::max(cpu_error,std::abs(cpu[i]-expected[i]));gpu_error=std::max(gpu_error,std::abs(candidate[i]-expected[i]));scale=std::max(scale,std::abs(expected[i]));}
      maximum=std::max({maximum,cpu_error/scale,gpu_error/scale});
      std::cout<<(comma?",":"")<<"{\"shape\":["<<shape[0]<<','<<shape[1]<<','<<shape[2]<<"],\"columns\":"<<columns
          <<",\"cpu_seconds\":"<<cpu_seconds<<",\"candidate_seconds\":"<<gpu_seconds<<",\"accelerated\":"<<(accelerated?"true":"false")<<",\"speedup\":";
      if(accelerated)std::cout<<cpu_seconds/gpu_seconds;else std::cout<<"null";
      std::cout
          <<",\"cpu_scaled_error\":"<<cpu_error/scale<<",\"candidate_scaled_error\":"<<gpu_error/scale<<"}";comma=true;
      {std::lock_guard lock(fft_planner_mutex());fftw_destroy_plan(forward);fftw_destroy_plan(backward);}fftw_free(work);
    }
    std::cout<<"],\"maximum_scaled_error\":"<<maximum<<",\"limit\":2e-11,\"passed\":"<<(maximum<2e-11?"true":"false")<<"}\n";
    write_performance_report("fft_backend.performance.json");return maximum<2e-11?0:3;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
