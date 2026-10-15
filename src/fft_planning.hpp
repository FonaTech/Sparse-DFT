#pragma once
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <fftw3.h>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif
namespace sdft {
// FFTW execute/new-array execute is thread-safe. Its planner/destructor are
// global-state APIs and must share this lock across density/operator modules.
inline std::mutex& fft_planner_mutex(){static std::mutex mutex;return mutex;}
inline int fast_fft_length(int minimum){
  if(minimum<1)throw std::invalid_argument("invalid FFT length");
  for(int n=minimum;;++n){int remainder=n;
    for(int prime:{2,3,5,7})while(remainder%prime==0)remainder/=prime;
    if(remainder==1)return n;
    if(n==std::numeric_limits<int>::max())throw std::overflow_error("FFT length overflow");
  }
}
inline int density_fft_threads(std::size_t count){
  int threads=1;
#ifdef SDFT_HAS_OPENMP
  if(!omp_in_parallel()&&count>=262144)threads=omp_get_max_threads();
#endif
  if(const char* value=std::getenv("SDFT_DENSITY_FFT_THREADS")){
    std::size_t end=0;threads=std::stoi(value,&end);
    if(end!=std::string(value).size()||threads<1)throw std::invalid_argument("SDFT_DENSITY_FFT_THREADS must be a positive integer");
  }
#ifndef SDFT_HAS_FFTW_THREADS
  if(threads>1)throw std::runtime_error("threaded density FFT requested but FFTW threads library was not compiled");
#endif
  return threads;
}
// Call with the planner mutex held. Reset to one after plan creation so an
// independently parallel band team does not inherit nested FFT workers.
inline void set_fft_planner_threads(int threads){
#ifdef SDFT_HAS_FFTW_THREADS
  static bool ready=fftw_init_threads()!=0;
  if(!ready)throw std::runtime_error("FFTW thread runtime initialization failed");
  fftw_plan_with_nthreads(threads);
#else
  (void)threads;
#endif
}
}
