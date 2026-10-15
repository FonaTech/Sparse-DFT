#include "density_json.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

int main() {
  using C=std::complex<double>;
  std::map<std::tuple<int,int,int>,C> values;
  values[{-9,3,-1}]=C(-0.,std::numeric_limits<double>::denorm_min());
  values[{0,0,0}]=C(1.234567890123456e200,-1.234567890123456e-210);
  for(int i=1;i<100000;++i)values[{i%101,i/101,-i%37}]=C(std::sin(i)*.1,std::cos(i)*.01);
  std::ostringstream legacy,optimized;
  legacy<<std::setprecision(16)<<'[';bool comma=false;
  const auto start=std::chrono::steady_clock::now();
  for(const auto& [key,value]:values) {
    if(comma)legacy<<", ";comma=true;
    legacy<<"{\"h\": "<<std::get<0>(key)<<", \"k\": "<<std::get<1>(key)
          <<", \"l\": "<<std::get<2>(key)<<", \"value_real\": "<<value.real()
          <<", \"value_imag\": "<<value.imag()<<'}';
  }
  legacy<<']';const auto middle=std::chrono::steady_clock::now();
  sdft::write_density_fourier_json(optimized,values);
  const auto finish=std::chrono::steady_clock::now();
  auto expected=legacy.str();expected.erase(std::remove(expected.begin(),expected.end(),' '),expected.end());
  if(expected!=optimized.str()){std::cerr<<"16-digit density formatting differs\n";return 1;}
  std::ostringstream empty;sdft::write_density_fourier_json(empty,{});
  if(empty.str()!="[]")return 2;
  values.clear();values[{0,0,0}]=C(std::numeric_limits<double>::infinity(),0.);
  try {sdft::write_density_fourier_json(empty,values);return 3;}catch(const std::runtime_error&){}
  std::ostringstream bad;bad.setstate(std::ios::badbit);
  try {sdft::write_density_fourier_json(bad,{});return 4;}catch(const std::runtime_error&){}
  std::cout<<"100001 reciprocal density entries: exact legacy formatting PASS; legacy_s="
           <<std::chrono::duration<double>(middle-start).count()<<" optimized_s="
           <<std::chrono::duration<double>(finish-middle).count()<<'\n';
}
