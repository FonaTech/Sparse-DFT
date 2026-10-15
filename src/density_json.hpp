#pragma once
#include <charconv>
#include <cmath>
#include <complex>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace sdft {
// Same public fields and 16-digit numeric policy as the SCF ostream writer.
// Formatting stays bounded independently of the reciprocal support size.
inline void write_density_fourier_json(std::ostream& out,
    const std::map<std::tuple<int,int,int>,std::complex<double>>& density) {
  std::string buffer;
  constexpr std::size_t chunk_size=256*1024;
  buffer.reserve(chunk_size+512);
  const auto integer=[&](int value) {
    char text[32];const auto converted=std::to_chars(text,text+sizeof(text),value);
    if(converted.ec!=std::errc{})throw std::runtime_error("density integer formatting failed");
    buffer.append(text,converted.ptr);
  };
  const auto real=[&](double value) {
    if(!std::isfinite(value))throw std::runtime_error("nonfinite density JSON value");
    char text[96];const auto converted=std::to_chars(text,text+sizeof(text),value,std::chars_format::general,16);
    if(converted.ec!=std::errc{})throw std::runtime_error("density real formatting failed");
    buffer.append(text,converted.ptr);
  };
  const auto flush=[&]() {
    out.write(buffer.data(),static_cast<std::streamsize>(buffer.size()));
    if(!out)throw std::runtime_error("density JSON write failed");
    buffer.clear();
  };
  buffer+='[';bool comma=false;
  for(const auto& [key,value]:density) {
    if(comma)buffer+=',';comma=true;
    buffer+="{\"h\":";integer(std::get<0>(key));
    buffer+=",\"k\":";integer(std::get<1>(key));
    buffer+=",\"l\":";integer(std::get<2>(key));
    buffer+=",\"value_real\":";real(value.real());
    buffer+=",\"value_imag\":";real(value.imag());buffer+='}';
    if(buffer.size()>=chunk_size)flush();
  }
  buffer+=']';flush();
}
} // namespace sdft
