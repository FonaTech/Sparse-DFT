#pragma once
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sdft {
// Diagnostic search seeds, bound to the actual parent PWO SHA-256. This is
// never an SCF checkpoint or an assertion that the upper states converged.
inline void save_search_guards(const std::string& path,const std::string& parent_sha,
                              std::size_t dimension,
                              const std::vector<std::complex<double>>& guards) {
  if(std::endian::native!=std::endian::little || parent_sha.size()!=64 || !dimension ||
     guards.size()%dimension || std::filesystem::exists(path))
    throw std::invalid_argument("invalid diagnostic search state or existing output");
  for(auto v:guards)if(!std::isfinite(v.real())||!std::isfinite(v.imag()))
    throw std::invalid_argument("nonfinite diagnostic search state");
  const std::uint64_t header[]{dimension,guards.size()/dimension};
  std::ofstream f(path,std::ios::binary);
  f.write("SDFTSG01",8);f.write(parent_sha.data(),64);
  f.write(reinterpret_cast<const char*>(header),sizeof(header));
  f.write(reinterpret_cast<const char*>(guards.data()),guards.size()*sizeof(guards[0]));
  f.close();if(!f)throw std::runtime_error("diagnostic search state write failed");
}

inline std::vector<std::complex<double>> load_search_guards(
    const std::string& path,const std::string& parent_sha,std::size_t dimension,
    std::size_t maximum_columns) {
  if(std::endian::native!=std::endian::little || parent_sha.size()!=64 || !dimension)
    throw std::invalid_argument("invalid diagnostic search state platform/identity");
  const auto bytes=std::filesystem::file_size(path);
  if(bytes<88)throw std::runtime_error("truncated diagnostic search header");
  char magic[8],sha[64];std::uint64_t header[2];
  std::ifstream f(path,std::ios::binary);
  f.read(magic,8);f.read(sha,64);f.read(reinterpret_cast<char*>(header),sizeof(header));
  if(!f || std::string(magic,8)!="SDFTSG01" || std::string(sha,64)!=parent_sha ||
     header[0]!=dimension || header[1]>maximum_columns ||
     header[1]>(std::numeric_limits<std::size_t>::max()-88)/dimension/sizeof(std::complex<double>) ||
     bytes!=88+dimension*header[1]*sizeof(std::complex<double>))
    throw std::runtime_error("diagnostic search state identity/dimensions/size mismatch");
  std::vector<std::complex<double>> guards(dimension*header[1]);
  f.read(reinterpret_cast<char*>(guards.data()),guards.size()*sizeof(guards[0]));
  if(!f)throw std::runtime_error("truncated diagnostic search payload");
  for(auto v:guards)if(!std::isfinite(v.real())||!std::isfinite(v.imag()))
    throw std::runtime_error("nonfinite diagnostic search payload");
  return guards;
}
}
