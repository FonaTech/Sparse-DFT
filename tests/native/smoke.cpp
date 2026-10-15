#include "sdft_core.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
int main(){
  std::array<sdft::Vec3,3> a{{{1,0,0},{0,1,0},{0,0,1}}}; assert(sdft::determinant(a)==1.0);
  std::random_device nonce;
  std::filesystem::path directory;
  for(int attempt=0;attempt<128;++attempt){
    auto candidate=std::filesystem::temp_directory_path()/("sparse-dft-core-smoke-"+std::to_string(nonce())+"-"+std::to_string(nonce()));
    if(std::filesystem::create_directory(candidate)){directory=candidate;break;}
  }
  if(directory.empty())throw std::runtime_error("cannot create isolated smoke fixture");
  struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}} cleanup{directory};
  auto p=directory/"metadata.json";
  { std::ofstream f(p); f<<R"({"record_id":"Si:standard","element":{"symbol":"Si"},"variant":{"family":"standard"},"metadata":{"potcar":{"exchange_correlation":"PE","valence_electrons":4.0,"enmax_ev":245.345},"files":{"POTCAR":{"sha256":"x"},"PSCTR":{"sha256":"y"}}}})"; }
  auto m=sdft::read_paw_cache_metadata(p); assert(m.record_id=="Si:standard"); assert(m.element=="Si"); assert(m.family=="standard"); assert(m.zval==4.0); assert(m.enmax_ev==245.345); std::filesystem::remove(p);
  std::cout << "sdft_core_smoke PASS\n"; return 0;
}
