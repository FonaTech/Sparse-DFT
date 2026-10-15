#include "sdft_core.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
int main(){try{
  std::random_device random;std::filesystem::path directory;
  for(int i=0;i<128;++i){auto p=std::filesystem::temp_directory_path()/("sparse-dft-hash-"+std::to_string(random())+"-"+std::to_string(random()));
    if(std::filesystem::create_directory(p)){directory=p;break;}}
  if(directory.empty())throw std::runtime_error("cannot create hash fixture");
  struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove_all(p,e);}} cleanup{directory};
  const auto require=[](bool ok){if(!ok)throw std::runtime_error("public SHA-256 oracle mismatch");};
  const auto p=directory/"bytes";
  {std::ofstream f(p,std::ios::binary);}
  require(sdft::file_sha256(p)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  {std::ofstream f(p,std::ios::binary);f<<"abc";}
  const std::string abc="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  require(sdft::file_sha256(p)==abc&&sdft::file_sha256_matches(p,abc));
  require(!sdft::file_sha256_matches(p,std::string(64,'0')));
  {std::ofstream f(p,std::ios::binary);for(int i=0;i<131089;++i)f.put(static_cast<char>(i%251));}
  require(sdft::file_sha256(p)=="1ec3437cee3cccf3647e130524ade52848571960a3340ae33573beef17d330b3");
  bool missing=false;try{sdft::file_sha256(directory/"absent");}catch(const std::runtime_error&){missing=true;}
  require(missing&&!sdft::file_sha256_matches(directory/"absent",abc));
  bool unreadable=false;try{sdft::file_sha256(directory);}catch(const std::runtime_error&){unreadable=true;}
  require(unreadable);
  std::cout<<"public SHA-256 empty/abc/multichunk/mismatch/missing/unreadable PASS\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
