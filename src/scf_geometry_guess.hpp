#pragma once
#include "scf_core.hpp"
#include "density_functionals.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>

namespace sdft {
// A guess is neither a checkpoint nor a frozen NSCF Hamiltonian. All
// operators, overlaps, projections and occupations are rebuilt by run_scf.
struct ScfGeometryGuess {
  int spin_channels{1};
  std::array<FourierDensity,2> spin_smooth;
  std::string signature;
  double volume{};
  FourierDensity smooth,atomic_smooth;
  std::vector<std::vector<std::complex<double>>> onsite;
  std::vector<std::vector<GVector>> g;
  std::vector<std::vector<std::complex<double>>> waves;
};
inline ScfGeometryGuess read_geometry_guess(const std::filesystem::path& path,
    const std::string& signature,std::size_t nk,std::size_t natoms,std::size_t bands,int spin_channels=1) {
  using C=std::complex<double>;
  std::ifstream hashes(path.string()+".sha256");std::string expected;
  if(!(hashes>>expected)||expected.size()!=64||file_sha256(path)!=expected)
    throw std::runtime_error("geometry guess SHA-256 mismatch");
  std::ifstream file(path);std::string tag;
  ScfGeometryGuess out;std::size_t count{};
  if(!(file>>tag>>out.signature>>out.volume)||(tag!="SDFT_GEOMETRY_GUESS_1"&&tag!="SDFT_GEOMETRY_GUESS_2"&&tag!="SDFT_GEOMETRY_GUESS_3")||
      out.signature!=signature||!std::isfinite(out.volume)||out.volume<=0)
    throw std::runtime_error("geometry guess identity/volume mismatch");
  out.spin_channels=tag=="SDFT_GEOMETRY_GUESS_3"?2:1;
  if(out.spin_channels!=spin_channels)throw std::runtime_error("geometry guess spin model mismatch");
  const bool atomic_difference=tag!="SDFT_GEOMETRY_GUESS_1";
  const auto value=[&](){double r{},i{};
    if(!(file>>r>>i)||!std::isfinite(r)||!std::isfinite(i))throw std::runtime_error("invalid geometry guess value");return C(r,i);};
  if(!(file>>count)||count>10000000)throw std::runtime_error("invalid geometry guess density count");
  for(std::size_t i=0;i<count;++i){int h{},k{},l{};if(!(file>>h>>k>>l))throw std::runtime_error("invalid guess density index");
    const auto v=value();if(!out.smooth.emplace(std::make_tuple(h,k,l),v).second)throw std::runtime_error("duplicate guess density index");}
  if(atomic_difference){
    if(!(file>>count)||count>10000000)throw std::runtime_error("invalid guess atomic density count");
    for(std::size_t i=0;i<count;++i){int h{},k{},l{};if(!(file>>h>>k>>l))throw std::runtime_error("invalid atomic density index");
      const auto v=value();if(!out.atomic_smooth.emplace(std::make_tuple(h,k,l),v).second)throw std::runtime_error("duplicate atomic density index");}
  }
  if(!(file>>count)||count!=natoms)throw std::runtime_error("geometry guess atom count mismatch");
  out.onsite.resize(count);
  for(auto& atom:out.onsite){std::size_t size{};if(!(file>>size)||size>100000)throw std::runtime_error("invalid guess onsite count");
    atom.resize(size);for(auto& v:atom)v=value();}
  if(!(file>>count)||count!=nk)throw std::runtime_error("geometry guess k count mismatch");
  out.g.resize(count);out.waves.resize(count);
  for(std::size_t k=0;k<count;++k){std::size_t n{},nb{};
    if(!(file>>n>>nb)||!n||n>10000000||nb!=bands||n>100000000/bands)throw std::runtime_error("invalid guess wave dimensions");
    out.g[k].resize(n);out.waves[k].resize(n*bands);
    for(auto& g:out.g[k])if(!(file>>g.h>>g.k>>g.l))throw std::runtime_error("invalid guess G index");
    for(auto& v:out.waves[k])v=value();
  }
  if(out.spin_channels==2)for(auto& q:out.spin_smooth){
    if(!(file>>count)||!count||count>10000000)throw std::runtime_error("invalid spin guess density count");
    for(std::size_t i=0;i<count;++i){int h{},k{},l{};if(!(file>>h>>k>>l))throw std::runtime_error("invalid spin guess density index");
      const auto v=value();if(!q.emplace(std::make_tuple(h,k,l),v).second)throw std::runtime_error("duplicate spin guess density index");}
  }
  if(!(file>>tag)||tag!="END"||(file>>tag))throw std::runtime_error("truncated or trailing geometry guess");
  return out;
}
inline void write_geometry_guess(const std::filesystem::path& path,const ScfGeometryGuess& guess) {
  if(path.empty())return;
  auto tmp=path;tmp+=".tmp";
  std::ofstream file(tmp);if(!file)throw std::runtime_error("cannot write geometry guess");
  file<<std::setprecision(17)<<(guess.spin_channels==2?"SDFT_GEOMETRY_GUESS_3 ":"SDFT_GEOMETRY_GUESS_2 ")<<guess.signature<<' '<<guess.volume<<'\n';
  file<<guess.smooth.size()<<'\n';
  for(const auto& [key,v]:guess.smooth){const auto [h,k,l]=key;file<<h<<' '<<k<<' '<<l<<' '<<v.real()<<' '<<v.imag()<<'\n';}
  file<<guess.atomic_smooth.size()<<'\n';
  for(const auto& [key,v]:guess.atomic_smooth){const auto [h,k,l]=key;file<<h<<' '<<k<<' '<<l<<' '<<v.real()<<' '<<v.imag()<<'\n';}
  file<<guess.onsite.size()<<'\n';for(const auto& atom:guess.onsite){file<<atom.size()<<'\n';for(const auto& v:atom)file<<v.real()<<' '<<v.imag()<<'\n';}
  file<<guess.g.size()<<'\n';for(std::size_t k=0;k<guess.g.size();++k){
    file<<guess.g[k].size()<<' '<<guess.waves[k].size()/guess.g[k].size()<<'\n';
    for(const auto& g:guess.g[k])file<<g.h<<' '<<g.k<<' '<<g.l<<'\n';
    for(const auto& v:guess.waves[k])file<<v.real()<<' '<<v.imag()<<'\n';}
  if(guess.spin_channels==2)for(const auto& q:guess.spin_smooth){file<<q.size()<<'\n';
    for(const auto& [key,v]:q){const auto [h,k,l]=key;file<<h<<' '<<k<<' '<<l<<' '<<v.real()<<' '<<v.imag()<<'\n';}}
  file<<"END\n";file.close();if(!file)throw std::runtime_error("cannot flush geometry guess");
  const auto hash=file_sha256(tmp);std::filesystem::rename(tmp,path);
  std::ofstream digest(path.string()+".sha256.tmp");digest<<hash<<'\n';digest.close();
  if(!digest)throw std::runtime_error("cannot flush geometry guess hash");
  std::filesystem::rename(path.string()+".sha256.tmp",path.string()+".sha256");
}
} // namespace sdft
