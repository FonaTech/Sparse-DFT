#include "scf_geometry_guess.hpp"
#include "compute_backend.hpp"
#include <iostream>
#include <chrono>
int main(){
  using namespace sdft;using C=std::complex<double>;
  const auto path=std::filesystem::temp_directory_path()/("sparse-dft-guess-smoke-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(new_immutable_matrix_identity()));
  ScfGeometryGuess guess;guess.signature="paw-sha:units:bands:kpoints";guess.volume=47.;
  guess.smooth={{{0,0,0},.1},{{1,0,-1},C(.03,.02)},{{-1,0,1},C(.03,-.02)}};
  guess.atomic_smooth={{{0,0,0},.1},{{1,0,-1},C(.01,-.03)},{{-1,0,1},C(.01,.03)}};
  guess.onsite={{1.,C(.1,.2),C(.1,-.2),2.}};
  guess.g={{{0,0,0},{1,0,0},{0,-1,0}}};guess.waves={{C(.9,.1),C(.1,.02),C(-.1,.2)}};
  write_geometry_guess(path,guess);
  auto loaded=read_geometry_guess(path,guess.signature,1,1,1);
  if(loaded.smooth!=guess.smooth||loaded.atomic_smooth!=guess.atomic_smooth||loaded.onsite!=guess.onsite||loaded.waves!=guess.waves)return 1;
  try{read_geometry_guess(path,"other-paw",1,1,1);return 2;}catch(const std::runtime_error&){}
  try{read_geometry_guess(path,guess.signature,2,1,1);return 3;}catch(const std::runtime_error&){}
  guess.spin_channels=2;guess.signature+="spin=2:Fe:l2:U4";
  guess.spin_smooth={guess.smooth,guess.atomic_smooth};
  guess.onsite.push_back(guess.onsite.front());guess.g.push_back(guess.g.front());guess.waves.push_back(guess.waves.front());
  write_geometry_guess(path,guess);
  loaded=read_geometry_guess(path,guess.signature,2,2,1,2);
  if(loaded.spin_smooth!=guess.spin_smooth||loaded.onsite!=guess.onsite||loaded.waves!=guess.waves)return 5;
  try{read_geometry_guess(path,guess.signature,2,2,1,1);return 6;}catch(const std::runtime_error&){}
  try{read_geometry_guess(path,"spin=2:Fe:l2:U0",2,2,1,2);return 7;}catch(const std::runtime_error&){}
  {std::ofstream corrupt(path,std::ios::app);corrupt<<"garbage";}
  try{read_geometry_guess(path,guess.signature,1,1,1);return 4;}catch(const std::runtime_error&){}
  std::filesystem::remove(path);std::filesystem::remove(path.string()+".sha256");
  std::cout<<"geometry guess complex roundtrip, PAW/k identity and corruption rejection PASS\n";
}
