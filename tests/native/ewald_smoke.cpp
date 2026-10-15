#include "ewald.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
double& component(sdft::Vec3& v, int i) { return i==0?v.x:(i==1?v.y:v.z); }
double component(const sdft::Vec3& v, int i) { return i==0?v.x:(i==1?v.y:v.z); }
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}

int main() {
  try {
    using namespace sdft;
    // A skew, charged cell exercises reciprocal bounds and the background
    // derivative. Negative and positive charges prevent symmetry hiding signs.
    const std::array<Vec3,3> a{{{8.,.4,.1},{.7,9.,-.2},{-.1,.3,10.}}};
    const std::vector<Vec3> positions{{.13,.21,.08},{.47,.61,.73},{.69,.24,.45}};
    const std::vector<double> charges{2.,-1.,.4};
    const auto ref=evaluate_ewald(a,positions,charges,.43);
    const double step=1e-5, volume=std::abs(determinant(a));
    const std::array<Vec3,3> dual{cross(a[1],a[2]),cross(a[2],a[0]),cross(a[0],a[1])};
    double force_error=0,stress_error=0,eta_error=0,invariance_error=0;
    for(std::size_t atom=0;atom<positions.size();++atom) for(int axis=0;axis<3;++axis) {
      auto p=positions,m=positions;
      for(int f=0;f<3;++f) {
        component(p[atom],f)+=step*component(dual[f],axis)/determinant(a);
        component(m[atom],f)-=step*component(dual[f],axis)/determinant(a);
      }
      const double fd=-(evaluate_ewald(a,p,charges,.43).energy-evaluate_ewald(a,m,charges,.43).energy)/(2*step);
      force_error=std::max(force_error,std::abs(fd-component(ref.forces[atom],axis)));
    }
    // All nine Cartesian strain derivatives, including shear, at fixed
    // fractional positions. This catches transpose and sign errors.
    for(int alpha=0;alpha<3;++alpha) for(int beta=0;beta<3;++beta) {
      auto p=a,m=a;
      for(int v=0;v<3;++v) {
        component(p[v],alpha)+=step*component(a[v],beta);
        component(m[v],alpha)-=step*component(a[v],beta);
      }
      const double fd=(evaluate_ewald(p,positions,charges,.43).energy-evaluate_ewald(m,positions,charges,.43).energy)/(2*step*volume);
      stress_error=std::max(stress_error,std::abs(fd-component(ref.stress[alpha],beta)));
    }
    for(double eta:{0.,.23,.71}) {
      const auto other=evaluate_ewald(a,positions,charges,eta);
      eta_error=std::max(eta_error,std::abs(other.energy-ref.energy));
      for(std::size_t atom=0;atom<positions.size();++atom) for(int axis=0;axis<3;++axis)
        require(std::abs(component(other.forces[atom],axis)-component(ref.forces[atom],axis))<1e-12,"eta-dependent Ewald forces");
    }
    auto shifted=positions;
    for(auto& f:shifted) { f.x+=.37; f.y-=.19; f.z+=.57; }
    shifted[0].x+=3.;shifted[1].y-=2.;
    invariance_error=std::abs(evaluate_ewald(a,shifted,charges,.43).energy-ref.energy);
    double sum_error=0;
    for(int axis=0;axis<3;++axis) {
      double total=0;
      for(const auto& force:ref.forces) total+=component(force,axis);
      sum_error=std::max(sum_error,std::abs(total));
    }
    // Independent NaCl Madelung constant: cubic cell, eight alternating ions,
    // nearest-neighbour distance L/2, four formula units. Background vanishes.
    std::vector<Vec3> salt;std::vector<double> salt_charges;
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int k=0;k<2;++k) {
      salt.push_back({i*.5,j*.5,k*.5});salt_charges.push_back((i+j+k)%2?-1.:1.);
    }
    const auto nacl=evaluate_ewald({{{10.,0,0},{0,10.,0},{0,0,10.}}},salt,salt_charges);
    const double madelung=-nacl.energy*5./4.;
    std::cout<<"ewald force_error="<<force_error<<" stress_error="<<stress_error
      <<" eta_energy_error="<<eta_error<<" translation_error="<<invariance_error
      <<" force_sum="<<sum_error<<" NaCl_Madelung="<<madelung<<'\n';
    require(force_error<5e-9,"Ewald force finite difference failed");
    require(stress_error<5e-10,"Ewald stress finite difference failed");
    require(eta_error<1e-12&&invariance_error<1e-12&&sum_error<1e-12,"Ewald invariance failed");
    require(std::abs(madelung-1.747564594633182)<2e-12,"NaCl independent Madelung benchmark failed");
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
