#include "ordered_fourier.hpp"
#include <iostream>
#include <limits>

int main() {
  using namespace sdft;using C=std::complex<double>;
  const FourierDensity a{{{-3,0,1},C(.1,.3)},{{0,0,0},C(.7,-.3)},{{2,1,-1},C(-.2,.4)}};
  const FourierDensity b{{{-4,1,0},C(.04,-.01)},{{-3,0,1},C(.6,.4)},{{1,0,0},C(.01,.8)},{{2,1,-1},C(.9,.02)}};
  const FourierDensity c{{{-2,-1,1},C(.04,.1)},{{0,0,0},C(.3,-.2)},{{3,0,-1},C(.07,.4)},{{8,-1,0},C(.01,.02)}};
  for(const auto& left:{FourierDensity{},a,b,c})for(const auto& right:{FourierDensity{},a,b,c}) {
    for(double w:{1.,.3,0.,-.4}) {
      auto expected=left,got=left,consumed=left,source=right;
      for(const auto& [key,value]:right)expected[key]+=w*value;
      accumulate_ordered_fourier(got,right,w);consume_ordered_fourier(consumed,std::move(source),w);
      if(got!=expected||consumed!=expected)return 1;
    }
    double expected=0;
    for(const auto& [key,value]:left){auto [h,k,l]=key;if(auto it=right.find({-h,-k,-l});it!=right.end())expected+=13.7*(value*it->second).real();}
    if(opposite_fourier_contract(left,right,13.7)!=expected)return 2;
  }
  auto aliased=a,expected=a;
  for(auto& [key,z]:expected){(void)key;z+=.3*z;}
  accumulate_ordered_fourier(aliased,aliased,.3);if(aliased!=expected)return 3;
  bool rejected=false;
  try{consume_ordered_fourier(aliased,std::move(aliased));}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)return 4;
  for(double w:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    rejected=false;try{accumulate_ordered_fourier(aliased,a,w);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)return 5;
  }
  std::cout<<"ordered Fourier complex map/consume/alias/absent-opposite arithmetic equivalence PASS\n";
}
