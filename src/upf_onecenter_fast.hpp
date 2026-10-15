#pragma once
#include "upf_onecenter.hpp"
namespace sdft {
UpfOneCenterResult upf_onecenter_fast(const UpfPawData&,const UpfPawOperator&,
    const std::vector<std::complex<double>>&,int angular_order=12);
struct UpfSpinOneCenterResult {
  double energy{},hartree_ae{},hartree_ps{},xc_ae{},xc_ps{};
  std::array<std::vector<std::complex<double>>,2> derivative;
  std::size_t negative_density_samples{};
};
UpfSpinOneCenterResult upf_onecenter_collinear(const UpfPawData&,const UpfPawOperator&,
    const std::array<std::vector<std::complex<double>>,2>&,int angular_order=12);
}
