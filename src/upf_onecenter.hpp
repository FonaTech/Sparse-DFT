#pragma once
#include "upf_paw_operator.hpp"

namespace sdft {
// Original UPF one-centre Hxc functional, in Hartree and Bohr. This result
// excludes the density-independent core energy and the bare PP_DIJ term.
struct UpfOneCenterResult {
  double energy{}, hartree_ae{},hartree_ps{},xc_ae{},xc_ps{};
  std::vector<std::complex<double>> derivative;
  std::size_t negative_density_samples{};
};
UpfOneCenterResult upf_onecenter(const UpfPawData&, const UpfPawOperator&,
    const std::vector<std::complex<double>>& density, int angular_order=16);
} // namespace sdft
