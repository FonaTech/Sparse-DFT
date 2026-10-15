#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sdft {
// Inexact SCF search must resolve the actual acceptance thresholds, including
// the safety margin. Refining the density functional alone does not refine
// the eigenvectors that determine the PAW and Hubbard occupation matrices.
class CollinearInnerAccuracy {
  double density_limit_, onsite_limit_, final_accuracy_;
  bool final_;
 public:
  CollinearInnerAccuracy(double density_limit, double onsite_limit, bool adaptive)
      : density_limit_(density_limit), onsite_limit_(onsite_limit), final_(!adaptive) {
    if (!std::isfinite(density_limit) || density_limit <= 0 ||
        !std::isfinite(onsite_limit) || onsite_limit <= 0)
      throw std::invalid_argument("invalid effective SCF acceptance threshold");
    final_accuracy_ = std::min(1e-12, std::max(5e-14,
        std::min(density_limit, onsite_limit) / 10000.));
  }
  bool observe(double density_residual, double onsite_residual) {
    if (!std::isfinite(density_residual) || density_residual < 0 ||
        !std::isfinite(onsite_residual) || onsite_residual < 0)
      throw std::invalid_argument("invalid SCF precision feedback");
    const bool transition = !final_ && density_residual <= 1000.*density_limit_ &&
        onsite_residual <= 1000.*onsite_limit_;
    final_ = final_ || transition;
    return transition;
  }
  double requested(double outer_error, double search_factor, bool last_iteration) const {
    if (!std::isfinite(outer_error) || outer_error < 0 ||
        !std::isfinite(search_factor) || search_factor <= 0)
      throw std::invalid_argument("invalid adaptive inner accuracy");
    return final_ || last_iteration ? final_accuracy_ :
        std::clamp(search_factor*outer_error, final_accuracy_, 1e-4);
  }
  bool final_refinement() const { return final_; }
  double final_accuracy() const { return final_accuracy_; }
};
} // namespace sdft
