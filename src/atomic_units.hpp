#pragma once
#include <string_view>

namespace sdft {
enum class AtomicUnitConvention { Codata, VaspLegacy, MatchRadialMode };
struct AtomicUnits {
  double bohr_per_angstrom;
  double ev_per_hartree;
};
inline constexpr AtomicUnits atomic_units(AtomicUnitConvention convention) {
  return convention==AtomicUnitConvention::VaspLegacy
      ? AtomicUnits{1.0/0.529177249, 2.0*13.605826}
      : AtomicUnits{1.8897261254578281, 27.211386245988};
}
inline constexpr std::string_view atomic_unit_name(AtomicUnitConvention convention) {
  return convention==AtomicUnitConvention::VaspLegacy ? "vasp_legacy" : "codata";
}
}
