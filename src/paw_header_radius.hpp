#pragma once
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace sdft {
// PSCTR RDEP is in Bohr; the stored radial grid is in Angstrom. RDEPT
// names another radius and must never override this integration boundary.
inline std::optional<double> paw_header_rdep_angstrom(const std::string& line) {
  const auto equal = line.find('=');
  if (equal == std::string::npos) return {};
  const auto first = line.find_first_not_of(" \t\r");
  const auto last = line.find_last_not_of(" \t\r", equal - 1);
  if (first == std::string::npos || last == std::string::npos || last < first ||
      line.substr(first, last - first + 1) != "RDEP") return {};
  std::size_t used{};
  double value{};
  try { value = std::stod(line.substr(equal + 1), &used); }
  catch (const std::exception&) { throw std::runtime_error("invalid PAW RDEP header"); }
  if (!used || !std::isfinite(value) || value <= 0)
    throw std::runtime_error("PAW RDEP must be positive and finite");
  // This converts the file format, independently of runtime atomic units.
  constexpr double potcar_bohr_angstrom = .529177249;
  return value * potcar_bohr_angstrom;
}
} // namespace sdft
