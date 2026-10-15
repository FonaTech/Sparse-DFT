#include "tabulated_spline.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  using sdft::TabulatedSpline;
  const TabulatedSpline linear({0,1,2,3}, {1,3,5,7});
  for (double x : {-0.5,0.0,0.3,1.8,3.0,3.5})
    if (std::abs(linear.value(x) - (1 + 2 * x)) > 1e-13 ||
        std::abs(linear.derivative(x) - 2) > 1e-13) return 1;
  // y=(x-3)^3 has y'(0)=27 and y''(3)=0, exactly matching the
  // clamped-left/natural-right boundary conditions.
  const TabulatedSpline cubic({0,1,2,3}, {-27,-8,-1,0}, 27.0);
  for (double x : {0.0,0.12,0.75,1.45,2.8,3.2})
    if (std::abs(cubic.value(x) - std::pow(x - 3, 3)) > 1e-12 ||
        std::abs(cubic.derivative(x) - 3 * (x - 3) * (x - 3)) > 1e-12) return 2;
  bool rejected = false;
  try { TabulatedSpline bad({0,0,1}, {1,2,3}); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 3;
  std::cout << "spline_smoke PASS nodes, polynomial, boundary, extrapolation, invalid mesh\n";
}
