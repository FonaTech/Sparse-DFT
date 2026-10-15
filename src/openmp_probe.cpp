#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <string>

#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
  bool require_parallel = argc > 1 && std::string(argv[1]) == "--require-parallel";
  int requested = -1;
  if (const char* env = std::getenv("OMP_NUM_THREADS")) {
    try { requested = std::max(1, std::stoi(env)); } catch (...) { requested = 1; }
  }
#ifdef SDFT_HAS_OPENMP
  omp_set_dynamic(0);
  if (requested > 0) omp_set_num_threads(requested);
  else requested = omp_get_max_threads();
  int team = 0;
#pragma omp parallel reduction(+:team)
  { team += 1; }
  const int max_threads = omp_get_max_threads();
  std::cout << "{\"compiled\":true,\"requested_threads\":" << requested
            << ",\"team_threads\":" << team
            << ",\"max_threads\":" << max_threads << "}\n";
  if (require_parallel && (requested > 1) && team < 2) return 2;
#else
  requested = 1;
  std::cout << "{\"compiled\":false,\"requested_threads\":1,\"team_threads\":1}\n";
  if (require_parallel) return 2;
#endif
  return 0;
}
