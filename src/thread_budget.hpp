#pragma once
// Explicit two-level OpenMP thread budget.
//
// Outer k-point teams set a per-thread inner budget; kernels ask
// kernel_threads() instead of testing !omp_in_parallel(). Outside any team
// a kernel may use the whole pool. Inside a team without a budget it stays
// serial, which is the historical behaviour. SDFT_DETERMINISTIC=1 keeps
// every historical default (summation order, FFT planning, inner budget 1).
#include <cstdlib>
#include <stdexcept>
#include <string>
#ifdef SDFT_HAS_OPENMP
#include <omp.h>
#endif

namespace sdft {

inline bool deterministic_mode() {
  static const bool value = [] {
    const char* flag = std::getenv("SDFT_DETERMINISTIC");
    if (flag && std::string(flag) != "0" && std::string(flag) != "1")
      throw std::invalid_argument("SDFT_DETERMINISTIC must be 0 or 1");
    return flag && std::string(flag) == "1";
  }();
  return value;
}

inline int& inner_thread_budget() {
  thread_local int budget = 0;
  return budget;
}

// RAII: set the inner budget for the calling (team member) thread.
class InnerThreadBudget {
 public:
  explicit InnerThreadBudget(int threads) : previous_(inner_thread_budget()) {
    inner_thread_budget() = threads < 1 ? 1 : threads;
  }
  ~InnerThreadBudget() { inner_thread_budget() = previous_; }
  InnerThreadBudget(const InnerThreadBudget&) = delete;
  InnerThreadBudget& operator=(const InnerThreadBudget&) = delete;

 private:
  int previous_;
};

// Threads a kernel may open at this call site.
inline int kernel_threads() {
#ifdef SDFT_HAS_OPENMP
  if (!omp_in_parallel()) return omp_get_max_threads();
  const int budget = inner_thread_budget();
  return budget > 1 && omp_get_active_level() < omp_get_max_active_levels() ? budget : 1;
#else
  return 1;
#endif
}

}  // namespace sdft
