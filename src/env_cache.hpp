#pragma once
// Cached environment reads for hot paths.
//
// getenv takes libc's environment lock (macOS __findenv_locked), so calling
// it per GEMM or per PAW product serializes k-point teams. Each thread keeps
// its own copy per variable and re-reads only when the process-wide
// generation changes. Code that calls setenv after start-up (smoke tests,
// benchmarks) must call refresh_environment_cache() afterwards.
#include <atomic>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace sdft {

inline std::atomic<unsigned>& environment_generation() {
  static std::atomic<unsigned> generation{1};
  return generation;
}

inline void refresh_environment_cache() {
  environment_generation().fetch_add(1, std::memory_order_relaxed);
}

// nullptr when unset, otherwise the variable's text. `name` must be a
// string literal (its address is the cache key).
inline const char* cached_getenv(const char* name) {
  struct Entry {
    unsigned generation = 0;
    bool present = false;
    std::string value;
  };
  thread_local std::unordered_map<const char*, Entry> cache;
  const unsigned now = environment_generation().load(std::memory_order_relaxed);
  auto& entry = cache[name];
  if (entry.generation != now) {
    const char* raw = std::getenv(name);
    entry.present = raw != nullptr;
    entry.value = raw ? raw : "";
    entry.generation = now;
  }
  return entry.present ? entry.value.c_str() : nullptr;
}

}  // namespace sdft
