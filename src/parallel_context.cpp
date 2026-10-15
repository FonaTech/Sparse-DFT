#include "parallel_context.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef SDFT_HAS_MPI
#include <mpi.h>
#endif

namespace sdft {
namespace {
int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end || parsed < 1 || parsed > 1 << 20)
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  return static_cast<int>(parsed);
}
#ifdef SDFT_HAS_MPI
MPI_Comm world_comm = MPI_COMM_NULL;
MPI_Comm band_comm = MPI_COMM_NULL;
void check(int code, const char* what) {
  if (code != MPI_SUCCESS) throw std::runtime_error(std::string("MPI failure in ") + what);
}
#endif
}  // namespace

ParallelContext& ParallelContext::instance() {
  static ParallelContext context;
  return context;
}

void ParallelContext::initialize(int* argc, char*** argv) {
#ifdef SDFT_HAS_MPI
  int already = 0;
  MPI_Initialized(&already);
  if (!already) {
    int provided = 0;
    // Only the main thread calls MPI; OpenMP teams never communicate.
    check(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    owns_init_ = true;
  }
  check(MPI_Comm_dup(MPI_COMM_WORLD, &world_comm), "MPI_Comm_dup");
  MPI_Comm_rank(world_comm, &world_rank_);
  MPI_Comm_size(world_comm, &world_size_);
  enabled_ = true;
  bandgroups_ = env_int("SDFT_BANDGROUPS", 1);
  if (world_size_ % bandgroups_)
    throw std::invalid_argument("SDFT_BANDGROUPS must divide the number of MPI ranks");
  const int requested_k = env_int("SDFT_KGROUPS", world_size_ / bandgroups_);
  if (requested_k * bandgroups_ != world_size_)
    throw std::invalid_argument("SDFT_KGROUPS x SDFT_BANDGROUPS must equal the number of MPI ranks");
  kgroups_ = requested_k;
  // Contiguous ranks form a k group; band rank 0 of a group is its root.
  kgroup_ = world_rank_ / bandgroups_;
  band_rank_ = world_rank_ % bandgroups_;
  check(MPI_Comm_split(world_comm, kgroup_, band_rank_, &band_comm), "MPI_Comm_split");
#else
  (void)argc;
  (void)argv;
  // Without MPI the job is one rank; group variables other than 1 are an
  // error rather than a silent serial run.
  if (env_int("SDFT_BANDGROUPS", 1) != 1 || env_int("SDFT_KGROUPS", 1) != 1)
    throw std::invalid_argument("SDFT_KGROUPS/SDFT_BANDGROUPS need a build with SDFT_ENABLE_MPI=ON");
#endif
}

void ParallelContext::finalize() {
#ifdef SDFT_HAS_MPI
  if (!enabled_) return;
  if (band_comm != MPI_COMM_NULL) MPI_Comm_free(&band_comm);
  if (world_comm != MPI_COMM_NULL) MPI_Comm_free(&world_comm);
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (owns_init_ && !finalized) MPI_Finalize();
  enabled_ = false;
#endif
}

void ParallelContext::abort(int code) {
#ifdef SDFT_HAS_MPI
  if (enabled_ && world_size_ > 1) {
    std::cerr.flush();
    MPI_Abort(world_comm, code);
  }
#endif
  std::exit(code);
}

int ParallelContext::owner_of(std::size_t k) const {
  if (kgroups_ <= 1) return 0;
  if (k < assignment_.size()) return assignment_[k];
  return static_cast<int>(k % static_cast<std::size_t>(kgroups_));
}

void ParallelContext::assign_kpoints(const std::vector<double>& costs) {
  assignment_.assign(costs.size(), 0);
  if (kgroups_ <= 1) return;
  // Longest processing time first; ties keep the lower k index first and
  // the lower group, so every rank computes the same assignment.
  std::vector<std::size_t> order(costs.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return costs[a] > costs[b]; });
  std::vector<double> load(static_cast<std::size_t>(kgroups_), 0.);
  for (const auto k : order) {
    const auto group = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
    assignment_[k] = group;
    load[static_cast<std::size_t>(group)] += costs[k] > 0 ? costs[k] : 1.;
  }
}

void ParallelContext::broadcast_from_group(int group, void* data, std::size_t bytes) const {
#ifdef SDFT_HAS_MPI
  if (!enabled_ || world_size_ == 1 || !bytes) return;
  const int root = group * bandgroups_;
  // Large payloads in INT_MAX-sized pieces.
  auto* cursor = static_cast<char*>(data);
  while (bytes) {
    const int piece = static_cast<int>(std::min<std::size_t>(bytes, 1u << 30));
    check(MPI_Bcast(cursor, piece, MPI_BYTE, root, world_comm), "MPI_Bcast");
    cursor += piece;
    bytes -= static_cast<std::size_t>(piece);
  }
#else
  (void)group; (void)data; (void)bytes;
#endif
}

void ParallelContext::broadcast_from_group(int group, std::vector<std::complex<double>>& data) const {
  std::size_t size = data.size();
  broadcast_from_group(group, &size, sizeof size);
  data.resize(size);
  broadcast_from_group(group, data.data(), size * sizeof(std::complex<double>));
}

void ParallelContext::broadcast_from_group(int group, std::vector<double>& data) const {
  std::size_t size = data.size();
  broadcast_from_group(group, &size, sizeof size);
  data.resize(size);
  broadcast_from_group(group, data.data(), size * sizeof(double));
}

void ParallelContext::broadcast_from_world_root(void* data, std::size_t bytes) const {
  broadcast_from_group(0, data, bytes);
}

void ParallelContext::ordered_sum(std::vector<double>& values) const {
#ifdef SDFT_HAS_MPI
  if (!enabled_ || world_size_ == 1) return;
  const int count = static_cast<int>(values.size());
  std::vector<double> all(world_rank_ == 0 ? values.size() * static_cast<std::size_t>(world_size_) : 0);
  check(MPI_Gather(values.data(), count, MPI_DOUBLE, all.data(), count, MPI_DOUBLE, 0, world_comm), "MPI_Gather");
  if (world_rank_ == 0)
    for (int i = 0; i < count; ++i) {
      double sum = 0;
      for (int r = 0; r < world_size_; ++r) sum += all[static_cast<std::size_t>(r) * values.size() + i];
      values[static_cast<std::size_t>(i)] = sum;
    }
  check(MPI_Bcast(values.data(), count, MPI_DOUBLE, 0, world_comm), "MPI_Bcast");
#else
  (void)values;
#endif
}

double ParallelContext::max(double value) const {
#ifdef SDFT_HAS_MPI
  if (enabled_ && world_size_ > 1) {
    double out = value;
    check(MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, world_comm), "MPI_Allreduce");
    return out;
  }
#endif
  return value;
}

bool ParallelContext::all(bool value) const {
#ifdef SDFT_HAS_MPI
  if (enabled_ && world_size_ > 1) {
    int in = value ? 1 : 0, out = 0;
    check(MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, world_comm), "MPI_Allreduce");
    return out != 0;
  }
#endif
  return value;
}

void ParallelContext::band_allreduce_sum(std::vector<std::complex<double>>& values) const {
#ifdef SDFT_HAS_MPI
  if (!enabled_ || bandgroups_ == 1) return;
  // Ordered: gather to band root, sum in band-rank order, broadcast.
  const int count = static_cast<int>(2 * values.size());
  std::vector<double> all(band_rank_ == 0 ? static_cast<std::size_t>(count) * bandgroups_ : 0);
  check(MPI_Gather(values.data(), count, MPI_DOUBLE, all.data(), count, MPI_DOUBLE, 0, band_comm), "MPI_Gather");
  if (band_rank_ == 0) {
    auto* out = reinterpret_cast<double*>(values.data());
    for (int i = 0; i < count; ++i) {
      double sum = 0;
      for (int r = 0; r < bandgroups_; ++r) sum += all[static_cast<std::size_t>(r) * count + i];
      out[i] = sum;
    }
  }
  check(MPI_Bcast(values.data(), count, MPI_DOUBLE, 0, band_comm), "MPI_Bcast");
#else
  (void)values;
#endif
}

void ParallelContext::band_broadcast(std::vector<std::complex<double>>& values, int root) const {
#ifdef SDFT_HAS_MPI
  if (!enabled_ || bandgroups_ == 1) return;
  check(MPI_Bcast(values.data(), static_cast<int>(2 * values.size()), MPI_DOUBLE, root, band_comm), "MPI_Bcast");
#else
  (void)values; (void)root;
#endif
}

std::size_t ParallelContext::band_first_column(std::size_t columns, int rank) const {
  const auto groups = static_cast<std::size_t>(bandgroups_);
  const auto r = static_cast<std::size_t>(rank);
  return columns / groups * r + std::min(r, columns % groups);
}

void ParallelContext::band_allgather_columns(std::complex<double>* data, std::size_t n,
                                             std::size_t columns) const {
#ifdef SDFT_HAS_MPI
  if (!enabled_ || bandgroups_ == 1) return;
  std::vector<int> counts(static_cast<std::size_t>(bandgroups_)), offsets(counts.size());
  for (int r = 0; r < bandgroups_; ++r) {
    const auto first = band_first_column(columns, r), last = band_first_column(columns, r + 1);
    if (2 * n * last > static_cast<std::size_t>(INT32_MAX)) throw std::overflow_error("band-group block too large for MPI int counts");
    counts[static_cast<std::size_t>(r)] = static_cast<int>(2 * n * (last - first));
    offsets[static_cast<std::size_t>(r)] = static_cast<int>(2 * n * first);
  }
  check(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data, counts.data(), offsets.data(),
                       MPI_DOUBLE, band_comm), "MPI_Allgatherv");
#else
  (void)data; (void)n; (void)columns;
#endif
}

void ParallelContext::barrier() const {
#ifdef SDFT_HAS_MPI
  if (enabled_ && world_size_ > 1) MPI_Barrier(world_comm);
#endif
}

std::string ParallelContext::description() const {
  std::ostringstream out;
  out << "{\"mpi_compiled\":"
#ifdef SDFT_HAS_MPI
      << "true"
#else
      << "false"
#endif
      << ",\"world_size\":" << world_size_ << ",\"kgroups\":" << kgroups_
      << ",\"bandgroups\":" << bandgroups_ << '}';
  return out.str();
}

}  // namespace sdft
