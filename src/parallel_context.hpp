#pragma once
// Process-level parallel context (MPI when SDFT_ENABLE_MPI=ON, otherwise a
// single-rank stub with identical semantics).
//
// Layout: world = kgroups x bandgroups. A k point (k x spin) is owned by
// one k group; inside a k group the band group ranks share its Davidson
// subspace. Every collective here is ordered: data are gathered by owner
// and combined in the global k order, so results do not depend on the
// rank count beyond the solver's own arithmetic.
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace sdft {

class ParallelContext {
 public:
  static ParallelContext& instance();

  // Call once from main before any solver work; reads SDFT_KGROUPS and
  // SDFT_BANDGROUPS (defaults: kgroups=world size, bandgroups=1).
  void initialize(int* argc, char*** argv);
  void finalize();
  [[noreturn]] void abort(int code);

  bool enabled() const { return enabled_; }
  int world_rank() const { return world_rank_; }
  int world_size() const { return world_size_; }
  bool is_root() const { return world_rank_ == 0; }
  int kgroups() const { return kgroups_; }
  int kgroup() const { return kgroup_; }
  int bandgroups() const { return bandgroups_; }
  int bandgroup_rank() const { return band_rank_; }

  // Static round-robin by default; assign_kpoints() can rebalance by cost.
  int owner_of(std::size_t k) const;
  bool owns(std::size_t k) const { return owner_of(k) == kgroup_; }
  // Greedy longest-processing-time assignment from per-k costs (seconds of
  // the previous iteration). Identical on every rank: costs are gathered.
  void assign_kpoints(const std::vector<double>& costs);
  void reset_assignment() { assignment_.clear(); }

  // Broadcast from the root of k group `group` (its band rank 0) to all.
  void broadcast_from_group(int group, void* data, std::size_t bytes) const;
  void broadcast_from_group(int group, std::vector<std::complex<double>>& data) const;
  void broadcast_from_group(int group, std::vector<double>& data) const;
  void broadcast_from_world_root(void* data, std::size_t bytes) const;
  // Elementwise sum over world ranks, summed in rank order on rank 0 and
  // broadcast (deterministic for a fixed rank count).
  void ordered_sum(std::vector<double>& values) const;
  double max(double value) const;
  bool all(bool value) const;

  // Band group (within one k group).
  void band_allreduce_sum(std::vector<std::complex<double>>& values) const;
  void band_broadcast(std::vector<std::complex<double>>& values, int root = 0) const;
  // Columns [first(r), first(r+1)) of an n x columns block belong to band
  // rank r. Each rank fills its own columns in `data`; afterwards every
  // band rank holds the full block.
  std::size_t band_first_column(std::size_t columns, int rank) const;
  void band_allgather_columns(std::complex<double>* data, std::size_t n, std::size_t columns) const;
  void barrier() const;

  std::string description() const;

 private:
  ParallelContext() = default;
  bool enabled_ = false, owns_init_ = false;
  int world_rank_ = 0, world_size_ = 1;
  int kgroups_ = 1, kgroup_ = 0, bandgroups_ = 1, band_rank_ = 0;
  std::vector<int> assignment_;
  void* world_ = nullptr;  // MPI_Comm handles, opaque in this header
  void* band_ = nullptr;
};

inline ParallelContext& parallel() { return ParallelContext::instance(); }

}  // namespace sdft
