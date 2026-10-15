// ParallelContext checks. Serial build: single-rank semantics. MPI build
// (run under mpiexec -n 2 or 4): ordered sums, k ownership, broadcasts and
// band-group column gathers agree on every rank.
#include "parallel_context.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  auto& mpi = sdft::parallel();
  try {
    mpi.initialize(&argc, &argv);
    const int size = mpi.world_size(), rank = mpi.world_rank();
    // Ordered sum: rank r contributes r+0.1 in two slots.
    std::vector<double> v{rank + .1, -(rank + .1)};
    mpi.ordered_sum(v);
    double expected = 0;
    for (int r = 0; r < size; ++r) expected += r + .1;
    if (v[0] != expected || v[1] != -expected) throw std::runtime_error("ordered_sum");
    // k ownership covers every k exactly once per group set, identically.
    const std::size_t nk = 11;
    std::vector<double> costs(nk);
    for (std::size_t k = 0; k < nk; ++k) costs[k] = double((k * 7) % 5 + 1);
    mpi.assign_kpoints(costs);
    std::vector<double> owners(nk);
    for (std::size_t k = 0; k < nk; ++k) owners[k] = mpi.owner_of(k);
    auto copy = owners;
    mpi.broadcast_from_world_root(copy.data(), sizeof(double) * nk);
    if (copy != owners) throw std::runtime_error("assignment differs between ranks");
    for (auto o : owners) if (o < 0 || o >= mpi.kgroups()) throw std::runtime_error("owner out of range");
    // Broadcast from the owner of k=3.
    std::vector<std::complex<double>> payload;
    if (mpi.owns(3)) payload.assign(5, {3., -1.});
    mpi.broadcast_from_group(mpi.owner_of(3), payload);
    if (payload.size() != 5 || payload[4] != std::complex<double>(3., -1.)) throw std::runtime_error("group broadcast");
    // Band-group column gather.
    const std::size_t n = 3, columns = 7;
    std::vector<std::complex<double>> block(n * columns);
    const auto first = mpi.band_first_column(columns, mpi.bandgroup_rank());
    const auto last = mpi.band_first_column(columns, mpi.bandgroup_rank() + 1);
    for (std::size_t c = first; c < last; ++c)
      for (std::size_t i = 0; i < n; ++i) block[c * n + i] = {double(c), double(i)};
    mpi.band_allgather_columns(block.data(), n, columns);
    for (std::size_t c = 0; c < columns; ++c)
      for (std::size_t i = 0; i < n; ++i)
        if (block[c * n + i] != std::complex<double>(double(c), double(i))) throw std::runtime_error("band gather");
    if (!mpi.all(true) || mpi.max(double(rank)) != size - 1) throw std::runtime_error("all/max");
    if (rank == 0) std::cout << "parallel context PASS " << mpi.description() << "\n";
    mpi.finalize();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rank " << mpi.world_rank() << ": " << e.what() << "\n";
    mpi.abort(1);
  }
}
