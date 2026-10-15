#include "sdft_core.hpp"
#include "plane_wave_operator.hpp"
#include "compute_backend.hpp"
#include "solver_search_state.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

int main(int argc,char** argv) {
  try {
    if(argc<3||argc>5)throw std::invalid_argument("usage: sdft_solver_replay snapshot.pwop report.json [max_iterations] [warm_snapshot.pwop]");
    if(std::filesystem::exists(argv[2]))throw std::runtime_error("refusing to overwrite replay report");
    std::size_t roots,iterations;double tolerance;std::vector<std::complex<double>> initial;
    auto op=sdft::PlaneWaveOperator::load_solver_snapshot(argv[1],roots,iterations,tolerance,initial);
    const auto stored_iterations=iterations;
    if(argc>=4){std::size_t end;const std::string value=argv[3];iterations=std::stoull(value,&end);
      if(end!=value.size()||!iterations||iterations>1000000||value[0]=='-')throw std::invalid_argument("invalid replay iteration limit");}
    const auto start=std::chrono::steady_clock::now();
    auto h=op->hamiltonian();const auto s=op->overlap();
    const std::string guard_path=std::string(argv[1])+".guards";
    if(std::filesystem::exists(guard_path))
      h.initial_search_guards=sdft::load_search_guards(guard_path,sdft::file_sha256(argv[1]),h.dimension,roots);
    const auto result=sdft::solve_generalized_hermitian_davidson(h,s,roots,iterations,tolerance,initial);
    // Independent fresh application to the returned vectors, without cached
    // subspace products. Never convert an iteration-limited solve to success.
    std::vector<std::complex<double>> hx,sx;
    h.apply_block(result.eigenvectors,roots,hx,true);s.apply_block(result.eigenvectors,roots,sx,true);
    double maximum=0,normalization_error=0;
    for(std::size_t band=0;band<roots;++band) {
      long double square=0,metric=0;
      for(std::size_t g=0;g<h.dimension;++g){const auto i=band*h.dimension+g;
        const double re=std::fma(-result.eigenvalues[band],sx[i].real(),hx[i].real());
        const double im=std::fma(-result.eigenvalues[band],sx[i].imag(),hx[i].imag());
        square+=static_cast<long double>(re)*re+static_cast<long double>(im)*im;
        metric+=static_cast<long double>(result.eigenvectors[i].real())*sx[i].real()
               +static_cast<long double>(result.eigenvectors[i].imag())*sx[i].imag();
      }
      maximum=std::max(maximum,std::sqrt(static_cast<double>(square)));
      normalization_error=std::max(normalization_error,std::abs(static_cast<double>(metric)-1));
    }
    const bool passed=result.converged&&maximum<=tolerance&&normalization_error<=1e-12;
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream f(argv[2]);f<<std::setprecision(17)
      <<"{\"schema_version\":\"sparse-dft-solver-replay-1\",\"scientific_scf_result\":false,\"passed\":"<<(passed?"true":"false")
      <<",\"dimension\":"<<h.dimension<<",\"roots\":"<<roots<<",\"stored_max_iterations\":"<<stored_iterations
      <<",\"max_iterations\":"<<iterations<<",\"iterations\":"<<result.iterations<<",\"tolerance\":"<<tolerance
      <<",\"solver_converged\":"<<(result.converged?"true":"false")<<",\"fresh_max_residual\":"<<maximum
      <<",\"s_normalization_error\":"<<normalization_error<<",\"seconds\":"<<seconds<<",\"residuals\":[";
    for(std::size_t i=0;i<roots;++i)f<<(i?",":"")<<result.residual_norms[i];
    f<<"],\"eigenvalues_hartree\":[";
    for(std::size_t i=0;i<roots;++i)f<<(i?",":"")<<result.eigenvalues[i];
    f<<"]}\n";f.close();if(!f)throw std::runtime_error("replay report write failed");
    sdft::write_performance_report(std::string(argv[2])+".performance.json");
    if(argc==5) {
      op->save_solver_snapshot(argv[4],roots,iterations,tolerance,result.eigenvectors);
      if(!result.search_guard_vectors.empty())
        sdft::save_search_guards(std::string(argv[4])+".guards",sdft::file_sha256(argv[4]),
                                 h.dimension,result.search_guard_vectors);
    }
    std::cout<<"solver_replay "<<(passed?"PASS":"FAIL")<<" residual="<<maximum<<" tolerance="<<tolerance<<" iterations="<<result.iterations<<'\n';
    return passed?0:2;
  }catch(const std::exception& e){std::cerr<<"SDFT_REPLAY_ERROR: "<<e.what()<<'\n';return 1;}
}
