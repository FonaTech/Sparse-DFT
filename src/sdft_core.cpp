#include "sdft_core.hpp"
#include "paw_header_radius.hpp"
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <cmath>
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#endif

namespace sdft {
double dot(Vec3 a, Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Vec3 cross(Vec3 a, Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double determinant(const std::array<Vec3,3>& l){return dot(l[0],cross(l[1],l[2]));}
static std::string value(const std::string& text,const std::string& key){
  auto p=text.find('"'+key+'"'); if(p==std::string::npos) return {};
  p=text.find(':',p); if(p==std::string::npos) return {}; ++p;
  while(p<text.size() && (text[p]==' '||text[p]=='\n'||text[p]=='\r'||text[p]=='\t')) ++p;
  if(p<text.size() && text[p]=='"'){++p; auto e=text.find('"',p); return e==std::string::npos?std::string{}:text.substr(p,e-p);}
  auto e=text.find_first_of(",}\n",p); return text.substr(p,e==std::string::npos?text.size()-p:e-p);
}
static std::string nested_value(const std::string& text,const std::string& parent,const std::string& key){
  auto p=text.find('"'+parent+'"'); if(p==std::string::npos) return {};
  auto b=text.find('{',p), e=text.find('}',b); if(b==std::string::npos||e==std::string::npos) return {};
  return value(text.substr(b,e-b+1),key);
}
PawCacheMetadata read_paw_cache_metadata(const std::filesystem::path& path){
  std::ifstream f(path); if(!f) throw std::runtime_error("cannot open PAW metadata: "+path.string());
  std::stringstream ss; ss<<f.rdbuf(); auto t=ss.str(); PawCacheMetadata m;
  m.record_id=value(t,"record_id"); m.element=nested_value(t,"element","symbol"); m.family=nested_value(t,"variant","family");
  m.xc=nested_value(t,"potcar","exchange_correlation");
  m.potcar_sha256=nested_value(t,"POTCAR","sha256"); m.psctr_sha256=nested_value(t,"PSCTR","sha256");
  auto z=nested_value(t,"potcar","valence_electrons"); auto e=nested_value(t,"potcar","enmax_ev");
  try {m.zval=std::stod(z); m.enmax_ev=std::stod(e);} catch(...) {throw std::runtime_error("invalid PAW numeric metadata");}
  if(m.record_id.empty()||m.element.empty()) throw std::runtime_error("incomplete PAW metadata"); return m;
}
std::string file_sha256(const std::filesystem::path& path){
  if(!std::filesystem::is_regular_file(path))
    throw std::runtime_error("SHA-256 input is not a regular file: "+path.string());
  std::ifstream f(path,std::ios::binary);
  if(!f) throw std::runtime_error("cannot open SHA-256 input: "+path.string());
  char buffer[1<<16];
#ifdef __APPLE__
  CC_SHA256_CTX context;
  if(!CC_SHA256_Init(&context)) throw std::runtime_error("SHA-256 initialization failed");
#else
  SHA256_CTX context;
  if(!SHA256_Init(&context)) throw std::runtime_error("SHA-256 initialization failed");
#endif
  while(f){
    f.read(buffer,sizeof(buffer));
    const auto size=f.gcount();
#ifdef __APPLE__
    if(size>0&&!CC_SHA256_Update(&context,buffer,static_cast<CC_LONG>(size)))
#else
    if(size>0&&!SHA256_Update(&context,buffer,static_cast<std::size_t>(size)))
#endif
      throw std::runtime_error("SHA-256 update failed");
  }
  if(f.bad()||!f.eof()) throw std::runtime_error("cannot read SHA-256 input: "+path.string());
  unsigned char output[32];
#ifdef __APPLE__
  if(!CC_SHA256_Final(output,&context))
#else
  if(!SHA256_Final(output,&context))
#endif
    throw std::runtime_error("SHA-256 finalization failed");
  std::ostringstream result;result<<std::hex;
  for(unsigned char byte:output){result.width(2);result.fill('0');result<<static_cast<unsigned int>(byte);}
  return result.str();
}
bool file_sha256_matches(const std::filesystem::path& path,const std::string& expected){
  try{return file_sha256(path)==expected;}
  catch(const std::runtime_error&){return false;}
}

namespace {
template <typename T>
T read_scalar(std::ifstream& f, const char* label) {
  T value{};
  f.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!f) throw std::runtime_error(std::string("truncated PAW binary cache while reading ") + label);
  return value;
}
std::vector<double> read_vector(std::ifstream& f, const char* label) {
  auto n = read_scalar<std::uint32_t>(f, label);
  if (n > 100000000U) throw std::runtime_error(std::string("unreasonable PAW vector length for ") + label);
  std::vector<double> values(n);
  if (n) {
    f.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n * sizeof(double)));
    if (!f) throw std::runtime_error(std::string("truncated PAW vector for ") + label);
  }
  return values;
}
void require_size(const std::vector<double>& values, std::size_t expected, const char* label) {
  if (values.size() != expected) throw std::runtime_error(std::string("PAW cache length mismatch for ") + label);
}
}

PawNumericCache read_paw_numeric_cache(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open PAW numeric cache: " + path.string());
  char magic[8]{};
  f.read(magic, sizeof(magic));
  if (!f || std::memcmp(magic, "SDFTPAW1", 8) != 0) throw std::runtime_error("invalid Sparse-DFT PAW binary magic");
  PawNumericCache cache;
  cache.version = read_scalar<std::uint32_t>(f, "version");
  cache.main_grid_size = read_scalar<std::uint32_t>(f, "main_grid_size");
  cache.radial_points = read_scalar<std::uint32_t>(f, "radial_points");
  auto channel_count = read_scalar<std::uint32_t>(f, "channel_count");
  auto projector_count = read_scalar<std::uint32_t>(f, "projector_count");
  if ((cache.version < 1 || cache.version > 4) || cache.main_grid_size == 0 || cache.radial_points == 0 || channel_count == 0 || projector_count == 0)
    throw std::runtime_error("invalid PAW numeric cache header");
  if (cache.version >= 3) cache.radial_rmax = read_scalar<double>(f, "radial_rmax");
  if (cache.version >= 4) {
    cache.atomic_reference_energy_ev = read_scalar<double>(f, "atomic_reference_energy_ev");
    cache.core_xc_outside_ev = read_scalar<double>(f, "core_xc_outside_ev");
    cache.atomic_reference_present = true;
    cache.kinetic_error_cutoff_ev = read_vector(f, "kinetic_error_cutoff_ev");
    cache.kinetic_error_ev = read_vector(f, "kinetic_error_ev");
  }
  cache.local_header = read_vector(f, "local_header");
  if (cache.local_header.size() > 1) throw std::runtime_error("PAW local header contains multiple values");
  cache.local_potential = read_vector(f, "local_potential");
  cache.core_charge_partial = read_vector(f, "core_charge_partial");
  cache.kinetic_energy_partial = read_vector(f, "kinetic_energy_partial");
  cache.atomic_pseudo_charge = read_vector(f, "atomic_pseudo_charge");
  cache.atomic_pseudo_charge_tail = read_vector(f, "atomic_pseudo_charge_tail");
  require_size(cache.local_potential, cache.main_grid_size, "local_potential");
  if (!cache.core_charge_partial.empty()) require_size(cache.core_charge_partial, cache.main_grid_size, "core_charge_partial");
  if (!(cache.kinetic_energy_partial.size() == cache.main_grid_size || cache.kinetic_energy_partial.size() % cache.main_grid_size == 0))
    throw std::runtime_error("PAW kinetic partial grid length mismatch");
  require_size(cache.atomic_pseudo_charge, cache.main_grid_size, "atomic_pseudo_charge");

  cache.nonlocal_channels.reserve(channel_count);
  std::size_t counted_projectors = 0;
  for (std::uint32_t c = 0; c < channel_count; ++c) {
    PawNumericChannel channel;
    channel.angular_momentum = read_scalar<std::int32_t>(f, "angular_momentum");
    channel.nprojectors = read_scalar<std::int32_t>(f, "nprojectors");
    channel.cutoff_radius = read_scalar<double>(f, "cutoff_radius");
    channel.reciprocal_cutoff = cache.version >= 2 ? read_scalar<double>(f, "reciprocal_cutoff") : 0.0;
    if (channel.nprojectors < 1 || channel.nprojectors > 64 || channel.angular_momentum < 0)
      throw std::runtime_error("invalid PAW nonlocal channel dimensions");
    counted_projectors += static_cast<std::size_t>(channel.nprojectors);
    channel.d_matrix.resize(channel.nprojectors);
    for (auto& row : channel.d_matrix) { row = read_vector(f, "d_matrix"); require_size(row, channel.nprojectors, "d_matrix"); }
    channel.reciprocal_projectors.resize(channel.nprojectors);
    channel.real_projectors.resize(channel.nprojectors);
    for (auto& values : channel.reciprocal_projectors) { values = read_vector(f, "reciprocal_projector"); require_size(values, 100, "reciprocal_projector"); }
    for (auto& values : channel.real_projectors) { values = read_vector(f, "real_projector"); require_size(values, 100, "real_projector"); }
    cache.nonlocal_channels.push_back(std::move(channel));
  }
  if (counted_projectors != projector_count) throw std::runtime_error("PAW projector count mismatch");

  cache.radial_grid = read_vector(f, "radial_grid");
  cache.ae_potential = read_vector(f, "ae_potential");
  cache.core_charge_density = read_vector(f, "core_charge_density");
  cache.kinetic_energy_density = read_vector(f, "kinetic_energy_density");
  cache.kinetic_energy_density_pseudized = read_vector(f, "kinetic_energy_density_pseudized");
  cache.local_pseudopotential_core = read_vector(f, "local_pseudopotential_core");
  cache.pseudopotential_valence = read_vector(f, "pseudopotential_valence");
  cache.core_charge_density_pseudized = read_vector(f, "core_charge_density_pseudized");
  cache.augmentation_coefficients = read_vector(f, "augmentation_coefficients");
  cache.occupancies = read_vector(f, "occupancies");
  for (const auto* values : {&cache.radial_grid, &cache.ae_potential, &cache.core_charge_density,
                             &cache.local_pseudopotential_core, &cache.pseudopotential_valence,
                             &cache.core_charge_density_pseudized}) require_size(*values, cache.radial_points, "radial array");
  if (!cache.kinetic_energy_density.empty()) require_size(cache.kinetic_energy_density, cache.radial_points, "kinetic_energy_density");
  if (!cache.kinetic_energy_density_pseudized.empty()) require_size(cache.kinetic_energy_density_pseudized, cache.radial_points, "kinetic_energy_density_pseudized");
  if (cache.augmentation_coefficients.size() != projector_count * projector_count || cache.occupancies.size() != projector_count * projector_count)
    throw std::runtime_error("PAW augmentation tensor length mismatch");
  cache.pseudo_wavefunctions.resize(projector_count);
  cache.ae_wavefunctions.resize(projector_count);
  for (auto& values : cache.pseudo_wavefunctions) { values = read_vector(f, "pseudo_wavefunction"); require_size(values, cache.radial_points, "pseudo_wavefunction"); }
  for (auto& values : cache.ae_wavefunctions) { values = read_vector(f, "ae_wavefunction"); require_size(values, cache.radial_points, "ae_wavefunction"); }
  if (cache.version < 3 && !cache.radial_grid.empty()) cache.radial_rmax = cache.radial_grid.back();
  cache.radial_rdep = cache.radial_grid.empty() ? cache.radial_rmax : cache.radial_grid.back();
  // RDEP is a POTCAR/PSCTR header parameter rather than part of the v3
  // numeric payload. Recover it from the restored sibling POTCAR so old and
  // new caches use the same truncated AE-norm radius without changing the file
  // format. A missing text file has a conservative full-mesh fallback.
  const auto potcar = path.parent_path() / "POTCAR";
  std::ifstream header(potcar);
  if (header) {
    std::string line;
    while (std::getline(header, line)) {
      if (line.find("END of PSCTR") != std::string::npos) break;
      if (cache.version < 4) {
        const auto scalar = [&](const char* tag, double& target) {
          const auto at = line.find(tag);
          if (at == std::string::npos) return false;
          const auto eq = line.find('=', at);
          if (eq == std::string::npos) return false;
          const char* begin = line.c_str() + eq + 1;
          char* end = nullptr;
          const double v = std::strtod(begin, &end);
          if (end == begin || !std::isfinite(v))
            throw std::runtime_error(std::string("invalid PAW header scalar ") + tag);
          target = v; return true;
        };
        if (scalar("EATOM", cache.atomic_reference_energy_ev)) cache.atomic_reference_present = true;
        scalar("DEXC", cache.core_xc_outside_ev);
        if (line.find("Error from kinetic energy argument") != std::string::npos) {
          std::string count_line, step_line;
          if (!std::getline(header, count_line) || !std::getline(header, step_line))
            throw std::runtime_error("truncated PAW kinetic error header");
          const auto count_eq = count_line.find('='), step_eq = step_line.find('=');
          if (count_eq == std::string::npos || step_eq == std::string::npos)
            throw std::runtime_error("invalid PAW kinetic error header");
          const int count = std::stoi(count_line.substr(count_eq + 1));
          double start{}, factor{};
          std::istringstream steps(step_line.substr(step_eq + 1)); steps >> start >> factor;
          if (!steps || count < 2 || count > 10000 || start <= 0.0 || factor <= 1.0)
            throw std::runtime_error("invalid PAW kinetic error dimensions");
          for (int i = 0; i < count; ++i) {
            double v{};
            if (!(header >> v)) throw std::runtime_error("truncated PAW kinetic error data");
            cache.kinetic_error_ev.push_back(v);
            cache.kinetic_error_cutoff_ev.push_back(start * std::pow(factor, i));
          }
        }
      }
      if (const auto radius = paw_header_rdep_angstrom(line))
        cache.radial_rdep = *radius;
    }
  }
  if (!std::isfinite(cache.atomic_reference_energy_ev) || !std::isfinite(cache.core_xc_outside_ev) ||
      cache.kinetic_error_ev.size() != cache.kinetic_error_cutoff_ev.size())
    throw std::runtime_error("invalid PAW atomic reference data");
  for (std::size_t i = 0; i < cache.kinetic_error_ev.size(); ++i)
    if (!std::isfinite(cache.kinetic_error_ev[i]) || !std::isfinite(cache.kinetic_error_cutoff_ev[i]) ||
        cache.kinetic_error_cutoff_ev[i] <= 0.0 ||
        (i && cache.kinetic_error_cutoff_ev[i] <= cache.kinetic_error_cutoff_ev[i - 1]))
      throw std::runtime_error("invalid PAW kinetic error table");
  char extra{};
  if (f.read(&extra, 1)) throw std::runtime_error("trailing bytes in PAW numeric cache");
  return cache;
}

double paw_numeric_checksum(const PawNumericCache& cache) {
  double sum = 0.0;
  auto add = [&sum](const std::vector<double>& values) { for (double value : values) sum += value; };
  add(cache.local_potential); add(cache.core_charge_partial); add(cache.atomic_pseudo_charge);
  add(cache.radial_grid); add(cache.ae_potential); add(cache.augmentation_coefficients);
  for (const auto& channel : cache.nonlocal_channels) {
    for (const auto& row : channel.d_matrix) add(row);
    for (const auto& row : channel.reciprocal_projectors) add(row);
    for (const auto& row : channel.real_projectors) add(row);
  }
  return sum;
}
}
