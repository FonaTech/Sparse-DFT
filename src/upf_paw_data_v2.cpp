#include "upf_paw_data.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#endif

namespace sdft {
namespace {
constexpr std::uint64_t maximum_bytes=64*1024*1024;
constexpr double pi=3.1415926535897932384626433832795;

std::string bytes(std::istream& f,std::size_t n) {
  if(n>maximum_bytes)throw std::invalid_argument("UPF container field exceeds bound");
  std::string value(n,'\0');f.read(value.data(),static_cast<std::streamsize>(n));
  if(!f)throw std::invalid_argument("truncated native UPF container");
  return value;
}

template<class T> T integer(std::istream& f) {
  static_assert(std::is_unsigned<T>::value,"unsigned little-endian integer required");
  const auto raw=bytes(f,sizeof(T));T value=0;
  for(std::size_t i=0;i<sizeof(T);++i)value|=T(static_cast<unsigned char>(raw[i]))<<(8*i);
  return value;
}

double real(std::istream& f) {
  const auto bits=integer<std::uint64_t>(f);double value;
  static_assert(sizeof(value)==sizeof(bits),"native UPF requires binary64");
  std::memcpy(&value,&bits,sizeof(value));
  if(!std::isfinite(value))throw std::invalid_argument("nonfinite native UPF value");
  return value;
}

std::string string(std::istream& f) {
  const auto n=integer<std::uint32_t>(f);
  if(n>4096)throw std::invalid_argument("oversized UPF string");
  return bytes(f,n);
}

std::string digest(const std::string& raw) {
  std::array<unsigned char,32> hash{};
#ifdef __APPLE__
  CC_SHA256(raw.data(),static_cast<CC_LONG>(raw.size()),hash.data());
#else
  SHA256(reinterpret_cast<const unsigned char*>(raw.data()),raw.size(),hash.data());
#endif
  return std::string(reinterpret_cast<const char*>(hash.data()),hash.size());
}

std::string hex(const std::string& raw) {
  std::ostringstream out;out<<std::hex<<std::setfill('0');
  for(unsigned char c:raw)out<<std::setw(2)<<unsigned(c);
  return out.str();
}

void require_size(const UpfPawData& data,const std::string& name,std::size_t n) {
  if(data.array(name).size()!=n)throw std::invalid_argument("native UPF array dimension mismatch: "+name);
}

void require_g(double g) {
  if(!std::isfinite(g)||g<0)throw std::invalid_argument("invalid UPF reciprocal magnitude");
}

// Compensated summation limits cancellation in short-range Coulomb transforms.
struct Sum {
  double total{},correction{};
  void add(double x) {const double t=total+x;
    correction+=std::abs(total)>=std::abs(x)?(total-t)+x:(x-t)+total;total=t;}
  double value()const{return total+correction;}
};
}

const std::vector<double>& UpfPawData::array(const std::string& name) const {
  auto found=arrays.find(name);
  if(found==arrays.end())throw std::invalid_argument("missing native UPF array: "+name);
  return found->second;
}

UpfPawData read_native_upf_paw(const std::filesystem::path& path) {
  std::ifstream f(path,std::ios::binary);
  if(!f)throw std::invalid_argument("cannot open native UPF dataset");
  if(bytes(f,8)!="SDFTUPF1")throw std::invalid_argument("unsupported native UPF container");
  const auto version=integer<std::uint32_t>(f);
  if(version!=1&&version!=2)throw std::invalid_argument("unsupported native UPF version");
  UpfPawData data;data.element=string(f);data.functional=string(f);data.relativistic=string(f);
  if(data.functional!="PBE"||data.relativistic!="scalar")
    throw std::invalid_argument("native UPF development requires scalar PBE PAW");
  const auto original_digest=bytes(f,32);data.original_sha256=hex(original_digest);
  data.z_valence=real(f);data.core_energy_ry=real(f);data.augmentation_epsilon=real(f);
  data.augmentation_count=integer<std::uint32_t>(f);const auto n=integer<std::uint32_t>(f);
  if(n<1||n>64||data.z_valence<=0||data.z_valence>118||data.augmentation_epsilon<0)
    throw std::invalid_argument("invalid native UPF projector/valence count");
  data.integration_count=data.augmentation_count;
  for(std::size_t i=0;i<n;++i) {
    const auto l=integer<std::uint32_t>(f),k=integer<std::uint32_t>(f);
    if(l>6||k<3)throw std::invalid_argument("invalid native UPF projector index");
    data.projector_l.push_back(l);data.projector_cutoff.push_back(k);
    data.integration_count=std::max(data.integration_count,std::size_t(k));
  }
  if(version==2) {
    const auto nchi=integer<std::uint32_t>(f);
    if(nchi>64)throw std::invalid_argument("oversized UPF atomic-wavefunction count");
    for(std::size_t i=0;i<nchi;++i) {
      const auto l=integer<std::uint32_t>(f);const double occupation=real(f);
      if(l>6)throw std::invalid_argument("invalid UPF atomic angular momentum");
      data.atomic_l.push_back(l);data.atomic_occupations.push_back(occupation);
    }
  }
  const auto section_count=integer<std::uint32_t>(f);
  if(section_count<10||section_count>10000)throw std::invalid_argument("invalid native UPF section count");
  std::uint64_t total_values=0;
  for(std::size_t s=0;s<section_count;++s) {
    const auto name=string(f);const auto count=integer<std::uint64_t>(f);
    if(count>maximum_bytes/8||total_values+count>maximum_bytes/8||data.arrays.count(name))
      throw std::invalid_argument("oversized or duplicate native UPF numeric section");
    total_values+=count;const auto expected=bytes(f,32);const auto raw=bytes(f,count*8);
    if(digest(raw)!=expected)throw std::invalid_argument("native UPF section SHA mismatch");
    std::istringstream stream(raw);std::vector<double> values(count);
    for(auto& x:values)x=real(stream);
    data.arrays.emplace(name,std::move(values));
  }
  data.metadata_json=bytes(f,integer<std::uint64_t>(f));
  data.original_xml=bytes(f,integer<std::uint64_t>(f));
  if(digest(data.original_xml)!=original_digest)throw std::invalid_argument("original UPF XML SHA mismatch");
  if(f.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing native UPF container bytes");
  const auto& r=data.array("UPF/PP_MESH/PP_R");const auto mesh=r.size();
  if(mesh<3||data.augmentation_count<3||data.integration_count>mesh)
    throw std::invalid_argument("native UPF radial integration range invalid");
  require_size(data,"UPF/PP_MESH/PP_RAB",mesh);
  const auto& rab=data.array("UPF/PP_MESH/PP_RAB");
  for(std::size_t i=0;i<mesh;++i)if(r[i]<0||rab[i]<=0||(i&&r[i]<=r[i-1]))
    throw std::invalid_argument("native UPF radial mesh/Jacobian invalid");
  for(const auto name:{"UPF/PP_LOCAL","UPF/PP_RHOATOM","UPF/PP_PAW/PP_AE_NLCC","UPF/PP_PAW/PP_AE_VLOC"})
    require_size(data,name,mesh);
  for(const auto name:{"UPF/PP_NONLOCAL/PP_DIJ","UPF/PP_NONLOCAL/PP_AUGMENTATION/PP_Q"}) {
    require_size(data,name,n*n);const auto& matrix=data.array(name);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)
      if(std::abs(matrix[i+n*j]-matrix[j+n*i])>1e-12*std::max({1.,std::abs(matrix[i+n*j]),std::abs(matrix[j+n*i])}))
        throw std::invalid_argument("native UPF D/Q symmetry failure");
  }
  for(std::size_t i=0;i<n;++i)for(const auto prefix:{"UPF/PP_NONLOCAL/PP_BETA.","UPF/PP_FULL_WFC/PP_AEWFC.","UPF/PP_FULL_WFC/PP_PSWFC."})
    require_size(data,std::string(prefix)+std::to_string(i+1),mesh);
  return data;
}

std::vector<double> upf_simpson_weights(const std::vector<double>& rab,std::size_t count) {
  if(count<3||count>rab.size())throw std::invalid_argument("invalid UPF integration extent");
  std::vector<double> w(rab.size(),0.);
  for(std::size_t i=0;i<rab.size();++i)if(!std::isfinite(rab[i])||rab[i]<=0)
    throw std::invalid_argument("invalid UPF radial Jacobian");
  for(std::size_t i=0;i<count;++i)w[i]=rab[i]*(i==0||i==count-1?1.:i%2?4.:2.)/3.;
  if(count%2==0){w[count-3]-=rab[count-3]/12.;w[count-2]+=rab[count-2]/3.;w[count-1]=rab[count-1]*5./12.;}
  return w;
}

double upf_spherical_bessel(int l,double x) {
  if(l<0||l>12||!std::isfinite(x))throw std::invalid_argument("invalid UPF Bessel input");
  if(x<0)return (l%2?-1.:1.)*upf_spherical_bessel(l,-x);
  if(x<.5) {
    double term=1.;for(int k=1;k<=l;++k)term*=x/(2*k+1);double result=term;
    for(int k=1;k<100;++k){term*=-x*x/(2.*k*(2*l+2*k+1));result+=term;
      if(std::abs(term)<1e-17*std::max(1.,std::abs(result)))break;}
    return result;
  }
  const double j0=std::sin(x)/x,j1=std::sin(x)/(x*x)-std::cos(x)/x;
  if(x>l+1){if(l==0)return j0;double lower=j0,current=j1;
    for(int k=1;k<l;++k){const double next=(2*k+1)*current/x-lower;lower=current;current=next;}return current;}
  const int top=l+40;std::array<double,54> v{};v[top]=1.;
  for(int k=top;k>0;--k){v[k-1]=(2*k+1)*v[k]/x-v[k+1];
    if(std::abs(v[k-1])>1e100)for(int q=k-1;q<=top+1;++q)v[q]*=1e-100;}
  return v[l]*(std::abs(j0)>std::abs(j1)?j0/v[0]:j1/v[1]);
}

double upf_spherical_bessel_derivative(int l,double x) {
  if(l<0||l>12||!std::isfinite(x))throw std::invalid_argument("invalid UPF Bessel derivative input");
  if(std::abs(x)<1e-12)return l==1?1./3.:0.;
  return l?upf_spherical_bessel(l-1,x)-(l+1)*upf_spherical_bessel(l,x)/x:-upf_spherical_bessel(1,x);
}

double upf_projector_form(const UpfPawData& data,std::size_t i,double g,bool derivative) {
  require_g(g);if(i>=data.projector_count())throw std::invalid_argument("UPF projector index outside range");
  const auto& r=data.array("UPF/PP_MESH/PP_R");const auto& beta=data.array("UPF/PP_NONLOCAL/PP_BETA."+std::to_string(i+1));
  const auto w=upf_simpson_weights(data.array("UPF/PP_MESH/PP_RAB"),data.integration_count);Sum sum;
  for(std::size_t k=0;k<data.integration_count;++k)sum.add(w[k]*r[k]*beta[k]*(derivative?
      r[k]*upf_spherical_bessel_derivative(data.projector_l[i],g*r[k]):upf_spherical_bessel(data.projector_l[i],g*r[k])));
  return sum.value();
}

std::vector<double> upf_augmentation_radial(const UpfPawData& data,std::size_t i,std::size_t j,int l) {
  const auto n=data.projector_count();if(i>=n||j>=n||l<0||l>12)throw std::invalid_argument("UPF augmentation index outside range");
  const auto mesh=data.array("UPF/PP_MESH/PP_R").size();const int li=data.projector_l[i],lj=data.projector_l[j];
  if(l<std::abs(li-lj)||l>li+lj||(li+lj-l)%2)return std::vector<double>(mesh,0.);
  const auto& multipoles=data.array("UPF/PP_NONLOCAL/PP_AUGMENTATION/PP_MULTIPOLES");
  const auto index=i+n*j+n*n*std::size_t(l);
  if(multipoles.size()%(n*n)||index>=multipoles.size())
    throw std::invalid_argument("UPF augmentation multipole dimensions invalid");
  if(std::abs(multipoles[index])<data.augmentation_epsilon)return std::vector<double>(mesh,0.);
  auto q=data.array("UPF/PP_NONLOCAL/PP_AUGMENTATION/PP_QIJL."+std::to_string(std::min(i,j)+1)+"."+
                    std::to_string(std::max(i,j)+1)+"."+std::to_string(l));
  if(q.size()!=mesh)throw std::invalid_argument("UPF augmentation mesh mismatch");
  std::fill(q.begin()+data.augmentation_count,q.end(),0.);return q;
}

double upf_augmentation_form(const UpfPawData& data,std::size_t i,std::size_t j,int l,double g,bool derivative) {
  require_g(g);const auto q=upf_augmentation_radial(data,i,j,l);const auto& r=data.array("UPF/PP_MESH/PP_R");
  const auto w=upf_simpson_weights(data.array("UPF/PP_MESH/PP_RAB"),data.integration_count);Sum sum;
  for(std::size_t k=0;k<data.integration_count;++k)sum.add(w[k]*q[k]*(derivative?r[k]*upf_spherical_bessel_derivative(l,g*r[k]):upf_spherical_bessel(l,g*r[k])));
  return sum.value();
}

double upf_local_form_ry(const UpfPawData& data,double g,bool derivative) {
  require_g(g);if(derivative&&g==0)throw std::invalid_argument("UPF local derivative requires nonzero G");
  const auto& r=data.array("UPF/PP_MESH/PP_R");const auto& v=data.array("UPF/PP_LOCAL");
  // As in QE, this numerical short-range transform is limited to
  // the first mesh node beyond 10 Bohr (odd Simpson count). The raw UPF
  // large-r local tail contains small numerical offsets, not physical
  // long-range charge; integrating its r^2 amplification creates a false
  // G=0 shift. Original arrays and all onecenter mesh data remain intact.
  std::size_t count=r.size();
  for(std::size_t k=0;k<r.size();++k)if(r[k]>10.){count=k+1;break;}
  if(count%2==0)--count;
  const auto w=upf_simpson_weights(data.array("UPF/PP_MESH/PP_RAB"),count);Sum sum;
  for(std::size_t k=0;k<count;++k){
    const double short_range=r[k]*r[k]*v[k]+2*data.z_valence*r[k]*(g==0?1.:std::erf(r[k]));
    sum.add(w[k]*short_range*(derivative?r[k]*upf_spherical_bessel_derivative(0,g*r[k]):upf_spherical_bessel(0,g*r[k])));
  }
  double analytic=0.;
  if(g>0){const double gaussian=std::exp(-g*g/4);
    analytic=derivative?data.z_valence*gaussian*(1/g+4/(g*g*g)):-2*data.z_valence*gaussian/(g*g);
  }
  return 4*pi*(sum.value()+analytic);
}

} // namespace sdft
