// Original portable two-float GEMM: about 48 significand bits within a bounded
// exponent range. This is NOT IEEE FP64 or a promise of bitwise FP64 equality.
static const char* compensatedSource=R"metal(
#include <metal_stdlib>
using namespace metal;
struct Shape { uint m,n,k,sa,sb,sc,is_complex; };
inline float2 add2(float2 a,float2 b) {
    float s=a.x+b.x;
    float v=s-a.x;
    float e=(a.x-(s-v))+(b.x-v);
    e=e+a.y+b.y;
    float h=s+e;
    return float2(h,e-(h-s));
}
inline float2 mul2(float2 a,float2 b) {
    float p=a.x*b.x;
    float e=fma(a.x,b.x,-p);
    e=e+a.x*b.y+a.y*b.x;
    float h=p+e;
    return float2(h,e-(h-p));
}
// IEEE binary64 storage -> hi/lo FP32 without binary64 arithmetic. Mantissa
// split and ties-to-even rounding are exact integer operations. This is a
// conversion kernel, not an emulated full IEEE binary64 arithmetic library.
inline float2 split_ieee64(ulong bits){
    uint field=uint((bits>>52)&0x7fful);
    if(field==0)return float2(0); // all double subnormals are below FP32 range
    if(field==0x7ffu)return float2(as_type<float>(0x7fc00000u),0);
    ulong mantissa=(bits&0xffffffffffffful)|0x10000000000000ul;
    long exponent=long(field)-1023;
    ulong rounded=(mantissa+0xffffffful+((mantissa>>29)&1ul))>>29;
    long remainder=long(mantissa)-long(rounded<<29);
    float hi=ldexp(float(rounded),int(exponent)-23);
    float lo=ldexp(float(remainder),int(exponent)-52);
    return (bits>>63)?-float2(hi,lo):float2(hi,lo);
}
struct PackShape{uint rows,cols,ld,adjoint;};
kernel void ieee64_pack(const device ulong2* input [[buffer(0)]],device float4* output [[buffer(1)]],
    constant PackShape& p [[buffer(2)]],uint index [[thread_position_in_grid]]){
    if(index>=p.rows*p.cols)return;
    uint row=index/p.cols,col=index%p.cols;
    ulong2 value=input[p.adjoint?col+row*p.ld:row+col*p.ld];
    if(p.adjoint)value.y^=0x8000000000000000ul;
    output[index]=float4(split_ieee64(value.x),split_ieee64(value.y));
}
kernel void compensated_gemm(const device float4* a [[buffer(0)]],
                              const device float4* b [[buffer(1)]],
                              device float4* c [[buffer(2)]],
                              constant Shape& p [[buffer(3)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float4 aa[16][16];
    threadgroup float4 bb[16][16];
    uint row=group.y*16+tid.y,col=group.x*16+tid.x;
    float2 real=float2(0),imag=float2(0);
    for(uint start=0;start<p.k;start+=16) {
        uint qa=start+tid.x,qb=start+tid.y;
        aa[tid.y][tid.x]=(row<p.m&&qa<p.k)?a[row*p.sa+qa]:float4(0);
        bb[tid.y][tid.x]=(qb<p.k&&col<p.n)?b[qb*p.sb+col]:float4(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<16;++q) {
            float4 x=aa[tid.y][q],y=bb[q][tid.x];
            real=add2(real,mul2(x.xy,y.xy));
            if(p.is_complex) {
                real=add2(real,-mul2(x.zw,y.zw));
                imag=add2(imag,mul2(x.xy,y.zw));
                imag=add2(imag,mul2(x.zw,y.xy));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<p.m&&col<p.n)c[row*p.sc+col]=float4(real,imag);
}
kernel void compensated_gemm_2x2(const device float4* a [[buffer(0)]],
                              const device float4* b [[buffer(1)]],device float4* c [[buffer(2)]],
                              constant Shape& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float4 aa[16][16],bb[16][16];
    float2 real[4]={float2(0),float2(0),float2(0),float2(0)};
    float2 imag[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint start=0;start<p.k;start+=16){
        for(uint r=0;r<2;++r)for(uint s=0;s<2;++s){
            uint ri=tid.y+r*8,ci=tid.x+s*8,row=group.y*16+ri,col=group.x*16+ci;
            aa[ri][ci]=(row<p.m&&start+ci<p.k)?a[row*p.sa+start+ci]:float4(0);
            bb[ri][ci]=(start+ri<p.k&&col<p.n)?b[(start+ri)*p.sb+col]:float4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<16;++q)for(uint r=0;r<2;++r)for(uint s=0;s<2;++s){
            uint v=r*2+s;float4 x=aa[tid.y+r*8][q],y=bb[q][tid.x+s*8];
            real[v]=add2(real[v],mul2(x.xy,y.xy));real[v]=add2(real[v],-mul2(x.zw,y.zw));
            imag[v]=add2(imag[v],mul2(x.xy,y.zw));imag[v]=add2(imag[v],mul2(x.zw,y.xy));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint r=0;r<2;++r)for(uint s=0;s<2;++s){uint row=group.y*16+tid.y+r*8,col=group.x*16+tid.x+s*8;
        if(row<p.m&&col<p.n)c[row*p.sc+col]=float4(real[r*2+s],imag[r*2+s]);}
}
// FP32 benchmark only. It has the same tiling/layout and dispatch overhead,
// making arithmetic cost directly comparable to the compensated kernel.
kernel void fp32_gemm(const device float4* a [[buffer(0)]],
                       const device float4* b [[buffer(1)]],device float4* c [[buffer(2)]],
                       constant Shape& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float2 aa[16][16],bb[16][16];
    uint row=group.y*16+tid.y,col=group.x*16+tid.x;float2 sum=float2(0);
    for(uint start=0;start<p.k;start+=16){uint qa=start+tid.x,qb=start+tid.y;
        float4 av=(row<p.m&&qa<p.k)?a[row*p.sa+qa]:float4(0);
        float4 bv=(qb<p.k&&col<p.n)?b[qb*p.sb+col]:float4(0);
        aa[tid.y][tid.x]=av.xz;bb[tid.y][tid.x]=bv.xz;threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint q=0;q<16;++q){float2 x=aa[tid.y][q],y=bb[q][tid.x];
            sum.x=fma(x.x,y.x,sum.x);sum.x=fma(-x.y,y.y,sum.x);
            sum.y=fma(x.x,y.y,sum.y);sum.y=fma(x.y,y.x,sum.y);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<p.m&&col<p.n)c[row*p.sc+col]=float4(sum.x,0,sum.y,0);
}
)metal";
