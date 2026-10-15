// Mixed-radix DIT FFT. Twiddles are generated in host FP64 and split into
// two FP32 components; no FP32 sin/cos is used in the transform. Search
// arithmetic has a bounded exponent range and is certified later by FP64.
static const char* metalFftSource=R"metal(
struct FFTShape {uint count,columns,length,stride,radix,previous,backward,npw;};
inline float4 complex_mul2(float4 a,float4 b){
    return float4(add2(mul2(a.xy,b.xy),-mul2(a.zw,b.zw)),
                  add2(mul2(a.xy,b.zw),mul2(a.zw,b.xy)));
}
kernel void fft_scatter(const device ulong2* input [[buffer(0)]],
    const device int* lookup [[buffer(1)]],device float4* output [[buffer(2)]],
    constant FFTShape& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.count*p.columns)return;
    int g=lookup[i%p.count];
    if(g<0){output[i]=float4(0);return;}
    ulong2 bits=input[(i/p.count)*p.npw+uint(g)];
    output[i]=float4(split_ieee64(bits.x),split_ieee64(bits.y));
}
kernel void fft_permute(const device float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],const device uint* permutation [[buffer(2)]],
    constant FFTShape& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.count*p.columns)return;
    uint coordinate=(i/p.stride)%p.length;
    output[i]=input[i-coordinate*p.stride+permutation[coordinate]*p.stride];
}
kernel void fft_butterfly(const device float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],const device float4* twiddles [[buffer(2)]],
    constant FFTShape& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.count*p.columns)return;
    uint coordinate=(i/p.stride)%p.length,stage=p.radix*p.previous;
    uint position=coordinate%stage,offset=position%p.previous;
    uint base=i-position*p.stride;
    float2 re=float2(0),im=float2(0);
    for(uint j=0;j<p.radix;++j){
        float4 w=twiddles[(position*j)%stage];if(p.backward)w.zw=-w.zw;
        float4 value=complex_mul2(input[base+(offset+j*p.previous)*p.stride],w);
        re=add2(re,value.xy);im=add2(im,value.zw);
    }
    output[i]=float4(re,im);
}
// The first DIT stage has previous=1. Read its operands through the exact
// digit-reversal table, avoiding a separate full-grid permutation pass.
// Twiddles and the order of the radix accumulation match fft_butterfly.
kernel void fft_permute_first_stage(const device float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],const device float4* twiddles [[buffer(2)]],
    constant FFTShape& p [[buffer(3)]],const device uint* permutation [[buffer(4)]],
    uint i [[thread_position_in_grid]]){
    if(i>=p.count*p.columns)return;
    uint coordinate=(i/p.stride)%p.length;
    uint position=coordinate%p.radix,block=coordinate-position;
    uint base=i-coordinate*p.stride;
    float2 re=float2(0),im=float2(0);
    for(uint j=0;j<p.radix;++j){
        float4 w=twiddles[(position*j)%p.radix];if(p.backward)w.zw=-w.zw;
        float4 value=complex_mul2(input[base+permutation[block+j]*p.stride],w);
        re=add2(re,value.xy);im=add2(im,value.zw);
    }
    output[i]=float4(re,im);
}
kernel void fft_multiply_potential(device float4* field [[buffer(0)]],
    const device float2* potential [[buffer(1)]],constant FFTShape& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]){
    if(i>=p.count*p.columns)return;
    float4 value=field[i];float2 v=potential[i%p.count];
    field[i]=float4(mul2(value.xy,v),mul2(value.zw,v));
}
kernel void fft_gather(const device float4* field [[buffer(0)]],
    const device uint* indices [[buffer(1)]],device float4* output [[buffer(2)]],
    constant FFTShape& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.npw*p.columns)return;
    output[i]=field[(i/p.npw)*p.count+indices[i%p.npw]];
}
)metal";
