// Native double-single PBE and forward derivatives. FP32 transcendental
// functions only seed Newton refinement; they do not determine the result.
static const char* metalPbeSource=R"metal(
inline float2 div2(float2 a,float2 b){
  float q=a.x/b.x;
  float2 result=float2(q,0);
  for(uint i=0;i<2;++i){float2 r=add2(a,-mul2(result,b));result=add2(result,float2(r.x/b.x,0));}
  return result;
}
inline float2 sqrt2(float2 a){
  float2 x=float2(sqrt(a.x),0);
  for(uint i=0;i<2;++i)x=add2(x,div2(add2(a,-mul2(x,x)),mul2(float2(2,0),x)));
  return x;
}
inline float2 cbrt2(float2 a){
  float2 x=float2(pow(a.x,1.f/3.f),0);
  for(uint i=0;i<2;++i)x=add2(x,div2(add2(a,-mul2(mul2(x,x),x)),mul2(float2(3,0),mul2(x,x))));
  return x;
}
inline float2 exp2_ds(float2 a,float2 ln2){
  int exponent=int(round(a.x/ln2.x));
  float2 reduced=add2(a,-mul2(float2(float(exponent),0),ln2));
  float2 sum=float2(1,0),term=sum;
  for(uint i=1;i<=17;++i){term=div2(mul2(term,reduced),float2(float(i),0));sum=add2(sum,term);}
  return float2(ldexp(sum.x,exponent),ldexp(sum.y,exponent));
}
inline float2 log2_ds(float2 a,float2 ln2){
  float2 seed=float2(log(a.x),0);
  float2 delta=add2(div2(a,exp2_ds(seed,ln2)),float2(-1,0));
  return add2(seed,add2(delta,-mul2(float2(.5,0),mul2(delta,delta))));
}
inline float2 log1p_ds(float2 a,float2 ln2){
  if(abs(a.x)>.5f)return log2_ds(add2(float2(1,0),a),ln2);
  // log(1+a)=2 atanh(a/(2+a)). Avoid adding a tiny a to 1 before
  // evaluation: that discards its relative precision in vacuum.
  float2 z=div2(a,add2(float2(2,0),a)),z2=mul2(z,z),term=z,sum=z;
  for(uint j=3;j<=35;j+=2){term=mul2(term,z2);sum=add2(sum,div2(term,float2(float(j),0)));}
  return mul2(float2(2,0),sum);
}
inline float2 expm1_ds(float2 a,float2 ln2){
  if(abs(a.x)>.5f)return add2(exp2_ds(a,ln2),float2(-1,0));
  float2 term=a,sum=a;
  for(uint j=2;j<=17;++j){term=div2(mul2(term,a),float2(float(j),0));sum=add2(sum,term);}
  return sum;
}
// The sigma response is analytic below. Carry only the density derivative
// through AD, saving registers and eliminating unused vacuum intermediates.
struct PbeDual {float2 v,n;};
inline PbeDual dc(float2 v){return {v,float2(0)};}
inline PbeDual da(PbeDual a,PbeDual b){return {add2(a.v,b.v),add2(a.n,b.n)};}
inline PbeDual dn(PbeDual a){return {-a.v,-a.n};}
inline PbeDual ds(PbeDual a,PbeDual b){return da(a,dn(b));}
inline PbeDual dm(PbeDual a,PbeDual b){return {mul2(a.v,b.v),add2(mul2(a.n,b.v),mul2(a.v,b.n))};}
inline PbeDual dd(PbeDual a,PbeDual b){
  // Form (a' - (a/b)*b')/b. Squaring 1/b can overflow FP32's
  // exponent range in vacuum even when the final derivative is finite.
  float2 value=div2(a.v,b.v);
  return {value,div2(add2(a.n,-mul2(value,b.n)),b.v)};
}
inline PbeDual dscale(PbeDual a,float2 b){return {mul2(a.v,b),mul2(a.n,b)};}
inline PbeDual droot(PbeDual a,bool cube,float2 third){
  float2 v=cube?cbrt2(a.v):sqrt2(a.v),factor=div2(mul2(cube?third:float2(.5,0),v),a.v);
  return {v,mul2(factor,a.n)};
}
inline PbeDual dlog1p(PbeDual a,float2 ln2){return {log1p_ds(a.v,ln2),div2(a.n,add2(float2(1,0),a.v))};}
inline PbeDual dexpm1(PbeDual a,float2 ln2){float2 v=expm1_ds(a.v,ln2);return {v,mul2(add2(float2(1,0),v),a.n)};}
kernel void pbe_points(const device ulong* input [[buffer(0)]],device float2* output [[buffer(1)]],
    constant float2* c [[buffer(2)]],constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]){
  if(i>=count)return;
  float2 density=split_ieee64(input[i]);
  // Differentiate with respect to log(n), then divide by n at output.
  // Relative derivatives avoid the 1/n exponent amplification in vacuum.
  PbeDual one=dc(float2(1,0)),n={density,density};
  float2 gradient=split_ieee64(input[count+i]);
  PbeDual g={mul2(gradient,gradient),float2(0)};
  PbeDual root=droot(n,true,c[0]),rs=dd(dc(c[1]),root),kf=dscale(root,c[2]);
  PbeDual s2=dd(g,dscale(dm(dm(kf,kf),dm(n,n)),float2(4,0)));
  PbeDual fx=ds(da(one,dc(c[4])),dd(dc(c[4]),da(one,dscale(s2,div2(c[3],c[4])))));
  PbeDual exchange=dm(dm(n,dscale(root,c[5])),fx);
  PbeDual rroot=droot(rs,false,c[0]);
  PbeDual poly=da(da(dscale(rroot,c[8]),dscale(rs,c[9])),da(dscale(dm(rs,rroot),c[10]),dscale(dm(rs,rs),c[11])));
  PbeDual eps=dscale(dm(da(one,dscale(rs,c[7])),dlog1p(dd(one,dscale(poly,mul2(float2(2,0),c[6]))),c[15])),mul2(float2(-2,0),c[6]));
  PbeDual ks2=dscale(kf,c[12]),t2=dd(g,dscale(dm(ks2,dm(n,n)),float2(4,0)));
  PbeDual A=dd(dc(div2(c[13],c[14])),dexpm1(dscale(eps,div2(float2(-1,0),c[14])),c[15]));
  PbeDual at=dm(A,t2);
  PbeDual ratio;
  if(at.v.x>1.f){PbeDual y=dd(one,at);
    ratio=dm(dd(dc(div2(c[13],c[14])),A),dd(da(one,y),da(da(one,y),dm(y,y))));
  }else ratio=dd(dm(dscale(t2,div2(c[13],c[14])),da(one,at)),da(da(one,at),dm(at,at)));
  PbeDual h=dscale(dlog1p(ratio,c[15]),c[14]);
  PbeDual correlation=dm(n,da(eps,h)),total=da(exchange,correlation);
  float2 u=mul2(div2(c[3],c[4]),s2.v),x=at.v;
  float2 plus=add2(float2(1,0),u);
  // For large reduced gradients both losses approach 1. Switch to the
  // small enhancement factors themselves, avoiding that second cancellation.
  float2 difference;
  if(u.x>.25f||x.x>.25f){
    float2 cf;
    if(x.x>1.f){float2 y=div2(float2(1,0),x),yd=add2(add2(float2(1,0),y),mul2(y,y));
      cf=div2(mul2(add2(float2(2,0),y),mul2(mul2(y,y),y)),mul2(yd,add2(yd,mul2(div2(div2(c[13],c[14]),A.v),add2(float2(1,0),y)))));
    }else{float2 den=add2(add2(float2(1,0),x),mul2(x,x)),bt=mul2(div2(c[13],c[14]),t2.v);
      cf=div2(add2(float2(1,0),mul2(float2(2,0),x)),mul2(den,add2(den,mul2(bt,add2(float2(1,0),x)))));}
    float2 inverse=div2(float2(1,0),plus);difference=add2(cf,-mul2(inverse,inverse));
  }else{
    float2 x2=mul2(x,x),den=add2(add2(float2(1,0),x),x2),bt=mul2(div2(c[13],c[14]),t2.v);
    float2 exchange_loss=div2(mul2(u,add2(float2(2,0),u)),mul2(plus,plus));
    float2 correlation_loss=div2(add2(mul2(x2,add2(float2(3,0),mul2(x,add2(float2(2,0),x)))),mul2(mul2(bt,add2(float2(1,0),x)),den)),mul2(den,add2(den,mul2(bt,add2(float2(1,0),x)))));
    difference=add2(exchange_loss,-correlation_loss);
  }
  float2 response=mul2(div2(c[13],mul2(float2(2,0),mul2(ks2.v,n.v))),difference);
  output[i*6]=total.v;output[i*6+1]=exchange.v;output[i*6+2]=correlation.v;
  output[i*6+3]=div2(exchange.n,n.v);output[i*6+4]=div2(correlation.n,n.v);output[i*6+5]=response;
}
)metal";
