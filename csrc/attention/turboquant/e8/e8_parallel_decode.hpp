#pragma once
#include <cstdint>
#ifdef E8_PARALLEL_CPU
#define E8P_FN inline
#else
#define E8P_FN __simt_callee__ inline __attribute__((always_inline))
#endif

namespace e8_parallel {
// Eight independent coordinate registers. There is no indexed device array.
struct I8 { int a,b,c,d,e,f,g,h; };
E8P_FN unsigned Xor8(unsigned a,unsigned b,unsigned c,unsigned d,
                    unsigned e,unsigned f,unsigned g,unsigned h) {
    const unsigned ab=a^b,cd=c^d,ef=e^f,gh=g^h;
    const unsigned abcd=ab^cd,efgh=ef^gh;
    return abcd^efgh;
}
E8P_FN int Sum8(int a,int b,int c,int d,int e,int f,int g,int h) {
    const int ab=a+b,cd=c+d,ef=e+f,gh=g+h;
    const int abcd=ab+cd,efgh=ef+gh;
    return abcd+efgh;
}
E8P_FN unsigned Max2(unsigned a,unsigned b) { return a>b?a:b; }
E8P_FN unsigned Max8(unsigned a,unsigned b,unsigned c,unsigned d,
                    unsigned e,unsigned f,unsigned g,unsigned h) {
    const unsigned ab=Max2(a,b),cd=Max2(c,d),ef=Max2(e,f),gh=Max2(g,h);
    const unsigned abcd=Max2(ab,cd),efgh=Max2(ef,gh);
    return Max2(abcd,efgh);
}
E8P_FN unsigned RepairKey(int value,unsigned index) {
    const unsigned positive=unsigned(value>=0);
    const unsigned absolute=unsigned(value<0?-value:value);
    // Frozen scan tie: first positive max; otherwise last negative max.
    return (absolute<<8)|(positive<<7)|(positive?7u-index:index);
}
E8P_FN I8 Repair(I8 r,unsigned parity) {
    const unsigned key=Max8(RepairKey(r.a,0),RepairKey(r.b,1),
        RepairKey(r.c,2),RepairKey(r.d,3),RepairKey(r.e,4),RepairKey(r.f,5),
        RepairKey(r.g,6),RepairKey(r.h,7));
    const unsigned positive=(key>>7)&1u;
    const unsigned index=positive?7u-(key&7u):key&7u;
    const int delta=positive?-8:8;
    return {r.a+int(parity&unsigned(index==0))*delta,
            r.b+int(parity&unsigned(index==1))*delta,
            r.c+int(parity&unsigned(index==2))*delta,
            r.d+int(parity&unsigned(index==3))*delta,
            r.e+int(parity&unsigned(index==4))*delta,
            r.f+int(parity&unsigned(index==5))*delta,
            r.g+int(parity&unsigned(index==6))*delta,
            r.h+int(parity&unsigned(index==7))*delta};
}
E8P_FN I8 Project(unsigned label) {
    const int a=2*int((label>>14)&1u)+int((label>>15)&1u)-2;
    const int b=int(label&3u)-2,c=int((label>>2)&3u)-2;
    const int d=int((label>>4)&3u)-2,e=int((label>>6)&3u)-2;
    const int f=int((label>>8)&3u)-2,g=int((label>>10)&3u)-2;
    const int h=int((label>>12)&3u)-2;
    return {a,a+2*b,a-2*b+2*c,a-2*c+2*d,a-2*d+2*e,
            a-2*e+2*f,a-2*f+2*g+2*h,a-2*g+2*h};
}
E8P_FN I8 ShapeResidualBits(I8 p) {
    const unsigned a=unsigned(p.a)+4u,b=unsigned(p.b)+4u;
    const unsigned c=unsigned(p.c)+4u,d=unsigned(p.d)+4u;
    const unsigned e=unsigned(p.e)+4u,f=unsigned(p.f)+4u;
    const unsigned g=unsigned(p.g)+4u,h=unsigned(p.h)+4u;
    const I8 r{int(a&7u)-4,int(b&7u)-4,int(c&7u)-4,int(d&7u)-4,
               int(e&7u)-4,int(f&7u)-4,int(g&7u)-4,int(h&7u)-4};
    const unsigned parity=Xor8((a>>3)&1u,(b>>3)&1u,(c>>3)&1u,(d>>3)&1u,
                               (e>>3)&1u,(f>>3)&1u,(g>>3)&1u,(h>>3)&1u);
    const unsigned negative=Xor8(unsigned(r.a<0),unsigned(r.b<0),
        unsigned(r.c<0),unsigned(r.d<0),unsigned(r.e<0),unsigned(r.f<0),
        unsigned(r.g<0),unsigned(r.h<0));
    const I8 q{r.a<0?r.a+4:r.a-4,r.b<0?r.b+4:r.b-4,
        r.c<0?r.c+4:r.c-4,r.d<0?r.d+4:r.d-4,r.e<0?r.e+4:r.e-4,
        r.f<0?r.f+4:r.f-4,r.g<0?r.g+4:r.g-4,r.h<0?r.h+4:r.h-4};
    // Keep the two uncorrected candidates until selection. A parity repair
    // changes exactly one coordinate by +/-8 toward zero: norm delta is
    // 64-16*abs(coordinate). Integer arithmetic is exact in this domain.
    const unsigned ikey=Max8(RepairKey(r.a,0),RepairKey(r.b,1),
        RepairKey(r.c,2),RepairKey(r.d,3),RepairKey(r.e,4),RepairKey(r.f,5),
        RepairKey(r.g,6),RepairKey(r.h,7));
    const unsigned hkey=Max8(RepairKey(q.a,0),RepairKey(q.b,1),
        RepairKey(q.c,2),RepairKey(q.d,3),RepairKey(q.e,4),RepairKey(q.f,5),
        RepairKey(q.g,6),RepairKey(q.h,7));
    const unsigned hp=parity^negative;
    const int id=Sum8(r.a*r.a,r.b*r.b,r.c*r.c,r.d*r.d,
                     r.e*r.e,r.f*r.f,r.g*r.g,r.h*r.h)
        +int(parity)*(64-16*int(ikey>>8));
    const int hd=Sum8(q.a*q.a,q.b*q.b,q.c*q.c,q.d*q.d,
                     q.e*q.e,q.f*q.f,q.g*q.g,q.h*q.h)
        +int(hp)*(64-16*int(hkey>>8));
    const unsigned ipos=(ikey>>7)&1u,hpos=(hkey>>7)&1u;
    const unsigned ii=ipos?7u-(ikey&7u):ikey&7u;
    const unsigned hi=hpos?7u-(hkey&7u):hkey&7u;
    const int ia=r.a+int(parity&unsigned(ii==0))*(ipos?-8:8);
    const int ha=q.a+int(hp&unsigned(hi==0))*(hpos?-8:8);
    const bool choose=(hd<id)|((hd==id)&(ha<ia));
    const unsigned selectedParity=choose?hp:parity;
    const unsigned selectedIndex=choose?hi:ii;
    const int delta=(choose?hpos:ipos)?-8:8;
    return {(choose?q.a:r.a)+int(selectedParity&unsigned(selectedIndex==0))*delta,
            (choose?q.b:r.b)+int(selectedParity&unsigned(selectedIndex==1))*delta,
            (choose?q.c:r.c)+int(selectedParity&unsigned(selectedIndex==2))*delta,
            (choose?q.d:r.d)+int(selectedParity&unsigned(selectedIndex==3))*delta,
            (choose?q.e:r.e)+int(selectedParity&unsigned(selectedIndex==4))*delta,
            (choose?q.f:r.f)+int(selectedParity&unsigned(selectedIndex==5))*delta,
            (choose?q.g:r.g)+int(selectedParity&unsigned(selectedIndex==6))*delta,
            (choose?q.h:r.h)+int(selectedParity&unsigned(selectedIndex==7))*delta};
}
E8P_FN I8 Decode(unsigned label) { return ShapeResidualBits(Project(label)); }
} // namespace e8_parallel
#undef E8P_FN
