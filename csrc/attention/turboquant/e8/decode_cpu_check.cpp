#define E8_CPU_TEST
#define ascend_micro frozen_micro
#include <routine_e8_bf16_dequant_simt.hpp>
#include <routine_e8_int8_dequant_simt.hpp>
#undef ascend_micro
#include "routines/routine_e8_bf16_dequant_simt.hpp"
#include "routines/routine_e8_int8_dequant_simt.hpp"
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
static std::uint16_t Bf16(float x) {
 std::uint32_t b=0;std::memcpy(&b,&x,4);return std::uint16_t((b+0x7fffu+((b>>16)&1u))>>16);
}
int main()try {
 std::fesetround(FE_TONEAREST);
 const std::array<float,16> scales{1,-1,0.5f,-0.5f,0.1f,127,
  0x0.fffffep-126f,-0x0.fffffep-126f,0x1p-126f,-0x1p-126f,
  0x1p-132f,-0x1p-132f,0.0f,-0.0f,16,0.03125f};
 std::uint64_t pairs=0;
 for(unsigned label=0;label<65536;++label) {
  int old[8],now[8];frozen_micro::e8::Project(label,old);frozen_micro::e8::ShapeResidualBits(old);
  ascend_micro::e8::Project(label,now);ascend_micro::e8::ShapeResidualBits(now);
  if(std::memcmp(old,now,sizeof(old)))throw std::runtime_error("finite label/tie mismatch");
  for(const float scale:scales)for(unsigned c=0;c<8;++c) {
   const float expected=(float(old[c])*0.5f)*scale,actual=(float(now[c])*0.5f)*scale;
   if(Bf16(expected)!=Bf16(actual))throw std::runtime_error("BF16 mismatch");
   if(frozen_micro::e8_int8_simt::detail::PhysicalInt8(expected)!=ascend_micro::e8_int8_simt::detail::PhysicalInt8(actual))throw std::runtime_error("INT8 mismatch");
   ++pairs;
  }
 }
 std::printf("PASS labels=65536 scales=16 coordinate-scale-pairs=%llu exact ties/BF16RNE/INT8RNE; CPU-only, no device claim\n",static_cast<unsigned long long>(pairs));return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
