// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
using namespace AscendC;
namespace {
constexpr uint32_t kTile = 4096;
constexpr uint32_t kScaleBlock = 32;
constexpr uint32_t kIndexTailBytes = 512;
constexpr uint32_t kActivationStageBytes = 1024;
constexpr uint32_t kProjectionOffset = 1024;
constexpr uint32_t kMathVectorStride = 128;
constexpr float kGateLimit = 10.0f;
// Mode 0: SIMD. Mode 1: direct cached SIMT. No undocumented bypass claim.
__simt_vf__ __launch_bounds__(1024) inline void unpack_vf(__gm__ const uint8_t* packed, __gm__ const uint8_t* scales,
                                                          __ubuf__ float* dst, uint32_t base, uint32_t count,
                                                          uint32_t bypass, uint32_t lut) {
  for (uint32_t pair = threadIdx.x; pair < count / 2; pair += blockDim.x) {
    const uint32_t byte =
        bypass ? asc_ldcg(const_cast<__gm__ uint8_t*>(packed) + base / 2 + pair) : packed[base / 2 + pair];
    const uint32_t scale = bypass ? asc_ldcg(const_cast<__gm__ uint8_t*>(scales) + (base / 2 + pair) / 16)
                                  : scales[(base / 2 + pair) / 16];
    union Bits {
      uint32_t u;
      float f;
    } s;
    s.u = scale == 0 ? 0x00400000U : (scale == 255 ? 0x7fc00000U : scale << 23);
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t code = (byte >> (half * 4)) & 15;
      const uint32_t mag = code & 7;
      union Bits v;
      v.u = mag == 0 ? 0 : (mag == 1 ? 0x3f000000U : (((mag >> 1) + 126) << 23) | ((mag & 1) << 22));
      v.u |= (code & 8) << 28;
      if (lut != 0) {
        // Register LUT: 2*abs(E2M1) = {0,1,2,3,4,6,8,12}. No UB lookup.
        const uint32_t doubled = (0xc8643210U >> (mag * 4)) & 15;
        v.f = static_cast<float>(doubled) * 0.5f;
        v.u |= (code & 8) << 28;
      }
      dst[2 * pair + half] = v.f * s.f;
    }
  }
}
}  // namespace
extern "C" __global__ __aicore__ void dsv4_unpack_probe(GM_ADDR packed, GM_ADDR scales, GM_ADDR decoded,
                                                        GM_ADDR activated, uint32_t elements, uint32_t columns,
                                                        uint32_t path, uint32_t math) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> pq, sq;
  TBuf<TPosition::VECCALC> dq, aq;
  TBuf<TPosition::VECCALC> scratch;
  // Common math/output storage; SIMD-only unpack scratch is not allocated in SIMT.
  pipe.InitBuffer(dq, kTile * sizeof(float));
  pipe.InitBuffer(aq, kActivationStageBytes);
  if (path == 0) {
    pipe.InitBuffer(pq, kTile / 2);
    pipe.InitBuffer(sq, kTile / kScaleBlock);
    pipe.InitBuffer(scratch, 5 * kTile * sizeof(float) + kIndexTailBytes);
  } else {
    pipe.InitBuffer(scratch, kTile * sizeof(float) + kIndexTailBytes);
  }
  GlobalTensor<uint8_t> p, s;
  GlobalTensor<float> out;
  GlobalTensor<bfloat16_t> act;
  p.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(packed), elements / 2);
  s.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(scales), elements / kScaleBlock);
  p.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  s.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decoded), elements);
  act.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(activated), elements / columns);
  const auto loadEvent = pipe.FetchEventID(HardEvent::MTE2_V);
  const auto storeEvent = pipe.FetchEventID(HardEvent::V_MTE3);
  const auto reuseEvent = pipe.FetchEventID(HardEvent::MTE3_V);
  const auto refillEvent = pipe.FetchEventID(HardEvent::V_MTE2);
  for (uint32_t base = 0; base < elements; base += kTile) {
    const uint32_t n = elements - base < kTile ? elements - base : kTile;
    auto dense = dq.Get<float>();
    auto a = scratch.Get<int32_t>();
    if (path != 0) {
      // Full phase drain, never a sub-tile attempt to overlap SIMT and SIMD.
      PipeBarrier<PIPE_ALL>();
      asc_vf_call<unpack_vf>(dim3{1024}, reinterpret_cast<__gm__ const uint8_t*>(packed),
                             reinterpret_cast<__gm__ const uint8_t*>(scales),
                             reinterpret_cast<__ubuf__ float*>(dense.GetPhyAddr()), base, n, path == 2, path == 3);
      PipeBarrier<PIPE_ALL>();
    } else {
      auto pb = pq.Get<uint8_t>();
      auto sb = sq.Get<uint8_t>();
      // V -> MTE2: input scratch is free before this refill (including first tile).
      SetFlag<HardEvent::V_MTE2>(refillEvent);
      WaitFlag<HardEvent::V_MTE2>(refillEvent);
      DataCopy(pb, p[base / 2], n / 2);
      DataCopy(sb, s[base / kScaleBlock], n / kScaleBlock);
      // MTE2 -> V: both inputs must land before vector reads.
      SetFlag<HardEvent::MTE2_V>(loadEvent);
      WaitFlag<HardEvent::MTE2_V>(loadEvent);
      auto b = scratch.GetWithOffset<int32_t>(kTile, kTile * 4);
      auto c = scratch.GetWithOffset<int32_t>(kTile, kTile * 8);
      auto codes = scratch.GetWithOffset<int32_t>(kTile, kTile * 12);
      auto scale = scratch.GetWithOffset<int32_t>(kTile, kTile * 16);
      auto mask = scratch.GetWithOffset<uint8_t>(kIndexTailBytes, kTile * 20);
      Cast(b.ReinterpretCast<uint16_t>(), pb, RoundMode::CAST_NONE, n / 2);
      Cast(a.ReinterpretCast<uint32_t>(), b.ReinterpretCast<uint16_t>(), RoundMode::CAST_NONE, n / 2);
      Duplicate(b, 15, n / 2);
      And(codes, a, b, n / 2);
      ShiftRight(codes[n / 2], a, 4, n / 2);
      Duplicate(c, 7, n);
      And(b, codes, c, n);
      ShiftRight(a, b, 1, n);
      Adds(a, a, 126, n);
      ShiftLeft(a, a, 23, n);
      Duplicate(c, 1, n);
      And(c, b, c, n);
      ShiftLeft(c, c, 22, n);
      Or(a, a, c, n);
      Duplicate(c, 0x3f000000, n);
      Compares(mask, b, 1, CMPMODE::EQ, n);
      Select(a, mask, c, a, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
      Duplicate(c, 0, n);
      Compares(mask, b, 0, CMPMODE::EQ, n);
      Select(a, mask, c, a, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
      Duplicate(c, 8, n);
      And(c, codes, c, n);
      ShiftLeft(c, c, 28, n);
      Or(codes, a, c, n);
      const uint32_t ns = n / kScaleBlock;
      Cast(b.ReinterpretCast<uint16_t>(), sb, RoundMode::CAST_NONE, ns);
      Cast(scale.ReinterpretCast<uint32_t>(), b.ReinterpretCast<uint16_t>(), RoundMode::CAST_NONE, ns);
      Adds(b, scale, 1, ns);
      Duplicate(c, 255, ns);
      And(b, b, c, ns);
      Compares(mask, b, 2, CMPMODE::LT, ns);
      ShiftLeft(scale, scale, 23, ns);
      Duplicate(c, 0x00400000, ns);
      Select(c, mask, c, 0, SELMODE::VSEL_TENSOR_SCALAR_MODE, ns);
      Add(scale, scale, c, ns);
      // Pair-planar -> natural order. Gather offsets are bytes.
      ArithProgression(a, 0, 1, n);
      Duplicate(b, 1, n);
      And(b, a, b, n);
      Muls(b, b, static_cast<int32_t>(n / 2), n);
      ShiftRight(a, a, 1, n);
      Add(a, a, b, n);
      ShiftLeft(a, a, 2, n);
      Gather(dense, codes.ReinterpretCast<float>(), a.ReinterpretCast<uint32_t>(), 0, n);
      ArithProgression(a, 0, 1, n);
      ShiftRight(a, a, 5, n);
      ShiftLeft(a, a, 2, n);
      Gather(b.ReinterpretCast<float>(), scale.ReinterpretCast<float>(), a.ReinterpretCast<uint32_t>(), 0, n);
      Mul(dense, dense, b.ReinterpretCast<float>(), n);
    }
    if (math != 0) {
      // Controlled projection: x=1, w1=w3. Same SIMD contraction on both paths.
      Muls(dense, dense, 1.0f, n);  // x=1 projection operand, common SIMD math.
      const uint32_t rows = n / columns;
      auto sums = a.ReinterpretCast<float>();
      for (uint32_t r = 0; r < rows; ++r) {
        // 32-byte reduction slots avoid partial-write aliasing.
        ReduceSum(sums[r * 8], dense[r * columns], sums[kTile / 2], columns);
      }
      auto av = aq.Get<bfloat16_t>();
      // Copy reduced slots into compact vectors with Gather.
      auto idx = scratch.GetWithOffset<int32_t>(128, kTile * 4);
      ArithProgression(idx, 0, 32, rows);
      Gather(sums[kProjectionOffset], sums, idx.ReinterpretCast<uint32_t>(), 0, rows);
      Mins(sums[kProjectionOffset + kMathVectorStride], sums[kProjectionOffset], kGateLimit, rows);
      Maxs(sums[kProjectionOffset + kMathVectorStride], sums[kProjectionOffset + kMathVectorStride], -kGateLimit, rows);
      auto activatedTile = sums[kProjectionOffset + 2 * kMathVectorStride];
      auto upTile = sums[kProjectionOffset];
      auto gateTile = sums[kProjectionOffset + kMathVectorStride];
      activatedTile.SetSize(rows);
      upTile.SetSize(rows);
      gateTile.SetSize(rows);
      SwiGLU<float, false>(activatedTile, upTile, gateTile, 1.0f, rows);
      Cast(av, sums[kProjectionOffset + 2 * kMathVectorStride], RoundMode::CAST_ROUND, rows);
      SetFlag<HardEvent::V_MTE3>(storeEvent);
      WaitFlag<HardEvent::V_MTE3>(storeEvent);
      DataCopyPad(act[base / columns], av, DataCopyExtParams{1, rows * 2, 0, 0, 0});
    }
    SetFlag<HardEvent::V_MTE3>(storeEvent);
    WaitFlag<HardEvent::V_MTE3>(storeEvent);
    DataCopy(out[base], dense, n);
    // MTE3 -> V: drain both stores before any UB slot is reused.
    SetFlag<HardEvent::MTE3_V>(reuseEvent);
    WaitFlag<HardEvent::MTE3_V>(reuseEvent);
  }
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t dsv4_unpack_probe_impl(void* stream, void* packed, void* scales, void* decoded, void* activated,
                                           uint32_t elements, uint32_t columns, uint32_t path, uint32_t math) {
  return dsv4_unpack_probe<<<1, nullptr, stream>>>(
      reinterpret_cast<uint8_t*>(packed), reinterpret_cast<uint8_t*>(scales), reinterpret_cast<uint8_t*>(decoded),
      reinterpret_cast<uint8_t*>(activated), elements, columns, path, math);
}
#endif

// A matched minimal-store control. Its differential includes VF instructions,
// dispatch and drain; it is deliberately not labeled isolated switch latency.
__simt_vf__ __launch_bounds__(1024) inline void switch_control_vf(__ubuf__ float* values) {
  volatile __ubuf__ float* target = values;
  target[threadIdx.x] = 1.0f;
}
extern "C" __global__ __aicore__ void dsv4_switch_control(GM_ADDR output, uint32_t iterations, uint32_t simt) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
  constexpr uint32_t kControlElements = 32;
  pipe.InitBuffer(buffer, kControlElements * sizeof(float));
  auto values = buffer.Get<float>();
  Duplicate(values, 1.0f, kControlElements);
  for (uint32_t i = 0; i < iterations; ++i) {
    PipeBarrier<PIPE_ALL>();
    if (simt != 0) asc_vf_call<switch_control_vf>(dim3{32}, reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()));
    PipeBarrier<PIPE_ALL>();
    Muls(values, values, 1.0f, kControlElements);
  }
  GlobalTensor<float> gm;
  gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), kControlElements);
  const auto store = pipe.FetchEventID(HardEvent::V_MTE3);
  SetFlag<HardEvent::V_MTE3>(store);
  WaitFlag<HardEvent::V_MTE3>(store);
  DataCopy(gm, values, kControlElements);
  const auto drain = pipe.FetchEventID(HardEvent::MTE3_V);
  SetFlag<HardEvent::MTE3_V>(drain);
  WaitFlag<HardEvent::MTE3_V>(drain);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t dsv4_switch_control_impl(void* stream, void* output, uint32_t iterations, uint32_t simt) {
  return dsv4_switch_control<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(output), iterations, simt);
}
#endif
