// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#pragma once

#include "turboquant_common.h"
#include "turboquant_strip_layout.h"

namespace vllm_ascend::turboquant {

// The BF16 writer stores consecutive signed INT4 coordinates, with q * scale
// already in the orthonormal basis. There is no centroid bias or codebook gain.
class TurboQuantStripCodec {
 public:
  static constexpr bool kIsAffine = true;
  static constexpr bool kGatherLut = false;
  static constexpr bool kLloydMaxLut = false;
  static constexpr float kGain = 1.0f;
  static constexpr uint32_t kStripRows = turboquant_wave::kReshapeTokensPerStrip;
  static constexpr uint32_t kGatherElems = kStripRows * kOperandC0;

  __aicore__ static constexpr uint32_t PackedBytes(uint32_t headSize) { return headSize / kPackFactor; }

  __aicore__ inline void Init(AscendC::TPipe* pipe, uint32_t headSize, uint32_t, float,
                              const AscendC::GlobalTensor<int32_t>&) {
    headSize_ = headSize;
    pipe->InitBuffer(halfBuf_, kStripRows * headSize * sizeof(half));
    pipe->InitBuffer(floatBuf_, kStripRows * headSize * sizeof(float));
    pipe->InitBuffer(gatherBuf_, kGatherElems * sizeof(float));
    pipe->InitBuffer(indexBuf_, kGatherElems * sizeof(uint32_t));
    if ASCEND_IS_AIV {
      if (AscendC::GetSubBlockIdx() != 0) {
        const auto indices = indexBuf_.Get<uint32_t>();
        // V is [channel, token-in-strip]; the Cube operand is grouped
        // by 32 channels, then token, then channel within the group.
        for (uint32_t token = 0; token < kStripRows; ++token)
          for (uint32_t channel = 0; channel < kOperandC0; ++channel)
            indices.SetValue(token * kOperandC0 + channel, StripValueGatherByte(token, channel));
        SyncScalarToVector();
      }
    }
  }

  template <typename OperandT>
  __aicore__ inline void CastToOperand(const AscendC::LocalTensor<OperandT>& dst,
                                       const AscendC::LocalTensor<float>& src, uint32_t count) {
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
  }

  template <typename OperandT>
  __aicore__ inline void UnpackStrip(const AscendC::LocalTensor<OperandT>& dst,
                                     const AscendC::LocalTensor<int8_t>& packed, bool valuePlane) {
    const uint32_t count = kStripRows * headSize_;
    const auto halfValues = halfBuf_.Get<half>();
    const auto values = floatBuf_.Get<float>();
    AscendC::Cast(halfValues, packed.ReinterpretCast<int4b_t>(), AscendC::RoundMode::CAST_NONE, count);
    AscendC::Cast(values, halfValues, AscendC::RoundMode::CAST_NONE, count);
    if (!valuePlane) {
      CastToOperand(dst, values, count);
    } else {
      const auto gathered = gatherBuf_.Get<float>();
      const auto indices = indexBuf_.Get<uint32_t>();
      for (uint32_t base = 0; base < count; base += kGatherElems) {
        AscendC::Gather(gathered, values[base], indices, kGatherSrcBase, kGatherElems);
        CastToOperand(dst[base], gathered, kGatherElems);
      }
    }
    // The next strip reuses both the cast scratch and the operand staging
    // buffer after its MTE3 transfer has been joined by the caller.
    AscendC::PipeBarrier<PIPE_V>();
  }

 private:
  AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
  uint32_t headSize_ = 0;
};

}  // namespace vllm_ascend::turboquant
