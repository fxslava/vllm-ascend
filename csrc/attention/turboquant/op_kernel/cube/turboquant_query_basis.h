#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_QUERY_BASIS_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_QUERY_BASIS_H

#include "../common/turboquant_common.h"
#include "../vector/turboquant_vector_service.h"
#include "routine_vector_rotate_butterfly.hpp"

namespace vllm_ascend {
namespace turboquant {

template <typename scalar_t, bool ENABLED>
class TurboQuantQueryBasis {
public:
    static constexpr bool kEnabled = true;
    using OperandT                 = scalar_t;

    __aicore__ inline void Init(AscendC::TPipe* pipe, __gm__ void* query, __gm__ void* piSigns,
                                __gm__ void* rotTables, __gm__ void* h16, __gm__ void* gate, __gm__ void* queryRot,
                                const uint32_t numHeads, const uint32_t headSize, const uint32_t cubeChunkVectors,
                                const uint32_t outputStage, const float invSqrtLen)
    {
        static_cast<void>(rotTables);
        static_cast<void>(h16);
        static_cast<void>(cubeChunkVectors);

        num_heads_    = numHeads;
        head_size_    = headSize;
        output_stage_ = outputStage;
        inv_sqrt_len_ = invSqrtLen;

        const uint32_t scratch_elems = kScratchRows * head_size_;

        query_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(query));
        pi_signs_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(piSigns), head_size_);
        gate_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(gate));
        query_rot_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryRot));

        pipe->InitBuffer(in_buf_, scratch_elems * sizeof(scalar_t));
        pipe->InitBuffer(float_buf_, scratch_elems * sizeof(float));
        pipe->InitBuffer(sign_buf_, head_size_ * sizeof(float));

        AscendC::DataCopy(sign_buf_.Get<float>(), pi_signs_gm_, head_size_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Rotate(const uint32_t numVectors, const uint32_t vectorsPerBlock)
    {
        if ASCEND_IS_AIV {
            const uint32_t start = MixBlockIdx() * vectorsPerBlock;
            uint32_t       end   = start + vectorsPerBlock;

            if (end > numVectors) {
                end = numVectors;
            }

            const uint32_t share = start < end ? end - start : 0;
            if (share > 0) {
                const uint32_t first_half     = CeilDiv(share, kVectorSubcoresPerBlock);
                const bool     second_subcore = AscendC::GetSubBlockIdx() != 0;
                const uint32_t from           = second_subcore ? start + first_half : start;
                const uint32_t to             = second_subcore ? end : start + first_half;

                const AscendC::LocalTensor<scalar_t> query_resident = in_buf_.Get<scalar_t>();
                const AscendC::LocalTensor<float>    query_f32      = float_buf_.Get<float>();
                const AscendC::LocalTensor<float>    signs          = sign_buf_.Get<float>();

                for (uint32_t v = from; v < to; v += kScratchRows) {
                    const uint32_t rows  = (v + kScratchRows <= to) ? kScratchRows : (to - v);
                    const uint32_t elems = rows * head_size_;
                    const uint64_t elem  = static_cast<uint64_t>(v) * head_size_;

                    AscendC::DataCopy(query_resident, query_gm_[elem], elems);
                    SyncMte2ToVector();

                    AscendC::Cast(query_f32, query_resident, AscendC::RoundMode::CAST_NONE, elems);

                    __ubuf__ float* query_base = reinterpret_cast<__ubuf__ float*>(query_f32.GetPhyAddr());
                    __ubuf__ float* signs_base = reinterpret_cast<__ubuf__ float*>(signs.GetPhyAddr());

                    ascend_micro::rotate_butterfly_shuffle_vf(query_base, query_base, signs_base, inv_sqrt_len_, static_cast<uint16_t>(rows));
                    SyncVectorToMte3();

                    AscendC::DataCopy(query_rot_gm_[elem], query_f32, elems);
                    SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
                    SyncMte3ToVector();
                }
            }
        }
        // SyncAll<false> is the MIX full-core barrier (AIC<->AIV per core, then across cores): every
        // AIC and every AIV of every block must reach it, including AICs and empty-share blocks.
        AscendC::SyncAll<false>();
    }

    __aicore__ inline void FinishHeads(const AscendC::LocalTensor<float>& acc, const uint32_t token,
                                       const uint32_t firstHead, const uint32_t heads)
    {
        if (output_stage_ == static_cast<uint32_t>(TurboQuantOutputStage::kRotatedBasis)) {
            return;
        }

        const AscendC::LocalTensor<float> signs = sign_buf_.Get<float>();

        __ubuf__ float* acc_base = reinterpret_cast<__ubuf__ float*>(acc.GetPhyAddr());
        __ubuf__ float* signs_base = reinterpret_cast<__ubuf__ float*>(signs.GetPhyAddr());

        ascend_micro::rotate_butterfly_gather_vf<ascend_micro::RotateDirection::Egress>(acc_base, acc_base, signs_base, inv_sqrt_len_, static_cast<uint16_t>(heads));

        if (output_stage_ != static_cast<uint32_t>(TurboQuantOutputStage::kGated)) {
            return;
        }

        const uint32_t elems = heads * head_size_;
        const AscendC::LocalTensor<scalar_t> gate_resident = in_buf_.Get<scalar_t>();
        const AscendC::LocalTensor<float>    gate_denom    = float_buf_.Get<float>();

        AscendC::DataCopy(gate_resident, gate_gm_[(static_cast<uint64_t>(token) * num_heads_ + firstHead) * head_size_], elems);
        SyncMte2ToVector();
        
        AscendC::Cast(gate_denom, gate_resident, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::Maxs(gate_denom, gate_denom, kGateLogitFloor, elems);
        AscendC::Muls(gate_denom, gate_denom, -1.0f, elems);
        AscendC::Exp(gate_denom, gate_denom, elems);
        AscendC::Adds(gate_denom, gate_denom, 1.0f, elems);
        AscendC::Div(acc, acc, gate_denom, elems);
    }

private:
    static constexpr uint32_t kScratchRows = kCubeTileM / kVectorSubcoresPerBlock;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> in_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> float_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sign_buf_;

    AscendC::GlobalTensor<scalar_t> query_gm_;
    AscendC::GlobalTensor<float>    pi_signs_gm_;
    AscendC::GlobalTensor<scalar_t> gate_gm_;
    AscendC::GlobalTensor<float>    query_rot_gm_;

    uint32_t num_heads_    = 0;
    uint32_t head_size_    = 0;
    uint32_t output_stage_ = 0;
    float    inv_sqrt_len_ = 1.0f;
};

template <typename scalar_t>
class TurboQuantQueryBasis<scalar_t, false> : public TurboQuantRotatedOutput {
public:
    static constexpr bool kEnabled = false;
};

}
}

#endif