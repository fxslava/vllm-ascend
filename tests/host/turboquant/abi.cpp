// SPDX-License-Identifier: Apache-2.0
#include <cstddef>
#include <type_traits>
#include "csrc/attention/turboquant/op_adapter/turboquant_torch_adpt.h"
#include "csrc/attention/turboquant/op_host/turboquant_tiling.h"
#include "csrc/attention/turboquant/e8/e8_decode.hpp"
#include "csrc/attention/turboquant/e8/e8_packet_layout.hpp"
#include "csrc/attention/turboquant/e8/e8_component.hpp"

namespace fd = ascend_micro::fused_decode;
static_assert(sizeof(fd::TurboQuantDecodeTilingData) == 64);
static_assert(alignof(fd::TurboQuantDecodeTilingData) == 64);
static_assert(sizeof(fd::TurboQuantDecodeDispatch) == 32);
#define WORD(FIELD, WORD) static_assert(offsetof(fd::TurboQuantDecodeTilingData, FIELD) == 4 * fd::WORD)
WORD(num_tiles, kTilingWordNumTiles);
WORD(regular_tiles, kTilingWordRegularTiles);
WORD(tail_rows, kTilingWordTailRows);
WORD(block_start_idx, kTilingWordBlockStart);
WORD(blocks_in_split, kTilingWordBlocksInSplit);
WORD(head_dim, kTilingWordHeadDim);
WORD(group_heads, kTilingWordGroupHeads);
WORD(split_k, kTilingWordSplitK);
WORD(tile_origin, kTilingWordTileOrigin);
WORD(tiles_per_page_shift, kTilingWordTilesPerPageShift);
WORD(flags, kTilingWordFlags);
WORD(chunks_per_group, kTilingWordChunksPerGroup);
#undef WORD
static_assert(sizeof(e8_decode::Launch) == 12);
static_assert(offsetof(e8_decode::Launch, attention_scale) == 8);
static_assert(sizeof(e8_decode::Buffers) == 8 * sizeof(void*));
static_assert(e8_packet_abi::kPacketBytes == 4352);
static_assert(e8_packet_abi::kScaleOffset == 4096);
static_assert(e8_decode::kKeyGroupStrideBytes == 1056);
using Reader = void (*)(void*, const e8_decode::Launch*, const e8_decode::Buffers*);
static_assert(std::is_same_v<decltype(&launch_e8_fused_decode_fp32), Reader>);
using Component = void (*)(void*, std::uint16_t*, float*, std::uint16_t*, std::int8_t*);
static_assert(std::is_same_v<decltype(&launch_e8_parallel_component), Component>);
int main() {}
