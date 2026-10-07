/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The authoritative weight byte source for startup ingestion.
//
// Ports `tools/dsv4_moe_runtime/protocols/provider.py` (WeightByteSource) and
// the container parsing of `hardware/safetensors_provider.py`: the safetensors
// format read directly (8-byte little-endian header length, JSON header,
// contiguous payload), no third-party dependency.
//
// The brief's hard rule -- "file descriptors are closed and no disk reads occur
// during execution" -- is enforced two ways rather than asserted in prose:
//
//   1. `Close()` closes every shard descriptor and latches `closed_`. Any later
//      read throws `Dsv4Error`, exactly as the Python partition's `read_param`
//      raises "runtime disk reads are forbidden in the exclusive hierarchy".
//   2. `AssertNoOpenWeightDescriptors()` walks /proc/self/fd and refuses if any
//      descriptor still resolves under the weights root -- which catches a
//      descriptor leaked by something *other* than this class.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "dsv4_expert_layout.hpp"

namespace vllm_ascend {
namespace dsv4 {

// How one checkpoint family names an expert's projections. Mirrors
// `hardware/safetensors_provider.py`'s ExpertNamingScheme.
enum class CheckpointNaming {
  kDsv4Flat,    // layers.{L}.ffn.experts.{E}.{w1|w2|w3}.{weight|scale}
  kHfDeepseek,  // model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.{weight|weight_scale_inv}
};

const char* CheckpointNamingName(CheckpointNaming naming);

// The three checkpoint projections that make up one expert, named by role
// rather than by the layout's fused regions.
enum class ExpertProjection { kGate, kUp, kDown };

class WeightByteSource {
 public:
  virtual ~WeightByteSource() = default;

  virtual const char* source_name() const = 0;

  virtual bool Contains(int32_t layer, int32_t expert) const = 0;

  // Read `count` bytes of one expert's slot image starting at `slot_offset`,
  // i.e. the destination is a window into a `ExpertSlotLayout`-shaped slot. This
  // is the only read primitive the staging engine needs: a host slot takes the
  // whole slot in one call, a device slot takes it in bounded transit chunks.
  virtual void ReadExpertSlotRange(uint8_t* destination, size_t destination_capacity, size_t slot_offset,
                                   size_t count, int32_t layer, int32_t expert) = 0;

  // Whole-slot convenience built on the primitive above.
  size_t FillExpertSlot(uint8_t* destination, size_t destination_capacity, int32_t layer, int32_t expert);

  // Backbone (non-expert) tensors, addressed by checkpoint name.
  virtual bool HasNamed(const std::string& name) const = 0;
  virtual void ReadNamed(const std::string& name, uint8_t* destination, size_t destination_capacity,
                         size_t byte_offset, size_t count) = 0;
  virtual size_t NamedByteSize(const std::string& name) const = 0;

  virtual void Close() = 0;
  bool closed() const { return closed_; }

  // Refuses if any open descriptor of this process still resolves under
  // `root`. Static so the runner can check after *all* sources are closed.
  static void AssertNoOpenWeightDescriptors(const std::string& root);

  uint64_t bytes_read() const { return bytes_read_; }
  uint64_t read_requests() const { return read_requests_; }

 protected:
  void RefuseIfClosed(const char* what) const;

  bool closed_ = false;
  uint64_t bytes_read_ = 0;
  uint64_t read_requests_ = 0;
  ExpertSlotLayout layout_ = ExpertSlotLayout::ForDeepSeekV4Flash();
};

// ---------------------------------------------------------------------------
// Deterministic synthetic source: no files, no descriptors
// ---------------------------------------------------------------------------
//
// Used by `--synthetic-weights` and by the contract test. Every byte is a pure
// function of (layer, expert, slot_offset), so the same expert reads back
// identically after an eviction round trip -- which is what makes the swap
// engine's round-trip check in `dsv4_contract_smoke` meaningful.
class SyntheticWeightSource : public WeightByteSource {
 public:
  SyntheticWeightSource(const ExpertSlotLayout& layout, int64_t num_layers, int64_t num_experts);

  const char* source_name() const override { return "synthetic"; }

  bool Contains(int32_t layer, int32_t expert) const override;
  void ReadExpertSlotRange(uint8_t* destination, size_t destination_capacity, size_t slot_offset, size_t count,
                           int32_t layer, int32_t expert) override;
  bool HasNamed(const std::string& name) const override;
  void ReadNamed(const std::string& name, uint8_t* destination, size_t destination_capacity, size_t byte_offset,
                 size_t count) override;
  size_t NamedByteSize(const std::string& name) const override;
  void Close() override { closed_ = true; }

  // The byte an expert's slot image carries at `slot_offset`.
  static uint8_t ByteAt(int32_t layer, int32_t expert, size_t slot_offset);

 private:
  int64_t num_layers_ = 0;
  int64_t num_experts_ = 0;
};

// ---------------------------------------------------------------------------
// Safetensors source
// ---------------------------------------------------------------------------

struct SafetensorsTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t payload_begin = 0;  // relative to the shard's data start
  uint64_t payload_end = 0;
  int32_t shard = -1;

  uint64_t num_bytes() const { return payload_end - payload_begin; }
};

class SafetensorsWeightSource : public WeightByteSource {
 public:
  // `path` is either a single `.safetensors` file, or a directory holding
  // shards. Every shard header is parsed at construction; nothing is read
  // lazily after `Close()`.
  SafetensorsWeightSource(const std::string& path, const ExpertSlotLayout& layout, CheckpointNaming naming,
                          int64_t num_layers, int64_t num_experts);
  ~SafetensorsWeightSource() override;

  const char* source_name() const override { return "safetensors"; }

  bool Contains(int32_t layer, int32_t expert) const override;
  void ReadExpertSlotRange(uint8_t* destination, size_t destination_capacity, size_t slot_offset, size_t count,
                           int32_t layer, int32_t expert) override;
  bool HasNamed(const std::string& name) const override;
  void ReadNamed(const std::string& name, uint8_t* destination, size_t destination_capacity, size_t byte_offset,
                 size_t count) override;
  size_t NamedByteSize(const std::string& name) const override;
  void Close() override;

  const std::string& root() const { return root_; }
  size_t tensor_count() const { return tensors_.size(); }
  CheckpointNaming naming() const { return naming_; }

  // Name of one expert projection under the active naming scheme.
  std::string ExpertTensorName(int32_t layer, int32_t expert, ExpertProjection projection, bool is_scale) const;

  // Throws unless every expert tensor the layout needs exists and has the byte
  // size the layout's region implies. Run once, at ingestion, before the first
  // slot is filled -- a mismatch found halfway through leaves a partition that
  // looks populated and is not.
  void ValidateExpertBinding(int32_t layer, int32_t expert) const;

 private:
  struct Shard {
    std::string path;
    int fd = -1;
    uint64_t data_start = 0;
    uint64_t file_size = 0;
  };

  void OpenShards(const std::string& path);
  void ParseShardHeader(int32_t shard_index);
  const SafetensorsTensor& Tensor(const std::string& name) const;
  void PreadExact(int32_t shard_index, uint8_t* destination, uint64_t file_offset, size_t count);

  // Maps a [slot_offset, slot_offset+count) window onto the checkpoint tensors
  // backing it and reads each piece. Handles the gate/up fusion: the `w13`
  // region's first half comes from gate_proj and the second from up_proj.
  void ReadRegionWindow(const ExpertRegionSpec& spec, uint8_t* destination, size_t window_begin, size_t window_end,
                        int32_t layer, int32_t expert);

  std::string root_;
  CheckpointNaming naming_ = CheckpointNaming::kDsv4Flat;
  int64_t num_layers_ = 0;
  int64_t num_experts_ = 0;
  std::vector<Shard> shards_;
  std::map<std::string, SafetensorsTensor> tensors_;
};

// Minimal JSON reader for safetensors headers. Exposed so the contract test can
// exercise it directly: the headers are machine-generated and flat (one object
// per tensor plus an optional `__metadata__`), which is the whole grammar this
// handles -- it is not a general JSON parser and says so by throwing on
// anything else.
std::map<std::string, SafetensorsTensor> ParseSafetensorsHeaderJson(const char* text, size_t length);

}  // namespace dsv4
}  // namespace vllm_ascend
