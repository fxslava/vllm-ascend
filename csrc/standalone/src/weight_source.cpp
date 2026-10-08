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

#include "moe/core/weight_source.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <sstream>

#include "moe/core/error.hpp"
#include "moe/core/config.hpp"

namespace ascend_moe {
namespace {

constexpr size_t kSafetensorsHeaderLengthBytes = 8;
// A safetensors header is JSON metadata, not payload. Anything past this is a
// malformed or hostile file rather than a big model.
constexpr uint64_t kMaxHeaderBytes = 512ull * 1024 * 1024;

bool IsDirectory(const std::string& path) {
  struct stat info {};
  return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string JoinPath(const std::string& directory, const std::string& leaf) {
  if (directory.empty() || EndsWith(directory, "/")) {
    return directory + leaf;
  }
  return directory + "/" + leaf;
}

// ---- the minimal JSON scanner -------------------------------------------

struct Scanner {
  const char* data = nullptr;
  size_t length = 0;
  size_t position = 0;

  void SkipSpace() {
    while (position < length &&
           (data[position] == ' ' || data[position] == '\t' || data[position] == '\n' || data[position] == '\r')) {
      ++position;
    }
  }

  char Peek() const {
    DSV4_REQUIRE(position < length, "safetensors header ended early at offset " << position);
    return data[position];
  }

  void Expect(char expected) {
    SkipSpace();
    DSV4_REQUIRE(position < length && data[position] == expected,
                 "safetensors header: expected '" << expected << "' at offset " << position << ", found '"
                                                  << (position < length ? data[position] : '?') << "'");
    ++position;
  }

  bool TryConsume(char expected) {
    SkipSpace();
    if (position < length && data[position] == expected) {
      ++position;
      return true;
    }
    return false;
  }

  std::string ReadString() {
    Expect('"');
    std::string out;
    while (position < length && data[position] != '"') {
      if (data[position] == '\\') {
        ++position;
        DSV4_REQUIRE(position < length, "safetensors header: trailing escape");
        // Tensor names in a safetensors header are plain identifiers with dots;
        // the only escapes that legally appear are these.
        switch (data[position]) {
          case '"':
            out.push_back('"');
            break;
          case '\\':
            out.push_back('\\');
            break;
          case '/':
            out.push_back('/');
            break;
          case 'n':
            out.push_back('\n');
            break;
          default:
            throw Dsv4Error("safetensors header: unsupported escape sequence in a tensor name");
        }
        ++position;
        continue;
      }
      out.push_back(data[position++]);
    }
    Expect('"');
    return out;
  }

  int64_t ReadInteger() {
    SkipSpace();
    const size_t begin = position;
    if (position < length && (data[position] == '-' || data[position] == '+')) {
      ++position;
    }
    while (position < length && data[position] >= '0' && data[position] <= '9') {
      ++position;
    }
    DSV4_REQUIRE(position > begin, "safetensors header: expected an integer at offset " << begin);
    return std::stoll(std::string(data + begin, position - begin));
  }

  // Skips any value; used for `__metadata__` and for fields this reader does
  // not need. Nesting is tracked so an object value cannot desynchronise the
  // outer scan.
  void SkipValue() {
    SkipSpace();
    const char first = Peek();
    if (first == '"') {
      ReadString();
      return;
    }
    if (first == '{' || first == '[') {
      const char closing = (first == '{') ? '}' : ']';
      int depth = 0;
      bool in_string = false;
      while (position < length) {
        const char current = data[position];
        if (in_string) {
          if (current == '\\') {
            ++position;
          } else if (current == '"') {
            in_string = false;
          }
        } else if (current == '"') {
          in_string = true;
        } else if (current == '{' || current == '[') {
          ++depth;
        } else if (current == '}' || current == ']') {
          --depth;
          if (depth == 0) {
            DSV4_REQUIRE(current == closing, "safetensors header: mismatched bracket at offset " << position);
            ++position;
            return;
          }
        }
        ++position;
      }
      throw Dsv4Error("safetensors header: unterminated object or array");
    }
    // A bare literal: number, true, false, null.
    while (position < length && data[position] != ',' && data[position] != '}' && data[position] != ']') {
      ++position;
    }
  }
};

}  // namespace

const char* CheckpointNamingName(CheckpointNaming naming) {
  return naming == CheckpointNaming::kDsv4Flat ? "dsv4-flat" : "hf-deepseek";
}

// ---------------------------------------------------------------------------
// WeightByteSource
// ---------------------------------------------------------------------------

void WeightByteSource::RefuseIfClosed(const char* what) const {
  DSV4_REQUIRE(!closed_, "runtime disk reads are forbidden in the exclusive hierarchy: "
                             << what << " after the weight source was closed");
}

size_t WeightByteSource::FillExpertSlot(uint8_t* destination, size_t destination_capacity, int32_t layer,
                                        int32_t expert) {
  const size_t slot_bytes = layout_.slot_num_bytes();
  DSV4_REQUIRE(destination_capacity >= slot_bytes,
               "expert slot fill needs " << slot_bytes << " bytes, destination holds " << destination_capacity);
  ReadExpertSlotRange(destination, destination_capacity, 0, slot_bytes, layer, expert);
  return slot_bytes;
}

void WeightByteSource::AssertNoOpenWeightDescriptors(const std::string& root) {
  if (root.empty()) {
    return;
  }
  DIR* directory = ::opendir("/proc/self/fd");
  if (directory == nullptr) {
    // No procfs: the latch in RefuseIfClosed is then the only guard, and the
    // caller is told rather than being given a false pass.
    throw Dsv4Error("cannot verify descriptor closure: /proc/self/fd is unavailable on this host");
  }
  std::vector<std::string> offenders;
  while (const dirent* entry = ::readdir(directory)) {
    if (entry->d_name[0] == '.') {
      continue;
    }
    const std::string link = std::string("/proc/self/fd/") + entry->d_name;
    char target[4096];
    const ssize_t written = ::readlink(link.c_str(), target, sizeof(target) - 1);
    if (written <= 0) {
      continue;
    }
    target[written] = '\0';
    if (std::string(target).compare(0, root.size(), root) == 0) {
      offenders.emplace_back(std::string(entry->d_name) + " -> " + target);
    }
  }
  ::closedir(directory);
  if (!offenders.empty()) {
    std::ostringstream message;
    message << "weight descriptors are still open after ingestion (" << offenders.size() << "):";
    for (const std::string& offender : offenders) {
      message << "\n  fd " << offender;
    }
    throw Dsv4Error(message.str());
  }
}

// ---------------------------------------------------------------------------
// SyntheticWeightSource
// ---------------------------------------------------------------------------

SyntheticWeightSource::SyntheticWeightSource(const ExpertSlotLayout& layout, int64_t num_layers,
                                             int64_t num_experts)
    : num_layers_(num_layers), num_experts_(num_experts) {
  layout_ = layout;
}

bool SyntheticWeightSource::Contains(int32_t layer, int32_t expert) const {
  return layer >= 0 && layer < num_layers_ && expert >= 0 && expert < num_experts_;
}

uint8_t SyntheticWeightSource::ByteAt(int32_t layer, int32_t expert, size_t slot_offset) {
  // splitmix64 over (key, offset): cheap, deterministic, and dependent on all
  // three inputs, so a slot that lands in the wrong place is detectable.
  uint64_t state = (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 40) ^
                   (static_cast<uint64_t>(static_cast<uint32_t>(expert)) << 24) ^
                   static_cast<uint64_t>(slot_offset);
  state += 0x9e3779b97f4a7c15ull;
  uint64_t mixed = state;
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebull;
  mixed = mixed ^ (mixed >> 31);
  return static_cast<uint8_t>(mixed & 0xffu);
}

void SyntheticWeightSource::ReadExpertSlotRange(uint8_t* destination, size_t destination_capacity,
                                                size_t slot_offset, size_t count, int32_t layer, int32_t expert) {
  RefuseIfClosed("ReadExpertSlotRange");
  DSV4_REQUIRE(Contains(layer, expert),
               "expert (layer=" << layer << ", id=" << expert << ") is outside the synthetic source coverage");
  DSV4_REQUIRE(count <= destination_capacity,
               "synthetic read of " << count << " bytes exceeds destination capacity " << destination_capacity);
  DSV4_REQUIRE(slot_offset + count <= layout_.slot_num_bytes(),
               "synthetic read [" << slot_offset << ", " << (slot_offset + count) << ") leaves the "
                                  << layout_.slot_num_bytes() << "-byte slot");
  for (size_t index = 0; index < count; ++index) {
    destination[index] = ByteAt(layer, expert, slot_offset + index);
  }
  bytes_read_ += count;
  ++read_requests_;
}

bool SyntheticWeightSource::HasNamed(const std::string&) const { return true; }

void SyntheticWeightSource::ReadNamed(const std::string& name, uint8_t* destination, size_t destination_capacity,
                                      size_t byte_offset, size_t count) {
  RefuseIfClosed("ReadNamed");
  DSV4_REQUIRE(count <= destination_capacity, "synthetic named read exceeds destination capacity");
  // Hash the name into the layer field so distinct backbone tensors differ.
  int32_t pseudo_layer = 0;
  for (char character : name) {
    pseudo_layer = pseudo_layer * 31 + static_cast<int32_t>(character);
  }
  for (size_t index = 0; index < count; ++index) {
    destination[index] = ByteAt(pseudo_layer, -1, byte_offset + index);
  }
  bytes_read_ += count;
  ++read_requests_;
}

size_t SyntheticWeightSource::NamedByteSize(const std::string&) const {
  // The synthetic source is unbounded by construction; the caller's reservation
  // decides how much it needs. 0 means "do not check".
  return 0;
}

// ---------------------------------------------------------------------------
// SafetensorsWeightSource
// ---------------------------------------------------------------------------

std::map<std::string, SafetensorsTensor> ParseSafetensorsHeaderJson(const char* text, size_t length) {
  Scanner scanner{text, length, 0};
  std::map<std::string, SafetensorsTensor> tensors;
  scanner.Expect('{');
  if (scanner.TryConsume('}')) {
    return tensors;
  }
  while (true) {
    const std::string name = scanner.ReadString();
    scanner.Expect(':');
    if (name == "__metadata__") {
      scanner.SkipValue();
    } else {
      SafetensorsTensor tensor;
      scanner.Expect('{');
      while (true) {
        const std::string field = scanner.ReadString();
        scanner.Expect(':');
        if (field == "dtype") {
          tensor.dtype = scanner.ReadString();
        } else if (field == "shape") {
          scanner.Expect('[');
          if (!scanner.TryConsume(']')) {
            while (true) {
              tensor.shape.push_back(scanner.ReadInteger());
              if (!scanner.TryConsume(',')) {
                break;
              }
            }
            scanner.Expect(']');
          }
        } else if (field == "data_offsets") {
          scanner.Expect('[');
          tensor.payload_begin = static_cast<uint64_t>(scanner.ReadInteger());
          scanner.Expect(',');
          tensor.payload_end = static_cast<uint64_t>(scanner.ReadInteger());
          scanner.Expect(']');
        } else {
          scanner.SkipValue();
        }
        if (!scanner.TryConsume(',')) {
          break;
        }
      }
      scanner.Expect('}');
      DSV4_REQUIRE(tensor.payload_end >= tensor.payload_begin,
                   "safetensors tensor " << name << " has data_offsets [" << tensor.payload_begin << ", "
                                         << tensor.payload_end << ")");
      tensors.emplace(name, std::move(tensor));
    }
    if (!scanner.TryConsume(',')) {
      break;
    }
  }
  scanner.Expect('}');
  return tensors;
}

SafetensorsWeightSource::SafetensorsWeightSource(const std::string& path, const ExpertSlotLayout& layout,
                                                 CheckpointNaming naming, int64_t num_layers, int64_t num_experts)
    : naming_(naming), num_layers_(num_layers), num_experts_(num_experts) {
  layout_ = layout;
  OpenShards(path);
  for (int32_t shard = 0; shard < static_cast<int32_t>(shards_.size()); ++shard) {
    ParseShardHeader(shard);
  }
  DSV4_REQUIRE(!tensors_.empty(), "no tensors found under " << path);
}

SafetensorsWeightSource::~SafetensorsWeightSource() {
  for (Shard& shard : shards_) {
    if (shard.fd >= 0) {
      ::close(shard.fd);
      shard.fd = -1;
    }
  }
}

void SafetensorsWeightSource::OpenShards(const std::string& path) {
  std::vector<std::string> files;
  if (IsDirectory(path)) {
    root_ = path;
    DIR* directory = ::opendir(path.c_str());
    DSV4_REQUIRE(directory != nullptr, "cannot open weights directory " << path << ": " << std::strerror(errno));
    while (const dirent* entry = ::readdir(directory)) {
      const std::string leaf = entry->d_name;
      if (EndsWith(leaf, ".safetensors")) {
        files.push_back(JoinPath(path, leaf));
      }
    }
    ::closedir(directory);
    std::sort(files.begin(), files.end());
  } else {
    files.push_back(path);
    const size_t slash = path.find_last_of('/');
    root_ = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
  }
  DSV4_REQUIRE(!files.empty(), "no .safetensors shard found at " << path);

  shards_.reserve(files.size());
  for (const std::string& file : files) {
    Shard shard;
    shard.path = file;
    shard.fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
    DSV4_REQUIRE(shard.fd >= 0, "cannot open " << file << ": " << std::strerror(errno));
    struct stat info {};
    DSV4_REQUIRE(::fstat(shard.fd, &info) == 0, "cannot stat " << file << ": " << std::strerror(errno));
    shard.file_size = static_cast<uint64_t>(info.st_size);
    shards_.push_back(shard);
  }
}

void SafetensorsWeightSource::ParseShardHeader(int32_t shard_index) {
  Shard& shard = shards_[static_cast<size_t>(shard_index)];
  DSV4_REQUIRE(shard.file_size >= kSafetensorsHeaderLengthBytes,
               shard.path << " is too small to be a safetensors container");
  uint64_t header_length = 0;
  PreadExact(shard_index, reinterpret_cast<uint8_t*>(&header_length), 0, kSafetensorsHeaderLengthBytes);
  DSV4_REQUIRE(header_length > 0 && header_length <= kMaxHeaderBytes,
               shard.path << " declares a " << header_length << "-byte header");
  DSV4_REQUIRE(kSafetensorsHeaderLengthBytes + header_length <= shard.file_size,
               shard.path << " header runs past the end of the file");
  std::vector<char> header(static_cast<size_t>(header_length));
  PreadExact(shard_index, reinterpret_cast<uint8_t*>(header.data()), kSafetensorsHeaderLengthBytes, header.size());
  shard.data_start = kSafetensorsHeaderLengthBytes + header_length;

  std::map<std::string, SafetensorsTensor> parsed = ParseSafetensorsHeaderJson(header.data(), header.size());
  for (auto& entry : parsed) {
    DSV4_REQUIRE(shard.data_start + entry.second.payload_end <= shard.file_size,
                 "tensor " << entry.first << " in " << shard.path << " ends past the file");
    entry.second.shard = shard_index;
    // Later shards never shadow an earlier definition: a duplicated tensor name
    // across shards is a corrupt checkpoint, not a precedence question.
    const auto inserted = tensors_.emplace(entry.first, entry.second);
    DSV4_REQUIRE(inserted.second, "tensor " << entry.first << " is defined by more than one shard");
  }
}

void SafetensorsWeightSource::PreadExact(int32_t shard_index, uint8_t* destination, uint64_t file_offset,
                                         size_t count) {
  const Shard& shard = shards_[static_cast<size_t>(shard_index)];
  DSV4_REQUIRE(shard.fd >= 0, "shard " << shard.path << " is closed");
  size_t done = 0;
  while (done < count) {
    const ssize_t read_bytes =
        ::pread(shard.fd, destination + done, count - done, static_cast<off_t>(file_offset + done));
    if (read_bytes < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw Dsv4Error("pread on " + shard.path + " failed: " + std::strerror(errno));
    }
    DSV4_REQUIRE(read_bytes > 0, "short read on " << shard.path << " at offset " << (file_offset + done));
    done += static_cast<size_t>(read_bytes);
  }
  bytes_read_ += count;
  ++read_requests_;
}

std::string SafetensorsWeightSource::ExpertTensorName(int32_t layer, int32_t expert, ExpertProjection projection,
                                                      bool is_scale) const {
  std::ostringstream name;
  if (naming_ == CheckpointNaming::kDsv4Flat) {
    const char* leaf = projection == ExpertProjection::kGate ? "w1"
                       : projection == ExpertProjection::kUp ? "w3"
                                                             : "w2";
    name << "layers." << layer << ".ffn.experts." << expert << "." << leaf << "." << (is_scale ? "scale" : "weight");
  } else {
    const char* leaf = projection == ExpertProjection::kGate ? "gate_proj"
                       : projection == ExpertProjection::kUp ? "up_proj"
                                                             : "down_proj";
    name << "model.layers." << layer << ".mlp.experts." << expert << "." << leaf << "."
         << (is_scale ? "weight_scale_inv" : "weight");
  }
  return name.str();
}

bool SafetensorsWeightSource::Contains(int32_t layer, int32_t expert) const {
  if (layer < 0 || layer >= num_layers_ || expert < 0 || expert >= num_experts_) {
    return false;
  }
  return tensors_.count(ExpertTensorName(layer, expert, ExpertProjection::kGate, false)) != 0;
}

const SafetensorsTensor& SafetensorsWeightSource::Tensor(const std::string& name) const {
  const auto found = tensors_.find(name);
  DSV4_REQUIRE(found != tensors_.end(), "checkpoint has no tensor " << name);
  return found->second;
}

bool SafetensorsWeightSource::HasNamed(const std::string& name) const { return tensors_.count(name) != 0; }

size_t SafetensorsWeightSource::NamedByteSize(const std::string& name) const {
  return static_cast<size_t>(Tensor(name).num_bytes());
}

void SafetensorsWeightSource::ReadNamed(const std::string& name, uint8_t* destination, size_t destination_capacity,
                                        size_t byte_offset, size_t count) {
  RefuseIfClosed(name.c_str());
  DSV4_REQUIRE(count <= destination_capacity,
               "named read of " << count << " bytes exceeds destination capacity " << destination_capacity);
  const SafetensorsTensor& tensor = Tensor(name);
  DSV4_REQUIRE(byte_offset + count <= tensor.num_bytes(),
               "named read [" << byte_offset << ", " << (byte_offset + count) << ") leaves tensor " << name
                              << " of " << tensor.num_bytes() << " bytes");
  const Shard& shard = shards_[static_cast<size_t>(tensor.shard)];
  PreadExact(tensor.shard, destination, shard.data_start + tensor.payload_begin + byte_offset, count);
}

void SafetensorsWeightSource::ValidateExpertBinding(int32_t layer, int32_t expert) const {
  const ExpertRegionSpec& gate_up = layout_.region(ExpertRegionId::kGateUpWeight);
  const ExpertRegionSpec& gate_up_scale = layout_.region(ExpertRegionId::kGateUpScale);
  const ExpertRegionSpec& down = layout_.region(ExpertRegionId::kDownWeight);
  const ExpertRegionSpec& down_scale = layout_.region(ExpertRegionId::kDownScale);

  // Header dtype strings accepted for the two routed storage kinds. Byte counts
  // alone cannot tell FP4-packed weights (ACL_FLOAT4_E2M1, 2 nibbles/byte) from
  // a half-width dense tensor of another dtype, nor an E8M0 block-32 scale
  // (ACL_FLOAT8_E8M0, 1 byte per 32 elements) from a per-element UINT8 scale,
  // so the unpack mapping is only provably right when the dtype string says
  // the checkpoint really is FP4 + UE8M0. Mirrors the dtype gate in
  // hardware/sharded_safetensors.py (_validate_dtype_and_shape).
  static const char* const kFp4WeightDtypes[] = {"MOE_F4", "F4_E2M1"};
  static const char* const kE8m0ScaleDtypes[] = {"MOE_F4_SCALE", "F8_E8M0", "UE8M0"};

  struct Binding {
    ExpertProjection projection;
    bool is_scale;
    uint64_t expected_bytes;
    int64_t expected_rows;
    int64_t expected_stored_cols;
  };
  // gate and up each contribute half of the fused w13 region. Shapes are the
  // BYTE view (rows x stored_cols): FP4 nibble pairs and E8M0 scale bytes stay
  // byte-shaped in the checkpoint, exactly as core/layout.py's logical_shape
  // defines it for these two kinds.
  const Binding bindings[] = {
      {ExpertProjection::kGate, false, gate_up.num_bytes() / 2, gate_up.rows / 2,
       static_cast<int64_t>(gate_up.stored_cols())},
      {ExpertProjection::kUp, false, gate_up.num_bytes() / 2, gate_up.rows / 2,
       static_cast<int64_t>(gate_up.stored_cols())},
      {ExpertProjection::kGate, true, gate_up_scale.num_bytes() / 2, gate_up_scale.rows / 2,
       static_cast<int64_t>(gate_up_scale.stored_cols())},
      {ExpertProjection::kUp, true, gate_up_scale.num_bytes() / 2, gate_up_scale.rows / 2,
       static_cast<int64_t>(gate_up_scale.stored_cols())},
      {ExpertProjection::kDown, false, down.num_bytes(), down.rows, static_cast<int64_t>(down.stored_cols())},
      {ExpertProjection::kDown, true, down_scale.num_bytes(), down_scale.rows,
       static_cast<int64_t>(down_scale.stored_cols())},
  };
  for (const Binding& binding : bindings) {
    const std::string name = ExpertTensorName(layer, expert, binding.projection, binding.is_scale);
    const SafetensorsTensor& tensor = Tensor(name);
    DSV4_REQUIRE(tensor.num_bytes() == binding.expected_bytes,
                 "tensor " << name << " holds " << tensor.num_bytes() << " bytes, the expert slot layout needs "
                           << binding.expected_bytes);
    const char* const* accepted = binding.is_scale ? kE8m0ScaleDtypes : kFp4WeightDtypes;
    const size_t accepted_count = binding.is_scale ? sizeof(kE8m0ScaleDtypes) / sizeof(kE8m0ScaleDtypes[0])
                                                   : sizeof(kFp4WeightDtypes) / sizeof(kFp4WeightDtypes[0]);
    bool dtype_ok = false;
    for (size_t index = 0; index < accepted_count; ++index) {
      dtype_ok = dtype_ok || tensor.dtype == accepted[index];
    }
    DSV4_REQUIRE(dtype_ok,
                 "tensor " << name << " has safetensors dtype '" << tensor.dtype
                           << "', but the slot layout is packed FP4 E2M1 weights with E8M0 block-"
                           << kRoutedScaleBlock << " scales; expected one of '"
                           << (binding.is_scale ? "MOE_F4_SCALE/F8_E8M0/UE8M0" : "MOE_F4/F4_E2M1") << "'");
    if (tensor.shape.size() == 2) {
      DSV4_REQUIRE(tensor.shape[0] == binding.expected_rows &&
                       tensor.shape[1] == binding.expected_stored_cols,
                   "tensor " << name << " has shape [" << tensor.shape[0] << ", " << tensor.shape[1]
                             << "], the byte view of its slot region is [" << binding.expected_rows << ", "
                             << binding.expected_stored_cols << "]");
    }
  }
}

void SafetensorsWeightSource::ReadRegionWindow(const ExpertRegionSpec& spec, uint8_t* destination,
                                               size_t window_begin, size_t window_end, int32_t layer,
                                               int32_t expert) {
  const bool fused = spec.id == ExpertRegionId::kGateUpWeight || spec.id == ExpertRegionId::kGateUpScale;
  const bool is_scale = spec.is_scale();
  const size_t region_bytes = spec.num_bytes();
  // Within the region, byte `r` belongs to gate for r < half and to up above
  // it, because the fusion stacks gate above up on the row axis and the region
  // is row-major.
  const size_t half = fused ? region_bytes / 2 : region_bytes;

  struct Piece {
    ExpertProjection projection;
    size_t region_begin;
    size_t region_end;
  };
  Piece pieces[2];
  size_t piece_count = 0;
  if (fused) {
    pieces[piece_count++] = {ExpertProjection::kGate, 0, half};
    pieces[piece_count++] = {ExpertProjection::kUp, half, region_bytes};
  } else {
    pieces[piece_count++] = {ExpertProjection::kDown, 0, region_bytes};
  }

  for (size_t index = 0; index < piece_count; ++index) {
    const Piece& piece = pieces[index];
    const size_t begin = std::max(window_begin, piece.region_begin);
    const size_t end = std::min(window_end, piece.region_end);
    if (begin >= end) {
      continue;
    }
    const std::string name = ExpertTensorName(layer, expert, piece.projection, is_scale);
    const SafetensorsTensor& tensor = Tensor(name);
    const size_t tensor_offset = begin - piece.region_begin;
    const size_t count = end - begin;
    DSV4_REQUIRE(tensor_offset + count <= tensor.num_bytes(),
                 "tensor " << name << " is too small for slot window [" << begin << ", " << end << ")");
    const Shard& shard = shards_[static_cast<size_t>(tensor.shard)];
    PreadExact(tensor.shard, destination + (begin - window_begin),
               shard.data_start + tensor.payload_begin + tensor_offset, count);
  }
}

void SafetensorsWeightSource::ReadExpertSlotRange(uint8_t* destination, size_t destination_capacity,
                                                  size_t slot_offset, size_t count, int32_t layer, int32_t expert) {
  RefuseIfClosed("ReadExpertSlotRange");
  // The unpack mapping is checked BEFORE any byte moves, on every window, so an
  // FP8 / BF16 / reshaped checkpoint bound against the packed-FP4 slot layout
  // is refused here rather than silently reinterpreted (byte counts alone
  // cannot tell the difference). FillExpertSlot -- the host-resident path --
  // funnels through here too, so both ingestion paths are covered.
  ValidateExpertBinding(layer, expert);
  DSV4_REQUIRE(count <= destination_capacity,
               "slot read of " << count << " bytes exceeds destination capacity " << destination_capacity);
  DSV4_REQUIRE(slot_offset + count <= layout_.slot_num_bytes(),
               "slot read [" << slot_offset << ", " << (slot_offset + count) << ") leaves the "
                             << layout_.slot_num_bytes() << "-byte slot");
  const size_t request_end = slot_offset + count;
  for (const ExpertRegionSpec& spec : layout_.regions()) {
    const size_t region_begin = spec.offset_bytes;
    const size_t region_end = spec.offset_bytes + spec.num_bytes();
    const size_t overlap_begin = std::max(slot_offset, region_begin);
    const size_t overlap_end = std::min(request_end, region_end);
    if (overlap_begin >= overlap_end) {
      continue;
    }
    ReadRegionWindow(spec, destination + (overlap_begin - slot_offset), overlap_begin - region_begin,
                     overlap_end - region_begin, layer, expert);
  }
}

void SafetensorsWeightSource::Close() {
  for (Shard& shard : shards_) {
    if (shard.fd >= 0) {
      DSV4_REQUIRE(::close(shard.fd) == 0, "closing " << shard.path << " failed: " << std::strerror(errno));
      shard.fd = -1;
    }
  }
  closed_ = true;
}

}  // namespace ascend_moe
