/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_buffer.hpp"
#include "turboquant/turboquant_launch.hpp"

#ifndef VLLM_ASCEND_TQ_STAGE_DUMP_PATH
  #error "The KV3 stage dump target must be set by the device test CMake target"
#endif

namespace vllm_ascend::test::turboquant_dump {

// CRC-32/ISO-HDLC, computed over logical token/head/word order, not NZ GM order.
inline uint32_t Crc32(const uint8_t* bytes, size_t count) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < count; ++i) {
    crc ^= bytes[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
    }
  }
  return ~crc;
}

inline std::ofstream OpenDump() {
  const std::filesystem::path path(VLLM_ASCEND_TQ_STAGE_DUMP_PATH);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Cannot open KV3 stage dump: " + path.string());
  }
  return out;
}

inline uint32_t WordAt(const std::vector<int8_t>& bytes, size_t row, size_t kv_head, size_t word, size_t kv_heads,
                       size_t packed_bytes, bool nz_tiled) {
  uint32_t result = 0;
  for (size_t lane = 0; lane < sizeof(uint32_t); ++lane) {
    const size_t column = word * sizeof(uint32_t) + lane;
    const size_t offset =
        nz_tiled ? static_cast<size_t>(turboquant::NzTiledPackedByte(row, kv_head, column, kv_heads, packed_bytes))
                 : (row * kv_heads + kv_head) * packed_bytes + column;
    result |= static_cast<uint32_t>(static_cast<uint8_t>(bytes.at(offset))) << (8 * lane);
  }
  return result;
}

inline size_t DumpPackedStage(const char* stage, const DeviceBuffer& debug_gm, const std::vector<int8_t>& expected,
                              size_t rows, size_t kv_heads, size_t packed_bytes, bool nz_tiled,
                              size_t actual_offset_bytes = 0, size_t storage_rows = 0) {
  const std::vector<int8_t> all_actual = debug_gm.ToHost<int8_t>();
  const size_t plane_bytes = rows * kv_heads * packed_bytes;
  const size_t physical_rows = nz_tiled ? storage_rows : rows;
  const size_t storage_bytes = physical_rows * kv_heads * packed_bytes;
  if (packed_bytes % sizeof(uint32_t) != 0 || (nz_tiled && storage_rows < rows) || expected.size() != storage_bytes ||
      all_actual.size() < actual_offset_bytes + storage_bytes) {
    throw std::invalid_argument("KV3 packed stage has an invalid plane extent");
  }
  const auto actual_begin = all_actual.begin() + static_cast<std::ptrdiff_t>(actual_offset_bytes);
  const auto actual_end = actual_begin + static_cast<std::ptrdiff_t>(storage_bytes);
  const std::vector<int8_t> actual(actual_begin, actual_end);
  const size_t words_per_head = packed_bytes / sizeof(uint32_t);
  const size_t total_words = rows * kv_heads * words_per_head;
  std::vector<uint8_t> actual_logical;
  std::vector<uint8_t> expected_logical;
  actual_logical.reserve(plane_bytes);
  expected_logical.reserve(plane_bytes);
  std::string differences;
  size_t mismatches = 0;
  char line[128];
  for (size_t token = 0; token < rows; ++token) {
    for (size_t head = 0; head < kv_heads; ++head) {
      for (size_t word = 0; word < words_per_head; ++word) {
        const uint32_t got = WordAt(actual, token, head, word, kv_heads, packed_bytes, nz_tiled);
        const uint32_t want = WordAt(expected, token, head, word, kv_heads, packed_bytes, nz_tiled);
        for (size_t lane = 0; lane < sizeof(uint32_t); ++lane) {
          actual_logical.push_back(static_cast<uint8_t>(got >> (8 * lane)));
          expected_logical.push_back(static_cast<uint8_t>(want >> (8 * lane)));
        }
        if (got != want) {
          ++mismatches;
          if (kv_heads == 1) {
            std::snprintf(line, sizeof(line), "T%02zu W%02zu: GOT 0x%08X WANT 0x%08X (XOR 0x%08X)\n", token, word, got,
                          want, got ^ want);
          } else {
            std::snprintf(line, sizeof(line), "T%02zu KV%02zu W%02zu: GOT 0x%08X WANT 0x%08X (XOR 0x%08X)\n", token,
                          head, word, got, want, got ^ want);
          }
          differences += line;
        }
      }
    }
  }
  const uint32_t actual_crc = Crc32(actual_logical.data(), actual_logical.size());
  const uint32_t expected_crc = Crc32(expected_logical.data(), expected_logical.size());
  std::ofstream out = OpenDump();
  if (mismatches == 0) {
    std::snprintf(line, sizeof(line), "[%s] ALL %zu WORDS MATCH (CRC=0x%08X) PASS\n", stage, total_words, actual_crc);
    out << line;
  } else {
    std::snprintf(line, sizeof(line),
                  "[%s] STATUS=FAIL TOPO=D256/H%zu/KV%zu MISMATCHES=%zu/%zu "
                  "CRC_ACTUAL=0x%08X CRC_EXPECTED=0x%08X\n",
                  stage, kv_heads, kv_heads, mismatches, total_words, actual_crc, expected_crc);
    out << line << differences;
  }
  out.flush();
  if (!out) {
    throw std::runtime_error("Failed to write KV3 packed stage dump");
  }
  return mismatches;
}

inline size_t DumpScaleStage(const char* stage, const DeviceBuffer& debug_gm, const std::vector<float>& expected,
                             size_t rows, size_t kv_heads, size_t plane, double relative_tolerance = 1.0e-3) {
  const std::vector<float> actual = debug_gm.ToHost<float>();
  const size_t stride = static_cast<size_t>(turboquant_host::ScaleSlotFloats(kv_heads));
  if (actual.size() < rows * stride || expected.size() < rows * stride || plane > 1) {
    throw std::invalid_argument("KV3 scale stage has an invalid plane extent");
  }
  std::vector<uint8_t> actual_bits;
  std::vector<uint8_t> expected_bits;
  std::string differences;
  size_t mismatches = 0;
  char line[160];
  for (size_t token = 0; token < rows; ++token) {
    for (size_t head = 0; head < kv_heads; ++head) {
      const size_t lane = token * stride + plane * kv_heads + head;
      uint32_t got_bits = 0;
      uint32_t want_bits = 0;
      std::memcpy(&got_bits, &actual[lane], sizeof(got_bits));
      std::memcpy(&want_bits, &expected[lane], sizeof(want_bits));
      for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
        actual_bits.push_back(static_cast<uint8_t>(got_bits >> (8 * byte)));
        expected_bits.push_back(static_cast<uint8_t>(want_bits >> (8 * byte)));
      }
      const double denominator = std::max(std::abs(static_cast<double>(expected[lane])), 1.0e-20);
      const double relative_error = std::abs(static_cast<double>(actual[lane]) - expected[lane]) / denominator;
      if (!std::isfinite(relative_error) || relative_error > relative_tolerance) {
        ++mismatches;
        std::snprintf(line, sizeof(line), "T%02zu KV%02zu: GOT 0x%08X (%+.4e) WANT 0x%08X (%+.4e) REL %.3e\n", token,
                      head, got_bits, actual[lane], want_bits, expected[lane], relative_error);
        differences += line;
      }
    }
  }
  const uint32_t actual_crc = Crc32(actual_bits.data(), actual_bits.size());
  const uint32_t expected_crc = Crc32(expected_bits.data(), expected_bits.size());
  const size_t total = rows * kv_heads;
  std::ofstream out = OpenDump();
  if (mismatches == 0) {
    std::snprintf(line, sizeof(line), "[%s] ALL %zu SCALE LANES MATCH (CRC=0x%08X) PASS\n", stage, total, actual_crc);
    out << line;
  } else {
    std::snprintf(line, sizeof(line),
                  "[%s] STATUS=FAIL TOPO=D256/H%zu/KV%zu MISMATCHES=%zu/%zu "
                  "CRC_ACTUAL=0x%08X CRC_EXPECTED=0x%08X\n",
                  stage, kv_heads, kv_heads, mismatches, total, actual_crc, expected_crc);
    out << line << differences;
  }
  out.flush();
  if (!out) {
    throw std::runtime_error("Failed to write KV3 scale stage dump");
  }
  return mismatches;
}

}  // namespace vllm_ascend::test::turboquant_dump
