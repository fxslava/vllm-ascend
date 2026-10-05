// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

namespace vllm_ascend::test {
template <typename Expert>
auto RunConcurrentExperts(const std::vector<std::unique_ptr<Expert>>& experts) {
  // Validate streams before any submission, then enqueue every expert before
  // reading any output. The expert owner drains its stream before HBM release.
  for (size_t i = 0; i < experts.size(); ++i) {
    if (!experts[i]) throw std::invalid_argument("null concurrent expert");
    for (size_t j = 0; j < i; ++j)
      if (experts[i]->stream() == experts[j]->stream()) throw std::invalid_argument("shared expert stream");
  }
  using Output = decltype(experts.front()->Read());
  std::vector<Output> outputs;
  outputs.reserve(experts.size());
  for (const auto& expert : experts) expert->Enqueue();
  for (const auto& expert : experts) outputs.push_back(expert->Read());
  return outputs;
}
}  // namespace vllm_ascend::test
