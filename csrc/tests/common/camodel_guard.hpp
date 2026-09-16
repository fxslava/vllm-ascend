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

#pragma once

#include <dirent.h>
#include <sys/stat.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace vllm_ascend {
namespace test {

constexpr int kCamodelHangExitCode = 3;

class LaunchWatchdog {
 public:
  LaunchWatchdog(int64_t budget_seconds, std::string tag)
      : budget_(budget_seconds), tag_(std::move(tag)), thread_([this] { Watch(); }) {}

  ~LaunchWatchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }

  LaunchWatchdog(const LaunchWatchdog&) = delete;
  LaunchWatchdog& operator=(const LaunchWatchdog&) = delete;

  void Arm(const std::string& what) {
    std::lock_guard<std::mutex> lock(mutex_);
    what_ = what;
    deadline_ = std::chrono::steady_clock::now() + budget_;
    armed_ = true;
    ++generation_;
    wake_.notify_all();
  }

  void Disarm() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    ++generation_;
    wake_.notify_all();
  }

 private:
  void Watch() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (!armed_) {
        wake_.wait(lock, [this] { return stop_ || armed_; });
        continue;
      }
      const uint64_t generation = generation_;
      const bool changed =
          wake_.wait_until(lock, deadline_, [this, generation] { return stop_ || generation_ != generation; });
      if (!changed) {
        std::printf("\n%s HANG: %s did not return within %lld s; ending the process with %d\n", tag_.c_str(),
                    what_.c_str(), static_cast<long long>(budget_.count()), kCamodelHangExitCode);
        std::fflush(stdout);
        std::_Exit(kCamodelHangExitCode);
      }
    }
  }

  const std::chrono::seconds budget_;
  const std::string tag_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::string what_;
  std::chrono::steady_clock::time_point deadline_;
  bool armed_ = false;
  bool stop_ = false;
  uint64_t generation_ = 0;
  std::thread thread_;
};

struct DumpCensus {
  size_t files = 0;
  size_t non_empty = 0;
  uint64_t bytes = 0;
};

inline DumpCensus ExceptionDumps() {
  DumpCensus census;
  DIR* dir = opendir(".");
  if (dir == nullptr) {
    return census;
  }
  while (const dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    const bool is_dump = name.size() > 5 && name.compare(name.size() - 5, 5, ".dump") == 0;
    if (!is_dump || name.find("excp_log") == std::string::npos) {
      continue;
    }
    struct stat info;
    if (stat(entry->d_name, &info) != 0) {
      continue;
    }
    ++census.files;
    census.bytes += static_cast<uint64_t>(info.st_size);
    if (info.st_size > 0) {
      ++census.non_empty;
    }
  }
  closedir(dir);
  return census;
}

}
}
