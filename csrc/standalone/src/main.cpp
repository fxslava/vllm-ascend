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

// dsv4_runner -- standalone DeepSeek-V4 Flash decode on Ascend 950PR.
//
//   ./dsv4_runner --weights <path> --prompt "..." --max-new-tokens 512 --vram-slots <K>
//
// No Python, no PyTorch: `ldd` on this binary carries libascendcl, libnnopbase
// and the libopapi family, and nothing else from a framework.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "moe/core/weight_source.hpp"

namespace ascend_moe {
namespace {

void PrintUsage() {
  std::printf(
      "dsv4_runner -- DeepSeek-V4 Flash decode, Ascend 950PR, ACLNN V5\n"
      "\n"
      "  --weights <path>          checkpoint directory or .safetensors file\n"
      "  --prompt \"...\"            prompt text (token ids are read from --prompt-ids)\n"
      "  --prompt-ids a,b,c        explicit prompt token ids; this runner embeds no tokenizer\n"
      "  --max-new-tokens <n>      tokens to generate (default 512)\n"
      "  --vram-slots <K>          resident routed-expert slots in HBM (default: as many as fit)\n"
      "  --device <id>             NPU device id (default 0)\n"
      "  --block-size <n>          paged KV block size (default 128)\n"
      "  --max-context <n>         reserved context length (default 8192)\n"
      "  --kv-lora-rank <n>        MLA latent width (default %lld, family default)\n"
      "  --qk-rope-head-dim <n>    MLA rope slice width (default %lld, family default)\n"
      "  --qk-nope-head-dim <n>    MLA nope head width (default %lld, family default)\n"
      "  --v-head-dim <n>          MLA value head width (default %lld, family default)\n"
      "  --moe-path fused|decomposed   expert GEMM chain (default fused; see README)\n"
      "  --gating-norm-type <n>    aclnnMoeGatingTopKV2 normType (default -1: scores arrive pre-normalized\n"
      "                            from the decomposed aclnnSoftplus -> aclnnSqrt sqrtsoftplus chain)\n"
      "  --dense-group-size <n>    aclnnQuantMatmulV5 groupSize (default 0, UNVERIFIED)\n"
      "  --routed-coverage <n>     |Set_Device| + |Set_Host|, default %lld (bring-up subset below that)\n"
      "  --synthetic-weights       deterministic in-memory weights; no files are opened\n"
      "  --dry-run                 build, plan, report and exit without decoding\n"
      "  --report <path>           write the full report there as well as to stdout\n"
      "  --verbose                 print the arena ledger and operator inventory\n"
      "  --help\n",
      static_cast<long long>(kDefaultKvLoraRank), static_cast<long long>(kDefaultQkRopeHeadDim),
      static_cast<long long>(kDefaultQkNopeHeadDim), static_cast<long long>(kDefaultVHeadDim),
      static_cast<long long>(kTotalRoutedExperts));
}

int64_t ParseInt(const char* text, const char* flag) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  DSV4_REQUIRE(end != nullptr && *end == '\0', flag << " expects an integer, got '" << text << "'");
  return static_cast<int64_t>(value);
}

std::vector<int32_t> ParseIdList(const std::string& text) {
  std::vector<int32_t> ids;
  std::stringstream stream(text);
  std::string piece;
  while (std::getline(stream, piece, ',')) {
    if (!piece.empty()) {
      ids.push_back(static_cast<int32_t>(ParseInt(piece.c_str(), "--prompt-ids")));
    }
  }
  return ids;
}

struct Arguments {
  RuntimeConfig config;
  std::vector<int32_t> prompt_ids;
  bool help = false;
};

Arguments ParseArguments(int argc, char** argv) {
  Arguments arguments;
  RuntimeConfig& config = arguments.config;
  bool mla_from_cli = false;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    auto next = [&](const char* name) -> const char* {
      DSV4_REQUIRE(index + 1 < argc, name << " needs a value");
      return argv[++index];
    };
    if (flag == "--help" || flag == "-h") {
      arguments.help = true;
    } else if (flag == "--weights") {
      config.weights_path = next("--weights");
    } else if (flag == "--prompt") {
      config.prompt = next("--prompt");
    } else if (flag == "--prompt-ids") {
      arguments.prompt_ids = ParseIdList(next("--prompt-ids"));
    } else if (flag == "--max-new-tokens") {
      config.max_new_tokens = ParseInt(next("--max-new-tokens"), "--max-new-tokens");
    } else if (flag == "--vram-slots") {
      config.vram_slots = ParseInt(next("--vram-slots"), "--vram-slots");
    } else if (flag == "--device") {
      config.device_id = static_cast<int32_t>(ParseInt(next("--device"), "--device"));
    } else if (flag == "--block-size") {
      config.block_size = ParseInt(next("--block-size"), "--block-size");
    } else if (flag == "--max-context") {
      config.max_context_len = ParseInt(next("--max-context"), "--max-context");
    } else if (flag == "--kv-lora-rank") {
      config.mla.kv_lora_rank = ParseInt(next("--kv-lora-rank"), "--kv-lora-rank");
      mla_from_cli = true;
    } else if (flag == "--qk-rope-head-dim") {
      config.mla.qk_rope_head_dim = ParseInt(next("--qk-rope-head-dim"), "--qk-rope-head-dim");
      mla_from_cli = true;
    } else if (flag == "--qk-nope-head-dim") {
      config.mla.qk_nope_head_dim = ParseInt(next("--qk-nope-head-dim"), "--qk-nope-head-dim");
      mla_from_cli = true;
    } else if (flag == "--v-head-dim") {
      config.mla.v_head_dim = ParseInt(next("--v-head-dim"), "--v-head-dim");
      mla_from_cli = true;
    } else if (flag == "--moe-path") {
      const std::string value = next("--moe-path");
      DSV4_REQUIRE(value == "fused" || value == "decomposed",
                   "--moe-path expects 'fused' or 'decomposed', got '" << value << "'");
      config.moe_path = value == "fused" ? MoePath::kFused : MoePath::kDecomposed;
    } else if (flag == "--gating-norm-type") {
      config.gating_norm_type = ParseInt(next("--gating-norm-type"), "--gating-norm-type");
    } else if (flag == "--dense-group-size") {
      config.dense_group_size = ParseInt(next("--dense-group-size"), "--dense-group-size");
    } else if (flag == "--routed-coverage") {
      config.routed_coverage = ParseInt(next("--routed-coverage"), "--routed-coverage");
    } else if (flag == "--synthetic-weights") {
      config.synthetic_weights = true;
    } else if (flag == "--dry-run") {
      config.dry_run = true;
    } else if (flag == "--report") {
      config.report_path = next("--report");
    } else if (flag == "--verbose") {
      config.verbose = true;
    } else {
      throw Dsv4Error("unknown flag: " + flag);
    }
  }
  if (mla_from_cli) {
    config.mla.provenance = GeometryProvenance::kCommandLine;
  }
  return arguments;
}

// A checkpoint's config.json, when present, is authoritative for the MLA
// geometry. Only the four fields this runner cannot otherwise know are read; a
// missing file leaves the family defaults in place and the report says so.
void LoadGeometryFromCheckpoint(const std::string& weights_path, MlaGeometry* mla) {
  if (weights_path.empty() || mla->provenance == GeometryProvenance::kCommandLine) {
    return;
  }
  const std::string path = weights_path + "/config.json";
  std::ifstream file(path);
  if (!file.is_open()) {
    return;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  bool found_any = false;
  auto read_field = [&](const char* key, int64_t* destination) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) {
      return;
    }
    const size_t colon = text.find(':', at + needle.size());
    if (colon == std::string::npos) {
      return;
    }
    *destination = std::strtoll(text.c_str() + colon + 1, nullptr, 10);
    found_any = true;
  };
  read_field("kv_lora_rank", &mla->kv_lora_rank);
  read_field("qk_rope_head_dim", &mla->qk_rope_head_dim);
  read_field("qk_nope_head_dim", &mla->qk_nope_head_dim);
  read_field("v_head_dim", &mla->v_head_dim);
  if (found_any) {
    mla->provenance = GeometryProvenance::kCheckpointConfig;
  }
}

std::string DescribeConfiguration(const RuntimeConfig& config, const ExpertSlotLayout& layout) {
  std::ostringstream out;
  out << "DeepSeek-V4 Flash standalone runner\n";
  out << "  topology      " << kNumLayers << " layers, hidden " << kHiddenSize << ", moe_intermediate "
      << kMoeIntermediateSize << "\n";
  out << "  experts       " << kNumRoutedExperts << " routed + " << kNumSharedExperts << " shared, top-"
      << kNumExpertsPerTok << ", scaling " << kRoutedScalingFactor << "\n";
  out << "  attention     MLA, " << kNumAttentionHeads << " heads, q_lora " << kQLoraRank << ", kv_lora "
      << config.mla.kv_lora_rank << " + rope " << config.mla.qk_rope_head_dim << " = row "
      << config.mla.kv_row_elements() << "\n";
  out << "  MLA geometry  from " << config.mla.provenance_name() << "\n";
  out << "  precision     dense FP8 E4M3 (" << kAclFloat8E4m3Fn << ") block-" << kDenseScaleBlock
      << " scales; routed FP4 E2M1 (" << kAclFloat4E2m1 << ") + E8M0 (" << kAclFloat8E8m0 << ") block-"
      << kRoutedScaleBlock << "\n";
  out << "  activation    SwiGLU, clamp " << kSwigluLimit << "\n";
  out << "  paged KV      block " << config.block_size << ", context " << config.max_context_len << "\n";
  out << layout.DescribeTable();
  return out.str();
}

int Run(int argc, char** argv) {
  Arguments arguments = ParseArguments(argc, argv);
  if (arguments.help || (argc == 1)) {
    PrintUsage();
    return 0;
  }
  RuntimeConfig& config = arguments.config;
  DSV4_REQUIRE(!config.weights_path.empty() || config.synthetic_weights,
               "--weights is required unless --synthetic-weights is given");

  LoadGeometryFromCheckpoint(config.weights_path, &config.mla);
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();

  std::ostringstream report;
  report << DescribeConfiguration(config, layout);

  OpTable ops;
  if (config.verbose) {
    report << ops.DescribeInventory();
  }
  DSV4_REQUIRE(ops.runtime_reachable(),
               "the aclnn runtime is not on this process's loader path: no operator resolved. Source "
               "set_env.sh, or add $ASCEND_TOOLKIT_HOME/<arch>-linux/lib64 to LD_LIBRARY_PATH.");

  // The device. A missing NPU is reported as itself, not as a fallback.
  AclDeviceOps device(config.device_id);
  report << "  device        " << config.device_id << ", SoC " << device.soc_name() << "\n";

  size_t free_hbm = 0;
  size_t total_hbm = 0;
  DSV4_REQUIRE(device.QueryDeviceMemory(&free_hbm, &total_hbm), "aclrtGetMemInfo(ACL_HBM_MEM) failed");
  const size_t backbone_bytes =
      Dsv4Pipeline::BackboneDeviceBytes(config.mla, config.block_size, config.max_context_len);
  const int64_t slots = ExclusiveExpertManager::PlanDeviceSlots(
      free_hbm, backbone_bytes, layout.slot_num_bytes(), config.routed_coverage, kDeviceReserveBytes,
      kTransferChunkBytes, kNumExpertsPerTok, config.vram_slots);
  report << "  HBM           " << (free_hbm >> 20) << " MiB free of " << (total_hbm >> 20) << " MiB; backbone "
         << (backbone_bytes >> 20) << " MiB; K = " << slots << " routed slots\n";

  ExclusiveExpertManager::Options options;
  options.routed_coverage = config.routed_coverage;
  options.device_slots = slots;
  ExclusiveExpertManager experts(device, device, layout, options);
  report << experts.DescribeHierarchy();

  std::unique_ptr<WeightByteSource> source;
  if (config.synthetic_weights) {
    source = std::make_unique<SyntheticWeightSource>(layout, kNumLayers, kNumRoutedExperts);
  } else {
    source = std::make_unique<SafetensorsWeightSource>(config.weights_path, layout, CheckpointNaming::kDsv4Flat,
                                                       kNumLayers, kNumRoutedExperts);
  }

  Dsv4Pipeline pipeline(device, device, ops, experts, config);
  // Order matters: the pipeline takes the backbone tensors first, then the
  // expert manager takes the routed slots and seals the source. Exactly one
  // owner closes it, and after that the hierarchy is the only copy.
  pipeline.Build(*source);
  experts.Ingest(*source, {});
  WeightByteSource::AssertNoOpenWeightDescriptors(config.synthetic_weights ? std::string() : config.weights_path);
  report << "  ingestion     " << (experts.stats().startup_bytes >> 20) << " MiB of routed experts, source '"
         << source->source_name() << "' closed, no descriptor left open\n";

  report << pipeline.DescribeStages();
  report << pipeline.DescribeSlotIndexMap();
  if (config.verbose) {
    report << pipeline.arena_manager().arena().DescribeLedger();
  }

  if (config.dry_run) {
    report << "\n--dry-run: built, planned and verified; no token was decoded.\n";
  } else {
    DSV4_REQUIRE(!arguments.prompt_ids.empty(),
                 "--prompt-ids is required to decode: this runner embeds no tokenizer, so the prompt text in "
                 "--prompt is recorded but not tokenized. Pass the ids your tokenizer produced.");
    std::vector<int32_t> generated;
    int64_t position = 0;
    // Prefill is run as a sequence of single-token steps: the decode graph is
    // shaped for one token (see kTokensPerStep), so a batched prefill would need
    // a second set of descriptors. It is correct, and it is O(prompt) steps.
    for (size_t index = 0; index < arguments.prompt_ids.size(); ++index) {
      pipeline.DecodeStep(arguments.prompt_ids[index], position++);
    }
    int32_t token = pipeline.ReadArgmaxToken();
    for (int64_t index = 0; index < config.max_new_tokens && position < config.max_context_len; ++index) {
      generated.push_back(token);
      pipeline.DecodeStep(token, position++);
      token = pipeline.ReadArgmaxToken();
    }
    report << "\ngenerated " << generated.size() << " token ids:";
    for (int32_t id : generated) {
      report << " " << id;
    }
    report << "\n";
    const StepCounters& counters = pipeline.counters();
    report << "steps " << counters.steps << ", layers " << counters.layers << ", launches " << counters.launches
           << "\n";
    report << "expert residency: " << counters.expert_slot_hits << " hits, " << counters.expert_slot_misses
           << " misses\n";
    report << "host synchronizations " << counters.host_synchronizations << " (expected "
           << counters.steps * (kNumLayers + 1) << ": one per MoE layer plus one per step readback)\n";
    report << "allocations inside a step " << counters.device_allocations_in_step << " (must be 0); descriptors "
           << counters.descriptors_built_in_step << " (must be 0)\n";
    DSV4_REQUIRE(counters.device_allocations_in_step == 0,
                 counters.device_allocations_in_step << " device allocations happened inside a decode step");
    DSV4_REQUIRE(counters.descriptors_built_in_step == 0,
                 counters.descriptors_built_in_step << " descriptors were built inside a decode step");
  }

  experts.Synchronize();
  experts.ValidateResidency();
  report << "exclusive residency invariant holds after the run.\n";

  const std::string text = report.str();
  std::fputs(text.c_str(), stdout);
  if (!config.report_path.empty()) {
    std::ofstream file(config.report_path);
    DSV4_REQUIRE(file.is_open(), "cannot write the report to " << config.report_path);
    file << text;
  }
  return 0;
}

}  // namespace
}  // namespace ascend_moe

int main(int argc, char** argv) {
  try {
    return ascend_moe::Run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "dsv4_runner: %s\n", error.what());
    return 1;
  }
}
