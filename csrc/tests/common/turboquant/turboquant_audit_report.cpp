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

#include "turboquant_audit_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <ios>
#include <sstream>

namespace vllm_ascend {
namespace test {
namespace turboquant_audit {

namespace tqh = turboquant_host;

namespace {

using bench::BenchmarkResult;
using bench::BenchmarkRunner;
using bench::TimingMode;
using bench::TimingModeLabel;

const BenchmarkResult* FindResult(const std::vector<BenchmarkRunner*>& runners, const std::string& name,
                                  TimingMode mode) {
  for (const BenchmarkRunner* runner : runners) {
    for (const BenchmarkResult& result : runner->results()) {
      if (result.case_name == name && result.mode == mode) {
        return &result;
      }
    }
  }
  return nullptr;
}

Sample SampleFor(const std::vector<BenchmarkRunner*>& runners, const char* leg, const Config& config) {
  Sample sample;
  const std::string name = CaseName(leg, config);
  for (const TimingMode mode : {TimingMode::kDeviceEvents, TimingMode::kPipelined}) {
    const BenchmarkResult* result = FindResult(runners, name, mode);
    if (result != nullptr && result->latency.median_us > 0.0) {
      sample.present = true;
      sample.median_us = result->latency.median_us;
      sample.p95_us = result->latency.p95_us;
      sample.gigabytes_per_second = result->gigabytes_per_second();
      sample.tflops = result->tflops();
      sample.mode = TimingModeLabel(mode);
      return sample;
    }
  }
  return sample;
}

Sample StructuralZero() {
  Sample sample;
  sample.present = true;
  sample.structural_zero = true;
  sample.mode = "folded";
  return sample;
}

// A decode that rotates its raw query in its own launch has no rotate_q to time.
Sample InLaunchZero() {
  Sample sample = StructuralZero();
  sample.mode = "in-launch";
  return sample;
}

Sample RotateOutputSample(const std::vector<BenchmarkRunner*>& runners, const char* leg, const Config& config) {
  return config.model.folds_output ? StructuralZero() : SampleFor(runners, leg, config);
}

double Speedup(const Sample& baseline, const Sample& candidate) {
  if (!baseline.present || !candidate.present || candidate.median_us <= 0.0) {
    return 0.0;
  }
  return baseline.median_us / candidate.median_us;
}

void PrintUs(const Sample& sample, int width) {
  if (sample.present) {
    std::printf(" %*.2f", width, sample.median_us);
  } else {
    std::printf(" %*s", width, "-");
  }
}

void PrintRatio(double ratio, int width) {
  if (ratio > 0.0) {
    std::printf(" %*.3fx", width - 1, ratio);
  } else {
    std::printf(" %*s", width, "-");
  }
}

void PrintHeaderAndRule(const char* header, int width) {
  std::printf("%s\n", header);
  const size_t indent = 2;
  const size_t rule = width > static_cast<int>(indent) ? static_cast<size_t>(width) - indent : 0;
  std::printf("  %s\n", std::string(rule, '-').c_str());
}

void PrintDouble(double value, int width, int precision) {
  if (value > 0.0) {
    std::printf(" %*.*f", width, precision, value);
  } else {
    std::printf(" %*s", width, "-");
  }
}

PrefillRow ReadPrefillRow(const std::vector<BenchmarkRunner*>& runners, const Config& config, const Traffic& traffic) {
  PrefillRow row;
  row.ingest = SampleFor(runners, kLegPfIngest, config);
  row.rot_q = SampleFor(runners, kLegPfRotQ, config);
  row.attn_core = SampleFor(runners, kLegPfAttnCore, config);
  row.rot_o = RotateOutputSample(runners, kLegPfRotO, config);
  row.e2e = SampleFor(runners, kLegPfE2E, config);
  row.native = SampleFor(runners, kLegPfV5, config);
  row.native_ingest = SampleFor(runners, kLegPfV5Ingest, config);
  row.sum_of_parts_us = (row.rot_q.present ? row.rot_q.median_us : 0.0) +
                        (row.attn_core.present ? row.attn_core.median_us : 0.0) +
                        (row.rot_o.present ? row.rot_o.median_us : 0.0);
  row.speedup = Speedup(row.native, row.e2e);
  if (row.e2e.present && row.e2e.median_us > 0.0) {
    row.gigabytes_per_second = traffic.pf_e2e / (row.e2e.median_us * 1.0e3);
  }
  return row;
}

DecodeRow ReadDecodeRow(const std::vector<BenchmarkRunner*>& runners, const Config& config, const Traffic& traffic,
                        RotationMode mode) {
  const bool separate = mode == RotationMode::kSeparate;
  DecodeRow row;
  row.rot_q = separate ? SampleFor(runners, kLegDecRotQ, config) : InLaunchZero();
  row.attn_core = SampleFor(runners, separate ? kLegDecAttnCore : kLegDecFusedQAttnCore, config);
  row.rot_o = separate || config.model.folds_output ? RotateOutputSample(runners, kLegDecRotO, config) : InLaunchZero();
  row.e2e = SampleFor(runners, separate ? kLegDecE2E : kLegDecFusedQE2E, config);
  row.native = SampleFor(runners, kLegDecV5, config);

  row.sum_of_parts_us = (row.rot_q.present ? row.rot_q.median_us : 0.0) +
                        (row.attn_core.present ? row.attn_core.median_us : 0.0) +
                        (row.rot_o.present ? row.rot_o.median_us : 0.0);
  row.speedup = Speedup(row.native, row.e2e);
  if (row.e2e.present && row.e2e.median_us > 0.0) {
    row.gigabytes_per_second = (separate ? traffic.dec_e2e : traffic.dec_fused_q_e2e) / (row.e2e.median_us * 1.0e3);
  }
  return row;
}

double MedianResidual(const std::vector<double>& residuals) {
  if (residuals.empty()) {
    return 0.0;
  }
  std::vector<double> sorted = residuals;
  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}

void WriteCell(std::ofstream& csv, const Sample& sample) {
  csv << ',';
  if (sample.present) {
    csv << sample.median_us;
  }
}

}  // namespace

void PrintTableA(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator) {
  std::printf("\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] TABLE A -- PREFILL PIPELINE PERFORMANCE\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] device time from ACL events, median us per step; native operator: %s\n",
              fia_operator.c_str());
  std::printf("[ascend-bench] TQ_E2E = T_rot_q + T_attn_core + T_rot_o, measured as ONE composite region\n");
  std::printf(
      "[ascend-bench] Context is the prefix S; every prefill step is chunked: min(S, %lld) query tokens "
      "against it (ASCEND_BENCH_TQ_AUDIT_CHUNK)\n\n",
      static_cast<long long>(PrefillChunkTokens()));

  char header[512];
  const int header_width =
      std::snprintf(header, sizeof(header), "  %-17s %13s %3s %9s %4s | %11s %10s %12s %10s %11s %11s | %8s %10s %12s",
                    "Model", "Context", "B", "H_Q/H_KV", "D", "TQ Ingest", "T_rot_q", "T_attn_core", "T_rot_o",
                    "TQ_E2E", "V5_Native", "Speedup", "HBM GB/s", "KV Saved MB");
  PrintHeaderAndRule(header, header_width);

  bool any = false;
  std::vector<double> residuals;
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const PrefillRow row = ReadPrefillRow(runners, config, traffic[index]);
    if (!row.ingest.present && !row.rot_q.present && !row.attn_core.present && !row.e2e.present &&
        !row.native.present) {
      continue;
    }
    any = true;
    std::ostringstream heads;
    heads << config.model.num_heads << '/' << config.model.num_kv_heads;
    std::printf("  %-17s %13lld %3lld %9s %4lld |", config.model.label, static_cast<long long>(config.seq_len),
                static_cast<long long>(config.batch), heads.str().c_str(),
                static_cast<long long>(config.model.head_size));
    PrintUs(row.ingest, 11);
    PrintUs(row.rot_q, 10);
    PrintUs(row.attn_core, 12);
    PrintUs(row.rot_o, 10);
    PrintUs(row.e2e, 11);
    PrintUs(row.native, 11);
    std::printf(" |");
    PrintRatio(row.speedup, 8);
    PrintDouble(row.gigabytes_per_second, 10, 1);
    PrintDouble(FullModelSavedMib(config, traffic[index]), 12, 1);
    std::printf("\n");

    if (row.e2e.present && row.sum_of_parts_us > 0.0) {
      residuals.push_back(std::fabs(row.e2e.median_us - row.sum_of_parts_us) / row.e2e.median_us);
    }
  }
  if (!any) {
    std::printf("  (no prefill configuration produced a sample)\n");
  }

  std::printf(
      "\n[ascend-bench]   Speedup is V5_Native / TQ_E2E: above 1.000x is TurboQuant ahead. TQ Ingest is\n"
      "[ascend-bench]   reported beside the pipeline and is NOT inside TQ_E2E, matching the accounting\n"
      "[ascend-bench]   this audit was specified with; the native operator's own cache write is timed as\n"
      "[ascend-bench]   pf_v5_ingest and carried in the CSV, so neither side hides a write.\n");
  std::printf(
      "[ascend-bench]   T_rot_o = 0.00 on a folded layer is exact, not missing: W_o' = W_o (I (x) Pi)\n"
      "[ascend-bench]   absorbs the de-rotation offline. Qwen3.5 cannot fold -- attn_output_gate sits\n"
      "[ascend-bench]   between attention and o_proj -- so its column is a measured kernel.\n");
  std::printf(
      "[ascend-bench]   HBM GB/s is the pipeline's compulsory traffic over its measured composite time.\n"
      "[ascend-bench]   KV Saved MB is the whole batch's context over the whole model: fp16 residency minus\n"
      "[ascend-bench]   TurboQuant's for the benchmarked kv head, times KV layers x cached heads per layer\n"
      "[ascend-bench]   (config.json; hybrid linear-attention layers keep no KV, an MLA latent is one slot):\n");
  for (const ModelSpec& model : Models()) {
    std::printf("[ascend-bench]     %-17s %lld KV layers x %lld kv heads\n", model.label,
                static_cast<long long>(model.kv_layers), static_cast<long long>(model.model_kv_heads));
  }
  if (!residuals.empty()) {
    std::printf(
        "[ascend-bench]   composite vs sum-of-parts: median gap %.1f%% over %zu rows. That gap is the\n"
        "[ascend-bench]   launch overhead between stages, which the per-component columns cannot show.\n",
        100.0 * MedianResidual(residuals), residuals.size());
  }
  std::printf(
      "[ascend-bench]   NOT TIMED: the fp32 -> fp16 narrowing between npu_turboquant_rotate_q and FIA.\n"
      "[ascend-bench]   See WHAT IS STILL NOT PRICED in this file's header; its modelled byte cost is in\n"
      "[ascend-bench]   the CSV as untimed_cast_bytes.\n");
  std::fflush(stdout);
}

void PrintTableB(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator, int64_t aiv_num) {
  std::printf("\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] TABLE B -- DECODE LATENCY BREAKDOWN\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] device time from ACL events, median us per decode step; native operator: %s\n",
              fia_operator.c_str());
  std::printf(
      "[ascend-bench] one decode token per sequence; B is the batch of sequences; split policy %s\n"
      "[ascend-bench] (ASCEND_BENCH_TQ_AUDIT_SPLIT); rotation modes %s (%s)\n\n",
      SplitPolicyLabel(SplitPolicy()), RotationModesLabel().c_str(), kRotationModeEnv);

  char header[512];
  const int header_width =
      std::snprintf(header, sizeof(header), "  %-17s %9s %3s %-11s %-19s | %10s %14s %8s %10s %11s %11s | %8s %10s",
                    "Model", "Context", "B", "Path", "Cfg [K/Tsk/Blk/Red]", "T_rot_q", "T_FusedDecode", "Launches",
                    "T_rot_o", "TQ_E2E", "V5_Decode", "Speedup", "Eff GB/s");
  PrintHeaderAndRule(header, header_width);

  bool any = false;
  std::vector<double> residuals;
  std::vector<std::string> deltas;
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    DecodeRow separate;
    DecodeRow fused_q;
    for (const RotationMode mode : kRotationModes) {
      if (!config.decodes_in_mode(mode)) {
        continue;
      }
      const DecodeRow row = ReadDecodeRow(runners, config, model, mode);
      if (mode == RotationMode::kSeparate) {
        separate = row;
      } else {
        fused_q = row;
      }
      if (!row.measured()) {
        continue;
      }
      any = true;
      std::printf("  %-17s %9lld %3lld %-11s %-19s |", config.model.label, static_cast<long long>(config.seq_len),
                  static_cast<long long>(config.batch), config.path_tag(mode).c_str(), model.dispatch.label().c_str());
      PrintUs(row.rot_q, 10);
      PrintUs(row.attn_core, 14);
      std::printf(" %8d", config.decode_launches(mode));
      PrintUs(row.rot_o, 10);
      PrintUs(row.e2e, 11);
      PrintUs(row.native, 11);
      std::printf(" |");
      PrintRatio(row.speedup, 8);
      PrintDouble(row.gigabytes_per_second, 10, 1);
      std::printf("\n");

      if (row.e2e.present && row.sum_of_parts_us > 0.0) {
        residuals.push_back(std::fabs(row.e2e.median_us - row.sum_of_parts_us) / row.e2e.median_us);
      }
    }
    if (separate.e2e.present && fused_q.e2e.present && separate.e2e.median_us > 0.0) {
      char line[256];
      const double delta = fused_q.e2e.median_us - separate.e2e.median_us;
      std::snprintf(line, sizeof(line), "  %-17s %9lld %3lld | TQ_E2E %+9.2f us (%+6.1f%%)", config.model.label,
                    static_cast<long long>(config.seq_len), static_cast<long long>(config.batch), delta,
                    100.0 * delta / separate.e2e.median_us);
      std::string text(line);
      if (separate.attn_core.present && fused_q.attn_core.present && separate.rot_q.present) {
        std::snprintf(line, sizeof(line), ", T_FusedDecode %+9.2f us against a %.2f us rotate_q launch",
                      fused_q.attn_core.median_us - separate.attn_core.median_us, separate.rot_q.median_us);
        text += line;
      }
      deltas.push_back(text);
    }
  }
  if (!any) {
    std::printf("  (no decode configuration produced a sample)\n");
  }

  const int64_t mix_blocks = std::max<int64_t>(1, aiv_num / tqh::kVectorSubcoresPerBlock);
  std::printf(
      "\n[ascend-bench]   Speedup is V5_Decode / TQ_E2E: above 1.000x is TurboQuant ahead. TQ_E2E is one\n"
      "[ascend-bench]   composite ACL-event region over the step's launches; T_FusedDecode is the\n"
      "[ascend-bench]   attention core, ONE launch on either path. Launches counts the host dispatches\n"
      "[ascend-bench]   of one decode step: rotate-q (-Sep only), the core, and rotate-o if unfolded.\n");
  std::printf(
      "[ascend-bench]   Path -Sep launches rotate_q ahead of the decode and, unfolded, rotate_q over its\n"
      "[ascend-bench]   output. Path -FusedQ hands the decode the raw fp16 query, which it rotates in the\n"
      "[ascend-bench]   same launch ahead of its split tasks, and un-rotates the output before its fp16\n"
      "[ascend-bench]   write (PRE_ROTATED=false, kUnrotated, Cube kv4fp8 only): T_rot_q and T_rot_o are\n"
      "[ascend-bench]   0.00 in-launch and T_FusedDecode includes both. Both share the same Cfg and V5.\n");
  std::printf(
      "[ascend-bench]   Cfg [K/Tsk/Blk/Red]: K context splits per sequence; Tsk tasks the launch\n"
      "[ascend-bench]   schedules (Cube B x H_KV x K x head chunks, AIV B x H_Q x K); Blk the blocks it\n"
      "[ascend-bench]   spans, of %lld MIX blocks on the Cube path and %lld vector cores on the AIV path;\n"
      "[ascend-bench]   Red ON when the launch reduces split partials after a SyncAll (NeedsReduction).\n",
      static_cast<long long>(mix_blocks), static_cast<long long>(aiv_num));
  std::printf(
      "[ascend-bench]   Every model takes the Cube decode, whatever its GQA group or head size;\n"
      "[ascend-bench]   ASCEND_BENCH_TQ_AUDIT_PATH=aiv forces the vector-only path for an A/B.\n");
  std::printf(
      "[ascend-bench]   Eff GB/s is the step's compulsory traffic over its measured composite time.\n"
      "[ascend-bench]   KV compression (fp16 over TurboQuant residency) is in the decode CSV.\n");
  if (!residuals.empty()) {
    std::printf("[ascend-bench]   composite vs sum-of-parts: median gap %.1f%% over %zu rows.\n",
                100.0 * MedianResidual(residuals), residuals.size());
  }
  if (!deltas.empty()) {
    std::printf(
        "\n[ascend-bench]   in-launch basis change (-FusedQ) against separate rotate_q launches (-Sep), "
        "negative is -FusedQ faster:\n");
    for (const std::string& line : deltas) {
      std::printf("%s\n", line.c_str());
    }
  }
  std::fflush(stdout);
}

void WritePrefillCsv(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                     const std::vector<Traffic>& traffic, const std::string& path, const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV='%s' for writing\n", path.c_str());
    return;
  }
  csv << "model,regime,seq_len,chunk_tokens,batch,num_heads,num_kv_heads,head_dim,path,native_operator,"
      << "tq_ingest_us,t_rot_q_us,t_attn_core_us,t_rot_o_us,rot_o_is_structural_zero,"
      << "tq_e2e_us,tq_e2e_sum_of_parts_us,tq_e2e_with_ingest_us,"
      << "v5_native_us,v5_native_ingest_us,net_speedup,net_speedup_with_ingest,"
      << "hbm_gbps,attn_core_gbps,v5_gbps,attn_core_tflops,v5_tflops,"
      << "fp16_kv_bytes,tq_kv_bytes,kv_saved_mib,compression_ratio,"
      << "e2e_bytes,attention_flops,untimed_cast_bytes,warmup,iterations,timing_mode\n";

  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    const PrefillRow row = ReadPrefillRow(runners, config, model);

    csv << config.model.key << ',' << config.regime << ',' << config.seq_len << ',' << config.chunk << ','
        << config.batch << ',' << config.model.num_heads << ',' << config.model.num_kv_heads << ','
        << config.model.head_size << ',' << PathLabel(config.path) << ',' << fia_operator;
    WriteCell(csv, row.ingest);
    WriteCell(csv, row.rot_q);
    WriteCell(csv, row.attn_core);
    WriteCell(csv, row.rot_o);
    csv << ',' << (row.rot_o.structural_zero ? 1 : 0);
    WriteCell(csv, row.e2e);
    csv << ',';
    if (row.sum_of_parts_us > 0.0) {
      csv << row.sum_of_parts_us;
    }
    csv << ',';
    if (row.e2e.present && row.ingest.present) {
      csv << row.e2e.median_us + row.ingest.median_us;
    }
    WriteCell(csv, row.native);
    WriteCell(csv, row.native_ingest);
    csv << ',';
    if (row.speedup > 0.0) {
      csv << row.speedup;
    }
    csv << ',';
    if (row.e2e.present && row.ingest.present && row.native.present && row.native_ingest.present) {
      const double tq = row.e2e.median_us + row.ingest.median_us;
      if (tq > 0.0) {
        csv << (row.native.median_us + row.native_ingest.median_us) / tq;
      }
    }
    csv << ',';
    if (row.gigabytes_per_second > 0.0) {
      csv << row.gigabytes_per_second;
    }
    csv << ',';
    if (row.attn_core.present) {
      csv << row.attn_core.gigabytes_per_second;
    }
    csv << ',';
    if (row.native.present) {
      csv << row.native.gigabytes_per_second;
    }
    csv << ',';
    if (row.attn_core.present) {
      csv << row.attn_core.tflops;
    }
    csv << ',';
    if (row.native.present) {
      csv << row.native.tflops;
    }
    csv << ',' << model.fp16_kv_bytes << ',' << model.tq_kv_bytes << ',' << model.saved_mib() << ','
        << model.compression_ratio() << ',' << model.pf_e2e << ',' << model.pf_flops << ',' << model.pf_untimed_cast
        << ',' << config.budget.warmup << ',' << config.budget.iterations << ',' << row.e2e.mode << '\n';
  }
  std::printf("[ascend-bench] Table A written to %s\n", path.c_str());
}

namespace {

// The timed columns of one decode CSV row, from t_rot_q_us on.
void WriteDecodeCsvTimings(std::ofstream& csv, const Config& config, const Traffic& model, const DecodeRow& row,
                           RotationMode mode) {
  WriteCell(csv, row.rot_q);
  WriteCell(csv, row.attn_core);
  csv << ',' << config.decode_launches(mode);
  WriteCell(csv, row.rot_o);
  csv << ',' << (row.rot_o.structural_zero ? 1 : 0);
  WriteCell(csv, row.e2e);
  csv << ',';
  if (row.sum_of_parts_us > 0.0) {
    csv << row.sum_of_parts_us;
  }
  WriteCell(csv, row.native);
  csv << ',';
  if (row.speedup > 0.0) {
    csv << row.speedup;
  }
  csv << ',';
  if (row.gigabytes_per_second > 0.0) {
    csv << row.gigabytes_per_second;
  }
  csv << ',';
  if (row.attn_core.present) {
    csv << row.attn_core.gigabytes_per_second;
  }
  csv << ',';
  if (row.native.present) {
    csv << row.native.gigabytes_per_second;
  }
  csv << ',';
  if (row.attn_core.present) {
    csv << row.attn_core.tflops;
  }
  csv << ',';
  if (row.native.present) {
    csv << row.native.tflops;
  }
  csv << ',' << model.fp16_kv_bytes << ',' << model.tq_kv_bytes << ',' << model.saved_mib() << ','
      << model.compression_ratio() << ',' << (mode == RotationMode::kSeparate ? model.dec_e2e : model.dec_fused_q_e2e)
      << ',' << model.dec_flops << ',';
  if (row.e2e.present) {
    csv << row.e2e.p95_us;
  }
  csv << ',';
  if (row.native.present) {
    csv << row.native.p95_us;
  }
  csv << ',' << config.budget.warmup << ',' << config.budget.iterations << ',' << row.e2e.mode << ','
      << SplitPolicyLabel(SplitPolicy()) << '\n';
}

}  // namespace

void WriteDecodeCsv(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                    const std::vector<Traffic>& traffic, const std::string& path, const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_AUDIT_DECODE_CSV='%s' for writing\n", path.c_str());
    return;
  }
  csv << "model,regime,seq_len,batch,num_heads,num_kv_heads,head_dim,path,rotation_mode,"
      << "split_k,total_tasks,active_blocks,device_blocks,needs_reduction,native_operator,"
      << "t_rot_q_us,t_fused_decode_us,decode_launches,t_rot_o_us,rot_o_is_structural_zero,"
      << "tq_e2e_us,tq_e2e_sum_of_parts_us,v5_decode_us,net_speedup,"
      << "effective_gbps,attn_core_gbps,v5_gbps,attn_core_tflops,v5_tflops,"
      << "fp16_kv_bytes,tq_kv_bytes,kv_saved_mib,compression_ratio,"
      << "e2e_bytes,attention_flops,tq_e2e_p95_us,v5_p95_us,warmup,iterations,timing_mode,split_policy\n";

  // One row per (configuration, rotation mode) the sweep decodes in.
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    for (const RotationMode mode : kRotationModes) {
      if (!config.decodes_in_mode(mode)) {
        continue;
      }
      const DecodeRow row = ReadDecodeRow(runners, config, model, mode);
      const DecodeDispatch& dispatch = model.dispatch;

      csv << config.model.key << ',' << config.regime << ',' << config.seq_len << ',' << config.batch << ','
          << config.model.num_heads << ',' << config.model.num_kv_heads << ',' << config.model.head_size << ','
          << PathLabel(config.path) << ',' << RotationModeKey(mode) << ',' << dispatch.split_k << ','
          << dispatch.total_tasks << ',' << dispatch.active_blocks << ',' << dispatch.device_blocks << ','
          << (dispatch.needs_reduction ? 1 : 0) << ',' << fia_operator;
      WriteDecodeCsvTimings(csv, config, model, row, mode);
    }
  }
  std::printf("[ascend-bench] Table B written to %s\n", path.c_str());
}

void PrintBanner(const std::vector<Config>& sweep, int64_t aiv_num, bool aiv_queried) {
  std::printf("[ascend-bench] TurboQuant end-to-end audit against the native CANN V5 attention operator\n");
  std::printf("[ascend-bench]   vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");
  std::printf("[ascend-bench]\n");
  std::printf(
      "[ascend-bench]   THE NATIVE OPERATOR IS aclnnFusedInferAttentionScoreV5, in both roles.\n"
      "[ascend-bench]   aclnnPromptFlashAttentionV5 and aclnnFusionIncrementalAttention DO NOT EXIST in\n"
      "[ascend-bench]   this toolkit -- the PFA family stops at V3 and the IFA family at V4, and the\n"
      "[ascend-bench]   second name appears in no header at all. FIA V5 is the unified interface that\n"
      "[ascend-bench]   replaces both on an Ascend950 (V1..V4 of it return 361001), and it covers the\n"
      "[ascend-bench]   prompt role and the incremental role through its argument list. V2 is the\n"
      "[ascend-bench]   fallback; every table says which one actually planned.\n");
  std::printf("[ascend-bench]\n");
  std::printf(
      "[ascend-bench]   NO NUMBER BELOW HAS EVER BEEN TAKEN ON SILICON. No Ascend 950PR part has been\n"
      "[ascend-bench]   available to this project; the kernels are verified on the arch35 camodel.\n");
  std::printf("[ascend-bench]\n");
  std::printf("[ascend-bench]   decode: split policy %s (ASCEND_BENCH_TQ_AUDIT_SPLIT), rotation modes %s (%s)\n",
              SplitPolicyLabel(SplitPolicy()), RotationModesLabel().c_str(), kRotationModeEnv);
  if (!RotationModeValid()) {
    std::printf("[ascend-bench]   %s='%s' is not separate, fused_prologue or both; timing both\n", kRotationModeEnv,
                env::String(kRotationModeEnv).c_str());
  }
  if (!UnpackModeValid()) {
    std::printf("[ascend-bench]   %s='%s' is not on, off, both, gather, lloydmax or all; timing the unablated decode\n",
                kUnpackEnv, env::String(kUnpackEnv).c_str());
  }
  if (UnpackAblationEnabled()) {
    std::printf(
        "[ascend-bench]   %s=%s: the %s leg is the decode with the KV unpack compiled out. Its\n"
        "[ascend-bench]   output is all-zero and its checksum means nothing; read it only as the\n"
        "[ascend-bench]   time %s would take without the expand.\n",
        kUnpackEnv, UnpackModeLabel(), kLegDecNoUnpack, kLegDecAttnCore);
  }
  if (UnpackGatherEnabled()) {
    std::printf(
        "[ascend-bench]   %s=%s: the %s leg is the same decode expanding through a 16-entry UB\n"
        "[ascend-bench]   Gather instead of a per-plane Adds. Traffic is identical, so %s minus\n"
        "[ascend-bench]   %s is Option C's instruction cost, and its Eff GB/s against %s's says\n"
        "[ascend-bench]   whether the expand is on the critical path at all. Its table is the\n"
        "[ascend-bench]   uniform grid, so its checksum must equal the unablated decode's.\n",
        kUnpackEnv, UnpackModeLabel(), kLegDecGather, kLegDecGather, kLegDecAttnCore, kLegDecAttnCore);
  }
  if (UnpackLloydMaxEnabled()) {
    std::printf(
        "[ascend-bench]   %s=%s: the %s leg is Cube Lloyd-Max LUT against %s's Cube\n"
        "[ascend-bench]   Uniform -- the same launch and the same traffic, expanding through the\n"
        "[ascend-bench]   e4m3-rounded non-uniform codebook instead of the grid. %s minus\n"
        "[ascend-bench]   %s is what the codebook costs, and next to %s it says how much\n"
        "[ascend-bench]   of that is the Gather rather than the table. This cache is uniformly\n"
        "[ascend-bench]   written, so its checksum must DIFFER and is not a fidelity result; the\n"
        "[ascend-bench]   codebook's SNR is measured on the simulator and host tiers.\n",
        kUnpackEnv, UnpackModeLabel(), kLegDecLloydMax, kLegDecAttnCore, kLegDecLloydMax, kLegDecAttnCore,
        kLegDecGather);
  }
  std::printf("[ascend-bench]\n");

  if (sweep.empty()) {
    std::printf(
        "[ascend-bench]   the sweep is empty: every configuration was filtered out by an\n"
        "[ascend-bench]   ASCEND_BENCH_TQ_AUDIT_* selector.\n");
    std::fflush(stdout);
    return;
  }

  std::printf("[ascend-bench]   %zu configurations:\n", sweep.size());
  std::string previous_model;
  for (const Config& config : sweep) {
    if (previous_model != config.model.key) {
      previous_model = config.model.key;
      std::printf("[ascend-bench]     %-17s D=%-4lld H_Q=%-3lld H_KV=%-2lld  %-6s  %s\n", config.model.label,
                  static_cast<long long>(config.model.head_size), static_cast<long long>(config.model.num_heads),
                  static_cast<long long>(config.model.num_kv_heads), config.model.folds_output ? "folded" : "UNFOLD",
                  PathLabel(config.path));
      std::printf("[ascend-bench]       %s\n", config.model.note);
    }
  }
  std::fflush(stdout);
}

}  // namespace turboquant_audit
}  // namespace test
}  // namespace vllm_ascend
