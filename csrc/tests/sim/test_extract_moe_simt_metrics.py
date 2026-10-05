# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Regression gates for report attribution; no fabricated additive stall metrics."""
import json
import tempfile
import unittest
from pathlib import Path

from extract_moe_simt_metrics import extract


class MetricsTest(unittest.TestCase):
    def test_modeled_clocks_and_missing_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({
                "kernel_info": {"kernel_total_clocks": 8192},
                "pipe_utilization": {"SIMT": 0.8, "MTE3": 0.4},
            }))
            (root / "kernel_launch_0_vf_report.json").write_text(json.dumps({
                "vfs": [{"type": "simd", "id": 0}, {"type": "simt", "id": 1}]
            }))
            got = extract(root, 4096)
            self.assertEqual(len(got["vector_functions"]), 2)
            self.assertEqual(got["simt_vfs"], [{"type": "simt", "id": 1}])
            self.assertEqual(got["modeled_cycles"], 8192)
            self.assertEqual(got["cycles_per_element_end_to_end"], 2)
            self.assertIsNone(got["exclusive_stall_cycles"])
            self.assertIsNone(got["isolated_mode_switch_cycles"])
            self.assertIsNone(got["data_cache_hit_rate"])

    def test_missing_or_ambiguous_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                extract(root, 4096)
            for name in ("a", "b"):
                sub = root / name
                sub.mkdir()
                (sub / "summary.json").write_text("{}")
            with self.assertRaises(ValueError):
                extract(root, 4096)


if __name__ == "__main__":
    unittest.main()
