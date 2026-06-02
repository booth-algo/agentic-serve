from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path


class ExtractorClassifierTest(unittest.TestCase):
    def test_classify_routes_known_kernels_into_buckets(self) -> None:
        from profiling.process._legacy.extract_nsys_prefill_breakdown import (
            classify,
        )

        # short, demangled → expected bucket
        cases = [
            ("device_kernel", "void cutlass::device_kernel<flash::FlashAttnFwdSm90<...>>", "attention"),
            ("nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN", "void nvjet::...", "gemm_linear"),
            ("splitKreduce_kernel", "void splitKreduce_kernel<...>", "gemm_linear"),
            ("reshape_and_cache_flash_kernel", "void vllm::reshape_and_cache_flash_kernel<...>", "kv_write"),
            ("triton_poi_fused_mul_silu_slice_1", "void triton_poi_fused...", "elementwise"),
            ("vectorized_elementwise_kernel", "void at::vectorized_elementwise_kernel...", "elementwise"),
            ("_topk_topp_kernel", "void _topk_topp_kernel...", "sampling"),
            ("vectorized_gather_kernel", "void vectorized_gather_kernel...", "other"),
        ]
        for short, dem, expected in cases:
            with self.subTest(short=short):
                self.assertEqual(classify(short, dem), expected)


class ExtractorEndToEndTest(unittest.TestCase):
    def test_extractor_emits_per_n_breakdown_csv(self) -> None:
        # Build a tiny in-memory sqlite that looks like an nsys capture.
        from profiling.process._legacy.extract_nsys_prefill_breakdown import (
            extract_trace,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / "prefill_N1024.sqlite"
            con = sqlite3.connect(sqlite_path)
            con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            con.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                (start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER)""")
            strings = [
                (1, "nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN"),
                (2, "void nvjet::..."),
                (3, "device_kernel"),
                (4, "void cutlass::device_kernel<flash::FlashAttnFwdSm90<...>>"),
                (5, "triton_poi_fused_mul_silu_slice_1"),
                (6, "void triton_poi_fused_..."),
                (7, "reshape_and_cache_flash_kernel"),
                (8, "void vllm::reshape_and_cache_flash_kernel<...>"),
            ]
            con.executemany("INSERT INTO StringIds VALUES (?, ?)", strings)
            # 1ms GEMM + 0.5ms attention + 0.2ms elementwise + 0.1ms KV write
            con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", [
                (0, 1_000_000, 1, 2),    # 1ms gemm
                (1_000_000, 1_500_000, 3, 4),  # 0.5ms attention
                (1_500_000, 1_700_000, 5, 6),  # 0.2ms elementwise
                (1_700_000, 1_800_000, 7, 8),  # 0.1ms kv_write
            ])
            con.commit()
            con.close()

            trace = extract_trace(sqlite_path)

        self.assertEqual(trace.prefill_tokens, 1024)
        self.assertEqual(trace.gemm_ns, 1_000_000)
        self.assertEqual(trace.attention_ns, 500_000)
        self.assertEqual(trace.elementwise_ns, 200_000)
        self.assertEqual(trace.kv_write_ns, 100_000)


class ExtractorReferenceCsvOutputShape(unittest.TestCase):
    def test_emitted_csv_has_expected_columns_and_sanity_ratio_one(self) -> None:
        from profiling.process._legacy.extract_nsys_prefill_breakdown import (
            main,
        )
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Build minimal fake nsys data: two N values
            nsys_dir = tmp / "nsys"
            nsys_dir.mkdir()
            for n in (64, 1024):
                sqlite_path = nsys_dir / f"prefill_N{n}.sqlite"
                con = sqlite3.connect(sqlite_path)
                con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
                con.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                    (start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER)""")
                con.executemany("INSERT INTO StringIds VALUES (?, ?)", [
                    (1, "nvjet_tst_x"), (2, "void nvjet::..."),
                    (3, "triton_poi_fused_x"), (4, "triton_poi_fused..."),
                    (5, "reshape_and_cache_flash_kernel"), (6, "reshape_and_cache_flash_kernel..."),
                ])
                con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", [
                    (0, 1_000_000, 1, 2),  # gemm
                    (1_000_000, 1_100_000, 3, 4),  # elementwise
                    (1_100_000, 1_120_000, 5, 6),  # kv_write
                ])
                con.commit()
                con.close()
            # Tiny reference CSVs
            ref = tmp / "ref.csv"
            ref.write_text("prefill_tokens,prefill_ms\n64,6.65\n1024,23.55\n")
            fa3 = tmp / "fa3.csv"
            fa3.write_text("prefill_tokens,flash_full_model_ms\n64,0.73\n1024,1.00\n")
            output = tmp / "out.csv"

            argv_backup = sys.argv
            sys.argv = [
                "extract",
                "--nsys-dir", str(nsys_dir),
                "--reference-vllm-prefill", str(ref),
                "--fa3-prefill", str(fa3),
                "--output", str(output),
            ]
            try:
                main()
            finally:
                sys.argv = argv_backup

            with output.open() as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([int(r["prefill_tokens"]) for r in rows], [64, 1024])
        # Sanity ratio = (gemm+elem+kv+other+fa3) / reference_vllm_prefill = exactly 1.0 by construction.
        for r in rows:
            self.assertAlmostEqual(float(r["sanity_ratio"]), 1.0, places=6)
        # All non-attention components are non-negative.
        for r in rows:
            for key in ("gemm_compiled_ms", "elementwise_ms", "kv_write_ms", "other_ms"):
                self.assertGreaterEqual(float(r[key]), 0.0)


if __name__ == "__main__":
    unittest.main()
