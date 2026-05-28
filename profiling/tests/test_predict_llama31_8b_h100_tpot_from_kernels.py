from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from profiling.process.predictors.predict_llama31_8b_h100_tpot_from_kernels import (
    DecodeTarget,
    build_prediction_rows,
    load_attention_profile,
    load_gemm_summary,
    load_small_kernel_summary,
)


class KernelComposedTpotPredictorTest(unittest.TestCase):
    def test_load_attention_profile_reads_flash_full_model_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flash.csv"
            path.write_text(
                "gpu,batch_size,context_len,flash_full_model_ms_median\n"
                "H100,1,512,0.66\n"
                "A100,1,512,0.88\n"
                "H100,32,8192,11.5\n"
            )

            values = load_attention_profile(path, gpu="H100")

        self.assertEqual(values, {(1, 512): 0.66, (32, 8192): 11.5})

    def test_load_attention_profile_reads_ncu_flash_full_model_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flash_ncu.csv"
            path.write_text(
                "batch_size,context_len,ncu_flash_full_model_ms_sum,"
                "flash_full_model_ms_median\n"
                "1,512,0.57,0.66\n"
            )

            values = load_attention_profile(path, gpu="H100")

        self.assertEqual(values, {(1, 512): 0.57})

    def test_load_gemm_summary_composes_layer_counted_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gemm.csv"
            path.write_text(
                "batch_size,op_name,calls_per_decode_step,ncu_gpu_time_ms_sum\n"
                "1,qkv_fused,32,0.02\n"
                "1,o_proj,32,0.01\n"
                "2,qkv_fused,32,0.03\n"
            )

            values = load_gemm_summary(path)

        self.assertAlmostEqual(values[1], 0.96)
        self.assertAlmostEqual(values[2], 0.96)

    def test_load_gemm_summary_aggregates_all_four_decode_projection_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gemm.csv"
            path.write_text(
                "batch_size,op_name,calls_per_decode_step,ncu_gpu_time_ms_sum\n"
                "8,qkv_fused,32,0.01\n"
                "8,o_proj,32,0.02\n"
                "8,gate_up_fused,32,0.03\n"
                "8,down_proj,32,0.04\n"
            )

            values = load_gemm_summary(path)

        self.assertEqual(set(values), {8})
        self.assertAlmostEqual(values[8], 3.2)

    def test_load_small_kernel_summary_groups_by_batch_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.csv"
            path.write_text(
                "batch_size,context_len,kernel_name,calls_per_decode_step,"
                "ncu_gpu_time_ms_sum\n"
                "1,512,rms_norm,64,0.004\n"
                "1,512,rotary_embedding,32,0.006\n"
                "1,1024,rms_norm,64,0.005\n"
            )

            values = load_small_kernel_summary(path)

        self.assertAlmostEqual(values[(1, 512)], 0.448)
        self.assertAlmostEqual(values[(1, 1024)], 0.32)

    def test_load_small_kernel_summary_skips_diagnostic_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.csv"
            path.write_text(
                "batch_size,context_len,kernel_name,source_status,"
                "calls_per_decode_step,ncu_gpu_time_ms_sum\n"
                "1,512,rms_norm,source_of_truth,64,0.004\n"
                "1,512,silu_and_mul,diagnostic,32,0.010\n"
                "1,512,kv_cache_write,unknown,32,0.020\n"
            )

            source_only = load_small_kernel_summary(path)
            with_diagnostic = load_small_kernel_summary(
                path,
                include_diagnostic=True,
            )

        self.assertAlmostEqual(source_only[(1, 512)], 0.256)
        self.assertAlmostEqual(with_diagnostic[(1, 512)], 1.216)

    def test_measured_tpot_is_validation_only(self) -> None:
        attention = {(1, 512): 1.0}
        gemm = {1: 5.0}
        small = {(1, 512): 0.5}
        low_target = [DecodeTarget(1, 512, 10.0)]
        high_target = [DecodeTarget(1, 512, 20.0)]

        low_row = build_prediction_rows(
            low_target,
            attention_ms_by_key=attention,
            gemm_ms_by_batch=gemm,
            small_ms_by_key=small,
        )[0]
        high_row = build_prediction_rows(
            high_target,
            attention_ms_by_key=attention,
            gemm_ms_by_batch=gemm,
            small_ms_by_key=small,
        )[0]

        self.assertAlmostEqual(low_row.pred_tpot_ms, 6.5)
        self.assertAlmostEqual(high_row.pred_tpot_ms, 6.5)
        self.assertNotEqual(low_row.pct_error, high_row.pct_error)


if __name__ == "__main__":
    unittest.main()
