from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from profiling.process.predictors.predict_llama31_8b_h100_tpot_from_interpolated_kernels import (
    BenchmarkTarget,
    KernelSample,
    LogLinearKernelModel,
    XGBoostKernelModel,
    build_component_models,
    build_prediction_rows,
    load_gemm_components,
    load_small_kernel_components,
)


class InterpolatedKernelTpotPredictorTest(unittest.TestCase):
    def test_log_linear_model_interpolates_and_extrapolates_by_batch(self) -> None:
        model = LogLinearKernelModel([
            KernelSample(batch_size=1, context_len=None, value_ms=1.0),
            KernelSample(batch_size=4, context_len=None, value_ms=3.0),
        ])

        exact = model.predict(1)
        interpolated = model.predict(2)
        extrapolated = model.predict(8)

        self.assertEqual(exact.status, "exact")
        self.assertAlmostEqual(exact.value_ms, 1.0)
        self.assertEqual(interpolated.status, "log_linear_interpolated")
        self.assertAlmostEqual(interpolated.value_ms, 2.0)
        self.assertEqual(extrapolated.status, "log_linear_extrapolated")
        self.assertAlmostEqual(extrapolated.value_ms, 4.0)

    def test_load_gemm_components_keeps_projection_families_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gemm.csv"
            path.write_text(
                "batch_size,op_name,calls_per_decode_step,ncu_gpu_time_ms_sum\n"
                "1,qkv_fused,32,0.01\n"
                "1,o_proj,32,0.02\n"
                "2,qkv_fused,32,0.03\n"
            )

            components = load_gemm_components(path)

        self.assertEqual(sorted(components), ["o_proj", "qkv_fused"])
        self.assertAlmostEqual(components["qkv_fused"][0].value_ms, 0.32)
        self.assertAlmostEqual(components["o_proj"][0].value_ms, 0.64)

    def test_small_kernel_loader_skips_diagnostic_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.csv"
            path.write_text(
                "batch_size,context_len,kernel_name,source_status,"
                "calls_per_decode_step,ncu_gpu_time_ms_sum\n"
                "1,512,rms_norm,source_of_truth,64,0.004\n"
                "1,512,silu_and_mul,diagnostic,32,0.010\n"
            )

            source_only = load_small_kernel_components(path)
            all_rows = load_small_kernel_components(path, include_diagnostic=True)

        self.assertEqual(sorted(source_only), ["rms_norm"])
        self.assertEqual(sorted(all_rows), ["rms_norm", "silu_and_mul"])

    def test_measured_tpot_is_validation_only_not_component_feature(self) -> None:
        attention = build_component_models(
            {"attention": [KernelSample(1, 512, 1.0), KernelSample(2, 512, 2.0)]},
            model_name="log_interp",
        )
        gemm = build_component_models(
            {"qkv_fused": [KernelSample(1, None, 3.0), KernelSample(2, None, 4.0)]},
            model_name="log_interp",
        )
        small = build_component_models(
            {"rms_norm": [KernelSample(1, 512, 0.5), KernelSample(2, 512, 0.6)]},
            model_name="log_interp",
        )
        low_target = BenchmarkTarget(
            batch_size=1,
            context_len=512,
            measured_tpot_ms=10.0,
            profile="p",
            concurrency=1,
            turn_index=0,
            primary_eval=True,
            diagnostic_reason="",
            row={},
        )
        high_target = BenchmarkTarget(
            batch_size=1,
            context_len=512,
            measured_tpot_ms=20.0,
            profile="p",
            concurrency=1,
            turn_index=0,
            primary_eval=True,
            diagnostic_reason="",
            row={},
        )

        rows = build_prediction_rows(
            [low_target, high_target],
            model_name="log_interp",
            attention_models=attention,
            gemm_models=gemm,
            small_kernel_models=small,
        )

        self.assertAlmostEqual(rows[0].pred_tpot_ms, 4.5)
        self.assertAlmostEqual(rows[1].pred_tpot_ms, 4.5)
        self.assertNotEqual(rows[0].pct_error, rows[1].pct_error)

    @unittest.skipUnless(
        importlib.util.find_spec("xgboost") is not None,
        "xgboost is not installed",
    )
    def test_xgboost_component_model_exact_rows_and_positive_interpolation(self) -> None:
        model = XGBoostKernelModel([
            KernelSample(1, 512, 1.0),
            KernelSample(2, 512, 2.0),
            KernelSample(4, 1024, 4.0),
            KernelSample(8, 2048, 8.0),
        ])

        exact = model.predict(1, 512)
        interp = model.predict(3, 768)

        self.assertEqual(exact.status, "exact")
        self.assertAlmostEqual(exact.value_ms, 1.0)
        self.assertEqual(interp.status, "xgboost_interpolated")
        self.assertGreater(interp.value_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
