from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from profiling.process._legacy.predict_llama31_8b_h100_tpot_from_interpolated_kernels import (
    ComponentPrediction,
)
from profiling.process._legacy.predict_llama31_8b_h100_tpot_with_engine_steps import (
    EngineStepComponentModels,
    ForwardPassAttentionModel,
    ForwardPassGemmModel,
    NsysCompiledPrefillModel,
    build_engine_step_prediction_rows,
)
from simulator._legacy.vllm_scheduler_shape import VllmSchedulerConfig


class EngineStepTpotPredictorTest(unittest.TestCase):
    def test_forward_gemm_model_sums_layer_components_and_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gemm.csv"
            path.write_text(
                "M,N,K,dtype_bytes,latency_us\n"
                "4,4096,4096,2,10\n"
                "4,14336,4096,2,20\n"
                "8,4096,4096,2,20\n"
                "8,14336,4096,2,40\n"
            )

            model = ForwardPassGemmModel.from_csv(path, num_layers=2)

        self.assertAlmostEqual(model.predict_ms(4), 0.06)
        self.assertAlmostEqual(model.predict_ms(8), 0.12)

    def test_forward_attention_model_scales_chunk_by_causal_work_fraction(self) -> None:
        model = ForwardPassAttentionModel([(4, 8.0)])

        cost = model.predict_chunks_ms([
            _chunk(request_id=0, scheduled_tokens=1, prefix_tokens=3),
        ])

        self.assertAlmostEqual(cost, 8.0 * 4.0 / 10.0)

    def test_engine_step_prediction_uses_per_request_itl_aggregation(self) -> None:
        rows = build_engine_step_prediction_rows(
            [_prediction_row()],
            benchmark_turns={
                ("swebench-multiturn-synth", 2, 0): {
                    "scheduled_request_count": "2",
                    "successful_request_count": "2",
                    "turn_num_requests": "2",
                    "context_len": "4",
                    "output_tokens": "1",
                    "new_prefill_tokens": "1",
                    "cached_context_tokens": "3",
                    "cache_hit_rate": "0.75",
                },
            },
            component_models=EngineStepComponentModels(
                attention_model=_AttentionModel(),
                gemm_models={},
                small_kernel_models={},
                forward_gemm_model=_ForwardGemm(),
                forward_attention_model=ForwardPassAttentionModel([(4, 8.0)]),
            ),
            component_model="xgboost",
            prefill_mode="synthetic_shared_prefix",
            dense_source="forward_pass",
            config=VllmSchedulerConfig(
                max_num_batched_tokens=4,
                shared_prefix_tokens=3,
                available_gpu_kv_blocks=1024,
            ),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.sim_steps, 3)
        self.assertEqual(row.sim_mixed_decode_prefill_steps, 1)
        self.assertEqual(row.sim_total_prefill_tokens, 5)
        self.assertGreater(row.prefill_attention_ms, 0.0)
        self.assertGreater(row.pred_tpot_ms, 0.0)
        self.assertAlmostEqual(row.pred_tpot_ms, row.pooled_itl_ms)

    def test_decode_residency_wave_policy_scales_decode_components_only(self) -> None:
        rows = build_engine_step_prediction_rows(
            [_prediction_row()],
            benchmark_turns={
                ("swebench-multiturn-synth", 2, 0): {
                    "scheduled_request_count": "2",
                    "successful_request_count": "2",
                    "turn_num_requests": "2",
                    "context_len": "64",
                    "output_tokens": "1",
                    "new_prefill_tokens": "0",
                    "cached_context_tokens": "64",
                    "cache_hit_rate": "1.0",
                },
            },
            component_models=EngineStepComponentModels(
                attention_model=_AttentionModel(),
                gemm_models={},
                small_kernel_models={},
                forward_gemm_model=_ForwardGemm(),
                forward_attention_model=None,
            ),
            component_model="xgboost",
            prefill_mode="benchmark_cache",
            dense_source="forward_pass",
            wave_policy="decode_residency",
            config=VllmSchedulerConfig(
                max_num_batched_tokens=8,
                cache_block_size=16,
                available_gpu_kv_blocks=8,
            ),
        )

        row = rows[0]
        self.assertEqual(row.sim_max_decode_batch, 1)
        self.assertAlmostEqual(row.decode_residency_wave_factor, 2.0)
        self.assertAlmostEqual(row.pred_tpot_ms, 150.0)
        self.assertEqual(row.wave_policy, "decode_residency")

    def test_effective_cache_policy_replaces_overoptimistic_benchmark_cache(self) -> None:
        rows = build_engine_step_prediction_rows(
            [
                dict(
                    _prediction_row(),
                    concurrency="80",
                    turn_index="12",
                    batch_size="80",
                    context_len="6188",
                )
            ],
            benchmark_turns={
                ("swebench-multiturn-synth", 80, 12): {
                    "scheduled_request_count": "80",
                    "successful_request_count": "80",
                    "turn_num_requests": "80",
                    "context_len": "6188",
                    "output_tokens": "1",
                    "new_prefill_tokens": "138",
                    "cached_context_tokens": "5904",
                    "cache_hit_rate": "0.968582",
                },
            },
            component_models=EngineStepComponentModels(
                attention_model=_AttentionModel(),
                gemm_models={},
                small_kernel_models={},
                forward_gemm_model=_ForwardGemm(),
                forward_attention_model=None,
            ),
            component_model="xgboost",
            prefill_mode="benchmark_cache",
            dense_source="forward_pass",
            effective_cache_policy="vllm_residency",
            config=VllmSchedulerConfig(
                max_num_batched_tokens=16_384,
                cache_block_size=16,
                available_gpu_kv_blocks=27_769,
            ),
        )

        row = rows[0]
        self.assertEqual(row.cache_residency_classification, "benchmark_cache_overoptimistic")
        self.assertEqual(row.cached_context_tokens, 5904)
        self.assertEqual(row.new_prefill_tokens, 138)
        self.assertEqual(row.sim_cached_context_tokens, 0)
        self.assertEqual(row.sim_new_prefill_tokens, 6188)
        self.assertEqual(row.engine_effective_cached_tokens, 0)
        self.assertEqual(row.capacity_feasible_cached_tokens, 5552)
        self.assertEqual(row.sim_total_prefill_tokens, 80 * 6188)

    def test_fluid_wave_scheduler_model_emits_envelope_prediction(self) -> None:
        rows = build_engine_step_prediction_rows(
            [dict(_prediction_row(), concurrency="4", batch_size="4")],
            benchmark_turns={
                ("swebench-multiturn-synth", 4, 0): {
                    "scheduled_request_count": "4",
                    "successful_request_count": "4",
                    "turn_num_requests": "4",
                    "context_len": "32",
                    "output_tokens": "1",
                    "new_prefill_tokens": "16",
                    "cached_context_tokens": "16",
                    "cache_hit_rate": "0.5",
                },
            },
            component_models=EngineStepComponentModels(
                attention_model=_AttentionModel(),
                gemm_models={},
                small_kernel_models={},
                forward_gemm_model=_ForwardGemm(),
                forward_attention_model=None,
            ),
            component_model="xgboost",
            prefill_mode="benchmark_cache",
            dense_source="forward_pass",
            scheduler_model="fluid_wave",
            config=VllmSchedulerConfig(
                max_num_batched_tokens=4,
                max_num_seqs=8,
                cache_block_size=16,
                available_gpu_kv_blocks=4,
            ),
        )

        row = rows[0]
        self.assertEqual(row.scheduler_model, "fluid_wave")
        self.assertEqual(row.engine_step_model, "fluid_wave_v1")
        self.assertEqual(row.fluid_active_decode_capacity, 1)
        self.assertAlmostEqual(row.decode_residency_wave_factor, 4.0)
        self.assertEqual(row.fluid_cache_optimistic_tokens, 16)
        self.assertEqual(row.fluid_cache_mid_tokens, 0)
        self.assertEqual(row.fluid_cache_pessimistic_tokens, 0)
        self.assertEqual(row.fluid_scheduler_sensitivity, "kv_scheduler_sensitive")
        self.assertLessEqual(row.fluid_pred_low_tpot_ms, row.pred_tpot_ms)
        self.assertGreaterEqual(row.fluid_pred_high_tpot_ms, row.pred_tpot_ms)


class _AttentionModel:
    name = "attention"

    def predict(self, batch_size: int, context_len: int | None = None) -> ComponentPrediction:
        return ComponentPrediction(batch_size * 10.0 + (context_len or 0), "test")


class _ForwardGemm:
    def predict_ms(self, tokens: int) -> float:
        return float(tokens)


def _prediction_row() -> dict[str, str]:
    return {
        "component_model": "xgboost",
        "profile": "swebench-multiturn-synth",
        "concurrency": "2",
        "turn_index": "0",
        "primary_eval": "true",
        "batch_size": "2",
        "context_len": "4",
        "measured_tpot_ms": "10.0",
        "pred_tpot_ms": "5.0",
        "pct_error": "50.0",
        "signed_error_ms": "-5.0",
        "diagnostic_reason": "",
    }


def _chunk(*, request_id: int, scheduled_tokens: int, prefix_tokens: int):
    from simulator._legacy.vllm_scheduler_shape import VllmPrefillChunk

    return VllmPrefillChunk(
        request_id=request_id,
        scheduled_tokens=scheduled_tokens,
        prefix_tokens=prefix_tokens,
    )


class NsysCompiledPrefillModelTest(unittest.TestCase):
    def test_from_csv_sums_non_attention_components_and_log_linear_interps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "breakdown.csv"
            path.write_text(
                "prefill_tokens,gemm_compiled_ms,elementwise_ms,kv_write_ms,other_ms,fa3_ms,total_ms,reference_vllm_prefill_ms,sanity_ratio,share_gemm,share_elementwise,share_kv_write,share_other\n"
                "64,8.0,1.0,0.1,0.05,0.7,9.85,9.85,1.0,0.87,0.11,0.01,0.005\n"
                "1024,80.0,10.0,1.0,0.5,7.0,98.5,98.5,1.0,0.87,0.11,0.01,0.005\n"
            )

            model = NsysCompiledPrefillModel.from_csv(path)

        # Exact endpoint values (gemm + elementwise + kv_write + other, FA3 excluded).
        self.assertAlmostEqual(model.predict_non_attention_ms(64), 9.15)
        self.assertAlmostEqual(model.predict_non_attention_ms(1024), 91.5)
        # Between endpoints: log-linear interpolation lies strictly between endpoints.
        mid = model.predict_non_attention_ms(256)
        self.assertGreater(mid, 9.15)
        self.assertLess(mid, 91.5)


if __name__ == "__main__":
    unittest.main()
