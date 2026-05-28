from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulator._legacy.roofline_step_predictor import (
    RooflineParams,
    RooflineStepCost,
    RooflineStepPredictor,
)
from simulator._legacy.vllm_scheduler_shape import VllmPrefillChunk, VllmStepShape


def _step(
    *,
    decode_batch: int = 0,
    decoded_ids: tuple[int, ...] = (),
    prefill_chunks: tuple[VllmPrefillChunk, ...] = (),
) -> VllmStepShape:
    prefill_tokens = sum(c.scheduled_tokens for c in prefill_chunks)
    return VllmStepShape(
        step_id=0,
        decode_batch=decode_batch,
        decoded_request_ids=decoded_ids,
        prefill_seqs=len(prefill_chunks),
        prefill_tokens=prefill_tokens,
        prefill_chunks=prefill_chunks,
        waiting_queue=0,
        running_queue=max(1, decode_batch + len(prefill_chunks)),
        free_kv_blocks=1,
        completed_prefill_request_ids=(),
        completed_request_ids=(),
    )


def _minimal_params() -> RooflineParams:
    # Clean round numbers so test arithmetic is easy.  10 GFLOPS model on a
    # 100 GFLOPS / 100 GB/s GPU at 50% utilization both ways.
    return RooflineParams(
        n_params=10_000_000,            # 10M
        peak_flops_per_s=100e9,         # 100 GFLOPS
        peak_bw_bytes_per_s=100e9,      # 100 GB/s
        kv_bytes_per_token=1024.0,      # 1 KB per token
        util_flops=0.5,
        util_bw=0.5,
        bytes_per_param=2.0,
    )


class RooflineParamsTest(unittest.TestCase):
    def test_rejects_nonpositive_flops(self) -> None:
        with self.assertRaises(ValueError):
            RooflineStepPredictor(
                RooflineParams(peak_flops_per_s=0.0)
            )

    def test_rejects_zero_utilization(self) -> None:
        with self.assertRaises(ValueError):
            RooflineStepPredictor(
                RooflineParams(util_flops=0.0)
            )

    def test_json_round_trip(self) -> None:
        params = _minimal_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "params.json"
            params.to_json(path)
            loaded = RooflineParams.from_json(path)
        self.assertEqual(loaded.n_params, params.n_params)
        self.assertAlmostEqual(loaded.util_flops, params.util_flops)

    def test_json_ignores_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "params.json"
            data = {
                **_minimal_params().__dict__,
                "anchor_note": "derived from c=40 t=12 step 1",
            }
            path.write_text(json.dumps(data))
            loaded = RooflineParams.from_json(path)
        self.assertEqual(loaded.n_params, 10_000_000)


class RooflineComputePathTest(unittest.TestCase):
    def test_compute_ms_for_n_tokens(self) -> None:
        # 10M params · 100 tokens · 2 flops/param = 2e9 flops
        # at 100 GFLOPS · 0.5 util = 50 GFLOPS → 40 ms
        pred = RooflineStepPredictor(_minimal_params())
        self.assertAlmostEqual(pred.compute_ms(100), 40.0, places=4)

    def test_compute_ms_scales_linearly_in_tokens(self) -> None:
        pred = RooflineStepPredictor(_minimal_params())
        self.assertAlmostEqual(
            pred.compute_ms(1000) / pred.compute_ms(100), 10.0, places=4
        )

    def test_compute_ms_zero_tokens_is_zero(self) -> None:
        pred = RooflineStepPredictor(_minimal_params())
        self.assertEqual(pred.compute_ms(0), 0.0)


class RooflineBandwidthPathTest(unittest.TestCase):
    def test_decode_only_bandwidth(self) -> None:
        # B=2 decoded requests with T=1000 ctx each.
        # Bytes = weights(2·10M=20MB) + KV_read(2·1000·1KB=2MB) + KV_write(2·1KB)
        #       = 22.002 MB
        # at 100 GB/s · 0.5 util = 50 GB/s → 22.002e6 / 50e9 = 0.44 ms
        pred = RooflineStepPredictor(_minimal_params())
        step = _step(decode_batch=2, decoded_ids=(0, 1))
        ms = pred.bandwidth_ms(step, context_lens={0: 1000, 1: 1000})
        self.assertAlmostEqual(ms, (20e6 + 2 * 1000 * 1024 + 2 * 1024) / 50e9 * 1e3, places=4)

    def test_prefill_only_bandwidth(self) -> None:
        # Full prefill (P=0) U=64 tokens.
        # Bytes = weights(20MB) + KV_write(64·1KB)
        pred = RooflineStepPredictor(_minimal_params())
        step = _step(
            prefill_chunks=(VllmPrefillChunk(0, 64, 0),),
        )
        ms = pred.bandwidth_ms(step, context_lens={})
        expected_bytes = 20e6 + 64 * 1024
        self.assertAlmostEqual(ms, expected_bytes / 50e9 * 1e3, places=4)

    def test_cached_prefill_bandwidth_includes_prefix_read(self) -> None:
        # Chunked prefill U=64 P=8192.  Bytes:
        # weights + KV_read(P=8192·1KB) + KV_write(U=64·1KB)
        pred = RooflineStepPredictor(_minimal_params())
        step = _step(prefill_chunks=(VllmPrefillChunk(0, 64, 8192),))
        ms = pred.bandwidth_ms(step, context_lens={})
        expected = 20e6 + 8192 * 1024 + 64 * 1024
        self.assertAlmostEqual(ms, expected / 50e9 * 1e3, places=4)


class RooflineStepDispatchTest(unittest.TestCase):
    def test_empty_step_returns_zero(self) -> None:
        pred = RooflineStepPredictor(_minimal_params())
        cost = pred.step_ms(_step(), context_lens={})
        self.assertEqual(cost.total_ms, 0.0)
        self.assertEqual(cost.classification, "empty")

    def test_bandwidth_bound_step_returns_bandwidth_ms(self) -> None:
        # Decode at B=2 T=10000: KV_read = 2·10000·1KB = 20MB.
        # Compute: 2 tokens · 2·10M flops/token = 40MFLOPs → tiny.
        # Bandwidth dominates.
        pred = RooflineStepPredictor(_minimal_params())
        step = _step(decode_batch=2, decoded_ids=(0, 1))
        cost = pred.step_ms(step, context_lens={0: 10000, 1: 10000})
        self.assertEqual(cost.classification, "bandwidth_bound")
        self.assertGreater(cost.total_ms, cost.compute_ms)

    def test_compute_bound_step_returns_compute_ms(self) -> None:
        # Big prefill chunk with P=0, U=100000 tokens.
        # Compute: 100000 · 2 · 10M = 2e12 flops → 40 seconds.
        # Bandwidth: weights + 100000·1KB write ≈ 120MB → 2.4 ms.
        pred = RooflineStepPredictor(_minimal_params())
        step = _step(prefill_chunks=(VllmPrefillChunk(0, 100000, 0),))
        cost = pred.step_ms(step, context_lens={})
        self.assertEqual(cost.classification, "compute_bound")
        self.assertGreater(cost.total_ms, cost.bandwidth_ms)


class RooflineAgainstH100AnchorsTest(unittest.TestCase):
    """Sanity-check the H100/Llama defaults against the plan's anchors."""

    def _h100_params(self) -> RooflineParams:
        return RooflineParams()  # defaults are H100 + Llama-3.1-8B

    def test_compute_anchor_c40_t12_step1_full_prefill(self) -> None:
        # Plan anchor: c=40 t=12 step 1, pure prefill at N=7189, real submit=178ms.
        # With util_flops=0.65 default, predicted should match within 10%.
        pred = RooflineStepPredictor(self._h100_params())
        predicted = pred.compute_ms(7189)
        # 2 · 8.03e9 · 7189 / 989e12 / 0.65 = 179.6 ms.  Within ±10% of 178 ms.
        self.assertLess(abs(predicted - 178.0) / 178.0, 0.10)

    def test_bandwidth_anchor_c40_t12_decode_steady_state(self) -> None:
        # Plan anchor: c=40 t=12 steps 17-22, B=40 T~7200, real wall ≈ 17 ms.
        # With util_bw=0.65 default.
        pred = RooflineStepPredictor(self._h100_params())
        step = _step(decode_batch=40, decoded_ids=tuple(range(40)))
        ms = pred.bandwidth_ms(step, context_lens={i: 7200 for i in range(40)})
        # Predicted within ±20% of 17 ms.
        self.assertLess(abs(ms - 17.0) / 17.0, 0.20)


if __name__ == "__main__":
    unittest.main()
