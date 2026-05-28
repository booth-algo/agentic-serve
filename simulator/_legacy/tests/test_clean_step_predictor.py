from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simulator._legacy.clean_step_predictor import (
    CleanStepPredictor,
    _bilinear_log2d,
    _log_linear_1d,
)
from simulator._legacy.vllm_scheduler_shape import VllmPrefillChunk, VllmStepShape


class LogLinearInterpTest(unittest.TestCase):
    def test_returns_endpoint_at_or_below_min(self) -> None:
        samples = [(64, 6.65), (1024, 23.55), (8192, 62.36)]
        self.assertEqual(_log_linear_1d(samples, 1), 6.65)
        self.assertEqual(_log_linear_1d(samples, 64), 6.65)

    def test_returns_endpoint_at_or_above_max(self) -> None:
        samples = [(64, 6.65), (1024, 23.55), (8192, 62.36)]
        self.assertEqual(_log_linear_1d(samples, 8192), 62.36)
        self.assertEqual(_log_linear_1d(samples, 100_000), 62.36)

    def test_interpolates_between_in_log_space(self) -> None:
        samples = [(64, 10.0), (1024, 20.0)]
        mid = _log_linear_1d(samples, 256)
        # log2(64)=6, log2(1024)=10, log2(256)=8 → halfway → 15.0
        self.assertAlmostEqual(mid, 15.0, places=4)


class BilinearLog2dTest(unittest.TestCase):
    def test_exact_lattice_lookup(self) -> None:
        lattice = {(64, 512): 12.0, (128, 512): 13.0, (64, 1024): 14.0, (128, 1024): 15.0}
        self.assertAlmostEqual(_bilinear_log2d(lattice, 64, 512), 12.0)
        self.assertAlmostEqual(_bilinear_log2d(lattice, 128, 1024), 15.0)

    def test_interpolates_within_lattice(self) -> None:
        # 4-corner square; interpolation at center yields the average.
        lattice = {(64, 512): 12.0, (128, 512): 14.0, (64, 1024): 16.0, (128, 1024): 18.0}
        mid = _bilinear_log2d(lattice, 90, 720)  # roughly center in log2 space
        self.assertGreater(mid, 12.0)
        self.assertLess(mid, 18.0)


class CleanStepPredictorFromCsvsTest(unittest.TestCase):
    def _write_csvs(self, tmp: Path) -> tuple[Path, Path, Path]:
        full = tmp / "full.csv"
        full.write_text(
            "gpu,prefill_tokens,prefill_ms,decode_ref_ms,T32_ms,T1_ms,runs\n"
            "H100,64,6.65,6.4,0,0,1\n"
            "H100,1024,23.55,6.3,0,0,1\n"
            "H100,8192,62.36,8.2,0,0,1\n"
        )
        cached = tmp / "cached.csv"
        cached.write_text(
            "U,P,prefill_ms,decode_ref_ms,t1_cached_ms,scheduled_tokens,cache_hit\n"
            "64,512,12.37,6.5,18.87,16,True\n"
            "128,512,12.55,6.5,19.06,16,True\n"
            "64,8192,24.27,6.6,30.83,64,True\n"
            "128,8192,24.00,6.6,30.61,64,True\n"
        )
        decode = tmp / "decode.csv"
        decode.write_text(
            "gpu,batch_size,context_len,observed_context_len,total_kv_tokens,decode_step_ms,generated_tokens,decode_intervals,gpu_ms\n"
            "H100,1,512,513,640,11.83,128,127,1502\n"
            "H100,1,1024,1025,1152,6.53,128,127,829\n"
            "H100,40,1024,1025,40960,8.0,128,127,1000\n"
            "H100,40,8192,8193,327680,30.0,128,127,3800\n"
        )
        return full, cached, decode

    def test_loads_3_csvs_and_predicts_at_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            full, cached, decode = self._write_csvs(Path(tmpdir))
            pred = CleanStepPredictor.from_csvs(
                full_prefill_path=full,
                cached_prefill_path=cached,
                decode_path=decode,
            )
        self.assertAlmostEqual(pred.full_prefill_ms(64), 6.65, places=4)
        self.assertAlmostEqual(pred.cached_prefill_ms(64, 512), 12.37, places=4)
        self.assertAlmostEqual(pred.decode_ms(1, 1024), 6.53, places=4)

    def test_cached_prefill_with_zero_prefix_falls_back_to_full_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            full, cached, decode = self._write_csvs(Path(tmpdir))
            pred = CleanStepPredictor.from_csvs(
                full_prefill_path=full,
                cached_prefill_path=cached,
                decode_path=decode,
            )
        # P=0 → full prefill at U=1024
        self.assertAlmostEqual(pred.cached_prefill_ms(1024, 0), 23.55, places=4)


class CleanStepPredictorDispatchTest(unittest.TestCase):
    def _make_predictor(self) -> CleanStepPredictor:
        return CleanStepPredictor(
            full_prefill_samples=[(64, 5.0), (1024, 20.0), (8192, 60.0)],
            cached_prefill_lattice={
                (64, 512): 10.0,
                (128, 512): 11.0,
                (64, 8192): 25.0,
                (128, 8192): 26.0,
            },
            decode_lattice={
                (1, 1024): 6.5,
                (1, 8192): 8.0,
                (40, 1024): 8.0,
                (40, 8192): 30.0,
            },
        )

    def _step(
        self,
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
            running_queue=decode_batch + len(prefill_chunks),
            free_kv_blocks=1,
            completed_prefill_request_ids=(),
            completed_request_ids=(),
        )

    def test_decode_only_step_uses_decode_predictor(self) -> None:
        pred = self._make_predictor()
        step = self._step(decode_batch=40, decoded_ids=tuple(range(40)))
        cost = pred.step_ms(step, context_lens={i: 1024 for i in range(40)})
        self.assertEqual(cost.classification, "decode_only")
        self.assertAlmostEqual(cost.total_ms, 8.0, places=4)
        self.assertEqual(cost.prefill_ms, 0.0)

    def test_prefill_only_step_sums_chunks(self) -> None:
        pred = self._make_predictor()
        step = self._step(
            prefill_chunks=(
                VllmPrefillChunk(request_id=0, scheduled_tokens=64, prefix_tokens=512),
                VllmPrefillChunk(request_id=1, scheduled_tokens=64, prefix_tokens=512),
            )
        )
        cost = pred.step_ms(step, context_lens={})
        self.assertEqual(cost.classification, "prefill_only")
        self.assertAlmostEqual(cost.total_ms, 20.0, places=4)  # 2 × 10.0
        self.assertEqual(cost.decode_ms, 0.0)

    def test_mixed_step_takes_max_of_prefill_and_decode(self) -> None:
        pred = self._make_predictor()
        # Decode_ms(40, 8192)=30.0; prefill_chunks sum to 10.0 (one (64, 512) chunk).
        # max(10, 30) = 30 → decode dominates.
        step = self._step(
            decode_batch=40,
            decoded_ids=tuple(range(40)),
            prefill_chunks=(
                VllmPrefillChunk(request_id=99, scheduled_tokens=64, prefix_tokens=512),
            ),
        )
        cost = pred.step_ms(step, context_lens={i: 8192 for i in range(40)})
        self.assertEqual(cost.classification, "mixed")
        self.assertAlmostEqual(cost.total_ms, 30.0, places=4)
        self.assertAlmostEqual(cost.decode_ms, 30.0, places=4)
        self.assertAlmostEqual(cost.prefill_ms, 10.0, places=4)

    def test_mixed_step_prefill_dominates(self) -> None:
        pred = self._make_predictor()
        # Decode (1, 1024) = 6.5; prefill 2 chunks (64, 8192) = 50 → prefill wins
        step = self._step(
            decode_batch=1,
            decoded_ids=(0,),
            prefill_chunks=(
                VllmPrefillChunk(request_id=1, scheduled_tokens=64, prefix_tokens=8192),
                VllmPrefillChunk(request_id=2, scheduled_tokens=64, prefix_tokens=8192),
            ),
        )
        cost = pred.step_ms(step, context_lens={0: 1024})
        self.assertEqual(cost.classification, "mixed")
        self.assertAlmostEqual(cost.total_ms, 50.0, places=4)
        self.assertAlmostEqual(cost.prefill_ms, 50.0, places=4)

    def test_empty_step_returns_zero(self) -> None:
        pred = self._make_predictor()
        step = self._step()
        cost = pred.step_ms(step, context_lens={})
        self.assertEqual(cost.classification, "empty")
        self.assertEqual(cost.total_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
