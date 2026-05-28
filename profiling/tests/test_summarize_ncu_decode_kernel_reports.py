from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from profiling.process.summarizers.summarize_ncu_decode_kernel_reports import (
    FUSED_BUCKETS,
    NUM_LAYERS,
    load_fused_metadata,
    load_flash_cuda_events,
    parse_args,
    parse_flash_tag,
    parse_gemm_tag,
    summarize_report,
    write_rows,
)


class SummarizeNcuDecodeKernelReportsTest(unittest.TestCase):
    def test_parse_flash_tag_reads_batch_and_context(self) -> None:
        fields = parse_flash_tag(Path("flash_attn_B32_T8192.csv"))

        self.assertEqual(fields["batch_size"], 32)
        self.assertEqual(fields["context_len"], 8192)
        self.assertEqual(fields["bucket"], "attention")
        self.assertEqual(fields["calls_per_decode_step"], NUM_LAYERS)

    def test_parse_gemm_tag_covers_llama31_decode_projection_shapes(self) -> None:
        cases = {
            "gemm_qkv_fused_B8.csv": {
                "op_name": "qkv_fused",
                "m": 8,
                "n": 6144,
                "k": 4096,
                "calls_per_decode_step": 32,
            },
            "gemm_o_proj_B8.csv": {
                "op_name": "o_proj",
                "m": 8,
                "n": 4096,
                "k": 4096,
                "calls_per_decode_step": 32,
            },
            "gemm_gate_up_fused_B8.csv": {
                "op_name": "gate_up_fused",
                "m": 8,
                "n": 28672,
                "k": 4096,
                "calls_per_decode_step": 32,
            },
            "gemm_down_proj_B8.csv": {
                "op_name": "down_proj",
                "m": 8,
                "n": 4096,
                "k": 14336,
                "calls_per_decode_step": 32,
            },
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                fields = parse_gemm_tag(Path(filename))
                self.assertEqual(fields["batch_size"], 8)
                self.assertEqual(fields["op_name"], expected["op_name"])
                self.assertEqual(fields["m"], expected["m"])
                self.assertEqual(fields["n"], expected["n"])
                self.assertEqual(fields["k"], expected["k"])
                self.assertEqual(
                    fields["calls_per_decode_step"],
                    expected["calls_per_decode_step"],
                )

    def test_load_flash_cuda_events_filters_gpu_and_reads_full_model_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flash_events.csv"
            path.write_text(
                "gpu,batch_size,context_len,flash_ms_median,"
                "flash_full_model_ms_median\n"
                "H100,1,512,0.02,0.64\n"
                "A100,1,512,0.03,0.96\n"
            )

            rows = load_flash_cuda_events(path, gpu="H100")

        self.assertEqual(set(rows), {(1, 512)})
        self.assertAlmostEqual(rows[(1, 512)].flash_layer_ms_median, 0.02)
        self.assertAlmostEqual(rows[(1, 512)].flash_full_model_ms_median, 0.64)

    def test_load_flash_cuda_events_rejects_ncu_wrapped_event_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cuda_events_under_ncu" / "flash_attn_B1_T512.csv"
            path.parent.mkdir()
            path.write_text(
                "gpu,batch_size,context_len,flash_ms_median,"
                "flash_full_model_ms_median\n"
                "H100,1,512,1000,32000\n"
            )

            with self.assertRaises(SystemExit):
                load_flash_cuda_events(path, gpu="H100")

    def test_write_flash_summary_includes_ncu_primary_and_cuda_event_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ncu_dir = root / "ncu"
            ncu_dir.mkdir()
            ncu_csv = ncu_dir / "flash_attn_B1_T512.csv"
            ncu_csv.write_text(
                "Kernel Name,gpu__time_duration.sum,dram__bytes_read.sum,"
                "dram__bytes_write.sum\n"
                "flash_kernel_a,20,10,2\n"
                "flash_kernel_a,30,11,3\n"
                "flash_kernel_b,10,4,1\n"
            )
            event_csv = root / "flash_events.csv"
            event_csv.write_text(
                "gpu,batch_size,context_len,flash_ms_median,"
                "flash_full_model_ms_median\n"
                "H100,1,512,0.002,0.064\n"
            )
            output = root / "flash_summary.csv"

            args = parse_args([
                "--kind",
                "flash",
                "--ncu-dir",
                str(ncu_dir),
                "--output",
                str(output),
                "--cuda-event-flash",
                str(event_csv),
            ])
            write_rows(args)

            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["batch_size"], "1")
        self.assertEqual(row["context_len"], "512")
        self.assertEqual(row["kernel_count"], "3")
        self.assertAlmostEqual(float(row["ncu_gpu_time_ms_sum"]), 0.06)
        self.assertAlmostEqual(float(row["ncu_flash_layer_ms_sum"]), 0.06)
        self.assertAlmostEqual(float(row["ncu_flash_full_model_ms_sum"]), 1.92)
        self.assertAlmostEqual(float(row["cuda_event_flash_full_model_ms_median"]), 0.064)
        self.assertAlmostEqual(float(row["ncu_minus_cuda_event_full_model_ms"]), 1.856)
        self.assertIn("flash_kernel_a", row["top_kernel_examples"])
        self.assertEqual(row["ncu_csv"], str(ncu_csv))

    def test_write_flash_summary_parses_alternate_name_and_normalizes_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ncu_dir = root / "ncu"
            ncu_dir.mkdir()
            ncu_csv = ncu_dir / "attention_B2_T1024_probe.csv"
            ncu_csv.write_text(
                "Kernel Name,gpu__time_duration.sum,dram__bytes_read.sum,"
                "dram__bytes_write.sum\n"
                ",ns,Gbyte,byte\n"
                "flash_kernel,2000000,0.002,2048\n"
            )
            output = root / "flash_summary.csv"

            args = parse_args([
                "--kind",
                "flash",
                "--ncu-dir",
                str(ncu_dir),
                "--pattern",
                "attention_*.csv",
                "--output",
                str(output),
            ])
            write_rows(args)

            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["batch_size"], "2")
        self.assertEqual(row["context_len"], "1024")
        self.assertAlmostEqual(float(row["ncu_gpu_time_ms_sum"]), 2.0)
        self.assertAlmostEqual(float(row["ncu_flash_full_model_ms_sum"]), 64.0)
        self.assertAlmostEqual(float(row["ncu_dram_read_mbytes_sum"]), 2.0)
        self.assertAlmostEqual(float(row["ncu_dram_write_mbytes_sum"]), 0.002048)

    def test_summarize_report_normalizes_ncu_memory_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flash_attn_B32_T8192.csv"
            path.write_text(
                "Kernel Name,gpu__time_duration.sum,dram__bytes_read.sum,"
                "dram__bytes_write.sum\n"
                ",us,Gbyte,byte\n"
                "flash_kernel_a,20,1.5,512\n"
                "flash_kernel_b,10,0.001,1024\n"
            )

            summary = summarize_report(path, top_kernels=2)

        self.assertEqual(summary.kernel_count, 2)
        self.assertAlmostEqual(summary.gpu_time_ms_sum, 0.03)
        self.assertAlmostEqual(summary.dram_read_mbytes_sum, 1501.0)
        self.assertAlmostEqual(summary.dram_write_mbytes_sum, 0.001536)

    def test_sampling_topk_is_small_kernel_bucket_not_lm_head(self) -> None:
        self.assertEqual(FUSED_BUCKETS["sampling_topk"], ("sampling_logits", 1))

    def test_load_fused_metadata_marks_vllm_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ncu_csv = root / "ncu" / "fused_rms_norm_B1_T512.csv"
            event_csv = root / "cuda_events_under_ncu" / ncu_csv.name
            ncu_csv.parent.mkdir()
            event_csv.parent.mkdir()
            event_csv.write_text(
                "kernel_name,implementation\n"
                "rms_norm,vllm\n"
            )

            metadata = load_fused_metadata(ncu_csv)

        self.assertEqual(metadata.implementation, "vllm")
        self.assertEqual(metadata.source_status, "source_of_truth")


if __name__ == "__main__":
    unittest.main()
