import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sweep_progress_report as report  # noqa: E402


def job(short="Llama-3.1-8B", model_path="/models/Llama-3.1-8B-Instruct", backend="sglang"):
    return report.Job(
        host="a100",
        model_path=model_path,
        tp=1,
        short=short,
        mode="single",
        backend=backend,
        max_len="2048",
        gpu_mem="0.5",
        concs="1",
        profiles="chat-singleturn-synth",
        extra_env="",
        scope="synthetic_distributional",
        line_no=1,
    )


def state_for(job_obj, age_seconds=3600, gpus="5"):
    return report.JobState(
        job=job_obj,
        status="running",
        gpus=gpus,
        port="8091",
        attempt="0",
        age_seconds=age_seconds,
        max_len_override="",
        run_id="",
    )


def proc(parent_cmd):
    return report.GpuProcess(
        gpu_index="5",
        gpu_uuid="GPU-test",
        pid="123",
        process_name="sglang::scheduler",
        used_memory_mib=1024,
        user="kevin",
        ppid="100",
        pgid="100",
        sid="100",
        stat="Sl",
        age_seconds=3600,
        cmd="sglang::scheduler",
        parent_user="kevin",
        parent_ppid="1",
        parent_pgid="100",
        parent_sid="100",
        parent_stat="Sl",
        parent_age_seconds=3600,
        parent_cmd=parent_cmd,
    )


class SweepProgressReportTests(unittest.TestCase):
    def test_old_assignment_without_matching_process_is_not_live(self):
        stale_state = state_for(job())
        processes = {
            "5": [
                proc(
                    "/data/env/bin/python -m sglang.launch_server "
                    "--model-path /models/Llama-3.1-70B-Instruct --port 8093"
                )
            ]
        }

        self.assertFalse(report.assignment_is_live_or_warming(stale_state, processes))

    def test_old_assignment_with_matching_process_is_live(self):
        live_state = state_for(job())
        processes = {
            "5": [
                proc(
                    "/data/env/bin/python -m sglang.launch_server "
                    "--model-path /models/Llama-3.1-8B-Instruct --port 8091"
                )
            ]
        }

        self.assertTrue(report.assignment_is_live_or_warming(live_state, processes))

    def test_recent_assignment_is_kept_during_warmup(self):
        warming_state = state_for(job(), age_seconds=120)

        self.assertTrue(report.assignment_is_live_or_warming(warming_state, {}))

    def test_parse_host_snapshot_includes_benchmark_lease_metadata(self):
        stdout = """__WHOAMI__
kevin
__GPU__
0, GPU-test, NVIDIA A100, 1024, 40960, 0
__PROC__
GPU-test, 123, VLLM::Worker, 1024
__PS__
123 kevin 100 100 100 Sl 3600 VLLM::Worker
__PARENTS__
100 kevin 50 100 100 Sl 3600 VLLM::EngineCore
__GRANDPARENTS__
50 kevin 1 50 50 Sl 3600 python -m vllm.entrypoints.openai.api_server --port 8089
__ENV__
123 BENCH_RUN_ID=run_20260512T120000Z_test
123 BENCH_JOB_ID=a100_Tiny_tp1_single
123 BENCH_SCOPE=synthetic_distributional
123 BENCH_PORT=8089
123 BENCH_GPUS=0
__PORTS__
8089 LISTEN 0 4096 0.0.0.0:8089 0.0.0.0:* users:((\"python\",pid=50,fd=3))
__PORT_PS__
50 kevin 1 50 50 Sl 3600 python -m vllm.entrypoints.openai.api_server --port 8089
__PORT_ENV__
50 BENCH_PORT=8089
"""

        snapshot = report.parse_host_snapshot("a100", stdout, "")

        self.assertTrue(snapshot.ok)
        self.assertEqual(snapshot.remote_user, "kevin")
        proc_state = snapshot.processes[0]
        self.assertEqual(proc_state.bench_run_id, "run_20260512T120000Z_test")
        self.assertEqual(proc_state.bench_job_id, "a100_Tiny_tp1_single")
        self.assertEqual(proc_state.bench_port, "8089")
        self.assertIn("api_server --port 8089", proc_state.grandparent_cmd)
        listener = snapshot.port_listeners[0]
        self.assertEqual(listener.port, "8089")
        self.assertEqual(listener.pid, "50")
        self.assertEqual(listener.user, "kevin")
        self.assertEqual(listener.bench_port, "8089")
        self.assertIn("api_server --port 8089", listener.cmd)

    def test_parse_host_snapshot_reports_nvidia_smi_failure(self):
        stdout = """__WHOAMI__
kevin
__GPU__
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.159
__PROC__
"""

        snapshot = report.parse_host_snapshot("3090", stdout, "")

        self.assertFalse(snapshot.ok)
        self.assertEqual(snapshot.gpus, [])
        self.assertIn("Failed to initialize NVML", snapshot.error)

    def test_vllm_worker_matches_job_from_benchmark_env(self):
        job_obj = job(short="Tiny", model_path="/models/Tiny", backend="vllm")
        worker = report.GpuProcess(
            gpu_index="0",
            gpu_uuid="GPU-test",
            pid="123",
            process_name="VLLM::Worker_TP0",
            used_memory_mib=1024,
            user="kevin",
            ppid="100",
            pgid="100",
            sid="100",
            stat="Sl",
            age_seconds=3600,
            cmd="VLLM::Worker_TP0",
            parent_user="kevin",
            parent_ppid="50",
            parent_pgid="100",
            parent_sid="100",
            parent_stat="Sl",
            parent_age_seconds=3600,
            parent_cmd="VLLM::EngineCore",
            grandparent_user="kevin",
            grandparent_ppid="1",
            grandparent_pgid="50",
            grandparent_sid="50",
            grandparent_stat="Sl",
            grandparent_age_seconds=3600,
            grandparent_cmd="python -m vllm.entrypoints.openai.api_server --model /models/Tiny",
            bench_job_id="a100_Tiny_tp1_single",
        )

        self.assertTrue(report.process_matches_job(worker, job_obj))

    def test_pending_state_with_active_run_lease_is_reported_running(self):
        job_obj = job(short="Tiny", model_path="/models/Tiny", backend="vllm")
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            scope_dir = state_root / "synthetic_distributional"
            runs_dir = scope_dir / "runs"
            runs_dir.mkdir(parents=True)
            jid = job_obj.job_id
            run_id = "run_20260512T120000Z_test"
            (scope_dir / f"{jid}.status").write_text("pending\n")
            (scope_dir / f"{jid}.run_id").write_text(f"{run_id}\n")
            started_at = datetime.now(timezone.utc).isoformat()
            (runs_dir / f"{run_id}.json").write_text(
                f"""{{
  "run_id": "run_20260512T120000Z_test",
  "job_id": "a100_Tiny_tp1_single",
  "status": "running",
  "port": "8092",
  "gpus": ["2"],
  "started_at": "{started_at}"
}}
"""
            )

            states = report.load_job_states([job_obj], state_root)

        self.assertEqual(states[0].status, "running")
        self.assertEqual(states[0].port, "8092")
        self.assertEqual(states[0].gpus, "2")

    def test_benchmark_metadata_process_is_sweep_even_without_assignment(self):
        process = proc("VLLM::EngineCore")
        process.process_name = "VLLM::Worker_TP0"
        process.cmd = "VLLM::Worker_TP0"
        process.bench_run_id = "run_20260512T120000Z_test"
        snapshot = report.HostSnapshot(host="a100", ok=True, remote_user="kevin")

        self.assertEqual(report.classify_process(process, snapshot, []), "sweep")

    def test_load_drained_hosts_from_state_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            control_dir = state_dir / "control"
            control_dir.mkdir(parents=True)
            (control_dir / "drained-hosts.txt").write_text("# comment\n3090\nh100 reason ignored\n")

            drained, path = report.load_drained_hosts(state_dir)

        self.assertEqual(drained, {"3090", "h100"})
        self.assertEqual(path.name, "drained-hosts.txt")


if __name__ == "__main__":
    unittest.main()
