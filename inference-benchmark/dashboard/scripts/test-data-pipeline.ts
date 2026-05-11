import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const dashboardRoot = path.resolve(__dirname, "..");

function runTsx(args: string[], env: NodeJS.ProcessEnv = {}) {
  return spawnSync("npx", ["tsx", ...args], {
    cwd: dashboardRoot,
    env: { ...process.env, ...env },
    encoding: "utf8",
  });
}

function baseRow(profile = "chat-singleturn", concurrency = 320, dataScope = "trace_replay") {
  return {
    config: {
      model: "/models/Llama-3.1-8B-Instruct",
      backend: "vllm",
      profile,
      concurrency,
      tensor_parallel: 1,
      max_model_len: 32768,
      dataset: profile,
      dashboard_scope: dataScope,
    },
    summary: {
      total_requests: concurrency,
      successful_requests: concurrency,
      failed_requests: 0,
      median_ttft_ms: 100,
      median_tpot_ms: 10,
      median_e2el_ms: 500,
    },
  };
}

function writeResult(
  root: string,
  scope: "trace_replay" | "synthetic_distributional" | "archived",
  profile = "chat-singleturn",
  concurrency = 320,
) {
  const dir = path.join(root, scope, "a100_Llama-3.1-8B_tp1_vllm");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, `${profile}_c${concurrency}.json`),
    JSON.stringify(baseRow(profile, concurrency, scope), null, 2),
  );
}

function testBuildDataKeepsScopesSeparate(tmp: string) {
  const resultsDir = path.join(tmp, "results");
  const outputPath = path.join(tmp, "data.json");
  writeResult(resultsDir, "synthetic_distributional", "chat-singleturn-synth");
  writeResult(resultsDir, "trace_replay");
  writeResult(resultsDir, "archived");

  const result = runTsx(["scripts/build-data.ts"], {
    BENCHMARK_RESULTS_DIR: resultsDir,
    DASHBOARD_DATA_OUTPUT: outputPath,
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);

  const rows = JSON.parse(fs.readFileSync(outputPath, "utf8"));
  const scopes = rows.map((row: { dataScope: string }) => row.dataScope).sort();
  assert.deepEqual(scopes, ["archived", "synthetic_distributional", "trace_replay"]);

  for (const scope of scopes) {
    const scopedPath = path.join(tmp, `data.${scope}.json`);
    assert.equal(fs.existsSync(scopedPath), true, `${scopedPath} should be written`);
    const scopedRows = JSON.parse(fs.readFileSync(scopedPath, "utf8"));
    assert.deepEqual(
      scopedRows.map((row: { dataScope: string }) => row.dataScope),
      [scope],
    );
  }
}

function testValidateRejectsEmptyCompletedScope(tmp: string) {
  const dataPath = path.join(tmp, "data-without-archived.json");
  const sweepStatePath = path.join(tmp, "sweep-state.json");

  const rows = [{
    dataScope: "trace_replay",
    config: {
      profile: "chat-singleturn",
      concurrency: 320,
    },
  }];

  fs.writeFileSync(dataPath, JSON.stringify(rows, null, 2));
  fs.writeFileSync(
    sweepStatePath,
    JSON.stringify(
      {
        cells: [
          {
            data_scope: "archived",
            status: "done",
            profiles: ["chat-singleturn"],
            concurrencies: [200],
          },
        ],
      },
      null,
      2,
    ),
  );

  const result = runTsx(["scripts/validate-data.ts", dataPath], {
    SWEEP_STATE_PATH: sweepStatePath,
  });
  assert.notEqual(result.status, 0, "validation should fail when expected archived scope is empty");
  assert.match(result.stderr + result.stdout, /archived/);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agentic-serve-data-pipeline-"));
try {
  testBuildDataKeepsScopesSeparate(tmp);
  testValidateRejectsEmptyCompletedScope(tmp);
  console.log("data pipeline regression tests passed");
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
