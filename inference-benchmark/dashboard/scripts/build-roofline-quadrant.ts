import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Model parameters ────────────────────────────────────────────────────────
interface ModelParams {
  params_B: number;
  d_model: number;
  n_layers: number;
  n_kv_heads: number;
  head_dim: number;
  is_moe?: boolean;
  active_params_B?: number;
}

const MODEL_PARAMS: Record<string, ModelParams> = {
  'Llama-3.1-8B':  { params_B: 8,   d_model: 4096, n_layers: 32,  n_kv_heads: 8,  head_dim: 128 },
  'Qwen3.5-9B':    { params_B: 9,   d_model: 3584, n_layers: 36,  n_kv_heads: 4,  head_dim: 128 },
  'gpt-oss-20b':   { params_B: 21,  d_model: 3072, n_layers: 52,  n_kv_heads: 8,  head_dim: 128, is_moe: true, active_params_B: 3.6 },
  'Qwen3.5-27B':   { params_B: 27,  d_model: 4096, n_layers: 48,  n_kv_heads: 4,  head_dim: 128 },
  'gpt-oss-120b':  { params_B: 117, d_model: 5120, n_layers: 120, n_kv_heads: 16, head_dim: 128, is_moe: true, active_params_B: 5.1 },
  'Qwen2.5-72B':   { params_B: 72,  d_model: 8192, n_layers: 80,  n_kv_heads: 8,  head_dim: 128 },
  'Llama-3.1-70B':  { params_B: 70,  d_model: 8192, n_layers: 80,  n_kv_heads: 8,  head_dim: 128 },
  'Llama-3.3-70B':  { params_B: 70,  d_model: 8192, n_layers: 80,  n_kv_heads: 8,  head_dim: 128 },
  'Qwen3-32B':     { params_B: 32,  d_model: 5120, n_layers: 64,  n_kv_heads: 8,  head_dim: 128 },
};

// ── Avg sequence lengths per profile (input + output) ────────────────────────
const PROFILE_AVG_SEQ: Record<string, number> = {
  // Canonical natural chat ST. chat-short/chat-medium are retired from the
  // main sweep because they mostly differ by output length.
  'chat-singleturn': 486,
  'coding-agent': 6600,
  'prefill-heavy': 4200,
  'decode-heavy': 2200,
  'random-1k': 1024,
  'chat-multiturn-short': 4500,
  'chat-multiturn-medium': 8000,
  'chat-multiturn-long': 16000,
  'swebench-multiturn-short': 16000,
  'swebench-multiturn-medium': 32000,
  'swebench-multiturn-long': 64000,
  'terminalbench-multiturn-short': 16000,
  'terminalbench-multiturn-medium': 32000,
  'terminalbench-multiturn-long': 64000,
};

const HISTORICAL_PROFILE_ALIASES: Record<string, string> = {
  'chat-long': 'chat-singleturn',
};

const BYTES_PER_PARAM = 2; // BF16

const H100_HARDWARE = {
  peak_tflops: 989,
  memory_bw_tbs: 3.35,
  hbm_gb: 80,
};

// ── Raw result shape (subset) ────────────────────────────────────────────────
interface RawConfig {
  model: string;
  backend: string;
  profile: string;
  concurrency: number;
  [key: string]: unknown;
}

interface RawSummary {
  output_token_throughput: number;
  mean_tpot_ms: number;
  median_tpot_ms: number;
  mean_ttft_ms: number;
  median_ttft_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  successful_requests: number;
  [key: string]: unknown;
}

interface RawResult {
  config: RawConfig;
  summary: RawSummary;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function shortenModel(model: string): string {
  let short = model;
  short = short.replace(/^.*\//, '');
  short = short.replace(/-Instruct$/i, '');
  short = short.replace(/-FP8$/i, '');
  return short;
}

function detectHardware(dirPath: string): string {
  const dir = dirPath.toLowerCase();
  if (/_tp4_/.test(dir)) return 'H100x4';
  if (/_tp2_/.test(dir)) return 'H100x2';
  return 'H100';
}

function detectEngine(dirPath: string, configBackend: string): string {
  const dir = dirPath.toLowerCase();
  if (dir.includes('sglang')) return 'sglang';
  if (dir.includes('vllm')) return 'vllm';
  return configBackend || 'vllm';
}

function findModelParams(modelShort: string): ModelParams | null {
  // Direct match
  if (MODEL_PARAMS[modelShort]) return MODEL_PARAMS[modelShort];
  // Fuzzy: check if any key is a prefix
  for (const [key, params] of Object.entries(MODEL_PARAMS)) {
    if (modelShort.startsWith(key) || modelShort.includes(key)) return params;
  }
  return null;
}

function computeOI(
  outputThroughput: number,
  activeParamsB: number,
  concurrency: number,
  avgInputTokens: number,
  successfulRequests: number,
): number {
  // Achieved OI based on actual throughput:
  // For each output token, the model performs ~2 * active_params FLOPs (forward pass)
  // The memory accessed is model weights (read once per batch) + KV cache reads
  //
  // Simplified: OI ≈ (total_FLOPs/s) / (memory_bandwidth_used)
  //           ≈ (tok/s * 2 * active_params * 1e9) / (active_params * 1e9 * 2 / concurrency + kv_overhead)
  //
  // More practically, separate prefill vs decode contribution:
  // - Prefill tokens have high OI (batched matmuls over long sequences)
  // - Decode tokens have low OI (sequential, one token at a time per request)
  //
  // Use the ratio of average input length to batch size as an OI proxy:
  // When ISL is high relative to batch size → prefill-dominated → high OI
  // When batch (concurrency) is high relative to ISL → decode-dominated but batch helps
  const avgISL = successfulRequests > 0 ? avgInputTokens / successfulRequests : 500;

  // OI = (2 * FLOPs_per_token) / (bytes_accessed_per_token)
  // For decode at batch B: bytes_accessed = model_weights / B (amortized over batch)
  // FLOPs = 2 * active_params per token
  // So decode OI ≈ B (batch_size)
  // For prefill at seq_len S: OI ≈ S (long sequence = many FLOPs per weight read)
  //
  // Blend: effective OI ≈ (prefill_fraction * avgISL + decode_fraction * concurrency)
  // where prefill_fraction ≈ avgISL / (avgISL + avg_OSL)
  const avgOSL = outputThroughput > 0 && successfulRequests > 0
    ? (outputThroughput * (successfulRequests > 0 ? 1 : 1)) // approximate from throughput
    : 300;

  // Simple effective OI: weighted by token count
  // Prefill processes avgISL tokens at OI ≈ avgISL
  // Decode generates tokens at OI ≈ concurrency
  const prefillTokens = avgISL;
  const decodeTokens = avgOSL > 0 ? avgOSL : 300;
  const totalTokens = prefillTokens + decodeTokens;

  const effectiveOI = (prefillTokens / totalTokens) * Math.min(avgISL, 10000)
                    + (decodeTokens / totalTokens) * concurrency;

  return effectiveOI;
}

function computeCF(
  paramsB: number,
  nLayers: number,
  nKvHeads: number,
  headDim: number,
  concurrency: number,
  avgSeqLen: number,
): number {
  const modelWeightBytes = paramsB * 1e9 * BYTES_PER_PARAM;
  // KV cache per token: 2 (key+value) * n_layers * n_kv_heads * head_dim * 2 (BF16)
  const kvCachePerToken = 2 * nLayers * nKvHeads * headDim * BYTES_PER_PARAM;
  const kvCachePerRequest = kvCachePerToken * avgSeqLen;
  const totalBytes = modelWeightBytes + kvCachePerRequest * concurrency;
  return totalBytes / 1e9; // GB
}

// ── Collect JSON files recursively ───────────────────────────────────────────
function collectJsonFiles(dir: string): Array<{ fullPath: string; relDir: string }> {
  const results: Array<{ fullPath: string; relDir: string }> = [];
  if (!fs.existsSync(dir)) return results;

  function walk(d: string, rel: string) {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) {
        walk(full, path.join(rel, entry.name));
      } else if (entry.isFile() && entry.name.endsWith('.json')) {
        results.push({ fullPath: full, relDir: rel });
      }
    }
  }
  walk(dir, '');
  return results;
}

// ── Main ─────────────────────────────────────────────────────────────────────
function main() {
  const RESULTS_DIR = path.resolve(__dirname, '../../results');
  const OUTPUT_FILE = path.resolve(__dirname, '../public/roofline-quadrant.json');

  console.log(`Scanning results from: ${RESULTS_DIR}`);

  const files = collectJsonFiles(RESULTS_DIR).filter(
    (f) => !f.relDir.includes('roofline') && !f.relDir.includes('archive'),
  );
  console.log(`Found ${files.length} result files`);

  interface QuadrantPoint {
    model: string;
    profile: string;
    concurrency: number;
    engine: string;
    hardware: string;
    oi: number;
    cf_gb: number;
    output_tput: number;
    tpot_ms: number;
    ttft_ms: number;
  }

  const points: QuadrantPoint[] = [];
  let skipped = 0;

  for (const { fullPath, relDir } of files) {
    try {
      const raw = JSON.parse(fs.readFileSync(fullPath, 'utf-8')) as RawResult;
      if (!raw.config || !raw.summary) continue;

      const modelShort = shortenModel(raw.config.model);
      const params = findModelParams(modelShort);
      if (!params) {
        skipped++;
        continue;
      }

      const profile = HISTORICAL_PROFILE_ALIASES[raw.config.profile] ?? raw.config.profile;
      const concurrency = raw.config.concurrency;
      if (!concurrency || concurrency < 1) continue;

      const avgSeqLen = PROFILE_AVG_SEQ[profile];
      if (!avgSeqLen) {
        skipped++;
        continue;
      }

      const activeParamsB = params.active_params_B ?? params.params_B;
      const outputTput = raw.summary.output_token_throughput ?? 0;
      const tpotMs = raw.summary.median_tpot_ms ?? raw.summary.mean_tpot_ms ?? 0;
      const ttftMs = raw.summary.median_ttft_ms ?? raw.summary.mean_ttft_ms ?? 0;
      const totalInputTokens = raw.summary.total_input_tokens ?? 0;
      const successfulRequests = raw.summary.successful_requests ?? 0;

      const oi = computeOI(outputTput, activeParamsB, concurrency, totalInputTokens, successfulRequests);
      const cf = computeCF(
        params.params_B,
        params.n_layers,
        params.n_kv_heads,
        params.head_dim,
        concurrency,
        avgSeqLen,
      );

      if (outputTput <= 0 || tpotMs <= 0) continue;

      points.push({
        model: modelShort,
        profile,
        concurrency,
        engine: detectEngine(relDir, raw.config.backend),
        hardware: detectHardware(relDir),
        oi: Math.round(oi * 100) / 100,
        cf_gb: Math.round(cf * 100) / 100,
        output_tput: Math.round(outputTput * 10) / 10,
        tpot_ms: Math.round(tpotMs * 100) / 100,
        ttft_ms: Math.round(ttftMs * 100) / 100,
      });
    } catch {
      // Skip unparseable files
    }
  }

  // Sort for deterministic output
  points.sort((a, b) => {
    if (a.model !== b.model) return a.model.localeCompare(b.model);
    if (a.profile !== b.profile) return a.profile.localeCompare(b.profile);
    if (a.engine !== b.engine) return a.engine.localeCompare(b.engine);
    return a.concurrency - b.concurrency;
  });

  const output = {
    hardware: H100_HARDWARE,
    points,
  };

  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output));

  console.log(`\nResults:`);
  console.log(`  Points: ${points.length}`);
  console.log(`  Skipped: ${skipped} (no model params or profile)`);
  console.log(`  Models: ${[...new Set(points.map((p) => p.model))].sort().join(', ')}`);
  console.log(`  Profiles: ${[...new Set(points.map((p) => p.profile))].sort().join(', ')}`);
  console.log(`  Output: ${OUTPUT_FILE}`);
}

main();
