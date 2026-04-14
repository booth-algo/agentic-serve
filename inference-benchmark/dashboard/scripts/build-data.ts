import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface RawResult {
  config: {
    model: string;
    backend: string;
    profile: string;
    concurrency: number;
    [key: string]: unknown;
  };
  summary: Record<string, unknown>;
}

interface ScatterPoint {
  input_tokens: number;
  ttft_ms: number;
  turn_index: number;
}

interface PerTurnEntry {
  turn_index: number;
  num_requests: number;
  successful: number;
  mean_ttft_ms: number;
  median_ttft_ms: number;
  p90_ttft_ms: number;
  p99_ttft_ms: number;
  mean_tpot_ms: number;
  median_tpot_ms: number;
  mean_e2el_ms: number;
  median_e2el_ms: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
}

interface EnrichedResult {
  config: RawResult['config'];
  summary: RawResult['summary'];
  hardware: string;
  quant: string;
  modelShort: string;
  seriesKey: string;
  filename: string;
  perTurn?: PerTurnEntry[];
  scatterData?: ScatterPoint[];
}

const RESULTS_DIR = path.resolve(__dirname, '../../results');
const OUTPUT_FILE = path.resolve(__dirname, '../public/data.json');

// Files to skip — test files, debug files, symlinks
const SKIP_PATTERNS = [
  /^smoke_test/,
  /^test_/,
  /^rng_/,
  /^doublewrap/,
  /^latest\.json$/,
  /^output_short_conc/,  // ambiguous naming, no hardware prefix
];

function detectHardware(filename: string, dirPath: string): string {
  const fp = filename.toLowerCase();
  const dir = dirPath.toLowerCase();

  if (fp.includes('h100x2') || dir.includes('h100x2')) return 'H100x2';
  if (fp.includes('h100_tcp') || dir.includes('h100_tcp')) return 'H100-TCP';
  if (fp.includes('a6000') || dir.includes('a6000')) return 'A6000';

  // Infer from directory name
  if (dir.includes('h100_70b_fp8')) return 'H100';

  // Infer from TP size in directory path (4xH100 RunPod)
  if (/_tp4_/.test(dir)) return 'H100x4';
  if (/_tp2_/.test(fp) || /_tp2_/.test(dir)) return 'H100x2';
  if (/_tp1_/.test(fp) || /_tp1_/.test(dir)) return 'H100';
  // Match _tp2 at end of dir name (e.g. tpot_validation_Qwen3.5-27B_tp2)
  if (/_tp2$/.test(dir)) return 'H100x2';
  if (/_tp4$/.test(dir)) return 'H100x4';
  if (/_tp1$/.test(dir)) return 'H100';
  if (fp.includes('h100') || dir.includes('h100')) return 'H100';

  return 'Unknown';
}

function detectQuant(filename: string, model: string, dirPath: string): string {
  const combined = `${filename} ${model} ${dirPath}`.toLowerCase();
  if (combined.includes('fp8')) return 'FP8';
  if (combined.includes('bf16') || combined.includes('bfloat16')) return 'BF16';
  // Default: if no explicit FP8 marker, assume BF16
  return 'BF16';
}

function shortenModel(model: string): string {
  let short = model;
  // Remove local path prefix (e.g. /workspace/models/Llama-3.1-8B-Instruct)
  short = short.replace(/^.*\//, '');
  // Remove common HF prefixes
  short = short.replace(/^meta-llama\/Meta-/i, '');
  short = short.replace(/^meta-llama\//i, '');
  short = short.replace(/^neuralmagic\/(Meta-)?/i, '');
  short = short.replace(/^Qwen\//i, '');
  // Remove -Instruct suffix
  short = short.replace(/-Instruct$/i, '');
  // Remove -FP8 suffix (captured separately in quant)
  short = short.replace(/-FP8$/i, '');
  return short;
}

// Old → new profile name mapping (matches profiles.py PROFILE_ALIASES)
const PROFILE_ALIASES: Record<string, string> = {
  'chatbot-short': 'chat-short',
  'chatbot-multi-turn': 'chat-medium',
  'chatbot-multi-turn': 'chat-medium',
  'rag-retrieval': 'chat-medium',
  'rag-heavy': 'chat-medium',
  'coding-assist': 'chat-medium',
  'coding-heavy': 'chat-medium',
  'summarization': 'chat-medium',
  'agentic-tool-use': 'chat-medium',
  'computer-use-basic': 'chat-short',
  'customer-support-basic': 'chat-short',
  'output-short': 'prefill-heavy',
  'output-long': 'decode-heavy',
  'random-inferencex': 'random-1k',
  'random-inferencex-legacy': 'random-1k',
  'random-inferencex-doublewrap': 'random-1k',
  'multi-turn-short': 'chat-multiturn-short',
  'multi-turn-medium': 'chat-multiturn-medium',
  'multi-turn-long': 'chat-multiturn-long',
};

function normalizeProfile(profile: string): string {
  // Normalize underscores to hyphens, then resolve aliases
  const normalized = profile.replace(/_/g, '-');
  return PROFILE_ALIASES[normalized] ?? normalized;
}

function detectBackendFromFilename(filename: string, configBackend: string): string {
  const fn = filename.toLowerCase();
  if (fn.startsWith('sglang_') || fn.includes('_sglang_')) return 'sglang';
  if (fn.startsWith('vllm_') || fn.includes('_vllm_')) return 'vllm';
  return configBackend || 'vllm';
}

function collectJsonFiles(dir: string, relDir: string = ''): Array<{ fullPath: string; filename: string; relDir: string }> {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: Array<{ fullPath: string; filename: string; relDir: string }> = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      // Recurse into subdirectories
      files.push(...collectJsonFiles(fullPath, path.join(relDir, entry.name)));
    } else if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push({ fullPath, filename: entry.name, relDir });
    }
  }

  return files;
}

function shouldSkip(filename: string, relDir: string): boolean {
  // Skip crossval, inferencex, and archive subdirectories
  if (relDir.includes('crossval') || relDir.includes('inferencex') || relDir.includes('archive')) return true;

  for (const pattern of SKIP_PATTERNS) {
    if (pattern.test(filename)) return true;
  }

  return false;
}

function main() {
  console.log(`Reading results from: ${RESULTS_DIR}`);

  const jsonFiles = collectJsonFiles(RESULTS_DIR);
  console.log(`Found ${jsonFiles.length} JSON files total`);

  const results: EnrichedResult[] = [];
  let skipped = 0;
  let errors = 0;

  for (const { fullPath, filename, relDir } of jsonFiles) {
    if (shouldSkip(filename, relDir)) {
      skipped++;
      continue;
    }

    try {
      const raw = JSON.parse(fs.readFileSync(fullPath, 'utf-8')) as RawResult;

      // Validate required fields
      if (!raw.config || !raw.summary || !raw.config.model || !raw.summary.median_ttft_ms === undefined) {
        skipped++;
        continue;
      }

      // Must have concurrency
      const concurrency = raw.config.concurrency ?? raw.summary.concurrency;
      if (!concurrency) {
        skipped++;
        continue;
      }

      const hardware = detectHardware(filename, relDir);
      const quant = detectQuant(filename, raw.config.model, relDir);
      const modelShort = shortenModel(raw.config.model);
      const backend = detectBackendFromFilename(filename, raw.config.backend);
      const profile = normalizeProfile(raw.config.profile || (raw.summary as Record<string, string>).profile || 'unknown');

      // Skip unknown hardware or unknown profiles
      if (hardware === 'Unknown') {
        skipped++;
        continue;
      }

      // Skip underloaded single-turn results (nreq < concurrency)
      // Multi-turn uses num_requests=num_sessions which is intentionally < concurrency
      const mode = raw.config.mode || (relDir.includes('multiturn') ? 'multi-turn' : 'single-turn');
      if (mode !== 'multi-turn' && concurrency > 1 && raw.config.num_requests && raw.config.num_requests < concurrency) {
        skipped++;
        continue;
      }

      const seriesKey = `${hardware} / ${modelShort} ${quant} / ${backend} / ${profile}`;

      // Extract scatter data from per_request (multi-turn results with turn_index)
      let scatterData: ScatterPoint[] | undefined;
      if (raw.summary && (raw as Record<string, unknown>).per_request) {
        const perReq = (raw as Record<string, unknown>).per_request as Array<Record<string, unknown>>;
        const points = perReq
          .filter((r) => r.success && r.turn_index !== undefined && r.ttft_ms != null && r.input_tokens != null)
          .map((r) => ({
            input_tokens: r.input_tokens as number,
            ttft_ms: r.ttft_ms as number,
            turn_index: r.turn_index as number,
          }));
        if (points.length > 0) {
          scatterData = points;
        }
      }

      // Check for matching _per_turn.json file
      let perTurn: PerTurnEntry[] | undefined;
      const perTurnPath = fullPath.replace(/\.json$/, '_per_turn.json');
      if (fs.existsSync(perTurnPath)) {
        try {
          const ptRaw = JSON.parse(fs.readFileSync(perTurnPath, 'utf-8'));
          if (ptRaw.per_turn && Array.isArray(ptRaw.per_turn)) {
            perTurn = ptRaw.per_turn as PerTurnEntry[];
          }
        } catch {
          // Skip if per-turn file is malformed
        }
      }

      results.push({
        config: { ...raw.config, backend, profile, concurrency },
        summary: raw.summary,
        hardware,
        quant,
        modelShort,
        seriesKey,
        filename: path.join(relDir, filename),
        ...(perTurn ? { perTurn } : {}),
        ...(scatterData ? { scatterData } : {}),
      });
    } catch (e) {
      errors++;
      console.error(`  Error parsing ${fullPath}: ${(e as Error).message}`);
    }
  }

  // Deduplicate: if same series+concurrency appears multiple times, keep the one
  // from the deeper directory (h100_70b_fp8/ subdir files are duplicates of root)
  const seen = new Map<string, EnrichedResult>();
  for (const r of results) {
    const dedupeKey = `${r.seriesKey}::${r.config.concurrency}`;
    const existing = seen.get(dedupeKey);
    if (!existing) {
      seen.set(dedupeKey, r);
    } else {
      // Prefer the file NOT in the subdirectory (root-level is canonical)
      // Unless root has no relDir and subdir does, keep whichever has more specific path
      if (r.filename.includes('/') && !existing.filename.includes('/')) {
        // existing is root-level, keep it
      } else if (!r.filename.includes('/') && existing.filename.includes('/')) {
        seen.set(dedupeKey, r); // r is root-level, prefer it
      }
      // Otherwise keep first seen
    }
  }

  const dedupedResults = Array.from(seen.values());

  // Sort by hardware, model, backend, profile, concurrency
  dedupedResults.sort((a, b) => {
    if (a.hardware !== b.hardware) return a.hardware.localeCompare(b.hardware);
    if (a.modelShort !== b.modelShort) return a.modelShort.localeCompare(b.modelShort);
    if (a.config.backend !== b.config.backend) return a.config.backend.localeCompare(b.config.backend);
    if (a.config.profile !== b.config.profile) return a.config.profile.localeCompare(b.config.profile);
    return a.config.concurrency - b.config.concurrency;
  });

  // Strip fields not used by the dashboard to reduce payload size
  const CONFIG_KEEP = new Set(['backend', 'profile', 'concurrency', 'model', 'mode', 'num_requests']);
  const SUMMARY_KEEP = new Set([
    'concurrency', 'num_requests', 'duration_s', 'successful_requests', 'failed_requests',
    'request_throughput', 'input_token_throughput', 'output_token_throughput', 'total_token_throughput',
    'total_input_tokens', 'total_output_tokens',
    'mean_ttft_ms', 'median_ttft_ms', 'p90_ttft_ms', 'p99_ttft_ms',
    'mean_tpot_ms', 'median_tpot_ms', 'p90_tpot_ms', 'p99_tpot_ms',
    'mean_itl_ms', 'median_itl_ms', 'p90_itl_ms', 'p99_itl_ms',
    'mean_e2el_ms', 'median_e2el_ms', 'p90_e2el_ms', 'p99_e2el_ms',
    'errors',
  ]);

  const slimResults = dedupedResults.map(r => {
    const config: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(r.config)) {
      if (CONFIG_KEEP.has(k)) config[k] = v;
    }
    const summary: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(r.summary)) {
      if (SUMMARY_KEEP.has(k)) summary[k] = v;
    }
    const slim: Record<string, unknown> = { config, summary, hardware: r.hardware, quant: r.quant, modelShort: r.modelShort, seriesKey: r.seriesKey };
    if ((r as Record<string, unknown>).perTurn) slim.perTurn = (r as Record<string, unknown>).perTurn;
    return slim;
  });

  // Ensure output directory exists
  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(slimResults));

  console.log(`\nResults:`);
  console.log(`  Included: ${dedupedResults.length}`);
  console.log(`  Skipped:  ${skipped}`);
  console.log(`  Errors:   ${errors}`);
  console.log(`  Output:   ${OUTPUT_FILE}`);

  // Print series summary
  const seriesMap = new Map<string, number>();
  for (const r of dedupedResults) {
    seriesMap.set(r.seriesKey, (seriesMap.get(r.seriesKey) || 0) + 1);
  }
  console.log(`\nSeries (${seriesMap.size}):`);
  for (const [key, count] of seriesMap) {
    console.log(`  ${key} (${count} points)`);
  }
}

main();
