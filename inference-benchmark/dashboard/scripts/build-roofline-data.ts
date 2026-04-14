import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface RawKernel {
  name: string;
  category: string;
  cuda_time_us: number;
  calls: number;
  flops: number;
  theoretical_bytes: number;
  arithmetic_intensity: number;
  achieved_tflops: number;
}

interface RawCategorySummary {
  total_cuda_time_us: number;
  total_flops: number;
  total_bytes: number;
  kernel_count: number;
  call_count: number;
  duration_pct: number;
  avg_arithmetic_intensity: number;
  avg_achieved_tflops: number;
}

interface RawRooflineFile {
  model: string;
  phase: string;
  batch_size: number;
  seq_len: number;
  num_layers_profiled: number;
  profiler: string;
  source_file?: string;
  total_kernels?: number;
  model_info?: Record<string, unknown>;
  kernels: RawKernel[];
  category_summary: Record<string, RawCategorySummary>;
}

export interface RooflineKernel {
  name: string;
  category: string;
  cuda_time_us: number;
  calls: number;
  flops: number;
  arithmetic_intensity: number;
  achieved_tflops: number;
  model: string;
  phase: string;
  batch_size: number;
}

export interface RooflineCategorySummary {
  category: string;
  total_cuda_time_us: number;
  duration_pct: number;
  avg_arithmetic_intensity: number;
  avg_achieved_tflops: number;
  kernel_count: number;
}

export interface RooflineEntry {
  model: string;
  phase: string;
  batch_size: number;
  seq_len: number;
  total_cuda_time_us: number;
  kernels: RooflineKernel[];
  category_summary: RooflineCategorySummary[];
}

export interface RooflineData {
  entries: RooflineEntry[];
  models: string[];
  phases: string[];
  batch_sizes: number[];
}

const PARSED_DIR = path.resolve(__dirname, '../../results/roofline/parsed');
const OUTPUT_FILE = path.resolve(__dirname, '../public/roofline-data.json');

function main() {
  console.log(`Reading roofline data from: ${PARSED_DIR}`);

  if (!fs.existsSync(PARSED_DIR)) {
    console.error(`Directory not found: ${PARSED_DIR}`);
    process.exit(1);
  }

  const files = fs.readdirSync(PARSED_DIR).filter((f) => f.endsWith('.json'));
  console.log(`Found ${files.length} roofline JSON files`);

  const entries: RooflineEntry[] = [];
  const modelsSet = new Set<string>();
  const phasesSet = new Set<string>();
  const batchSizesSet = new Set<number>();

  for (const filename of files) {
    const fullPath = path.join(PARSED_DIR, filename);
    try {
      const raw = JSON.parse(fs.readFileSync(fullPath, 'utf-8')) as RawRooflineFile;

      // Filter out invalid kernels
      const validKernels = raw.kernels.filter(
        (k) => k.arithmetic_intensity > 0 && k.achieved_tflops > 0
      );

      const totalCudaTime = Object.values(raw.category_summary).reduce(
        (sum, s) => sum + s.total_cuda_time_us,
        0
      );

      const kernels: RooflineKernel[] = validKernels.map((k) => ({
        name: k.name,
        category: k.category,
        cuda_time_us: k.cuda_time_us,
        calls: k.calls,
        flops: k.flops,
        arithmetic_intensity: k.arithmetic_intensity,
        achieved_tflops: k.achieved_tflops,
        model: raw.model,
        phase: raw.phase,
        batch_size: raw.batch_size,
      }));

      const category_summary: RooflineCategorySummary[] = Object.entries(
        raw.category_summary
      ).map(([cat, s]) => ({
        category: cat,
        total_cuda_time_us: s.total_cuda_time_us,
        duration_pct: s.duration_pct,
        avg_arithmetic_intensity: s.avg_arithmetic_intensity,
        avg_achieved_tflops: s.avg_achieved_tflops,
        kernel_count: s.kernel_count,
      }));

      const entry: RooflineEntry = {
        model: raw.model,
        phase: raw.phase,
        batch_size: raw.batch_size,
        seq_len: raw.seq_len,
        total_cuda_time_us: totalCudaTime,
        kernels,
        category_summary,
      };

      entries.push(entry);
      modelsSet.add(raw.model);
      phasesSet.add(raw.phase);
      batchSizesSet.add(raw.batch_size);

      console.log(
        `  ${filename}: ${kernels.length} valid kernels (of ${raw.kernels.length} total)`
      );
    } catch (e) {
      console.error(`  Error parsing ${filename}: ${(e as Error).message}`);
    }
  }

  // Sort entries for deterministic output
  entries.sort((a, b) => {
    if (a.model !== b.model) return a.model.localeCompare(b.model);
    if (a.phase !== b.phase) return a.phase.localeCompare(b.phase);
    return a.batch_size - b.batch_size;
  });

  const output: RooflineData = {
    entries,
    models: Array.from(modelsSet).sort(),
    phases: Array.from(phasesSet).sort(),
    batch_sizes: Array.from(batchSizesSet).sort((a, b) => a - b),
  };

  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output));

  console.log(`\nResults:`);
  console.log(`  Entries: ${entries.length}`);
  console.log(`  Models:  ${output.models.join(', ')}`);
  console.log(`  Phases:  ${output.phases.join(', ')}`);
  console.log(`  Batches: ${output.batch_sizes.join(', ')}`);
  console.log(`  Output:  ${OUTPUT_FILE}`);
}

main();
