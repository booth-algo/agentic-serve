import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface DataRow {
  dataScope?: 'trace_replay' | 'synthetic_distributional' | 'archived' | 'synthetic' | 'latest' | 'current' | 'archive' | 'fixed' | 'mse';
  config?: {
    profile?: string;
  };
}

interface SweepState {
  cells?: SweepCell[];
}

interface SweepCell {
  data_scope?: string;
  status?: string;
  profiles?: unknown[];
  concurrencies?: unknown[];
}

function fail(message: string): never {
  console.error(`data validation failed: ${message}`);
  process.exit(1);
}

type ValidScope = 'trace_replay' | 'synthetic_distributional' | 'archived';

function normalizeScope(scope: string | undefined): ValidScope {
  if (scope === 'synthetic_distributional' || scope === 'synthetic' || scope === 'latest') return 'synthetic_distributional';
  if (scope === 'trace_replay' || scope === 'archive') return 'trace_replay';
  if (scope === 'archived' || scope === 'current' || scope === 'canonical' || scope === 'fixed' || scope === 'fixed-grid' || scope === 'mse') return 'archived';
  return 'trace_replay';
}

function readExpectedScopes(dataPath: string): Set<ValidScope> {
  const configuredPath = process.env.SWEEP_STATE_PATH;
  const sweepStatePath = path.resolve(configuredPath ?? path.join(path.dirname(dataPath), 'sweep-state.json'));
  const expectedScopes = new Set<ValidScope>();
  if (!fs.existsSync(sweepStatePath)) {
    return expectedScopes;
  }

  const parsed = JSON.parse(fs.readFileSync(sweepStatePath, 'utf8')) as SweepState;
  for (const cell of parsed.cells ?? []) {
    const scope = normalizeScope(cell.data_scope);
    if (scope === 'synthetic_distributional') {
      // Synthetic distributional can be visible as a pending sweep surface before the first
      // batch of rows lands. Coverage still comes from sweep-state.json.
      continue;
    }
    if (cell.status !== 'done') {
      continue;
    }
    if ((cell.profiles?.length ?? 0) === 0 || (cell.concurrencies?.length ?? 0) === 0) {
      continue;
    }
    expectedScopes.add(scope);
  }
  return expectedScopes;
}

const dataPath = path.resolve(process.argv[2] ?? path.join(__dirname, '../public/data.json'));

if (!fs.existsSync(dataPath)) {
  fail(`missing data file at ${dataPath}`);
}

const parsed = JSON.parse(fs.readFileSync(dataPath, 'utf8')) as unknown;
if (!Array.isArray(parsed)) {
  fail('data file must contain a JSON array');
}

const rows = parsed as DataRow[];
const scopeCounts: Record<ValidScope, number> = { trace_replay: 0, synthetic_distributional: 0, archived: 0 };
const expectedScopes = readExpectedScopes(dataPath);

for (const row of rows) {
  const rawScope = row.dataScope ?? 'trace_replay';
  const scope = normalizeScope(rawScope);
  scopeCounts[scope] += 1;
}

if (scopeCounts.trace_replay === 0) {
  fail('expected at least one trace_replay row; real trace replay data would disappear');
}

for (const scope of expectedScopes) {
  if (scopeCounts[scope] === 0) {
    fail(`expected at least one ${scope} row because sweep-state.json has runnable ${scope} cells`);
  }
}

console.log(JSON.stringify({
  path: dataPath,
  rows: rows.length,
  scopes: scopeCounts,
  expectedScopes: [...expectedScopes].sort(),
}, null, 2));
