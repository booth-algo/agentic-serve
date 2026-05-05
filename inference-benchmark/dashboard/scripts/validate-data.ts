import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REQUIRED_CURRENT_PROFILES = [
  'chat-singleturn',
  'coding-singleturn',
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
] as const;

interface DataRow {
  dataScope?: 'synthetic' | 'latest' | 'current' | 'archive' | 'fixed' | 'mse';
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

type ValidScope = 'synthetic' | 'current' | 'fixed' | 'mse';

function normalizeScope(scope: string | undefined): 'synthetic' | 'current' | 'archive' | 'fixed' | 'mse' {
  if (scope === 'latest') return 'synthetic';
  if (scope === 'synthetic' || scope === 'current' || scope === 'archive' || scope === 'fixed' || scope === 'mse') return scope;
  return 'archive';
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
    if (scope !== 'synthetic' && scope !== 'current' && scope !== 'fixed' && scope !== 'mse') {
      continue;
    }
    if (scope === 'synthetic') {
      // Synthetic can be visible as a pending sweep surface before the first
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
const scopeCounts = { synthetic: 0, current: 0, archive: 0, fixed: 0, mse: 0 };
const currentProfiles = new Map<string, number>();
const expectedScopes = readExpectedScopes(dataPath);

for (const row of rows) {
  const rawScope = row.dataScope ?? 'archive';
  if (rawScope !== 'synthetic' && rawScope !== 'latest' && rawScope !== 'current' && rawScope !== 'archive' && rawScope !== 'fixed' && rawScope !== 'mse') {
    fail(`invalid dataScope ${JSON.stringify(rawScope)}`);
  }
  const scope = normalizeScope(rawScope);
  scopeCounts[scope] += 1;
  if (scope === 'synthetic' || scope === 'current' || scope === 'fixed') {
    const profile = row.config?.profile;
    if (profile) currentProfiles.set(profile, (currentProfiles.get(profile) ?? 0) + 1);
  }
}

if (scopeCounts.current === 0) {
  fail('expected at least one current row; generated data would make canonical coverage zero');
}

if (scopeCounts.archive === 0) {
  fail('expected at least one archive row; archived benchmark data would disappear');
}

for (const scope of expectedScopes) {
  if (scopeCounts[scope] === 0) {
    fail(`expected at least one ${scope} row because sweep-state.json has runnable ${scope} cells`);
  }
}

const missingProfiles = REQUIRED_CURRENT_PROFILES.filter((profile) => !currentProfiles.has(profile));
if (missingProfiles.length > 0) {
  fail(`missing current canonical profiles: ${missingProfiles.join(', ')}`);
}

console.log(JSON.stringify({
  path: dataPath,
  rows: rows.length,
  scopes: scopeCounts,
  expectedScopes: [...expectedScopes].sort(),
  currentProfiles: Object.fromEntries(
    REQUIRED_CURRENT_PROFILES.map((profile) => [profile, currentProfiles.get(profile) ?? 0]),
  ),
}, null, 2));
