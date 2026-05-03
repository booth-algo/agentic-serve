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
  dataScope?: 'current' | 'archive' | 'fixed';
  config?: {
    profile?: string;
  };
}

function fail(message: string): never {
  console.error(`data validation failed: ${message}`);
  process.exit(1);
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
const scopeCounts = { current: 0, archive: 0, fixed: 0 };
const currentProfiles = new Map<string, number>();

for (const row of rows) {
  const scope = row.dataScope ?? 'archive';
  if (scope !== 'current' && scope !== 'archive' && scope !== 'fixed') {
    fail(`invalid dataScope ${JSON.stringify(scope)}`);
  }
  scopeCounts[scope] += 1;
  if (scope === 'current' || scope === 'fixed') {
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

const missingProfiles = REQUIRED_CURRENT_PROFILES.filter((profile) => !currentProfiles.has(profile));
if (missingProfiles.length > 0) {
  fail(`missing current canonical profiles: ${missingProfiles.join(', ')}`);
}

console.log(JSON.stringify({
  path: dataPath,
  rows: rows.length,
  scopes: scopeCounts,
  currentProfiles: Object.fromEntries(
    REQUIRED_CURRENT_PROFILES.map((profile) => [profile, currentProfiles.get(profile) ?? 0]),
  ),
}, null, 2));
