export interface ProfileMeta {
  displayName: string;
  workloadGroup: string;
  benchmarkVisible?: boolean;
  agentType: 'chat' | 'coding' | 'terminal' | 'computer-use' | 'stress';
  turnStyle: 'single-turn' | 'multi-turn';
  dataSource: string;    // "ShareGPT", "SWEBench", "TerminalBench", "OSWorld", "Random", "Test"
  isl: string;
  osl: string;
  description: string;
}

export type DataScope = 'trace_replay' | 'synthetic_distributional' | 'archived';

export interface DataScopeMeta {
  label: string;
  shortLabel: string;
  eyebrow: string;
  description: string;
  accent: string;
  rowsLabel: string;
}

export const DATA_SCOPE_OPTIONS = ['trace_replay', 'synthetic_distributional', 'archived'] as const;

export const DATA_SCOPE_META: Record<DataScope, DataScopeMeta> = {
  trace_replay: {
    label: 'Trace replay',
    shortLabel: 'Trace replay',
    eyebrow: 'HF subset',
    description: 'Filtered real trace replay data aligned with the Hugging Face dataset naming.',
    accent: '#8b949e',
    rowsLabel: 'trace replay rows',
  },
  synthetic_distributional: {
    label: 'Synthetic distributional',
    shortLabel: 'Synthetic',
    eyebrow: 'APC-aware',
    description: 'Validated APC-aware synthetic replay grid. Uses synthetic-suffixed profiles derived from the active sweep state, without coding-singleturn.',
    accent: '#3fb950',
    rowsLabel: 'synthetic rows',
  },
  archived: {
    label: 'Archived',
    shortLabel: 'Archived',
    eyebrow: 'retired scopes',
    description: 'Retired canonical, fixed-grid, and MSE result scopes kept for reference but not used as the active Hugging Face dataset surface.',
    accent: '#00bcd4',
    rowsLabel: 'archived rows',
  },
};

export function isDataScope(value: string | null): value is DataScope {
  return value === 'trace_replay' || value === 'synthetic_distributional' || value === 'archived';
}

export function normalizeDataScope(value: string | null): DataScope | null {
  if (value === 'latest' || value === 'synthetic' || value === 'synthetic-distributional') return 'synthetic_distributional';
  if (value === 'archive') return 'trace_replay';
  if (value === 'current' || value === 'canonical' || value === 'fixed' || value === 'fixed-grid' || value === 'mse') return 'archived';
  return isDataScope(value) ? value : null;
}

export function hasSyntheticRuntime(scope: DataScope): boolean {
  return scope === 'synthetic_distributional';
}

export const CURRENT_PROFILES = [
  'chat-singleturn',
  'coding-singleturn',
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
] as const;

export const FIXED_PROFILES = [
  'chat-singleturn',
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
] as const;

export const SYNTHETIC_PROFILES = [
  'chat-singleturn-synth',
  'chat-multiturn-synth',
  'swebench-multiturn-synth',
  'terminalbench-multiturn-synth',
  'osworld-multiturn-synth',
] as const;

export const MSE_PROFILES = [
  'swebench-multiturn-mse',
  'swebench-multiturn-short',
  'terminalbench-multiturn-mse',
  'terminalbench-multiturn-short',
  'osworld-multiturn-mse',
  'osworld-multiturn-short',
] as const;

export const ARCHIVE_PROFILES = [
  'chat-short',
  'chat-medium',
  'chat-singleturn',
  'coding-singleturn',
  'prefill-heavy',
  'decode-heavy',
  'random-1k',
  'fixed-seq128',
  'chat-multiturn-short',
  'chat-multiturn-medium',
  'chat-multiturn-long',
  'swebench-multiturn-short',
  'swebench-multiturn-medium',
  'swebench-multiturn-long',
  'terminalbench-multiturn-short',
  'terminalbench-multiturn-medium',
  'terminalbench-multiturn-long',
  'osworld-multiturn-short',
  'osworld-multiturn-medium',
  'osworld-multiturn-long',
] as const;

export const ARCHIVED_PROFILES = [
  ...CURRENT_PROFILES,
  ...FIXED_PROFILES,
  ...MSE_PROFILES,
] as const;

const SYNTHETIC_PROFILE_SET = new Set<string>(SYNTHETIC_PROFILES);
const ARCHIVE_PROFILE_SET = new Set<string>(ARCHIVE_PROFILES);
const ARCHIVED_PROFILE_SET = new Set<string>(ARCHIVED_PROFILES);

const PROFILE_ALIASES: Record<string, string> = {
  'coding-agent': 'coding-singleturn',
  'chat-long': 'chat-singleturn',
};

export const PROFILE_META: Record<string, ProfileMeta> = {

  // Tier 1: Real Agent Data
  'coding-singleturn':            { displayName: 'coding-singleturn', workloadGroup: 'Agentic coding ST', agentType: 'coding',   turnStyle: 'single-turn', dataSource: 'SWEBench',      isl: 'med ~6.3K', osl: 'med ~280',  description: 'Single planning/model call from SWE-Bench-style coding prompts. Published runs are long-input single-turn workloads, but not ShareGPT chat.' },
  'coding-agent':                 { displayName: 'coding-singleturn', workloadGroup: 'Agentic coding ST', agentType: 'coding',   turnStyle: 'single-turn', dataSource: 'SWEBench',      isl: 'med ~6.3K', osl: 'med ~280',  description: 'Legacy profile tag for coding-singleturn.' },
  'swebench-multiturn':           { displayName: 'swebench-multiturn', workloadGroup: 'Agentic coding MT', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: 'sampled', osl: 'sampled', description: 'Canonical distributional SWE-bench multi-turn workload sampled from empirical turn-count, new-prefill, and output-token distributions.' },
  'swebench-multiturn-synth':     { displayName: 'swebench-multiturn-synth', workloadGroup: 'Synthetic coding MT', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: 'sampled', osl: 'sampled', description: 'APC-aware morphology-calibrated synthetic SWE-bench replay profile.' },
  'swebench-multiturn-mse':       { displayName: 'swebench-multiturn-mse', workloadGroup: 'MSE validation', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: '<=32K', osl: '<=2000', description: 'MSE validation synthetic SWE-bench workload filtered to match the real short-trajectory population.' },
  'swebench-multiturn-short':     { displayName: 'swebench-multiturn-short', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: 'med ~8.0K',   osl: '<=2000', description: 'Real SWE-bench agent sessions in the shorter step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'swebench-multiturn-medium':    { displayName: 'swebench-multiturn-medium', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: 'med ~13.4K',   osl: '<=2000', description: 'Real SWE-bench agent sessions in the medium step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'swebench-multiturn-long':      { displayName: 'swebench-multiturn-long', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: '<=128K',  osl: '<=2000', description: 'Long SWE-bench agent sessions. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'terminalbench-multiturn':      { displayName: 'terminalbench-multiturn', workloadGroup: 'Agentic terminal MT', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: 'sampled', osl: 'sampled', description: 'Canonical distributional TerminalBench multi-turn workload sampled from empirical turn-count, new-prefill, and output-token distributions.' },
  'terminalbench-multiturn-synth': { displayName: 'terminalbench-multiturn-synth', workloadGroup: 'Synthetic terminal MT', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: 'sampled', osl: 'sampled', description: 'APC-aware morphology-calibrated synthetic TerminalBench replay profile.' },
  'terminalbench-multiturn-mse':  { displayName: 'terminalbench-multiturn-mse', workloadGroup: 'MSE validation', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: '<=32K', osl: '<=2000', description: 'MSE validation synthetic TerminalBench workload filtered to match the real short-trajectory population.' },
  'terminalbench-multiturn-short': { displayName: 'terminalbench-multiturn-short', workloadGroup: 'Agentic terminal', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: 'med ~5.0K',   osl: '<=2000', description: 'Real TerminalBench CLI-agent sessions in the shorter step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'terminalbench-multiturn-medium': { displayName: 'terminalbench-multiturn-medium', workloadGroup: 'Agentic terminal', agentType: 'terminal', turnStyle: 'multi-turn', dataSource: 'TerminalBench', isl: 'med ~10.5K',   osl: '<=2000', description: 'Real TerminalBench CLI-agent sessions in the medium step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'terminalbench-multiturn-long':  { displayName: 'terminalbench-multiturn-long', workloadGroup: 'Agentic terminal', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: '<=128K',  osl: '<=2000', description: 'Long TerminalBench CLI-agent sessions. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },

  // Tier 2: Chat (ShareGPT, honest shape labels)
  'chat-short':                   { displayName: 'chat-short', workloadGroup: 'Legacy chat ST', benchmarkVisible: false, agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~129',   osl: 'med ~169',  description: 'Retired ShareGPT single-turn variant. It differs mostly by shorter output length, so it is hidden from the main benchmark view.' },
  'chat-medium':                  { displayName: 'chat-medium', workloadGroup: 'Legacy chat ST', benchmarkVisible: false, agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~157',  osl: 'med ~286', description: 'Retired ShareGPT single-turn variant. It overlaps heavily with chat-singleturn, so new sweeps use chat-singleturn as the canonical natural chat workload.' },
  'chat-singleturn':                    { displayName: 'chat-singleturn', workloadGroup: 'Natural chat ST', agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~187',  osl: 'med ~299', description: 'Canonical natural ShareGPT single-turn workload. This represents ordinary chat; it is not a long-context prefill stress workload.' },
  'chat-singleturn-synth':              { displayName: 'chat-singleturn-synth', workloadGroup: 'Synthetic chat ST', agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~187',  osl: 'med ~299', description: 'Synthetic-scope ShareGPT single-turn chat baseline.' },

  // Tier 3: Synthetic Stress Tests
  'prefill-heavy':                { displayName: 'prefill-heavy', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '8192',   osl: '256',   description: 'Synthetic prefill stress: long random input and short output. Use this for controlled long-context prefill behavior, not ShareGPT chat.' },
  'decode-heavy':                 { displayName: 'decode-heavy', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '256',    osl: '4096',  description: 'Synthetic decode stress: short random input and long output. Isolates sustained generation speed.' },
  'random-1k':                    { displayName: 'random-1k', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '1024',   osl: '1024',  description: 'Random-token balanced workload with ISL=1024 and OSL=1024. Kept for InferenceX-style cross-validation.' },
  'fixed-seq128':                 { displayName: 'fixed-seq128', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '128',    osl: '128',   description: 'Fixed random-token shape used for predictor validation.' },

  // Tier 4: Multi-turn Chat (ShareGPT)
  'chat-multiturn':               { displayName: 'chat-multiturn', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'sampled', osl: 'sampled', description: 'Canonical distributional ShareGPT multi-turn chat workload sampled from empirical turn-count, new-prefill, and output-token summaries.' },
  'chat-multiturn-synth':         { displayName: 'chat-multiturn-synth', workloadGroup: 'Synthetic chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'sampled', osl: 'sampled', description: 'APC-aware synthetic ShareGPT multi-turn replay profile.' },
  'chat-multiturn-short':         { displayName: 'chat-multiturn-short', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'med ~673',    osl: 'med ~298', description: 'Natural ShareGPT multi-turn chat in the shortest turn-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'chat-multiturn-medium':        { displayName: 'chat-multiturn-medium', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'med ~835',   osl: 'med ~246', description: 'Natural ShareGPT multi-turn chat in the medium turn-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'chat-multiturn-long':          { displayName: 'chat-multiturn-long', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'med ~937',   osl: 'med ~149', description: 'Natural ShareGPT multi-turn chat in the deepest current turn bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },

  // Tier 5: Computer-Use (OSWorld WebArena trajectories)
  'osworld-multiturn':            { displayName: 'osworld-multiturn', workloadGroup: 'Computer-use MT', agentType: 'computer-use', turnStyle: 'multi-turn', dataSource: 'OSWorld',     isl: 'sampled', osl: 'sampled', description: 'Canonical distributional OSWorld computer-use workload sampled from empirical turn-count, new-prefill, and output-token distributions. Use with the OSWorld trace caveat noted in paper notes.' },
  'osworld-multiturn-synth':      { displayName: 'osworld-multiturn-synth', workloadGroup: 'Synthetic computer-use MT', agentType: 'computer-use', turnStyle: 'multi-turn', dataSource: 'OSWorld',     isl: 'sampled', osl: 'sampled', description: 'APC-aware synthetic OSWorld computer-use replay profile.' },
  'osworld-multiturn-mse':        { displayName: 'osworld-multiturn-mse', workloadGroup: 'MSE validation', agentType: 'computer-use', turnStyle: 'multi-turn', dataSource: 'OSWorld',     isl: '<=32K', osl: '<=500', description: 'MSE validation synthetic OSWorld workload filtered to match the real short-trajectory population.' },
  'osworld-multiturn-short':      { displayName: 'osworld-multiturn-short', workloadGroup: 'Computer-use', agentType: 'computer-use', turnStyle: 'multi-turn', dataSource: 'OSWorld',     isl: 'med ~5.0K',   osl: '<=800',  description: 'Real OSWorld computer-use sessions in the short step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'osworld-multiturn-medium':     { displayName: 'osworld-multiturn-medium', workloadGroup: 'Computer-use', agentType: 'computer-use', turnStyle: 'multi-turn', dataSource: 'OSWorld',     isl: 'med ~4.7K',   osl: '<=1000', description: 'Real OSWorld computer-use sessions in the medium step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'osworld-multiturn-long':       { displayName: 'osworld-multiturn-long', workloadGroup: 'Computer-use', agentType: 'computer-use', turnStyle: 'multi-turn', dataSource: 'OSWorld',     isl: '<=64K',   osl: '<=1200', description: 'Longest OSWorld computer-use step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
};

// Color for agent type badges
export const AGENT_TYPE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'chat':         { bg: 'rgba(63,185,80,0.12)',   text: '#3fb950', border: 'rgba(63,185,80,0.3)' },
  'coding':       { bg: 'rgba(0,188,212,0.12)',   text: '#00bcd4', border: 'rgba(0,188,212,0.3)' },
  'terminal':     { bg: 'rgba(249,117,131,0.12)', text: '#f97583', border: 'rgba(249,117,131,0.3)' },
  'computer-use': { bg: 'rgba(236,72,153,0.12)',  text: '#ec4899', border: 'rgba(236,72,153,0.3)' },
  'stress':       { bg: 'rgba(255,152,0,0.12)',   text: '#ff9800', border: 'rgba(255,152,0,0.3)' },
};

// Color for data source badges
export const DATA_SOURCE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'ShareGPT':      { bg: 'rgba(168,85,247,0.12)',  text: '#a855f7', border: 'rgba(168,85,247,0.3)' },
  'SWEBench':      { bg: 'rgba(121,192,255,0.12)', text: '#79c0ff', border: 'rgba(121,192,255,0.3)' },
  'TerminalBench': { bg: 'rgba(255,183,77,0.12)',  text: '#ffb74d', border: 'rgba(255,183,77,0.3)' },
  'OSWorld':       { bg: 'rgba(20,184,166,0.12)',  text: '#14b8a6', border: 'rgba(20,184,166,0.3)' },
  'Random':        { bg: 'rgba(255,152,0,0.12)',   text: '#ff9800', border: 'rgba(255,152,0,0.3)' },
};

export const FALLBACK_META_COLORS = {
  bg: 'rgba(139,148,158,0.12)',
  text: '#8b949e',
  border: 'rgba(139,148,158,0.3)',
};

export function profileDisplayName(profile: string): string {
  const normalized = normalizeProfileName(profile);
  return PROFILE_META[normalized]?.displayName ?? normalized;
}

export function isBenchmarkProfile(profile: string): boolean {
  const normalized = normalizeProfileName(profile);
  return PROFILE_META[normalized]?.benchmarkVisible !== false;
}

export function normalizeProfileName(profile: string): string {
  return PROFILE_ALIASES[profile] ?? profile;
}

export function isProfileInScope(profile: string, scope: DataScope): boolean {
  const normalized = normalizeProfileName(profile);
  if (scope === 'trace_replay') {
    return ARCHIVE_PROFILE_SET.has(normalized);
  }
  if (scope === 'synthetic_distributional') {
    return SYNTHETIC_PROFILE_SET.has(normalized);
  }
  return ARCHIVED_PROFILE_SET.has(normalized);
}

export function scopeLabel(scope: DataScope): string {
  return DATA_SCOPE_META[scope].label;
}
