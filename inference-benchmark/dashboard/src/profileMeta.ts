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

export const PROFILE_META: Record<string, ProfileMeta> = {

  // Tier 1: Real Agent Data
  'coding-agent':                 { displayName: 'coding-agent', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'single-turn', dataSource: 'SWEBench',      isl: 'med ~6.3K', osl: 'med ~280',  description: 'Single planning/model call from SWE-Bench-style coding prompts. Published runs are long-input single-turn workloads, but not ShareGPT chat.' },
  'swebench-multiturn-short':     { displayName: 'swebench-multiturn-short', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: 'med ~8.0K',   osl: '<=2000', description: 'Real SWE-bench coding-agent sessions in the shorter step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'swebench-multiturn-medium':    { displayName: 'swebench-multiturn-medium', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: 'med ~13.4K',   osl: '<=2000', description: 'Real SWE-bench coding-agent sessions in the medium step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'swebench-multiturn-long':      { displayName: 'swebench-multiturn-long', workloadGroup: 'Agentic coding', agentType: 'coding',   turnStyle: 'multi-turn',  dataSource: 'SWEBench',      isl: '<=128K',  osl: '<=2000', description: 'Long SWE-bench coding-agent sessions. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'terminalbench-multiturn-short': { displayName: 'terminalbench-multiturn-short', workloadGroup: 'Agentic terminal', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: 'med ~5.0K',   osl: '<=2000', description: 'Real TerminalBench CLI-agent sessions in the shorter step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'terminalbench-multiturn-medium': { displayName: 'terminalbench-multiturn-medium', workloadGroup: 'Agentic terminal', agentType: 'terminal', turnStyle: 'multi-turn', dataSource: 'TerminalBench', isl: 'med ~10.5K',   osl: '<=2000', description: 'Real TerminalBench CLI-agent sessions in the medium step-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'terminalbench-multiturn-long':  { displayName: 'terminalbench-multiturn-long', workloadGroup: 'Agentic terminal', agentType: 'terminal', turnStyle: 'multi-turn',  dataSource: 'TerminalBench', isl: '<=128K',  osl: '<=2000', description: 'Long TerminalBench CLI-agent sessions. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },

  // Tier 2: Chat (ShareGPT, honest shape labels)
  'chat-short':                   { displayName: 'chat-short', workloadGroup: 'Legacy chat ST', benchmarkVisible: false, agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~129',   osl: 'med ~169',  description: 'Retired ShareGPT single-turn variant. It differs mostly by shorter output length, so it is hidden from the main benchmark view.' },
  'chat-medium':                  { displayName: 'chat-medium', workloadGroup: 'Legacy chat ST', benchmarkVisible: false, agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~157',  osl: 'med ~286', description: 'Retired ShareGPT single-turn variant. It overlaps heavily with chat-singleturn, so new sweeps use chat-singleturn as the canonical natural chat workload.' },
  'chat-singleturn':                    { displayName: 'chat-singleturn', workloadGroup: 'Natural chat ST', agentType: 'chat',     turnStyle: 'single-turn', dataSource: 'ShareGPT',      isl: 'med ~187',  osl: 'med ~299', description: 'Canonical natural ShareGPT single-turn workload. This represents ordinary chat; it is not a long-context prefill stress workload.' },

  // Tier 3: Synthetic Stress Tests
  'prefill-heavy':                { displayName: 'prefill-heavy', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '8192',   osl: '256',   description: 'Synthetic prefill stress: long random input and short output. Use this for controlled long-context prefill behavior, not ShareGPT chat.' },
  'decode-heavy':                 { displayName: 'decode-heavy', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '256',    osl: '4096',  description: 'Synthetic decode stress: short random input and long output. Isolates sustained generation speed.' },
  'random-1k':                    { displayName: 'random-1k', workloadGroup: 'Stress', agentType: 'stress',     turnStyle: 'single-turn', dataSource: 'Random',        isl: '1024',   osl: '1024',  description: 'Random-token balanced workload with ISL=1024 and OSL=1024. Kept for InferenceX-style cross-validation.' },

  // Tier 4: Multi-turn Chat (ShareGPT)
  'chat-multiturn-short':         { displayName: 'chat-multiturn-short', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'med ~673',    osl: 'med ~298', description: 'Natural ShareGPT multi-turn chat in the shortest turn-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'chat-multiturn-medium':        { displayName: 'chat-multiturn-medium', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'med ~835',   osl: 'med ~246', description: 'Natural ShareGPT multi-turn chat in the medium turn-depth bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },
  'chat-multiturn-long':          { displayName: 'chat-multiturn-long', workloadGroup: 'Natural chat MT', agentType: 'chat',     turnStyle: 'multi-turn',  dataSource: 'ShareGPT',      isl: 'med ~937',   osl: 'med ~149', description: 'Natural ShareGPT multi-turn chat in the deepest current turn bucket. Short/medium/long denote turn depth, not monotonic ISL or OSL.' },

  // Tier 5: Computer-Use (OSWorld WebArena trajectories)
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
  return profile;
}

export function isBenchmarkProfile(profile: string): boolean {
  return PROFILE_META[profile]?.benchmarkVisible !== false;
}
