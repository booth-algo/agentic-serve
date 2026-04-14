export interface ProfileMeta {
  agentType: 'chat' | 'coding' | 'terminal';
  turnStyle: 'single-turn' | 'multi-turn';
  servingStyle: 'disaggregated' | 'not-disaggregated';
  dataSource: string;    // "ShareGPT", "SWEBench", "TerminalBench", "Random", "Test"
  isl: string;
  osl: string;
  description: string;
}

export const PROFILE_META: Record<string, ProfileMeta> = {

  // Tier 1: Real Agent Data
  'coding-agent':                 { agentType: 'coding',   turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'SWEBench',      isl: '~17000', osl: '~800',  description: 'Real coding-agent prompts from Sequrity SWEBench runs. PLLM planning calls with ~17K input tokens, ~800 output. Very long prefill stress test.' },
  'swebench-multiturn-short':     { agentType: 'coding',   turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'SWEBench',      isl: '≤32K',   osl: '≤2000', description: 'Real SWEBench coding agent: 13-30 step sessions (shortest). Growing context from actual Codex agent trajectories.' },
  'swebench-multiturn-medium':    { agentType: 'coding',   turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'SWEBench',      isl: '≤64K',   osl: '≤2000', description: 'Real SWEBench coding agent: 30-80 step sessions. Medium-length agent runs with command execution and file edits.' },
  'swebench-multiturn-long':      { agentType: 'coding',   turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'SWEBench',      isl: '≤128K',  osl: '≤2000', description: 'Real SWEBench coding agent: 80-150 step sessions. Long agent runs stressing KV cache with accumulated context.' },
  'swebench-multiturn-xl':        { agentType: 'coding',   turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'SWEBench',      isl: '≤128K',  osl: '≤2000', description: 'Real SWEBench coding agent: 150+ step sessions (longest). Extended runs pushing context window limits.' },
  'terminalbench-multiturn-short': { agentType: 'terminal', turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'TerminalBench', isl: '≤32K',   osl: '≤2000', description: 'Real TerminalBench CLI agent: 2-20 step sessions (shortest). Includes shell output in growing context.' },
  'terminalbench-multiturn-medium': { agentType: 'terminal', turnStyle: 'multi-turn', servingStyle: 'not-disaggregated', dataSource: 'TerminalBench', isl: '≤64K',   osl: '≤2000', description: 'Real TerminalBench CLI agent: 20-60 step sessions. Medium-length terminal agent runs.' },
  'terminalbench-multiturn-long':  { agentType: 'terminal', turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'TerminalBench', isl: '≤128K',  osl: '≤2000', description: 'Real TerminalBench CLI agent: 60-150 step sessions. Long terminal sessions with extensive command history.' },
  'terminalbench-multiturn-xl':    { agentType: 'terminal', turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'TerminalBench', isl: '≤128K',  osl: '≤2000', description: 'Real TerminalBench CLI agent: 150+ step sessions. Extended terminal sessions pushing context limits.' },

  // Tier 2: Chat (ShareGPT, honest shape labels)
  'chat-short':                   { agentType: 'chat',     turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤500',   osl: '≤300',  description: 'Short Q&A chat — most common pattern. ShareGPT conversations filtered to ≤500 input / ≤300 output tokens.' },
  'chat-medium':                  { agentType: 'chat',     turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤2000',  osl: '≤1000', description: 'Medium chat — longer conversations and detailed answers. ShareGPT filtered to ≤2K input / ≤1K output tokens.' },
  'chat-long':                    { agentType: 'chat',     turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤8000',  osl: '≤2000', description: 'Long chat — longest natural ShareGPT conversations. Filtered to ≤8K input / ≤2K output tokens.' },

  // Tier 3: Synthetic Stress Tests
  'prefill-heavy':                { agentType: 'chat',     turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'Random',        isl: '8192',   osl: '256',   description: 'Synthetic prefill stress: long input (8192 tokens), short output (256 tokens). Random tokens, no prefix caching. Isolates prefill performance.' },
  'decode-heavy':                 { agentType: 'chat',     turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'Random',        isl: '256',    osl: '4096',  description: 'Synthetic decode stress: short input (256 tokens), long output (4096 tokens). Random tokens. Isolates sustained token generation speed.' },
  'random-1k':                    { agentType: 'chat',     turnStyle: 'single-turn', servingStyle: 'not-disaggregated', dataSource: 'Random',        isl: '1024',   osl: '1024',  description: 'InferenceX cross-validation: random tokens ISL=1024, OSL=1024. For comparing TPOT with InferenceX benchmarks.' },

  // Tier 4: Multi-turn Chat (ShareGPT)
  'chat-multiturn-short':         { agentType: 'chat',     turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤8K',    osl: '≤1000', description: 'ShareGPT multi-turn: 3-5 turns per session. Tests KV cache reuse with moderate growing context.' },
  'chat-multiturn-medium':        { agentType: 'chat',     turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤16K',   osl: '≤1500', description: 'ShareGPT multi-turn: 5-10 turns per session. Larger growing context, more KV cache pressure.' },
  'chat-multiturn-long':          { agentType: 'chat',     turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤32K',   osl: '≤2000', description: 'ShareGPT multi-turn: 10-20 turns per session. Deep KV cache stress with large accumulated history.' },
  'chat-multiturn-xl':            { agentType: 'chat',     turnStyle: 'multi-turn',  servingStyle: 'not-disaggregated', dataSource: 'ShareGPT',      isl: '≤64K',   osl: '≤2000', description: 'ShareGPT multi-turn: 20-30 turns per session. Extreme context length stress testing.' },
};

// Color for agent type badges
export const AGENT_TYPE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'chat':     { bg: 'rgba(63,185,80,0.12)',   text: '#3fb950', border: 'rgba(63,185,80,0.3)' },
  'coding':   { bg: 'rgba(0,188,212,0.12)',   text: '#00bcd4', border: 'rgba(0,188,212,0.3)' },
  'terminal': { bg: 'rgba(249,117,131,0.12)', text: '#f97583', border: 'rgba(249,117,131,0.3)' },
};

// Color for data source badges
export const DATA_SOURCE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'ShareGPT':      { bg: 'rgba(168,85,247,0.12)',  text: '#a855f7', border: 'rgba(168,85,247,0.3)' },
  'SWEBench':      { bg: 'rgba(121,192,255,0.12)', text: '#79c0ff', border: 'rgba(121,192,255,0.3)' },
  'TerminalBench': { bg: 'rgba(255,183,77,0.12)',  text: '#ffb74d', border: 'rgba(255,183,77,0.3)' },
  'Random':        { bg: 'rgba(255,152,0,0.12)',   text: '#ff9800', border: 'rgba(255,152,0,0.3)' },
};
