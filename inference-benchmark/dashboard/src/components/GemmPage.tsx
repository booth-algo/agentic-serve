import { useState, useEffect } from 'react';
import { gemmEvalJsonUrl } from '../dataUrls';

interface GemmPoint {
  M: number; N: number; K: number;
  measured: number; predicted: number; roofline: number;
}
interface ModelResult {
  model: string; status: string;
  n_shapes?: number; mape?: number; roofline_mape?: number;
  points?: { M: number; N: number; K: number; measured_us: number; predicted_us: number; roofline_us: number }[];
}
interface GpuData {
  n_shapes: number; roofline_mape: number; xgb_mape: number;
  model_results: ModelResult[];
  scatter: GemmPoint[];
}
interface GemmEval {
  generated_at: string;
  gpus: Record<string, GpuData>;
}

function mapeColor(v: number): string {
  if (v < 5) return '#3fb950';
  if (v < 10) return '#58a6ff';
  if (v < 20) return '#ff9800';
  return '#f85149';
}

export function GemmPage() {
  const [data, setData] = useState<GemmEval | null>(null);
  const [gpu, setGpu] = useState('H100');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(gemmEvalJsonUrl)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-[#8b949e] p-8">Loading GEMM evaluation data...</div>;
  if (!data) return <div className="text-[#f85149] p-8">Failed to load gemm-eval.json</div>;

  const gpus = Object.keys(data.gpus);
  const gpuData = data.gpus[gpu];
  if (!gpuData) return null;

  const testedModels = gpuData.model_results.filter(m => m.status === 'tested' && m.mape !== undefined);
  const sharedModels = gpuData.model_results.filter(m => m.status === 'shared');

  const scatterMax = Math.max(...gpuData.scatter.map(s => Math.max(s.measured, s.predicted)));

  return (
    <div className="space-y-6">
      {/* Explainer */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] px-5 py-4 text-sm text-[#c9d1d9]">
        <h3 className="text-base font-semibold text-[#e6edf3] mb-2">Per-Kernel GEMM Predictor</h3>
        <p className="mb-2">
          In transformer inference, <strong className="text-[#e6edf3]">GEMM (matrix multiply) kernels account for
          92–97% of decode compute time</strong> and 60–80% of prefill time. Predicting GEMM latency accurately
          is the single most important driver for end-to-end serving latency prediction.
        </p>
        <p className="mb-2">
          We profile each GPU's cuBLAS/cuBLASLt GEMM dispatch by sweeping <code className="text-[#79c0ff] bg-[#21262d] px-1 rounded">nn.Linear</code> across
          the exact (M, N, K) shapes that LLM serving emits — QKV projections, FFN gate/up/down, and LM head.
          An XGBoost model learns the residual between a roofline baseline and measured ncu kernel time:
        </p>
        <div className="bg-[#0d1117] rounded px-3 py-2 font-mono text-xs text-[#8b949e] mb-2">
          prediction = roofline(M, N, K) × exp(XGBoost_residual(log₂M, log₂N, log₂K, OI, log(roofline)))
        </div>
        <p className="text-[#8b949e] text-xs">
          Evaluation: leave-one-model-out — train on all models except one, predict the held-out model's unique GEMM shapes.
          Models sharing identical shapes with training models are marked "shared" (no extrapolation test needed).
        </p>
      </div>

      {/* GPU selector */}
      <div className="flex gap-2">
        {gpus.map(g => (
          <button
            key={g}
            onClick={() => setGpu(g)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              g === gpu
                ? 'bg-[#1f6feb] text-white'
                : 'bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3]'
            }`}
          >
            {g}
          </button>
        ))}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <SummaryCard label="Shapes Profiled" value={gpuData.n_shapes.toString()} />
        <SummaryCard label="Roofline-Only MAPE" value={`${gpuData.roofline_mape}%`} color="#f85149" />
        <SummaryCard label="XGBoost In-Sample MAPE" value={`${gpuData.xgb_mape}%`} color="#3fb950" />
        <SummaryCard
          label="Leave-One-Model-Out Median MAPE"
          value={`${median(testedModels.map(m => m.mape!)).toFixed(1)}%`}
          color="#58a6ff"
        />
      </div>

      {/* Two-column: roofline vs xgb scatter + model bar chart */}
      <div className="grid grid-cols-2 gap-4">
        {/* Roofline vs XGBoost comparison scatter */}
        <div className="rounded-lg border border-[#21262d] bg-[#0d1117] p-4">
          <h4 className="text-sm font-semibold text-[#e6edf3] mb-1">Roofline vs XGBoost+Roofline — Predicted vs Measured</h4>
          <p className="text-xs text-[#484f58] mb-3">Each dot is one GEMM shape. Closer to the diagonal = better prediction.</p>
          <svg viewBox="0 0 400 380" className="w-full">
            {/* Perfect diagonal */}
            <line x1="50" y1="340" x2="380" y2="10" stroke="#30363d" strokeWidth="1" strokeDasharray="4" />
            {/* Axes */}
            <line x1="50" y1="340" x2="380" y2="340" stroke="#484f58" strokeWidth="1" />
            <line x1="50" y1="340" x2="50" y2="10" stroke="#484f58" strokeWidth="1" />
            {/* Roofline points (red, behind) */}
            {gpuData.scatter.map((s, i) => {
              const logMax = Math.log10(Math.max(scatterMax, 10));
              const x = 50 + (Math.log10(Math.max(s.measured, 0.1)) / logMax) * 330;
              const y = 340 - (Math.log10(Math.max(s.roofline, 0.1)) / logMax) * 330;
              return <circle key={`r${i}`} cx={x} cy={y} r="2.5" fill="#f85149" opacity="0.35" />;
            })}
            {/* XGBoost points (blue, on top) */}
            {gpuData.scatter.map((s, i) => {
              const logMax = Math.log10(Math.max(scatterMax, 10));
              const x = 50 + (Math.log10(Math.max(s.measured, 0.1)) / logMax) * 330;
              const y = 340 - (Math.log10(Math.max(s.predicted, 0.1)) / logMax) * 330;
              return <circle key={`x${i}`} cx={x} cy={y} r="2.5" fill="#58a6ff" opacity="0.6" />;
            })}
            {/* Legend */}
            <circle cx="70" cy="20" r="4" fill="#f85149" opacity="0.5" />
            <text x="80" y="24" fill="#f85149" fontSize="10">Roofline only ({gpuData.roofline_mape}% MAPE)</text>
            <circle cx="230" cy="20" r="4" fill="#58a6ff" opacity="0.7" />
            <text x="240" y="24" fill="#58a6ff" fontSize="10">XGBoost+Roofline ({gpuData.xgb_mape}% MAPE)</text>
            {/* Axis labels */}
            <text x="215" y="370" fill="#8b949e" fontSize="11" textAnchor="middle">Measured (μs, log scale)</text>
            <text x="15" y="175" fill="#8b949e" fontSize="11" textAnchor="middle" transform="rotate(-90 15 175)">Predicted (μs, log scale)</text>
          </svg>
        </div>

        {/* Leave-one-model-out bar chart */}
        <div className="rounded-lg border border-[#21262d] bg-[#0d1117] p-4 min-h-[380px]">
          <h4 className="text-sm font-semibold text-[#e6edf3] mb-3">Leave-One-Model-Out MAPE</h4>
          <div className="space-y-2">
            {testedModels.sort((a, b) => (b.mape ?? 0) - (a.mape ?? 0)).map(m => (
              <div key={m.model} className="flex items-center gap-2">
                <span className="text-xs text-[#8b949e] w-28 truncate text-right">{m.model}</span>
                <div className="flex-1 bg-[#21262d] rounded-full h-5 relative">
                  <div
                    className="h-5 rounded-full flex items-center justify-end pr-2"
                    style={{
                      width: `${Math.min((m.mape ?? 0) / 50 * 100, 100)}%`,
                      backgroundColor: mapeColor(m.mape ?? 0),
                      opacity: 0.8,
                      minWidth: '40px',
                    }}
                  >
                    <span className="text-[10px] font-bold text-white">{m.mape?.toFixed(1)}%</span>
                  </div>
                </div>
                <span className="text-[10px] text-[#484f58] w-16">{m.n_shapes} shapes</span>
              </div>
            ))}
            {sharedModels.map(m => (
              <div key={m.model} className="flex items-center gap-2">
                <span className="text-xs text-[#8b949e] w-28 truncate text-right">{m.model}</span>
                <div className="flex-1">
                  <span className="text-xs text-[#484f58] italic ml-2">shared shapes — no extrapolation needed</span>
                </div>
              </div>
            ))}
          </div>
          {/* Roofline comparison line */}
          <div className="mt-3 pt-2 border-t border-[#21262d] flex items-center gap-2">
            <span className="text-xs text-[#8b949e] w-28 text-right">Roofline only</span>
            <div className="flex-1 bg-[#21262d] rounded-full h-5 relative">
              <div
                className="h-5 rounded-full flex items-center justify-end pr-2"
                style={{
                  width: `${Math.min(gpuData.roofline_mape / 50 * 100, 100)}%`,
                  backgroundColor: '#f85149',
                  opacity: 0.5,
                  minWidth: '60px',
                }}
              >
                <span className="text-[10px] font-bold text-white">{gpuData.roofline_mape}%</span>
              </div>
            </div>
            <span className="text-[10px] text-[#484f58] w-16">baseline</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-[#21262d] bg-[#161b22] px-4 py-3">
      <div className="text-xs text-[#8b949e]">{label}</div>
      <div className="text-2xl font-bold mt-1" style={{ color: color ?? '#e6edf3' }}>{value}</div>
    </div>
  );
}

function median(arr: number[]): number {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}
