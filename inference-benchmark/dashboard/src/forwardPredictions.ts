// Forward predictor output (simulator.forward, no-GT path) joined per
// (gpu_key, model, profile, concurrency) against the backtester's serving rows. Shared by the
// Predictions matrix and the Simulator page so both join the same way.

export interface ForwardRow {
  model?: string;
  profile?: string;
  concurrency?: number;
  fwd_tpot_pred?: number;
  fwd_ttft_pred?: number;
  fwd_e2el_pred?: number;
  fwd_tpot_err?: number | null;
  fwd_ttft_err?: number | null;
  fwd_e2el_err?: number | null;
}

export type FwdLookup = Map<string, ForwardRow>;

// The composite key is internal to the lookup Map (built and read only via this helper), so the
// separator is arbitrary. Models/profiles are dashed identifiers and concurrency is numeric, so a
// space separator cannot shift a field boundary into a collision.
export function fwdKey(gpuKey: string, model: string, profile: string, conc: number | string): string {
  return [gpuKey, model, profile, conc].join(' ');
}

export function buildFwdLookup(json: Record<string, ForwardRow[]> | null): FwdLookup {
  const m: FwdLookup = new Map();
  if (!json) return m;
  for (const [gpuKey, rows] of Object.entries(json)) {
    for (const r of rows) {
      if (r.model && r.profile != null && r.concurrency != null) {
        m.set(fwdKey(gpuKey, r.model, r.profile, r.concurrency), r);
      }
    }
  }
  return m;
}
