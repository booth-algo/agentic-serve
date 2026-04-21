import { useEffect, useState } from 'react';
import type { ProfilingState } from '../types-profiling';

declare const __BUILD_HASH__: string;

const PROFILING_URL = '/profiling-state.json';

export function useProfilingState() {
  const [profilingState, setProfilingState] = useState<ProfilingState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${PROFILING_URL}?v=${__BUILD_HASH__}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: ProfilingState) => {
        setProfilingState(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { profilingState, loading, error };
}
