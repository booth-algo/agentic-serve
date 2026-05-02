import { useEffect, useState } from 'react';
import { profilingStateJsonUrl } from '../dataUrls';
import type { ProfilingState } from '../types-profiling';

export function useProfilingState() {
  const [profilingState, setProfilingState] = useState<ProfilingState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(profilingStateJsonUrl)
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
