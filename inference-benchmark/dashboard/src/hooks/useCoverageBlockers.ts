import { useEffect, useState } from 'react';
import { coverageBlockersJsonUrl } from '../dataUrls';
import type { CoverageBlockersState } from '../types-coverage-blockers';

export function useCoverageBlockers(enabled = true) {
  const [blockersState, setBlockersState] = useState<CoverageBlockersState | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setBlockersState(null);
      setLoading(false);
      setError(null);
      return;
    }

    let active = true;
    let controller: AbortController | null = null;

    const load = () => {
      controller?.abort();
      controller = new AbortController();
      const separator = coverageBlockersJsonUrl.includes('?') ? '&' : '?';
      fetch(`${coverageBlockersJsonUrl}${separator}_=${Date.now()}`, {
        cache: 'no-store',
        signal: controller.signal,
      })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((json: CoverageBlockersState) => {
          if (!active) return;
          setBlockersState(json);
          setError(null);
          setLoading(false);
        })
        .catch((err) => {
          if (!active || err.name === 'AbortError') return;
          setError(err.message);
          setLoading(false);
        });
    };

    setLoading(true);
    load();
    const interval = window.setInterval(load, 30_000);
    return () => {
      active = false;
      window.clearInterval(interval);
      controller?.abort();
    };
  }, [enabled]);

  return { blockersState, loading, error };
}
