import { useEffect, useState } from 'react';
import type { SweepState } from '../types-sweep';

declare const __BUILD_HASH__: string;

const R2_URL = 'https://pub-38e30ed030784867856634f1625c7130.r2.dev/sweep-state.json';

export function useSweepState() {
  const [sweepState, setSweepState] = useState<SweepState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${R2_URL}?v=${__BUILD_HASH__}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: SweepState) => {
        setSweepState(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { sweepState, loading, error };
}
