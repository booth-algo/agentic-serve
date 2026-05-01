import { useEffect, useState } from 'react';
import type { SweepState } from '../types-sweep';
import { sweepStateUrl } from '../dataUrls';

export function useSweepState() {
  const [sweepState, setSweepState] = useState<SweepState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(sweepStateUrl)
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
