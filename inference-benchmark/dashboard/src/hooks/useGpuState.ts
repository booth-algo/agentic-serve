import { useEffect, useState } from 'react';
import { gpuStateJsonUrl } from '../dataUrls';
import type { GpuState } from '../types-gpu-state';

export function useGpuState() {
  const [gpuState, setGpuState] = useState<GpuState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(gpuStateJsonUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: GpuState) => {
        setGpuState(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { gpuState, loading, error };
}
