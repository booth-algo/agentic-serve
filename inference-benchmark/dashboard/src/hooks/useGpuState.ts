import { useEffect, useState } from 'react';
import { gpuStateJsonUrl } from '../dataUrls';
import type { GpuState } from '../types-gpu-state';

export function useGpuState() {
  const [gpuState, setGpuState] = useState<GpuState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;

    const load = () => {
      controller?.abort();
      controller = new AbortController();
      const separator = gpuStateJsonUrl.includes('?') ? '&' : '?';
      fetch(`${gpuStateJsonUrl}${separator}_=${Date.now()}`, {
        cache: 'no-store',
        signal: controller.signal,
      })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((json: GpuState) => {
          if (!active) return;
          setGpuState(json);
          setError(null);
          setLoading(false);
        })
        .catch((err) => {
          if (!active || err.name === 'AbortError') return;
          setError(err.message);
          setLoading(false);
        });
    };

    load();
    const interval = window.setInterval(load, 30_000);
    return () => {
      active = false;
      window.clearInterval(interval);
      controller?.abort();
    };
  }, []);

  return { gpuState, loading, error };
}
