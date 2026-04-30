import { useState, useEffect, useMemo, useCallback } from 'react';
import type { BenchmarkResult, FilterState, FilterOptions } from '../types';
import {
  PROFILE_META,
  type DataScope,
  isProfileInScope,
  normalizeProfileName,
} from '../profileMeta';

declare const __BUILD_HASH__: string;

const R2_JSON_BASE = 'https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current';

export function useData(dataScope: DataScope) {
  const [allData, setAllData] = useState<BenchmarkResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    hardware: [],
    model: [],
    backend: [],
    agentType: [],
    turnStyle: [],
    profile: [],
  });

  useEffect(() => {
    // Cache-bust with build-time hash so deploys always serve fresh data
    fetch(`${R2_JSON_BASE}/data.json?v=${__BUILD_HASH__}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: BenchmarkResult[]) => {
        const normalized = data.map((r) => {
          const profile = normalizeProfileName(r.config.profile);
          const dataScope = r.dataScope ?? 'archive';
          if (profile === r.config.profile && dataScope === r.dataScope) return r;
          return {
            ...r,
            config: { ...r.config, profile },
            seriesKey: `${r.hardware} / ${r.modelShort} ${r.quant} / ${r.config.backend} / ${profile}`,
            dataScope,
          };
        });
        setAllData(normalized);
        // Default to first hardware config to avoid chart clutter
        const hwSet = new Set(normalized.map((r) => r.hardware));
        const sortedHardware = Array.from(hwSet).sort();
        const defaultHardware = sortedHardware.includes('H100') ? 'H100' : sortedHardware[0];
        if (defaultHardware) {
          setFilters((prev) => ({ ...prev, hardware: [defaultHardware] }));
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const scopedData = useMemo(
    () => allData.filter((r) => (r.dataScope ?? 'archive') === dataScope && isProfileInScope(r.config.profile, dataScope)),
    [allData, dataScope],
  );

  const filterOptions = useMemo<FilterOptions>(() => {
    const hw = new Set<string>();
    const model = new Set<string>();
    const backend = new Set<string>();
    const agentType = new Set<string>();
    const turnStyle = new Set<string>();
    const profile = new Set<string>();

    for (const r of scopedData) {
      hw.add(r.hardware);
      model.add(r.modelShort);
      backend.add(r.config.backend);
      profile.add(r.config.profile);
      const meta = PROFILE_META[r.config.profile];
      if (meta) {
        agentType.add(meta.agentType);
        turnStyle.add(meta.turnStyle);
      }
    }

    return {
      hardware: Array.from(hw).sort(),
      model: Array.from(model).sort(),
      backend: Array.from(backend).sort(),
      agentType: Array.from(agentType).sort(),
      turnStyle: Array.from(turnStyle).sort(),
      profile: Array.from(profile).sort(),
    };
  }, [scopedData]);

  const filteredData = useMemo(() => {
    return scopedData.filter((r) => {
      if (filters.hardware.length > 0 && !filters.hardware.includes(r.hardware)) return false;
      if (filters.model.length > 0 && !filters.model.includes(r.modelShort)) return false;
      if (filters.backend.length > 0 && !filters.backend.includes(r.config.backend)) return false;
      if (filters.profile.length > 0 && !filters.profile.includes(r.config.profile)) return false;

      // Tag-based filtering via profile metadata
      const meta = PROFILE_META[r.config.profile];
      if (meta) {
        if (filters.agentType.length > 0 && !filters.agentType.includes(meta.agentType)) return false;
        if (filters.turnStyle.length > 0 && !filters.turnStyle.includes(meta.turnStyle)) return false;
      } else {
        if (filters.agentType.length > 0 || filters.turnStyle.length > 0) return false;
      }

      return true;
    });
  }, [scopedData, filters]);

  // Group data by series key for chart rendering
  const seriesData = useMemo(() => {
    const map = new Map<string, BenchmarkResult[]>();
    for (const r of filteredData) {
      const existing = map.get(r.seriesKey) || [];
      existing.push(r);
      map.set(r.seriesKey, existing);
    }
    // Sort each series by concurrency
    for (const [, arr] of map) {
      arr.sort((a, b) => a.config.concurrency - b.config.concurrency);
    }
    return map;
  }, [filteredData]);

  const toggleFilter = useCallback((category: keyof FilterState, value: string) => {
    setFilters((prev) => {
      const arr = prev[category];
      const next = arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value];
      return { ...prev, [category]: next };
    });
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({ hardware: [], model: [], backend: [], agentType: [], turnStyle: [], profile: [] });
  }, []);

  const clearWorkloadFilters = useCallback(() => {
    setFilters((prev) => ({ ...prev, agentType: [], turnStyle: [], profile: [] }));
  }, []);

  return {
    allData: scopedData,
    data: filteredData,
    seriesData,
    loading,
    error,
    filters,
    filterOptions,
    toggleFilter,
    clearFilters,
    clearWorkloadFilters,
  };
}
