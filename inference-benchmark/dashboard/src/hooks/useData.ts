import { useState, useEffect, useMemo, useCallback } from 'react';
import type { BenchmarkResult, FilterState, FilterOptions } from '../types';
import { PROFILE_META } from '../profileMeta';

declare const __BUILD_HASH__: string;

export function useData() {
  const [allData, setAllData] = useState<BenchmarkResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    hardware: [],
    model: [],
    backend: [],
    agentType: [],
    turnStyle: [],
    servingStyle: [],
    profile: [],
  });

  useEffect(() => {
    // Cache-bust with build-time hash so deploys always serve fresh data
    fetch(`https://pub-38e30ed030784867856634f1625c7130.r2.dev/data.json?v=${__BUILD_HASH__}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: BenchmarkResult[]) => {
        setAllData(data);
        // Default to first hardware config to avoid chart clutter
        const hwSet = new Set(data.map((r) => r.hardware));
        const firstHw = Array.from(hwSet).sort()[0];
        if (firstHw) {
          setFilters((prev) => ({ ...prev, hardware: [firstHw] }));
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const filterOptions = useMemo<FilterOptions>(() => {
    const hw = new Set<string>();
    const model = new Set<string>();
    const backend = new Set<string>();
    const agentType = new Set<string>();
    const turnStyle = new Set<string>();
    const servingStyle = new Set<string>();
    const profile = new Set<string>();

    for (const r of allData) {
      hw.add(r.hardware);
      model.add(r.modelShort);
      backend.add(r.config.backend);
      profile.add(r.config.profile);
      const meta = PROFILE_META[r.config.profile];
      if (meta) {
        agentType.add(meta.agentType);
        turnStyle.add(meta.turnStyle);
        servingStyle.add(meta.servingStyle);
      }
    }

    return {
      hardware: Array.from(hw).sort(),
      model: Array.from(model).sort(),
      backend: Array.from(backend).sort(),
      agentType: Array.from(agentType).sort(),
      turnStyle: Array.from(turnStyle).sort(),
      servingStyle: Array.from(servingStyle).sort(),
      profile: Array.from(profile).sort(),
    };
  }, [allData]);

  const filteredData = useMemo(() => {
    return allData.filter((r) => {
      if (filters.hardware.length > 0 && !filters.hardware.includes(r.hardware)) return false;
      if (filters.model.length > 0 && !filters.model.includes(r.modelShort)) return false;
      if (filters.backend.length > 0 && !filters.backend.includes(r.config.backend)) return false;
      if (filters.profile.length > 0 && !filters.profile.includes(r.config.profile)) return false;

      // Tag-based filtering via profile metadata
      const meta = PROFILE_META[r.config.profile];
      if (meta) {
        if (filters.agentType.length > 0 && !filters.agentType.includes(meta.agentType)) return false;
        if (filters.turnStyle.length > 0 && !filters.turnStyle.includes(meta.turnStyle)) return false;
        if (filters.servingStyle.length > 0 && !filters.servingStyle.includes(meta.servingStyle)) return false;
      } else {
        // If no metadata, exclude when tag filters are active
        if (filters.agentType.length > 0 || filters.turnStyle.length > 0 || filters.servingStyle.length > 0) return false;
      }

      return true;
    });
  }, [allData, filters]);

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
    setFilters({ hardware: [], model: [], backend: [], agentType: [], turnStyle: [], servingStyle: [], profile: [] });
  }, []);

  return {
    allData,
    data: filteredData,
    seriesData,
    loading,
    error,
    filters,
    filterOptions,
    toggleFilter,
    clearFilters,
  };
}
