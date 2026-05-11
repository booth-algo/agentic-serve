import { useState, useEffect, useMemo, useCallback } from 'react';
import type { BenchmarkResult, FilterState, FilterOptions } from '../types';
import {
  PROFILE_META,
  type DataScope,
  isProfileInScope,
  normalizeProfileName,
  normalizeDataScope,
} from '../profileMeta';
import { dataJsonUrl, dataJsonUrlForScope } from '../dataUrls';

interface UseDataOptions {
  deriveBenchmarkData?: boolean;
  enabled?: boolean;
}

const dataCache: Partial<Record<DataScope, BenchmarkResult[]>> = {};

const EMPTY_FILTER_OPTIONS: FilterOptions = {
  hardware: [],
  model: [],
  backend: [],
  agentType: [],
  turnStyle: [],
  profile: [],
};

const EMPTY_SERIES_DATA = new Map<string, BenchmarkResult[]>();

async function fetchBenchmarkRows(dataScope: DataScope): Promise<BenchmarkResult[]> {
  const scopedResponse = await fetch(dataJsonUrlForScope(dataScope));
  if (scopedResponse.ok) return (await scopedResponse.json()) as BenchmarkResult[];
  if (scopedResponse.status !== 404) throw new Error(`HTTP ${scopedResponse.status}`);

  const aggregateResponse = await fetch(dataJsonUrl);
  if (!aggregateResponse.ok) throw new Error(`HTTP ${aggregateResponse.status}`);
  return (await aggregateResponse.json()) as BenchmarkResult[];
}

function normalizeRows(rows: BenchmarkResult[], targetScope: DataScope): BenchmarkResult[] {
  return rows
    .map((r) => {
      const profile = normalizeProfileName(r.config.profile);
      const dataScope = normalizeDataScope(r.dataScope ?? null) ?? 'trace_replay';
      if (profile === r.config.profile && dataScope === r.dataScope) return r;
      return {
        ...r,
        config: { ...r.config, profile },
        seriesKey: `${r.hardware} / ${r.modelShort} ${r.quant} / ${r.config.backend} / ${profile}`,
        dataScope,
      };
    })
    .filter((row) => row.dataScope === targetScope && isProfileInScope(row.config.profile, targetScope));
}

function defaultHardwareForScope(rows: BenchmarkResult[]): string | undefined {
  const hardware = Array.from(new Set(rows.map((r) => r.hardware))).sort();
  const preferred = [
    'H100x8',
    'H100x4',
    'H100x2',
    'H100',
    'A100-40GBx4',
    'A100-40GBx2',
    'A100-40GB',
  ];
  return preferred.find((label) => hardware.includes(label)) ?? hardware[0];
}

export function useData(dataScope: DataScope, options: UseDataOptions = {}) {
  const deriveBenchmarkData = options.deriveBenchmarkData ?? true;
  const enabled = options.enabled ?? true;
  const [allData, setAllData] = useState<BenchmarkResult[]>([]);
  const [loading, setLoading] = useState(enabled);
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
    if (!enabled) {
      setAllData([]);
      setLoading(false);
      setError(null);
      return;
    }

    const cached = dataCache[dataScope];
    if (cached) {
      setAllData(cached);
      setError(null);
      setLoading(false);
      const defaultHardware = defaultHardwareForScope(cached);
      setFilters((prev) => ({ ...prev, hardware: defaultHardware ? [defaultHardware] : [] }));
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchBenchmarkRows(dataScope)
      .then((data: BenchmarkResult[]) => {
        const normalized = normalizeRows(data, dataScope);
        if (cancelled) return;
        dataCache[dataScope] = normalized;
        setAllData(normalized);
        const defaultHardware = defaultHardwareForScope(normalized);
        setFilters((prev) => ({ ...prev, hardware: defaultHardware ? [defaultHardware] : [] }));
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataScope, enabled]);

  const scopedData = useMemo(() => {
    if (!enabled) return [];
    return allData.filter((row) => {
      const scope = normalizeDataScope(row.dataScope ?? null) ?? dataScope;
      return scope === dataScope && isProfileInScope(row.config.profile, dataScope);
    });
  }, [allData, dataScope, enabled]);

  const filterOptions = useMemo<FilterOptions>(() => {
    if (!deriveBenchmarkData) return EMPTY_FILTER_OPTIONS;

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
  }, [deriveBenchmarkData, scopedData]);

  const filteredData = useMemo(() => {
    if (!deriveBenchmarkData) return [];

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
  }, [deriveBenchmarkData, scopedData, filters]);

  // Group data by series key for chart rendering
  const seriesData = useMemo(() => {
    if (!deriveBenchmarkData) return EMPTY_SERIES_DATA;

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
  }, [deriveBenchmarkData, filteredData]);

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
    setFilters({ hardware: [], model: [], backend: [], agentType: [], turnStyle: [], profile: [] });
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
