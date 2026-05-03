import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import type { BenchmarkResult, FilterState, FilterOptions } from '../types';
import {
  PROFILE_META,
  type DataScope,
  isProfileInScope,
  normalizeProfileName,
} from '../profileMeta';
import { dataJsonUrl } from '../dataUrls';

interface UseDataOptions {
  deriveBenchmarkData?: boolean;
}

const EMPTY_FILTER_OPTIONS: FilterOptions = {
  hardware: [],
  model: [],
  backend: [],
  agentType: [],
  turnStyle: [],
  profile: [],
};

const EMPTY_SERIES_DATA = new Map<string, BenchmarkResult[]>();

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
  const initialDataScope = useRef<DataScope>(dataScope);
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
    fetch(dataJsonUrl)
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
        // Default within the active scope to avoid filtering current rows with
        // archive-only labels such as plain "H100".
        const scopedForDefault = normalized.filter(
          (r) => (r.dataScope ?? 'archive') === initialDataScope.current
            && isProfileInScope(r.config.profile, initialDataScope.current),
        );
        const defaultHardware = defaultHardwareForScope(scopedForDefault);
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

  const scopedDataByScope = useMemo<Record<DataScope, BenchmarkResult[]>>(() => {
    const next: Record<DataScope, BenchmarkResult[]> = { current: [], archive: [], fixed: [] };
    for (const row of allData) {
      const scope = row.dataScope ?? 'archive';
      if (isProfileInScope(row.config.profile, scope)) next[scope].push(row);
    }
    return next;
  }, [allData]);

  const scopedData = scopedDataByScope[dataScope];

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
