# Dashboard — GitHub Pages Publishing

## Overview

The reporting dashboard is a React + TypeScript + Vite app at `dashboard/`.
It's deployed to GitHub Pages at `https://sequrity-ai.github.io/inference-benchmark/`.

## Tech Stack

- **React 19** + **TypeScript 5.8** + **Vite 6**
- **Tailwind CSS 4** for styling
- **Recharts 3** for charts (latency, throughput, comparison, per-turn)
- Base path: `/inference-benchmark/` (set in `vite.config.ts`)

## Key Files

| File | Purpose |
|------|---------|
| `dashboard/scripts/build-data.ts` | Reads `results/**/*.json`, enriches with hardware/quant metadata, writes `public/data.json` |
| `dashboard/public/data.json` | Pre-built data bundle consumed by the frontend |
| `dashboard/src/App.tsx` | Main app with filters, charts, tables |
| `dashboard/src/components/` | Filters, KPICards, DataTable, Layout, Tabs |
| `dashboard/src/components/charts/` | LatencyChart, ThroughputChart, ComparisonChart, PerTurnChart |
| `dashboard/src/hooks/useData.ts` | Data loading hook |
| `dashboard/src/profileMeta.ts` | Profile display names and metadata |
| `dashboard/src/types.ts` | TypeScript interfaces |

## Build & Deploy Workflow

### 1. Rebuild data from results
```bash
cd dashboard && npx tsx scripts/build-data.ts
```
This reads all `results/**/*.json`, enriches each entry (hardware detection, quant, model short name, series key), and writes `dashboard/public/data.json`.

### 2. Build the site
```bash
cd dashboard && npm run build
# Output: dashboard/dist/
```

### 3. Deploy to GitHub Pages
```bash
# Commit data.json and push
git add dashboard/public/data.json
git commit -m 'Update benchmark data'
git push

# GitHub Actions deploys from dist/ to Pages (if configured)
# Or manual: push dist/ contents to gh-pages branch
```

## Local Development

```bash
cd dashboard
npm install
npm run dev     # http://localhost:5173/inference-benchmark/
```

## Data Pipeline Summary

```
results/*.json → build-data.ts → public/data.json → Vite build → GitHub Pages
```
