# Workgroup Occupancy Analyzer

**Source:** `src/04-workgroup-analyzer.js`  
**Export:** `WorkgroupOccupancyAnalyzer` class  
**Singleton:** `workgroupAnalyzer` — instantiated in `src/06-init-logging.js`, used by Modules 10 and 11.

---

## What it does

Every `dispatchWorkgroups` call has a geometry: workgroup size × dispatch size. Choosing the wrong geometry is one of the most common WebGPU performance mistakes — too few threads per workgroup leaves execution units idle, workgroup sizes that aren't multiples of the SIMD width waste lanes silently, and dispatch dimensions that exceed device limits crash the kernel outright.

The Workgroup Occupancy Analyzer checks each dispatch against the device limits and a set of heuristic rules, then produces a structured analysis with a score from 0 to 100 and a list of issues with severity, impact, and a concrete fix.

Module 10 calls `analyzeDispatch` on every `dispatchWorkgroups`. Module 11 reads the cached analyses to build the workgroup summary table in the dashboard and to power `getWorkgroupAnalysis()`.

---

## How it integrates

### Device limits

The analyzer needs device limits before it can score anything. Module 10 sets them once, after the first adapter/device is acquired:

```js
workgroupAnalyzer.setDeviceLimits(device.limits);
```

Until `setDeviceLimits` is called, `analyzeDispatch` returns `null` and logs a warning.

### Per-dispatch analysis

Module 10 calls this immediately before or after `dispatchWorkgroups`:

```js
workgroupAnalyzer.analyzeDispatch({
  kernelId:      string,
  workgroupSize: [x, y, z],
  dispatchSize:  [x, y, z]
});
```

The analysis is stored in `this.analyses` keyed by `kernelId`. Only the **worst-scoring** analysis per kernel is kept — so the dashboard always surfaces the most problematic geometry, not a lucky run.

---

## Scoring

Every analysis starts at **100** and has points deducted by rule. Final score is clamped to `[0, 100]`.

| Rule | Severity | Penalty |
|---|---|---|
| Any dispatch dimension exceeds `maxComputeWorkgroupsPerDimension` | critical | Score set to 0 |
| Workgroup thread count exceeds `maxComputeInvocationsPerWorkgroup` | critical | −50 |
| Workgroup threads < 64 (not a small utility dispatch) | high | −30 |
| Workgroup threads not a multiple of SIMD width | medium | −20 |
| Workgroup dimensions not powers of 2 | low | −10 |
| 1D dispatch with multi-dimensional workgroup | low | −5 |
| Small utility dispatch (< 256 workgroups, 8–63 threads) | info | Score set to 70 |

Small utility dispatches — histogram finalization, prefix sum tail, reduction final pass — are handled specially. They are common, intentionally small, and not a bug. The rule recognises them and scores them 70 (acceptable) rather than penalising them for low thread count.

---

## Issue structure

Each issue in `analysis.issues` follows this shape:

```js
{
  severity:       'critical' | 'high' | 'medium' | 'low' | 'info',
  type:           string,   // machine-readable issue code
  message:        string,   // human-readable description
  impact:         string,   // what goes wrong
  recommendation: string    // concrete fix
}
```

---

## Analysis output

`analyzeDispatch` returns an analysis object:

```js
{
  kernelId:       string,
  workgroupSize:  [x, y, z],
  dispatchSize:   [x, y, z],
  totalThreads:   number,      // wg[0] * wg[1] * wg[2]
  totalWorkgroups: number,     // ds[0] * ds[1] * ds[2]
  totalInvocations: number,    // totalThreads * totalWorkgroups
  score:          number,      // 0–100
  issues:         Issue[],
  recommendations: object[],
  workgroupUtilization: {
    total:      number,
    xDim:       number,
    yDim:       number,
    zDim:       number,
    xPercent:   number,        // (xDim / maxPerDim) * 100
    yPercent:   number,
    zPercent:   number,
    maxPerDim:  number,
    xExceeds:   boolean,
    yExceeds:   boolean,
    zExceeds:   boolean
  }
}
```

---

## API reference

### `setDeviceLimits(limits)`

Must be called before any `analyzeDispatch` call. `limits` is the `GPUSupportedLimits` object from the device. Extracts `maxComputeInvocationsPerWorkgroup`, `maxComputeWorkgroupsPerDimension`, and `simdWidth` (with a fallback of 32 if the browser does not expose it).

---

### `analyzeDispatch(dispatch) → analysis | null`

Scores a single dispatch geometry. Returns `null` if device limits have not been set or if `workgroupSize`/`dispatchSize` are missing. Stores the worst-scoring result per `kernelId` in `this.analyses`.

| Parameter | Type | Description |
|---|---|---|
| `dispatch.kernelId` | `string` | Pipeline label or UUID identifying the kernel |
| `dispatch.workgroupSize` | `Array \| object \| number` | Workgroup dimensions — normalized by `_normalizeDims` |
| `dispatch.dispatchSize` | `Array \| object \| number` | Dispatch grid dimensions — normalized by `_normalizeDims` |

---

### `getAllAnalyses() → analysis[]`

Returns all stored analyses as a flat array. One entry per unique `kernelId`.

---

### `getSummary() → object | null`

Aggregates across all stored analyses. Returns `null` if no analyses exist yet.

```js
{
  totalKernels:    number,
  averageScore:    string,   // toFixed(1)
  grade:           'A' | 'B' | 'C' | 'D' | 'F',
  criticalIssues:  number,   // kernels with at least one critical issue
  highIssues:      number,   // kernels with at least one high-severity issue
  needsAttention:  analysis[] // score < 70
}
```

Grade thresholds: A ≥ 90, B ≥ 80, C ≥ 65, D ≥ 50, F < 50.

---

### `_normalizeDims(dims) → [x, y, z]`

Accepts an array, an `{x, y, z}` object, or a plain number. Returns a 3-element array with each dimension floored at 1. Handles `null`/`undefined` gracefully — returns `[1, 1, 1]`.

---

### `roundUpToMultiple(value, multiple) → number`

Rounds `value` up to the next multiple of `multiple`. Used in issue recommendations to suggest the nearest valid SIMD-aligned workgroup size.

---

### `nearestPowerOf2(value) → number`

Rounds `value` to the nearest power of 2. Used in issue recommendations to suggest the nearest power-of-2 workgroup dimension.

---

## Usage via the public API

| `WebSight` method | What it calls |
|---|---|
| `getWorkgroupAnalysis()` | `getSummary()` + `getAllAnalyses()` — logs full report, returns both |
| `getWorkgroupSummaryData()` | Reads `dispatch.occupancyAnalysis` stored by Module 10 — builds dashboard table rows |

Note: `getWorkgroupSummaryData` reads analyses that were attached to dispatch records by Module 10 at dispatch time (`d.occupancyAnalysis`), not directly from `workgroupAnalyzer.analyses`. This means it reflects the geometry at the time of each dispatch, while `getWorkgroupAnalysis()` reflects only the worst-case stored result per kernel.

---

## Known limitations

| Scenario | Behaviour |
|---|---|
| Device limits not yet set | `analyzeDispatch` returns `null`, no analysis stored |
| Render pass draw calls | Not analyzed — only compute dispatches have workgroup geometry |
| Dynamic workgroup sizes (spec extension) | Not supported — analyzer assumes static workgroup sizes from pipeline creation |
| Multi-dimensional occupancy modeling | Not implemented — score is heuristic, not a hardware occupancy calculator |
| `simdWidth` device feature | Falls back to 32 if not exposed; actual SIMD width varies by vendor and driver |

---

## Data flow

```
10-webgpu-hooks.js
  dispatchWorkgroups(x, y, z)
    └─ workgroupAnalyzer.analyzeDispatch({ kernelId, workgroupSize, dispatchSize })
         └─ analyses.set(kernelId, worst)   ←─ only if new score < existing score
              │
              └─ result attached to dispatch record as d.occupancyAnalysis

11-public-api.js
  getWorkgroupAnalysis()
    └─ workgroupAnalyzer.getSummary()
    └─ workgroupAnalyzer.getAllAnalyses()

  getWorkgroupSummaryData()
    └─ profilerData.dispatches[].occupancyAnalysis   ← per-dispatch snapshot
```
