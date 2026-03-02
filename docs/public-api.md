# Module 11 — Public API (`11-public-api.js`)

All methods are on `window.WebSight`. Available after `profiler-standalone.js` loads.

---

## Control

### `WebSight.start()`
Installs all WebGPU hooks. Called automatically on page load unless `window.__webSightIsUIWindow` is set. Safe to call manually if auto-start is disabled.

### `WebSight.clear()`
Resets `dispatches`, `logs`, `kernels`, `buffers`, `bufferHeatMap`, `activeEncoders`, and `pipelines` to empty. Does not remove GPU hooks.

### `WebSight.configure(options)` → `config`
Updates runtime configuration. Returns the full applied config object.

| Option | Type | Effect |
|---|---|---|
| `broadcastEnabled` | `boolean` | Enable / disable BroadcastChannel pushes to the dashboard |
| `broadcastDebounceMs` | `number` | Debounce interval for broadcasts (ms) |
| `timeUnit` | `'ns' \| 'us' \| 'ms'` | Unit used by `normalizeTime` / `listKernels` |
| `minimalOverhead` | `boolean` | `true` → CPU-only timing on all devices, broadcast interval 10 s |
| `peakMemoryBandwidthGBs` | `number` | Device peak bandwidth; used to compute memory efficiency % |
| `simdWidth` | `number` | SIMD lane width propagated to workgroup analyzer |
| `leakThresholdMs` | `number` | Age (ms) after which an un-destroyed resource is flagged as a leak |

### `WebSight.benchmarkMode()`
Disables GPU timing and broadcasts, sets all per-device `timingMode` to `'cpu-only'`. Minimal profiler overhead for accurate application benchmarks.

### `WebSight.normalMode()`
Re-enables GPU timing and broadcasts, sets all per-device `timingMode` to `'gpu'`.

---

## Data access

### `WebSight.getData()` → `profilerData`
Returns the raw internal data store object directly. All other methods derive from this.

### `WebSight.getStats()` → `object`
Aggregated overview stats.

```js
{
  totalDispatches,        // number — total recorded dispatches/draws
  gpuTimedDispatches,     // number — dispatches with a real GPU timestamp
  cpuFallbackDispatches,  // number — dispatches that fell back to CPU timing
  avgGpuTime,             // ms
  totalGpuTime,           // ms
  minGpuTime,             // ms
  maxGpuTime,             // ms
  timeUnit: 'ms',
  totalCpuTimeMs,
  totalKernelTimeMs,
  totalBufferSizeBytes,
  bandwidth: {
    totalBytesRead, totalBytesWritten, totalBytes,
    avgBandwidthGBs, peakBandwidthGBs,
    totalBytesReadFormatted, totalBytesWrittenFormatted, totalBytesFormatted
  }
}
```

### `WebSight.listKernels()` → `KernelSummary[]`
One entry per unique kernel (pipeline + workgroup size combination).

```js
{
  id,             // string UUID
  label,          // string
  workgroupSize,  // [x, y, z] or { x, y, z }
  dispatchCount,  // number
  avgTime,        // normalized (respects timeUnit config)
  totalTime,
  minTime,
  maxTime,
  timeUnit        // 'ns' | 'us' | 'ms'
}
```

### `WebSight.getDispatchList()` → `DispatchEntry[]`
Last 20 dispatches/draws, newest first, filtered to those with `cpuTimeMs >= 0.001 ms`.

```js
{
  title,          // e.g. "Compute #42", "Draw Call #7"
  pipelineLabel,  // string
  workgroupInfo,  // e.g. "Dispatch: 64×1×1 | Workgroup: 256×1×1"
  timeDisplay     // e.g. "0.123 ms" or "0.456 ms (CPU fallback)"
}
```

### `WebSight.getMultiGPUStats()` → `object`
Snapshot of all adapters, devices, and `TimingHelper` pool metrics.

```js
{
  adapters: [{ index, hasTimestampFeature, powerPreference, deviceCount, requestedAt }],
  devices:  [{ index, label, hasTimestampQuery, timingMode, encoderCount,
               passCount, dispatchCount, features, limits, createdAt }],
  pools:    [{ deviceLabel, poolSize, maxSize, available, inUse, missedCount,
               failed, limitReached, mode, utilizationPercent }],
  totals:   { adapterCount, deviceCount, totalEncoders, totalPasses, totalDispatches }
}
```

---

## Graph / chart data

### `WebSight.getGraphData()` → `object`
Bandwidth vs input-size series, grouped by kernel label, for GPU- and CPU-timed dispatches.

```js
{
  kernels: [{
    label,
    gpu: { inputSizes, bandwidth, times },
    cpu: { inputSizes, bandwidth, times }
  }],
  gpu: { inputSizes, bandwidth, times },  // aggregate across all kernels
  cpu: { inputSizes, bandwidth, times },
  hasGpuTiming  // boolean
}
```

### `WebSight.getKernelGraphData()` → `KernelTimeSeries[]`
Per-kernel dispatch-over-time series (GPU-timed dispatches only).

```js
{
  id, label,
  count,       // total dispatch count
  avgTimeMs,
  dispatches: [{ index, timeMs }]
}
```

### `WebSight.getBufferData()` → `object`
Buffer sizes classified into four categories for stacked-bar charts.

```js
{
  labels,                   // string[] — buffer labels
  series: { input, output, atomic, uniform },  // number[] in KB
  warnings: [{ label, sizeFormatted, percentOfMax }],
  maxBufferSizeFormatted
}
```

### `WebSight.getAtomicContentionData()` → `{ x, y }`
Scatter data: `x` = dispatch index, `y` = threads-per-atomic-bin (contention proxy). Only dispatches with small atomic/histogram buffers and a recorded occupancy analysis are included.

---

## Analysis

### `WebSight.getWorkgroupSummaryData()` → `object`
Groups dispatches by `pipelineLabel|workgroupSize` key. Keeps the worst-scoring occupancy analysis per group.

```js
{
  groups: [{
    pipeline, workgroupSize, dispatchSize,
    minWGs, maxWGs,
    analysis: { score, issues },
    count,
    dimensionViolation  // boolean
  }],
  totalConfigs,
  goodConfigs,    // score >= 80 and no dimension violation
  failedConfigs   // dimension violations blocked by profiler
}
```

### `WebSight.getWorkgroupAnalysis()` → `{ summary, analyses }`
Full workgroup occupancy report. Prints a formatted report to the console and returns:

```js
{
  summary: {
    totalKernels, averageScore, grade,
    criticalIssues, highIssues,
    needsAttention: [{ kernelId, score, workgroupSize, dispatchSize,
                       totalThreads, totalWorkgroups, totalInvocations, issues }]
  },
  analyses  // Map of all individual analyses from workgroupAnalyzer
}
```

### `WebSight.getShaderAnalysis()` → `{ summary, analyses }`
Runs `shaderAnalyzer.analyzeShader` on any kernel not yet analyzed, then prints a complexity report to the console.

```js
{
  summary: {
    totalShaders, averageScore, overallGrade: { letter, desc },
    averageComplexity, criticalIssues,
    needsOptimization: [{
      shaderId, score, grade, complexity, lineCount,
      metrics: { branches, loops, mathOps, memoryOps, atomicOps, variableCount },
      issues
    }]
  },
  analyses  // Map of all individual shader analyses
}
```

### `WebSight.analyzeShader(shaderId)` → `analysis | null`
Retrieves and prints the stored analysis for a specific shader UUID. Returns `null` if no analysis exists for that ID.

```js
{
  score, grade: { letter },
  complexity, lineCount,
  issues: [{ severity, type, message, impact, recommendation, example? }]
}
```

### `WebSight.getBandwidthAnalysis()` → `object | null`
Prints a full bandwidth report to the console. Returns `null` if no GPU-timed dispatches with bandwidth data exist.

```js
{
  overall: {
    totalBytesRead, totalBytesWritten, totalBytes,
    totalTimeMs, avgBandwidthGBs, peakBandwidthGBs,
    memoryEfficiency,   // null if peakMemoryBandwidthGBs not configured
    peakMemoryBandwidthGBs
  },
  kernels: [{
    label, count, totalBytes, totalTimeMs,
    avgBandwidthGBs, peakBandwidthGBs,
    memoryPatterns: string[]
  }],
  memoryBound,   // kernels above 50% of peak (or > 200 GB/s)
  computeBound   // kernels below 20% of peak (or < 100 GB/s)
}
```

### `WebSight.getMemoryLeaks()` → `LeakReport`
Calls `memoryLeakDetector.checkForLeaks()`, prints a report, and returns the full leak report object (see [memory-leak.md](memory-leak.md) for shape).

### `WebSight.getMemoryStats()` → `stats`
Raw `memoryLeakDetector.stats` object: `{ createdCount, destroyedCount, currentMemory, peakMemory }`.

### `WebSight.getFullAnalysisReport()` → `object`
Runs memory, workgroup, and shader summaries and prints a combined console report.

```js
{
  memory,    // LeakReport from memoryLeakDetector.getLeakReport()
  workgroup, // workgroupAnalyzer.getSummary()
  shader,    // shaderAnalyzer.getSummary()
  timestamp  // Date.now()
}
```

---

## Utilities

### `WebSight.formatBytes(bytes)` → `string`
Human-readable byte size (e.g. `"4.00 MB"`). Delegates to `memoryLeakDetector.formatBytes`.

### `WebSight.formatTime(ms)` → `string`
Human-readable duration. Delegates to `memoryLeakDetector.formatTime`.

### `WebSight.export()`
Serializes the full `profilerData` to JSON and triggers a browser download as `websight-profile-<timestamp>.json`.

### `WebSight.getTimingHelperStats()` → `object`
Legacy stub — returns `{ message: 'Each encoder now has its own TimingHelper. Stats are per-encoder.' }`.

---

## Auto-launch behaviour

On `window load`:

- If `profilerData.gpuCharacteristics.limits` is set, calls `workgroupAnalyzer.setDeviceLimits`.
- Unless `window.__webSightDisableAutoUI` or `window.__webSightIsUIWindow` is set, opens `index.html` in a new 1400×900 popup window (`WebSightProfiler`).

Immediately after module evaluation:

- Unless `window.__webSightIsUIWindow` is set, calls `hookWebGPU()` automatically.

### Control flags (set before loading the script)

| Flag | Effect |
|---|---|
| `window.__webSightIsUIWindow = true` | Marks this page as the dashboard; skips auto-hook and auto-popup |
| `window.__webSightDisableAutoUI = true` | Prevents the profiler popup from opening |
| `window.__webSightDisableGPUTiming = true` | Forces CPU-only timing on every device (read by Module 10) |
