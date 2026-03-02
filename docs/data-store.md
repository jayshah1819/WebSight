# Data Store

**Source:** `src/02-data-store.js`  
**Export:** `profilerData` — a single shared object imported by every module that needs to read or write profiler state.

---

## What it does

`profilerData` is the central state store for the entire profiler. There is one instance per page — all modules share the same reference. No copying, no message passing, no sync needed between modules on the same thread.

Every dispatch recorded by Module 10, every buffer created, every kernel seen, every log line — it all lands here. Module 11 reads from it to build every public API response.

---

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `dispatches` | `Array` | One entry per `dispatchWorkgroups` or draw call recorded. Each entry holds timing, workgroup geometry, buffer accesses, and occupancy analysis. |
| `pipelines` | `Object` | Keyed by pipeline label. Metadata accumulated at `createComputePipeline` time — shader source, workgroup size, entry point. |
| `bindGroups` | `Object` | Metadata about bind groups created by the app. Keyed by internal UUID. |
| `buffers` | `Object` | Keyed by internal UUID. Every `GPUBuffer` created, with size, label, and access pattern data. |
| `timingHelper` | `null \| object` | Legacy field — per-device TimingHelper instances are now managed in Module 10. Kept for backward compatibility. |
| `logs` | `Array` | Profiler event log. Each entry is `{ time, message }`. Writable via `addLog()` in Module 8. |
| `gpuCharacteristics` | `null \| object` | Adapter/device limits and features populated by Module 10 on first device acquisition. Used by the workgroup analyzer for occupancy scoring. |
| `bufferHeatMap` | `Object` | Keyed by buffer UUID. Tracks how many times each buffer was bound per dispatch — used by the bandwidth analysis to find hotspot buffers. |
| `runId` | `null \| string` | UUID identifying the current benchmark run. Set by Module 10 on first dispatch, reset by `clear()`. |
| `kernels` | `Object` | Keyed by pipeline label. Accumulated per-kernel stats: `count`, `totalTime`, `minTime`, `maxTime`, `avgTime`, `shaderId`. |
| `runs` | `Object` | Keyed by `runId`. Aggregated run-level stats for multi-run benchmark comparisons. |
| `timingMode` | `string` | Legacy global timing mode: `'unknown'`, `'cpu'`, or `'gpu'`. Module 10 reads per-device `timingMode` from `device.__webSightInfo` instead. Updated by `configure()`, `benchmarkMode()`, and `normalMode()` for backward compatibility. |
| `sessionStart` | `number` | `Date.now()` at module load time. Used to compute session duration in reports. |
| `totalKernelTime` | `number` | Running sum of GPU time across all dispatches. Incremented by Module 10 on each timing result. |
| `memoryUsage` | `object` | `{ peak, current, allocations[] }` — maintained by the memory leak detector, not this module directly. Kept here for snapshot export. |
| `activeEncoders` | `WeakMap` | Tracks in-flight `GPUCommandEncoder` instances. `WeakMap` so finished encoders are collected automatically without manual cleanup. |

---

## `config` sub-object

Runtime configuration. All fields are readable and writable, but prefer `WebSight.configure()` for any field that has a side effect (timing mode propagation, analyzer reconfig, etc.).

| Field | Default | Description |
|---|---|---|
| `broadcastEnabled` | `true` | Whether Module 7 sends `profiler-update` messages over `BroadcastChannel`. Disable for pure in-page use or benchmark mode to eliminate serialization overhead. |
| `broadcastDebounceMs` | `3000` | Minimum ms between broadcasts. 3 s default prevents UI thrashing during fast benchmark loops. Set to `10000` in minimal-overhead mode. |
| `normalizeTimeUnit` | `'us'` | Time unit for `normalizeTime()` output: `'ns'`, `'us'`, or `'ms'`. |
| `verboseLogging` | `false` | Enables extra `console.log` output from hooks. Useful for debugging missed dispatches. |
| `minimalOverhead` | `false` | When `true`, GPU timing is disabled and broadcasts are suppressed. Set by `benchmarkMode()`. |
| `enableMemoryLeakDetection` | `false` | Whether Module 3 runs leak checks. Off by default — leak detection adds per-resource bookkeeping overhead. |
| `enableWorkgroupAnalysis` | `true` | Whether Module 4 scores every dispatch for occupancy. On by default — the cost is small and the data is useful. |
| `enableShaderAnalysis` | `false` | Whether shaders are analyzed for complexity at `createShaderModule` time. Off by default — WGSL parsing is not free. |
| `captureStacks` | `false` | Whether JS call stacks are captured at `trackResource` time. Only useful for hunting down which call site is leaking. |
| `peakMemoryBandwidthGBs` | `null` | Device peak memory bandwidth. Must be set by the caller via `configure()`. Used to compute memory efficiency percentage in `getBandwidthAnalysis()`. |
| `simdWidth` | `32` | Logical SIMD width used by the workgroup analyzer's occupancy scoring. Override if running on a known non-32-wide architecture. |
| `memoryLeakThresholdMs` | `10000` | Passed to `memoryLeakDetector.leakThreshold` on init. Resources alive longer than this many ms are classified as potential leaks. |
| `memoryWarningThresholdMB` | `100` | Passed to `memoryLeakDetector.sizeThreshold`. `checkForLeaks()` emits an extra warning if `currentMemory` exceeds this. |

---

## Design notes

### Single shared reference

Every module imports `profilerData` directly:

```js
import { profilerData } from './02-data-store.js';
```

Because ES module imports are live bindings to the same object, all modules operate on the same reference — no copying or syncing required. Mutations made in Module 10 are immediately visible to Module 11.

### `activeEncoders` is a `WeakMap`

Command encoders are tracked while in flight (between `createCommandEncoder` and `queue.submit`). Using `WeakMap` means finished encoders are garbage collected automatically without any explicit cleanup, and there is no risk of the profiler holding stale encoder references after a submit.

For the same reason, `activeEncoders` is re-initialized to a fresh `WeakMap` on every `clear()` call — clearing a `WeakMap` is not possible directly.

### `timingMode` vs per-device `timingMode`

The top-level `profilerData.timingMode` is a legacy field kept for broadcast payload compatibility (Module 7 includes it in the snapshot so the dashboard can display it). Module 10 does **not** read it. The timing mode Module 10 actually consults is `device.__webSightInfo.timingMode`, set per-device when the device is acquired. `configure()`, `benchmarkMode()`, and `normalMode()` update both.

---

## Data flow

```
02-data-store.js
  └─ profilerData (shared object)
        │
        ├─ written by Module 10 (hooks)
        │     createBuffer      →  buffers[uuid]
        │     createTexture     →  buffers[uuid]
        │     createComputePipeline → pipelines[label], kernels[label]
        │     createCommandEncoder  → activeEncoders.set(encoder, meta)
        │     queue.submit      →  dispatches[], kernels[].stats, runs[]
        │
        ├─ written by Module 3 (memory leak detector)
        │     trackResource     →  memoryUsage.current, .peak
        │
        ├─ read by Module 7 (broadcast)
        │     scheduleBroadcast →  serializes dispatches, kernels, buffers, runs
        │
        └─ read by Module 11 (public API)
              getData()         →  returns profilerData directly
              getStats()        →  aggregates dispatches[]
              getGraphData()    →  groups dispatches[] by kernel label
              getBufferData()   →  iterates buffers{}
              getDispatchList() →  slices and formats dispatches[]
              clear()           →  resets all mutable fields
```

---

## Cleared on `WebSight.clear()`

| Field | Reset to |
|---|---|
| `dispatches` | `[]` |
| `logs` | `[]` |
| `kernels` | `{}` |
| `buffers` | `{}` |
| `bufferHeatMap` | `{}` |
| `activeEncoders` | `new Map()` |
| `pipelines` | `{}` |

`config`, `gpuCharacteristics`, `sessionStart`, `memoryUsage`, and `timingMode` are preserved across `clear()` — they describe the device and session, not the profile data.
