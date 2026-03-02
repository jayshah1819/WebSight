# Module 10 — WebGPU API Hooks (`10-webgpu-hooks.js`)

## What it does

Module 10 is the interception layer. It monkey-patches every relevant WebGPU entry point at runtime so the profiler can observe GPU operations without requiring any changes to application code.

Two functions are exported:

- **`hookWebGPU()`** — installs all hooks. Must be called once, before the application calls `navigator.gpu.requestAdapter`.
- **`getMultiGPUStats()`** — returns a structured snapshot of every adapter, device, and timing-helper pool that has been created since `hookWebGPU` ran.

---

## How it integrates

```
Application code
      │
      ▼
 hookWebGPU()               ← one-time install, guarded by __webSightHooked
      │
      ├── HTMLCanvasElement.prototype.getContext
      │       └── context.getCurrentTexture          → memoryLeakDetector
      │
      ├── navigator.gpu.requestAdapter
      │       └── adapter.requestDevice
      │               ├── device.createShaderModule    → module.__source / __shaderId
      │               ├── device.createComputePipeline
      │               │   device.createComputePipelineAsync
      │               │       └── setupPipeline()      → profilerData.pipelines / .kernels
      │               ├── device.createBuffer          → profilerData.buffers + memoryLeakDetector
      │               ├── device.createTexture         → texture.__websight_metadata + memoryLeakDetector
      │               ├── device.createBindGroupLayout → layout.__websight_entries
      │               ├── device.createBindGroup       → bg.__websight_resources / .__capture
      │               └── device.createCommandEncoder
      │                       ├── encoder.beginComputePass
      │                       │       └── pass.dispatchWorkgroups
      │                       │               ├── dimension-violation check
      │                       │               ├── workgroupAnalyzer.analyzeDispatch
      │                       │               ├── calculateDispatchBandwidth
      │                       │               └── profilerData.dispatches.push
      │                       ├── encoder.beginRenderPass
      │                       │       ├── pass.draw
      │                       │       └── pass.drawIndexed
      │                       │               └── recordDrawCall() → profilerData.dispatches.push
      │                       ├── encoder.finish       → commandBuffer.__passTimings
      │                       └── device.queue.submit
      │                               └── onSubmittedWorkDone → GPU timing resolution
      │
      └── broadcastData()   ← called after every dispatch / draw / timing resolution
```

---

## Window globals

`hookWebGPU` writes several globals that persist for the page lifetime.

| Global | Type | Purpose |
|---|---|---|
| `__webSightHooked` (on `navigator.gpu`) | `boolean` | Re-entry guard; prevents double-hooking |
| `window.__webSightAdapters` | `AdapterInfo[]` | One entry per `requestAdapter` call |
| `window.__webSightDevices` | `DeviceInfo[]` | One entry per `requestDevice` call |
| `window.__webSightTimingHelperPools` | `Map<GPUDevice, Pool>` | Per-device `TimingHelper` object pools |
| `window.__webSightGlobalTimingResults` | `number[]` | Rolling buffer of pass durations (nanoseconds); capped at `__webSightMaxTimingResults` (10,000) |
| `window.__webSightMaxTimingResults` | `number` | Cap for the above buffer |
| `window.__WebSightTimingHelper` | `{ getResult, reset }` | Compatibility shim consumed by `primitive.mjs` benchmarking harness |
| `window.__webSightTimingEvents` | `EventTarget` | Fires a `'timing'` `CustomEvent` after each submit resolves |

Pre-flags (set by application before loading the profiler):

| Flag | Effect |
|---|---|
| `window.__webSightDisableGPUTiming = true` | Skips `timestamp-query` feature request; forces `'cpu-only'` mode on every device |

---

## Hook installation walkthrough

### Canvas texture tracking

`HTMLCanvasElement.prototype.getContext` is wrapped once. When it returns a `'webgpu'` context, `context.getCurrentTexture` is replaced with a version that:

1. Checks whether the returned texture has already been tagged (`__websight_id`).
2. If the context previously held a different canvas texture (`context.__lastCanvasTexture`), calls `memoryLeakDetector.markDestroyed` on it — handles implicit destruction on canvas resize.
3. Tags the new texture with `__websight_id`, `__websight_metadata`, and registers it with the memory leak detector (assumed 4 bytes/pixel RGBA).
4. Wraps `texture.createView` so every view inherits `__websight_texture` pointing back to the metadata.

### Adapter hook

`navigator.gpu.requestAdapter` is replaced. On success it stores an `AdapterInfo` record (including `hasTimestampFeature`) in `window.__webSightAdapters` and immediately wraps `adapter.requestDevice`.

### Device hook

`adapter.requestDevice` is replaced. It:

1. Unconditionally adds `'timestamp-query'` to `requiredFeatures` when the adapter supports it (and the pre-flag is not set).
2. Determines `deviceTimingMode`: `'gpu'` when `timestamp-query` was granted, otherwise `'cpu-only'`.
3. Stores a `DeviceInfo` record in `window.__webSightDevices` — this is the object that `11-public-api.js` reads when propagating `benchmarkMode`/`normalMode` changes.
4. Calls `workgroupAnalyzer.setDeviceLimits(device.limits)` the first time a device is created.
5. Listens for `uncapturederror` to catch late-breaking QuerySet allocation failures and demote the device to `'cpu-only'`.
6. Installs all subsequent per-device hooks described below.

### `setupPipeline(pipeline, desc, defaultLabel)` — shared helper

Called by both `createComputePipeline` and `createComputePipelineAsync`. Extracts from the pipeline descriptor:

- WGSL source (`desc.compute.module.__source`)
- Entry point (`desc.compute.entryPoint`)
- Workgroup size via `extractWorkgroupSize` (Module 08)
- WGSL complexity analysis via `analyzeWGSL` (Module 08)

Stores a `pipeline.__capture` object and registers the pipeline in `profilerData.pipelines`. Registers the kernel in `profilerData.kernels` if not already present (initialises `stats` with sentinel values `minTime: Infinity`, `maxTime: 0`).

Also wraps `pipeline.getBindGroupLayout` to populate `layout.__websight_entries` from `parseWGSLBindings` (Module 09) when the application fetches a layout reflectively.

### Buffer hook (`createBuffer`)

Assigns a UUID, stores `{ id, label, size, usage }` in `profilerData.buffers`, registers the buffer with the memory leak detector (skipping `TimingHelper`-internal buffers by label prefix), and wraps `buffer.destroy` to call `markDestroyed`.

### Texture hook (`createTexture`)

Assigns a UUID, stores full dimension/format metadata in `texture.__websight_metadata`, wraps `createView` to propagate the metadata to views, calculates byte size via the same format table used by `calculateTextureSize` (Module 09), registers with the memory leak detector, and wraps `destroy`.

### Bind group layout hook (`createBindGroupLayout`)

Iterates `desc.entries` and stores per-binding visibility, buffer type, and texture/storageTexture flags in `layout.__websight_entries`. This is the ground truth used by the bind group hook below.

### Bind group hook (`createBindGroup`)

For each entry:

- **Buffer resource:** determines `accessType` (`'read-only'` vs `'read-write'`) by preferring `layout.__websight_entries[binding].bufferType`; falls back to `GPUBufferUsage` flags.
- **Texture resource:** determines `accessType` from `hasStorageTexture` in the layout entry; calls `calculateTextureSize` for byte size.

Stores normalised resource descriptors in `bg.__websight_resources` (consumed by bandwidth tracking) and `bg.__capture.entries` (debug snapshot).

### Command encoder hook (`createCommandEncoder`)

Creates an `encoderData` record (dispatches list, start time, UUID) stored in `profilerData.activeEncoders`. Builds a `proxyEncoder` shim exposing only the methods `TimingHelper` needs. Then wraps `beginComputePass`, `beginRenderPass`, and `finish`.

---

## Pass hooks

### `beginComputePass`

Optionally acquires a `TimingHelper` from the per-device pool. Initialises:

- `pass.__dispatches` — list of dispatch records for this pass
- `pass.__boundPipeline` / `pass.__boundBindGroups` — state shadow
- `pass.__bandwidthTracker` (new `BandwidthTracker` instance) / `pass.__bandwidthSnapshot`

Wraps `pass.setPipeline`, `pass.setBindGroup`, and `pass.dispatchWorkgroups`.

#### `dispatchWorkgroups` hook

1. **Dimension violation check** — compares `x/y/z` against `device.limits.maxComputeWorkgroupsPerDimension`. If any dimension exceeds the limit, the dispatch is blocked (the real `origDispatch` is not called), an error log is pushed to `profilerData.logs`, `workgroupAnalyzer.analyzeDispatch` is called with `dimensionViolation: true`, `broadcastData()` fires, and the function returns early.
2. **CPU timing** — wraps `origDispatch` with `performance.now()`.
3. **Dispatch record** — populated with kernel ID, pipeline label, workgroup/dispatch sizes, CPU timing, device label, pass type, and `timingSource: 'pending'`.
4. **Buffer access snapshot** — flattens `__websight_resources` from all bound bind groups.
5. **Memory pattern analysis** — calls `analyzeMemoryAccessPattern` (Module 09).
6. **Bandwidth snapshot** — calls `calculateDispatchBandwidth` (Module 09); updates `pass.__bandwidthSnapshot`.
7. **Workgroup analysis** — calls `workgroupAnalyzer.analyzeDispatch` if `enableWorkgroupAnalysis` is set; appends critical/warning logs.
8. **Kernel stat increment** — `kernel.stats.count++` (timing will be added later by `queue.submit`).
9. **Registry** — pushes to `profilerData.dispatches`, `pass.__dispatches`, `encoderData.dispatches`, increments `device.__webSightInfo.dispatchCount`.
10. **Buffer heat map** — increments `profilerData.bufferHeatMap[id]` for each accessed buffer.
11. **Broadcast** — calls `broadcastData()`.

### `beginRenderPass`

Follows the same structure. Wraps `setPipeline`, `setBindGroup`, `draw`, and `drawIndexed` via the shared `recordDrawCall` helper.

#### `recordDrawCall(drawParams, origFn, origArgs)`

Calls `calculateRenderDrawBandwidth` (Module 09), which deduplicates framebuffer attachment bandwidth across draw calls in the same pass (`pass.__fbBandwidthCounted`). Sets `timingSource: 'render_pass_timing'`.

### `encoder.finish`

Attaches `__dispatches`, `__passTimings`, and a `__gpuTiming` promise (resolved by the submit hook) to the returned command buffer.

---

## `TimingHelper` pool

Each device gets its own pool stored in `window.__webSightTimingHelperPools`.

| Field | Default | Meaning |
|---|---|---|
| `helpers` | `[]` | All created `TimingHelper` instances |
| `available` | `[]` | Queue of idle helpers ready to hand out |
| `inUse` | `Set` | Currently active helpers |
| `maxSize` | `8` | Hard cap on helpers per device |
| `passesPerHelper` | `1` | Passes per `TimingHelper` (always 1) |
| `failed` | `false` | Set when 0 helpers can be created |
| `limitReached` | `false` | Set when GPU QuerySet cap is hit |
| `missedCount` | `0` | Passes that received no timing; logged every 50 misses |

**Acquire** (`getTimingHelper`): returns `null` if `pool.failed`, dequeues from `available` if possible, tries to create a new helper otherwise, or records a miss and returns `null`.

**Release** (`releaseTimingHelper`): moves the helper from `inUse` back to `available`.

---

## `queue.submit` — GPU timing resolution

After `origSubmit` returns, `device.queue.onSubmittedWorkDone()` is awaited to ensure GPU work is finished before reading timestamps. For each pass entry:

| Condition | Behaviour |
|---|---|
| `entry.helper` is `null` | Pushes `0` to `allDurations` |
| `helper.__valid === false` | Pushes `0`, releases helper |
| Normal | Calls `helper.getResult()` → `durations[0]` (nanoseconds) |

**Single dispatch per pass** — assigns `gpuTimeMs` and `timingSource: 'gpu_timestamp'` directly to the dispatch record; recalculates `bandwidth.bandwidthGBs`; updates all four kernel stats (`totalTime`, `avgTime`, `minTime`, `maxTime`); accumulates per-kernel bandwidth totals.

**Multiple dispatches per pass** — sets `dispatch.passGpuTimeMs` and `timingSource: 'pass_aggregate'`; divides the pass time evenly for `kernel.stats.totalTime/avgTime`; only seeds `minTime`/`maxTime` when they are still at sentinel values (`Infinity`/`0`) to avoid corrupting real single-dispatch measurements.

After all passes are processed, resolved durations are appended to `window.__webSightGlobalTimingResults` (trimmed with `slice(-n)` when over cap), the `commandBuffer.__gpuTiming` promise is resolved, the `'timing'` event fires on `window.__webSightTimingEvents`, and `broadcastData()` is called.

---

## `getMultiGPUStats()`

Reads the three window globals and returns:

```
{
  adapters: [{ index, hasTimestampFeature, powerPreference, deviceCount, requestedAt }],
  devices:  [{ index, label, hasTimestampQuery, timingMode, encoderCount,
               passCount, dispatchCount, features, limits, createdAt }],
  pools:    [{ deviceLabel, poolSize, maxSize, available, inUse, missedCount,
               currentIndex, failed, limitReached, mode, utilizationPercent }],
  totals:   { adapterCount, deviceCount, totalEncoders, totalPasses, totalDispatches }
}
```

`limits` in the device snapshot is a subset: `maxComputeWorkgroupsPerDimension`, `maxComputeInvocationsPerWorkgroup`, `maxStorageBufferBindingSize`, `maxBufferSize`.

---

## Design notes

**Single hook install.** The `__webSightHooked` flag on `navigator.gpu` prevents duplicate installation if `hookWebGPU()` is called more than once (e.g., from multiple script tags).

**`setupPipeline` shared helper.** Both the synchronous and async pipeline constructors call the same helper so the capture logic is not duplicated.

**`proxyEncoder` shim.** `TimingHelper` needs access to `encoder.resolveQuerySet` and `encoder.copyBufferToBuffer`. Rather than passing the full encoder, a minimal proxy is constructed to avoid inadvertently exposing the full hooked encoder to the timing helper.

**Canvas texture lifecycle.** WebGPU destroys the previous canvas texture silently when the canvas is resized. By tracking `context.__lastCanvasTexture` and calling `markDestroyed` on mismatch, the memory leak detector does not permanently count canvas textures as leaked.

**`slice(-n)` instead of `splice`.** When the timing results buffer exceeds its cap, `slice(-n)` is used to retain only the newest entries. This is O(n) allocation but avoids the O(n) element-shift that `splice(0, excess)` performs in place.

**Dimension violation is non-fatal.** The dispatch is blocked to prevent GPU device loss, but the record is still passed to `workgroupAnalyzer.analyzeDispatch` with `dimensionViolation: true` so the issue appears in downstream analysis views.

---

## Known limitations

- **Bandwidth is estimated, not measured.** All `bandwidth.*` values are derived from buffer sizes times an access-count factor (see Module 09). Actual GPU memory traffic may differ from cache effects, tiling, and compression.
- **`pass_aggregate` timing is approximate.** When a pass contains multiple dispatches the pass duration is split evenly. This is inaccurate for passes with heterogeneous kernels.
- **`TimingHelper` pool cap is a heuristic.** `maxSize: 8` was chosen to stay within common GPU QuerySet limits, but some devices allow fewer and some more. The pool degrades gracefully to `cpu-only` when the cap is exceeded.
- **Hooks are per-device.** `hookWebGPU` installs hooks at `requestDevice` time. Devices created before `hookWebGPU()` is called are not instrumented.
- **Canvas texture memory is assumed RGBA8.** The hardcoded 4 bytes/pixel estimate is wrong for HDR (`rgba16float`) or compressed canvas formats.
- **`window.__webSightTimingHelper_executionId`** is incremented but not consumed anywhere; it is a legacy field from an earlier API iteration.

---

## Data flow

```
dispatchWorkgroups / draw / drawIndexed
          │
          ├─► profilerData.dispatches[]          (Module 02)
          ├─► profilerData.kernels[id].stats      (Module 02)
          ├─► profilerData.bufferHeatMap          (Module 02)
          ├─► profilerData.logs[]                 (Module 02)  ← violations + workgroup warnings
          └─► broadcastData()                     (Module 07)

queue.submit → onSubmittedWorkDone
          │
          ├─► dispatch.gpuTimeMs / .timingSource  (in-place on Module 02 records)
          ├─► kernel.stats (totalTime, avg, min, max)
          ├─► kernel.bandwidth (totalBytes, avgGBs, peakGBs)
          ├─► window.__webSightGlobalTimingResults
          ├─► commandBuffer.__gpuTiming (Promise resolved)
          ├─► window.__webSightTimingEvents 'timing' event
          └─► broadcastData()                     (Module 07)

getMultiGPUStats()
          └─► reads window.__webSightAdapters / __webSightDevices / __webSightTimingHelperPools
              ← consumed by 11-public-api.js WebSight.getMultiGPUStats()
```
