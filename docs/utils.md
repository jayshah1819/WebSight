# Utility Functions

**Source:** `src/08-utils.js`  
**Imports:** `profilerData` (config reads), `broadcastData` (called by `addLog`)

---

## What it does

Module 8 provides stateless helpers used across the codebase. All functions are pure except `addLog`, which writes to `profilerData.logs` and triggers a broadcast. No class, no singleton — just exported functions.

---

## Functions

### `normalizeTime(timeNs) → number`

Converts a nanosecond value to the unit configured in `profilerData.config.normalizeTimeUnit`.

| Config value | Output |
|---|---|
| `'ns'` | nanoseconds (no conversion) |
| `'us'` | microseconds (÷ 1000) |
| `'ms'` | milliseconds (÷ 1,000,000) |
| default | microseconds |

Used by Module 11's `listKernels()` to normalize per-kernel timing before returning it to callers.

---

### `getTimeUnitLabel() → string`

Returns the display label for the current time unit: `'ns'`, `'µs'`, or `'ms'`. Matches `normalizeTime()` so callers can display the correct unit suffix alongside values.

---

### `hashString(str) → string`

Produces a short base-36 string hash of any string using a 32-bit djb2-style hash. Used internally by `generateKernelId` to produce compact, deterministic ID components from shader source and workgroup config. Not cryptographic.

---

### `generateKernelId(shaderSource, workgroupSize, label?) → string`

Builds a stable kernel identity string from the shader source, workgroup dimensions, and optional label:

```
kernel_{sourceHash}_{configHash}[_{labelHash}]
```

Two kernels with the same shader and workgroup geometry but different labels get different IDs. Used by Module 10 to key the `profilerData.kernels` map.

---

### `extractWorkgroupSize(source, entryPoint?) → { x, y, z }`

Parses `@workgroup_size(...)` from WGSL source and returns resolved integer dimensions.

**Constant resolution:** Scans for `const` and `override` declarations and builds a name→value map. When a `@workgroup_size` component is a name rather than a literal, it is looked up in this map. Unresolved names default to 1.

**Entry point scoping:** If `entryPoint` is provided, the parser finds the `fn entryPoint(` declaration and takes the last `@workgroup_size` attribute that appears before it. This handles shaders with multiple entry points correctly. Without an entry point, the first `@workgroup_size` match is used.

Returns `{ x: 1, y: 1, z: 1 }` if no `@workgroup_size` is found.

---

### `analyzeWGSL(source) → { warnings, metrics }`

Lightweight WGSL scan used for quick pre-pass warnings. Checks for:

- Atomic operations (any of the 7 `atomic*` variants) → `metrics.hasAtomics = true`
- More than 5 `if (` occurrences → `metrics.hasBranching = true`

Returns `{ warnings: [], metrics: {} }` on empty input. This is a coarser check than Module 5's `ShaderComplexityAnalyzer` — it runs synchronously at pipeline creation time, not on-demand.

---

### `addLog(message, level?) → void`

Appends a structured log entry to `profilerData.logs` and triggers a debounced broadcast.

```js
profilerData.logs.push({
  timestamp: new Date().toLocaleTimeString(),
  level,      // default 'info'
  message,
  time: Date.now()
})
```

If `verboseLogging` is enabled, also logs to `console.log`. Every `addLog` call triggers `broadcastData()`, so the dashboard receives a snapshot shortly after any significant profiler event.

---

## Data flow

```
08-utils.js
  normalizeTime / getTimeUnitLabel  ──▶  11-public-api.js  (listKernels, getStats)
  hashString / generateKernelId     ──▶  10-webgpu-hooks.js (kernel ID generation)
  extractWorkgroupSize              ──▶  10-webgpu-hooks.js (pipeline creation hook)
  analyzeWGSL                       ──▶  10-webgpu-hooks.js (pre-pass warning at createComputePipeline)
  addLog                            ──▶  everywhere          (profilerData.logs + broadcastData)
```
