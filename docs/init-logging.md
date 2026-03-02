# Analyzer Instances & Logging Helpers

**Source:** `src/06-init-logging.js`  
**Exports:** `memoryLeakDetector`, `workgroupAnalyzer`, `shaderAnalyzer`, `log`, `warn`, `error`

---

## What it does

Module 6 is the shared singleton layer. It instantiates the three analyzer classes exactly once and exports the instances so every downstream module imports the same object — never constructing its own copy. It also provides three logging helpers that gate output behind `profilerData.config.verboseLogging`, keeping the console clean by default.

---

## Why a dedicated module for this

ES modules are evaluated once per page and their exports are live bindings. Any module that imports `memoryLeakDetector` gets the same object that Module 10's hooks write to and Module 11's public API reads from. If each module called `new MemoryLeakDetector()` directly, they would each have a separate instance with separate state, and no cross-module accumulation would work.

Module 6 is the single point where all three analyzers are constructed, and all other modules import from here.

---

## Singleton exports

| Export | Class | Source |
|---|---|---|
| `memoryLeakDetector` | `MemoryLeakDetector` | `src/03-memory-leak-detector.js` |
| `workgroupAnalyzer` | `WorkgroupOccupancyAnalyzer` | `src/04-workgroup-analyzer.js` |
| `shaderAnalyzer` | `ShaderComplexityAnalyzer` | `src/05-shader-analyzer.js` |

All three are constructed with no arguments at module evaluation time. Any configuration (e.g. `workgroupAnalyzer.setDeviceLimits`, `memoryLeakDetector.leakThreshold`) is applied later — by Module 10 on device acquisition, or by Module 11 when `configure()` is called.

---

## Logging helpers

### `log(...args)`

Wraps `console.log`. Output is suppressed unless `profilerData.config.verboseLogging` is `true`. Used by Modules 10 and 11 for routine profiler events (hook registration, timing results, dispatch records).

### `warn(...args)`

Wraps `console.warn`. Gated by the same `verboseLogging` flag. Used for non-critical anomalies — missing labels, fallback paths, timing estimation.

### `error(...args)`

Wraps `console.error`. **Always emits**, regardless of `verboseLogging`. Reserved for conditions that indicate a real problem: device loss, invalid dispatch geometry, hook setup failure.

---

## Enabling verbose output

```js
WebSight.configure({ verboseLogging: true });
```

Or directly:

```js
profilerData.config.verboseLogging = true;
```

This gates `log()` and `warn()` calls only. `error()` is always shown.

---

## Data flow

```
06-init-logging.js
  memoryLeakDetector  ──┬──▶  10-webgpu-hooks.js   (trackResource, markDestroyed)
                        └──▶  11-public-api.js      (getMemoryLeaks, formatBytes, formatTime)

  workgroupAnalyzer   ──┬──▶  10-webgpu-hooks.js   (analyzeDispatch, setDeviceLimits)
                        └──▶  11-public-api.js      (getWorkgroupAnalysis, getWorkgroupSummaryData)

  shaderAnalyzer      ──┬──▶  10-webgpu-hooks.js   (shaderId stored on kernel)
                        └──▶  11-public-api.js      (getShaderAnalysis, analyzeShader)

  log / warn / error  ──────▶  10-webgpu-hooks.js, 11-public-api.js
```
