# Memory Leak Detector

**Source:** `src/03-memory-leak-detector.js`  
**Singleton:** `memoryLeakDetector` — instantiated in `src/06-init-logging.js`, shared by Modules 10 and 11.

---

## What it does

Tracks every `GPUBuffer` and `GPUTexture` created and destroyed by the application.
Any resource that stays alive longer than `leakThreshold` ms (default 60 s) is
classified as a potential leak. It also tracks running memory totals so you can see
current usage, peak usage, and allocation/free counts at any time.

---

## How resources are tracked

Module 10 (`10-webgpu-hooks.js`) intercepts `device.createBuffer` and
`device.createTexture`. On every creation it calls `trackResource`; on every
`.destroy()` it calls `markDestroyed`.

```
device.createBuffer(desc)
  └─ memoryLeakDetector.trackResource(buffer, 'GPUBuffer', desc.size)

buffer.destroy()
  └─ memoryLeakDetector.markDestroyed(buffer)

device.createTexture(desc)
  └─ memoryLeakDetector.trackResource(texture, 'GPUTexture', estimatedSize)

texture.destroy()
  └─ memoryLeakDetector.markDestroyed(texture)
```

TimingHelper's internal buffers are explicitly excluded (label starts with
`"TimingHelper"`) so profiler overhead never appears in leak reports.

---

## Internal design

### `#resourceIds` WeakMap

The detector uses a private `WeakMap<GPUObject, integer>` rather than stamping a
property directly onto the resource object. This avoids colliding with the UUID that
Module 10 writes to `__websight_id` for canvas texture reuse tracking.

### `resources` Map

Stores metadata keyed by integer id. Deliberately does **not** hold a reference to
the GPU object itself — that would prevent destroyed buffers from being garbage
collected. The WeakMap handles the reverse lookup.

```javascript
resources.get(id) → {
  type:      'GPUBuffer' | 'GPUTexture',
  size:      number,          // bytes
  createdAt: number,          // Date.now()
  destroyed: boolean,
  destroyedAt: number | undefined,
  lifetime:  number | undefined,  // ms, set on destroy
  label:     string,
  stack:     string           // JS call stack, only if config.captureStacks = true
}
```

### Map pruning

Every 100 `markDestroyed` calls, entries that have been destroyed for more than 60 s
are removed from the Map. This keeps memory pressure bounded for long profiling
sessions with thousands of buffer allocations.

### `#findLeaks()` — single source of truth

Both `checkForLeaks()` and `getLeakReport()` delegate to the private `#findLeaks()`
method. This ensures `stats.leakCount` is always in sync with what both callers see
and that the threshold logic has one place to change.

```
#findLeaks()
  ├─ iterates resources Map
  ├─ age = now - createdAt
  ├─ age > leakThreshold  →  leaks[]   (includes stack for debugging)
  ├─ age ≤ leakThreshold  →  active[]
  ├─ destroyed = true     →  destroyed[]
  └─ sets stats.leakCount = leaks.length
```

---

## API reference

### `trackResource(resource, type, size) → id`

Registers a GPU resource. Returns the integer id assigned to it.
Safe to call twice on the same resource — the second call is a no-op if the resource
is still alive (prevents `currentMemory` double-counting).

| Parameter | Type | Description |
|---|---|---|
| `resource` | `GPUBuffer \| GPUTexture` | The GPU object to track |
| `type` | `string` | `'GPUBuffer'` or `'GPUTexture'` |
| `size` | `number` | Size in bytes |

---

### `markDestroyed(resource)`

Records that a resource has been freed. Updates `totalFreed`, `currentMemory`, and
`destroyedCount`. Safe to call on an untracked resource — silently returns.

---

### `checkForLeaks()`

Runs `#findLeaks()` and logs a `console.warn` + `console.table` grouped by type if
any leaks are found. Also warns if `currentMemory > sizeThreshold` (default 100 MB).

---

### `getLeakReport() → object`

Returns the full report without logging. Shape:

```javascript
{
  stats: {
    totalAllocated:  number,   // cumulative bytes ever allocated
    totalFreed:      number,   // cumulative bytes freed
    peakMemory:      number,   // highest currentMemory ever seen
    currentMemory:   number,   // bytes currently live
    leakCount:       number,   // resources currently classified as leaks
    createdCount:    number,
    destroyedCount:  number
  },
  leaks:     [ { id, type, size, age, label, isLeak, stack } ],  // sorted by size desc
  active:    [ { id, type, size, age, label } ],                  // sorted by size desc
  destroyed: [ { id, type, size, lifetime, label } ],             // sorted by lifetime desc, top 100
  summary: {
    totalLeaks:        number,
    leakedMemory:      number,   // total bytes in leak entries
    activeResources:   number,
    activeMemory:      number,
    destroyedResources: number,
    leakRate:          string    // e.g. "2.50%"
  }
}
```

---

### `enableAutoCheck(intervalMs = 30000)`

Starts a repeating timer that calls `checkForLeaks()` every `intervalMs` milliseconds.
Not called automatically — must be opted in explicitly.

```javascript
// Check every 30 s (default)
memoryLeakDetector.enableAutoCheck();

// Check every 2 minutes for long benchmark suites
memoryLeakDetector.enableAutoCheck(120000);
```

---

### `disableAutoCheck()`

Clears the auto-check interval.

---

### `formatBytes(bytes) → string`

Human-readable byte count. Returns `"0 B"`, `"1.50 KB"`, `"256.00 MB"`, etc.
Used as the single source of truth for all byte formatting across the public API
(`getStats`, `getBufferData`, `WebSight.formatBytes`).

---

### `formatTime(ms) → string`

Human-readable duration. Returns `"500ms"`, `"3.2s"`, `"1.5min"`.

---

## Configuration

### `leakThreshold` (default `60000` ms)

Resources alive longer than this are classified as leaks. Configurable at runtime
via `WebSight.configure()`:

```javascript
// For a regression suite that runs 2+ minutes
WebSight.configure({ leakThresholdMs: 180000 });
```

### `sizeThreshold` (default `100 MB`)

If `currentMemory` exceeds this value, `checkForLeaks()` emits an additional warning.
Changeable directly on the instance: `memoryLeakDetector.sizeThreshold = 512 * 1024 * 1024`.

### `captureStacks` (default `false`)

Enable via `WebSight.configure` → `profilerData.config.captureStacks = true`.  
When true, a JS stack trace is captured at `trackResource` time and stored per entry.
Stack traces appear only on leak entries in the report (not active or destroyed).

---

## Usage via the public API

| `WebSight` method | What it calls |
|---|---|
| `getMemoryLeaks()` | `checkForLeaks()` then `getLeakReport()` — logs and returns report |
| `getMemoryStats()` | Returns `memoryLeakDetector.stats` directly |
| `getFullAnalysisReport()` | Calls `getLeakReport()` as part of combined analysis |
| `configure({ leakThresholdMs })` | Sets `memoryLeakDetector.leakThreshold` |
| `formatBytes(n)` | Delegates to `memoryLeakDetector.formatBytes(n)` |
| `formatTime(ms)` | Delegates to `memoryLeakDetector.formatTime(ms)` |

---

## Known limitations

| Scenario | Behaviour |
|---|---|
| Resources created before WebSight loads | Not tracked — invisible to the detector |
| WebWorker / OffscreenCanvas WebGPU | Not tracked — hooks only apply to the main thread |
| `GPUSampler`, `GPURenderPipeline`, etc. | Not tracked — only `GPUBuffer` and `GPUTexture` |
| Canvas textures (`getCurrentTexture`) | Tracked on creation, never marked destroyed (browser owns lifetime) — will appear as leak after threshold |
| GPU device lost | App never calls `.destroy()` so all live resources appear as leaks after threshold |

---

## Data flow summary

```
10-webgpu-hooks.js
  createBuffer / createTexture  →  trackResource()   ─┐
  buffer.destroy / texture.destroy  →  markDestroyed() ─┤
                                                         ↓
                                               resources Map  ←→  #resourceIds WeakMap
                                                         ↓
                                                    #findLeaks()
                                                  ┌──────┴──────┐
                                           checkForLeaks()  getLeakReport()
                                                         ↓
11-public-api.js
  getMemoryLeaks()  →  logs + returns report
  getMemoryStats()  →  { currentMemory, peakMemory, ... }
  formatBytes()     →  human-readable strings everywhere
```
