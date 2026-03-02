# Bandwidth Tracking & Memory Pattern Analysis

**Source:** `src/09-bandwidth-tracker.js`  
**Exports:** `analyzeMemoryAccessPattern`, `calculateTextureSize`, `calculateFramebufferBandwidth`, `parseWGSLBindings`, `BandwidthTracker`, `calculateDispatchBandwidth`, `calculateRenderDrawBandwidth`

---

## What it does

Module 9 estimates how many bytes a GPU pass reads and writes. WebGPU does not expose hardware memory counters, so the approach is bound-buffer-size estimation: sum the sizes of buffers and textures bound to a pass, split by access mode (read-only vs read-write), and use the GPU time from Module 10's timing helpers to compute GB/s.

The module also parses WGSL binding declarations to determine access modes, and computes framebuffer bandwidth for render passes from attachment descriptors.

---

## Estimation model

All bandwidth values carry `measurementNote: 'bound-buffer-size'` in their returned objects. This is an intentional reminder that:

- The values reflect the size of bound resources, not actual bytes transferred by the GPU hardware.
- A buffer bound but only partially accessed counts in full.
- Read-write buffers (`var<storage, read_write>`) are counted once as read and once as written.
- `bandwidthGBs` is backfilled by Module 10 after `onSubmittedWorkDone` resolves the GPU time — at dispatch record time it is always `0`.

---

## Functions

### `analyzeMemoryAccessPattern(shaderCode, bufferAccesses, workgroupSize, dispatchSize) → object`

Scans shader source for structural hints about memory access behaviour:

| Hint | Trigger |
|---|---|
| `'uses-shared-memory'` | `var<workgroup>` present |
| `'uses-atomics'` | any `atomic*` builtin present |
| `'uses-barriers'` | `workgroupBarrier` or `storageBarrier` present |
| `'math-heavy'` | > 10 occurrences of `sin`, `cos`, `exp`, `log`, `pow`, or `sqrt` |

Returns:

```js
{
  accessPatternType: 'unknown',  // reserved for future pattern detection
  hints:             string[],
  coalesced:         null,       // reserved
  cacheEfficiency:   'unknown'   // reserved
}
```

If `shaderCode` is empty, returns the same shape with an empty `hints` array.

---

### `calculateTextureSize(textureView) → number`

Estimates the memory size of a texture in bytes, including all mip levels.

Reads `textureView.__websight_texture` (a back-reference stored by Module 10's texture hook) to get `width`, `height`, `depthOrArrayLayers`, `mipLevelCount`, and `format`. Uses a built-in format→bytes-per-pixel table covering all standard WebGPU texture formats plus common compressed formats (BC, ETC2, ASTC).

Returns `0` if the back-reference is absent.

---

### `calculateFramebufferBandwidth(passDescriptor) → { bytesRead, bytesWritten }`

Estimates render pass framebuffer bandwidth from a `GPURenderPassDescriptor`:

- Each color attachment is counted as a write. If `loadOp === 'load'`, it is also counted as a read.
- The depth/stencil attachment is counted as a write. If `depthLoadOp === 'load'`, also as a read.
- Size comes from `calculateTextureSize`. If that returns 0 (back-reference missing), a fallback path tries `attachment.view.texture`, `view.descriptor.size`, and `view.__texture` in order. If dimensions are still unavailable, the attachment contributes 0 bytes.

---

### `parseWGSLBindings(shaderCode) → { [group]: { [binding]: string } }`

Parses `@group` / `@binding` declarations from WGSL source and returns a nested map of access types.

**Buffer bindings** (`var<addressSpace[, accessMode]>`):

| Address space | Access mode | Resolved type |
|---|---|---|
| `uniform` | — | `'uniform'` |
| `storage` | `read_write` | `'storage'` |
| `storage` | absent or `read` | `'read-only-storage'` |

**Texture / sampler bindings** (plain `var` without angle brackets):

| WGSL type prefix | Resolved type |
|---|---|
| `texture_storage` | `'storage-texture'` |
| `texture_` | `'sampled-texture'` |
| `sampler` / `sampler_comparison` | `'sampler'` |

Used by Module 10 to annotate bound resources with their WGSL-declared access mode at `setBindGroup` time.

---

### `BandwidthTracker` class

Tracks cumulative bytes read and written across all `setBindGroup` calls within a single command encoder pass. One instance per pass, created by Module 10.

#### `trackBindGroup(bindGroup, bindGroupLayout)`

Iterates `bindGroup.__websight_resources` (populated by Module 10's bind group hook). For each resource not yet seen in this pass:

- `read-only` → adds `size` to `totalBytesRead`
- `read-write` → adds `size` to `totalBytesReadWrite`
- `write-only` → adds `size` to `totalBytesWritten`

Resources are deduped by `res.id` — a buffer bound to multiple bind groups in the same pass is counted only once.

#### `calculateBandwidth(durationNs) → object`

Returns cumulative bandwidth totals. `bandwidthGBs` is computed from `durationNs`; pass `0` to get `bandwidthGBs: 0` when time is not yet known.

```js
{
  bytesRead:       totalBytesRead + totalBytesReadWrite,
  bytesWritten:    totalBytesWritten + totalBytesReadWrite,
  totalBytes:      bytesRead + bytesWritten,
  totalDataMB:     string,
  measurementNote: 'bound-buffer-size',
  bandwidthGBs:    number,
  resourceCount:   number
}
```

Read-write buffers appear in both `bytesRead` and `bytesWritten` because `var<storage, read_write>` performs both operations.

---

### `calculateDispatchBandwidth(tracker, snapshotBefore, accessPatternType) → { bandwidth, newSnapshot }`

Computes per-dispatch bandwidth by diffing the tracker's current totals against a snapshot taken after the previous dispatch. This is how Module 10 attributes bandwidth to individual dispatches within a multi-dispatch pass.

```js
bytesRead    = tracker.bytesRead    - snapshotBefore.read
bytesWritten = tracker.bytesWritten - snapshotBefore.written
```

Returns:

```js
{
  bandwidth: {
    bytesRead, bytesWritten, totalBytes,
    measurementNote: 'bound-buffer-size',
    bandwidthGBs: 0,          // backfilled later
    arithmeticIntensity: 0,   // reserved
    resourceCount, accessPattern
  },
  newSnapshot: { read, written }  // caller advances its snapshot to this for the next dispatch
}
```

---

### `calculateRenderDrawBandwidth(tracker, snapshotBefore, passDescriptor, fbAlreadyCounted) → { bandwidth, newSnapshot, fbCountedNow }`

Same as `calculateDispatchBandwidth` but for render pass draw calls. Adds framebuffer bandwidth (from `calculateFramebufferBandwidth`) to the buffer delta, but only if `fbAlreadyCounted` is `false`. The first draw call in a render pass pays the framebuffer cost; subsequent draws do not.

`fbCountedNow` is returned so the caller can set its own `fbAlreadyCounted` flag for subsequent draws in the same pass.

---

## Data flow

```
10-webgpu-hooks.js
  createRenderPassEncoder / createComputePassEncoder
    └─ new BandwidthTracker()  ─────────────────────────────────┐
                                                                 │
  setBindGroup()                                                 │
    └─ tracker.trackBindGroup(bindGroup, layout)                │
                                                                 │
  dispatchWorkgroups()                                           ▼
    └─ calculateDispatchBandwidth(tracker, snapshot)
         └─ dispatch.bandwidth = { bytesRead, bytesWritten, ... }

  draw() / drawIndexed()
    └─ calculateRenderDrawBandwidth(tracker, snapshot, passDesc, fbCounted)
         └─ dispatch.bandwidth = { bytesRead, bytesWritten, ... }

  queue.submit → onSubmittedWorkDone
    └─ dispatch.bandwidth.bandwidthGBs = bytes / gpuTimeNs / 1e9

11-public-api.js
  getBandwidthAnalysis()  ──▶  reads dispatch.bandwidth
  getStats()              ──▶  reads dispatch.bandwidth via _computeBandwidthStats()
  getGraphData()          ──▶  reads dispatch.bandwidth.totalBytes for series grouping
```
