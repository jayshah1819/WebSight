# Broadcast & Control Messages

**Source:** `src/07-broadcast.js`  
**Exports:** `profilerChannel`, `broadcastData`

---

## What it does

Module 7 handles all cross-page communication between the profiled application and the dashboard (`index.html`). It opens a `BroadcastChannel` named `websight-profiler`, listens for control messages sent by the dashboard, and sends debounced profiler state snapshots back to it.

The benchmark page and the dashboard both run in separate browser tabs. This is the only module that bridges them.

---

## Channel name

```
'websight-profiler'
```

Both the application page (which loads the profiler) and the dashboard page listen on the same channel name. `BroadcastChannel` connects any tabs on the same origin — the channel does not need to be created before the other tab opens.

---

## Control messages (inbound)

The benchmark page listens for messages sent by the dashboard. Messages are ignored if `window.__webSightIsUIWindow` is set — the dashboard page has the channel open but should never process its own commands as if it were the profiled app.

| `msg.type` | Effect |
|---|---|
| `enable-profiling` | Sets `profilerData.config.broadcastEnabled = true` — subsequent `broadcastData()` calls will send snapshots |
| `disable-profiling` | Sets `profilerData.config.broadcastEnabled = false` — snapshots are suppressed |
| `clear-data` | Resets `dispatches`, `pipelines`, `buffers`, `kernels`, `logs`, `runs` to empty, then immediately calls `broadcastData()` to push the empty state to the dashboard |

---

## `broadcastData()`

Schedules a debounced snapshot of `profilerData` to be sent over the channel.

### Guards

1. **UI window guard** — returns immediately if `window.__webSightIsUIWindow` is set. The dashboard imports Module 11 (which imports this module), but should never broadcast.
2. **Enabled guard** — returns if `profilerData.config.broadcastEnabled` is `false`. Benchmark mode sets this to `false` to eliminate serialization overhead.
3. **Debounce guard** — if a timer is already pending, the call is a no-op. Only one broadcast fires per debounce interval. The interval is `profilerData.config.broadcastDebounceMs` (default 3000 ms, set to 10000 ms in minimal-overhead mode).

### Buffer filtering

Sending the full `profilerData.buffers` object on every broadcast would include buffers that were created but never bound to any dispatch — staging buffers, intermediate copies, etc. Instead, the payload contains only buffers that appear in at least one `dispatch.bufferAccesses` entry:

```
usedBufferIds = union of bufferAccess.id across all dispatches
filteredBuffers = profilerData.buffers filtered to usedBufferIds
```

This keeps payload size proportional to active GPU work rather than total allocation history.

### Payload shape

```js
{
  type: 'profiler-update',
  data: {
    dispatches:          profilerData.dispatches,
    pipelines:           profilerData.pipelines,
    buffers:             filteredBuffers,
    kernels:             profilerData.kernels,
    logs:                profilerData.logs,
    gpuCharacteristics:  profilerData.gpuCharacteristics,
    runs:                profilerData.runs,
    runId:               profilerData.runId,
    timingMode:          profilerData.timingMode,
    sessionStart:        profilerData.sessionStart,
    timestamp:           Date.now()
  }
}
```

All array and object fields are direct references to the live `profilerData` objects. `BroadcastChannel.postMessage` performs a structured clone — this creates a deep copy at send time, so the dashboard receives a snapshot, not a live reference.

---

## Dashboard-side handling

The dashboard (`index.html`) receives `profiler-update` messages and stores the snapshot in `lastReceivedData`. When `updateUI()` runs and `WebSight.getData()` returns an empty local profilerData (because the dashboard itself runs no WebGPU), the snapshot is injected into the local store so all downstream API calls (`getStats()`, `getGraphData()`, etc.) see the broadcast data.

---

## Debounce behaviour

`broadcastData()` is called by Module 10 on every queue submit. Without debouncing, a benchmark loop running at 60 fps would attempt hundreds of serializations per second. The debounce timer means only one snapshot fires per `broadcastDebounceMs` window, regardless of how many submits happen in that interval.

```
submit → broadcastData() → timer starts (3000ms)
submit → broadcastData() → no-op, timer already running
submit → broadcastData() → no-op
... 3000ms later → snapshot sent, timer cleared
submit → broadcastData() → new timer starts
```

---

## Data flow

```
index.html (dashboard)
  BroadcastChannel('websight-profiler')
    ├─ send: { type: 'enable-profiling' }
    ├─ send: { type: 'disable-profiling' }
    ├─ send: { type: 'clear-data' }
    └─ receive: { type: 'profiler-update', data: snapshot }
               └─ lastReceivedData = snapshot
               └─ updateUI() → Object.assign(profilerData, snapshot) if local empty

benchmark page
  07-broadcast.js
    profilerChannel.onmessage
      ├─ 'enable-profiling'  → broadcastEnabled = true
      ├─ 'disable-profiling' → broadcastEnabled = false
      └─ 'clear-data'        → reset profilerData fields → broadcastData()

    10-webgpu-hooks.js
      queue.submit() → broadcastData() → debounce → postMessage(payload)
```
