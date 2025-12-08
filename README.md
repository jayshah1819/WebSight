# WebSight

WebGPU profiler that shows you where your compute shaders are slow.

## What it does

Hooks into WebGPU and tracks everything:
- How long each compute dispatch takes
- Which buffers you're using
- What's causing bottlenecks (memory, atomics, etc.)
- Bandwidth and throughput numbers

## Quick start

**Browser extension (easiest):**
1. Open `chrome://extensions/`
2. Turn on Developer Mode
3. Click "Load unpacked" and select the `extension` folder
4. Click the extension icon on any WebGPU site

**Standalone:**
Just open `index.html` in Chrome

**In your code:**
```javascript
await WebSight.start();
// your webgpu code here
const data = WebSight.getData();
```

## What you get

- Real-time stats on every compute dispatch
- Timeline showing GPU activity
- Buffer inspector (size, usage, access count)
- Scalability test tool (`scalability-benchmark.html`)
- Export everything as JSON

## Bottleneck types it finds

- **MEMORY_BANDWIDTH** - You're hitting memory limits
- **ATOMIC_SERIALIZATION** - Atomics causing slowdown
- **OCCUPANCY** - Not enough threads running
- **COMPUTE_BOUND** - Math heavy, not memory limited
- **DIVERGENCE** - Branches causing issues
- **REGISTER_PRESSURE** - Too many variables

## Browser support

Chrome/Edge 113+ works best. Safari 18+ also works.

## Files

```
index.html                  - Main UI
scalability-benchmark.html  - Performance testing tool
profiler-standalone.js      - Core profiler library
test-histogram.html         - Example with atomic ops
extension/                  - Chrome extension version
```

## API

```javascript
WebSight.start()           // Start profiling
WebSight.getData()         // Get all captured data
WebSight.getStats()        // Get summary statistics
WebSight.export()          // Download JSON report
WebSight.clear()           // Clear captured data
```

## License

MIT - do whatever you want with it

---

Made for WebGPU developers who want to know why their shaders are slow.
