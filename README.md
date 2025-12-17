# WebSight

A **real-time performance profiler for WebGPU** that shows you exactly what your GPU is doing—without modifying your code.

## What does it do?

WebSight hooks into the WebGPU API and captures:
- **GPU execution times** (actual hardware timing, not just JavaScript overhead)
- **Memory allocations** (buffers, textures, pipelines)
- **Compute shader performance** (dispatch patterns, workgroup efficiency)
- **Draw call metrics** (vertex counts, instance counts)
- **Memory leaks** (resources that never get cleaned up)
- **Workgroup optimization issues** (inefficient GPU utilization)

All without touching your application code. Just include the profiler, open the dashboard, and start coding.

## Quick Start

### 1. Add the profiler to your app

```html
<script src="profiler-standalone.js"></script>
```

That's it. WebSight automatically hooks into WebGPU when your page loads.

### 2. Open the dashboard

Open `index.html` in another browser tab. It'll automatically connect and start showing real-time metrics.

### 3. Run your WebGPU code

Everything gets profiled automatically:
- Compute shader dispatches
- Render passes
- Buffer allocations
- Pipeline creation

## What you'll see

### Live Performance Graphs
- **Bandwidth vs Input Size**: How fast your GPU moves data
- **Execution Time vs Input Size**: How your shaders scale
- Color-coded for GPU (green) vs CPU (black) timing

### Buffer Analysis
- Which buffers are eating your memory
- Atomic contention hotspots
- Buffer size warnings (when you're near GPU limits)

### Kernel Performance
- Per-dispatch timing for every compute shader
- Average/min/max execution times
- Workgroup configuration analysis

### Automatic Warnings

WebSight catches common issues automatically:

**Workgroup Problems:**
- "Workgroup size not multiple of 64" → GPU hardware expects this
- "Dispatch exceeds GPU limit" → Your dispatch is literally too big
- "X dimension exceeds 65535" → Hardware maximum violation

**Memory Issues:**
- Buffer leaks (resources never destroyed)
- Buffers approaching GPU limits (> 90% of max size)
- Memory pressure warnings

**Shader Issues:**
- Unsupported types (`u64`, `i64`, `f64` don't work in WebGPU)
- Divergent branches (performance killers)
- Expensive math operations

## Features Explained

### GPU Timestamp Queries (The Magic)

Most profilers only tell you how long the *CPU* took to submit work. That's useless.

WebSight uses WebGPU's `timestamp-query` feature to measure **actual GPU execution time**:

```javascript
// What timestamp queries do:
GPU starts executing → [timestamp 1]
GPU finishes → [timestamp 2]
Real execution time = timestamp 2 - timestamp 1
```

This gives you **nanosecond-precision** timing of what the GPU actually did.

**Note:** If your GPU doesn't support timestamp queries, WebSight falls back to CPU timing (still useful for relative comparisons).

### Memory Leak Detection

Creates a resource tracker that watches:
```javascript
device.createBuffer()   // Tracked
device.createTexture()  // Tracked
buffer.destroy()        // Marked as destroyed
```

After 10 seconds (configurable), if a resource hasn't been destroyed, it flags it as a potential leak.

### Workgroup Optimization Analysis

Checks every compute dispatch for:
- **SIMD efficiency**: Is your workgroup size a multiple of 64? (GPU wavefront size)
- **Power-of-2 dimensions**: GPUs like 64x1x1, 128x1x1, etc.
- **GPU limit violations**: Did you exceed hardware maximums?
- **Workgroup utilization**: Are you wasting GPU cores?

Example warning:
```
Workgroup size 63x1x1 not multiple of 64
→ Wasting 1 thread per workgroup
→ Recommendation: Use 64x1x1 instead
```

### Shader Complexity Analysis

Parses your WGSL code to find:
- **Divergent branches** (`if` statements inside loops → slow)
- **Expensive math** (sin, cos, sqrt → use lookup tables if possible)
- **Atomic operations** (contention = performance death)
- **Unsupported types** (u64 doesn't exist in WebGPU)

## Configuration

```javascript
WebSight.configure({
  // Disable real-time broadcasting (for benchmarking mode)
  broadcastEnabled: false,
  
  // How often to send updates (milliseconds)
  broadcastDebounceMs: 3000,
  
  // Enable/disable specific analyzers
  enableMemoryLeakDetection: true,
  enableWorkgroupAnalysis: true,
  enableShaderAnalysis: true,
  
  // Memory warning threshold (MB)
  memoryWarningThresholdMB: 100
});
```

## API Reference

### Get profiler data
```javascript
const data = WebSight.getData();
console.log(data.dispatches);  // All compute/render dispatches
console.log(data.kernels);     // Aggregated kernel stats
console.log(data.buffers);     // All allocated buffers
```

### Get statistics
```javascript
const stats = WebSight.getStats();
console.log(stats.totalGpuTime);       // Total GPU execution time
console.log(stats.gpuTimedDispatches); // # of dispatches with GPU timing
console.log(stats.avgBandwidth);       // Average bandwidth utilization
```

### Memory analysis
```javascript
const leaks = WebSight.getMemoryLeaks();
console.log(leaks.summary.totalLeaks);  // # of potential leaks
console.log(leaks.leaks);               // Array of leaked resources
```

### Workgroup analysis
```javascript
const analysis = WebSight.getWorkgroupAnalysis();
console.log(analysis.summary.grade);    // A-F grade
console.log(analysis.summary.criticalIssues); // # of critical issues
```

### Clear data
```javascript
WebSight.clear();  // Reset all profiling data
```

## Dashboard Controls

**Stop/Start Profiling**: Pause data collection without losing current data

**Clear Data**: Wipe everything and start fresh

**Graph Settings**:
- **Line Shape**: Linear (accurate) vs Smooth (prettier)
- **Scale**: Log (better for large ranges) vs Linear

**Analysis Buttons**:
- **Run Memory Analysis**: Check for resource leaks
- **Run Shader Analysis**: Find optimization opportunities
- **Full Report**: Export everything

## Common Issues

### "No GPU timing data"
Your GPU doesn't support the `timestamp-query` feature. You'll still get CPU timing (submission time), which is useful for comparisons but not absolute measurements.

Check with:
```javascript
const adapter = await navigator.gpu.requestAdapter();
console.log(adapter.features.has('timestamp-query')); // Should be true
```

### "WGSL parse error: unresolved type 'u64'"
WebGPU doesn't support 64-bit integers. Use `u32` instead, or emulate with:
```wgsl
struct U64 {
  lo: u32,  // Lower 32 bits
  hi: u32   // Upper 32 bits
}
```

### "Dispatch exceeds GPU limit"
Your dispatch dimensions are too large. Maximum is typically:
- X: 65,535 workgroups
- Y: 65,535 workgroups  
- Z: 65,535 workgroups

Split large dispatches into multiple smaller ones.

### Dashboard shows "Waiting for WebGPU app..."
Make sure:
1. Profiler is loaded in your app (`<script src="profiler-standalone.js"></script>`)
2. Both pages are served from the same origin (localhost works)
3. Your app is actually using WebGPU

## Benchmarking Mode

For accurate benchmarks, disable real-time updates:

```javascript
WebSight.benchmarkMode();  // Disables broadcasting, minimal overhead
// Run your benchmark
const stats = WebSight.getStats();
console.log(stats.totalGpuTime);
WebSight.normalMode();  // Re-enable broadcasting
```


## Browser Support

- **Chrome/Edge**:  Full support (with timestamp-query flag enabled)
- **Firefox**: WebGPU behind flag
- **Safari**: WebGPU experimental

Enable timestamp queries in Chrome:
```
chrome://flags/#enable-webgpu-developer-features
```

## License

MIT - Do whatever you want with it.

## Contributing

Found a bug? Open an issue.
Want a feature? Open a PR.

---
