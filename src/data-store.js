// =============================================================================
// MODULE 2: Profiler Data Store
// =============================================================================

export const profilerData = {
  dispatches: [],
  pipelines: {},
  bindGroups: {},
  buffers: {},
  timingHelper: null,
  logs: [],
  gpuCharacteristics: null,
  bufferHeatMap: {},
  runId: null,
  kernels: {},
  runs: {},
  timingMode: 'unknown',
  sessionStart: Date.now(),
  totalKernelTime: 0,
  memoryUsage: {
    peak: 0,
    current: 0,
    allocations: []
  },
  activeEncoders: new WeakMap(),
  config: {
    broadcastEnabled: true,  // Enable broadcasting to UI
    broadcastDebounceMs: 3000,  // Update every 3 seconds to reduce flickering
    normalizeTimeUnit: 'us',

    verboseLogging: false,

    minimalOverhead: false,

    enableMemoryLeakDetection: false,
    enableWorkgroupAnalysis: true,  // Enable by default to catch dispatch geometry issues
    enableShaderAnalysis: false,
    captureStacks: false,

    peakMemoryBandwidthGBs: null,  // User must set via configure() for efficiency %
    simdWidth: 32, 

    memoryLeakThresholdMs: 10000,
    memoryWarningThresholdMB: 100
  }
};
