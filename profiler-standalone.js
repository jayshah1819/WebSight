// WebGPU Performance Profiler
// Hooks into WebGPU API to capture GPU timing, pipeline usage, memory allocations,
// and dispatch statistics. Broadcasts profiler data via BroadcastChannel for real-time visualization.

(function() {
  'use strict';

  function assert(cond, msg = "") {
    if (!cond) {
      throw new Error(msg);
    }
  }

  class TimingHelper {
    #canTimestamp;
    #device;
    #querySet;
    #resolveBuffer;
    #resultBuffer;
    #resultBuffers = [];
    #passNumber;
    #numKernels;
    #state = "free";

    constructor(device, numKernels = 1) {
      this.#device = device;
      this.#canTimestamp = device.features.has("timestamp-query");
      this.#numKernels = numKernels;
      this.reset(numKernels);
    }

    destroy() {
      if (this.#querySet) this.#querySet.destroy();
      if (this.#resolveBuffer) this.#resolveBuffer.destroy();
      while (this.#resultBuffers.length > 0) {
        const resultBuffer = this.#resultBuffers.pop();
        resultBuffer.destroy();
      }
    }

    reset(numKernels) {
      this.#passNumber = 0;
      this.#numKernels = numKernels;
      if (this.#canTimestamp) {
        if (this.#querySet) {
          this.#querySet.destroy();
        }
        try {
          this.#device.pushErrorScope('validation');
          this.#querySet = this.#device.createQuerySet({
            type: "timestamp",
            label: `TimingHelper query set buffer of count ${numKernels * 2}`,
            count: numKernels * 2,
          });
          this.#device.popErrorScope().then(error => {
            if (error) {
              console.warn(`[WebSight] QuerySet creation failed: ${error.message}`);
              this.#canTimestamp = false;
              this.#querySet = null;
            }
          });
        } catch (e) {
          this.#canTimestamp = false;
          throw new Error(`Failed to create QuerySet: ${e.message}`);
        }
        if (this.#resolveBuffer) {
          this.#resolveBuffer.destroy();
        }
        if (this.#querySet) {
          this.#resolveBuffer = this.#device.createBuffer({
            size: this.#querySet.count * 8,
            label: `TimingHelper resolve buffer of count ${this.#querySet.count}`,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
          });
        }
      }
    }

    get numKernels() {
      return this.#numKernels;
    }

    get canTimestamp() {
      return this.#canTimestamp;
    }

    #beginTimestampPass(encoder, fnName, descriptor) {
      if (this.#canTimestamp && this.#querySet) {
        assert(
          this.#state === "free" || this.#state == "in progress",
          `state not free (state = ${this.#state})`
        );

        const pass = encoder[fnName]({
          ...descriptor,
          ...{
            timestampWrites: {
              querySet: this.#querySet,
              beginningOfPassWriteIndex: this.#passNumber * 2,
              endOfPassWriteIndex: this.#passNumber * 2 + 1,
            },
          },
        });

        this.#passNumber++;
        if (this.#passNumber == this.#numKernels) {
          this.#state = "need resolve";
        } else {
          this.#state = "in progress";
        }

        const resolve = () => this.#resolveTiming(encoder);
        pass.end = (function (origFn) {
          return function () {
            origFn.call(this);
            resolve();
          };
        })(pass.end);

        return pass;
      } else {
        return encoder[fnName](descriptor);
      }
    }

    beginRenderPass(encoder, descriptor = {}) {
      return this.#beginTimestampPass(encoder, "beginRenderPass", descriptor);
    }

    beginComputePass(encoder, descriptor = {}) {
      return this.#beginTimestampPass(encoder, "beginComputePass", descriptor);
    }

    #resolveTiming(encoder) {
      if (!this.#canTimestamp) {
        return;
      }
      if (this.#passNumber != this.#numKernels) {
        return;
      }
      assert(
        this.#state === "need resolve",
        `must call addTimestampToPass (state is '${this.#state}')`
      );
      this.#state = "wait for result";

      this.#resultBuffer =
        this.#resultBuffers.pop() ||
        this.#device.createBuffer({
          size: this.#resolveBuffer.size,
          label: `TimingHelper result buffer of count ${this.#querySet.count}`,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

      encoder.resolveQuerySet(
        this.#querySet,
        0,
        this.#querySet.count,
        this.#resolveBuffer,
        0
      );
      encoder.copyBufferToBuffer(
        this.#resolveBuffer,
        0,
        this.#resultBuffer,
        0,
        this.#resultBuffer.size
      );
    }

    async getResult() {
      if (!this.#canTimestamp) {
        return [0];
      }
      assert(
        this.#state === "wait for result",
        `must call resolveTiming (state === ${this.#state})`
      );
      this.#state = "free";

      this.#passNumber = 0;

      const resultBuffer = this.#resultBuffer;
      await resultBuffer.mapAsync(GPUMapMode.READ);
      const times = new BigInt64Array(resultBuffer.getMappedRange());
      const durations = [];
      for (let idx = 0; idx < times.length; idx += 2) {
        durations.push(Number(times[idx + 1] - times[idx]));
      }
      resultBuffer.unmap();
      this.#resultBuffers.push(resultBuffer);
      return durations;
    }

    getStats() {
      return {
        canTimestamp: this.#canTimestamp,
        numKernels: this.#numKernels,
        currentPass: this.#passNumber,
        state: this.#state
      };
    }
  }

  const profilerData = {
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
      
      memoryLeakThresholdMs: 10000,
      memoryWarningThresholdMB: 100
    }
  };

  class MemoryLeakDetector {
    constructor() {
      this.resources = new Map();
      this.nextId = 0;
      this.leakThreshold = 10000;
      this.sizeThreshold = 100 * 1024 * 1024;
      this.autoCheckInterval = null;
      
      this.stats = {
        totalAllocated: 0,
        totalFreed: 0,
        peakMemory: 0,
        currentMemory: 0,
        leakCount: 0,
        createdCount: 0,
        destroyedCount: 0
      };
    }
    
    enableAutoCheck() {
      if (!this.autoCheckInterval) {
        this.autoCheckInterval = setInterval(() => this.checkForLeaks(), 5000);
      }
    }
    
    disableAutoCheck() {
      if (this.autoCheckInterval) {
        clearInterval(this.autoCheckInterval);
        this.autoCheckInterval = null;
      }
    }
    
    trackResource(resource, type, size) {
      const id = this.nextId++;
      const now = Date.now();
      
      let stack = '';
      if (profilerData.config.captureStacks) {
        try {
          throw new Error();
        } catch (e) {
          stack = e.stack;
        }
      }
      
      this.resources.set(id, {
        resource: resource,
        type: type,
        size: size,
        createdAt: now,
        stack: stack,
        destroyed: false,
        label: resource.label || 'unlabeled'
      });
      
      this.stats.totalAllocated += size;
      this.stats.currentMemory += size;
      this.stats.createdCount++;
      
      if (this.stats.currentMemory > this.stats.peakMemory) {
        this.stats.peakMemory = this.stats.currentMemory;
      }
      
      resource.__websight_id = id;
      
      return id;
    }
    
    markDestroyed(resource) {
      const id = resource.__websight_id;
      if (id === undefined) return;
      
      const info = this.resources.get(id);
      if (!info || info.destroyed) return;
      
      info.destroyed = true;
      info.destroyedAt = Date.now();
      info.lifetime = info.destroyedAt - info.createdAt;
      
      this.stats.totalFreed += info.size;
      this.stats.currentMemory -= info.size;
      this.stats.destroyedCount++;
    }
    
    checkForLeaks() {
      const now = Date.now();
      const leaks = [];
      
      for (const [id, info] of this.resources.entries()) {
        if (!info.destroyed) {
          const age = now - info.createdAt;
          
          if (age > this.leakThreshold) {
            leaks.push({
              id: id,
              type: info.type,
              size: info.size,
              age: age,
              label: info.label,
              stack: info.stack
            });
          }
        }
      }
      
      if (leaks.length > 0) {
        this.stats.leakCount = leaks.length;
        console.warn(`[WebSight] ${leaks.length} potential memory leaks detected!`);
        
        const byType = {};
        leaks.forEach(leak => {
          if (!byType[leak.type]) byType[leak.type] = [];
          byType[leak.type].push(leak);
        });
        
        console.table(Object.entries(byType).map(([type, items]) => ({
          Type: type,
          Count: items.length,
          TotalSize: this.formatBytes(items.reduce((sum, l) => sum + l.size, 0)),
          OldestAge: this.formatTime(Math.max(...items.map(l => l.age)))
        })));
      }
      
      if (this.stats.currentMemory > this.sizeThreshold) {
        console.warn(`[WebSight] High memory usage: ${this.formatBytes(this.stats.currentMemory)}`);
      }
    }
    
    getLeakReport() {
      const now = Date.now();
      const leaks = [];
      const active = [];
      const destroyed = [];
      
      for (const [id, info] of this.resources.entries()) {
        if (!info.destroyed) {
          const age = now - info.createdAt;
          const item = {
            id: id,
            type: info.type,
            size: info.size,
            age: age,
            label: info.label,
            isLeak: age > this.leakThreshold
          };
          
          if (item.isLeak) {
            leaks.push(item);
          } else {
            active.push(item);
          }
        } else {
          destroyed.push({
            id: id,
            type: info.type,
            size: info.size,
            lifetime: info.lifetime,
            label: info.label
          });
        }
      }
      
      return {
        stats: this.stats,
        leaks: leaks.sort((a, b) => b.size - a.size),
        active: active.sort((a, b) => b.size - a.size),
        destroyed: destroyed.sort((a, b) => b.lifetime - a.lifetime).slice(0, 100), // Top 100
        summary: {
          totalLeaks: leaks.length,
          leakedMemory: leaks.reduce((sum, l) => sum + l.size, 0),
          activeResources: active.length,
          activeMemory: active.reduce((sum, a) => sum + a.size, 0),
          destroyedResources: destroyed.length,
          leakRate: this.stats.createdCount > 0 ? (leaks.length / this.stats.createdCount * 100).toFixed(2) + '%' : '0%'
        }
      };
    }
    
    formatBytes(bytes) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
    }
    
    formatTime(ms) {
      if (ms < 1000) return ms + 'ms';
      if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
      return (ms / 60000).toFixed(1) + 'min';
    }
  }

  class WorkgroupOccupancyAnalyzer {
    constructor() {
      this.deviceLimits = null;
      this.analyses = new Map(); // kernelId -> analysis
    }
    
    setDeviceLimits(limits) {
      this.deviceLimits = limits;
      
      this.optimal = {
        simdWidth: 64,
        sharedMemoryPerWorkgroup: 32 * 1024,
        maxThreadsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup || 256,
        recommendations: {
          '1D': [256, 128, 64],
          '2D': [[16, 16], [8, 8], [32, 8]],
          '3D': [[8, 8, 8], [4, 4, 4]]
        }
      };
    }
    
    analyzeDispatch(dispatch) {
      if (!this.deviceLimits) {
        console.warn('[WebSight] Cannot analyze dispatch - device limits not set yet');
        return null;
      }
      
      const wgSize = dispatch.workgroupSize;
      const dispatchSize = dispatch.dispatchSize;
      
      if (!wgSize || !dispatchSize) {
        console.warn('[WebSight] Cannot analyze dispatch - missing workgroup or dispatch size');
        return null;
      }
      
      const analysis = {
        kernelId: dispatch.kernelId,
        workgroupSize: wgSize,
        dispatchSize: dispatchSize,
        totalThreads: wgSize[0] * wgSize[1] * wgSize[2],
        totalWorkgroups: dispatchSize[0] * dispatchSize[1] * dispatchSize[2],
        totalInvocations: 0,
        issues: [],
        score: 100, // Start at 100, deduct for issues
        recommendations: []
      };
      
      analysis.totalInvocations = analysis.totalThreads * analysis.totalWorkgroups;
      
      // Check if dispatch dimensions exceed device limits
      const maxDim = this.deviceLimits.maxComputeWorkgroupsPerDimension || 65535;
      const totalWorkgroupCount = dispatchSize[0] * dispatchSize[1] * dispatchSize[2];
      
      let hasCriticalDimensionError = false;
      
      if (dispatchSize[0] > maxDim) {
        hasCriticalDimensionError = true;
        analysis.issues.push({
          severity: 'critical',
          type: 'exceeds-x-limit',
          message: `X dimension (${dispatchSize[0]}) exceeds limit (${maxDim})`,
          impact: 'GPU will REJECT this dispatch',
          recommendation: `Split X into Y/Z dimensions. Check getDispatchGeometry() logic.`
        });
        analysis.score = 0;
      }
      
      if (dispatchSize[1] > maxDim) {
        hasCriticalDimensionError = true;
        analysis.issues.push({
          severity: 'critical',
          type: 'exceeds-y-limit',
          message: `Y dimension (${dispatchSize[1]}) exceeds limit (${maxDim})`,
          impact: 'GPU will REJECT this dispatch',
          recommendation: `BUG DETECTED: Y overflow suggests nested if-inside-while in getSimpleDispatchGeometry(). Use separate sequential while loops: first reduce X→Y, then reduce Y→Z.`
        });
        analysis.score = 0;
      }
      
      if (dispatchSize[2] > maxDim) {
        hasCriticalDimensionError = true;
        analysis.issues.push({
          severity: 'critical',
          type: 'exceeds-z-limit',
          message: `Z dimension (${dispatchSize[2]}) exceeds limit (${maxDim})`,
          impact: 'GPU will REJECT this dispatch',
          recommendation: `Total workgroups (${totalWorkgroupCount}) too large. Reduce workgroup count or use tiling.`
        });
        analysis.score = 0;
      }
      
      // Small utility dispatches get lenient scoring
      const isSmallUtilityDispatch = !hasCriticalDimensionError 
        && totalWorkgroupCount < 256 
        && analysis.totalThreads >= 8
        && analysis.totalThreads < 64;
      
      // Calculate workgroup utilization
      const maxWorkgroupsPerDim = this.deviceLimits.maxComputeWorkgroupsPerDimension;
      const xUtilization = ((dispatchSize[0] / maxWorkgroupsPerDim) * 100).toFixed(1);
      const yUtilization = ((dispatchSize[1] / maxWorkgroupsPerDim) * 100).toFixed(1);
      const zUtilization = ((dispatchSize[2] / maxWorkgroupsPerDim) * 100).toFixed(1);
      
      analysis.workgroupUtilization = {
        total: totalWorkgroupCount,
        xPercent: parseFloat(xUtilization),
        yPercent: parseFloat(yUtilization),
        zPercent: parseFloat(zUtilization),
        xDim: dispatchSize[0],
        yDim: dispatchSize[1],
        zDim: dispatchSize[2],
        maxPerDim: maxWorkgroupsPerDim,
        xExceeds: dispatchSize[0] > maxWorkgroupsPerDim,
        yExceeds: dispatchSize[1] > maxWorkgroupsPerDim,
        zExceeds: dispatchSize[2] > maxWorkgroupsPerDim
      };
      
      if (isSmallUtilityDispatch) {
        analysis.score = 70;
        analysis.issues.push({
          severity: 'info',
          type: 'utility-dispatch',
          message: `Small utility dispatch (${totalWorkgroupCount} workgroups)`,
          impact: 'Acceptable for finalization/reduction passes',
          recommendation: 'No action needed for small auxiliary kernels'
        });
      }
      
      // Issue 1: Workgroup size not multiple of SIMD width (skip for small utility dispatches)
      if (!isSmallUtilityDispatch && analysis.totalThreads % this.optimal.simdWidth !== 0) {
        const wastedThreads = this.optimal.simdWidth - (analysis.totalThreads % this.optimal.simdWidth);
        const efficiency = ((analysis.totalThreads / (analysis.totalThreads + wastedThreads)) * 100).toFixed(1);
        
        analysis.issues.push({
          severity: 'medium',
          type: 'simd-inefficiency',
          message: `Workgroup size (${analysis.totalThreads}) not a multiple of SIMD width (${this.optimal.simdWidth})`,
          impact: `${efficiency}% efficiency - ${wastedThreads} threads wasted per SIMD group`,
          recommendation: `Use size that's multiple of ${this.optimal.simdWidth}: ${this.roundUpToMultiple(analysis.totalThreads, this.optimal.simdWidth)}`
        });
        analysis.score -= 20;
      }
      
      if (!isSmallUtilityDispatch && analysis.totalThreads < 64) {
        analysis.issues.push({
          severity: 'high',
          type: 'low-occupancy',
          message: `Workgroup size (${analysis.totalThreads}) is very small`,
          impact: 'GPU underutilized - many execution units idle',
          recommendation: `Increase to at least 64 threads, prefer 128-256`
        });
        analysis.score -= 30;
      }
      
      if (analysis.totalThreads > this.optimal.maxThreadsPerWorkgroup) {
        analysis.issues.push({
          severity: 'critical',
          type: 'exceeds-limit',
          message: `Workgroup size (${analysis.totalThreads}) exceeds device limit (${this.optimal.maxThreadsPerWorkgroup})`,
          impact: 'Kernel may fail on some devices',
          recommendation: `Reduce to ${this.optimal.maxThreadsPerWorkgroup} or less`
        });
        analysis.score -= 50;
      }
      
      const dimensions = [wgSize[0], wgSize[1], wgSize[2]].filter(d => d > 1).length;
      if (dimensions === 1 && (wgSize[1] > 1 || wgSize[2] > 1)) {
        analysis.issues.push({
          severity: 'low',
          type: 'dimension-mismatch',
          message: 'Workgroup appears 1D but uses multiple dimensions',
          impact: 'Minor cache inefficiency',
          recommendation: 'Consider using [256, 1, 1] instead of [256, Y, Z] for 1D problems'
        });
        analysis.score -= 5;
      }
      
      const isPowerOf2 = (n) => n > 0 && (n & (n - 1)) === 0;
      if (!isPowerOf2(wgSize[0]) || (wgSize[1] > 1 && !isPowerOf2(wgSize[1])) || (wgSize[2] > 1 && !isPowerOf2(wgSize[2]))) {
        analysis.issues.push({
          severity: 'low',
          type: 'non-power-of-2',
          message: 'Workgroup dimensions are not powers of 2',
          impact: 'May reduce performance on some GPUs (especially AMD)',
          recommendation: `Use power-of-2 sizes like [${this.nearestPowerOf2(wgSize[0])}, ${this.nearestPowerOf2(wgSize[1])}, ${this.nearestPowerOf2(wgSize[2])}]`
        });
        analysis.score -= 10;
      }
      
      if (analysis.score > 80) {
        analysis.recommendations.push({
          type: 'good',
          message: 'Workgroup configuration looks good!'
        });
      }
      
      if (analysis.issues.length > 0) {
        const maxImpact = Math.max(...analysis.issues.filter(i => i.severity === 'high' || i.severity === 'critical').map(() => 2), 1.5);
        analysis.potentialSpeedup = maxImpact.toFixed(1) + 'x';
      } else {
        analysis.potentialSpeedup = 'Already optimal';
      }
      
      this.analyses.set(dispatch.kernelId, analysis);
      return analysis;
    }
    
    roundUpToMultiple(value, multiple) {
      return Math.ceil(value / multiple) * multiple;
    }
    
    nearestPowerOf2(value) {
      if (value <= 1) return 1;
      return Math.pow(2, Math.round(Math.log2(value)));
    }
    
    getAllAnalyses() {
      return Array.from(this.analyses.values());
    }
    
    getSummary() {
      const analyses = this.getAllAnalyses();
      if (analyses.length === 0) return null;
      
      const avgScore = analyses.reduce((sum, a) => sum + a.score, 0) / analyses.length;
      const criticalIssues = analyses.filter(a => a.issues.some(i => i.severity === 'critical')).length;
      const highIssues = analyses.filter(a => a.issues.some(i => i.severity === 'high')).length;
      
      return {
        totalKernels: analyses.length,
        averageScore: avgScore.toFixed(1),
        grade: avgScore >= 90 ? 'A' : avgScore >= 80 ? 'B' : avgScore >= 70 ? 'C' : avgScore >= 60 ? 'D' : 'F',
        criticalIssues: criticalIssues,
        highIssues: highIssues,
        needsAttention: analyses.filter(a => a.score < 70)
      };
    }
  }

  class ShaderComplexityAnalyzer {
    constructor() {
      this.analyses = new Map();
      
      this.instructionCosts = {
        'sqrt': 8, 'rsqrt': 8, 'sin': 16, 'cos': 16, 'tan': 16,
        'exp': 12, 'exp2': 12, 'log': 12, 'log2': 12,
        'pow': 16, 'atan': 16, 'atan2': 16,
        
        'textureSample': 20, 'textureLoad': 10, 'textureStore': 10,
        'atomicAdd': 15, 'atomicSub': 15, 'atomicMax': 15,
        'atomicMin': 15, 'atomicAnd': 15, 'atomicOr': 15,
        
        // Control flow
        'if': 2, 'else': 1, 'for': 3, 'while': 3, 'loop': 3,
        'switch': 4, 'break': 1, 'continue': 1, 'return': 1,
        
        // Memory operations
        'load': 4, 'store': 4,
        
        // Cheap operations
        'add': 1, 'sub': 1, 'mul': 1, 'div': 2,
        'select': 1, 'clamp': 1, 'min': 1, 'max': 1
      };
    }
    
    analyzeShader(shaderId, wgslCode) {
      const analysis = {
        shaderId: shaderId,
        code: wgslCode,
        lineCount: 0,
        instructionCount: 0,
        complexity: 0,
        score: 100,
        issues: [],
        recommendations: [],
        metrics: {
          branches: 0,
          loops: 0,
          mathOps: 0,
          memoryOps: 0,
          atomicOps: 0,
          textureOps: 0,
          registerPressure: 0
        }
      };
      
      if (!wgslCode) {
        analysis.issues.push({
          severity: 'info',
          type: 'no-code',
          message: 'Shader code not available for analysis'
        });
        return analysis;
      }
      
      const lines = wgslCode.split('\n');
      analysis.lineCount = lines.filter(l => l.trim() && !l.trim().startsWith('//')).length;
      
      const divergentBranches = this.findDivergentBranches(wgslCode);
      if (divergentBranches.length > 0) {
        analysis.metrics.branches = divergentBranches.length;
        analysis.issues.push({
          severity: 'high',
          type: 'divergent-branch',
          message: `${divergentBranches.length} divergent branch(es) detected`,
          impact: 'Causes thread divergence - SIMD lanes waste cycles',
          locations: divergentBranches,
          recommendation: 'Use select() or bitwise operations instead of if/else when possible',
          example: 'let result = select(falseValue, trueValue, condition); // instead of if'
        });
        analysis.score -= 15 * divergentBranches.length;
      }
      
      const expensiveMath = this.findExpensiveMath(wgslCode);
      if (expensiveMath.length > 0) {
        analysis.metrics.mathOps = expensiveMath.length;
        analysis.issues.push({
          severity: 'medium',
          type: 'expensive-math',
          message: `${expensiveMath.length} expensive math operation(s)`,
          impact: `~${this.estimateCost(expensiveMath)}x slower than simple ALU ops`,
          operations: expensiveMath,
          recommendation: 'Consider approximations or lookup tables for non-critical calculations'
        });
        analysis.score -= 10;
      }
      
      const variables = this.countVariables(wgslCode);
      if (variables > 32) {
        analysis.metrics.registerPressure = variables;
        analysis.issues.push({
          severity: 'high',
          type: 'register-pressure',
          message: `High register usage: ${variables} variables`,
          impact: 'May cause register spilling to memory - significant slowdown',
          recommendation: 'Split kernel into multiple passes or reuse variables'
        });
        analysis.score -= 20;
      }
      
      const uncoalescedAccess = this.findUncoalescedAccess(wgslCode);
      if (uncoalescedAccess.length > 0) {
        analysis.metrics.memoryOps = uncoalescedAccess.length;
        analysis.issues.push({
          severity: 'critical',
          type: 'uncoalesced-access',
          message: `${uncoalescedAccess.length} potential uncoalesced memory access(es)`,
          impact: 'Can be 10-100x slower than coalesced access!',
          patterns: uncoalescedAccess,
          recommendation: 'Access memory sequentially: thread[i] accesses data[i], not data[i * stride]'
        });
        analysis.score -= 30;
      }
      
      const barriers = (wgslCode.match(/workgroupBarrier|storageBarrier/g) || []).length;
      if (barriers > 5) {
        analysis.issues.push({
          severity: 'medium',
          type: 'excessive-barriers',
          message: `${barriers} barrier synchronizations`,
          impact: 'Each barrier stalls all threads - reduces throughput',
          recommendation: 'Minimize barriers by restructuring algorithm'
        });
        analysis.score -= 10;
      }
      
      const atomics = this.findAtomicOps(wgslCode);
      if (atomics.length > 0) {
        analysis.metrics.atomicOps = atomics.length;
        analysis.issues.push({
          severity: 'medium',
          type: 'atomic-operations',
          message: `${atomics.length} atomic operation(s) found`,
          impact: 'Atomics serialize execution - avoid in hot paths',
          operations: atomics,
          recommendation: 'Use local reduction then single atomic, or avoid atomics entirely'
        });
        analysis.score -= 15;
      }
      
      const loops = this.analyzeLoops(wgslCode);
      if (loops.nested > 0) {
        analysis.metrics.loops = loops.total;
        analysis.issues.push({
          severity: 'medium',
          type: 'nested-loops',
          message: `${loops.nested} nested loop(s) with depth ${loops.maxDepth}`,
          impact: 'High iteration count - consider loop unrolling or tiling',
          recommendation: 'Unroll small loops (< 8 iterations) or use loop hints'
        });
        analysis.score -= 10 * loops.maxDepth;
      }
      
      if (wgslCode.includes('workgroupUniformLoad')) {
        analysis.recommendations.push({
          type: 'good',
          message: ' Uses workgroupUniformLoad - good for shared memory optimization'
        });
      }
      
      if (wgslCode.includes('subgroupAdd') || wgslCode.includes('subgroup')) {
        analysis.recommendations.push({
          type: 'excellent',
          message: ' Uses subgroup operations - excellent for performance!'
        });
        analysis.score += 10;
      }
      
      analysis.complexity = this.calculateComplexity(analysis);
      analysis.grade = this.getGrade(analysis.score);
      
      // Estimate potential speedup
      if (analysis.issues.length > 0) {
        const highSeverity = analysis.issues.filter(i => i.severity === 'critical' || i.severity === 'high');
        if (highSeverity.length > 0) {
          analysis.potentialSpeedup = '2-10x if critical issues fixed';
        } else {
          analysis.potentialSpeedup = '1.2-2x with optimizations';
        }
      } else {
        analysis.potentialSpeedup = 'Already well optimized';
      }
      
      this.analyses.set(shaderId, analysis);
      return analysis;
    }
    
    findDivergentBranches(code) {
      const branches = [];
      const lines = code.split('\n');
      
      // Look for if statements inside loops
      let inLoop = false;
      let loopDepth = 0;
      
      lines.forEach((line, idx) => {
        if (/\bfor\b|\bwhile\b|\bloop\b/.test(line)) {
          inLoop = true;
          loopDepth++;
        }
        if (inLoop && /\bif\b/.test(line) && !line.includes('//')) {
          branches.push({
            line: idx + 1,
            code: line.trim(),
            inLoopDepth: loopDepth
          });
        }
        if (line.includes('}') && loopDepth > 0) {
          loopDepth--;
          if (loopDepth === 0) inLoop = false;
        }
      });
      
      return branches;
    }
    
    findExpensiveMath(code) {
      const expensive = ['sqrt', 'rsqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'pow', 'atan'];
      const found = [];
      
      expensive.forEach(op => {
        const regex = new RegExp(`\\b${op}\\s*\\(`, 'g');
        const matches = code.match(regex);
        if (matches) {
          found.push({
            operation: op,
            count: matches.length,
            cost: this.instructionCosts[op]
          });
        }
      });
      
      return found;
    }
    
    countVariables(code) {
      // Count 'var' and 'let' declarations
      const vars = (code.match(/\b(var|let)\s+\w+/g) || []).length;
      return vars;
    }
    
    findUncoalescedAccess(code) {
      const patterns = [];
      const lines = code.split('\n');
      
      lines.forEach((line, idx) => {
        // Look for array[expression * stride] patterns
        if (/\[\s*\w+\s*\*\s*\d+\s*\]/.test(line)) {
          patterns.push({
            line: idx + 1,
            code: line.trim(),
            type: 'strided-access'
          });
        }
        
        // Look for indirect indexing
        if (/\[\s*\w+\[\w+\]\s*\]/.test(line)) {
          patterns.push({
            line: idx + 1,
            code: line.trim(),
            type: 'indirect-indexing'
          });
        }
      });
      
      return patterns;
    }
    
    findAtomicOps(code) {
      const atomics = ['atomicAdd', 'atomicSub', 'atomicMax', 'atomicMin', 'atomicAnd', 'atomicOr', 'atomicXor', 'atomicExchange', 'atomicCompareExchangeWeak'];
      const found = [];
      
      atomics.forEach(op => {
        const regex = new RegExp(`\\b${op}\\s*\\(`, 'g');
        const matches = code.match(regex);
        if (matches) {
          found.push({
            operation: op,
            count: matches.length
          });
        }
      });
      
      return found;
    }
    
    analyzeLoops(code) {
      const lines = code.split('\n');
      let depth = 0;
      let maxDepth = 0;
      let total = 0;
      let nested = 0;
      
      lines.forEach(line => {
        if (/\bfor\b|\bwhile\b|\bloop\b/.test(line)) {
          total++;
          depth++;
          if (depth > 1) nested++;
          maxDepth = Math.max(maxDepth, depth);
        }
        if (line.includes('}')) {
          depth = Math.max(0, depth - 1);
        }
      });
      
      return { total, nested, maxDepth };
    }
    
    estimateCost(operations) {
      return operations.reduce((sum, op) => sum + (op.cost * op.count), 0);
    }
    
    calculateComplexity(analysis) {
      // Weighted complexity score
      return (
        analysis.metrics.branches * 3 +
        analysis.metrics.loops * 5 +
        analysis.metrics.mathOps * 2 +
        analysis.metrics.memoryOps * 4 +
        analysis.metrics.atomicOps * 5 +
        analysis.metrics.registerPressure * 0.5
      );
    }
    
    getGrade(score) {
      if (score >= 90) return { letter: 'A', color: 'green', desc: 'Excellent' };
      if (score >= 80) return { letter: 'B', color: 'lightgreen', desc: 'Good' };
      if (score >= 70) return { letter: 'C', color: 'yellow', desc: 'Acceptable' };
      if (score >= 60) return { letter: 'D', color: 'orange', desc: 'Needs Work' };
      return { letter: 'F', color: 'red', desc: 'Poor' };
    }
    
    getAllAnalyses() {
      return Array.from(this.analyses.values());
    }
    
    getSummary() {
      const analyses = this.getAllAnalyses();
      if (analyses.length === 0) return null;
      
      const avgScore = analyses.reduce((sum, a) => sum + a.score, 0) / analyses.length;
      const avgComplexity = analyses.reduce((sum, a) => sum + a.complexity, 0) / analyses.length;
      
      return {
        totalShaders: analyses.length,
        averageScore: avgScore.toFixed(1),
        averageComplexity: avgComplexity.toFixed(1),
        overallGrade: this.getGrade(avgScore),
        criticalIssues: analyses.filter(a => a.issues.some(i => i.severity === 'critical')).length,
        needsOptimization: analyses.filter(a => a.score < 70)
      };
    }
  }

  // Initialize analyzers
  const memoryLeakDetector = new MemoryLeakDetector();
  const workgroupAnalyzer = new WorkgroupOccupancyAnalyzer();
  const shaderAnalyzer = new ShaderComplexityAnalyzer();
  
  // Helper functions for conditional logging
  function log(...args) {
    if (profilerData.config.verboseLogging) {
      console.log(...args);
    }
  }
  
  function warn(...args) {
    if (profilerData.config.verboseLogging) {
      console.warn(...args);
    }
  }
  
  function error(...args) {
    console.error(...args);
  }

  const profilerChannel = new BroadcastChannel('websight-profiler');
  let broadcastTimer = null;
  
  // Listen for control messages from UI
  profilerChannel.onmessage = (event) => {
    if (window.__webSightIsUIWindow) return; // Only profiler window handles these
    
    const msg = event.data;
    
    if (msg.type === 'enable-profiling') {
      profilerData.config.broadcastEnabled = true;
      console.log('[WebSight] Profiling ENABLED');
    } else if (msg.type === 'disable-profiling') {
      profilerData.config.broadcastEnabled = false;
      console.log('[WebSight] Profiling DISABLED');
    } else if (msg.type === 'clear-data') {
      // Clear all profiling data
      profilerData.dispatches = [];
      profilerData.pipelines = {};
      profilerData.buffers = {};
      profilerData.kernels = {};
      profilerData.logs = [];
      profilerData.runs = {};
      console.log('[WebSight] Profiling data CLEARED');
      broadcastData(); // Send empty data to UI
    }
  };

  function broadcastData() {
    if (window.__webSightIsUIWindow) {
      return;
    }
    
    if (!profilerData.config.broadcastEnabled) {
      return;
    }
    if (broadcastTimer) return;
    
    broadcastTimer = setTimeout(() => {
      try {
        // Filter buffers to only include those that are actually used in dispatches
        const usedBufferIds = new Set();
        profilerData.dispatches.forEach(dispatch => {
          if (dispatch.bufferAccesses) {
            dispatch.bufferAccesses.forEach(bufferAccess => {
              if (bufferAccess && bufferAccess.id) {
                usedBufferIds.add(bufferAccess.id);
              }
            });
          }
        });
        
        // Only send buffers that have been accessed
        const filteredBuffers = {};
        usedBufferIds.forEach(id => {
          if (profilerData.buffers[id]) {
            filteredBuffers[id] = profilerData.buffers[id];
          }
        });
        
        const payload = {
          type: 'profiler-update',
          data: {
            dispatches: profilerData.dispatches,
            pipelines: profilerData.pipelines,
            buffers: filteredBuffers, // Only send used buffers
            kernels: profilerData.kernels,
            logs: profilerData.logs,
            gpuCharacteristics: profilerData.gpuCharacteristics,
            runs: profilerData.runs,
            runId: profilerData.runId,
            timingMode: profilerData.timingMode,  // Add timing mode
            sessionStart: profilerData.sessionStart,
            timestamp: Date.now()
          }
        };
        profilerChannel.postMessage(payload);
      } catch (e) {
        console.error('[WebSight] Broadcast failed:', e);
      }
      broadcastTimer = null;
    }, profilerData.config.broadcastDebounceMs);
  }

  function normalizeTime(timeNs) {
    const unit = profilerData.config.normalizeTimeUnit;
    switch (unit) {
      case 'ns': return timeNs;
      case 'us': return timeNs / 1000;
      case 'ms': return timeNs / 1000000;
      default: return timeNs / 1000;
    }
  }

  function getTimeUnitLabel() {
    const unit = profilerData.config.normalizeTimeUnit;
    switch (unit) {
      case 'ns': return 'ns';
      case 'us': return 'µs';
      case 'ms': return 'ms';
      default: return 'µs';
    }
  }

  function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  function generateKernelId(shaderSource, workgroupSize, label = '') {
    const config = `${workgroupSize.x}x${workgroupSize.y}x${workgroupSize.z}`;
    const sourceHash = hashString(shaderSource);
    const configHash = hashString(config);
    const labelHash = label ? `_${hashString(label)}` : '';
    return `kernel_${sourceHash}_${configHash}${labelHash}`;
  }

  function extractWorkgroupSize(source) {
    const match = source.match(/@workgroup_size\((\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?\)/);
    if (match) {
      return {
        x: parseInt(match[1]) || 1,
        y: parseInt(match[2]) || 1,
        z: parseInt(match[3]) || 1
      };
    }
    return { x: 1, y: 1, z: 1 };
  }

  function analyzeWGSL(source) {
    if (!source) return { warnings: [], metrics: {} };
    const warnings = [];
    const metrics = { 
      hasAtomics: false, 
      hasBranching: false, 
      estimatedRegisters: 0, 
      sharedMemorySize: 0 
    };

    if (source.match(/atomic(Add|Sub|Max|Min|And|Or|Xor|Exchange|CompareExchange)/)) {
      metrics.hasAtomics = true;
      warnings.push({ 
        severity: 'warning', 
        type: 'ATOMIC_CONTENTION', 
        message: 'Atomic operations detected' 
      });
    }
    if (source.match(/\bif\s*\(/g)?.length > 5) {
      metrics.hasBranching = true;
      warnings.push({ 
        severity: 'warning', 
        type: 'EXCESSIVE_BRANCHING', 
        message: 'Excessive branching detected' 
      });
    }
    return { warnings, metrics };
  }

  function addLog(message, level = 'info') {
    profilerData.logs.push({ 
      timestamp: new Date().toLocaleTimeString(), 
      level, 
      message, 
      time: Date.now() 
    });
    // Only log to console if verbose logging is enabled
    if (profilerData.config.verboseLogging) {
      console.log(`[WebSight] ${message}`);
    }
    broadcastData();
  }

  async function hookWebGPU() {
    if (!navigator.gpu) {
      addLog('WebGPU not available', 'error');
      return;
    }

    if (!window.__webSightAdapters) {
      window.__webSightAdapters = [];
    }
    if (!window.__webSightDevices) {
      window.__webSightDevices = [];
    }

    const originalRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    
    navigator.gpu.requestAdapter = async function(options) {
      const adapter = await originalRequestAdapter(options);
      if (!adapter) return adapter;

      const hasTimestampFeature = adapter.features.has('timestamp-query');
      
      // Track this adapter
      const adapterInfo = {
        adapter,
        hasTimestampFeature,
        requestedAt: Date.now(),
        options: options || {},
        devices: []
      };
      window.__webSightAdapters.push(adapterInfo);
      
      addLog(`Adapter ${window.__webSightAdapters.length}: timestamp-query=${hasTimestampFeature}, powerPreference=${options?.powerPreference || 'default'}`);

      const originalRequestDevice = adapter.requestDevice.bind(adapter);
      
      adapter.requestDevice = async function(descriptor) {
        try {
          const requiredFeatures = new Set(descriptor?.requiredFeatures || []);
          
          if (hasTimestampFeature) {
            requiredFeatures.add('timestamp-query');
          }
          
          const modifiedDescriptor = {
            ...descriptor,
            requiredFeatures: Array.from(requiredFeatures)
          };
          
          const device = await originalRequestDevice(modifiedDescriptor);
          
          const hasTimestampQuery = device.features.has('timestamp-query');
          
          const shouldUseGPUTiming = (window.__webSightDisableGPUTiming !== true) && hasTimestampQuery;
          profilerData.timingMode = shouldUseGPUTiming ? 'gpu' : 'cpu-only';
          
          const deviceInfo = {
            hasTimestampQuery,
            createdAt: Date.now(),
            label: descriptor?.label || `device_${window.__webSightDevices.length + 1}`,
            features: Array.from(device.features),
            limits: { ...device.limits },
            timingMode: profilerData.timingMode, // Use same value as profilerData
            encoderCount: 0,
            passCount: 0,
            dispatchCount: 0
          };
          window.__webSightDevices.push(deviceInfo);
          adapterInfo.devices.push(deviceInfo);
          
          // Initialize workgroup analyzer with device limits immediately
          if (!workgroupAnalyzer.deviceLimits && device.limits) {
            workgroupAnalyzer.setDeviceLimits(device.limits);
            console.log('[WebSight] Workgroup analysis initialized - monitoring dispatch geometry');
          }
          
          // Add uncaptured error handler to catch QuerySet allocation failures
          device.addEventListener('uncapturederror', (event) => {
            if (event.error.message && event.error.message.includes('Cannot allocate sample buffer')) {
              console.error('[WebSight] GPU QuerySet allocation failed!');
              console.log('[WebSight] Your GPU has reached its QuerySet limit.');
        
              console.log('[WebSight]   <script>window.__webSightDisableGPUTiming = true;</script>');
              // Automatically disable GPU timing for this device
              deviceInfo.timingMode = 'cpu-only';
              profilerData.timingMode = 'cpu-only';
            }
          });
          
          // Store device reference on the device object itself for easy lookup
          device.__webSightInfo = deviceInfo;
          
          addLog(`Device ${window.__webSightDevices.length} created - Timing: ${profilerData.timingMode}, Label: ${deviceInfo.label}`);

          // Hook createShaderModule
          const origCreateShaderModule = device.createShaderModule.bind(device);
          device.createShaderModule = function(desc) {
            const module = origCreateShaderModule(desc);
            const shaderId = crypto.randomUUID();
            
            module.__source = desc.code;
            module.__shaderId = shaderId;
            
            // Store shader code for on-demand analysis
            // Don't analyze automatically 
            
            return module;
          };

          // Hook createComputePipeline
          const origCreateComputePipeline = device.createComputePipeline.bind(device);
          device.createComputePipeline = function(desc) {
            const pipeline = origCreateComputePipeline(desc);
            const source = desc.compute.module.__source || '';
            const workgroupSize = extractWorkgroupSize(source);
            const label = desc.label || 'compute_pipeline';
            
            pipeline.__capture = {
              id: crypto.randomUUID(),
              label: label,
              workgroupSize: workgroupSize,
              shader: source,
              shaderId: desc.compute.module.__shaderId,
              analysis: analyzeWGSL(source)
            };
            
            profilerData.pipelines[pipeline.__capture.id] = pipeline.__capture;
            
            const kernelId = generateKernelId(source, workgroupSize, label);
            if (!profilerData.kernels[kernelId]) {
              profilerData.kernels[kernelId] = {
                id: kernelId,
                label: label,
                workgroupSize: workgroupSize,
                shader: source,
                shaderId: desc.compute.module.__shaderId,
                stats: { count: 0, totalTime: 0, avgTime: 0, minTime: Infinity, maxTime: 0 }
              };
            }
            
            return pipeline;
          };

          // Hook createComputePipelineAsync
          const origCreateComputePipelineAsync = device.createComputePipelineAsync.bind(device);
          device.createComputePipelineAsync = async function(desc) {
            const pipeline = await origCreateComputePipelineAsync(desc);
            const source = desc.compute.module.__source || '';
            const workgroupSize = extractWorkgroupSize(source);
            const label = desc.label || 'compute_pipeline_async';
            
            pipeline.__capture = {
              id: crypto.randomUUID(),
              label: label,
              workgroupSize: workgroupSize,
              shader: source,
              shaderId: desc.compute.module.__shaderId,
              analysis: analyzeWGSL(source)
            };
            
            profilerData.pipelines[pipeline.__capture.id] = pipeline.__capture;
            
            const kernelId = generateKernelId(source, workgroupSize, label);
            if (!profilerData.kernels[kernelId]) {
              profilerData.kernels[kernelId] = {
                id: kernelId,
                label: label,
                workgroupSize: workgroupSize,
                shader: source,
                shaderId: desc.compute.module.__shaderId,
                stats: { count: 0, totalTime: 0, avgTime: 0, minTime: Infinity, maxTime: 0 }
              };
            }
            
            return pipeline;
          };

          // Hook createBuffer
          const origCreateBuffer = device.createBuffer.bind(device);
          device.createBuffer = function(desc) {
            const buffer = origCreateBuffer(desc);
            const bufferId = crypto.randomUUID();
            
            buffer.__capture = {
              id: bufferId,
              label: desc.label || 'buffer',
              size: desc.size,
              usage: desc.usage
            };
            profilerData.buffers[bufferId] = buffer.__capture;
            
            // Track for memory leak detection
            memoryLeakDetector.trackResource(buffer, 'GPUBuffer', desc.size);
            
            // Hook destroy method
            const origDestroy = buffer.destroy.bind(buffer);
            buffer.destroy = function() {
              memoryLeakDetector.markDestroyed(buffer);
              origDestroy();
            };
            
            return buffer;
          };

          // Hook createBindGroup
          const origCreateBindGroup = device.createBindGroup.bind(device);
          device.createBindGroup = function(desc) {
            const bg = origCreateBindGroup(desc);
            bg.__capture = {
              entries: desc.entries.map(e => ({ 
                binding: e.binding, 
                resource: e.resource.buffer?.__capture || e.resource 
              }))
            };
            return bg;
          };

          // Global timing accumulator for exposing to BasePrimitive.__timingHelper interface
          // This accumulates ALL pass timings across all encoders in current execution
          if (!window.__webSightGlobalTimingResults) {
            window.__webSightGlobalTimingResults = [];
            window.__webSightMaxTimingResults = 10000; 
          }
          

          // WebGPU limit: Varies by GPU (typically 30-60 QuerySets on integrated GPUs)
          // Each TimingHelper uses 1 QuerySet with 2 timestamps (begin/end)
          // Strategy: Create helpers up to GPU limit, then WAIT for completion before reuse
          if (!window.__webSightTimingHelperPools) {
            window.__webSightTimingHelperPools = new Map(); // Per-device pools
          }
          
          const getTimingHelper = (device) => {
            // Get or create pool for this device
            if (!window.__webSightTimingHelperPools.has(device)) {
              window.__webSightTimingHelperPools.set(device, {
                helpers: [],
                available: [], // Queue of helpers ready to use
                inUse: new Set(), // Helpers currently in use
                index: 0,
                maxSize: 1, // Ultra-conservative: only 1 TimingHelper to avoid GPU QuerySet limits
                passesPerHelper: 1, // 1 pass per helper = most reliable for rapid single-pass operations
                failed: false, // Track if timing completely unavailable
                limitReached: false // Track if we've hit the GPU QuerySet limit
              });
            }
            
            const pool = window.__webSightTimingHelperPools.get(device);
            
            // If timing previously failed completely, don't keep trying
            if (pool.failed) {
              return null;
            }
            
            // If we have available helpers in queue, use those first
            if (pool.available.length > 0) {
              const helper = pool.available.shift();
              pool.inUse.add(helper);
              return helper;
            }
            
            // If we haven't hit the limit yet, try to create new helper WITH GPU-level validation
            if (pool.helpers.length < pool.maxSize && !pool.limitReached) {
              try {
                // Push error scope BEFORE creating TimingHelper to catch GPU-level failures
                device.pushErrorScope('validation');
                const helper = new TimingHelper(device, pool.passesPerHelper);
                
                // Check for GPU-level errors synchronously-ish
                device.popErrorScope().then(error => {
                  if (error) {
                    // GPU rejected the QuerySet - mark pool as failed
                    console.error('[WebSight]  GPU QuerySet creation failed:', error.message);
                    pool.limitReached = true;
                    pool.failed = true;
                    // Remove the failed helper from pool
                    const idx = pool.helpers.indexOf(helper);
                    if (idx >= 0) pool.helpers.splice(idx, 1);
                    pool.inUse.delete(helper);
                  }
                });
                
                // Only add to pool if creation succeeded (no exception thrown)
                pool.helpers.push(helper);
                pool.inUse.add(helper);
                if (pool.helpers.length <= 5 || pool.helpers.length % 10 === 0) {
                  console.log(`[WebSight] Created TimingHelper ${pool.helpers.length}/${pool.maxSize} (${pool.passesPerHelper} passes/helper) for device`);
                }
                return helper;
              } catch (e) {
                if (e.message.includes('Cannot allocate') || e.message.includes('sample buffer') || e.message.includes('QuerySet')) {
                  pool.limitReached = true;
                  console.warn(`[WebSight] GPU QuerySet limit reached at ${pool.helpers.length} helpers.`);
                  console.warn(`[WebSight] Your GPU cannot allocate more QuerySets for timestamp queries.`);
                  console.log(`[WebSight] Workaround: Add "window.__webSightDisableGPUTiming = true;" before loading profiler-standalone.js`);
                  
                  if (pool.helpers.length === 0) {
                    pool.failed = true;
                    console.error('[WebSight] GPU timing completely unavailable. Profiler will continue without CPU timestamps.');
                    console.log(`[WebSight] Falling back to CPU timing. Set window.__webSightDisableGPUTiming = true to suppress this error.`);
                    profilerData.timingMode = 'cpu-only';
                    return null;
                  }
                  console.log(`[WebSight] Max concurrent timing operations: ${pool.helpers.length}. Render passes will continue without timing until helpers become available.`);
                  return null;
                } else {
                  console.warn('[WebSight] Cannot create TimingHelper:', e.message);
                  if (pool.helpers.length === 0) {
                    pool.failed = true;
                    console.error('[WebSight] GPU timing completely unavailable.');
                    return null;
                  }
                  return null;
                }
              }
            }
            
            if (pool.limitReached && pool.available.length === 0) {
              if (pool.index % 50 === 0) {
                console.warn(`[WebSight] All ${pool.helpers.length} timing helpers busy. Some passes will not be timed.`);
              }
              pool.index++;
            }
            
            return null; // No available helpers - caller handles graceful degradation
          };
          
          // Helper function to release a TimingHelper back to the pool after use
          const releaseTimingHelper = (device, helper) => {
            const pool = window.__webSightTimingHelperPools.get(device);
            if (!pool || !helper) return;
            
            pool.inUse.delete(helper);
            pool.available.push(helper);
          };
          
          // Hook createCommandEncoder
          const origCreateCommandEncoder = device.createCommandEncoder.bind(device);
          device.createCommandEncoder = function(desc) {
            const encoder = origCreateCommandEncoder(desc);
            const origBeginComputePass = encoder.beginComputePass.bind(encoder);
            const origBeginRenderPass = encoder.beginRenderPass.bind(encoder);
            const origFinish = encoder.finish.bind(encoder);

            // Track encoder creation for multi-GPU statistics
            device.__webSightInfo.encoderCount++;

            // Track encoder lifetime
            const encoderData = {
              dispatches: [],
              startTime: performance.now(),
              id: crypto.randomUUID(),
              passCount: 0, // Count passes for this encoder
              device: device, // Store device reference
              deviceLabel: device.__webSightInfo.label
            };
            profilerData.activeEncoders.set(encoder, encoderData);

            // Create NEW TimingHelper for THIS encoder (like primitive.mjs does)
            // We don't know how many passes in advance, so start with 1 and adjust per pass
            let encoderTimingHelper = null;
            let passTimings = []; // Collect pass timings manually
            
            // Skip GPU timing in minimal overhead mode
            if (profilerData.timingMode === 'gpu' && !profilerData.config.minimalOverhead) {
              try {
                // Don't use TimingHelper - it requires exact kernel count up front
                // Instead, track passes manually
                console.log(`[WebSight] GPU timing enabled for encoder "${desc?.label || 'unlabeled'}"`);
              } catch (e) {
                console.error(`[WebSight] GPU timing setup failed: ${e.message}`);
              }
            }

            // Create a proxy encoder with ORIGINAL beginComputePass for TimingHelper
            // Also include resolveQuerySet and copyBufferToBuffer which are used by #resolveTiming
            const proxyEncoder = {
              beginComputePass: origBeginComputePass,
              beginRenderPass: origBeginRenderPass,
              resolveQuerySet: encoder.resolveQuerySet.bind(encoder),
              copyBufferToBuffer: encoder.copyBufferToBuffer.bind(encoder)
            };

            encoder.beginComputePass = function(passDesc) {
              // Get or reuse a TimingHelper from pool (instead of creating unlimited new ones)
              let passTimingHelper = null;
              if (profilerData.timingMode === 'gpu') {
                passTimingHelper = getTimingHelper(device);
                if (passTimingHelper) {
                  passTimings.push(passTimingHelper);
                }
                // No error logging here - getTimingHelper already handles that
              }
              
              // Use TimingHelper's beginComputePass on PROXY encoder (prevents recursion)
              // If no timing helper available, just use original pass (graceful degradation)
              let pass;
              if (passTimingHelper) {
                try {
                  pass = passTimingHelper.beginComputePass(proxyEncoder, passDesc);
                } catch (e) {
                  // If beginComputePass fails (e.g., invalid QuerySet), fall back to regular pass
                  console.warn('[WebSight] TimingHelper.beginComputePass failed, continuing without timing:', e.message);
                  pass = origBeginComputePass(passDesc);
                  passTimingHelper = null; // Don't track this failed helper
                  passTimings.pop(); // Remove from passTimings array
                }
              } else {
                pass = origBeginComputePass(passDesc);
              }
              
              encoderData.passCount++;
              device.__webSightInfo.passCount++; // Track per-device
              
              pass.__dispatches = [];
              pass.__boundPipeline = null;
              pass.__boundBindGroups = {};
              pass.__timingHelper = passTimingHelper; // Store on pass for later
              pass.__passType = 'compute';
              pass.__deviceLabel = device.__webSightInfo.label;

              const origSetPipeline = pass.setPipeline.bind(pass);
              pass.setPipeline = function(p) {
                this.__boundPipeline = p;
                origSetPipeline(p);
              };

              const origSetBindGroup = pass.setBindGroup.bind(pass);
              pass.setBindGroup = function(i, bg) {
                this.__boundBindGroups[i] = bg;
                origSetBindGroup(i, bg);
              };

              const origDispatch = pass.dispatchWorkgroups.bind(pass);
              pass.dispatchWorkgroups = function(x, y, z) {
                const pipelineObj = this.__boundPipeline;
                const pipeline = pipelineObj?.__capture;
                
                if (!pipeline) {
                  console.warn('[WebSight] Dispatch without pipeline!');
                  origDispatch(x, y, z);
                  return;
                }
                
                // Get label from pipeline object or capture
                const pipelineLabel = pipelineObj?.label || pipeline?.label || 'compute_pipeline';

                const kernelId = generateKernelId(
                  pipeline.shader, 
                  pipeline.workgroupSize, 
                  pipelineLabel
                );

                if (!profilerData.kernels[kernelId]) {
                  profilerData.kernels[kernelId] = {
                    id: kernelId,
                    label: pipelineLabel,
                    workgroupSize: pipeline.workgroupSize,
                    shader: pipeline.shader,
                    stats: { count: 0, totalTime: 0, avgTime: 0, minTime: Infinity, maxTime: 0 }
                  };
                }

                // CRITICAL: Check dispatch dimensions BEFORE executing
                const maxDim = device.limits?.maxComputeWorkgroupsPerDimension || 65535;
                const dispatchX = x || 1;
                const dispatchY = y || 1;
                const dispatchZ = z || 1;
                
                let dimensionViolation = false;
                let violationMsg = '';
                
                // Check 1: Any dimension exceeds max limit
                if (dispatchX > maxDim) {
                  dimensionViolation = true;
                  violationMsg += `X dimension (${dispatchX}) exceeds limit ${maxDim}. `;
                }
                if (dispatchY > maxDim) {
                  dimensionViolation = true;
                  violationMsg += `Y dimension (${dispatchY}) exceeds limit ${maxDim}. `;
                }
                if (dispatchZ > maxDim) {
                  dimensionViolation = true;
                  violationMsg += `Z dimension (${dispatchZ}) exceeds limit ${maxDim}. `;
                }
                
                // Report dimension violations without trying to auto-correct
                if (dimensionViolation) {
                  const errorMsg = `DISPATCH GEOMETRY ERROR: ${violationMsg}`;
                  
                  profilerData.logs.push({
                    timestamp: Date.now(),
                    level: 'error',
                    category: 'dispatch-geometry',
                    message: errorMsg,
                    details: `Pipeline: "${pipelineLabel}"\nDispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}]\nWorkgroup: [${pipeline.workgroupSize?.join(', ') || '?'}]\nMax Allowed Per Dimension: ${maxDim}\n\nWARNING: This dispatch will be REJECTED by the GPU driver!\nRoot Cause: Likely nested if-inside-while in getSimpleDispatchGeometry(). Y dimension must be reduced in a separate while loop AFTER X is fully reduced.`
                  });
                  
                  // Trigger immediate broadcast for critical errors
                  broadcastData();
                }

                const cpuStart = performance.now();
                origDispatch(x, y, z);
                const cpuEnd = performance.now();
                const cpuTimeMs = cpuEnd - cpuStart;
                const cpuTimeNs = cpuTimeMs * 1000000;

                // Convert workgroupSize to array format (handles both object and array)
                let workgroupSizeArray = [1, 1, 1];
                if (pipeline.workgroupSize) {
                  if (Array.isArray(pipeline.workgroupSize)) {
                    workgroupSizeArray = pipeline.workgroupSize;
                  } else if (typeof pipeline.workgroupSize === 'object') {
                    workgroupSizeArray = [
                      pipeline.workgroupSize.x || 1,
                      pipeline.workgroupSize.y || 1,
                      pipeline.workgroupSize.z || 1
                    ];
                  }
                }

                const dispatchRecord = {
                  index: profilerData.dispatches.length,
                  kernelId: kernelId,
                  pipelineLabel: pipelineLabel,
                  workgroupSize: workgroupSizeArray,
                  dispatchSize: [x || 1, y || 1, z || 1],
                  x, y, z,
                  cpuStart,
                  cpuEnd,
                  cpuTimeNs: cpuTimeNs,
                  cpuTimeUs: cpuTimeNs / 1000,
                  cpuTimeMs: cpuTimeMs,
                  gpuTimeNs: cpuTimeNs, // Default to CPU time
                  gpuTimeUs: cpuTimeNs / 1000,
                  gpuTimeMs: cpuTimeMs,
                  timingSource: 'cpu_timing',
                  normalizedTime: normalizeTime(cpuTimeNs),
                  timeUnit: getTimeUnitLabel(),
                  timestampStart: -1,
                  timestampEnd: -1,
                  deviceLabel: device.__webSightInfo.label, // Multi-GPU tracking
                  passType: 'compute',
                  dimensionViolation: dimensionViolation, // Mark failed dispatches
                  bufferAccesses: Object.values(this.__boundBindGroups).flatMap(bg => 
                    bg.__capture?.entries.filter(e => e.resource?.id).map(e => ({
                      ...profilerData.buffers[e.resource.id],
                      binding: e.binding
                    })) || []
                  )
                };
                
                if (profilerData.config.enableWorkgroupAnalysis) {
                  const analysis = workgroupAnalyzer.analyzeDispatch(dispatchRecord);
                  if (analysis) {
                    dispatchRecord.occupancyAnalysis = analysis;
                    
                    // CRITICAL: Send to UI logs instead of console
                    const criticalIssues = analysis.issues.filter(i => i.severity === 'critical');
                    if (criticalIssues.length > 0) {
                      const criticalMsg = `CRITICAL DISPATCH ERROR for "${pipeline.label}"`;
                      const details = [
                        `Dispatch: [${dispatchRecord.dispatchSize.join(', ')}]`,
                        `Workgroup: [${dispatchRecord.workgroupSize.join(', ')}]`,
                        `Score: ${analysis.score}/100`,
                        '',
                        ...criticalIssues.map(issue => 
                          `${issue.message}\n   ${issue.impact}\n   FIX: ${issue.recommendation}`
                        )
                      ].join('\n   ');
                      
                      profilerData.logs.push({
                        timestamp: Date.now(),
                        level: 'error',
                        category: 'dispatch-geometry',
                        message: criticalMsg,
                        details: details
                      });
                      
                      // Also trigger immediate broadcast for critical issues
                      broadcastData();
                    } else if (analysis.score < 70) {
                      const warningMsg = `Suboptimal workgroup config for "${pipeline.label}" (Score: ${analysis.score})`;
                      const issueList = analysis.issues.map(i => i.message).join(', ');
                      
                      profilerData.logs.push({
                        timestamp: Date.now(),
                        level: 'warning',
                        category: 'workgroup-analysis',
                        message: warningMsg,
                        details: issueList
                      });
                    }
                  }
                }

                const kernel = profilerData.kernels[kernelId];
                if (kernel) {
                  kernel.stats.count++;
                  kernel.stats.totalTime += dispatchRecord.gpuTimeNs;
                  kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                  kernel.stats.minTime = Math.min(kernel.stats.minTime, dispatchRecord.gpuTimeNs);
                  kernel.stats.maxTime = Math.max(kernel.stats.maxTime, dispatchRecord.gpuTimeNs);
                }

                profilerData.dispatches.push(dispatchRecord);
                pass.__dispatches.push(dispatchRecord);
                encoderData.dispatches.push(dispatchRecord);
                device.__webSightInfo.dispatchCount++; // Track per-device

                dispatchRecord.bufferAccesses.forEach(b => {
                  if (b?.id) {
                    profilerData.bufferHeatMap[b.id] = (profilerData.bufferHeatMap[b.id] || 0) + 1;
                  }
                });

                broadcastData();
              };

              return pass;
            };

            // Hook beginRenderPass for graphical applications (same timing strategy as compute)
            encoder.beginRenderPass = function(passDesc) {
              try {
                // Get or reuse a TimingHelper from pool
                let passTimingHelper = null;
                if (profilerData.timingMode === 'gpu') {
                  passTimingHelper = getTimingHelper(device);
                  if (passTimingHelper) {
                    passTimings.push(passTimingHelper);
                  }
                  // No error logging here - getTimingHelper already handles that
                }

                // Use TimingHelper's beginRenderPass on PROXY encoder (prevents recursion)
                // If no timing helper available, just use original pass (graceful degradation)
                let pass;
                if (passTimingHelper) {
                  try {
                    pass = passTimingHelper.beginRenderPass(proxyEncoder, passDesc);
                  } catch (e) {
                    // If beginRenderPass fails (e.g., invalid QuerySet), fall back to regular pass
                    console.warn('[WebSight] TimingHelper.beginRenderPass failed, continuing without timing:', e.message);
                    pass = origBeginRenderPass(passDesc);
                    passTimingHelper = null; // Don't track this failed helper
                    passTimings.pop(); // Remove from passTimings array
                  }
                } else {
                  pass = origBeginRenderPass(passDesc);
                }
                
                encoderData.passCount++;
                device.__webSightInfo.passCount++; // Track per-device

                pass.__dispatches = [];
                pass.__boundBindGroups = {};
                pass.__passType = 'render';
                pass.__timingHelper = passTimingHelper; // Store for later timing retrieval
                pass.__deviceLabel = device.__webSightInfo.label;

                // Track render pass operations (draw calls instead of dispatches)
                const origSetPipeline = pass.setPipeline.bind(pass);
                pass.setPipeline = function(p) {
                  this.__boundPipeline = p;
                  origSetPipeline(p);
                };

                const origSetBindGroup = pass.setBindGroup.bind(pass);
                pass.setBindGroup = function(i, bg) {
                  this.__boundBindGroups[i] = bg;
                  origSetBindGroup(i, bg);
                };

                // Track draw calls as "dispatches" for consistency
                const origDraw = pass.draw.bind(pass);
                pass.draw = function(vertexCount, instanceCount, firstVertex, firstInstance) {
                  const pipeline = this.__boundPipeline?.__capture;
                  
                  const drawRecord = {
                    index: profilerData.dispatches.length,
                    type: 'draw',
                    vertexCount: vertexCount || 0,
                    instanceCount: instanceCount || 1,
                    firstVertex: firstVertex || 0,
                    firstInstance: firstInstance || 0,
                    pipelineLabel: pipeline?.label || 'unknown',
                    cpuStart: performance.now(),
                    cpuEnd: 0,
                    cpuTimeNs: 0,
                    gpuTimeNs: 0,
                    timingSource: 'render_pass_timing',
                    deviceLabel: device.__webSightInfo.label,
                    passType: 'render',
                    bufferAccesses: Object.values(this.__boundBindGroups).flatMap(bg => 
                      bg.__capture?.entries.filter(e => e.resource?.id).map(e => ({
                        ...profilerData.buffers[e.resource.id],
                        binding: e.binding
                      })) || []
                    )
                  };

                  origDraw(vertexCount, instanceCount, firstVertex, firstInstance);
                  
                  drawRecord.cpuEnd = performance.now();
                  drawRecord.cpuTimeNs = (drawRecord.cpuEnd - drawRecord.cpuStart) * 1000000;
                  drawRecord.gpuTimeNs = drawRecord.cpuTimeNs; // Default to CPU time

                  profilerData.dispatches.push(drawRecord);
                  pass.__dispatches.push(drawRecord);
                  encoderData.dispatches.push(drawRecord);
                  device.__webSightInfo.dispatchCount++; // Track per-device

                  broadcastData();
                };

                const origDrawIndexed = pass.drawIndexed.bind(pass);
                pass.drawIndexed = function(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
                  const pipeline = this.__boundPipeline?.__capture;
                  
                  const drawRecord = {
                    index: profilerData.dispatches.length,
                    type: 'drawIndexed',
                    indexCount: indexCount || 0,
                    instanceCount: instanceCount || 1,
                    firstIndex: firstIndex || 0,
                    baseVertex: baseVertex || 0,
                    firstInstance: firstInstance || 0,
                    pipelineLabel: pipeline?.label || 'unknown',
                    cpuStart: performance.now(),
                    cpuEnd: 0,
                    cpuTimeNs: 0,
                    gpuTimeNs: 0,
                    timingSource: 'render_pass_timing',
                    deviceLabel: device.__webSightInfo.label,
                    passType: 'render',
                    bufferAccesses: Object.values(this.__boundBindGroups).flatMap(bg => 
                      bg.__capture?.entries.filter(e => e.resource?.id).map(e => ({
                        ...profilerData.buffers[e.resource.id],
                        binding: e.binding
                      })) || []
                    )
                  };

                  origDrawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
                  
                  drawRecord.cpuEnd = performance.now();
                  drawRecord.cpuTimeNs = (drawRecord.cpuEnd - drawRecord.cpuStart) * 1000000;
                  drawRecord.gpuTimeNs = drawRecord.cpuTimeNs;

                  profilerData.dispatches.push(drawRecord);
                  pass.__dispatches.push(drawRecord);
                  encoderData.dispatches.push(drawRecord);
                  device.__webSightInfo.dispatchCount++; // Track per-device

                  broadcastData();
                };

                return pass;
              } catch (error) {
                console.error('[WebSight] Error in beginRenderPass hook:', error);
                return origBeginRenderPass.call(proxyEncoder, passDesc);
              }
            };

            encoder.finish = function(descriptor) {
              const commandBuffer = origFinish.call(this, descriptor);
              commandBuffer.__dispatches = encoderData.dispatches;
              commandBuffer.__encoderId = encoderData.id;
              commandBuffer.__passTimings = passTimings; // Store array of TimingHelpers (one per pass)
              commandBuffer.__passCount = encoderData.passCount;
              
              // Method 3: Expose timing API on command buffer for universal access
              commandBuffer.__gpuTiming = {
                available: false,
                passes: [],
                totalTimeNs: 0,
                
                // Promise that resolves when timing is ready
                ready: new Promise((resolve) => {
                  commandBuffer.__gpuTimingResolve = resolve;
                })
              };
              
              return commandBuffer;
            };

            return encoder;
          };

          // Hook queue.submit
          const origSubmit = device.queue.submit.bind(device.queue);
          
          device.queue.submit = function(cmds) {
            // Collect dispatches and timing helpers from command buffers
            let dispatchesInSubmit = [];
            let passTimingHelpers = [];
            
            for (const cmd of cmds) {
              if (cmd.__dispatches) {
                dispatchesInSubmit.push(...cmd.__dispatches);
              }
              if (cmd.__passTimings) {
                passTimingHelpers.push(...cmd.__passTimings);
              }
            }
            
            const result = origSubmit(cmds);
            
            // Get GPU timing results AFTER submission completes
            if (passTimingHelpers.length > 0) {
              // Track which command buffers we're timing
              const cmdBuffersWithTiming = cmds.filter(cmd => cmd.__gpuTiming);
              
              device.queue.onSubmittedWorkDone().then(async () => {
                try {
                  // Get results from each TimingHelper (each has 1 pass)
                  const allDurations = [];
                  for (const helper of passTimingHelpers) {
                    if (!helper) {
                      allDurations.push(0n); // No helper was available
                      continue;
                    }
                    
                    try {
                      const durations = await helper.getResult();
                      allDurations.push(...durations);
                      
                      releaseTimingHelper(device, helper);
                    } catch (e) {
                      console.warn('[WebSight] Failed to get timing from one pass:', e.message);
                      allDurations.push(0n);
                      
                      releaseTimingHelper(device, helper);
                    }
                  }
                  
                  const nonZeroCount = allDurations.filter(d => d > 0n).length;
                  if (nonZeroCount > 0) {
                    console.log(`[WebSight] Got GPU timing for ${nonZeroCount}/${allDurations.length} passes:`, allDurations.map(d => `${(Number(d)/1000000).toFixed(3)}ms`));
                  }
                  
                  // Accumulate ALL timings for Method 2 (direct access)
                  // Add memory leak protection - limit total stored results
                  window.__webSightGlobalTimingResults.push(...allDurations);
                  if (window.__webSightGlobalTimingResults.length > window.__webSightMaxTimingResults) {
                    // Keep only the most recent results
                    const excess = window.__webSightGlobalTimingResults.length - window.__webSightMaxTimingResults;
                    window.__webSightGlobalTimingResults.splice(0, excess);
                    console.warn(`[WebSight] Timing results buffer full (${window.__webSightMaxTimingResults}). Oldest ${excess} results discarded.`);
                  }
                  
                  // Method 3: Populate command buffer timing
                  if (cmdBuffersWithTiming.length > 0) {
                    const totalTimeNs = allDurations.reduce((sum, t) => sum + Number(t), 0);
                    
                    cmdBuffersWithTiming.forEach(cmd => {
                      cmd.__gpuTiming.available = true;
                      cmd.__gpuTiming.passes = allDurations.map(d => Number(d));
                      cmd.__gpuTiming.totalTimeNs = totalTimeNs;
                      cmd.__gpuTiming.totalTimeMs = totalTimeNs / 1000000;
                      
                      // Resolve the ready promise
                      if (cmd.__gpuTimingResolve) {
                        cmd.__gpuTimingResolve(cmd.__gpuTiming);
                      }
                    });
                  }
                  
                  // Method 4: Fire timing event
                  if (window.__webSightTimingEvents) {
                    window.__webSightTimingEvents.dispatchEvent(new CustomEvent('timing', {
                      detail: {
                        passes: allDurations.map(d => Number(d)),
                        totalTimeNs: allDurations.reduce((sum, t) => sum + Number(t), 0),
                        commandBuffers: cmdBuffersWithTiming.length
                      }
                    }));
                  }
                  
                  // Update dispatch records for profiler UI
                  if (dispatchesInSubmit.length > 0) {
                    for (let i = 0; i < Math.min(allDurations.length, dispatchesInSubmit.length); i++) {
                      const dispatch = dispatchesInSubmit[i];
                      const gpuTimeNs = Number(allDurations[i]);
                      
                      // Update dispatch with GPU timing
                      dispatch.gpuTimeNs = gpuTimeNs;
                      dispatch.gpuTimeUs = gpuTimeNs / 1000;
                      dispatch.gpuTimeMs = gpuTimeNs / 1000000;
                      dispatch.normalizedTime = normalizeTime(gpuTimeNs);
                      dispatch.timingSource = 'gpu_timestamp';
                      
                      // Update kernel stats
                      const kernel = profilerData.kernels[dispatch.kernelId];
                      if (kernel) {
                        // Replace CPU time with GPU time in totals
                        kernel.stats.totalTime = (kernel.stats.totalTime - dispatch.cpuTimeNs) + gpuTimeNs;
                        kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                        kernel.stats.minTime = Math.min(kernel.stats.minTime, gpuTimeNs);
                        kernel.stats.maxTime = Math.max(kernel.stats.maxTime, gpuTimeNs);
                      }
                    }
                  }
                  
                  broadcastData();
                } catch (e) {
                  console.error('[WebSight] Failed to get GPU timing:', e);
                }
              }).catch(e => {
                console.error('[WebSight] onSubmittedWorkDone failed:', e);
              });
            }
            
            return result;
          };

          if (!window.__WebSightTimingHelper) {
            window.__webSightTimingHelper_executionId = 0;
            
            window.__WebSightTimingHelper = {
              async getResult() {
                const results = window.__webSightGlobalTimingResults || [];
                const total = results.reduce((a,b) => a + Number(b), 0) / 1000000; // Convert to ms
                console.log(`[WebSight] Application requested timing: ${results.length} passes, ${total.toFixed(3)}ms total`);
                
                // Return copy and clear for next execution
                const returnValue = [...results];
                window.__webSightGlobalTimingResults = [];
                
                return returnValue;
              },
              
              reset(numKernels) {
                console.log(`[WebSight] Timing helper reset (expecting ${numKernels} kernels) - data will be cleared on next getResult()`);
              }
            };
            
            // Method 4: Event-based timing notifications (for real-time apps)
            window.__webSightTimingEvents = new EventTarget();
            
            log('[WebSight] GPU Timing enabled - Multiple access methods:');
            log('  1. window.__WebSightTimingHelper.getResult() - for primitive.mjs');
            log('  2. window.__webSightGlobalTimingResults - direct access array');
            log('  3. commandBuffer.__gpuTiming - per-command timing');
            log('  4. window.__webSightTimingEvents.addEventListener("timing") - events');
          }

          addLog('WebGPU hooks installed');
          return device;
          
        } catch (e) {
          addLog(`Device creation failed: ${e.message}`, 'error');
          throw e;
        }
      };
      return adapter;
    };
  }

  if (typeof window !== 'undefined') {
    window.WebSight = {
      getData: () => profilerData,
      
      getMultiGPUStats: () => {
        return {
          adapters: window.__webSightAdapters?.map((a, i) => ({
            index: i,
            hasTimestampFeature: a.hasTimestampFeature,
            powerPreference: a.options?.powerPreference || 'default',
            deviceCount: a.devices.length,
            requestedAt: a.requestedAt
          })) || [],
          
          devices: window.__webSightDevices?.map((d, i) => ({
            index: i,
            label: d.label,
            hasTimestampQuery: d.hasTimestampQuery,
            timingMode: d.timingMode,
            encoderCount: d.encoderCount,
            passCount: d.passCount,
            dispatchCount: d.dispatchCount,
            features: d.features,
            limits: {
              maxComputeWorkgroupsPerDimension: d.limits.maxComputeWorkgroupsPerDimension,
              maxComputeInvocationsPerWorkgroup: d.limits.maxComputeInvocationsPerWorkgroup,
              maxStorageBufferBindingSize: d.limits.maxStorageBufferBindingSize,
              maxBufferSize: d.limits.maxBufferSize
            },
            createdAt: d.createdAt
          })) || [],
          
          pools: (() => {
            const pools = [];
            window.__webSightTimingHelperPools?.forEach((pool, device) => {
              const deviceInfo = device.__webSightInfo;
              pools.push({
                deviceLabel: deviceInfo?.label || 'unknown',
                poolSize: pool.helpers.length,
                maxSize: pool.maxSize,
                available: pool.available?.length || 0,
                inUse: pool.inUse?.size || 0,
                currentIndex: pool.index,
                failed: pool.failed,
                limitReached: pool.limitReached || false,
                mode: pool.limitReached ? 'limited (waiting)' : 'growing',
                utilizationPercent: pool.helpers.length > 0 ? 
                  Math.round(((pool.inUse?.size || 0) / pool.helpers.length) * 100) : 0
              });
            });
            return pools;
          })(),
          
          // Aggregate statistics
          totals: {
            adapterCount: window.__webSightAdapters?.length || 0,
            deviceCount: window.__webSightDevices?.length || 0,
            totalEncoders: window.__webSightDevices?.reduce((sum, d) => sum + d.encoderCount, 0) || 0,
            totalPasses: window.__webSightDevices?.reduce((sum, d) => sum + d.passCount, 0) || 0,
            totalDispatches: window.__webSightDevices?.reduce((sum, d) => sum + d.dispatchCount, 0) || 0
          }
        };
      },
      
      clear: () => { 
        profilerData.dispatches = []; 
        profilerData.logs = [];
        profilerData.kernels = {};
        // Note: Each encoder has its own TimingHelper now
        addLog('Profiler cleared');
      },
      
      start: hookWebGPU,
      
      // Configuration
      configure: (options) => {
        if (options.broadcastEnabled !== undefined) {
          profilerData.config.broadcastEnabled = options.broadcastEnabled;
          console.log(`[WebSight] Broadcasting ${options.broadcastEnabled ? 'enabled' : 'disabled'}`);
        }
        
        if (options.broadcastDebounceMs !== undefined) {
          profilerData.config.broadcastDebounceMs = options.broadcastDebounceMs;
        }
        
        if (options.timeUnit !== undefined && ['ns', 'us', 'ms'].includes(options.timeUnit)) {
          profilerData.config.normalizeTimeUnit = options.timeUnit;
          console.log(`[WebSight] Time unit: ${options.timeUnit}`);
        }
        
        if (options.minimalOverhead !== undefined) {
          profilerData.config.minimalOverhead = options.minimalOverhead;
          if (options.minimalOverhead) {
            // Disable GPU timing and increase broadcast interval
            profilerData.timingMode = 'cpu';
            profilerData.config.broadcastDebounceMs = 10000; // 10 seconds
            console.log('[WebSight] MINIMAL OVERHEAD MODE: GPU timing disabled, broadcast interval 10s');
          } else {
            profilerData.timingMode = 'gpu';
            profilerData.config.broadcastDebounceMs = 1000;
            console.log('[WebSight] Normal mode: GPU timing enabled, broadcast interval 1s');
          }
        }
        
        // Note: maxPasses and allowDynamicGrowth are now per-encoder, not global
        
        return {
          broadcastEnabled: profilerData.config.broadcastEnabled,
          broadcastDebounceMs: profilerData.config.broadcastDebounceMs,
          timeUnit: profilerData.config.normalizeTimeUnit,
          minimalOverhead: profilerData.config.minimalOverhead
        };
      },
      
      benchmarkMode: () => {
        profilerData.config.minimalOverhead = true;
        profilerData.timingMode = 'cpu';
        profilerData.config.broadcastEnabled = false;
        console.log('[WebSight] BENCHMARK MODE: Minimal overhead, no broadcasts, CPU timing only');
      },
      
      normalMode: () => {
        profilerData.config.minimalOverhead = false;
        profilerData.timingMode = 'gpu';
        profilerData.config.broadcastEnabled = true;
        profilerData.config.broadcastDebounceMs = 1000;
        console.log('[WebSight] NORMAL MODE: GPU timing enabled, broadcasts active');
      },
      
      getStats: () => {
        const dispatches = profilerData.dispatches || [];
        const validGpuTimes = dispatches
          .filter(d => d.timingSource === 'gpu_timestamp')
          .map(d => d.gpuTimeNs);
        
        const unit = getTimeUnitLabel();
        
        return {
          totalDispatches: dispatches.length,
          gpuTimedDispatches: validGpuTimes.length,
          cpuFallbackDispatches: dispatches.filter(d => d.timingSource === 'cpu_timing').length,
          avgGpuTime: validGpuTimes.length > 0 
            ? normalizeTime(validGpuTimes.reduce((a, b) => a + b, 0) / validGpuTimes.length)
            : 0,
          totalGpuTime: validGpuTimes.length > 0
            ? normalizeTime(validGpuTimes.reduce((a, b) => a + b, 0))
            : 0,
          minGpuTime: validGpuTimes.length > 0
            ? normalizeTime(Math.min(...validGpuTimes))
            : 0,
          maxGpuTime: validGpuTimes.length > 0
            ? normalizeTime(Math.max(...validGpuTimes))
            : 0,
          timeUnit: unit
        };
      },
      
      getTimingHelperStats: () => {
        return { message: 'Each encoder now has its own TimingHelper. Stats are per-encoder.' };
      },
      
      listKernels: () => {
        return Object.values(profilerData.kernels).map(k => ({
          id: k.id,
          label: k.label || 'Unnamed Kernel',
          workgroupSize: k.workgroupSize || { x: 0, y: 0, z: 0 },
          dispatchCount: k.stats?.count || 0,
          avgTime: normalizeTime(k.stats?.avgTime || 0),
          totalTime: normalizeTime(k.stats?.totalTime || 0),
          minTime: normalizeTime(k.stats?.minTime === Infinity ? 0 : k.stats?.minTime || 0),
          maxTime: normalizeTime(k.stats?.maxTime || 0),
          timeUnit: getTimeUnitLabel()
        }));
      },
      
      export: () => {
        const dataStr = JSON.stringify(profilerData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `websight-profile-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        addLog('Profile exported');
      },
      
      // =========================================================================
      // ADVANCED ANALYSIS API
      // =========================================================================
      
      // Memory Leak Detection
      getMemoryLeaks: () => {
        // Run leak check immediately before getting report
        memoryLeakDetector.checkForLeaks();
        
        const report = memoryLeakDetector.getLeakReport();
        
        console.log('\n [WebSight] Memory Leak Report');
        console.log('═'.repeat(60));
        console.log(`Total Resources Created: ${report.stats.createdCount}`);
        console.log(`Total Resources Destroyed: ${report.stats.destroyedCount}`);
        console.log(`Active Resources: ${report.summary.activeResources}`);
        console.log(`Potential Leaks: ${report.summary.totalLeaks}`);
        console.log(`Leak Rate: ${report.summary.leakRate}`);
        console.log(`Current Memory: ${memoryLeakDetector.formatBytes(report.stats.currentMemory)}`);
        console.log(`Peak Memory: ${memoryLeakDetector.formatBytes(report.stats.peakMemory)}`);
        
        if (report.leaks.length > 0) {
          console.log('\n Detected Leaks:');
          console.table(report.leaks.slice(0, 10).map(l => ({
            Type: l.type,
            Size: memoryLeakDetector.formatBytes(l.size),
            Age: memoryLeakDetector.formatTime(l.age),
            Label: l.label
          })));
        } else {
          console.log('\nNo memory leaks detected!');
        }
        
        return report;
      },
      
      getMemoryStats: () => memoryLeakDetector.stats,
      
      // Workgroup Occupancy Analysis
      getWorkgroupAnalysis: () => {
        // Analyze all dispatches on-demand
        console.log('\n [WebSight] Analyzing workgroup configurations...');
        
        // Analyze all recorded dispatches
        profilerData.dispatches.forEach(dispatch => {
          if (dispatch.workgroupSize && dispatch.dispatchSize) {
            workgroupAnalyzer.analyzeDispatch(dispatch);
          }
        });
        
        const summary = workgroupAnalyzer.getSummary();
        const analyses = workgroupAnalyzer.getAllAnalyses();
        
        console.log('\n [WebSight] Workgroup Occupancy Report');
        console.log('═'.repeat(60));
        
        if (summary) {
          console.log(`Total Kernels Analyzed: ${summary.totalKernels}`);
          console.log(`Average Score: ${summary.averageScore}/100 (Grade: ${summary.grade})`);
          console.log(`Critical Issues: ${summary.criticalIssues}`);
          console.log(`High Priority Issues: ${summary.highIssues}`);
          
          if (summary.needsAttention.length > 0) {
            console.log(`\n ${summary.needsAttention.length} Kernel(s) Need Attention:`);
            summary.needsAttention.forEach(a => {
              console.log(`\n  Kernel: ${a.kernelId}`);
              console.log(`  Score: ${a.score}/100`);
              console.log(`  Workgroup Size: [${a.workgroupSize.join(', ')}] = ${a.totalThreads} threads`);
              console.log(`  Dispatch Size: [${a.dispatchSize.join(', ')}] = ${a.totalWorkgroups} workgroups`);
              console.log(`  Total Invocations: ${a.totalInvocations.toLocaleString()}`);
              console.log(`  Potential Speedup: ${a.potentialSpeedup}`);
              console.log('  Issues:');
              a.issues.forEach(issue => {
                console.log(`    - [${issue.severity.toUpperCase()}] ${issue.message}`);
                console.log(`      Impact: ${issue.impact}`);
                console.log(`      Fix: ${issue.recommendation}`);
              });
            });
          } else {
            console.log('\n All workgroup configurations look good!');
          }
        } else {
          console.log('\n No dispatches to analyze yet. Run your WebGPU code first.');
        }
        
        return { summary, analyses };
      },
      
      // Shader Complexity Analysis
      getShaderAnalysis: () => {
        // Analyze all shaders on-demand
        console.log('\n [WebSight] Analyzing shader complexity...');
        
        // Analyze all kernels' shaders
        Object.values(profilerData.kernels).forEach(kernel => {
          if (kernel.shaderId && kernel.shader && !shaderAnalyzer.analyses.has(kernel.shaderId)) {
            shaderAnalyzer.analyzeShader(kernel.shaderId, kernel.shader);
          }
        });
        
        const summary = shaderAnalyzer.getSummary();
        const analyses = shaderAnalyzer.getAllAnalyses();
        
        console.log('\n [WebSight] Shader Complexity Report');
        console.log('═'.repeat(60));
        
        if (summary) {
          console.log(`Total Shaders Analyzed: ${summary.totalShaders}`);
          console.log(`Average Score: ${summary.averageScore}/100`);
          console.log(`Overall Grade: ${summary.overallGrade.letter} (${summary.overallGrade.desc})`);
          console.log(`Average Complexity: ${summary.averageComplexity}`);
          console.log(`Critical Issues: ${summary.criticalIssues}`);
          
          if (summary.needsOptimization.length > 0) {
            console.log(`\n🔧 ${summary.needsOptimization.length} Shader(s) Need Optimization:`);
            summary.needsOptimization.forEach(a => {
              console.log(`\n  Shader ID: ${a.shaderId}`);
              console.log(`  Score: ${a.score}/100 (Grade: ${a.grade.letter})`);
              console.log(`  Complexity: ${a.complexity.toFixed(1)}`);
              console.log(`  Lines of Code: ${a.lineCount}`);
              console.log(`  Potential Speedup: ${a.potentialSpeedup}`);
              console.log('  Metrics:');
              console.log(`    - Branches: ${a.metrics.branches}`);
              console.log(`    - Loops: ${a.metrics.loops}`);
              console.log(`    - Math Ops: ${a.metrics.mathOps}`);
              console.log(`    - Memory Ops: ${a.metrics.memoryOps}`);
              console.log(`    - Atomic Ops: ${a.metrics.atomicOps}`);
              console.log(`    - Register Pressure: ${a.metrics.registerPressure}`);
              console.log('  Issues:');
              a.issues.forEach(issue => {
                console.log(`    - [${issue.severity.toUpperCase()}] ${issue.message}`);
                console.log(`      Impact: ${issue.impact}`);
                console.log(`      Fix: ${issue.recommendation}`);
              });
            });
          }
        }
        
        return { summary, analyses };
      },
      
      // Get specific shader analysis by ID
      analyzeShader: (shaderId) => {
        const analysis = shaderAnalyzer.analyses.get(shaderId);
        if (!analysis) {
          console.warn(`[WebSight] No analysis found for shader: ${shaderId}`);
          return null;
        }
        
        console.log(`\n Shader Analysis: ${shaderId}`);
        console.log('═'.repeat(60));
        console.log(`Score: ${analysis.score}/100 (${analysis.grade.letter})`);
        console.log(`Complexity: ${analysis.complexity.toFixed(1)}`);
        console.log(`Lines: ${analysis.lineCount}`);
        console.log(`Potential Speedup: ${analysis.potentialSpeedup}`);
        
        if (analysis.issues.length > 0) {
          console.log('\nIssues:');
          analysis.issues.forEach(issue => {
            console.log(`  [${issue.severity.toUpperCase()}] ${issue.type}`);
            console.log(`  ${issue.message}`);
            console.log(`   Impact: ${issue.impact}`);
            console.log(`   Fix: ${issue.recommendation}`);
            if (issue.example) {
              console.log(`    Example: ${issue.example}`);
            }
          });
        }
        
        return analysis;
      },
      
      // Get comprehensive analysis report
      getFullAnalysisReport: () => {
        console.log('\n🚀 [WebSight] COMPREHENSIVE ANALYSIS REPORT');
        console.log('═'.repeat(80));
        
        const memoryReport = memoryLeakDetector.getLeakReport();
        const workgroupReport = workgroupAnalyzer.getSummary();
        const shaderReport = shaderAnalyzer.getSummary();
        
        console.log('\n SUMMARY');
        console.log('-'.repeat(80));
        console.log(`Memory: ${memoryLeakDetector.formatBytes(memoryReport.stats.currentMemory)} / Peak: ${memoryLeakDetector.formatBytes(memoryReport.stats.peakMemory)}`);
        console.log(`Potential Leaks: ${memoryReport.summary.totalLeaks} (${memoryReport.summary.leakRate})`);
        
        if (workgroupReport) {
          console.log(`Workgroup Optimization: Grade ${workgroupReport.grade} (${workgroupReport.averageScore}/100)`);
          console.log(`Critical Workgroup Issues: ${workgroupReport.criticalIssues}`);
        }
        
        if (shaderReport) {
          console.log(`Shader Optimization: Grade ${shaderReport.overallGrade.letter} (${shaderReport.averageScore}/100)`);
          console.log(`Critical Shader Issues: ${shaderReport.criticalIssues}`);
        }
        
        console.log('\n✨ Call specific methods for detailed reports:');
        console.log('  - WebSight.getMemoryLeaks()');
        console.log('  - WebSight.getWorkgroupAnalysis()');
        console.log('  - WebSight.getShaderAnalysis()');
        
        return {
          memory: memoryReport,
          workgroup: workgroupReport,
          shader: shaderReport,
          timestamp: Date.now()
        };
      }
    };
    
    // Initialize device limits when adapter is available
    window.addEventListener('load', () => {
      if (profilerData.gpuCharacteristics?.limits) {
        workgroupAnalyzer.setDeviceLimits(profilerData.gpuCharacteristics.limits);
      }
      
      // Auto-launch profiler UI window (unless disabled or already UI window)
      if (!window.__webSightDisableAutoUI && !window.__webSightIsUIWindow) {
        const scripts = document.querySelectorAll('script[src*="profiler-standalone.js"]');
        let uiPath = 'index.html';
        if (scripts.length > 0) {
          const scriptSrc = scripts[0].src;
          const scriptDir = scriptSrc.substring(0, scriptSrc.lastIndexOf('/') + 1);
          uiPath = scriptDir + 'index.html';
        }
        
        const profilerWindow = window.open(uiPath, 'WebSightProfiler', 'width=1400,height=900');
        if (!profilerWindow) {
          // Only warn if popup blocked
          console.warn(`[WebSight] Could not open profiler UI (popup blocked?). Manually open: ${uiPath}`);
        }
      }
    });
    
    hookWebGPU();
    addLog('WebSight initialized');
  }
})();