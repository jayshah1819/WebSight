// WebGPU Compute Profiler - Production Ready with Accurate Timestamps

(function() {
  'use strict';

  // Simple TimingHelper class (like webgpufundamentals)
  class TimingHelper {
    constructor(device, origSubmit) {
      this.device = device;
      this.origSubmit = origSubmit; // Store ORIGINAL submit to avoid recursion
      this.querySetPool = []; // Pool of reusable QuerySets
      this.maxPoolSize = 128; // Very large pool for extreme stress tests (100+ rapid submits)
      this.activeQuerySets = new Set(); // Track QuerySets currently in use
      this.poolExhaustedWarned = false; // Track if we've warned about pool exhaustion
      
      // Pre-allocate a few QuerySets to avoid allocation overhead
      this.preallocateQuerySets(16);
    }
    
    preallocateQuerySets(count) {
      for (let i = 0; i < count && this.querySetPool.length < this.maxPoolSize; i++) {
        try {
          const qs = this.device.createQuerySet({
            type: 'timestamp',
            count: 2
          });
          this.querySetPool.push(qs);
        } catch (e) {
          console.warn(`[WebSight] Failed to preallocate QuerySet ${i+1}/${count}:`, e.message);
          break;
        }
      }
      if (this.querySetPool.length > 0) {
        console.log(`[WebSight] Pre-allocated ${this.querySetPool.length} QuerySets`);
      }
    }

    getQuerySet() {
      // Try to get a free QuerySet from the pool
      let querySet = this.querySetPool.find(qs => !this.activeQuerySets.has(qs));
      
      // If no free QuerySet and pool isn't full, create a new one
      if (!querySet && this.querySetPool.length < this.maxPoolSize) {
        try {
          querySet = this.device.createQuerySet({
            type: 'timestamp',
            count: 2
          });
          this.querySetPool.push(querySet);
          console.log(`[WebSight] Created QuerySet ${this.querySetPool.length}/${this.maxPoolSize}`);
        } catch (e) {
          console.warn('[WebSight] Failed to create QuerySet:', e.message);
          return null;
        }
      }
      
      // If still no QuerySet available, we need to wait or return null
      // Better to return null and skip timing than to corrupt data with reuse
      if (!querySet) {
        const exhaustedCount = (this.poolExhaustedCount || 0) + 1;
        this.poolExhaustedCount = exhaustedCount;
        
        if (!this.poolExhaustedWarned || exhaustedCount % 10 === 0) {
          console.warn(`[WebSight] QuerySet pool exhausted (${this.activeQuerySets.size}/${this.maxPoolSize} active, ${exhaustedCount} times). Some dispatches will use CPU timing.`);
          this.poolExhaustedWarned = true;
        }
        return null;
      }
      
      if (querySet) {
        this.activeQuerySets.add(querySet);
      }
      
      return querySet;
    }

    releaseQuerySet(querySet) {
      this.activeQuerySets.delete(querySet);
    }

    beginComputePass(encoder, descriptor, origBeginComputePass) {
      // Get a QuerySet from the pool
      const querySet = this.getQuerySet();
      
      if (!querySet) {
        console.warn('[WebSight] No QuerySet available, pass will not have GPU timing');
        return origBeginComputePass.call(encoder, descriptor);
      }
      
      const passDesc = {
        ...descriptor,
        timestampWrites: {
          querySet: querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1
        }
      };
      
      // Call ORIGINAL beginComputePass for THIS encoder!
      const pass = origBeginComputePass.call(encoder, passDesc);
      
      // Store querySet on the pass so we can read it later
      pass.__querySet = querySet;
      
      return pass;
    }

    async getResult(querySet) {
      // Read timestamps from the specific querySet for this pass
      const resolveBuffer = this.device.createBuffer({
        size: 2 * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
      });
      const readBuffer = this.device.createBuffer({
        size: 2 * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      
      const encoder = this.device.createCommandEncoder();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
      
      // Use original submit to avoid recursion!
      this.origSubmit([encoder.finish()]);
      
      // Release QuerySet IMMEDIATELY after submitting the resolve command
      // The GPU has captured the timestamp data, so the QuerySet can be reused
      this.releaseQuerySet(querySet);
      
      // Now wait for the data to be read from GPU
      await this.device.queue.onSubmittedWorkDone();
      
      await readBuffer.mapAsync(GPUMapMode.READ);
      const times = new BigUint64Array(readBuffer.getMappedRange());
      const timestampStart = Number(times[0]);
      const timestampEnd = Number(times[1]);
      const gpuTime = timestampEnd - timestampStart;
      readBuffer.unmap();
      
      // Clean up temporary buffers
      resolveBuffer.destroy();
      readBuffer.destroy();
      
      return { timestampStart, timestampEnd, gpuTime }; // Return timestamps and duration in nanoseconds
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
    pendingTimestamps: [] // Track which dispatches need timestamp resolution
  };

  const profilerChannel = new BroadcastChannel('websight-profiler');
  let broadcastTimer = null;

  function broadcastData() {
    if (broadcastTimer) return;
    broadcastTimer = setTimeout(() => {
      try {
        profilerChannel.postMessage({
          type: 'profiler-update',
          data: {
            dispatches: profilerData.dispatches,
            pipelines: profilerData.pipelines,
            buffers: profilerData.buffers,
            kernels: profilerData.kernels,
            logs: profilerData.logs,
            gpuCharacteristics: profilerData.gpuCharacteristics,
            runs: profilerData.runs,
            runId: profilerData.runId,
            timestamp: Date.now()
          }
        });
      } catch (e) {
        console.error('[WebSight] Broadcast failed:', e);
      }
      broadcastTimer = null;
    }, 100);
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
    console.log(`[WebSight] ${message}`);
    broadcastData();
  }

  async function hookWebGPU() {
    if (!navigator.gpu) {
      addLog('WebGPU not available', 'error');
      return;
    }

    const originalRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    
    navigator.gpu.requestAdapter = async function(options) {
      console.log('[WebSight] Adapter requested with options:', options);
      const adapter = await originalRequestAdapter(options);
      if (!adapter) {
        console.error('[WebSight] ERROR: No adapter returned!');
        return adapter;
      }

      const hasTimestampFeature = adapter.features.has('timestamp-query');
      const hasTimestampInPasses = adapter.features.has('chromium-experimental-timestamp-query-inside-passes');
      
      console.log('[WebSight] Adapter features:', Array.from(adapter.features).join(', '));
      console.log('[WebSight] timestamp-query:', hasTimestampFeature);
      console.log('[WebSight] timestamp-query-inside-passes:', hasTimestampInPasses);
      
      addLog(`Timestamp support: ${hasTimestampFeature}, In-pass: ${hasTimestampInPasses}`);

      const originalRequestDevice = adapter.requestDevice.bind(adapter);
      
      adapter.requestDevice = async function(descriptor) {
        console.log('[WebSight] Device requested');
        
        try {
          const requiredFeatures = new Set(descriptor?.requiredFeatures || []);
          
          if (hasTimestampFeature) {
            requiredFeatures.add('timestamp-query');
            console.log('[WebSight] Enabling timestamp-query');
          }
          
          if (hasTimestampInPasses) {
            requiredFeatures.add('chromium-experimental-timestamp-query-inside-passes');
            console.log('[WebSight] Enabling timestamp-query-inside-passes');
          }
          
          const modifiedDescriptor = {
            ...descriptor,
            requiredFeatures: Array.from(requiredFeatures)
          };
          
          const device = await originalRequestDevice(modifiedDescriptor);
          
          const hasTimestampQuery = device.features.has('timestamp-query');
          
          profilerData.timingMode = hasTimestampQuery ? 'gpu' : 'cpu-only';
          
          console.log('[WebSight] Device features:', Array.from(device.features).join(', '));
          console.log('[WebSight] GPU timing available:', hasTimestampQuery);
          
          addLog(`Device created - Timing mode: ${profilerData.timingMode}`);

        // TimingHelper will be created later in encoder hook where we have origBeginComputePass

        // Hook createShaderModule FIRST (before pipeline creation)
        const origCreateShaderModule = device.createShaderModule.bind(device);
        device.createShaderModule = function(desc) {
          const module = origCreateShaderModule(desc);
          module.__source = desc.code;
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
            analysis: analyzeWGSL(source)
          };
          
          profilerData.pipelines[pipeline.__capture.id] = pipeline.__capture;
          
          // Pre-register kernel
          const kernelId = generateKernelId(source, workgroupSize, label);
          if (!profilerData.kernels[kernelId]) {
            profilerData.kernels[kernelId] = {
              id: kernelId,
              label: label,
              workgroupSize: workgroupSize,
              shader: source,
              stats: { count: 0, totalTime: 0, avgTime: 0, minTime: Infinity, maxTime: 0 }
            };
            addLog(`Kernel registered: ${label} (${kernelId})`);
          }
          
          return pipeline;
        };

        // Hook createBuffer
        const origCreateBuffer = device.createBuffer.bind(device);
        device.createBuffer = function(desc) {
          const buffer = origCreateBuffer(desc);
          buffer.__capture = {
            id: crypto.randomUUID(),
            label: desc.label || 'buffer',
            size: desc.size,
            usage: desc.usage
          };
          profilerData.buffers[buffer.__capture.id] = buffer.__capture;
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

        // Hook createCommandEncoder
        const origCreateCommandEncoder = device.createCommandEncoder.bind(device);
        device.createCommandEncoder = function(desc) {
          const encoder = origCreateCommandEncoder(desc);
          const origBeginComputePass = encoder.beginComputePass.bind(encoder);
          const origFinish = encoder.finish.bind(encoder);

          // Track dispatches for this encoder
          encoder.__allDispatches = [];

          // Store original submit BEFORE creating TimingHelper
          const origSubmit = device.queue.submit.bind(device.queue);

          // Create TimingHelper on first encoder creation if we don't have one yet
          if (!profilerData.timingHelper && profilerData.timingMode === 'gpu') {
            try {
              profilerData.timingHelper = new TimingHelper(device, origSubmit);
              addLog('TimingHelper created successfully');
            } catch (e) {
              addLog(`Failed to create TimingHelper: ${e.message}`, 'error');
              profilerData.timingHelper = null;
              profilerData.timingMode = 'cpu-only';
            }
          }

          // Hook finish() to track which dispatches are in this command buffer
          encoder.finish = function() {
            const commandBuffer = origFinish();
            // Store reference to dispatches and querySet from this encoder
            commandBuffer.__dispatches = encoder.__allDispatches;
            commandBuffer.__querySet = encoder.__querySet;
            return commandBuffer;
          };

          encoder.beginComputePass = function(passDesc) {
            // Use TimingHelper if available, passing the original function for THIS encoder
            const pass = profilerData.timingHelper ?
              profilerData.timingHelper.beginComputePass(encoder, passDesc, origBeginComputePass) :
              origBeginComputePass(passDesc);
            
            // Store querySet from pass onto encoder for later retrieval
            if (pass.__querySet) {
              encoder.__querySet = pass.__querySet;
            }
            
            pass.__dispatches = [];
            pass.__boundPipeline = null;
            pass.__boundBindGroups = {};
            pass.__cpuStart = performance.now();

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
              const pipeline = this.__boundPipeline?.__capture;
              
              if (!pipeline) {
                console.warn('[WebSight] Dispatch without pipeline!');
                origDispatch(x, y, z);
                return;
              }

              const kernelId = generateKernelId(
                pipeline.shader, 
                pipeline.workgroupSize, 
                pipeline.label
              );

              // Ensure kernel exists
              if (!profilerData.kernels[kernelId]) {
                profilerData.kernels[kernelId] = {
                  id: kernelId,
                  label: pipeline.label,
                  workgroupSize: pipeline.workgroupSize,
                  shader: pipeline.shader,
                  stats: { count: 0, totalTime: 0, avgTime: 0, minTime: Infinity, maxTime: 0 }
                };
              }

              // Execute dispatch with CPU timing
              const cpuStart = performance.now();
              origDispatch(x, y, z);
              const cpuEnd = performance.now();
              const cpuTimeMs = cpuEnd - cpuStart;

              const dispatchRecord = {
                index: profilerData.dispatches.length,
                kernelId: kernelId,
                pipelineLabel: pipeline.label,
                x, y, z,
                cpuStart,
                cpuEnd,
                cpuTimeMs: cpuTimeMs,
                gpuTime: cpuTimeMs * 1000, // Convert to microseconds as default
                timingSource: 'cpu_timing',
                timestampStart: -1,
                timestampEnd: -1,
                bufferAccesses: Object.values(this.__boundBindGroups).flatMap(bg => 
                  bg.__capture?.entries.filter(e => e.resource?.id).map(e => ({
                    ...profilerData.buffers[e.resource.id],
                    binding: e.binding
                  })) || []
                )
              };

              // Update kernel stats immediately
              const kernel = profilerData.kernels[kernelId];
              if (kernel) {
                kernel.stats.count++;
                kernel.stats.totalTime += dispatchRecord.gpuTime;
                kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                kernel.stats.minTime = Math.min(kernel.stats.minTime, dispatchRecord.gpuTime);
                kernel.stats.maxTime = Math.max(kernel.stats.maxTime, dispatchRecord.gpuTime);
              }

              profilerData.dispatches.push(dispatchRecord);
              pass.__dispatches.push(dispatchRecord);
              encoder.__allDispatches.push(dispatchRecord);

              // Track buffer heat map
              dispatchRecord.bufferAccesses.forEach(b => {
                if (b?.id) {
                  profilerData.bufferHeatMap[b.id] = (profilerData.bufferHeatMap[b.id] || 0) + 1;
                }
              });

              console.log(`[WebSight] Dispatch #${dispatchRecord.index} (${pipeline.label}):`, {
                workgroups: `[${x}, ${y}, ${z}]`,
                cpuTime: `${cpuTimeMs.toFixed(3)} ms`,
                gpuTime: dispatchRecord.gpuTime ? 
                  `${(dispatchRecord.gpuTime / 1000).toFixed(3)} ms` : 
                  'pending',
                timingSource: dispatchRecord.timingSource,
                kernelId: kernelId
              });

              broadcastData();
            };

            return pass;
          };
          return encoder;
        };

        // Hook queue.submit to get GPU timing
        const origSubmit = device.queue.submit.bind(device.queue);
        let isInternalSubmit = false; // Flag to prevent recursion
        
        device.queue.submit = function(cmds) {
          // Get dispatches and querySet from the submitted command buffer(s)
          let dispatchesInSubmit = [];
          let querySet = null;
          for (const cmd of cmds) {
            if (cmd.__dispatches) {
              dispatchesInSubmit.push(...cmd.__dispatches);
            }
            if (cmd.__querySet) {
              querySet = cmd.__querySet;
            }
          }
          
          const result = origSubmit(cmds);
          
          // If we have TimingHelper and this has dispatches/querySet (not an internal submit), get GPU time
          if (profilerData.timingHelper && dispatchesInSubmit.length > 0 && querySet) {
            // Schedule the async timing resolution but DON'T await it here
            // This allows the synchronous submit loop to continue
            device.queue.onSubmittedWorkDone().then(async () => {
              try {
                isInternalSubmit = true; // Mark next submit as internal
                const timingResult = await profilerData.timingHelper.getResult(querySet);
                isInternalSubmit = false; // Reset flag
                
                const { timestampStart, timestampEnd, gpuTime: gpuTimeNs } = timingResult;
                
                // Validate timestamps - if both are 0, timing failed
                if (timestampStart === 0 && timestampEnd === 0) {
                  console.warn('[WebSight] Invalid timestamps (0, 0) - QuerySet may have been corrupted or not written');
                  // Keep CPU timing as fallback
                  return;
                }
                
                const gpuTimeUs = gpuTimeNs / 1000; // Convert to microseconds
                
                console.log(`[WebSight] GPU timing: ${(gpuTimeUs / 1000).toFixed(3)} ms for ${dispatchesInSubmit.length} dispatch(es)`);
                
                // WITHOUT in-pass timestamps, we can only measure the ENTIRE pass time.
                // Distribute time proportionally based on workgroup counts (x * y * z)
                const totalWorkgroups = dispatchesInSubmit.reduce((sum, d) => 
                  sum + (d.x * (d.y || 1) * (d.z || 1)), 0);
                
                dispatchesInSubmit.forEach((d, index) => {
                  const workgroups = d.x * (d.y || 1) * (d.z || 1);
                  const proportion = totalWorkgroups > 0 ? workgroups / totalWorkgroups : 1 / dispatchesInSubmit.length;
                  
                  // Distribute GPU time proportionally
                  d.gpuTime = gpuTimeUs * proportion;
                  d.timingSource = 'gpu_timestamp';
                  
                  // Estimate timestamps proportionally (not accurate, but needed for validation)
                  // First dispatch starts at timestampStart, last ends at timestampEnd
                  const cumulativeWorkgroups = dispatchesInSubmit.slice(0, index).reduce(
                    (sum, dispatch) => sum + (dispatch.x * (dispatch.y || 1) * (dispatch.z || 1)), 0
                  );
                  const cumulativeProportion = totalWorkgroups > 0 ? cumulativeWorkgroups / totalWorkgroups : index / dispatchesInSubmit.length;
                  
                  d.timestampStart = timestampStart + Math.floor(gpuTimeNs * cumulativeProportion);
                  d.timestampEnd = d.timestampStart + Math.floor(gpuTimeNs * proportion);
                  
                  // Update kernel stats with proportional GPU time
                  const kernel = profilerData.kernels[d.kernelId];
                  if (kernel) {
                    kernel.stats.totalTime = (kernel.stats.totalTime - d.cpuTimeMs * 1000) + d.gpuTime;
                    kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                    kernel.stats.minTime = Math.min(kernel.stats.minTime, d.gpuTime);
                    kernel.stats.maxTime = Math.max(kernel.stats.maxTime, d.gpuTime);
                  }
                });
                
                broadcastData();
              } catch (e) {
                isInternalSubmit = false; // Reset flag on error
                console.error('[WebSight] GPU timing failed:', e);
              }
            });
          }
          
          return result;
        };

        addLog('WebGPU hooks installed successfully');
        return device;
        
        } catch (e) {
          console.error('[WebSight] Device creation failed:', e);
          addLog(`Device creation failed: ${e.message}`, 'error');
          throw e;
        }
      };
      return adapter;
    };
  }

  // Public API
  if (typeof window !== 'undefined') {
    window.WebSight = {
      getData: () => profilerData,
      
      clear: () => { 
        profilerData.dispatches = []; 
        profilerData.logs = [];
        profilerData.timestampCount = 0;
        profilerData.kernels = {};
        addLog('Profiler data cleared');
      },
      
      start: hookWebGPU,
      
      getStats: () => {
        const dispatches = profilerData.dispatches || [];
        const validGpuTimes = dispatches
          .filter(d => d.gpuTime != null && d.gpuTime > 0 && 
                      d.timingSource !== 'pending_gpu_timestamp')
          .map(d => d.gpuTime);
        
        const gpuTimedDispatches = dispatches.filter(d => 
          d.timingSource === 'gpu_timestamp'
        ).length;
        
        return {
          totalDispatches: dispatches.length,
          dispatchesWithTiming: validGpuTimes.length,
          gpuTimedDispatches: gpuTimedDispatches,
          cpuFallbackDispatches: profilerData.cpuFallbackCount,
          avgGpuTime: validGpuTimes.length > 0 
            ? validGpuTimes.reduce((a, b) => a + b, 0) / validGpuTimes.length
            : 0,
          totalGpuTime: validGpuTimes.length > 0
            ? validGpuTimes.reduce((a, b) => a + b, 0)
            : 0,
          minGpuTime: validGpuTimes.length > 0
            ? Math.min(...validGpuTimes)
            : 0,
          maxGpuTime: validGpuTimes.length > 0
            ? Math.max(...validGpuTimes)
            : 0
        };
      },
      
      getSessionInfo: () => {
        return {
          timingMode: profilerData.timingMode,
          hasTimestamps: profilerData.hasTimestamps,
          timestampCount: profilerData.timestampCount,
          maxTimestamps: 8192,
          memoryPeak: profilerData.memoryUsage.peak,
          kernelCount: Object.keys(profilerData.kernels).length,
          cpuFallbackCount: profilerData.cpuFallbackCount
        };
      },
      
      listKernels: () => {
        return Object.values(profilerData.kernels).map(k => ({
          id: k.id,
          label: k.label || 'Unnamed Kernel',
          workgroupSize: k.workgroupSize || { x: 0, y: 0, z: 0 },
          dispatchCount: k.stats?.count || 0,
          avgTime: k.stats?.avgTime || 0,
          totalTime: k.stats?.totalTime || 0
        }));
      },
      
      getKernelSummary: (kernelId) => {
        const kernel = profilerData.kernels[kernelId];
        if (!kernel) {
          return {
            label: 'Unknown',
            workgroupSize: { x: 0, y: 0, z: 0 },
            dispatchCount: 0,
            avgTime: 0,
            totalTime: 0,
            minTime: 0,
            maxTime: 0
          };
        }
        return {
          label: kernel.label || 'Unnamed Kernel',
          workgroupSize: kernel.workgroupSize || { x: 0, y: 0, z: 0 },
          dispatchCount: kernel.stats?.count || 0,
          avgTime: kernel.stats?.avgTime || 0,
          totalTime: kernel.stats?.totalTime || 0,
          minTime: kernel.stats?.minTime === Infinity ? 0 : kernel.stats?.minTime || 0,
          maxTime: kernel.stats?.maxTime || 0
        };
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
      
      // Configuration API
      configure: (options) => {
        if (options.maxQuerySets !== undefined) {
          const oldSize = profilerData.timingHelper?.maxPoolSize || 128;
          if (profilerData.timingHelper) {
            profilerData.timingHelper.maxPoolSize = options.maxQuerySets;
            console.log(`[WebSight] QuerySet pool size changed: ${oldSize} → ${options.maxQuerySets}`);
            addLog(`QuerySet pool size: ${options.maxQuerySets}`);
          } else {
            console.warn('[WebSight] Cannot configure: TimingHelper not initialized yet');
          }
        }
        return {
          maxQuerySets: profilerData.timingHelper?.maxPoolSize || 128,
          currentPoolSize: profilerData.timingHelper?.querySetPool.length || 0,
          activeQuerySets: profilerData.timingHelper?.activeQuerySets.size || 0
        };
      },
      
      // Get current pool status
      getPoolStatus: () => {
        if (!profilerData.timingHelper) {
          return { available: false, reason: 'TimingHelper not initialized' };
        }
        return {
          available: true,
          maxSize: profilerData.timingHelper.maxPoolSize,
          poolSize: profilerData.timingHelper.querySetPool.length,
          activeCount: profilerData.timingHelper.activeQuerySets.size,
          freeCount: profilerData.timingHelper.querySetPool.length - profilerData.timingHelper.activeQuerySets.size,
          exhaustedCount: profilerData.timingHelper.poolExhaustedCount || 0
        };
      }
    };
    
    // Auto-initialize
    hookWebGPU();
    addLog('WebSight profiler initialized');
  }
})();