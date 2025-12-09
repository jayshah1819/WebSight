// WebGPU Compute Profiler - Standalone Version

(function() {
  'use strict';

  const profilerData = {
    dispatches: [],
    pipelines: {},
    bindGroups: {},
    buffers: {},
    querySet: null,
    resolveBuffer: null,
    readBuffer: null,
    timestampCount: 0,
    hasTimestamps: false,
    logs: [],
    gpuCharacteristics: null,
    bufferHeatMap: {},
    runId: null,
    kernels: {},
    runs: {},
    timingMode: 'unknown',
    sessionStart: Date.now(),
    totalKernelTime: 0,
    cpuFallbackCount: 0,
    memoryUsage: {
      peak: 0,
      current: 0,
      allocations: []
    }
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

  function generateKernelId(shaderSource, workgroupSize, variant = '') {
    const config = `${workgroupSize.x}x${workgroupSize.y}x${workgroupSize.z}`;
    const sourceHash = hashString(shaderSource);
    const configHash = hashString(config);
    const variantHash = variant ? `_${hashString(variant)}` : '';
    return `kernel_${sourceHash}_${configHash}${variantHash}`;
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
    const metrics = { hasAtomics: false, hasBranching: false, estimatedRegisters: 0, sharedMemorySize: 0 };

    if (source.match(/atomic(Add|Sub|Max|Min|And|Or|Xor|Exchange|CompareExchange)/)) {
      metrics.hasAtomics = true;
      warnings.push({ severity: 'warning', type: 'ATOMIC_CONTENTION', message: 'Atomic operations detected' });
    }
    if (source.match(/\bif\s*\(/g)?.length > 5) {
      metrics.hasBranching = true;
      warnings.push({ severity: 'warning', type: 'EXCESSIVE_BRANCHING', message: 'Excessive branching detected' });
    }
    return { warnings, metrics };
  }

  function addLog(message, level = 'info') {
    profilerData.logs.push({ timestamp: new Date().toLocaleTimeString(), level, message, time: Date.now() });
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
      console.log('[WebSight] Adapter features:', Array.from(adapter.features).join(', '));
      console.log('[WebSight] Timestamp query supported:', hasTimestampFeature);
      
      addLog(`Timestamp query support: ${hasTimestampFeature}`);

      const originalRequestDevice = adapter.requestDevice.bind(adapter);
      
      adapter.requestDevice = async function(descriptor) {
        console.log('[WebSight] Device requested with descriptor:', descriptor);
        console.log('[WebSight] Adapter supports timestamp-query:', hasTimestampFeature);
        
        const requiredFeatures = new Set(descriptor?.requiredFeatures || []);
        const originalFeatures = Array.from(requiredFeatures);
        
        if (hasTimestampFeature) {
          requiredFeatures.add('timestamp-query');
          console.log('[WebSight] FORCING timestamp-query feature');
          console.log('[WebSight] Features before:', originalFeatures);
          console.log('[WebSight] Features after:', Array.from(requiredFeatures));
        } else {
          console.warn('[WebSight] WARNING: Adapter does NOT support timestamp-query - GPU profiling unavailable');
        }
        
        const modifiedDescriptor = {
          ...descriptor,
          requiredFeatures: Array.from(requiredFeatures)
        };
        
        console.log('[WebSight] Calling requestDevice with:', modifiedDescriptor);
        const device = await originalRequestDevice(modifiedDescriptor);
        
        profilerData.hasTimestamps = device.features.has('timestamp-query');
        profilerData.timingMode = profilerData.hasTimestamps ? 'gpu' : 'cpu-fallback';
        
        console.log('[WebSight] Device created');
        console.log('[WebSight] Device has timestamp-query:', profilerData.hasTimestamps);
        console.log('[WebSight] All device features:', Array.from(device.features).join(', '));
        
        if (!profilerData.hasTimestamps && hasTimestampFeature) {
          console.error('[WebSight] CRITICAL ERROR: timestamp-query NOT enabled despite being supported!');
        }
        
        addLog(`Device created with timestamps: ${profilerData.hasTimestamps}`);
        addLog(`Device features: ${Array.from(device.features).join(', ')}`);

        if (profilerData.hasTimestamps) {
          try {
            profilerData.querySet = device.createQuerySet({ 
              type: 'timestamp', 
              count: 512 
            });
            profilerData.resolveBuffer = device.createBuffer({ 
              size: 512 * 8, 
              usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC 
            });
            profilerData.readBuffer = device.createBuffer({ 
              size: 512 * 8, 
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ 
            });
            addLog('Timestamp query buffers created successfully');
          } catch (e) {
            addLog(`Failed to create timestamp buffers: ${e.message}`, 'error');
            profilerData.hasTimestamps = false;
            profilerData.timingMode = 'cpu-fallback';
          }
        }

        const origCreateComputePipeline = device.createComputePipeline.bind(device);
        device.createComputePipeline = function(desc) {
          const pipeline = origCreateComputePipeline(desc);
          const source = desc.compute.module.__source || '';
          pipeline.__capture = {
            id: crypto.randomUUID(),
            label: desc.label || 'compute_pipeline',
            workgroupSize: extractWorkgroupSize(source),
            shader: source,
            analysis: analyzeWGSL(source)
          };
          profilerData.pipelines[pipeline.__capture.id] = pipeline.__capture;
          addLog(`Pipeline created: ${pipeline.__capture.label}`);
          return pipeline;
        };

        const origCreateShaderModule = device.createShaderModule.bind(device);
        device.createShaderModule = function(desc) {
          const module = origCreateShaderModule(desc);
          module.__source = desc.code;
          return module;
        };

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

        const origCreateCommandEncoder = device.createCommandEncoder.bind(device);
        device.createCommandEncoder = function(desc) {
          const encoder = origCreateCommandEncoder(desc);
          const origBeginComputePass = encoder.beginComputePass.bind(encoder);

          encoder.beginComputePass = function(passDesc) {
            const pass = origBeginComputePass(passDesc);
            pass.__dispatches = [];

            const origDispatch = pass.dispatchWorkgroups.bind(pass);

            pass.dispatchWorkgroups = function(x, y, z) {
              const pipeline = this.__pipeline?.__capture;
              const kid = pipeline ? generateKernelId(pipeline.shader, pipeline.workgroupSize) : 'unknown';
              
              if (kid !== 'unknown' && !profilerData.kernels[kid]) {
                profilerData.kernels[kid] = {
                  id: kid,
                  label: pipeline.label,
                  workgroupSize: pipeline.workgroupSize,
                  stats: { count: 0, totalTime: 0, avgTime: 0 }
                };
                addLog(`Kernel registered: ${pipeline.label} (${kid})`);
              }

              let tsStart = -1;
              let tsEnd = -1;

              const canTimestamp =
                profilerData.hasTimestamps &&
                profilerData.querySet &&
                profilerData.timestampCount + 2 <= 512 &&
                typeof pass.writeTimestamp === 'function';

              if (canTimestamp) {
                tsStart = profilerData.timestampCount++;
                tsEnd = profilerData.timestampCount++;
                pass.writeTimestamp(profilerData.querySet, tsStart);
              }

              const cpuStart = performance.now();
              origDispatch(x, y, z);
              const cpuEnd = performance.now();

              if (canTimestamp) {
                pass.writeTimestamp(profilerData.querySet, tsEnd);
              }

              const dispatchRecord = {
                index: profilerData.dispatches.length,
                kernelId: kid,
                pipelineLabel: pipeline?.label || 'unknown',
                x, y, z,
                cpuStart,
                cpuEnd,
                gpuTime: null,
                timingSource: canTimestamp ? 'gpu_timestamp' : 'cpu-fallback',
                timestampStart: tsStart,
                timestampEnd: tsEnd,
                bufferAccesses: this.__boundBindGroups ? 
                  Object.values(this.__boundBindGroups).flatMap(bg => 
                    bg.__capture?.entries.filter(e => e.resource?.id).map(e => ({
                      ...profilerData.buffers[e.resource.id],
                      binding: e.binding
                    }))
                  ) : []
              };

              profilerData.dispatches.push(dispatchRecord);
              pass.__dispatches.push(dispatchRecord);

              dispatchRecord.bufferAccesses.forEach(b => {
                if (b?.id) {
                  profilerData.bufferHeatMap[b.id] = (profilerData.bufferHeatMap[b.id] || 0) + 1;
                }
              });

              if (!canTimestamp) {
                dispatchRecord.gpuTime = (cpuEnd - cpuStart) * 1000;
              }
              
              const cpuTimeMs = dispatchRecord.cpuEnd - dispatchRecord.cpuStart;
              console.log(`[WebSight] Dispatch #${dispatchRecord.index} (${pipeline?.label || 'unknown'}):`, {
                cpuTime: `${cpuTimeMs.toFixed(3)} ms`,
                gpuTime: dispatchRecord.gpuTime ? `${(dispatchRecord.gpuTime / 1000).toFixed(3)} ms (${dispatchRecord.gpuTime.toFixed(2)} μs)` : 'pending',
                timingSource: dispatchRecord.timingSource
              });
              
              addLog(`Dispatch #${dispatchRecord.index}: ${pipeline?.label || 'unknown'} [${x}, ${y}, ${z}]`);
              broadcastData();
            };

            const origSetPipeline = pass.setPipeline.bind(pass);
            pass.setPipeline = function(p) {
              this.__pipeline = p;
              origSetPipeline(p);
            };

            const origSetBindGroup = pass.setBindGroup.bind(pass);
            pass.setBindGroup = function(i, bg) {
              this.__boundBindGroups = this.__boundBindGroups || {};
              this.__boundBindGroups[i] = bg;
              origSetBindGroup(i, bg);
            };

            return pass;
          };
          return encoder;
        };

        const origSubmit = device.queue.submit.bind(device.queue);
        device.queue.submit = function(cmds) {
          const result = origSubmit(cmds);

          if (profilerData.hasTimestamps && profilerData.timestampCount > 0 && !profilerData.readBuffer.__mapped) {
            profilerData.readBuffer.__mapped = true;
            
            addLog(`Resolving ${profilerData.timestampCount} timestamps...`);
            
            const encoder = device.createCommandEncoder();
            encoder.resolveQuerySet(
              profilerData.querySet, 
              0, 
              profilerData.timestampCount, 
              profilerData.resolveBuffer, 
              0
            );
            encoder.copyBufferToBuffer(
              profilerData.resolveBuffer, 
              0, 
              profilerData.readBuffer, 
              0, 
              profilerData.timestampCount * 8
            );
            
            origSubmit([encoder.finish()]);

            device.queue.onSubmittedWorkDone().then(() => {
              return profilerData.readBuffer.mapAsync(GPUMapMode.READ);
            }).then(() => {
              const times = new BigUint64Array(
                profilerData.readBuffer.getMappedRange(0, profilerData.timestampCount * 8)
              );
              
              addLog(`Successfully read ${times.length} timestamps`);
              
              profilerData.dispatches.forEach(d => {
                if (
                  d.timestampStart >= 0 &&
                  d.timestampEnd >= 0 &&
                  d.timestampEnd < times.length
                ) {
                  const start = Number(times[d.timestampStart]);
                  const end = Number(times[d.timestampEnd]);

                  if (end > start) {
                    d.gpuTime = (end - start) / 1000;
                    d.timingSource = 'gpu_timestamp';

                    if (profilerData.kernels[d.kernelId]) {
                      const k = profilerData.kernels[d.kernelId];
                      k.stats.count++;
                      k.stats.totalTime += d.gpuTime;
                      k.stats.avgTime = k.stats.totalTime / k.stats.count;
                    }

                    const cpuTimeMs = d.cpuEnd - d.cpuStart;
                    console.log(`[WebSight] GPU Timestamp Resolved - Dispatch #${d.index} (${d.pipelineLabel}):`, {
                      cpuTime: `${cpuTimeMs.toFixed(3)} ms`,
                      gpuTime: `${(d.gpuTime / 1000).toFixed(3)} ms (${d.gpuTime.toFixed(2)} μs)`,
                      timingSource: 'gpu_timestamp',
                      speedup: `${(cpuTimeMs / (d.gpuTime / 1000)).toFixed(2)}x`
                    });

                    addLog(`Dispatch #${d.index} GPU time: ${d.gpuTime.toFixed(2)} μs`);
                  }
                }
              });

              addLog(`Resolved ${profilerData.dispatches.length} dispatch timestamps`);

              profilerData.readBuffer.unmap();
              profilerData.readBuffer.__mapped = false;
              profilerData.timestampCount = 0;
              broadcastData();
            }).catch(e => {
              addLog(`Timestamp readback failed: ${e.message}`, 'error');
              profilerData.readBuffer.__mapped = false;
            });
          }
          return result;
        };

        addLog('WebGPU hooks installed successfully');
        return device;
      };
      return adapter;
    };
  }

  if (typeof window !== 'undefined') {
    window.WebSight = {
      getData: () => profilerData,
      clear: () => { 
        profilerData.dispatches = []; 
        profilerData.logs = [];
        profilerData.timestampCount = 0;
        addLog('Profiler data cleared');
      },
      start: hookWebGPU,
      getStats: () => {
        const dispatches = profilerData.dispatches || [];
        const validGpuTimes = dispatches
          .filter(d => d.gpuTime != null && d.gpuTime > 0)
          .map(d => d.gpuTime);
        
        return {
          totalDispatches: dispatches.length,
          dispatchesWithTiming: validGpuTimes.length,
          avgGpuTime: validGpuTimes.length > 0 
            ? validGpuTimes.reduce((a, b) => a + b, 0) / validGpuTimes.length
            : 0,
          totalGpuTime: validGpuTimes.length > 0
            ? validGpuTimes.reduce((a, b) => a + b, 0) / 1000
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
          maxTimestamps: 512,
          memoryPeak: profilerData.memoryUsage.peak
        };
      },
      listKernels: () => {
        return Object.values(profilerData.kernels).map(k => ({
          id: k.id,
          label: k.label || 'Unnamed Kernel',
          workgroupSize: k.workgroupSize || { x: 0, y: 0, z: 0 },
          dispatchCount: k.stats?.count || 0
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
            totalTime: 0
          };
        }
        return {
          label: kernel.label || 'Unnamed Kernel',
          workgroupSize: kernel.workgroupSize || { x: 0, y: 0, z: 0 },
          dispatchCount: kernel.stats?.count || 0,
          avgTime: kernel.stats?.avgTime || 0,
          totalTime: kernel.stats?.totalTime || 0
        };
      },
      export: () => {
        const dataStr = JSON.stringify(profilerData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'websight-profile.json';
        a.click();
        URL.revokeObjectURL(url);
      },
      benchmarkScalability: async (shaderTemplate, options = {}) => {
        throw new Error('benchmarkScalability is not implemented in profiler-standalone.js. This function should be called from your application code, not from WebSight profiler. Please run your own benchmarks and use console.log({ gpuTotalTimeNSArray: [...] }) to report timing to WebSight.');
      }
    };
    hookWebGPU();
  }
})();