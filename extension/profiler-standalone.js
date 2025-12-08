// WebGPU Compute Profiler - Standalone Version
// This can be injected into any page to profile WebGPU operations

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
    bufferHeatMap: {}
  };

  function addLog(message) {
    const timestamp = new Date().toLocaleTimeString();
    const logMessage = `[${timestamp}] ${message}`;
    profilerData.logs.push(logMessage);
    console.log(`[WebGPU Profiler] ${message}`);
    
    // Update UI if it exists
    if (window.updateWebSightUI) {
      window.updateWebSightUI();
    }
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
      memoryAccesses: 0,
      sharedMemorySize: 0
    };

    // Check for atomics
    if (source.match(/atomic(Add|Sub|Max|Min|And|Or|Xor|Exchange|CompareExchange)/)) {
      metrics.hasAtomics = true;
      warnings.push({
        severity: 'warning',
        type: 'ATOMIC_CONTENTION',
        message: 'Atomic operations detected - potential serialization bottleneck'
      });
    }

    // Check for branching
    const ifMatches = source.match(/\bif\s*\(/g);
    if (ifMatches && ifMatches.length > 0) {
      metrics.hasBranching = true;
      if (ifMatches.length > 5) {
        warnings.push({
          severity: 'warning',
          type: 'EXCESSIVE_BRANCHING',
          message: `${ifMatches.length} conditional branches - may cause warp divergence`
        });
      }
    }

    // Estimate registers
    const varMatches = source.match(/\b(let|var)\s+\w+/g);
    metrics.estimatedRegisters = varMatches ? varMatches.length : 0;
    if (metrics.estimatedRegisters > 40) {
      warnings.push({
        severity: 'critical',
        type: 'REGISTER_PRESSURE',
        message: `~${metrics.estimatedRegisters} variables - high register pressure`
      });
    }

    // Check shared memory
    const workgroupVars = source.match(/var<workgroup>\s*:\s*array<[^,>]+,\s*(\d+)/g);
    if (workgroupVars) {
      workgroupVars.forEach(match => {
        const size = parseInt(match.match(/,\s*(\d+)/)?.[1] || '0');
        metrics.sharedMemorySize += size * 4;
      });
      
      if (metrics.sharedMemorySize > 32768) {
        warnings.push({
          severity: 'critical',
          type: 'SHARED_MEMORY_LIMIT',
          message: `${(metrics.sharedMemorySize / 1024).toFixed(1)}KB shared memory - may exceed limits`
        });
      }
    }

    return { warnings, metrics };
  }

  function estimateBottleneck(dispatch, gpuTime, shaderAnalysis) {
    if (!gpuTime) return { type: 'UNKNOWN', confidence: 0, description: 'No timing data' };

    const totalThreads = dispatch.x * dispatch.y * dispatch.z * 
                        dispatch.workgroupSize.x * dispatch.workgroupSize.y * dispatch.workgroupSize.z;
    const wgSize = dispatch.workgroupSize.x * dispatch.workgroupSize.y * dispatch.workgroupSize.z;

    // Low occupancy check
    if (totalThreads < 256) {
      return {
        type: 'OCCUPANCY',
        confidence: 90,
        description: `Only ${totalThreads} threads - GPU severely underutilized`,
        suggestion: 'Increase workgroup count or size'
      };
    }

    // Small workgroup check
    if (wgSize < 32) {
      return {
        type: 'OCCUPANCY',
        confidence: 80,
        description: `Workgroup size ${wgSize} < 32 - poor warp/wave utilization`,
        suggestion: 'Increase workgroup size to 64-256'
      };
    }

    // Atomic check
    if (shaderAnalysis.metrics.hasAtomics) {
      return {
        type: 'ATOMIC_SERIALIZATION',
        confidence: 70,
        description: 'Atomics detected - likely serialization',
        suggestion: 'Use local accumulation then atomic once per workgroup'
      };
    }

    // Divergence check
    if (shaderAnalysis.metrics.hasBranching && wgSize >= 32) {
      return {
        type: 'DIVERGENCE',
        confidence: 65,
        description: 'Branching with wide workgroups - warp divergence likely',
        suggestion: 'Reorganize to reduce divergence'
      };
    }

    // Register pressure check
    if (shaderAnalysis.metrics.estimatedRegisters > 40) {
      return {
        type: 'REGISTER_PRESSURE',
        confidence: 60,
        description: 'High register usage may limit active warps',
        suggestion: 'Reduce local variables'
      };
    }

    return {
      type: 'COMPUTE_BOUND',
      confidence: 50,
      description: 'Likely ALU limited - no obvious bottlenecks',
      suggestion: 'Use native GPU profiling tools'
    };
  }

  async function benchmarkGPU(device) {
    addLog('🔬 Running GPU microbenchmarks...');
    
    const testSize = 64 * 1024 * 1024;
    const inputBuffer = device.createBuffer({
      size: testSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const outputBuffer = device.createBuffer({
      size: testSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const copyShader = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          if (i < arrayLength(&input)) {
            output[i] = input[i];
          }
        }
      `
    });

    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: copyShader, entryPoint: 'main' }
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } }
      ]
    });

    const iterations = 10;
    const times = [];

    for (let i = 0; i < iterations; i++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(testSize / (256 * 16)));
      pass.end();
      
      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - start);
    }

    const avgTime = times.slice(2).reduce((a, b) => a + b, 0) / (times.length - 2);
    const bandwidth = (testSize / 1e9) / (avgTime / 1000);

    profilerData.gpuCharacteristics = {
      estimatedBandwidth: bandwidth,
      estimatedL2Size: 4 * 1024 * 1024,
      estimatedComputeUnits: 32,
      timestamp: Date.now()
    };

    addLog(` Estimated bandwidth: ${bandwidth.toFixed(1)} GB/s`);
    
    inputBuffer.destroy();
    outputBuffer.destroy();
    return profilerData.gpuCharacteristics;
  }

  async function hookWebGPU() {
    if (!navigator.gpu) {
      addLog(' WebGPU not supported');
      return null;
    }

    // Intercept requestAdapter
    const originalRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = async function(options) {
      const adapter = await originalRequestAdapter(options);
      if (!adapter) return adapter;

      // Intercept requestDevice
      const originalRequestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async function(descriptor) {
        const features = ['timestamp-query', 'chromium-experimental-timestamp-query-inside-passes'];
        const availableFeatures = Array.from(adapter.features);
        const requestFeatures = features.filter(f => availableFeatures.includes(f));
        
        profilerData.hasTimestamps = requestFeatures.includes('timestamp-query') || 
                                     requestFeatures.includes('chromium-experimental-timestamp-query-inside-passes');
        
        const device = await originalRequestDevice({
          ...descriptor,
          requiredFeatures: [...(descriptor?.requiredFeatures || []), ...requestFeatures]
        });

        addLog(` Device created (timestamps: ${profilerData.hasTimestamps})`);

        // Run benchmark
        await benchmarkGPU(device);

        // Hook createComputePipeline
        const origCreateComputePipeline = device.createComputePipeline.bind(device);
        device.createComputePipeline = function(desc) {
          const pipeline = origCreateComputePipeline(desc);
          const id = crypto.randomUUID();
          
          const source = desc.compute.module.__source || desc.compute.entryPoint || '';
          const workgroupSize = extractWorkgroupSize(source);
          const analysis = analyzeWGSL(source);
          
          pipeline.__capture = {
            id,
            label: desc.label || `compute_${id.slice(0, 8)}`,
            workgroupSize,
            shader: source,
            entryPoint: desc.compute.entryPoint || 'main',
            analysis
          };

          profilerData.pipelines[id] = pipeline.__capture;
          addLog(` Pipeline: ${pipeline.__capture.label} (${analysis.warnings.length} warnings)`);
          
          return pipeline;
        };

        // Hook createShaderModule to capture source
        const origCreateShaderModule = device.createShaderModule.bind(device);
        device.createShaderModule = function(desc) {
          const module = origCreateShaderModule(desc);
          module.__source = desc.code;
          return module;
        };

        // Hook createBuffer
        const origCreateBuffer = device.createBuffer.bind(device);
        device.createBuffer = function(desc) {
          const buffer = origCreateBuffer(desc);
          const id = crypto.randomUUID();
          
          buffer.__capture = {
            id,
            label: desc.label || `buffer_${id.slice(0, 8)}`,
            size: desc.size,
            usage: desc.usage
          };
          
          profilerData.buffers[id] = buffer.__capture;
          return buffer;
        };

        // Hook createBindGroup
        const origCreateBindGroup = device.createBindGroup.bind(device);
        device.createBindGroup = function(desc) {
          const bindGroup = origCreateBindGroup(desc);
          const id = crypto.randomUUID();
          
          bindGroup.__capture = {
            id,
            label: desc.label || `bindgroup_${id.slice(0, 8)}`,
            entries: desc.entries.map(e => ({
              binding: e.binding,
              resource: e.resource.buffer?.__capture || e.resource
            }))
          };
          
          profilerData.bindGroups[id] = bindGroup.__capture;
          return bindGroup;
        };

        // Create timestamp query resources
        if (profilerData.hasTimestamps) {
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
        }

        // Hook setPipeline
        const origSetPipeline = GPUComputePassEncoder.prototype.setPipeline;
        GPUComputePassEncoder.prototype.setPipeline = function(pipeline) {
          this.__pipeline = pipeline;
          return origSetPipeline.call(this, pipeline);
        };

        // Hook setBindGroup
        const origSetBindGroup = GPUComputePassEncoder.prototype.setBindGroup;
        GPUComputePassEncoder.prototype.setBindGroup = function(index, bg) {
          this.__boundBindGroups = this.__boundBindGroups || {};
          this.__boundBindGroups[index] = bg;
          return origSetBindGroup.call(this, index, bg);
        };

        // Hook dispatchWorkgroups
        const origDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
        GPUComputePassEncoder.prototype.dispatchWorkgroups = function(x = 1, y = 1, z = 1) {
          const pipeline = this.__pipeline?.__capture;
          const bindGroups = this.__boundBindGroups || {};
          
          const dispatchRecord = {
            index: profilerData.dispatches.length,
            x, y, z,
            pipeline: pipeline?.id,
            pipelineLabel: pipeline?.label || 'unknown',
            workgroupSize: pipeline?.workgroupSize || { x: 1, y: 1, z: 1 },
            bindGroups: Object.entries(bindGroups).map(([idx, bg]) => ({
              index: parseInt(idx),
              ...bg.__capture
            })),
            shaderAnalysis: pipeline?.analysis || { warnings: [], metrics: {} },
            timestampStart: null,
            timestampEnd: null,
            gpuTime: null,
            bottleneck: null
          };

          // Update buffer heat map
          if (dispatchRecord.bindGroups && dispatchRecord.bindGroups.length > 0) {
            dispatchRecord.bindGroups.forEach(bg => {
              if (bg.entries) {
                bg.entries.forEach(e => {
                  if (e.resource?.id) {
                    profilerData.bufferHeatMap[e.resource.id] = 
                      (profilerData.bufferHeatMap[e.resource.id] || 0) + 1;
                  }
                });
              }
            });
          }

          if (profilerData.hasTimestamps && profilerData.querySet) {
            const tsIndex = profilerData.timestampCount;
            dispatchRecord.timestampStart = tsIndex;
            dispatchRecord.timestampEnd = tsIndex + 1;
            
            try {
              this.writeTimestamp(profilerData.querySet, tsIndex);
              const result = origDispatch.call(this, x, y, z);
              this.writeTimestamp(profilerData.querySet, tsIndex + 1);
              
              profilerData.timestampCount += 2;
              profilerData.dispatches.push(dispatchRecord);
              
              addLog(`🔄 Dispatch #${dispatchRecord.index}: ${dispatchRecord.pipelineLabel} (${x}×${y}×${z})`);
              
              return result;
            } catch (e) {
              // Fallback if timestamp writes fail
              addLog(` Timestamp query failed: ${e.message}`);
              profilerData.hasTimestamps = false;
              profilerData.dispatches.push(dispatchRecord);
              addLog(`🔄 Dispatch #${dispatchRecord.index}: ${dispatchRecord.pipelineLabel} (${x}×${y}×${z}) [no timing]`);
              return origDispatch.call(this, x, y, z);
            }
          } else {
            profilerData.dispatches.push(dispatchRecord);
            addLog(`🔄 Dispatch #${dispatchRecord.index}: ${dispatchRecord.pipelineLabel} (${x}×${y}×${z}) [no timing]`);
            return origDispatch.call(this, x, y, z);
          }
        };

        // Hook queue.submit
        const origSubmit = device.queue.submit.bind(device.queue);
        device.queue.submit = async function(commandBuffers) {
          const submitStartTime = performance.now();
          const result = origSubmit(commandBuffers);
          
          // Use CPU timing as fallback if timestamps not available
          if (!profilerData.hasTimestamps && profilerData.dispatches.length > 0) {
            // Wait for GPU completion to get approximate timing
            device.queue.onSubmittedWorkDone().then(() => {
              const submitEndTime = performance.now();
              const totalTime = (submitEndTime - submitStartTime) * 1000; // Convert to microseconds
              
              // Distribute time evenly across dispatches in this submit
              // (This is approximate - real timing would use GPU timestamps)
              const recentDispatches = profilerData.dispatches.filter(d => d.gpuTime === null);
              if (recentDispatches.length > 0) {
                const timePerDispatch = totalTime / recentDispatches.length;
                recentDispatches.forEach(d => {
                  d.gpuTime = timePerDispatch;
                  d.bottleneck = estimateBottleneck(d, timePerDispatch, d.shaderAnalysis);
                });
                
                // Update UI
                if (window.updateWebSightUI) {
                  window.updateWebSightUI();
                }
              }
            });
          }
          
          if (profilerData.hasTimestamps && profilerData.timestampCount > 0) {
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
            device.queue.submit([encoder.finish()]);
            
            await profilerData.readBuffer.mapAsync(GPUMapMode.READ);
            const timestamps = new BigInt64Array(
              profilerData.readBuffer.getMappedRange(0, profilerData.timestampCount * 8)
            );
            
            profilerData.dispatches.forEach(dispatch => {
              if (dispatch.timestampStart !== null) {
                const start = Number(timestamps[dispatch.timestampStart]);
                const end = Number(timestamps[dispatch.timestampEnd]);
                dispatch.gpuTime = (end - start) / 1000;
                
                dispatch.bottleneck = estimateBottleneck(dispatch, dispatch.gpuTime, dispatch.shaderAnalysis);
              }
            });
            
            profilerData.readBuffer.unmap();
            
            // Update UI
            if (window.updateWebSightUI) {
              window.updateWebSightUI();
            }
          }
          
          return result;
        };

        addLog(' WebGPU hooks installed');
        return device;
      };

      return adapter;
    };

    addLog('🚀 Profiler hooks initialized');
    return true;
  }

  // Public API
  window.WebSight = {
    start: async function() {
      addLog(' Starting WebGPU Profiler...');
      return await hookWebGPU();
    },
    
    getData: function() {
      return profilerData;
    },
    
    clear: function() {
      profilerData.dispatches = [];
      profilerData.timestampCount = 0;
      profilerData.logs = [];
      profilerData.bufferHeatMap = {};
      addLog('🗑 Profiler data cleared');
    },
    
    export: function() {
      const data = {
        dispatches: profilerData.dispatches,
        pipelines: profilerData.pipelines,
        gpuCharacteristics: profilerData.gpuCharacteristics,
        bufferHeatMap: profilerData.bufferHeatMap,
        timestamp: new Date().toISOString()
      };
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `webgpu-profile-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      
      addLog(' Profile exported');
      return data;
    },
    
    getStats: function() {
      const dispatches = profilerData.dispatches;
      const totalGpuTime = dispatches.reduce((sum, d) => sum + (d.gpuTime || 0), 0);
      const avgGpuTime = dispatches.length > 0 ? totalGpuTime / dispatches.length : 0;
      
      const issues = dispatches.reduce((acc, d) => {
        d.shaderAnalysis.warnings.forEach(w => {
          if (w.severity === 'critical') acc.critical++;
          else if (w.severity === 'warning') acc.warnings++;
          else acc.info++;
        });
        return acc;
      }, { critical: 0, warnings: 0, info: 0 });
      
      return {
        totalDispatches: dispatches.length,
        activePipelines: Object.keys(profilerData.pipelines).length,
        totalBuffers: Object.keys(profilerData.buffers).length,
        totalGpuTime,
        avgGpuTime,
        issues,
        gpuCharacteristics: profilerData.gpuCharacteristics
      };
    }
  };

  // Auto-start
  console.log('🚀 WebGPU Compute Profiler loaded');
  console.log('Use WebSight.start() to begin profiling');
  console.log('Use WebSight.getData() to get captured data');
  console.log('Use WebSight.export() to download report');
})();
