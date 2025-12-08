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
    bufferHeatMap: {},
    // New: Stable identifiers and structured data
    runId: null,
    kernels: {},  // kernelId -> { id, label, shaderHash, workgroupSize, dispatches: [], stats: {} }
    runs: {},     // runId -> { timestamp, kernels: [], metadata: {} }
    phaseShifts: [] // Detected performance regime changes
  };

  // Helper: Generate deterministic hash
  function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }

  // Helper: Generate kernelId from shader source + config
  function generateKernelId(shaderSource, workgroupSize, variant = '') {
    const config = `${workgroupSize.x}x${workgroupSize.y}x${workgroupSize.z}`;
    const sourceHash = hashString(shaderSource);
    const configHash = hashString(config);
    const variantHash = variant ? `_${hashString(variant)}` : '';
    return `kernel_${sourceHash}_${configHash}${variantHash}`;
  }
  
  // Helper: Extract variant info from shader (constants, unroll factors, etc.)
  function extractVariantInfo(source) {
    const variants = {};
    
    // Extract workgroup size
    const wgMatch = source.match(/@workgroup_size\((\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?\)/);
    if (wgMatch) {
      variants.workgroupSize = `${wgMatch[1] || 1}x${wgMatch[2] || 1}x${wgMatch[3] || 1}`;
    }
    
    // Extract const declarations (optimization knobs)
    const constMatches = source.matchAll(/const\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([^;]+);/g);
    for (const match of constMatches) {
      const name = match[1];
      const value = match[2].trim();
      // Track optimization-relevant constants
      if (name.match(/TILE|BLOCK|UNROLL|BATCH/i)) {
        variants[name] = value;
      }
    }
    
    return variants;
  }

  // Helper: Compute normalized metrics
  function computeNormalizedMetrics(dispatch) {
    const totalThreads = dispatch.x * dispatch.y * dispatch.z * 
                        (dispatch.workgroupSize?.x || 1) * 
                        (dispatch.workgroupSize?.y || 1) * 
                        (dispatch.workgroupSize?.z || 1);
    
    const totalBytes = dispatch.bufferAccesses?.reduce((sum, buf) => sum + (buf.size || 0), 0) || 0;
    
    return {
      nsPerElement: dispatch.gpuTime ? (dispatch.gpuTime * 1000) / totalThreads : null,
      bytesPerThread: totalThreads > 0 ? totalBytes / totalThreads : 0,
      opsPerElement: 1, // Can be enhanced with WGSL analysis
      threadsPerWorkgroup: (dispatch.workgroupSize?.x || 1) * 
                          (dispatch.workgroupSize?.y || 1) * 
                          (dispatch.workgroupSize?.z || 1),
      totalThreads,
      totalBytes
    };
  }

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
    if (!gpuTime) return { type: 'UNKNOWN', confidence: 0, description: 'No timing data', causes: [] };

    const totalThreads = dispatch.x * dispatch.y * dispatch.z * 
                        dispatch.workgroupSize.x * dispatch.workgroupSize.y * dispatch.workgroupSize.z;
    const wgSize = dispatch.workgroupSize.x * dispatch.workgroupSize.y * dispatch.workgroupSize.z;

    // Find largest buffer for memory bandwidth analysis
    let primaryBuffer = null;
    let totalBytes = 0;
    if (dispatch.bufferAccesses && dispatch.bufferAccesses.length > 0) {
      primaryBuffer = dispatch.bufferAccesses.reduce((max, buf) => 
        (buf.size > (max?.size || 0)) ? buf : max, null);
      totalBytes = dispatch.bufferAccesses.reduce((sum, buf) => sum + buf.size, 0);
    }

    // Low occupancy check
    if (totalThreads < 256) {
      return {
        type: 'OCCUPANCY',
        confidence: 90,
        description: `Only ${totalThreads} threads - GPU severely underutilized`,
        suggestion: 'Increase workgroup count or size',
        causes: [`Workgroup count: ${dispatch.x}×${dispatch.y}×${dispatch.z}`],
        primaryBuffer: null,
        affectedBindings: []
      };
    }

    // Small workgroup check
    if (wgSize < 32) {
      return {
        type: 'OCCUPANCY',
        confidence: 80,
        description: `Workgroup size ${wgSize} < 32 - poor warp/wave utilization`,
        suggestion: 'Increase workgroup size to 64-256',
        causes: [`Workgroup size: ${dispatch.workgroupSize.x}×${dispatch.workgroupSize.y}×${dispatch.workgroupSize.z}`],
        primaryBuffer: null,
        affectedBindings: []
      };
    }

    // Atomic check with causal link
    if (shaderAnalysis.metrics.hasAtomics) {
      const atomicBuffers = dispatch.bufferAccesses?.filter(b => 
        b.bufferLabel.toLowerCase().includes('atomic') || 
        (b.usage & 0x0080) // STORAGE
      ) || [];
      
      return {
        type: 'ATOMIC_SERIALIZATION',
        confidence: 70,
        description: 'Atomics detected - likely serialization',
        suggestion: 'Use local accumulation then atomic once per workgroup',
        causes: ['Atomic operations in shader'],
        primaryBuffer: atomicBuffers[0]?.bufferLabel || null,
        affectedBindings: atomicBuffers.map(b => b.binding)
      };
    }

    // Memory bandwidth check with causal link
    if (primaryBuffer && totalBytes > 1024 * 1024 && gpuTime > 100) { // > 1MB and > 100μs
      const bandwidth = (totalBytes / 1e9) / (gpuTime / 1e6); // GB/s
      if (bandwidth > 100) { // Suspiciously high - likely memory bound
        return {
          type: 'MEMORY_BANDWIDTH',
          confidence: 82,
          description: `High memory traffic: ${(totalBytes / 1024 / 1024).toFixed(1)} MB`,
          suggestion: 'Optimize memory access patterns, use coalesced reads/writes',
          causes: [`Primary buffer: ${primaryBuffer.bufferLabel} (${(primaryBuffer.size / 1024 / 1024).toFixed(1)} MB)`],
          primaryBuffer: primaryBuffer.bufferLabel,
          affectedBindings: [primaryBuffer.binding],
          bandwidth
        };
      }
    }

    // Divergence check
    if (shaderAnalysis.metrics.hasBranching && wgSize >= 32) {
      return {
        type: 'DIVERGENCE',
        confidence: 65,
        description: 'Branching with wide workgroups - warp divergence likely',
        suggestion: 'Reorganize to reduce divergence',
        causes: ['Conditional branches in shader'],
        primaryBuffer: null,
        affectedBindings: []
      };
    }

    // Register pressure check
    if (shaderAnalysis.metrics.estimatedRegisters > 40) {
      return {
        type: 'REGISTER_PRESSURE',
        confidence: 60,
        description: 'High register usage may limit active warps',
        suggestion: 'Reduce local variables',
        causes: [`~${shaderAnalysis.metrics.estimatedRegisters} variables`],
        primaryBuffer: null,
        affectedBindings: []
      };
    }

    return {
      type: 'COMPUTE_BOUND',
      confidence: 50,
      description: 'Likely ALU limited - no obvious bottlenecks',
      suggestion: 'Use native GPU profiling tools',
      causes: ['No clear memory or occupancy issues'],
      primaryBuffer: null,
      affectedBindings: []
    };
  }

  async function benchmarkGPU(device) {
    addLog('Running GPU microbenchmarks...');
    
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

    addLog(`Estimated bandwidth: ${bandwidth.toFixed(1)} GB/s`);
    
    inputBuffer.destroy();
    outputBuffer.destroy();
    return profilerData.gpuCharacteristics;
  }

  async function hookWebGPU() {
    if (!navigator.gpu) {
      addLog('WebGPU not supported');
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

        addLog(`Device created (timestamps: ${profilerData.hasTimestamps})`);

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
          addLog(`Pipeline: ${pipeline.__capture.label} (${analysis.warnings.length} warnings)`);
          
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
          
          // Generate stable identifiers
          const workgroupSize = pipeline?.workgroupSize || { x: 1, y: 1, z: 1 };
          const shaderSource = pipeline?.shader || '';
          const variantInfo = extractVariantInfo(shaderSource);
          const variantString = JSON.stringify(variantInfo);
          const kernelId = generateKernelId(shaderSource, workgroupSize, variantString);
          
          // Initialize run if not started
          if (!profilerData.runId) {
            profilerData.runId = `run_${Date.now()}`;
            profilerData.runs[profilerData.runId] = {
              timestamp: Date.now(),
              kernels: [],
              metadata: {}
            };
          }
          
          // Track kernel if new
          if (!profilerData.kernels[kernelId]) {
            profilerData.kernels[kernelId] = {
              id: kernelId,
              label: pipeline?.label || 'unknown',
              shaderHash: hashString(shaderSource),
              workgroupSize,
              variantInfo, // NEW: Track what makes this variant unique
              dispatches: [],
              stats: {
                count: 0,
                totalTime: 0,
                avgTime: 0,
                totalBytes: 0,
                avgBandwidth: 0
              }
            };
            profilerData.runs[profilerData.runId].kernels.push(kernelId);
          }
          
          // Collect buffer accesses for causal analysis
          const bufferAccesses = [];
          if (bindGroups) {
            Object.entries(bindGroups).forEach(([idx, bg]) => {
              if (bg.__capture?.entries) {
                bg.__capture.entries.forEach(e => {
                  if (e.resource?.id && profilerData.buffers[e.resource.id]) {
                    bufferAccesses.push({
                      bufferId: e.resource.id,
                      bufferLabel: profilerData.buffers[e.resource.id].label,
                      size: profilerData.buffers[e.resource.id].size,
                      usage: profilerData.buffers[e.resource.id].usage,
                      binding: e.binding
                    });
                  }
                });
              }
            });
          }
          
          const dispatchRecord = {
            index: profilerData.dispatches.length,
            runId: profilerData.runId,
            kernelId: kernelId,
            x, y, z,
            pipeline: pipeline?.id,
            pipelineLabel: pipeline?.label || 'unknown',
            workgroupSize,
            bindGroups: Object.entries(bindGroups).map(([idx, bg]) => ({
              index: parseInt(idx),
              ...bg.__capture
            })),
            bufferAccesses, // NEW: for causal analysis
            shaderAnalysis: pipeline?.analysis || { warnings: [], metrics: {} },
            timestampStart: null,
            timestampEnd: null,
            gpuTime: null,
            timingSource: null, // NEW: 'gpu_timestamp' | 'cpu_proxy'
            timingConfidence: 1.0, // NEW: 1.0 for GPU timestamps, 0.5 for CPU proxy
            bottleneck: null,
            normalized: null // Will be computed after timing
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
              
              addLog(`Dispatch #${dispatchRecord.index}: ${dispatchRecord.pipelineLabel} (${x}×${y}×${z})`);
              
              return result;
            } catch (e) {
              // Fallback if timestamp writes fail
              addLog(`Timestamp query failed: ${e.message}`);
              profilerData.hasTimestamps = false;
              profilerData.dispatches.push(dispatchRecord);
              addLog(`Dispatch #${dispatchRecord.index}: ${dispatchRecord.pipelineLabel} (${x}×${y}×${z}) [no timing]`);
              return origDispatch.call(this, x, y, z);
            }
          } else {
            profilerData.dispatches.push(dispatchRecord);
            addLog(`Dispatch #${dispatchRecord.index}: ${dispatchRecord.pipelineLabel} (${x}×${y}×${z}) [no timing]`);
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
                  d.timingSource = 'cpu_proxy';
                  d.timingConfidence = 0.5; // CPU proxy is less reliable
                  
                  // Compute normalized metrics
                  d.normalized = computeNormalizedMetrics(d);
                  
                  // Estimate bottleneck (with reduced confidence)
                  d.bottleneck = estimateBottleneck(d, timePerDispatch, d.shaderAnalysis);
                  if (d.bottleneck) {
                    d.bottleneck.confidence = Math.round(d.bottleneck.confidence * d.timingConfidence);
                  }
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
                dispatch.timingSource = 'gpu_timestamp';
                dispatch.timingConfidence = 1.0; // Full confidence for GPU timestamps
                
                // Compute normalized metrics
                dispatch.normalized = computeNormalizedMetrics(dispatch);
                
                // Estimate bottleneck with causal analysis
                dispatch.bottleneck = estimateBottleneck(dispatch, dispatch.gpuTime, dispatch.shaderAnalysis);
                
                // Update kernel stats
                if (dispatch.kernelId && profilerData.kernels[dispatch.kernelId]) {
                  const kernel = profilerData.kernels[dispatch.kernelId];
                  kernel.dispatches.push(dispatch.index);
                  kernel.stats.count++;
                  kernel.stats.totalTime += dispatch.gpuTime;
                  kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                  
                  if (dispatch.normalized) {
                    kernel.stats.totalBytes += dispatch.normalized.totalBytes;
                    const bandwidth = (dispatch.normalized.totalBytes / 1e9) / (dispatch.gpuTime / 1e6);
                    kernel.stats.avgBandwidth = (kernel.stats.avgBandwidth * (kernel.stats.count - 1) + bandwidth) / kernel.stats.count;
                  }
                }
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

        addLog('WebGPU hooks installed');
        return device;
      };

      return adapter;
    };

    addLog('Profiler hooks initialized');
    return true;
  }

  // Scalability benchmark function
  async function runScalabilityBenchmark(shaderCode, options = {}) {
    addLog('Starting scalability benchmark...');
    
    const {
      minSize = 256,           // Start with 256 elements (1KB)
      maxSize = 16777216,      // Up to 16M elements (64MB)
      steps = 10,              // Number of test points
      workgroupSize = 256,
      dataType = 'f32',        // f32, u32, i32
      iterations = 3           // Iterations per size for averaging
    } = options;
    
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    const bytesPerElement = 4; // All types are 4 bytes
    const results = [];
    
    // Detect if shader uses atomics
    const hasAtomics = /atomic/.test(shaderCode);
    const atomicBinCount = hasAtomics ? 64 : 0; // Assume 64 histogram bins
    
    // Generate logarithmic sizes
    const logMin = Math.log2(minSize);
    const logMax = Math.log2(maxSize);
    const logStep = (logMax - logMin) / (steps - 1);
    
    for (let i = 0; i < steps; i++) {
      const size = Math.pow(2, Math.round(logMin + i * logStep));
      const sizeBytes = size * bytesPerElement;
      
      addLog(`  Testing size: ${size.toLocaleString()} elements (${(sizeBytes / 1024 / 1024).toFixed(2)} MB)`);
      
      // Create input buffer
      const inputBuffer = device.createBuffer({
        label: `benchmark_input_${size}`,
        size: sizeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      
      // Create output buffer (atomic or regular)
      const outputSize = hasAtomics ? atomicBinCount * 4 : sizeBytes;
      const outputBuffer = device.createBuffer({
        label: `benchmark_output_${size}`,
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      
      // Memory stats
      const inputMemoryMB = sizeBytes / 1024 / 1024;
      const outputMemoryMB = outputSize / 1024 / 1024;
      const totalMemoryMB = inputMemoryMB + outputMemoryMB;
      
      // Create shader with dynamic size
      const finalShaderCode = shaderCode.replace('INPUT_SIZE', size.toString());
      const shaderModule = device.createShaderModule({ code: finalShaderCode });
      
      const pipeline = device.createComputePipeline({
        label: `benchmark_pipeline_${size}`,
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' }
      });
      
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } }
        ]
      });
      
      const cpuTimes = [];
      const gpuTimes = [];
      
      // Calculate workgroup dispatch dimensions (max 65535 per dimension)
      const totalWorkgroups = Math.ceil(size / workgroupSize);
      const maxWorkgroupsPerDim = 65535;
      let dispatchX = totalWorkgroups;
      let dispatchY = 1;
      let dispatchZ = 1;
      
      if (totalWorkgroups > maxWorkgroupsPerDim) {
        // Use 2D dispatch
        dispatchX = maxWorkgroupsPerDim;
        dispatchY = Math.ceil(totalWorkgroups / maxWorkgroupsPerDim);
        if (dispatchY > maxWorkgroupsPerDim) {
          // Use 3D dispatch if needed
          dispatchY = maxWorkgroupsPerDim;
          dispatchZ = Math.ceil(totalWorkgroups / (maxWorkgroupsPerDim * maxWorkgroupsPerDim));
        }
      }
      
      // Run multiple iterations
      for (let iter = 0; iter < iterations; iter++) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        pass.end();
        
        // CPU time measurement
        const cpuStart = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const cpuEnd = performance.now();
        
        cpuTimes.push(cpuEnd - cpuStart);
        
        // GPU time would come from timestamp queries (if available)
        // For now, use CPU time as proxy
        gpuTimes.push(cpuEnd - cpuStart);
      }
      
      // Calculate average times (excluding first warmup run)
      const avgCPUTime = cpuTimes.length > 1 
        ? cpuTimes.slice(1).reduce((a, b) => a + b, 0) / (cpuTimes.length - 1)
        : cpuTimes[0];
        
      const avgGPUTime = gpuTimes.length > 1 
        ? gpuTimes.slice(1).reduce((a, b) => a + b, 0) / (gpuTimes.length - 1)
        : gpuTimes[0];
      
      // Calculate bandwidth (read + write)
      const bytesTransferred = sizeBytes + outputSize;
      const bandwidth = (bytesTransferred / 1e9) / (avgGPUTime / 1000); // GB/s
      
      // Calculate atomic contention factor
      const threadCount = Math.ceil(size / workgroupSize) * workgroupSize;
      const atomicContentionRatio = hasAtomics ? threadCount / atomicBinCount : 0;
      
      results.push({
        size,
        sizeBytes,
        sizeMB: sizeBytes / 1024 / 1024,
        
        // Timing
        cpuTimeMS: avgCPUTime,
        gpuTimeMS: avgGPUTime,
        
        // Memory
        inputMemoryMB,
        outputMemoryMB,
        totalMemoryMB,
        atomicBufferBytes: hasAtomics ? outputSize : 0,
        atomicBinCount: hasAtomics ? atomicBinCount : 0,
        
        // Performance
        bandwidth,
        throughputGEPS: (size / 1e9) / (avgGPUTime / 1000),
        
        // Atomic analysis
        hasAtomics,
        threadCount,
        atomicContentionRatio,
        avgThreadsPerBin: hasAtomics ? atomicContentionRatio : 0
      });
      
      addLog(`    CPU: ${avgCPUTime.toFixed(2)}ms, GPU: ${avgGPUTime.toFixed(2)}ms, BW: ${bandwidth.toFixed(2)} GB/s`);
      if (hasAtomics) {
        addLog(`    Atomic contention: ${atomicContentionRatio.toFixed(0)}x threads per bin`);
      }
      
      inputBuffer.destroy();
      outputBuffer.destroy();
    }
    
    addLog('Scalability benchmark complete!');
    
    // Calculate total input memory processed
    const totalInputMemory = results.reduce((sum, r) => sum + r.inputMemoryMB, 0);
    
    // Return results for plotting
    return {
      results,
      peakBandwidth: profilerData.gpuCharacteristics?.estimatedBandwidth || 0,
      summary: {
        minBandwidth: Math.min(...results.map(r => r.bandwidth)),
        maxBandwidth: Math.max(...results.map(r => r.bandwidth)),
        avgBandwidth: results.reduce((sum, r) => sum + r.bandwidth, 0) / results.length,
        totalInputMemoryMB: totalInputMemory,
        hasAtomics,
        atomicBinCount: hasAtomics ? atomicBinCount : 0
      }
    };
  }

  // Public API
  window.WebSight = {
    start: async function() {
      addLog('Starting WebGPU Profiler...');
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
      addLog('Profiler data cleared');
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
      
      addLog('Profile exported');
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
    },
    
    // New scalability benchmark API
    benchmarkScalability: async function(shaderCode, options) {
      return await runScalabilityBenchmark(shaderCode, options);
    },
    
    // ===== MCP TOOLS FOR AI REASONING =====
    
    // Tool: list_kernels
    listKernels: function() {
      return Object.values(profilerData.kernels).map(k => ({
        id: k.id,
        label: k.label,
        dispatchCount: k.stats.count,
        avgTime: k.stats.avgTime,
        avgBandwidth: k.stats.avgBandwidth,
        workgroupSize: k.workgroupSize
      }));
    },
    
    // Tool: get_kernel_summary
    getKernelSummary: function(kernelId) {
      const kernel = profilerData.kernels[kernelId];
      if (!kernel) return null;
      
      const dispatches = kernel.dispatches.map(idx => profilerData.dispatches[idx]);
      const bottlenecks = {};
      dispatches.forEach(d => {
        if (d.bottleneck) {
          bottlenecks[d.bottleneck.type] = (bottlenecks[d.bottleneck.type] || 0) + 1;
        }
      });
      
      return {
        id: kernel.id,
        label: kernel.label,
        dispatchCount: kernel.stats.count,
        avgTime: kernel.stats.avgTime,
        avgBandwidth: kernel.stats.avgBandwidth,
        totalTime: kernel.stats.totalTime,
        workgroupSize: kernel.workgroupSize,
        bottlenecks,
        dispatches: dispatches.map(d => ({
          index: d.index,
          gpuTime: d.gpuTime,
          bottleneck: d.bottleneck,
          normalized: d.normalized
        }))
      };
    },
    
    // Tool: compare_kernels
    compareKernels: function(kernelIdA, kernelIdB) {
      const summaryA = this.getKernelSummary(kernelIdA);
      const summaryB = this.getKernelSummary(kernelIdB);
      
      if (!summaryA || !summaryB) return null;
      
      const timeDiff = summaryB.avgTime - summaryA.avgTime;
      const bandwidthDiff = summaryB.avgBandwidth - summaryA.avgBandwidth;
      
      return {
        kernelA: { id: kernelIdA, label: summaryA.label },
        kernelB: { id: kernelIdB, label: summaryB.label },
        metrics: {
          avgTime: { a: summaryA.avgTime, b: summaryB.avgTime, diff: timeDiff, percentChange: (timeDiff / summaryA.avgTime * 100).toFixed(1) },
          avgBandwidth: { a: summaryA.avgBandwidth, b: summaryB.avgBandwidth, diff: bandwidthDiff, percentChange: (bandwidthDiff / summaryA.avgBandwidth * 100).toFixed(1) }
        },
        bottlenecks: {
          a: summaryA.bottlenecks,
          b: summaryB.bottlenecks
        },
        conclusion: timeDiff > 0 
          ? `Kernel B is ${(timeDiff / summaryA.avgTime * 100).toFixed(1)}% slower` 
          : `Kernel B is ${Math.abs(timeDiff / summaryA.avgTime * 100).toFixed(1)}% faster`
      };
    },
    
    // NEW: Tool: compare_runs (KILLER FOR CI)
    compareRuns: function(runIdA, runIdB) {
      const runA = profilerData.runs[runIdA];
      const runB = profilerData.runs[runIdB];
      
      if (!runA || !runB) {
        return { error: 'One or both runs not found', availableRuns: Object.keys(profilerData.runs) };
      }
      
      // Find common kernels (by shader hash, not full kernelId)
      const kernelsA = runA.kernels.map(kid => profilerData.kernels[kid]);
      const kernelsB = runB.kernels.map(kid => profilerData.kernels[kid]);
      
      const commonKernels = [];
      const addedKernels = [];
      const removedKernels = [];
      const regressions = [];
      const improvements = [];
      
      // Group by shader hash to compare variants
      const kernelsByHash = {};
      kernelsA.forEach(k => {
        if (!kernelsByHash[k.shaderHash]) kernelsByHash[k.shaderHash] = { a: [], b: [] };
        kernelsByHash[k.shaderHash].a.push(k);
      });
      kernelsB.forEach(k => {
        if (!kernelsByHash[k.shaderHash]) kernelsByHash[k.shaderHash] = { a: [], b: [] };
        kernelsByHash[k.shaderHash].b.push(k);
      });
      
      // Compare each kernel
      Object.entries(kernelsByHash).forEach(([hash, { a, b }]) => {
        if (a.length > 0 && b.length > 0) {
          // Compare best variant from each run
          const bestA = a.reduce((min, k) => k.stats.avgTime < min.stats.avgTime ? k : min, a[0]);
          const bestB = b.reduce((min, k) => k.stats.avgTime < min.stats.avgTime ? k : min, b[0]);
          
          const timeDiff = bestB.stats.avgTime - bestA.stats.avgTime;
          const percentChange = (timeDiff / bestA.stats.avgTime * 100);
          
          const comparison = {
            label: bestA.label,
            shaderHash: hash,
            runA: {
              kernelId: bestA.id,
              avgTime: bestA.stats.avgTime,
              avgBandwidth: bestA.stats.avgBandwidth,
              dispatchCount: bestA.stats.count,
              variantInfo: bestA.variantInfo
            },
            runB: {
              kernelId: bestB.id,
              avgTime: bestB.stats.avgTime,
              avgBandwidth: bestB.stats.avgBandwidth,
              dispatchCount: bestB.stats.count,
              variantInfo: bestB.variantInfo
            },
            diff: {
              avgTime: timeDiff,
              percentChange: percentChange.toFixed(1),
              status: Math.abs(percentChange) < 5 ? 'neutral' : percentChange > 0 ? 'regressed' : 'improved'
            }
          };
          
          commonKernels.push(comparison);
          
          if (percentChange > 5) {
            regressions.push(comparison);
          } else if (percentChange < -5) {
            improvements.push(comparison);
          }
        } else if (a.length > 0) {
          removedKernels.push({ label: a[0].label, stats: a[0].stats });
        } else if (b.length > 0) {
          addedKernels.push({ label: b[0].label, stats: b[0].stats });
        }
      });
      
      // Calculate total time
      const totalTimeA = kernelsA.reduce((sum, k) => sum + k.stats.totalTime, 0);
      const totalTimeB = kernelsB.reduce((sum, k) => sum + k.stats.totalTime, 0);
      const totalTimeDiff = totalTimeB - totalTimeA;
      
      // Generate CI-friendly summary
      const summary = {
        status: regressions.length > 0 ? 'FAILED' : improvements.length > 0 ? 'IMPROVED' : 'NEUTRAL',
        totalTime: {
          runA: totalTimeA,
          runB: totalTimeB,
          diff: totalTimeDiff,
          percentChange: ((totalTimeDiff / totalTimeA) * 100).toFixed(1)
        },
        kernelCounts: {
          common: commonKernels.length,
          added: addedKernels.length,
          removed: removedKernels.length,
          regressed: regressions.length,
          improved: improvements.length
        }
      };
      
      return {
        runA: { id: runIdA, timestamp: runA.timestamp, kernelCount: kernelsA.length },
        runB: { id: runIdB, timestamp: runB.timestamp, kernelCount: kernelsB.length },
        summary,
        regressions: regressions.sort((a, b) => parseFloat(b.diff.percentChange) - parseFloat(a.diff.percentChange)),
        improvements: improvements.sort((a, b) => parseFloat(a.diff.percentChange) - parseFloat(b.diff.percentChange)),
        commonKernels,
        addedKernels,
        removedKernels,
        ciMessage: this._generateCIMessage(summary, regressions, improvements)
      };
    },
    
    // Helper: Generate CI-friendly message
    _generateCIMessage: function(summary, regressions, improvements) {
      let message = `## WebSight Performance Report\n\n`;
      message += `**Status:** ${summary.status}\n`;
      message += `**Total Time Change:** ${summary.totalTime.diff > 0 ? '+' : ''}${summary.totalTime.percentChange}%\n\n`;
      
      if (regressions.length > 0) {
        message += `### ⚠️ Regressions (${regressions.length})\n\n`;
        regressions.slice(0, 5).forEach(r => {
          message += `- **${r.label}**: +${r.diff.percentChange}% slower (${r.runA.avgTime.toFixed(2)}μs → ${r.runB.avgTime.toFixed(2)}μs)\n`;
        });
        if (regressions.length > 5) {
          message += `- ...and ${regressions.length - 5} more\n`;
        }
        message += `\n`;
      }
      
      if (improvements.length > 0) {
        message += `### ✅ Improvements (${improvements.length})\n\n`;
        improvements.slice(0, 5).forEach(i => {
          message += `- **${i.label}**: ${i.diff.percentChange}% faster (${i.runA.avgTime.toFixed(2)}μs → ${i.runB.avgTime.toFixed(2)}μs)\n`;
        });
        if (improvements.length > 5) {
          message += `- ...and ${improvements.length - 5} more\n`;
        }
      }
      
      return message;
    },
    
    // Tool: explain_bottleneck
    explainBottleneck: function(dispatchId) {
      const dispatch = profilerData.dispatches[dispatchId];
      if (!dispatch || !dispatch.bottleneck) return null;
      
      const b = dispatch.bottleneck;
      return {
        type: b.type,
        confidence: b.confidence,
        description: b.description,
        suggestion: b.suggestion,
        causes: b.causes || [],
        primaryBuffer: b.primaryBuffer,
        affectedBindings: b.affectedBindings || [],
        context: {
          workgroups: `${dispatch.x}×${dispatch.y}×${dispatch.z}`,
          workgroupSize: `${dispatch.workgroupSize.x}×${dispatch.workgroupSize.y}×${dispatch.workgroupSize.z}`,
          totalThreads: dispatch.normalized?.totalThreads,
          gpuTime: dispatch.gpuTime
        }
      };
    },
    
    // Tool: focus_ui (for UI control)
    focusUI: function(view, target) {
      if (view === 'kernel' && target) {
        // Broadcast event for UI to handle
        window.dispatchEvent(new CustomEvent('websight:focus', { 
          detail: { view, target } 
        }));
        return { success: true, view, target };
      }
      return { success: false, error: 'Invalid view or target' };
    },
    
    // Tool: detect_phase_shifts (for scalability analysis)
    detectPhaseShifts: function(benchmarkResults) {
      if (!benchmarkResults || !benchmarkResults.results) return [];
      
      const results = benchmarkResults.results;
      const shifts = [];
      
      for (let i = 1; i < results.length - 1; i++) {
        const prev = results[i - 1];
        const curr = results[i];
        const next = results[i + 1];
        
        // Detect slope change (derivative change)
        const slope1 = (curr.bandwidth - prev.bandwidth) / (curr.sizeMB - prev.sizeMB);
        const slope2 = (next.bandwidth - curr.bandwidth) / (next.sizeMB - curr.sizeMB);
        const slopeChange = Math.abs(slope2 - slope1);
        
        // Detect saturation (bandwidth stops growing)
        const isSaturated = slope2 < 0.1 && curr.bandwidth > 1.0;
        
        if (slopeChange > 5.0 || isSaturated) {
          shifts.push({
            index: i,
            size: curr.size,
            sizeMB: curr.sizeMB,
            bandwidth: curr.bandwidth,
            type: isSaturated ? 'saturation' : 'slope_change',
            description: isSaturated 
              ? `Performance plateaus after ${curr.sizeMB.toFixed(1)} MB → likely cache capacity boundary`
              : `Significant slope change at ${curr.sizeMB.toFixed(1)} MB`,
            slopeChange: slopeChange.toFixed(2)
          });
        }
      }
      
      return shifts;
    },
    
    // Enhanced export with analysis
    exportAnalysis: function() {
      const rawData = this.getData();
      const stats = this.getStats();
      const kernels = this.listKernels();
      
      // Compute derived metrics
      const derivedMetrics = {
        totalDispatches: rawData.dispatches.length,
        uniqueKernels: kernels.length,
        bottleneckDistribution: {},
        avgNormalizedMetrics: {
          nsPerElement: 0,
          bytesPerThread: 0
        }
      };
      
      rawData.dispatches.forEach(d => {
        if (d.bottleneck) {
          derivedMetrics.bottleneckDistribution[d.bottleneck.type] = 
            (derivedMetrics.bottleneckDistribution[d.bottleneck.type] || 0) + 1;
        }
        if (d.normalized) {
          derivedMetrics.avgNormalizedMetrics.nsPerElement += d.normalized.nsPerElement || 0;
          derivedMetrics.avgNormalizedMetrics.bytesPerThread += d.normalized.bytesPerThread || 0;
        }
      });
      
      derivedMetrics.avgNormalizedMetrics.nsPerElement /= rawData.dispatches.length || 1;
      derivedMetrics.avgNormalizedMetrics.bytesPerThread /= rawData.dispatches.length || 1;
      
      // Generate conclusions
      const conclusions = [];
      const topBottleneck = Object.entries(derivedMetrics.bottleneckDistribution)
        .sort((a, b) => b[1] - a[1])[0];
      if (topBottleneck) {
        conclusions.push(`Primary bottleneck: ${topBottleneck[0]} (${topBottleneck[1]} dispatches)`);
      }
      
      // Confidence scores
      const confidenceScores = {};
      rawData.dispatches.forEach(d => {
        if (d.bottleneck) {
          confidenceScores[d.bottleneck.type] = 
            (confidenceScores[d.bottleneck.type] || []).concat(d.bottleneck.confidence);
        }
      });
      
      Object.keys(confidenceScores).forEach(key => {
        const scores = confidenceScores[key];
        confidenceScores[key] = scores.reduce((a, b) => a + b, 0) / scores.length;
      });
      
      const analysisData = {
        metadata: {
          timestamp: new Date().toISOString(),
          runId: profilerData.runId,
          tool: 'WebSight',
          version: '1.0.0'
        },
        rawData,
        derivedMetrics,
        conclusions,
        confidenceScores,
        kernels: kernels.map(k => this.getKernelSummary(k.id))
      };
      
      const blob = new Blob([JSON.stringify(analysisData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `websight-analysis-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      
      addLog('Analysis exported');
      return analysisData;
    }
  };

  // Auto-start
  console.log('WebGPU Compute Profiler loaded');
  console.log('Use WebSight.start() to begin profiling');
  console.log('Use WebSight.getData() to get captured data');
  console.log('Use WebSight.export() to download report');
})();
