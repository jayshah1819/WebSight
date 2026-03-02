// =============================================================================
// MODULE 10: WebGPU API Hooks
// =============================================================================


import { TimingHelper } from './timing-helper.js';
import { profilerData } from './data-store.js';
import { memoryLeakDetector, workgroupAnalyzer, log, warn, error } from './init-logging.js';
import { broadcastData } from './broadcast.js';
import { normalizeTime, getTimeUnitLabel, generateKernelId, extractWorkgroupSize, analyzeWGSL, addLog } from './utils.js';
import { analyzeMemoryAccessPattern, calculateTextureSize, parseWGSLBindings, BandwidthTracker,
         calculateDispatchBandwidth, calculateRenderDrawBandwidth } from './bandwidth-tracker.js';

  async function hookWebGPU() {
    if (!navigator.gpu) {
      addLog('WebGPU not available', 'error');
      return;
    }

    if (navigator.gpu.__webSightHooked) {
      addLog('hookWebGPU() called again — already hooked, ignoring', 'warn');
      return;
    }
    navigator.gpu.__webSightHooked = true;

    if (!window.__webSightAdapters) {
      window.__webSightAdapters = [];
    }
    if (!window.__webSightDevices) {
      window.__webSightDevices = [];
    }

    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(contextType, ...args) {
      const context = origGetContext.call(this, contextType, ...args);
      
      if (contextType === 'webgpu' && context) {
  
        const origGetCurrentTexture = context.getCurrentTexture.bind(context);
  
        let canvasTextureId = crypto.randomUUID();
        
        context.getCurrentTexture = function() {
          const texture = origGetCurrentTexture();
          
          if (texture && !texture.__websight_id) {
   
            if (context.__lastCanvasTexture && context.__lastCanvasTexture !== texture) {
              memoryLeakDetector.markDestroyed(context.__lastCanvasTexture);
            }
            context.__lastCanvasTexture = texture;

            const canvas = this.canvas;
            texture.__websight_id = canvasTextureId;
            texture.__websight_metadata = {
              id: canvasTextureId,
              label: 'canvas_texture',
              width: canvas.width,
              height: canvas.height,
              depthOrArrayLayers: 1,
              mipLevelCount: 1,
              format: navigator.gpu.getPreferredCanvasFormat(),
              usage: GPUTextureUsage.RENDER_ATTACHMENT
            };
        
            const canvasTextureSize = (canvas.width || 1) * (canvas.height || 1) * 4;
            memoryLeakDetector.trackResource(texture, 'GPUTexture (canvas)', canvasTextureSize);

            const origCreateView = texture.createView.bind(texture);
            texture.createView = function(viewDesc) {
              const view = origCreateView(viewDesc);
              view.__websight_texture = texture.__websight_metadata;
              return view;
            };
          }
          
          return texture;
        };
      }
      
      return context;
    };

    const originalRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    
    navigator.gpu.requestAdapter = async function(options) {
      const adapter = await originalRequestAdapter(options);
      if (!adapter) return adapter;

      const hasTimestampFeature = adapter.features.has('timestamp-query');
      
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

          const deviceTimingMode = shouldUseGPUTiming ? 'gpu' : 'cpu-only';
          
          const deviceInfo = {
            hasTimestampQuery,
            createdAt: Date.now(),
            label: descriptor?.label || `device_${window.__webSightDevices.length + 1}`,
            features: Array.from(device.features),
            limits: { ...device.limits },
            timingMode: deviceTimingMode,
            encoderCount: 0,
            passCount: 0,
            dispatchCount: 0
          };
          window.__webSightDevices.push(deviceInfo);
          adapterInfo.devices.push(deviceInfo);

          if (!workgroupAnalyzer.deviceLimits && device.limits) {
            workgroupAnalyzer.setDeviceLimits(device.limits);
            console.log('[WebSight] Workgroup analysis initialized - monitoring dispatch geometry');
          }
          

          device.addEventListener('uncapturederror', (event) => {
            if (event.error.message && event.error.message.includes('Cannot allocate sample buffer')) {
              console.error('[WebSight] GPU QuerySet allocation failed!');
              console.log('[WebSight] Your GPU has reached its QuerySet limit.');
              console.log('[WebSight]   <script>window.__webSightDisableGPUTiming = true;</script>');
              deviceInfo.timingMode = 'cpu-only';
            }
          });
          
          device.__webSightInfo = deviceInfo;
          
          addLog(`Device ${window.__webSightDevices.length} created - Timing: ${deviceTimingMode}, Label: ${deviceInfo.label}`);

          // Hook createShaderModule
          const origCreateShaderModule = device.createShaderModule.bind(device);
          device.createShaderModule = function(desc) {
            const module = origCreateShaderModule(desc);
            const shaderId = crypto.randomUUID();
            
            module.__source = desc.code;
            module.__shaderId = shaderId;
            
            return module;
          };


          function setupPipeline(pipeline, desc, defaultLabel) {
            const source      = desc.compute.module.__source || '';
            const entryPoint  = desc.compute.entryPoint || null;
            const workgroupSize = extractWorkgroupSize(source, entryPoint);
            const label       = desc.label || defaultLabel;

            pipeline.__capture = {
              id: crypto.randomUUID(),
              label,
              entryPoint,
              workgroupSize,
              shader: source,
              shaderId: desc.compute.module.__shaderId,
              analysis: analyzeWGSL(source)
            };
            profilerData.pipelines[pipeline.__capture.id] = pipeline.__capture;

            const kernelId = generateKernelId(source, workgroupSize, label);
            if (!profilerData.kernels[kernelId]) {
              profilerData.kernels[kernelId] = {
                id: kernelId,
                label,
                workgroupSize,
                shader: source,
                shaderId: desc.compute.module.__shaderId,
                stats: { count: 0, totalTime: 0, avgTime: 0, minTime: Infinity, maxTime: 0 }
              };
            }

            const parsedBindings = parseWGSLBindings(source);
            const origGetBGL = pipeline.getBindGroupLayout.bind(pipeline);
            pipeline.getBindGroupLayout = function(groupIndex) {
              const layout = origGetBGL(groupIndex);
              if (!layout.__websight_entries && parsedBindings[groupIndex]) {
                layout.__websight_entries = {};
                for (const [binding, bufType] of Object.entries(parsedBindings[groupIndex])) {
                  layout.__websight_entries[binding] = {
                    visibility: 0,
                    bufferType: bufType,
                    hasTexture: false,
                    hasStorageTexture: false,
                  };
                }
              }
              return layout;
            };

            return pipeline;
          }

          const origCreateComputePipeline = device.createComputePipeline.bind(device);
          device.createComputePipeline = function(desc) {
            const pipeline = origCreateComputePipeline(desc);
            return setupPipeline(pipeline, desc, 'compute_pipeline');
          };


          const origCreateComputePipelineAsync = device.createComputePipelineAsync.bind(device);
          device.createComputePipelineAsync = async function(desc) {
            const pipeline = await origCreateComputePipelineAsync(desc);
            return setupPipeline(pipeline, desc, 'compute_pipeline_async');
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
            
            // Skip profiler-internal TimingHelper buffers
            const isProfilerInternal = (desc.label || '').startsWith('TimingHelper');
            if (!isProfilerInternal) {
              memoryLeakDetector.trackResource(buffer, 'GPUBuffer', desc.size);
            }
            
            const origDestroy = buffer.destroy.bind(buffer);
            buffer.destroy = function() {
              memoryLeakDetector.markDestroyed(buffer);
              origDestroy();
            };
            
            return buffer;
          };

          // Hook createTexture to store metadata for bandwidth calculation
          const origCreateTexture = device.createTexture.bind(device);
          device.createTexture = function(desc) {
            const texture = origCreateTexture(desc);
            const textureId = crypto.randomUUID();
            
            texture.__websight_id = textureId;
            texture.__websight_metadata = {
              id: textureId,
              label: desc.label || 'texture',
              width: desc.size.width,
              height: desc.size.height,
              depthOrArrayLayers: desc.size.depthOrArrayLayers || 1,
              mipLevelCount: desc.mipLevelCount || 1,
              format: desc.format,
              usage: desc.usage
            };
            
            // Hook createView to link views back to texture
            const origCreateView = texture.createView.bind(texture);
            texture.createView = function(viewDesc) {
              const view = origCreateView(viewDesc);
              view.__websight_texture = texture.__websight_metadata;
              return view;
            };
            
            // Track for memory leak detection — use same format map as calculateTextureSize
            const formatBytesPerPixel = {
              'r8unorm': 1, 'r8snorm': 1, 'r8uint': 1, 'r8sint': 1,
              'r16uint': 2, 'r16sint': 2, 'r16float': 2, 'rg8unorm': 2, 'rg8snorm': 2,
              'r32uint': 4, 'r32sint': 4, 'r32float': 4, 'rg16uint': 4, 'rg16sint': 4, 'rg16float': 4,
              'rgba8unorm': 4, 'rgba8unorm-srgb': 4, 'rgba8snorm': 4, 'rgba8uint': 4, 'rgba8sint': 4,
              'bgra8unorm': 4, 'bgra8unorm-srgb': 4,
              'rgb10a2unorm': 4, 'rg11b10ufloat': 4,
              'rg32uint': 8, 'rg32sint': 8, 'rg32float': 8, 'rgba16uint': 8, 'rgba16sint': 8, 'rgba16float': 8,
              'rgba32uint': 16, 'rgba32sint': 16, 'rgba32float': 16,
              'depth32float': 4, 'depth24plus': 4, 'depth24plus-stencil8': 5, 'depth32float-stencil8': 5
            };
            const bytesPerPixel = formatBytesPerPixel[desc.format] || 4;
            const textureSize = desc.size.width * desc.size.height * 
                               (desc.size.depthOrArrayLayers || 1) * bytesPerPixel;
            memoryLeakDetector.trackResource(texture, 'GPUTexture', textureSize);
            
            // Hook destroy
            const origDestroy = texture.destroy.bind(texture);
            texture.destroy = function() {
              memoryLeakDetector.markDestroyed(texture);
              origDestroy();
            };
            
            return texture;
          };

          // Hook createBindGroupLayout to capture per-binding buffer types
          const origCreateBindGroupLayout = device.createBindGroupLayout.bind(device);
          device.createBindGroupLayout = function(desc) {
            const layout = origCreateBindGroupLayout(desc);
     
            layout.__websight_entries = {};
            for (const entry of (desc.entries || [])) {
              layout.__websight_entries[entry.binding] = {
                visibility: entry.visibility,
                bufferType: entry.buffer?.type || null,   // 'uniform' | 'storage' | 'read-only-storage'
                hasTexture: !!entry.texture,
                hasStorageTexture: !!entry.storageTexture,
              };
            }
            return layout;
          };

          // Hook createBindGroup to track resources with access patterns
          const origCreateBindGroup = device.createBindGroup.bind(device);
          device.createBindGroup = function(desc) {
            const bg = origCreateBindGroup(desc);
            
        
            const layoutEntries = desc.layout?.__websight_entries || {};
            
            bg.__websight_resources = [];
            bg.__capture = {
              entries: desc.entries.map(e => {
                let resource = null;
                let accessType = 'read-only';
                
                if (e.resource.buffer) {
                  const buffer = e.resource.buffer;
                  const usage = buffer.__capture?.usage || 0;
                  
                  // Prefer the bind group layout's buffer.type (precise)
                  const layoutEntry = layoutEntries[e.binding];
                  if (layoutEntry && layoutEntry.bufferType) {
                    // 'read-only-storage' → read-only, 'storage' → read-write, 'uniform' → read-only
                    if (layoutEntry.bufferType === 'read-only-storage') {
                      accessType = 'read-only';
                    } else if (layoutEntry.bufferType === 'storage') {
                      accessType = 'read-write';
                    } else {
                      accessType = 'read-only'; // uniform
                    }
                  } else if (usage & GPUBufferUsage.STORAGE) {
                    // Fallback: no layout info — guess from usage flags
                    accessType = 'read-write';
                  } else if (usage & GPUBufferUsage.UNIFORM) {
                    accessType = 'read-only';
                  }
                  
                  resource = {
                    id: buffer.__capture?.id || crypto.randomUUID(),
                    size: e.resource.size || buffer.size || 0,
                    type: 'buffer',
                    usage: usage,
                    accessType: accessType
                  };
                  
                  bg.__websight_resources.push(resource);
                } else if (e.resource instanceof GPUTextureView || e.resource.texture) {
                  const textureView = e.resource;
                  const textureSize = calculateTextureSize(textureView);
                  
                  const layoutEntry = layoutEntries[e.binding];
                  if (layoutEntry && layoutEntry.hasStorageTexture) {
                    accessType = 'read-write';
                  } else {
                    accessType = 'read-only';
                  }
                  
                  resource = {
                    id: textureView.__websight_texture?.id || crypto.randomUUID(),
                    size: textureSize,
                    type: 'texture',
                    usage: 0,
                    accessType: accessType
                  };
                  
                  bg.__websight_resources.push(resource);
                }
                
                return { 
                  binding: e.binding, 
                  resource: resource || (e.resource.buffer?.__capture || e.resource)
                };
              })
            };
            
            return bg;
          };

          // Global timing accumulator
          if (!window.__webSightGlobalTimingResults) {
            window.__webSightGlobalTimingResults = [];
            window.__webSightMaxTimingResults = 10000; 
          }
          

   
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
                maxSize: 8,
                passesPerHelper: 1,
                failed: false,
                limitReached: false
              });
            }
            
            const pool = window.__webSightTimingHelperPools.get(device);
            
            if (pool.failed) {
              return null;
            }
            
            if (pool.available.length > 0) {
              const helper = pool.available.shift();
              pool.inUse.add(helper);
              return helper;
            }
            
            if (pool.helpers.length < pool.maxSize && !pool.limitReached) {
              try {
                device.pushErrorScope('validation');
                const helper = new TimingHelper(device, pool.passesPerHelper);
                
                helper.__valid = true;

                device.popErrorScope().then(error => {
                  if (error) {
                    console.error('[WebSight]  GPU QuerySet creation failed:', error.message);
                    helper.__valid = false;
                    pool.limitReached = true;
                    pool.failed = true;
                    const idx = pool.helpers.indexOf(helper);
                    if (idx >= 0) pool.helpers.splice(idx, 1);
                    pool.inUse.delete(helper);
                  }
                });
                
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
                  
                    device.__webSightInfo.timingMode = 'cpu-only';
                    return null;
                  }
                  console.log(`[WebSight] Max concurrent timing operations: ${pool.helpers.length}. Render passes will continue without timing until helpers become available.`);
                  return null;
                } else {
                  console.warn('[WebSight] Cannot create TimingHelper:', e.message);
                  if (pool.helpers.length === 0) {
                    pool.failed = true;
                    console.error('[WebSight] GPU timing completely unavailable.');
                    device.__webSightInfo.timingMode = 'cpu-only';
                    return null;
                  }
                  return null;
                }
              }
            }
            
         
            pool.missedCount = (pool.missedCount || 0) + 1;
            if (pool.missedCount % 50 === 0) {
              const reason = pool.limitReached ? 'GPU QuerySet limit reached' : 'all helpers in use';
              console.warn(`[WebSight] Timing dropped for ${pool.missedCount} passes (${reason}).`);
            }
            if (pool.limitReached) pool.index++; 

            return null;
          };
          
          const releaseTimingHelper = (device, helper) => {
            const pool = window.__webSightTimingHelperPools.get(device);
            if (!pool || !helper) return;
            
            pool.inUse.delete(helper);
            pool.available.push(helper);
          };
          
          const origCreateCommandEncoder = device.createCommandEncoder.bind(device);
          device.createCommandEncoder = function(desc) {
            const encoder = origCreateCommandEncoder(desc);
            const origBeginComputePass = encoder.beginComputePass.bind(encoder);
            const origBeginRenderPass = encoder.beginRenderPass.bind(encoder);
            const origFinish = encoder.finish.bind(encoder);

     
            device.__webSightInfo.encoderCount++;

       
            const encoderData = {
              dispatches: [],
              startTime: performance.now(),
              id: crypto.randomUUID(),
              passCount: 0, // Count passes for this encoder
              device: device,
              deviceLabel: device.__webSightInfo.label
            };
            profilerData.activeEncoders.set(encoder, encoderData);

            let passTimings = [];
            
 
            if (device.__webSightInfo.timingMode === 'gpu' && !profilerData.config.minimalOverhead) {
              try {
                console.log(`[WebSight] GPU timing enabled for encoder "${desc?.label || 'unlabeled'}"`);
              } catch (e) {
                console.error(`[WebSight] GPU timing setup failed: ${e.message}`);
              }
            }

      
            const proxyEncoder = {
              beginComputePass: origBeginComputePass,
              beginRenderPass: origBeginRenderPass,
              resolveQuerySet: encoder.resolveQuerySet.bind(encoder),
              copyBufferToBuffer: encoder.copyBufferToBuffer.bind(encoder)
            };

            encoder.beginComputePass = function(passDesc) {
              let passTimingHelper = null;
              if (device.__webSightInfo.timingMode === 'gpu' && !profilerData.config.minimalOverhead) {
                passTimingHelper = getTimingHelper(device);
              }
              
              let pass;
              if (passTimingHelper) {
                try {
                  pass = passTimingHelper.beginComputePass(proxyEncoder, passDesc);
                } catch (e) {
                  console.warn('[WebSight] TimingHelper.beginComputePass failed, continuing without timing:', e.message);
                  pass = origBeginComputePass(passDesc);
                  passTimingHelper = null;
                }
              } else {
                pass = origBeginComputePass(passDesc);
              }
              
              encoderData.passCount++;
              device.__webSightInfo.passCount++;
              
              pass.__dispatches = [];
              pass.__passId = crypto.randomUUID();

              if (passTimingHelper) {
                passTimings.push({ helper: passTimingHelper, dispatches: pass.__dispatches });
              }

              pass.__boundPipeline = null;
              pass.__boundBindGroups = {};
              pass.__timingHelper = passTimingHelper;
              pass.__passType = 'compute';
              pass.__deviceLabel = device.__webSightInfo.label;
              pass.__bandwidthTracker = new BandwidthTracker();
       
              pass.__bandwidthSnapshot = { read: 0, written: 0 };

              const origSetPipeline = pass.setPipeline.bind(pass);
              pass.setPipeline = function(p) {
                this.__boundPipeline = p;
                origSetPipeline(p);
              };

              const origSetBindGroup = pass.setBindGroup.bind(pass);
              pass.setBindGroup = function(i, bg) {
                this.__boundBindGroups[i] = bg;
                this.__bandwidthTracker.trackBindGroup(bg);
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
                
  
                const baseLabel = pipelineObj?.label || pipeline?.label || 'compute_pipeline';
                const entryPoint = pipeline?.entryPoint || null;
                const pipelineLabel = entryPoint ? `${baseLabel} [${entryPoint}]` : baseLabel;

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

                // Check dispatch dimensions
                const maxDim = device.limits?.maxComputeWorkgroupsPerDimension || 65535;
                const dispatchX = x || 1;
                const dispatchY = y || 1;
                const dispatchZ = z || 1;
                
                let dimensionViolation = false;
                let violationMsg = '';
                
 
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
                
              
                if (dimensionViolation) {
                  const errorMsg = `DISPATCH GEOMETRY ERROR: ${violationMsg}`;

                  profilerData.logs.push({
                    timestamp: Date.now(),
                    level: 'error',
                    category: 'dispatch-geometry',
                    message: errorMsg,
                    details: `Pipeline: "${pipelineLabel}"\nDispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}]\nWorkgroup: [${pipeline.workgroupSize?.join(', ') || '?'}]\nMax Allowed Per Dimension: ${maxDim}\n\nDISPATCH BLOCKED by profiler to prevent GPU device loss.\nRoot Cause: Likely nested if-inside-while in getSimpleDispatchGeometry(). Y dimension must be reduced in a separate while loop AFTER X is fully reduced.`
                  });

                  // Still run the workgroup analyzer so the violation appears in getAllAnalyses().
                  // Without this the blocked dispatch is invisible to any downstream analysis UI.
                  if (profilerData.config.enableWorkgroupAnalysis) {
                    workgroupAnalyzer.analyzeDispatch({
                      kernelId,
                      pipelineLabel,
                      workgroupSize: workgroupSizeArray,
                      dispatchSize: [dispatchX, dispatchY, dispatchZ],
                      dimensionViolation: true
                    });
                  }

                  broadcastData();
                  return;
                }

                const cpuStart = performance.now();
                origDispatch(x, y, z);
                const cpuEnd = performance.now();
                const cpuTimeMs = cpuEnd - cpuStart;

                const dispatchRecord = {
                  index: profilerData.dispatches.length,
                  kernelId: kernelId,
                  pipelineLabel: pipelineLabel,
                  workgroupSize: workgroupSizeArray,
                  dispatchSize: [x || 1, y || 1, z || 1],
                  x, y, z,
                  cpuStart,
                  cpuEnd,
                  cpuTimeMs: cpuTimeMs,
                  gpuTimeMs: 0,
                  timingSource: 'pending',
                  normalizedTime: cpuTimeMs,
                  timeUnit: getTimeUnitLabel(),
                  timestampStart: -1,
                  timestampEnd: -1,
                  deviceLabel: device.__webSightInfo.label,
                  passType: 'compute',
                  passId: pass.__passId,
                  dimensionViolation: dimensionViolation
                };

                const bufferAccesses = Object.values(this.__boundBindGroups).flatMap(bg =>
                    bg.__websight_resources || []
                ).map(res => ({
                    id: res.id,
                    size: res.size,
                    type: res.type,
                    accessType: res.accessType
                }));

        
                const memoryPattern = analyzeMemoryAccessPattern(
                    pipeline.shader,
                    bufferAccesses,
                    workgroupSizeArray,
                    [x || 1, y || 1, z || 1]
                );
                dispatchRecord.memoryPattern = memoryPattern;

                const { bandwidth: dispatchBandwidth, newSnapshot } = calculateDispatchBandwidth(
                    this.__bandwidthTracker, this.__bandwidthSnapshot, memoryPattern.accessPatternType
                );
                this.__bandwidthSnapshot = newSnapshot;

                dispatchRecord.bufferAccesses = bufferAccesses;
                dispatchRecord.bandwidth = dispatchBandwidth;
                
                if (profilerData.config.enableWorkgroupAnalysis) {
                  const analysis = workgroupAnalyzer.analyzeDispatch(dispatchRecord);
                  if (analysis) {
                    dispatchRecord.occupancyAnalysis = analysis;
                    
   
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
        
                }

                profilerData.dispatches.push(dispatchRecord);
                pass.__dispatches.push(dispatchRecord);
                encoderData.dispatches.push(dispatchRecord);
                device.__webSightInfo.dispatchCount++;

                dispatchRecord.bufferAccesses.forEach(b => {
                  if (b?.id) {
                    profilerData.bufferHeatMap[b.id] = (profilerData.bufferHeatMap[b.id] || 0) + 1;
                  }
                });

                broadcastData();
              };

              return pass;
            };

            encoder.beginRenderPass = function(passDesc) {
           
              let pass = null;
              try {
                let passTimingHelper = null;
                if (device.__webSightInfo.timingMode === 'gpu' && !profilerData.config.minimalOverhead) {
                  passTimingHelper = getTimingHelper(device);
                }

                if (passTimingHelper) {
                  try {
                    pass = passTimingHelper.beginRenderPass(proxyEncoder, passDesc);
                  } catch (e) {
                    console.warn('[WebSight] TimingHelper.beginRenderPass failed, continuing without timing:', e.message);
                    pass = origBeginRenderPass(passDesc);
                    passTimingHelper = null;
                  }
                } else {
                  pass = origBeginRenderPass(passDesc);
                }
                
                encoderData.passCount++;
                device.__webSightInfo.passCount++;

                pass.__dispatches = [];
                pass.__passId = crypto.randomUUID();

                if (passTimingHelper) {
                  passTimings.push({ helper: passTimingHelper, dispatches: pass.__dispatches });
                }

                pass.__boundBindGroups = {};
                pass.__passType = 'render';
                pass.__timingHelper = passTimingHelper;
                pass.__deviceLabel = device.__webSightInfo.label;
                pass.__passDescriptor = passDesc;
                pass.__bandwidthTracker = new BandwidthTracker();
               
                pass.__bandwidthSnapshot = { read: 0, written: 0 };
             
                pass.__fbBandwidthCounted = false;

                const origSetPipeline = pass.setPipeline.bind(pass);
                pass.setPipeline = function(p) {
                  this.__boundPipeline = p;
                  origSetPipeline(p);
                };

                const origSetBindGroup = pass.setBindGroup.bind(pass);
                pass.setBindGroup = function(i, bg) {
                  this.__boundBindGroups[i] = bg;
                  this.__bandwidthTracker.trackBindGroup(bg);
                  origSetBindGroup(i, bg);
                };

                function recordDrawCall(drawParams, origFn, origArgs) {
                  const cpuStart = performance.now();
                  const pipeline = pass.__boundPipeline?.__capture;

                  const drawRecord = {
                    index: profilerData.dispatches.length,
                    ...drawParams,
                    pipelineLabel: pipeline?.label || 'unknown',
                    cpuStart,
                    cpuEnd: 0,
                    cpuTimeMs: 0,
                    gpuTimeMs: 0,
                    timingSource: 'render_pass_timing',
                    deviceLabel: device.__webSightInfo.label,
                    passType: 'render',
                    passId: pass.__passId,
                    bufferAccesses: []
                  };

                  origFn(...origArgs);

                  drawRecord.cpuEnd = performance.now();
                  drawRecord.cpuTimeMs = drawRecord.cpuEnd - drawRecord.cpuStart;

                  const { bandwidth: drawBandwidth, newSnapshot: drawSnapshot, fbCountedNow } =
                    calculateRenderDrawBandwidth(
                      pass.__bandwidthTracker, pass.__bandwidthSnapshot,
                      pass.__passDescriptor, pass.__fbBandwidthCounted
                    );
                  pass.__bandwidthSnapshot = drawSnapshot;
                  if (fbCountedNow) pass.__fbBandwidthCounted = true;

                  drawRecord.bandwidth = drawBandwidth;

                  profilerData.dispatches.push(drawRecord);
                  pass.__dispatches.push(drawRecord);
                  encoderData.dispatches.push(drawRecord);
                  device.__webSightInfo.dispatchCount++;

                  broadcastData();
                }

                const origDraw = pass.draw.bind(pass);
                pass.draw = function(vertexCount, instanceCount, firstVertex, firstInstance) {
                  recordDrawCall(
                    { type: 'draw', vertexCount: vertexCount || 0, instanceCount: instanceCount || 1,
                      firstVertex: firstVertex || 0, firstInstance: firstInstance || 0 },
                    origDraw,
                    [vertexCount, instanceCount, firstVertex, firstInstance]
                  );
                };

                const origDrawIndexed = pass.drawIndexed.bind(pass);
                pass.drawIndexed = function(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
                  recordDrawCall(
                    { type: 'drawIndexed', indexCount: indexCount || 0, instanceCount: instanceCount || 1,
                      firstIndex: firstIndex || 0, baseVertex: baseVertex || 0, firstInstance: firstInstance || 0 },
                    origDrawIndexed,
                    [indexCount, instanceCount, firstIndex, baseVertex, firstInstance]
                  );
                };

                return pass;
              } catch (error) {
                console.error('[WebSight] Error in beginRenderPass hook:', error);
                if (pass) return pass;
                return origBeginRenderPass(passDesc);
              }
            };

            encoder.finish = function(descriptor) {
              const commandBuffer = origFinish(descriptor);
              commandBuffer.__dispatches = encoderData.dispatches;
              commandBuffer.__encoderId = encoderData.id;
              commandBuffer.__passTimings = passTimings;
              commandBuffer.__passCount = encoderData.passCount;

              commandBuffer.__gpuTiming = {
                available: false,
                passes: [],
                totalTimeNs: 0,

                ready: new Promise((resolve) => {
                  commandBuffer.__gpuTimingResolve = resolve;
                })
              };
              
              return commandBuffer;
            };

            return encoder;
          };


          const origSubmit = device.queue.submit.bind(device.queue);
          
          device.queue.submit = function(cmds) {
            let passEntries = [];
            
            for (const cmd of cmds) {
              if (cmd.__passTimings) {
                passEntries.push(...cmd.__passTimings);
              }
            }
            
            const result = origSubmit(cmds);
            
            if (passEntries.length > 0) {
              const cmdBuffersWithTiming = cmds.filter(cmd => cmd.__gpuTiming);
              
              device.queue.onSubmittedWorkDone().then(async () => {
                try {
                  const allDurations = [];
                  for (const entry of passEntries) {
                    if (!entry.helper) {
                      allDurations.push(0);
                      continue;
                    }
                    
                    try {
                      if (entry.helper.__valid === false) {
                        allDurations.push(0);
                        releaseTimingHelper(device, entry.helper);
                        continue;
                      }
                      const durations = await entry.helper.getResult();
                      const passDurationNs = Number(durations[0] || 0);
                      allDurations.push(passDurationNs);
                      
                      // Single-dispatch: assign GPU time directly.
                      // Multi-dispatch: store pass aggregate; don't fabricate per-dispatch times.
                      const dispatches = entry.dispatches || [];
                      if (passDurationNs > 0 && dispatches.length > 0) {
                        if (dispatches.length === 1) {
                          const dispatch = dispatches[0];
                          const gpuTimeMs = passDurationNs / 1e6;

                          dispatch.gpuTimeMs = gpuTimeMs;
                          dispatch.normalizedTime = gpuTimeMs;
                          dispatch.timingSource = 'gpu_timestamp';

                          // Recalculate bandwidth with GPU timing
                          if (dispatch.bandwidth && gpuTimeMs > 0) {
                            dispatch.bandwidth.bandwidthGBs = dispatch.bandwidth.totalBytes / (gpuTimeMs / 1e3) / 1e9;
                            const peakBW = profilerData.config.peakMemoryBandwidthGBs;
                            if (peakBW) {
                              dispatch.bandwidth.memoryEfficiency = (dispatch.bandwidth.bandwidthGBs / peakBW) * 100;
                            }
                          }

                          // Update kernel stats (times in ns — matches normalizeTime() and profiler-standalone.js)
                          const kernel = profilerData.kernels[dispatch.kernelId];
                          if (kernel) {
                            kernel.stats.totalTime += passDurationNs;
                            kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                            kernel.stats.minTime = Math.min(kernel.stats.minTime, passDurationNs);
                            kernel.stats.maxTime = Math.max(kernel.stats.maxTime, passDurationNs);

                            if (!kernel.bandwidth) {
                              kernel.bandwidth = {
                                totalBytesRead: 0, totalBytesWritten: 0, totalBytes: 0,
                                avgBandwidthGBs: 0, peakBandwidthGBs: 0
                              };
                            }
                            if (dispatch.bandwidth) {
                              kernel.bandwidth.totalBytesRead += dispatch.bandwidth.bytesRead;
                              kernel.bandwidth.totalBytesWritten += dispatch.bandwidth.bytesWritten;
                              kernel.bandwidth.totalBytes += dispatch.bandwidth.totalBytes;
                              kernel.bandwidth.avgBandwidthGBs = kernel.bandwidth.totalBytes / (kernel.stats.totalTime / 1e9) / 1e9;
                              kernel.bandwidth.peakBandwidthGBs = Math.max(
                                kernel.bandwidth.peakBandwidthGBs || 0,
                                dispatch.bandwidth.bandwidthGBs
                              );
                            }
                          }
                        } else {
                          for (const dispatch of dispatches) {
                            dispatch.passGpuTimeMs = passDurationNs / 1e6;
                            dispatch.passGpuTimeNs = passDurationNs;
                            dispatch.passDispatchCount = dispatches.length;
                            dispatch.timingSource = 'pass_aggregate';

                            // Evenly divide for totalTime/avgTime (approximate).
                            // Do NOT update minTime/maxTime with the estimate if real per-dispatch
                            // measurements already exist — estimates would corrupt real benchmarks.
                            // But if only aggregate timing has ever been recorded, leaving minTime
                            // at Infinity and maxTime at 0 is more misleading than the estimate.
                            const perDispatchEstimate = passDurationNs / dispatches.length;
                            const kernel = profilerData.kernels[dispatch.kernelId];
                            if (kernel) {
                              kernel.stats.totalTime += perDispatchEstimate;
                              kernel.stats.avgTime = kernel.stats.totalTime / kernel.stats.count;
                              // Seed min/max only when no real measurement has set them yet.
                              if (kernel.stats.minTime === Infinity) kernel.stats.minTime = perDispatchEstimate;
                              if (kernel.stats.maxTime === 0)        kernel.stats.maxTime = perDispatchEstimate;
                            }
                          }
                        }
                      }
                      
                      // Release helper back to pool after getResult() completes
                      releaseTimingHelper(device, entry.helper);
                    } catch (e) {
                      console.warn('[WebSight] Failed to get timing from one pass:', e.message);
                      allDurations.push(0);
                      releaseTimingHelper(device, entry.helper);
                    }
                  }
                  
                  const nonZeroCount = allDurations.filter(d => d > 0).length;
                  if (nonZeroCount > 0) {
                    console.log(`[WebSight] Got GPU timing for ${nonZeroCount}/${allDurations.length} passes:`, allDurations.map(d => `${(d/1000000).toFixed(3)}ms`));
                  }
                  
                  // Accumulate timings for direct access
                  window.__webSightGlobalTimingResults.push(...allDurations);
                  if (window.__webSightGlobalTimingResults.length > window.__webSightMaxTimingResults) {
                    // slice(-n) is O(1) allocation and copies only the kept tail;
                    // splice(0, excess) on a 10 000-element array shifts every remaining
                    // element and causes a GC pause on every submit.
                    window.__webSightGlobalTimingResults =
                      window.__webSightGlobalTimingResults.slice(-window.__webSightMaxTimingResults);
                    console.warn(`[WebSight] Timing results buffer trimmed to ${window.__webSightMaxTimingResults} entries.`);
                  }
                  
                  // Populate command buffer timing
                  if (cmdBuffersWithTiming.length > 0) {
                    const totalTimeNs = allDurations.reduce((sum, t) => sum + t, 0);
                    
                    cmdBuffersWithTiming.forEach(cmd => {
                      cmd.__gpuTiming.available = true;
                      cmd.__gpuTiming.passes = allDurations;
                      cmd.__gpuTiming.totalTimeNs = totalTimeNs;
                      cmd.__gpuTiming.totalTimeMs = totalTimeNs / 1000000;
                      
                      if (cmd.__gpuTimingResolve) {
                        cmd.__gpuTimingResolve(cmd.__gpuTiming);
                      }
                    });
                  }
                  
                  // Fire timing event
                  if (window.__webSightTimingEvents) {
                    window.__webSightTimingEvents.dispatchEvent(new CustomEvent('timing', {
                      detail: {
                        passes: allDurations,
                        totalTimeNs: allDurations.reduce((sum, t) => sum + t, 0),
                        commandBuffers: cmdBuffersWithTiming.length
                      }
                    }));
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
            
            // Event-based timing notifications
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

// Returns a structured snapshot of all tracked adapters, devices, and timing-helper pools.
// Owns the window globals written by hookWebGPU — the sole reader lives here.
function getMultiGPUStats() {
  const adapters = (window.__webSightAdapters || []).map((a, i) => ({
    index: i,
    hasTimestampFeature: a.hasTimestampFeature,
    powerPreference: a.options?.powerPreference || 'default',
    deviceCount: a.devices.length,
    requestedAt: a.requestedAt
  }));

  const devices = (window.__webSightDevices || []).map((d, i) => ({
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
  }));

  const pools = [];
  window.__webSightTimingHelperPools?.forEach((pool, device) => {
    const info = device.__webSightInfo;
    pools.push({
      deviceLabel: info?.label || 'unknown',
      poolSize: pool.helpers.length,
      maxSize: pool.maxSize,
      available: pool.available?.length || 0,
      inUse: pool.inUse?.size || 0,
      missedCount: pool.missedCount || 0,
      currentIndex: pool.index,
      failed: pool.failed,
      limitReached: pool.limitReached || false,
      mode: pool.limitReached ? 'limited (waiting)' : 'growing',
      utilizationPercent: pool.helpers.length > 0
        ? Math.round(((pool.inUse?.size || 0) / pool.helpers.length) * 100) : 0
    });
  });

  return {
    adapters,
    devices,
    pools,
    totals: {
      adapterCount:    adapters.length,
      deviceCount:     devices.length,
      totalEncoders:   devices.reduce((s, d) => s + d.encoderCount,  0),
      totalPasses:     devices.reduce((s, d) => s + d.passCount,     0),
      totalDispatches: devices.reduce((s, d) => s + d.dispatchCount, 0)
    }
  };
}

export { hookWebGPU, getMultiGPUStats };