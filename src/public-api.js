

import { profilerData } from './data-store.js';
import { memoryLeakDetector, workgroupAnalyzer, shaderAnalyzer } from './init-logging.js';
import { normalizeTime, getTimeUnitLabel, addLog } from './utils.js';
import { hookWebGPU, getMultiGPUStats } from './webgpu-hooks.js';

function _dedupDispatches(onlyWithBandwidth = false) {
  const dispatches = profilerData.dispatches || [];
  const seenPassIds = new Set();
  return dispatches
    .filter(d => {
      if (onlyWithBandwidth && !d.bandwidth) return false;
      if (d.timingSource === 'gpu_timestamp') return true;
      if (d.timingSource === 'pass_aggregate') {
        if (!d.passId || !(d.passGpuTimeMs > 0 || d.passGpuTimeNs > 0)) return false;
        if (seenPassIds.has(d.passId)) return false;
        seenPassIds.add(d.passId);
        return true;
      }
      return !onlyWithBandwidth; 
    })
    .map(d => {
      if (d.timingSource !== 'pass_aggregate') return d;
      const gpuTimeMs = d.passGpuTimeMs || (d.passGpuTimeNs ? d.passGpuTimeNs / 1e6 : 0);
      return {
        ...d,
        gpuTimeMs,
        bandwidth: d.bandwidth ? {
          ...d.bandwidth,
          bandwidthGBs: _bwGBs(d.bandwidth.totalBytes, gpuTimeMs)
        } : d.bandwidth
      };
    });
}

function _bwGBs(bytes, timeMs) {
  return timeMs > 0 ? bytes / (timeMs / 1e3) / 1e9 : 0;
}

function _computeBandwidthStats(dispatches) {
  const totalBytesRead    = dispatches.reduce((s, d) => s + (d.bandwidth?.bytesRead    || 0), 0);
  const totalBytesWritten = dispatches.reduce((s, d) => s + (d.bandwidth?.bytesWritten || 0), 0);
  const totalBytes        = totalBytesRead + totalBytesWritten;
  const totalTimeMs       = dispatches.reduce((s, d) => s + (d.gpuTimeMs || 0), 0);
  const avgBandwidthGBs   = _bwGBs(totalBytes, totalTimeMs);
  // Only include dispatches with at least 10µs of GPU time AND at least 1MB
  // transferred for peak — excludes tiny utility passes (e.g. 1-element
  // reductions) that bind large buffers but move almost no data, which
  // would otherwise produce unrealistically high peak figures.
  const timedForPeak = dispatches.filter(d =>
    (d.gpuTimeMs || 0) >= 0.01 &&
    (d.bandwidth?.totalBytes || 0) >= 1 * 1024 * 1024
  );
  const peakBandwidthGBs = timedForPeak.length > 0
    ? Math.max(...timedForPeak.map(d => {
        const ms = d.gpuTimeMs || 0;
        const bytes = (d.bandwidth?.totalBytes || 0);
        // Recalculate from raw values — don't trust the stored field
        // which may have been computed with binary (1024^3) vs decimal (1e9) GB units.
        return _bwGBs(bytes, ms);
      }))
    : 0;
  return { totalBytesRead, totalBytesWritten, totalBytes, totalTimeMs, avgBandwidthGBs, peakBandwidthGBs };
}

function _computeGpuTimingStats(validGpuTimes) {
  if (validGpuTimes.length === 0) return { avg: 0, total: 0, min: 0, max: 0 };
  const sum = validGpuTimes.reduce((a, b) => a + b, 0);
  return {
    avg:   sum / validGpuTimes.length,
    total: sum,
    min:   Math.min(...validGpuTimes),
    max:   Math.max(...validGpuTimes)
  };
}

function _computeStorageTotals(allDispatches) {

  const totalCpuTimeMs       = allDispatches.reduce((s, d) => s + (d.cpuTimeMs || 0), 0);

  const totalKernelTimeMs    = Object.values(profilerData.kernels).reduce((s, k) => s + (k.stats?.totalTime || 0), 0) / 1e6;

  const totalBufferSizeBytes = Object.values(profilerData.buffers).reduce((s, b) => s + (b.size || 0), 0);
  const cpuFallbackCount     = allDispatches.filter(d => d.timingSource === 'cpu_timing').length;
  return { totalCpuTimeMs, totalKernelTimeMs, totalBufferSizeBytes, cpuFallbackCount };
}

function _sumGroup(dispatches, timeKey) {
  return {
    totalBytes: dispatches.reduce((s, d) => s + (d.bandwidth?.totalBytes || 0), 0),
    totalMs:    dispatches.reduce((s, d) => s + (d[timeKey] || 0), 0)
  };
}

function _seriesArrays(points) {
  return {
    inputSizes: points.map(d => d.inputSizeMB),
    bandwidth:  points.map(d => d.bandwidthGBs),
    times:      points.map(d => d.timeMs)
  };
}

function _classifyBuffer(ll) {
  if (ll.includes('atomic'))  return 'atomic';
  if (ll.includes('uniform')) return 'uniform';
  if (ll.includes('output') || ll.includes('result') || ll.includes('dst')) return 'output';
  return 'input';
}

  if (typeof window !== 'undefined') {
    window.WebSight = {
      getData: () => profilerData,
      
      getMultiGPUStats: () => getMultiGPUStats(),
      
      clear: () => { 
        profilerData.dispatches = []; 
        profilerData.logs = [];
        profilerData.kernels = {};
        profilerData.buffers = {};
        profilerData.bufferHeatMap = {};
        profilerData.activeEncoders = new Map();
        profilerData.pipelines = {};
        addLog('Profiler cleared');
      },
      
      start: hookWebGPU,
      

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
          const deviceMode = options.minimalOverhead ? 'cpu-only' : 'gpu';

          (window.__webSightDevices || []).forEach(d => { d.timingMode = deviceMode; });
          if (options.minimalOverhead) {
            profilerData.timingMode = 'cpu';
            profilerData.config.broadcastDebounceMs = 10000;
            console.log('[WebSight] MINIMAL OVERHEAD MODE: GPU timing disabled, broadcast interval 10s');
          } else {
            profilerData.timingMode = 'gpu';
            profilerData.config.broadcastDebounceMs = 1000;
            console.log('[WebSight] Normal mode: GPU timing enabled, broadcast interval 1s');
          }
        }
        
        if (options.peakMemoryBandwidthGBs !== undefined) {
          profilerData.config.peakMemoryBandwidthGBs = options.peakMemoryBandwidthGBs;
          console.log(`[WebSight] Peak memory bandwidth: ${options.peakMemoryBandwidthGBs} GB/s`);
        }
        
        if (options.simdWidth !== undefined) {
          profilerData.config.simdWidth = options.simdWidth;

          if (workgroupAnalyzer.optimal) {
            workgroupAnalyzer.optimal.simdWidth = options.simdWidth;
          }
          console.log(`[WebSight] SIMD width: ${options.simdWidth}`);
        }

        if (options.leakThresholdMs !== undefined) {
          memoryLeakDetector.leakThreshold = options.leakThresholdMs;
          console.log(`[WebSight] Leak threshold: ${options.leakThresholdMs} ms`);
        }
        

        
        return {
          broadcastEnabled: profilerData.config.broadcastEnabled,
          broadcastDebounceMs: profilerData.config.broadcastDebounceMs,
          timeUnit: profilerData.config.normalizeTimeUnit,
          minimalOverhead: profilerData.config.minimalOverhead,
          peakMemoryBandwidthGBs: profilerData.config.peakMemoryBandwidthGBs,
          simdWidth: profilerData.config.simdWidth,
          leakThresholdMs: memoryLeakDetector.leakThreshold
        };
      },
      
      benchmarkMode: () => {
        profilerData.config.minimalOverhead = true;
        profilerData.timingMode = 'cpu';
        profilerData.config.broadcastEnabled = false;

        (window.__webSightDevices || []).forEach(d => { d.timingMode = 'cpu-only'; });
        console.log('[WebSight] BENCHMARK MODE: Minimal overhead, no broadcasts, CPU timing only');
      },
      
      normalMode: () => {
        profilerData.config.minimalOverhead = false;
        profilerData.timingMode = 'gpu';
        profilerData.config.broadcastEnabled = true;
        profilerData.config.broadcastDebounceMs = 1000;

        (window.__webSightDevices || []).forEach(d => { d.timingMode = 'gpu'; });
        console.log('[WebSight] NORMAL MODE: GPU timing enabled, broadcasts active');
      },
      
      getStats: () => {
        const allDispatches = profilerData.dispatches || [];

        const timedDispatches = _dedupDispatches(false);
        const validGpuTimes   = timedDispatches.filter(d => d.gpuTimeMs > 0).map(d => d.gpuTimeMs);

        const bwDispatches    = timedDispatches.filter(d => d.bandwidth);
        const { totalBytesRead, totalBytesWritten, totalBytes, avgBandwidthGBs, peakBandwidthGBs }
          = _computeBandwidthStats(bwDispatches);

        const unit = getTimeUnitLabel();
        const gpuTiming = _computeGpuTimingStats(validGpuTimes);
        const { totalCpuTimeMs, totalKernelTimeMs, totalBufferSizeBytes, cpuFallbackCount } = _computeStorageTotals(allDispatches);

        return {
          totalDispatches: allDispatches.length,
          gpuTimedDispatches: validGpuTimes.length,
          cpuFallbackDispatches: cpuFallbackCount,
          avgGpuTime:   gpuTiming.avg,
          totalGpuTime: gpuTiming.total,
          minGpuTime:   gpuTiming.min,
          maxGpuTime:   gpuTiming.max,
          timeUnit: 'ms',
          totalCpuTimeMs,

          totalKernelTimeMs,
          totalBufferSizeBytes,

          bandwidth: {
              totalBytesRead,
              totalBytesWritten,
              totalBytes,
              avgBandwidthGBs,
              peakBandwidthGBs,
              totalBytesReadFormatted:    memoryLeakDetector.formatBytes(totalBytesRead),
              totalBytesWrittenFormatted: memoryLeakDetector.formatBytes(totalBytesWritten),
              totalBytesFormatted:        memoryLeakDetector.formatBytes(totalBytes)
          }
        };
      },
      
      getTimingHelperStats: () => {
        return { message: 'Each encoder now has its own TimingHelper. Stats are per-encoder.' };
      },

      getGraphData: () => {
        const allDispatches = profilerData.dispatches.filter(d => d.gpuTimeMs > 0 || d.cpuStart);

        const byKernel = new Map();

        allDispatches.forEach(d => {
          const kernelLabel = d.pipelineLabel || '(unlabeled)';
          if (!byKernel.has(kernelLabel)) byKernel.set(kernelLabel, new Map());
          const sizeMap = byKernel.get(kernelLabel);

          const buffers = d.bufferAccesses || [];
          const dataBuffers = buffers.filter(b => {
            const label = (b.label || '').toLowerCase();
            const size  = b.size || 0;
            return size > 1024 * 1024 ||
              label.includes('in') || label.includes('out') || label.includes('key') ||
              label.includes('payload') || label.includes('data') || label.includes('buffer');
          });
          const largestBuffer = dataBuffers.length > 0
            ? Math.max(...dataBuffers.map(b => b.size || 0))
            : (buffers.length > 0 ? Math.max(...buffers.map(b => b.size || 0))
               : (d.bandwidth?.totalBytes || 0));

          if (!sizeMap.has(largestBuffer)) sizeMap.set(largestBuffer, { gpuItems: [], cpuItems: [] });
          const bucket = sizeMap.get(largestBuffer);
            if (d.timingSource === 'gpu_timestamp' && d.bandwidth) bucket.gpuItems.push(d);
          else if (d.timingSource === 'cpu_timing'    && d.bandwidth) bucket.cpuItems.push(d);
        });

        const kernels = [];
        let anyGpuTiming = false;

        byKernel.forEach((sizeMap, label) => {
          const gpuPoints = [], cpuPoints = [];

          const sortedSizes = [...sizeMap.keys()].sort((a, b) => a - b);
          sortedSizes.forEach(largestBuffer => {
            const { gpuItems, cpuItems } = sizeMap.get(largestBuffer);
            const inputSizeMB = largestBuffer / (1024 * 1024);
            const inputBytes  = largestBuffer;

            if (gpuItems.length > 0) {
              const s = _sumGroup(gpuItems, 'gpuTimeMs');
              gpuPoints.push({ inputSizeMB, inputBytes, timeMs: s.totalMs, bandwidthGBs: _bwGBs(s.totalBytes, s.totalMs) });
            }
            if (cpuItems.length > 0) {
              const s = _sumGroup(cpuItems, 'cpuTimeMs');
              cpuPoints.push({ inputSizeMB, inputBytes, timeMs: s.totalMs, bandwidthGBs: _bwGBs(s.totalBytes, s.totalMs) });
            }
          });

          if (gpuPoints.length > 0) anyGpuTiming = true;
          kernels.push({ label, gpu: _seriesArrays(gpuPoints), cpu: _seriesArrays(cpuPoints) });
        });

        const allGpuPoints = kernels.flatMap(k => k.gpu.inputSizes.map((x, i) => ({
          inputSizeMB: x, bandwidthGBs: k.gpu.bandwidth[i], timeMs: k.gpu.times[i]
        }))).sort((a, b) => a.inputSizeMB - b.inputSizeMB);
        const allCpuPoints = kernels.flatMap(k => k.cpu.inputSizes.map((x, i) => ({
          inputSizeMB: x, bandwidthGBs: k.cpu.bandwidth[i], timeMs: k.cpu.times[i]
        }))).sort((a, b) => a.inputSizeMB - b.inputSizeMB);

        return {
          kernels,
          gpu: _seriesArrays(allGpuPoints),
          cpu: _seriesArrays(allCpuPoints),
          hasGpuTiming: anyGpuTiming
        };
      },

      getBufferData: () => {
        const maxBufferSize = profilerData.gpuCharacteristics?.limits?.maxBufferSize || (2 * 1024 * 1024 * 1024);
        const warningThreshold = maxBufferSize * 0.9;
        const labels = [], input = [], output = [], atomic = [], uniform = [];
        const warnings = [];

        Object.values(profilerData.buffers).forEach((b, i) => {
          const label = b.label || `Buffer ${i}`;
          labels.push(label);
          const sizeKB = b.size / 1024;
          const ll = label.toLowerCase();

          if (b.size >= warningThreshold) {
            warnings.push({
              label,
              sizeFormatted:  memoryLeakDetector.formatBytes(b.size),
              percentOfMax:   ((b.size / maxBufferSize) * 100).toFixed(1)
            });
          }

          const cat = _classifyBuffer(ll);
          input.push(cat === 'input'   ? sizeKB : 0);
          output.push(cat === 'output'  ? sizeKB : 0);
          atomic.push(cat === 'atomic'  ? sizeKB : 0);
          uniform.push(cat === 'uniform' ? sizeKB : 0);
        });

        return { labels, series: { input, output, atomic, uniform }, warnings, maxBufferSizeFormatted: memoryLeakDetector.formatBytes(maxBufferSize) };
      },

      getAtomicContentionData: () => {
        const x = [], y = [];
        profilerData.dispatches.forEach((d, i) => {
          const atomicBuffers = (d.bufferAccesses || []).filter(b => {
            const ll = (b.label || '').toLowerCase();
            return (ll.includes('atomic') || ll.includes('hist') || ll.includes('histogram') || ll.includes('counter'))
              && (b.size || 0) < 10 * 1024 * 1024;
          });
          if (atomicBuffers.length === 0) return;

          const wgAnalysis = d.occupancyAnalysis;
          if (!wgAnalysis) return;
          const totalThreads = wgAnalysis.totalInvocations;
          const bins = (atomicBuffers[0].size || 0) / 4;
          const threadsPerBin = bins > 0 ? totalThreads / bins : 0;
          if (threadsPerBin > 0) { x.push(i); y.push(threadsPerBin); }
        });
        return { x, y };
      },

      getKernelGraphData: () => {
        return Object.values(profilerData.kernels)
          .map((kernel, idx) => {
            const dispatches = profilerData.dispatches
              .filter(d => d.kernelId === kernel.id && d.gpuTimeMs > 0)
              .map((d, i) => ({ index: i, timeMs: d.gpuTimeMs }));
            if (dispatches.length === 0) return null;
            return {
              id: kernel.id,
              label: (kernel.label && kernel.label.trim()) ? kernel.label : `Kernel ${idx + 1}`,
              count: kernel.stats?.count || 0,
              avgTimeMs: kernel.stats?.avgTime ?? 0,
              dispatches
            };
          })
          .filter(Boolean);
      },

      getWorkgroupSummaryData: () => {
        const configGroups = new Map();
        profilerData.dispatches.forEach(d => {
          const wgSize = d.workgroupSize || [1, 1, 1];
          const key = `${d.pipelineLabel}|${wgSize.join('x')}`;
          const totalWGs = (d.dispatchSize || [1, 1, 1]).reduce((a, b) => a * b, 1);
          if (!configGroups.has(key)) {
            configGroups.set(key, {
              pipeline: d.pipelineLabel || 'compute_pipeline',
              workgroupSize: wgSize,
              dispatchSize: d.dispatchSize || [1, 1, 1],
              minWGs: totalWGs, maxWGs: totalWGs,
              analysis: d.occupancyAnalysis || { score: -1, issues: [] },
              count: 0, dimensionViolation: false
            });
          }
          const g = configGroups.get(key);
          g.count++;

          if (d.occupancyAnalysis && d.occupancyAnalysis.score < g.analysis.score) {
            g.analysis = d.occupancyAnalysis;
            g.dispatchSize = d.dispatchSize;
          }
          if (totalWGs < g.minWGs) g.minWGs = totalWGs;
          if (totalWGs > g.maxWGs) g.maxWGs = totalWGs;
          if (d.dimensionViolation) g.dimensionViolation = true;
        });

        const groups = Array.from(configGroups.values()).sort((a, b) => a.analysis.score - b.analysis.score);
        return {
          groups,
          totalConfigs: groups.length,
          goodConfigs: groups.filter(g => g.analysis.score >= 80 && !g.dimensionViolation).length,
          failedConfigs: groups.filter(g => g.dimensionViolation).length
        };
      },

      getDispatchList: () => {
        return profilerData.dispatches
          .filter(d => (d.cpuTimeMs || 0) >= 0.001)
          .slice(-20)
          .reverse()
          .map(d => {
            const timeDisplay = d.gpuTimeMs && d.timingSource === 'gpu_timestamp'
              ? `${d.gpuTimeMs.toFixed(3)} ms`
              : d.cpuTimeMs
              ? `${d.cpuTimeMs.toFixed(3)} ms (CPU fallback)`
              : 'pending';

            const pipelineLabel = d.pipelineLabel && d.pipelineLabel !== 'unknown'
              ? d.pipelineLabel
              : 'Unlabeled Pipeline';

            const dispatchType = d.type || d.passType || 'unknown';
            const title = (dispatchType === 'draw' || dispatchType === 'drawIndexed')
              ? `Draw Call #${d.index}`
              : dispatchType === 'compute'
              ? `Compute #${d.index}`
              : `Dispatch #${d.index}`;

            const workgroupInfo = dispatchType === 'compute'
              ? `Dispatch: ${d.dispatchSize ? d.dispatchSize.join('×') : `${d.x}×${d.y}×${d.z}`} | Workgroup: ${d.workgroupSize ? d.workgroupSize.join('×') : '?'}`
              : (dispatchType === 'draw' || dispatchType === 'drawIndexed')
              ? `Vertices: ${d.vertexCount ?? d.indexCount ?? 0}, Instances: ${d.instanceCount ?? 1}`
              : `Type: ${dispatchType}`;

            return { title, pipelineLabel, workgroupInfo, timeDisplay };
          });
      },

      formatBytes: (bytes) => memoryLeakDetector.formatBytes(bytes),

      formatTime: (ms) => memoryLeakDetector.formatTime(ms),

      getBandwidthAnalysis: () => {
        console.log('\n[WebSight] Bandwidth Analysis Report');
        console.log('═'.repeat(80));

        const dispatches = _dedupDispatches(true);

        if (dispatches.length === 0) {
            console.log('\nWARNING: No GPU-timed dispatches with bandwidth data available.');
            console.log('   Run your WebGPU application first.');
            return null;
        }

        const { totalBytesRead, totalBytesWritten, totalBytes, totalTimeMs, avgBandwidthGBs, peakBandwidthGBs }
          = _computeBandwidthStats(dispatches);

        const kernelBandwidth = {};
        dispatches.forEach(d => {
            if (!kernelBandwidth[d.kernelId]) {
                kernelBandwidth[d.kernelId] = {
                    label: d.pipelineLabel,
                    count: 0,
                    totalBytes: 0,
                    totalTimeMs: 0,
                    peakBandwidthGBs: 0,
                    memoryPatterns: []
                };
            }
            const kb = kernelBandwidth[d.kernelId];
            kb.count++;
            kb.totalBytes += d.bandwidth.totalBytes;
            kb.totalTimeMs += d.gpuTimeMs;
            kb.peakBandwidthGBs = Math.max(kb.peakBandwidthGBs, d.bandwidth.bandwidthGBs);
            
            if (d.memoryPattern && !kb.memoryPatterns.includes(d.memoryPattern.accessPatternType)) {
                kb.memoryPatterns.push(d.memoryPattern.accessPatternType);
            }
        });

        Object.values(kernelBandwidth).forEach(kb => {
            kb.avgBandwidthGBs = _bwGBs(kb.totalBytes, kb.totalTimeMs);
        });

        const sortedKernels = Object.values(kernelBandwidth)
            .sort((a, b) => b.totalBytes - a.totalBytes);

        console.log('\nOVERALL BANDWIDTH');
        console.log('-'.repeat(80));
        console.log(`Analyzed Dispatches: ${dispatches.length}`);
        console.log(`Total Data Transferred: ${(totalBytes / 1e9).toFixed(3)} GB  (bound-buffer-size estimate)`);
        console.log(`  Read:    ${(totalBytesRead / 1e9).toFixed(3)} GB (${totalBytes > 0 ? (totalBytesRead    / totalBytes * 100).toFixed(1) : '0.0'}%)`);
        console.log(`  Written: ${(totalBytesWritten / 1e9).toFixed(3)} GB (${totalBytes > 0 ? (totalBytesWritten / totalBytes * 100).toFixed(1) : '0.0'}%)`);  
        console.log(`Total GPU Time: ${totalTimeMs.toFixed(2)} ms`);
        console.log(`Average Bandwidth: ${avgBandwidthGBs.toFixed(2)} GB/s  (estimated from bound buffer sizes)`);
        console.log(`Peak Bandwidth: ${peakBandwidthGBs.toFixed(2)} GB/s  (estimated from bound buffer sizes)`);

        const peakBW = profilerData.config.peakMemoryBandwidthGBs;
        let memoryEfficiency = null;
        if (peakBW) {
          memoryEfficiency = (avgBandwidthGBs / peakBW * 100);
          console.log(`\nMemory Efficiency: ${memoryEfficiency.toFixed(1)}% of ${peakBW} GB/s peak`);

          if (memoryEfficiency < 20) {
              console.log('   VERY LOW (<20%) - compute-bound or significant overhead');
          } else if (memoryEfficiency < 50) {
              console.log('   LOW (20-50%) - poor memory access patterns');
          } else if (memoryEfficiency > 70) {
              console.log('   HIGH (>70%) - memory-bound workload');
          }
        } else {
          console.log('\nTip: Set peak bandwidth for efficiency %:');
          console.log('   WebSight.configure({ peakMemoryBandwidthGBs: 400 })');
        }

        console.log('\nTOP BANDWIDTH CONSUMERS');
        console.log('-'.repeat(80));
        console.log('Kernel'.padEnd(35) + 'Calls'.padEnd(8) + 'Total GB'.padEnd(12) + 'Avg GB/s'.padEnd(12) + 'Hints');
        console.log('-'.repeat(80));

        sortedKernels.slice(0, 10).forEach(kb => {
            console.log(
                kb.label.padEnd(35).substring(0, 35) +
                kb.count.toString().padEnd(8) +
                (kb.totalBytes / 1e9).toFixed(3).padEnd(12) +
                kb.avgBandwidthGBs.toFixed(2).padEnd(12) +
                (kb.memoryPatterns.filter(p => p !== 'unknown').join(', ') || '-')
            );
        });

        const memBoundThreshold = peakBW ? peakBW * 0.5 : 200;
        const compBoundThreshold = peakBW ? peakBW * 0.2 : 100;
        console.log(`\nMEMORY-BOUND KERNELS (>${memBoundThreshold.toFixed(0)} GB/s${peakBW ? ' = 50% of peak' : ''})`);
        console.log('-'.repeat(80));
        const memoryBound = sortedKernels.filter(kb => kb.avgBandwidthGBs > memBoundThreshold);
        if (memoryBound.length > 0) {
            memoryBound.forEach(kb => {
                console.log(`  ${kb.label}: ${kb.avgBandwidthGBs.toFixed(2)} GB/s`);
            });
            console.log('\nOptimize by: using shared memory, coalescing accesses, reducing redundant reads');
        } else {
            console.log('   None - workload is compute-bound or low utilization.');
        }

        console.log(`\nCOMPUTE-BOUND KERNELS (<${compBoundThreshold.toFixed(0)} GB/s${peakBW ? ' = 20% of peak' : ''})`);
        console.log('-'.repeat(80));
        const computeBound = sortedKernels.filter(kb => kb.avgBandwidthGBs < compBoundThreshold);
        if (computeBound.length > 0) {
            computeBound.forEach(kb => {
                console.log(`  ${kb.label}: ${kb.avgBandwidthGBs.toFixed(2)} GB/s`);
            });
            console.log('\nOptimize by: reducing math ops, using approximations, increasing workgroup size');
        } else {
            console.log('   None - all kernels are memory-bound.');
        }

        return {
            overall: {
                totalBytesRead,
                totalBytesWritten,
                totalBytes,
                totalTimeMs,
                avgBandwidthGBs,
                peakBandwidthGBs,
                memoryEfficiency,
                peakMemoryBandwidthGBs: peakBW
            },
            kernels: sortedKernels,
            memoryBound,
            computeBound
        };
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
      

      

      getMemoryLeaks: () => {

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
      

      getWorkgroupAnalysis: () => {

        console.log('\n [WebSight] Analyzing workgroup configurations...');
        
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
      

      getShaderAnalysis: () => {

        console.log('\n [WebSight] Analyzing shader complexity...');
        
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
            console.log(`\n${summary.needsOptimization.length} Shader(s) Need Optimization:`);
            summary.needsOptimization.forEach(a => {
              console.log(`\n  Shader ID: ${a.shaderId}`);
              console.log(`  Score: ${a.score}/100 (Grade: ${a.grade.letter})`);
              console.log(`  Complexity: ${a.complexity.toFixed(1)}`);
              console.log(`  Lines of Code: ${a.lineCount}`);
              console.log('  Metrics:');
              console.log(`  - Branches: ${a.metrics.branches}`);
              console.log(`   - Loops: ${a.metrics.loops}`);
              console.log(`   - Math Ops: ${a.metrics.mathOps}`);
              console.log(`   - Memory Ops: ${a.metrics.memoryOps}`);
              console.log(`   - Atomic Ops: ${a.metrics.atomicOps}`);
              console.log(`    - Variable Count: ${a.metrics.variableCount || 0}`);
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
      

      getFullAnalysisReport: () => {
        console.log('\n[WebSight] COMPREHENSIVE ANALYSIS REPORT');
        console.log('═'.repeat(80));
        
        const memoryReport = memoryLeakDetector.getLeakReport();
        const workgroupReport = workgroupAnalyzer.getSummary();
        const shaderReport = shaderAnalyzer.getSummary();
        
        console.log('\nSUMMARY');
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
        
        console.log('\nCall specific methods for detailed reports:');
        console.log('  - WebSight.getMemoryLeaks()');
        console.log('  - WebSight.getWorkgroupAnalysis()');
        console.log('  - WebSight.getShaderAnalysis()');
        console.log('  - WebSight.getBandwidthAnalysis()');
        
        return {
          memory: memoryReport,
          workgroup: workgroupReport,
          shader: shaderReport,
          timestamp: Date.now()
        };
      }
    };
    

    window.addEventListener('load', () => {
      if (profilerData.gpuCharacteristics?.limits) {
        workgroupAnalyzer.setDeviceLimits(profilerData.gpuCharacteristics.limits);
      }
      

      if (!window.__webSightDisableAutoUI && !window.__webSightIsUIWindow) {
        const scripts = document.querySelectorAll('script[src*="profiler-standalone.js"], script[src*="public-api.js"]');
        let uiPath = 'index.html';
        if (scripts.length > 0) {
          const scriptSrc = scripts[0].src;
          let scriptDir = scriptSrc.substring(0, scriptSrc.lastIndexOf('/') + 1);
          // public-api.js lives in src/ — go up one level to find index.html
          if (scriptSrc.includes('public-api.js')) {
            scriptDir = scriptDir.replace(/src\/$/, '');
          }
          uiPath = scriptDir + 'index.html';
        }
        
        const profilerWindow = window.open(uiPath, 'WebSightProfiler', 'width=1400,height=900');
        if (!profilerWindow) {

          console.warn(`[WebSight] Could not open profiler UI (popup blocked?). Manually open: ${uiPath}`);
        }
      }
    });
    
    if (!window.__webSightIsUIWindow) {
      hookWebGPU();
      addLog('WebSight initialized');
    }
  }