export class WorkgroupOccupancyAnalyzer {
  constructor() {
    this.deviceLimits = null;
    this.analyses = new Map(); 
  }

  setDeviceLimits(limits) {
    this.deviceLimits = limits;

    this.optimal = {
      simdWidth: (limits && limits.simdWidth) || 32,
      maxThreadsPerWorkgroup: (limits && limits.maxComputeInvocationsPerWorkgroup) || 256
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


    const wg = this._normalizeDims(wgSize);
    const ds = this._normalizeDims(dispatchSize);

    const analysis = {
      kernelId: dispatch.kernelId,
      workgroupSize: wg,
      dispatchSize: ds,
      totalThreads: wg[0] * wg[1] * wg[2],
      totalWorkgroups: ds[0] * ds[1] * ds[2],
      totalInvocations: 0,
      issues: [],
      score: 100, 
      recommendations: []
    };

    analysis.totalInvocations = analysis.totalThreads * analysis.totalWorkgroups;


    const maxDim = this.deviceLimits.maxComputeWorkgroupsPerDimension || 65535;
    const totalWorkgroupCount = ds[0] * ds[1] * ds[2];

    let hasCriticalDimensionError = false;

    if (ds[0] > maxDim) {
      hasCriticalDimensionError = true;
      analysis.issues.push({
        severity: 'critical',
        type: 'exceeds-x-limit',
        message: `X dimension (${ds[0]}) exceeds limit (${maxDim})`,
        impact: 'GPU will REJECT this dispatch',
        recommendation: `Split X into Y/Z dimensions. Check getDispatchGeometry() logic.`
      });
    }

    if (ds[1] > maxDim) {
      hasCriticalDimensionError = true;
      analysis.issues.push({
        severity: 'critical',
        type: 'exceeds-y-limit',
        message: `Y dimension (${ds[1]}) exceeds limit (${maxDim})`,
        impact: 'GPU will REJECT this dispatch',
        recommendation: `BUG DETECTED: Y overflow suggests nested if-inside-while in getSimpleDispatchGeometry(). Use separate sequential while loops: first reduce X→Y, then reduce Y→Z.`
      });
    }

    if (ds[2] > maxDim) {
      hasCriticalDimensionError = true;
      analysis.issues.push({
        severity: 'critical',
        type: 'exceeds-z-limit',
        message: `Z dimension (${ds[2]}) exceeds limit (${maxDim})`,
        impact: 'GPU will REJECT this dispatch',
        recommendation: `Total workgroups (${totalWorkgroupCount}) too large. Reduce workgroup count or use tiling.`
      });
    }

 
    if (hasCriticalDimensionError) {
      analysis.score = 0;
    }


    const isSmallUtilityDispatch = !hasCriticalDimensionError
      && totalWorkgroupCount < 256
      && analysis.totalThreads >= 8
      && analysis.totalThreads < 64;


    const maxWorkgroupsPerDim = this.deviceLimits.maxComputeWorkgroupsPerDimension || 65535;
    const xUtilization = ((ds[0] / maxWorkgroupsPerDim) * 100).toFixed(1);
    const yUtilization = ((ds[1] / maxWorkgroupsPerDim) * 100).toFixed(1);
    const zUtilization = ((ds[2] / maxWorkgroupsPerDim) * 100).toFixed(1);

    analysis.workgroupUtilization = {
      total: totalWorkgroupCount,
      xPercent: parseFloat(xUtilization),
      yPercent: parseFloat(yUtilization),
      zPercent: parseFloat(zUtilization),
      xDim: ds[0],
      yDim: ds[1],
      zDim: ds[2],
      maxPerDim: maxWorkgroupsPerDim,
      xExceeds: ds[0] > maxWorkgroupsPerDim,
      yExceeds: ds[1] > maxWorkgroupsPerDim,
      zExceeds: ds[2] > maxWorkgroupsPerDim
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

    if (!isSmallUtilityDispatch && !hasCriticalDimensionError && analysis.totalThreads % this.optimal.simdWidth !== 0) {
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

    if (!isSmallUtilityDispatch && !hasCriticalDimensionError && analysis.totalThreads < 64) {
      analysis.issues.push({
        severity: 'high',
        type: 'low-occupancy',
        message: `Workgroup size (${analysis.totalThreads}) is very small`,
        impact: 'GPU underutilized - many execution units idle',
        recommendation: `Increase to at least 64 threads, prefer 128-256`
      });
      analysis.score -= 30;
    }

    if (!hasCriticalDimensionError && analysis.totalThreads > this.optimal.maxThreadsPerWorkgroup) {
      analysis.issues.push({
        severity: 'critical',
        type: 'exceeds-limit',
        message: `Workgroup size (${analysis.totalThreads}) exceeds device limit (${this.optimal.maxThreadsPerWorkgroup})`,
        impact: 'Kernel may fail on some devices',
        recommendation: `Reduce to ${this.optimal.maxThreadsPerWorkgroup} or less`
      });
      analysis.score -= 50;
    }


    if (ds[1] === 1 && ds[2] === 1 && (wg[1] > 1 || wg[2] > 1)) {
      analysis.issues.push({
        severity: 'low',
        type: 'dimension-mismatch',
        message: 'Dispatch is 1D but workgroup uses multiple dimensions',
        impact: 'Minor cache inefficiency',
        recommendation: `Consider using [${analysis.totalThreads}, 1, 1] instead of [${wg[0]}, ${wg[1]}, ${wg[2]}] for 1D problems`
      });
      analysis.score -= 5;
    }

    const isPowerOf2 = (n) => n > 0 && (n & (n - 1)) === 0;
    if (!hasCriticalDimensionError && analysis.totalThreads <= this.optimal.maxThreadsPerWorkgroup
        && (!isPowerOf2(wg[0]) || (wg[1] > 1 && !isPowerOf2(wg[1])) || (wg[2] > 1 && !isPowerOf2(wg[2])))) {
      analysis.issues.push({
        severity: 'low',
        type: 'non-power-of-2',
        message: 'Workgroup dimensions are not powers of 2',
        impact: 'May reduce performance on some GPUs (especially AMD)',
        recommendation: `Use power-of-2 sizes like [${this.nearestPowerOf2(wg[0])}, ${this.nearestPowerOf2(wg[1])}, ${this.nearestPowerOf2(wg[2])}]`
      });
      analysis.score -= 10;
    }

    analysis.score = Math.max(0, Math.min(100, analysis.score));

    if (analysis.score > 80) {
      analysis.recommendations.push({
        type: 'good',
        message: 'Workgroup configuration looks good!'
      });
    }

    // Keep worst-scoring analysis per kernel
    const existing = this.analyses.get(dispatch.kernelId);
    if (!existing || analysis.score < existing.score) {
      this.analyses.set(dispatch.kernelId, analysis);
    }
    return analysis;
  }

  _normalizeDims(dims) {
    if (Array.isArray(dims)) {
      return [
        Math.max(1, dims[0] ?? 1),
        Math.max(1, dims[1] ?? 1),
        Math.max(1, dims[2] ?? 1)
      ];
    }
    if (typeof dims === 'object' && dims !== null) {
      return [
        Math.max(1, dims.x ?? dims[0] ?? 1),
        Math.max(1, dims.y ?? dims[1] ?? 1),
        Math.max(1, dims.z ?? dims[2] ?? 1)
      ];
    }
    if (typeof dims === 'number') {
      return [Math.max(1, dims), 1, 1];
    }
    return [1, 1, 1];
  }

  roundUpToMultiple(value, multiple) {
    if (multiple === 0) return value;
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
      grade: avgScore >= 90 ? 'A' : avgScore >= 80 ? 'B' : avgScore >= 65 ? 'C' : avgScore >= 50 ? 'D' : 'F',
      criticalIssues,
      highIssues,
      needsAttention: analyses.filter(a => a.score < 70)
    };
  }
}