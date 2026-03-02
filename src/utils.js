// =============================================================================
// MODULE 8: Utility Functions
// =============================================================================


import { profilerData } from './data-store.js';
import { broadcastData } from './broadcast.js';

export function normalizeTime(timeNs) {
  const unit = profilerData.config.normalizeTimeUnit;
  switch (unit) {
    case 'ns': return timeNs;
    case 'us': return timeNs / 1000;
    case 'ms': return timeNs / 1000000;
    default: return timeNs / 1000;
  }
}

export function getTimeUnitLabel() {
  const unit = profilerData.config.normalizeTimeUnit;
  switch (unit) {
    case 'ns': return 'ns';
    case 'us': return 'µs';
    case 'ms': return 'ms';
    default: return 'µs';
  }
}

export function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

export function generateKernelId(shaderSource, workgroupSize, label = '') {
  const config = `${workgroupSize.x}x${workgroupSize.y}x${workgroupSize.z}`;
  const sourceHash = hashString(shaderSource);
  const configHash = hashString(config);
  const labelHash = label ? `_${hashString(label)}` : '';
  return `kernel_${sourceHash}_${configHash}${labelHash}`;
}

export function extractWorkgroupSize(source, entryPoint) {
  if (!source) return { x: 1, y: 1, z: 1 };


  const constMap = {};
  for (const m of source.matchAll(/(?:const|override)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(\d+)/g)) {
    constMap[m[1]] = parseInt(m[2]);
  }

  function resolveComponent(raw) {
    if (raw === undefined || raw === null) return 1;
    const s = raw.trim();
    if (/^\d+$/.test(s)) return parseInt(s) || 1;
    return constMap[s] || 1;
  }


  if (entryPoint) {
    const fnRe = new RegExp(`\\bfn\\s+${entryPoint}\\s*\\(`);
    const fnMatch = fnRe.exec(source);
    if (fnMatch) {
      const before = source.slice(0, fnMatch.index);
      const allWs = [...before.matchAll(/@workgroup_size\(\s*([^)]+)\s*\)/g)];
      if (allWs.length > 0) {
        const last = allWs[allWs.length - 1];
        const parts = last[1].split(/\s*,\s*/);
        return {
          x: resolveComponent(parts[0]),
          y: resolveComponent(parts[1]),
          z: resolveComponent(parts[2]),
        };
      }
    }
  }

  const match = source.match(/@workgroup_size\(\s*([^)]+)\s*\)/);
  if (match) {
    const parts = match[1].split(/\s*,\s*/);
    return {
      x: resolveComponent(parts[0]),
      y: resolveComponent(parts[1]),
      z: resolveComponent(parts[2]),
    };
  }

  return { x: 1, y: 1, z: 1 };
}

export function analyzeWGSL(source) {
  if (!source) return { warnings: [], metrics: {} };
  const warnings = [];
  const metrics = {
    hasAtomics: false,
    hasBranching: false,
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

export function addLog(message, level = 'info') {
  profilerData.logs.push({
    timestamp: new Date().toLocaleTimeString(),
    level,
    message,
    time: Date.now()
  });
  if (profilerData.config.verboseLogging) {
    console.log(`[WebSight] ${message}`);
  }
  broadcastData();
}
