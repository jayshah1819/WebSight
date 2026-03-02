// =============================================================================
// MODULE 6: Analyzer Instances & Logging Helpers
// =============================================================================


import { profilerData } from './data-store.js';
import { MemoryLeakDetector } from './memory-leak-detector.js';
import { WorkgroupOccupancyAnalyzer } from './workgroup-analyzer.js';
import { ShaderComplexityAnalyzer } from './shader-analyzer.js';


export const memoryLeakDetector = new MemoryLeakDetector();
export const workgroupAnalyzer = new WorkgroupOccupancyAnalyzer();
export const shaderAnalyzer = new ShaderComplexityAnalyzer();

// Helper functions for conditional logging
export function log(...args) {
  if (profilerData.config.verboseLogging) {
    console.log(...args);
  }
}

export function warn(...args) {
  if (profilerData.config.verboseLogging) {
    console.warn(...args);
  }
}

// Always show critical errors
export function error(...args) {
  console.error(...args);
}
