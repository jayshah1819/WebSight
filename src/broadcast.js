// =============================================================================
// MODULE 7: Broadcast & Control Messages
// =============================================================================


import { profilerData } from './data-store.js';

export const profilerChannel = new BroadcastChannel('websight-profiler');
let broadcastTimer = null;

// Listen for control messages from UI
profilerChannel.onmessage = (event) => {
  if (window.__webSightIsUIWindow) return; 

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
    broadcastData(); 
  }
};

export function broadcastData() {
  if (window.__webSightIsUIWindow) {
    return;
  }

  if (!profilerData.config.broadcastEnabled) {
    return;
  }
  if (broadcastTimer) return;

  broadcastTimer = setTimeout(() => {
    try {

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
          timingMode: profilerData.timingMode,
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
