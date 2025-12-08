// Background service worker for WebGPU Profiler

chrome.runtime.onInstalled.addListener(() => {
  console.log('WebGPU Compute Profiler installed');
});

// Handle messages from popup or content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'log') {
    console.log('[WebGPU Profiler]', request.message);
  }
  
  sendResponse({ success: true });
  return true;
});
