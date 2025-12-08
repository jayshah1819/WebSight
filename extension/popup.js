// Popup script for WebGPU Profiler Extension

let profilerActive = false;

function updateUI() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'getStats' }, (response) => {
        if (chrome.runtime.lastError) {
          console.log('Could not connect to content script');
          return;
        }

        if (response && response.stats) {
          const stats = response.stats;
          
          document.getElementById('totalDispatches').textContent = stats.totalDispatches;
          document.getElementById('activePipelines').textContent = stats.activePipelines;
          
          if (stats.gpuCharacteristics) {
            document.getElementById('gpuBandwidth').textContent = 
              stats.gpuCharacteristics.estimatedBandwidth.toFixed(1) + ' GB/s';
          }
          
          const statusIndicator = document.getElementById('statusIndicator');
          const statusMessage = document.getElementById('statusMessage');
          
          if (response.active) {
            statusIndicator.className = 'status-indicator status-active';
            statusIndicator.textContent = ' Monitoring';
            statusMessage.textContent = 'Capturing WebGPU operations on this page';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
          } else {
            statusIndicator.className = 'status-indicator status-inactive';
            statusIndicator.textContent = '⚫ Not Monitoring';
            statusMessage.textContent = 'Click "Start Profiling" to begin';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
          }
        }
      });
    }
  });
}

document.getElementById('startBtn').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'start' }, () => {
      profilerActive = true;
      updateUI();
    });
  });
});

document.getElementById('stopBtn').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'stop' }, () => {
      profilerActive = false;
      updateUI();
    });
  });
});

document.getElementById('exportBtn').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'export' });
  });
});

document.getElementById('clearBtn').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'clear' }, () => {
      updateUI();
    });
  });
});

// Update UI on load and periodically
updateUI();
setInterval(updateUI, 2000);
