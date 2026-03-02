// =============================================================================
// MODULE 9: Bandwidth Tracking & Memory Pattern Analysis
// =============================================================================



export function analyzeMemoryAccessPattern(shaderCode, bufferAccesses, workgroupSize, dispatchSize) {
  const hints = [];

  if (!shaderCode) return {
    accessPatternType: 'unknown',
    hints,
    coalesced: null,
    cacheEfficiency: 'unknown'
  };

  if (/var\s*<\s*workgroup\s*>/.test(shaderCode))
    hints.push('uses-shared-memory');

  if (/\batomicAdd\b|\batomicOr\b|\batomicAnd\b|\batomicMin\b|\batomicMax\b|\batomicExchange\b|\batomicCompareExchangeWeak\b/.test(shaderCode))
    hints.push('uses-atomics');

  if (/\bworkgroupBarrier\b|\bstorageBarrier\b/.test(shaderCode))
    hints.push('uses-barriers');

  const mathOps = (shaderCode.match(/\b(sin|cos|exp|log|pow|sqrt)\b/g) || []).length;
  if (mathOps > 10) hints.push('math-heavy');

  return {
    accessPatternType: 'unknown', 
    hints,
    coalesced: null,           
    cacheEfficiency: 'unknown'
  };
}

export function calculateTextureSize(textureView) {
  // Helper to calculate texture memory size
  const texture = textureView.__websight_texture;

  if (!texture) return 0;

  const formatBytesPerPixel = {
    'r8unorm': 1, 'r8snorm': 1, 'r8uint': 1, 'r8sint': 1,
    'r16uint': 2, 'r16sint': 2, 'r16float': 2, 'rg8unorm': 2, 'rg8snorm': 2,
    'r32uint': 4, 'r32sint': 4, 'r32float': 4, 'rg16uint': 4, 'rg16sint': 4, 'rg16float': 4,
    'rgba8unorm': 4, 'rgba8unorm-srgb': 4, 'rgba8snorm': 4, 'rgba8uint': 4, 'rgba8sint': 4,
    'bgra8unorm': 4, 'bgra8unorm-srgb': 4,
    'rgb10a2unorm': 4, 'rg11b10ufloat': 4,
    'rg32uint': 8, 'rg32sint': 8, 'rg32float': 8, 'rgba16uint': 8, 'rgba16sint': 8, 'rgba16float': 8,
    'rgba32uint': 16, 'rgba32sint': 16, 'rgba32float': 16,
    'depth32float': 4, 'depth24plus': 4, 'depth24plus-stencil8': 5, 'depth32float-stencil8': 5,
 
    'bc1-rgba-unorm': 0.5, 'bc1-rgba-unorm-srgb': 0.5,
    'bc7-rgba-unorm': 1, 'bc7-rgba-unorm-srgb': 1,
    'etc2-rgb8unorm': 0.5, 'etc2-rgb8unorm-srgb': 0.5,
    'astc-4x4-unorm': 1, 'astc-4x4-unorm-srgb': 1
  };

  const bytesPerPixel = formatBytesPerPixel[texture.format] || 4;
  const width = texture.width || 1;
  const height = texture.height || 1;
  const depth = texture.depthOrArrayLayers || 1;
  const mipLevels = texture.mipLevelCount || 1;

  let totalSize = 0;
  for (let mip = 0; mip < mipLevels; mip++) {
    const mipWidth = Math.max(1, width >> mip);
    const mipHeight = Math.max(1, height >> mip);
    totalSize += mipWidth * mipHeight * depth * bytesPerPixel;
  }

  return totalSize;
}

export function calculateFramebufferBandwidth(passDescriptor) {
  let totalBytesRead = 0;
  let totalBytesWritten = 0;

  if (!passDescriptor) return { bytesRead: 0, bytesWritten: 0 };

  const colorAttachments = passDescriptor.colorAttachments || [];
  const depthAttachment = passDescriptor.depthStencilAttachment;

  colorAttachments.forEach((attachment, idx) => {
    if (attachment && attachment.view) {
      let attachmentBytes = calculateTextureSize(attachment.view);


      if (attachmentBytes === 0) {
        const view = attachment.view;
        let width, height;

        if (view.texture) {
          width = view.texture.width;
          height = view.texture.height;
        }

        if (!width && view.descriptor) {
          width = view.descriptor.size?.width;
          height = view.descriptor.size?.height;
        }

        if (!width && view.__texture) {
          width = view.__texture.width;
          height = view.__texture.height;
        }

        if (width && height) {
          // Fallback: assume 4 bytes/pixel (RGBA8) when texture format is unavailable
          const bytesPerPixel = 4;
          attachmentBytes = width * height * bytesPerPixel;
        }
        // If dimensions can't be determined, leave attachmentBytes = 0
      }

      // Always write to color attachment
      totalBytesWritten += attachmentBytes;

      // Read if blending or loading previous contents
      if (attachment.loadOp === 'load') {
        totalBytesRead += attachmentBytes;
      }
    }
  });

  // Depth/stencil bandwidth
  if (depthAttachment && depthAttachment.view) {
    const depthBytes = calculateTextureSize(depthAttachment.view);

    totalBytesWritten += depthBytes;
    if (depthAttachment.depthLoadOp === 'load') {
      totalBytesRead += depthBytes;
    }
  }

  return { bytesRead: totalBytesRead, bytesWritten: totalBytesWritten };
}


export function parseWGSLBindings(shaderCode) {
  if (!shaderCode) return {};
  const groups = {};

  // Buffer bindings: @group(G) @binding(B) var<addressSpace[, accessMode]> name: type
  const bufRe = /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s*<\s*(\w+)(?:\s*,\s*(\w+))?\s*>/g;
  let m;
  while ((m = bufRe.exec(shaderCode)) !== null) {
    const groupIdx = parseInt(m[1]);
    const bindingIdx = parseInt(m[2]);
    const addressSpace = m[3];
    const accessMode = m[4]; 

    if (!groups[groupIdx]) groups[groupIdx] = {};

    if (addressSpace === 'uniform') {
      groups[groupIdx][bindingIdx] = 'uniform';
    } else if (addressSpace === 'storage') {
      // WGSL default access for storage is read; read_write is explicit
      groups[groupIdx][bindingIdx] = accessMode === 'read_write' ? 'storage' : 'read-only-storage';
    }
  }


  // These use plain `var` (no angle-bracket address space).
  const texRe = /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s+\w+\s*:\s*(\w+)/g;
  while ((m = texRe.exec(shaderCode)) !== null) {
    const groupIdx = parseInt(m[1]);
    const bindingIdx = parseInt(m[2]);
    const typeName = m[3];

    if (!groups[groupIdx]) groups[groupIdx] = {};

    if (typeName.startsWith('texture_storage')) {
      groups[groupIdx][bindingIdx] = 'storage-texture';
    } else if (typeName.startsWith('texture_')) {
      groups[groupIdx][bindingIdx] = 'sampled-texture';
    } else if (typeName === 'sampler' || typeName === 'sampler_comparison') {
      groups[groupIdx][bindingIdx] = 'sampler';
    }
  }

  return groups;
}

export class BandwidthTracker {
  constructor() {
    this.passResources = new Map();
    this.totalBytesRead = 0;
    this.totalBytesWritten = 0;
    this.totalBytesReadWrite = 0;
  }

  trackBindGroup(bindGroup, bindGroupLayout) {
    if (!bindGroup || !bindGroup.__websight_resources) return;

    bindGroup.__websight_resources.forEach(res => {
      if (!this.passResources.has(res.id)) {
        this.passResources.set(res.id, {
          size: res.size,
          type: res.type,
          usage: res.usage,
          accessType: res.accessType
        });

        
        if (res.accessType === 'read-only') {
          this.totalBytesRead += res.size;
        } else if (res.accessType === 'read-write') {
          this.totalBytesReadWrite += res.size;
        }
      }
    });
  }

  calculateBandwidth(durationNs) {
    // read-write buffers (var<storage, read_write>) count as both read and written
    const bytesRead = this.totalBytesRead + this.totalBytesReadWrite;
    const bytesWritten = this.totalBytesWritten + this.totalBytesReadWrite;
    const totalBytes = bytesRead + bytesWritten;

    return {
      bytesRead,
      bytesWritten,
      totalBytes,
      totalDataMB: (totalBytes / (1024 ** 2)).toFixed(2),
    
     
      // counters (unavailable in WebGPU) would be required for accurate measurement.
      measurementNote: 'bound-buffer-size',
      bandwidthGBs: durationNs > 0 ? parseFloat((totalBytes / (durationNs / 1e9) / 1e9).toFixed(2)) : 0,
      resourceCount: this.passResources.size
    };
  }
}


/**
 * Returns the bandwidth record for a single compute dispatch plus a snapshot
 * to advance the caller's running tally to the next dispatch.
 *
 * @param {BandwidthTracker} tracker      - the pass-level tracker
 * @param {{read:number, written:number}} snapshotBefore - tracker totals after the previous dispatch
 * @param {string} accessPatternType      - from analyzeMemoryAccessPattern
 */
export function calculateDispatchBandwidth(tracker, snapshotBefore, accessPatternType) {
  const totals     = tracker.calculateBandwidth(0);
  const bytesRead    = totals.bytesRead    - snapshotBefore.read;
  const bytesWritten = totals.bytesWritten - snapshotBefore.written;
  return {
    bandwidth: {
      bytesRead,
      bytesWritten,
      totalBytes:          bytesRead + bytesWritten,
      measurementNote:     'bound-buffer-size',
      bandwidthGBs:        0, // backfilled by onSubmittedWorkDone once GPU time is known
      arithmeticIntensity: 0,
      resourceCount:       totals.resourceCount,
      accessPattern:       accessPatternType || 'unknown'
    },
    newSnapshot: { read: totals.bytesRead, written: totals.bytesWritten }
  };
}

/**
 * multiplying the framebuffer cost by the number of draws in the pass.
 *
 * @param {BandwidthTracker} tracker
 * @param {{read:number, written:number}} snapshotBefore
 * @param {GPURenderPassDescriptor|null} passDescriptor
 * @param {boolean} fbAlreadyCounted
 */
export function calculateRenderDrawBandwidth(tracker, snapshotBefore, passDescriptor, fbAlreadyCounted) {
  const totals       = tracker.calculateBandwidth(0);
  const deltaRead    = totals.bytesRead    - snapshotBefore.read;
  const deltaWritten = totals.bytesWritten - snapshotBefore.written;

  let fbRead = 0, fbWritten = 0, fbCountedNow = false;
  if (!fbAlreadyCounted && passDescriptor) {
    const fb  = calculateFramebufferBandwidth(passDescriptor);
    fbRead    = fb.bytesRead;
    fbWritten = fb.bytesWritten;
    fbCountedNow = true;
  }

  return {
    bandwidth: {
      bytesRead:           deltaRead    + fbRead,
      bytesWritten:        deltaWritten + fbWritten,
      totalBytes:          (deltaRead + fbRead) + (deltaWritten + fbWritten),
      measurementNote:     'bound-buffer-size',
      bandwidthGBs:        0,
      arithmeticIntensity: 0,
      resourceCount:       totals.resourceCount,
      accessPattern:       'render'
    },
    newSnapshot:  { read: totals.bytesRead, written: totals.bytesWritten },
    fbCountedNow
  };
}