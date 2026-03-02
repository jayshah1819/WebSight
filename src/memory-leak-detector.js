// =============================================================================
// MODULE 3: Memory Leak Detector
// =============================================================================



import { profilerData } from './data-store.js';

export class MemoryLeakDetector {

  #resourceIds = new WeakMap();

  constructor() {
    this.resources = new Map();
    this.nextId = 0;
    this.leakThreshold = 60000;
    this.sizeThreshold = 100 * 1024 * 1024;
    this.autoCheckInterval = null;

    this.stats = {
      totalAllocated: 0,
      totalFreed: 0,
      peakMemory: 0,
      currentMemory: 0,
      leakCount: 0,
      createdCount: 0,
      destroyedCount: 0
    };
  }

  
  enableAutoCheck(intervalMs = 30000) {
    if (!this.autoCheckInterval) {
      this.autoCheckInterval = setInterval(() => this.checkForLeaks(), intervalMs);
    }
  }

  disableAutoCheck() {
    if (this.autoCheckInterval) {
      clearInterval(this.autoCheckInterval);
      this.autoCheckInterval = null;
    }
  }

  trackResource(resource, type, size) {
 
    if (this.#resourceIds.has(resource)) {
      const existingId = this.#resourceIds.get(resource);
      const existing = this.resources.get(existingId);
      if (existing && !existing.destroyed) return existingId;
    }

    const id = this.nextId++;
    const now = Date.now();

    let stack = '';
    if (profilerData.config.captureStacks) {
      try {
        throw new Error();
      } catch (e) {
        stack = e.stack;
      }
    }

    this.resources.set(id, {
      type: type,
      size: size,
      createdAt: now,
      stack: stack,
      destroyed: false,
      label: resource.label || 'unlabeled'
    });


    this.#resourceIds.set(resource, id);

    this.stats.totalAllocated += size;
    this.stats.currentMemory += size;
    this.stats.createdCount++;

    if (this.stats.currentMemory > this.stats.peakMemory) {
      this.stats.peakMemory = this.stats.currentMemory;
    }

    return id;
  }

  markDestroyed(resource) {

    const id = this.#resourceIds.get(resource);
    if (id === undefined) return;

    const info = this.resources.get(id);
    if (!info || info.destroyed) return;

    info.destroyed = true;
    info.destroyedAt = Date.now();
    info.lifetime = info.destroyedAt - info.createdAt;

    this.stats.totalFreed += info.size;
    this.stats.currentMemory -= info.size;
    this.stats.destroyedCount++;

    if (this.stats.destroyedCount % 100 === 0) {
      const cutoff = Date.now() - 60000;
      for (const [eid, einfo] of this.resources.entries()) {
        if (einfo.destroyed && einfo.destroyedAt < cutoff) {
          this.resources.delete(eid);
        }
      }
    }
  }

  
  #findLeaks() {
    const now = Date.now();
    const leaks = [], active = [], destroyed = [];

    for (const [id, info] of this.resources.entries()) {
      if (!info.destroyed) {
        const age = now - info.createdAt;
        // stack is only included on leak entries where it's actionable for debugging;

        const base = { id, type: info.type, size: info.size, age, label: info.label };
        if (age > this.leakThreshold) {
          leaks.push({ ...base, isLeak: true, stack: info.stack });
        } else {
          active.push(base);
        }
      } else {
        destroyed.push({ id, type: info.type, size: info.size, lifetime: info.lifetime, label: info.label });
      }
    }

    this.stats.leakCount = leaks.length;
    return { leaks, active, destroyed };
  }

  checkForLeaks() {
    const { leaks } = this.#findLeaks();

    if (leaks.length > 0) {
      console.warn(`[WebSight] ${leaks.length} potential memory leaks detected!`);

      const byType = {};
      leaks.forEach(leak => {
        if (!byType[leak.type]) byType[leak.type] = [];
        byType[leak.type].push(leak);
      });

      console.table(Object.entries(byType).map(([type, items]) => ({
        Type: type,
        Count: items.length,
        TotalSize: this.formatBytes(items.reduce((sum, l) => sum + l.size, 0)),
        OldestAge: this.formatTime(Math.max(...items.map(l => l.age)))
      })));
    }

    if (this.stats.currentMemory > this.sizeThreshold) {
      console.warn(`[WebSight] High memory usage: ${this.formatBytes(this.stats.currentMemory)}`);
    }
  }

  getLeakReport() {
    const { leaks, active, destroyed } = this.#findLeaks();

    return {
      stats: this.stats,
      leaks: leaks.sort((a, b) => b.size - a.size),
      active: active.sort((a, b) => b.size - a.size),
      destroyed: destroyed.sort((a, b) => b.lifetime - a.lifetime).slice(0, 100),
      summary: {
        totalLeaks: leaks.length,
        leakedMemory: leaks.reduce((sum, l) => sum + l.size, 0),
        activeResources: active.length,
        activeMemory: active.reduce((sum, a) => sum + a.size, 0),
        destroyedResources: destroyed.length,
        leakRate: this.stats.createdCount > 0 ? (leaks.length / this.stats.createdCount * 100).toFixed(2) + '%' : '0%'
      }
    };
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
  }

  formatTime(ms) {
    if (ms < 1000) return ms + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
    return (ms / 60000).toFixed(1) + 'min';
  }
}
