// =============================================================================
// MODULE 5: Shader Complexity Analyzer
// =============================================================================


export class ShaderComplexityAnalyzer {
  constructor() {
    this.analyses = new Map();

    // List of operations known to have higher latency than basic ALU
    this.expensiveOps = ['sqrt', 'rsqrt', 'sin', 'cos', 'tan', 'exp', 'exp2', 'log', 'log2', 'pow', 'atan', 'atan2'];
  }

  analyzeShader(shaderId, wgslCode) {
    const analysis = {
      shaderId: shaderId,
      code: wgslCode,
      lineCount: 0,
      instructionCount: 0,
      complexity: 0,
      score: 100,
      issues: [],
      recommendations: [],
      metrics: {
        branches: 0,
        loops: 0,
        mathOps: 0,
        memoryOps: 0,
        atomicOps: 0,
        textureOps: 0,  
        variableCount: 0
      }
    };

    if (!wgslCode) {
      analysis.issues.push({
        severity: 'info',
        type: 'no-code',
        message: 'Shader code not available for analysis'
      });
      return analysis;
    }

    const lines = wgslCode.split('\n');
    analysis.lineCount = lines.filter(l => l.trim() && !l.trim().startsWith('//')).length;

    const divergentBranches = this.findDivergentBranches(wgslCode);
    if (divergentBranches.length > 0) {
      analysis.metrics.branches = divergentBranches.length;
      analysis.issues.push({
        severity: 'high',
        type: 'divergent-branch',
        message: `${divergentBranches.length} divergent branch(es) detected`,
        impact: 'Causes thread divergence - SIMD lanes waste cycles',
        locations: divergentBranches,
        recommendation: 'Use select() or bitwise operations instead of if/else when possible',
        example: 'let result = select(falseValue, trueValue, condition); // instead of if'
      });
  
      analysis.score -= Math.min(15 * divergentBranches.length, 45);
    }

    const expensiveMath = this.findExpensiveMath(wgslCode);
    if (expensiveMath.length > 0) {
      const totalCount = expensiveMath.reduce((sum, op) => sum + op.count, 0);
      analysis.metrics.mathOps = totalCount;
      analysis.issues.push({
        severity: 'medium',
        type: 'expensive-math',
        message: `${totalCount} expensive math operation(s): ${expensiveMath.map(o => `${o.operation}(${o.count})`).join(', ')}`,
        impact: 'Transcendental functions have higher latency than basic ALU ops',
        operations: expensiveMath,
        recommendation: 'Consider approximations or lookup tables for non-critical calculations'
      });
      analysis.score -= 10;
    }

    const variables = this.countVariables(wgslCode);
    if (variables > 32) {
      analysis.metrics.variableCount = variables;
      analysis.issues.push({
        severity: 'medium',
        type: 'high-variable-count',
        message: `${variables} variable declarations in shader`,
        impact: 'Many variables may increase register pressure (actual register allocation depends on GPU compiler)',
        recommendation: 'Consider reducing live variables if performance is lower than expected'
      });
      analysis.score -= 10;
    }

    const uncoalescedAccess = this.findUncoalescedAccess(wgslCode);
    if (uncoalescedAccess.length > 0) {
      analysis.metrics.memoryOps = uncoalescedAccess.length;
      analysis.issues.push({
        severity: 'critical',
        type: 'uncoalesced-access',
        message: `${uncoalescedAccess.length} potential uncoalesced memory access(es)`,
        impact: 'Significantly degrades memory bandwidth — actual penalty depends on GPU architecture and cache behavior',
        patterns: uncoalescedAccess,
        recommendation: 'Access memory sequentially: thread[i] accesses data[i], not data[i * stride]'
      });
      analysis.score -= 30;
    }

    const barriers = (wgslCode.match(/workgroupBarrier|storageBarrier/g) || []).length;
    if (barriers > 5) {
      analysis.issues.push({
        severity: 'medium',
        type: 'excessive-barriers',
        message: `${barriers} barrier synchronizations`,
        impact: 'Each barrier stalls all threads - reduces throughput',
        recommendation: 'Minimize barriers by restructuring algorithm'
      });
      analysis.score -= 10;
    }

    const atomics = this.findAtomicOps(wgslCode);
    if (atomics.length > 0) {
      analysis.metrics.atomicOps = atomics.length;
      analysis.issues.push({
        severity: 'medium',
        type: 'atomic-operations',
        message: `${atomics.length} atomic operation(s) found`,
        impact: 'Atomics serialize execution - avoid in hot paths',
        operations: atomics,
        recommendation: 'Use local reduction then single atomic, or avoid atomics entirely'
      });
      analysis.score -= 15;
    }

    const texOps = (wgslCode.match(/\b(textureSample|textureSampleLevel|textureLoad|textureStore)\s*\(/g) || []).length;
    if (texOps > 0) {
      analysis.metrics.textureOps = texOps;
      analysis.issues.push({
        severity: 'medium',
        type: 'texture-operations',
        message: `${texOps} texture operation(s) found`,
        impact: 'Texture ops have higher latency than buffer reads and may stall the pipeline',
        recommendation: 'Cache texture samples in local variables if the same texel is read multiple times'
      });
      analysis.score -= 5 * Math.min(texOps, 4); // cap at -20
    }

    const loops = this.analyzeLoops(wgslCode);
    if (loops.nested > 0) {
      analysis.metrics.loops = loops.total;
      analysis.issues.push({
        severity: 'medium',
        type: 'nested-loops',
        message: `${loops.nested} nested loop(s) with depth ${loops.maxDepth}`,
        impact: 'High iteration count - consider loop unrolling or tiling',
        recommendation: 'Unroll small loops (< 8 iterations) or use loop hints'
      });
      analysis.score -= 10 * loops.maxDepth;
    }

    if (wgslCode.includes('workgroupUniformLoad')) {
      analysis.recommendations.push({
        type: 'good',
        message: ' Uses workgroupUniformLoad - good for shared memory optimization'
      });
    }

    if (/\bsubgroup\w*\s*\(/.test(wgslCode)) {
      analysis.recommendations.push({
        type: 'excellent',
        message: ' Uses subgroup operations - excellent for performance!'
      });
      analysis.score += 10;
    }

    analysis.complexity = this.calculateComplexity(analysis);
  
    analysis.score = Math.max(0, Math.min(100, analysis.score));
    analysis.grade = this.getGrade(analysis.score);

    this.analyses.set(shaderId, analysis);
    return analysis;
  }


  _walkLines(code, callback) {
    const lines = code.split('\n');
    let loopDepth = 0;
    const braceStack = [];

    lines.forEach((line, idx) => {
      const trimmed = line.trim();
      if (trimmed.startsWith('//')) return;
      const noComment = trimmed.replace(/\/\/.*$/, '');
      const isLoopLine = /\b(for|while|loop)\b/.test(noComment);
      const opens = (noComment.match(/{/g) || []).length;
      const closes = (noComment.match(/}/g) || []).length;

      // Closes first: a line like `} for (...) {` exits one scope before entering the next,
      // so depth at the callback reflects the loop the code on this line actually belongs to.
      for (let i = 0; i < closes; i++) {
        if (braceStack.length > 0) {
          if (braceStack.pop() === 'loop') loopDepth--;
        }
      }

      let loopsOpened = 0;
      for (let i = 0; i < opens; i++) {
        if (isLoopLine && loopsOpened === 0) {
          braceStack.push('loop');
          loopDepth++;
  
          loopsOpened++;
        } else {
          braceStack.push('block');
        }
      }

      callback(noComment, trimmed, idx, loopDepth, loopsOpened);
    });
  }

  findDivergentBranches(code) {
    const branches = [];


    const perThreadBuiltins =
      /\b(global_invocation_id|local_invocation_id|local_invocation_index|subgroup_invocation_id)\b/;

    this._walkLines(code, (noComment, trimmed, idx, loopDepth) => {
  
      if (/\bif\b/.test(noComment) && perThreadBuiltins.test(noComment)) {
        branches.push({ line: idx + 1, code: trimmed, inLoopDepth: loopDepth });
      }
    });

    return branches;
  }

  findExpensiveMath(code) {
    const found = [];

    this.expensiveOps.forEach(op => {
      const regex = new RegExp(`\\b${op}\\s*\\(`, 'g');
      const matches = code.match(regex);
      if (matches) {
        found.push({
          operation: op,
          count: matches.length
        });
      }
    });

    return found;
  }

  countVariables(code) {

    const lets = (code.match(/\blet\s+\w+/g) || []).length;
    const vars = (code.match(/\bvar\s+\w+/g) || []).length;
    return lets + vars;
  }

  findUncoalescedAccess(code) {
    const patterns = [];
    const lines = code.split('\n');

    lines.forEach((line, idx) => {
      const trimmed = line.trim();

      if (/\[\s*(global_invocation_id|local_invocation_id)[\w.]*\s*\*\s*\w+\s*\]/.test(line)) {
        patterns.push({ line: idx + 1, code: trimmed, type: 'thread-id-stride' });
      } else {
        
        const strideMatch = line.match(/\[\s*\w+\s*\*\s*(\d+)\s*\]/);
        if (strideMatch && parseInt(strideMatch[1]) > 4) {
          patterns.push({ line: idx + 1, code: trimmed, type: 'strided-access' });
        }
      }


      if (/\[\s*\w+\[\w+\]\s*\]/.test(line)) {
        patterns.push({ line: idx + 1, code: trimmed, type: 'indirect-indexing' });
      }
    });

    return patterns;
  }

  findAtomicOps(code) {
    const atomics = ['atomicAdd', 'atomicSub', 'atomicMax', 'atomicMin', 'atomicAnd', 'atomicOr', 'atomicXor', 'atomicExchange', 'atomicCompareExchangeWeak'];
    const found = [];

    atomics.forEach(op => {
      const regex = new RegExp(`\\b${op}\\s*\\(`, 'g');
      const matches = code.match(regex);
      if (matches) {
        found.push({
          operation: op,
          count: matches.length
        });
      }
    });

    return found;
  }

  analyzeLoops(code) {
    let total = 0, nested = 0, maxDepth = 0;

    this._walkLines(code, (noComment, trimmed, idx, loopDepth, loopsOpened) => {
      if (loopsOpened > 0) {
        total++;

        if (loopDepth > 1) nested++;
        maxDepth = Math.max(maxDepth, loopDepth);
      }
    });

    return { total, nested, maxDepth };
  }

  calculateComplexity(analysis) {
    return (
      analysis.metrics.branches * 3 +
      analysis.metrics.loops * 5 +
      analysis.metrics.mathOps * 2 +
      analysis.metrics.memoryOps * 4 +
      analysis.metrics.atomicOps * 5
    );
  }

  getGrade(score) {
    if (score >= 90) return { letter: 'A', color: 'green', desc: 'Excellent' };
    if (score >= 80) return { letter: 'B', color: 'lightgreen', desc: 'Good' };
    if (score >= 70) return { letter: 'C', color: 'yellow', desc: 'Acceptable' };
    if (score >= 60) return { letter: 'D', color: 'orange', desc: 'Needs Work' };
    return { letter: 'F', color: 'red', desc: 'Poor' };
  }

  getAllAnalyses() {
    return Array.from(this.analyses.values());
  }

  getSummary() {
    const analyses = this.getAllAnalyses();
    if (analyses.length === 0) return null;

    const avgScore = analyses.reduce((sum, a) => sum + a.score, 0) / analyses.length;
    const avgComplexity = analyses.reduce((sum, a) => sum + a.complexity, 0) / analyses.length;

    return {
      totalShaders: analyses.length,
      averageScore: avgScore.toFixed(1),
      averageComplexity: avgComplexity.toFixed(1),
      overallGrade: this.getGrade(avgScore),
      criticalIssues: analyses.filter(a => a.issues.some(i => i.severity === 'critical')).length,
      needsOptimization: analyses.filter(a => a.score < 70)
    };
  }
}
