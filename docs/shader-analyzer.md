# Shader Complexity Analyzer

**Source:** `src/05-shader-analyzer.js`  
**Export:** `ShaderComplexityAnalyzer` class  
**Singleton:** `shaderAnalyzer` — instantiated in `src/06-init-logging.js`, used by Modules 10 and 11.

---

## What it does

WGSL shaders are compiled by the browser's GPU driver, not by WebGPU itself. You cannot inspect register pressure, instruction counts, or occupancy from the JavaScript side after compilation. The Shader Complexity Analyzer operates on the source text instead — scanning for patterns that commonly cause performance problems: divergent branches, uncoalesced memory access, transcendental math, nested loops, excessive atomics, and texture operations.

The output is a score from 0 to 100 and a list of structured issues, each with a severity, a plain-English impact description, and a concrete fix. This is intentionally heuristic — it flags likely problems, not hardware-measured ones.

Module 10 stores the WGSL source on each kernel at `createShaderModule` time. Module 11's `getShaderAnalysis()` runs analysis on-demand across all recorded kernels and returns the results.

---

## How it integrates

Module 10 captures the shader source and attaches a fresh UUID to it:

```js
kernel.shaderId = crypto.randomUUID();
kernel.shader   = descriptor.code;
```

Module 11 calls `analyzeShader` when the caller invokes `WebSight.getShaderAnalysis()`:

```js
shaderAnalyzer.analyzeShader(kernel.shaderId, kernel.shader);
```

Because `shaderId` is a fresh UUID per `createShaderModule` call, the `!analyses.has(shaderId)` guard in Module 11 correctly means "not yet analyzed" — it never returns a stale result for a new shader.

---

## Scoring

Every analysis starts at **100** and has points deducted by rule. After all deductions, a subgroup bonus may be added. Final score is clamped to `[0, 100]`.

| Rule | Severity | Penalty |
|---|---|---|
| Divergent branches (if on per-thread builtin) | high | −15 per branch, capped at −45 total |
| Uncoalesced memory access | critical | −30 |
| Atomic operations present | medium | −15 |
| Expensive math (transcendentals) | medium | −10 |
| High variable count (> 32) | medium | −10 |
| Excessive barriers (> 5) | medium | −10 |
| Nested loops | medium | −10 × maxDepth |
| Texture operations | medium | −5 per op, capped at −20 (4 ops) |
| Non-power-of-2 workgroup dimensions | low | −10 |
| Uses subgroup operations | — | **+10 bonus** |

---

## Issue structure

Each entry in `analysis.issues` follows this shape:

```js
{
  severity:       'critical' | 'high' | 'medium' | 'low' | 'info',
  type:           string,    // machine-readable code
  message:        string,    // human-readable description
  impact:         string,    // what goes wrong
  recommendation: string,    // concrete fix
  // type-specific fields:
  locations:      [{ line, code, inLoopDepth }],   // divergent-branch
  patterns:       [{ line, code, type }],           // uncoalesced-access
  operations:     [{ operation, count }]            // expensive-math, atomic-operations
}
```

---

## Analysis output

`analyzeShader` returns and stores an analysis object:

```js
{
  shaderId:         string,
  code:             string,       // original WGSL source
  lineCount:        number,       // non-empty, non-comment lines
  instructionCount: number,       // reserved (not currently populated)
  complexity:       number,       // weighted sum — see calculateComplexity()
  score:            number,       // 0–100
  grade:            { letter, color, desc },
  issues:           Issue[],
  recommendations:  object[],     // positive findings (subgroup use, workgroupUniformLoad)
  metrics: {
    branches:       number,
    loops:          number,
    mathOps:        number,
    memoryOps:      number,
    atomicOps:      number,
    textureOps:     number,
    variableCount:  number
  }
}
```

---

## API reference

### `analyzeShader(shaderId, wgslCode) → analysis`

Runs all checks, computes score and grade, stores the result in `this.analyses` keyed by `shaderId`, and returns it. If `wgslCode` is empty or null, returns a minimal analysis with a single `info` issue and score 100.

---

### `_walkLines(code, callback)`

Shared line-by-line traversal used by `findDivergentBranches` and `analyzeLoops`. Maintains a brace stack to track loop nesting depth, passing `(noComment, trimmed, lineIndex, loopDepth, loopsOpened)` to the callback.

**Ordering detail:** closes are processed before opens on the same line. This means a line like `} for (...) {` exits the previous scope and only then increments depth — so `loopDepth` at callback time reflects the scope the line's code actually belongs to. `analyzeLoops` relies on this: it checks `loopDepth > 1` for nesting, which requires the increment to have already happened when the callback fires.

---

### `findDivergentBranches(code) → locations[]`

Finds `if` statements that branch on a per-thread builtin value. The builtins checked are `global_invocation_id`, `local_invocation_id`, `local_invocation_index`, and `subgroup_invocation_id`. `lane_id` is not a WGSL built-in and is not included.

Divergence outside loops is flagged — an `if (id.x == 0)` reduction tail at the top level is just as divergent as one inside a loop.

Each result includes `inLoopDepth` so callers can distinguish top-level from in-loop divergence.

---

### `findExpensiveMath(code) → operations[]`

Scans for calls to the 12 transcendental/expensive WGSL functions:

```
sqrt, rsqrt, sin, cos, tan, exp, exp2, log, log2, pow, atan, atan2
```

Returns `[{ operation, count }]` for each function found at least once.

---

### `countVariables(code) → number`

Counts `let` and `var` declarations. `var<storage>` and `var<uniform>` bindings are included in the count because they match `\bvar\s+\w+` — they are not excluded, since they do represent live names the compiler must handle.

---

### `findUncoalescedAccess(code) → patterns[]`

Checks each line for three patterns:

| Type | Pattern | Example |
|---|---|---|
| `thread-id-stride` | `[global/local_invocation_id.x * anyVar]` | `data[id.x * stride]` |
| `strided-access` | `[var * literalConstant]` where constant > 4 | `data[i * 16]` |
| `indirect-indexing` | `[arr[idx]]` — index comes from another array lookup | `data[indices[i]]` |

The thread-id-stride check runs first with an `else` branch for the literal-stride check, preventing a line like `data[id.x * 16]` from being reported twice.

---

### `findAtomicOps(code) → operations[]`

Scans for all 9 WGSL atomic functions:  
`atomicAdd`, `atomicSub`, `atomicMax`, `atomicMin`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicExchange`, `atomicCompareExchangeWeak`.

Returns `[{ operation, count }]` for each found.

---

### `analyzeLoops(code) → { total, nested, maxDepth }`

Delegates to `_walkLines`. Counts loop-opening lines and uses the post-increment `loopDepth` to detect nesting (`loopDepth > 1` = this loop opened inside another loop).

---

### `calculateComplexity(analysis) → number`

Weighted sum of metric counts:

```
branches × 3 + loops × 5 + mathOps × 2 + memoryOps × 4 + atomicOps × 5
```

Higher weight on memory and atomics reflects their typically higher real-world impact.

---

### `getGrade(score) → { letter, color, desc }`

| Score | Grade | Desc |
|---|---|---|
| ≥ 90 | A | Excellent |
| ≥ 80 | B | Good |
| ≥ 70 | C | Acceptable |
| ≥ 60 | D | Needs Work |
| < 60 | F | Poor |

---

### `getAllAnalyses() → analysis[]`

Returns all stored analyses as a flat array. One entry per unique `shaderId`.

---

### `getSummary() → object | null`

Aggregates across all stored analyses. Returns `null` if nothing has been analyzed yet.

```js
{
  totalShaders:       number,
  averageScore:       string,    // toFixed(1)
  averageComplexity:  string,    // toFixed(1)
  overallGrade:       { letter, color, desc },
  criticalIssues:     number,    // shaders with at least one critical issue
  needsOptimization:  analysis[] // score < 70
}
```

---

## Usage via the public API

| `WebSight` method | What it calls |
|---|---|
| `getShaderAnalysis()` | Iterates `profilerData.kernels`, calls `analyzeShader` for any unseen `shaderId`, then calls `getSummary()` + `getAllAnalyses()` — logs full report, returns both |
| `analyzeShader(shaderId)` | Returns the cached analysis for a specific ID, or `null` if not found |
| `getFullAnalysisReport()` | Calls `getSummary()` as part of the combined analysis summary |

---

## Known limitations

| Scenario | Behaviour |
|---|---|
| Shader source not captured | `kernel.shader` is `undefined` — analysis returns score 100 with a single `no-code` info issue |
| Minified / preprocessed WGSL | Pattern matching may miss or double-count constructs across joined lines |
| Subgroup detection | Regex `/\bsubgroup\w*\s*\(/` requires an actual call site — names or comments containing `subgroup` are not counted |
| `var<storage>` bindings | Counted in variable total — they are binding declarations, not local variables, but the regex does not distinguish |
| GPU compiler optimization | All findings are source-level heuristics. The driver may optimize away divergence, reuse registers, or vectorize atomics in ways that make the score misleading |

---

## Data flow

```
10-webgpu-hooks.js
  createShaderModule(desc)
    └─ kernel.shaderId = crypto.randomUUID()
    └─ kernel.shader   = desc.code

11-public-api.js
  getShaderAnalysis()
    └─ for each kernel with unseen shaderId:
         shaderAnalyzer.analyzeShader(shaderId, shader)
           └─ analyses.set(shaderId, result)
    └─ shaderAnalyzer.getSummary()
    └─ shaderAnalyzer.getAllAnalyses()

  analyzeShader(shaderId)
    └─ shaderAnalyzer.analyses.get(shaderId)
```
