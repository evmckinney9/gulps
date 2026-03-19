# Segment Rearrangement: Exploiting Monodromy Symmetry for Solver Dispatch

## Problem

The segment solver finds `k = kron(u1, u0) ∈ SU(2)⊗SU(2)` such that
`G · k · C ~ T` (locally equivalent via Makhlin invariants). The Makhlin
GN's difficulty depends on the **target's** Weyl coordinates: at c₁≈c₂,
the Jacobian undergoes a rank 3→2 transition during convergence, causing
oscillation and requiring many restarts (the documented tail problem).

The key insight: the triple (C, G, T) satisfying the monodromy constraint
encodes the **same information** regardless of which entity plays which role.
There are three valid formulations of the same segment problem, each with a
different entity as the Makhlin target. We should solve whichever is easiest.

## The Three Formulations

Given `T ∈ P(C, G)` (monodromy polytope feasibility), three equivalent
segment problems exist:

| # | Notation | Solver inputs | Target |
|---|----------|---------------|--------|
| 1 | L(C, G, T) | prefix=Can(C), gate=G_mat, target=Can(T) | LP intermediate T |
| 2 | L(T, G†, C) | prefix=Can(T), gate=G_mat†, target=Can(C) | LP intermediate C |
| 3 | L(C†, T, G) | prefix=Can(C)†, gate=Can(T), target=Can(G) | ISA gate G |

### Derivations

**Formulation 2** (swap prefix/target, invert gate):
From `G · k · C ~ T`, take the dagger: `C† · k† · G† ~ T†`. Rearrange using
Weyl group to get `T · k' · G† ~ C` in the Weyl chamber.

**Formulation 3** (gate becomes target):
From `G · k · C = k_L · T · k_R`, isolate G:
`G = k_L · T · k_R · C⁻¹ · k⁻¹`, so `Weyl(T · k' · C⁻¹) = Weyl(G)`.
Equivalently, from monodromy: `T ∈ P(C, G) ⟹ G ∈ P(C†, T)`.

## Recovery Algebra

Each rearranged solution requires converting back to the original `u0, u1`.
The recovery uses one call to `recover_local_equivalence` (~56μs).

### Formulation 2: L(T, G†, C) → original u0, u1

```
1. Solve: find v0, v1 s.t. Makhlin(G† · kron(v1,v0) · Can(T)) = Makhlin(Can(C))
2. Compute: U_rev = G_mat† @ kron(v1,v0) @ Can(T)
3. Recovery: recover_local_equivalence(Can(C), U_rev) → (k1, k2, k3, k4, gphase)
4. Result:  u0 = k3†,  u1 = k4†   (LEFT K-factors, conjugate-transposed)
```

**Proof**: From `U_rev = k_L · Can(C) · k_R` and `Can(C) = k_L⁻¹ · U_rev · k_R⁻¹`:
```
G · kron(k4†,k3†) · Can(C)
  = G · k_L⁻¹ · U_rev · k_R⁻¹      [substituting Can(C)]
  = G · G† · kron(v1,v0) · Can(T) · kron(k2,k1) · k_R⁻¹
  = kron(v1,v0) · Can(T) · (local)
  → Weyl = T  ✓   [since G·G† = I]
```

### Formulation 3: L(C†, T, G) → original u0, u1

```
1. Solve: find w0, w1 s.t. Makhlin(Can(T) · kron(w1,w0) · Can(C)†) = Makhlin(Can(G))
2. Compute: U_3 = Can(T) @ kron(w1,w0) @ Can(C).conj().T
3. Recovery: recover_local_equivalence(Can(G), U_3) → (k1, k2, k3, k4, gphase)
4. Result:  u0 = k1†,  u1 = k2†   (RIGHT K-factors, conjugate-transposed)
```

**Proof**: From `U_3 = k_L · Can(G) · k_R` with `k_R = kron(k2†, k1†)`:
```
G · k_R · Can(C)
  = k_L⁻¹ · U_3 · Can(C)
  = k_L⁻¹ · Can(T) · kron(w1,w0) · Can(C)† · Can(C)
  = k_L⁻¹ · Can(T) · kron(w1,w0)
  → Weyl = T  ✓   [since Can(C)† · Can(C) = I]
```

### Summary table

| Formulation | Compute U_rearranged | Recovery call | u0 | u1 |
|---|---|---|---|---|
| L(C, G, T) | — | — | direct | direct |
| L(T, G†, C) | `G† @ kron(v) @ Can(T)` | `recover(Can(C), U)` | k3† | k4† |
| L(C†, T, G) | `Can(T) @ kron(w) @ Can(C)†` | `recover(Can(G), U)` | k1† | k2† |

## When Each Formulation Helps

The solver difficulty is determined by the **target's** Weyl face:

| Target face | Jacobian behavior | Convergence |
|---|---|---|
| Generic (interior) | Rank 3 throughout | ~100%, fast |
| c₁≈c₂ (iSWAP face) | Rank 3→2 transition | 8-54%, slow |
| c₂≈c₃ with c₂>0 | Rank 3→2 transition | Hard (80% of slow cases) |
| c₂=c₃=0 (CX face) | Consistently rank 2 | 90-100%, fast |
| SWAP (π/4,π/4,π/4) | Trivial (any k works) | 100%, instant |

**Formulation 3 is the key insight.** For mixed ISAs (e.g., sq2cx+sq3iswap):
- LP intermediates may all lie on c₁=c₂ → formulations 1 & 2 both target c₁=c₂ face (hard)
- But the CX-type gate (sq2cx) is on c₂=c₃=0 → formulation 3 targets CX face (easy!)

For pure iSWAP ISAs: all three formulations target c₁=c₂. No help (theoretical wall).

## Dispatch Metric

The raw gap `min(|c₁-c₂|, |c₂-c₃|)` is wrong — it scores CX targets as
degenerate (gap=0) when they're actually easy. The `|c₁-c₂|` gap alone
misses the c₂≈c₃ face (80% of slow cases).

### Proposed approach

Use a two-component score in monodromy space:

```python
def _target_difficulty(inv):
    """Estimate solver difficulty when inv is the Makhlin target.
    Lower = easier. Returns 0 for trivially easy targets."""
    m = inv.monodromy
    c1_c2_gap = abs(m[1] - m[2])     # |c₁-c₂| in Weyl
    c2_c3_gap = abs(m[0] - m[1])     # |c₂-c₃| in Weyl
    c2_plus_c3 = abs(m[1] + m[2])    # magnitude of c₂+c₃

    # CX face (c₂≈c₃≈0): consistently rank 2, no transition → easy
    # iSWAP face (c₁≈c₂): rank transition → hard
    # c₂≈c₃ with c₂>0: rank transition → hard
    ...
```

The exact metric needs empirical calibration via diagnostics on `make ab`.
A reasonable starting point:

- **Primary**: `|c₁-c₂|` in monodromy = `|m[1]-m[2]|`
- **Secondary**: `|c₂-c₃|` in monodromy = `|m[0]-m[1]|`, weighted by magnitude
  of `c₂` to discount the easy CX face

**Empirical validation needed**: for each hard segment in the benchmark suite,
log the three formulations' target gaps and estimated difficulties. This will
reveal whether the metric correctly identifies the easiest formulation.

## Implementation Plan

### Phase 1: Diagnostic (validate the theory)

Write a diagnostic script (`scripts/temp_diagnostics/rearrange_diagnostic.py`) that:
1. Runs the benchmark suite (weyl_speed grid + Haar-random)
2. For each segment in each decomposition, computes:
   - The three target Weyl coordinates
   - The three target difficulty scores
   - The actual solve time and convergence stats
3. Reports: how many hard segments (>5ms) could have been dispatched to an
   easier formulation, and the estimated time savings.

This tells us the **ceiling** for the rearrangement optimization.

### Phase 2: Core implementation

**Files to modify:**

1. **`segments_solver.py`** — Add dispatch logic in `_synthesize_batch`:
   - Before calling `jax_lm.try_solve`, compute difficulty of all three formulations
   - Choose the easiest formulation
   - If rearranged: compute rearranged solver inputs, call solver, run recovery
   - Cache the result in original format

2. **`jax_lm.py`** — No changes to the JIT'd solver or XLA graph.
   Add a helper method `_solve_rearranged(formulation, prefix_inv, basis_inv, target_inv)`
   that:
   - Computes the rearranged matrices (gate inverse, prefix inverse, etc.)
   - Calls `self._solve` with rearranged inputs
   - Runs algebraic recovery to produce original u0, u1
   - Handles PRNG retry on failure

3. **`segments_cache.py`** — Make cache key invariant under rearrangement:
   - For c₃=0 gates (most ISAs): canonical key = `(min(C._key, T._key), G._key, max(C._key, T._key))`
   - For c₃≠0 gates: include formulation 3 key using gate inverse monodromy
   - On hit: convert stored solution if formulation differs from request

### Phase 3: Benchmark

Run `make ab` before and after. Metrics:
- Grand total (weyl_speed): should improve for mixed ISAs
- Per-ISA max: the tail metric
- Median: must not regress
- Correctness: all 25 tests must pass

### What this does NOT add

- Zero new solver parameters or tolerances
- Zero changes to the JIT'd XLA graph
- Zero overhead on generic segments (dispatch check is O(1))
- Recovery cost (~56μs) only fires on rearranged segments

### Architectural constraints

- The rearrangement is purely a **dispatch** optimization — it changes which
  problem the existing solver sees, not how the solver works.
- The existing solver, stagnation abort, Weyl early-exit, and Makhlin fallback
  all work identically on rearranged inputs.
- The recovery uses the existing `recover_local_equivalence` function.

## Open Questions

1. **Dispatch metric calibration**: The exact threshold for when to rearrange
   needs empirical data. The diagnostic (Phase 1) will provide this.

2. **Formulation 3 with non-canonical gate matrices**: The solver currently
   uses the ISA gate's actual unitary (with local parts) as `basis_gate`.
   Formulation 3 uses `Can(T)` as the gate — always canonical. This might
   interact differently with the Makhlin residual computation. Needs testing.

3. **Cache hit rate improvement**: How often do different targets produce the
   same (C, G, T) triple in rearranged form? Likely rare for the segment cache,
   but could matter for repeated decompositions of similar targets.

4. **Interaction with PRNG retry**: The retry uses `fold_in(key, step ^ 0xCAFE)`.
   Should rearranged solves use a different diversification to avoid correlated
   failures?

## Mathematical Context

### Why the nonlinear solve cannot be eliminated entirely

The Chevalley restriction theorem guarantees that **every** smooth function
characterizing the double coset K\SU(4)/K has rank ≤ 2 at c₁=c₂. This is
not a property of the Makhlin invariants specifically — it's a property of
the symmetric space structure. No smooth change of invariants can resolve this.

The rearrangement doesn't change the invariants — it changes **which Weyl
point** the solver targets. This sidesteps the Chevalley obstruction by
avoiding degenerate targets entirely (when possible).

### SWAP as the extreme case

SWAP at (π/4, π/4, π/4) has **maximal isotropy**: the monodromy polytope
from SWAP collapses to a single point. Every k ∈ K gives the same Weyl
output. This trivializes the solve — any k works, and the remaining work
is purely KAK recovery.

The rearrangement generalizes this: at non-SWAP degenerate points, the
monodromy polytope is reduced but not trivial. The solve is still needed,
but choosing a well-conditioned target makes it fast.

### The circular dependency

The triple (C, G, T) involves three K-valued unknowns (L, k₁, k₂) coupled
by `C · L · G = k₁ · T · k₂`. Each rearrangement makes a different unknown
"hard" (requiring the nonlinear solve) while the others become "easy" (KAK
recovery). The hard solve cannot be eliminated — it can only be dispatched
to the best-conditioned formulation.
