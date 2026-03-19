# Solver Knobs Explored

A comprehensive catalog of every tunable parameter, dispatch condition, and algorithmic variant
explored during the optimization of the GULPS two-qubit gate decomposition solver.  Organized by
subsystem.  Each entry records the parameter, the values tested, and the outcome.

---

## 1. Current Production Parameters

These are the parameters that survived optimization and are shipped in the current solver.

### 1.1 Config Parameters (`GulpsConfig`)

| Parameter | Value | Role |
|-----------|-------|------|
| `makhlin_conv_tol` | `1e-9` | Makhlin invariant convergence threshold for GN restart loop early-exit |
| `weyl_conv_tol` | `1e-5` | Weyl coordinate convergence threshold for Weyl early-exit and inline polish |
| `lp_feasibility_tol` | `1e-10` | Primal/dual feasibility tolerance for the LP solver |
| `flag_duration` | `0.05` | Warning threshold (seconds) for slow decompositions |
| `segment_cache_size` | `3` | LFU cache size per step index |

### 1.2 Compile-Time Constants (`jax_lm.py`)

| Constant | Value | Role |
|----------|-------|------|
| `_MAXITER` | `512` | Max GN iterations per restart (inner loop) |
| `_MAX_RESTARTS` | `256` | Max random restarts (outer loop) |
| Init range | `[-0.1, 0.1]` | Uniform random init for quaternion params |
| GN regularization | `1e-14` | Tikhonov damping `(J J^T + λI)` in normal equations |
| Weyl polish budget | `128` | Max LM iterations for inline Weyl polish |
| Weyl polish init λ | `1e-3` | Initial LM damping for polish |
| Weyl polish λ range | `[1e-12, 1e6]` | Clipped adaptive damping bounds |

### 1.3 Stagnation Abort Tiers

Three-tier absolute stagnation check in the GN inner loop.  Aborts a restart early
when the residual is above a threshold after a given number of iterations.

| Tier | Iteration threshold | Residual threshold | Catch rate |
|------|--------------------|--------------------|------------|
| 1 | `j >= 8` | `prev_norm > 3e-1` | ~70% of bad basins |
| 2 | `j >= 16` | `prev_norm > 5e-2` | ~80% of remaining bad basins |
| 3 | `j >= 32` | `prev_norm > 1e-3` | Final sweep; 100x gap between good/bad median residuals |

### 1.4 Dispatch Conditions (Rearranged Dispatch)

| Condition | Threshold | Purpose |
|-----------|-----------|---------|
| `gap_t < 0.02` | Target on degenerate Weyl face | Gate the entire dispatch block |
| `basis_on_c1c2` | `abs(c1-c2) < 0.02` AND `min(c1,c2) > 0.05` | Only fire for iSwap-family basis gates |
| `gap_c > 2*gap_t + 0.01` | Prefix is generic | Ensure rearranged formulation targets an easy invariant |

### 1.5 LP Objective Tiebreaker

| Parameter | Value | Mechanism |
|-----------|-------|-----------|
| m0-prefer eps | `1e-12` | `c[::3] -= 1e-12` breaks co-optimal LP vertices toward c1 > c2 |

### 1.6 PRNG Retry

| Parameter | Value | Mechanism |
|-----------|-------|-----------|
| Retry key | `fold_in(key, step ^ 0xCAFE)` | Diversified key when both acceptance criteria fail |

### 1.7 Stitching Architecture

The pipeline decomposes a target unitary into a sequence of segments, each solved
independently.  The stitching strategy determines how errors accumulate and are corrected.

| Choice | Description | Status |
|--------|-------------|--------|
| **Recover after every segment** | After each segment solve, run `recover_local_equivalence(C_i, P)` to extract k3, k4 and snap the accumulated product back toward the canonical intermediate. Final recovery absorbs all remaining drift. | **Shipped.** |
| Feedforward (no intermediate recovery) | Let errors accumulate freely; only correct at the end. | Not tested — would compound Makhlin→Weyl quadratic error across all segments, likely hitting branch ambiguity at a≈π/4. |
| Recover every k steps | Amortize recovery overhead by correcting every k segments. | Not tested — recovery is ~55μs/call, negligible vs solver cost. No motivation to skip. |

### 1.8 Segment Cache

| Parameter | Value | Mechanism |
|-----------|-------|-----------|
| `segment_cache_size` | `3` | LFU cache per step index. Identical (prefix, basis, target) triples reuse prior numeric solutions. |

### 1.9 LP Solver and Warm-Start

| Choice | Description | Status |
|--------|-------------|--------|
| **Dual revised simplex (scipy)** | `DualRevisedSimplex` with warm-start across targets for the same ISA. Process-global `_solver_cache`. | **Shipped.** |
| Warm-start basis reset | Reset dual basis on degenerate targets only (`abs(m[0]-m[1]) < 1e-3`). | Explored — targeted cold-start preserves warm-start for 95%+ of targets. |
| Blanket cold-start | Reset for every target. | -30% median regression. |
| HiGHS / interior point | Alternative LP backend. | Not tested — dual simplex warm-start is critical for amortizing solve cost across targets. |

### 1.10 Enumeration and Depth

| Parameter | Description | Status |
|-----------|-------------|--------|
| `max_sequence_length` | Maximum number of basis gates in a decomposition. ISA-specific (default 6, sq4cx needs 12). | **Shipped.** |
| `enumerate()` | Priority-queue sentence search in cost order. Tries sentences until LP is feasible. | **Shipped.** |
| `precompute_polytopes` | Pre-build monodromy polytope per sentence. Eliminates enumerate overhead but costs 3-7s upfront. | Available, not used by benchmarks. |

### 1.11 Weyl Coordinate Computation

The Weyl coordinates are computed via `U_tilde = (σ_y ⊗ σ_y) · U^T · (σ_y ⊗ σ_y)`, then
eigenvalues of `U · U_tilde`.  The `SYSY` prefactor is mathematically fixed (transpose in
the symplectic algebra), not tunable.

| Choice | Description | Status |
|--------|-------------|--------|
| **σ_y ⊗ σ_y (SYSY)** | Standard transpose-in-magic-basis for M = U^T U computation. | **Shipped.** Mathematical identity, not a free parameter. |
| Alternative numeric prefactors | Tested I⊗Z and similar Pauli products as alternative conjugations for M-matrix computation. | Did not change the algebraic structure — the Makhlin invariants are basis-independent. |

---

## 2. Parameters Explored and Rejected

### 2.1 Restart Budgets

| Variant | Values tested | Outcome |
|---------|--------------|---------|
| `makhlin_restarts` reduced | 256 → 64 | **Lost 5 tests.** Hard segments at c1=c2 need many restarts. |
| `weyl_restarts` reduced | 128 → 16 | **Lost 5 tests** (same commit as above). |
| `makhlin_maxiter` increased | 512 → 640, 768 | ISA-specific tradeoff: isa3 max -30%, but isa1 max +23%. Not universal. |
| `makhlin_maxiter` at 256 | 256 (from 512) | Halves the tail — 50% fewer restarts reach the oscillation floor. |

**Key finding:** Budget cuts are the #1 correctness killer.  The 256x512 budget is the minimum
that handles all ISAs.  Budget rebalancing (fewer restarts x more iters or vice versa) trades
between ISA types.

### 2.2 Stagnation / Early-Exit Heuristics

| Variant | Outcome |
|---------|---------|
| **Relative stagnation** (`prev_norm > init_n * 0.1`) | Self-defeating: bad basins improve 50x in first 32 iters, so relative condition never fires. |
| **Rate-based stagnation** (progress_ratio / stagnation_window) | Kills slow-but-steady convergence at rank-deficient faces. Lost 5 tests. |
| **Makhlin restart patience** (exit after 12 restarts without improvement) | Compounds with budget cuts. At rank-deficient faces, progress is erratic. |
| **GN inner-loop patience** (best_j tracking, checkpoint-cumulative) | Cuts iterations that produce useful oscillation minima captured by best-tracking. Regressed isa1 P95. |
| **Fourth stagnation tier** (j>=64) | Good/bad basin distributions overlap at iter 64. Any useful threshold kills good restarts. |
| **Soft inner-loop exit** (relax inner tol to 2x-10x) | Catastrophic: isa3 max 22→246ms at 2x. The 1e-9 inner tol IS the binding constraint at degenerate faces. |
| **2x-10x Weyl early-exit margin** | Correctness failures: accepts solutions on wrong Weyl branch at degenerate faces. 1e-5 is a correctness boundary. |

### 2.3 GN Damping and Step Control

| Variant | Values tested | Outcome |
|---------|--------------|---------|
| Tikhonov λ sweep | 1e-14, 1e-10, 1e-8, 1e-6, 1e-4 | **Monotonically worse** at degenerate faces. λ=1e-14 always best. Oscillation IS the mechanism. |
| LM adaptive damping (Stage 1) | Adaptive λ with monotonic descent | Stuck at 8.9e-8 vs GN's 2.3e-9. Monotonic descent prevents oscillation minima. |
| Higher constant damping | Systematic sweep | Uniformly degrades c1=c2 convergence. |

### 2.4 Init Range

| Range | ||q|| | First-step size | Outcome |
|-------|-------|-----------------|---------|
| ±0.05 | ~0.058 | Too conservative | Loses diversity, some outliers |
| **±0.1** | ~0.115 | **Sweet spot** | **Shipped.** -8% total, -42% isa3 max vs ±0.5. |
| ±0.15 | ~0.17 | OK | Hit an 85ms outlier (PRNG interaction) |
| ±0.25 | ~0.29 | Moderate | Overshoot on first GN step |
| ±0.5 | ~0.58 | Large | Old default. 5x larger first steps. |
| ±0.7, ±1.0 | >0.8 | Too large | Wasted restarts on bad basins |
| Spherical (Gaussian+normalize) | 1.0 | ||q||=1 optimal point | +37% regression. Different PRNG sequence loses key=0 advantage. |

### 2.5 LP Objective

| Variant | eps | Direction | Outcome |
|---------|-----|-----------|---------|
| Asymmetric perturbation | 1e-8 on m1/m2 | Steer away from c1=c2 | **Lost 2 tests.** Fights geometry for iSwap ISAs. |
| Anti-spread tiebreaker | 1e-12 | `[-1, -(1+2ε), -1]` per intermediate | Fixes sq4iswap (-95% outlier) but **catastrophic** on Weyl grid (+2700%). |
| **m0-prefer tiebreaker** | 1e-12 | `c[::3] -= 1e-12` | **Shipped.** -12% total, -59% isa1 max. Steers toward c1>c2. |
| Conditional LP re-solve | - | Re-solve on near-degenerate intermediates | Only 5 triggers / 192 targets. -3.7% total. Not worth complexity. |
| Blanket LP cold-start | - | Reset basis for every target | -30% median regression. Cold-start only needed for degenerate targets. |
| Per-segment key diversification | - | `fold_in(key, step)` per segment | +20% regression. key=0 is above-average for most ISAs. |

### 2.6 Parameterization

| Variant | Params | Outcome |
|---------|--------|---------|
| **Quaternion (q/||q||)** | 8 (4 per SU(2)) | **Shipped.** Natural step-size regularization, no gimbal lock. |
| Euler angle (ZYZ) | 6 (3 per SU(2)) | 30% hit rate vs 51% quaternion. Gimbal lock singularities. 2x slower per restart. |
| CAN-sandwich (algebraic) | Constructive R∈SO(4) | tr(M) linear in orthostochastic P[k,j]=R[k,j]², but tr(M²) has signed entries. Even if solvable, numpy 4×4 is ~1000x slower than JIT'd JAX. |

### 2.7 Residual Functions

| Residual | Dim | Jacobian rank at c1=c2 | Outcome |
|----------|-----|----------------------|---------|
| **Makhlin invariants** | R^8 → R^3 | Rank 2 (topological) | **Shipped.** Polynomial, smooth AD, fast per-step. |
| Weyl coordinates | R^8 → R^3 | Rank 3 (breaks symmetry) | 10-13x slower (eigvals+sort AD). |
| Augmented 4D (Makhlin + |c1-c2|) | R^8 → R^4 | Rank 3 | 3x per-step overhead. Catastrophic end-to-end regression. |
| Full M (Um^T @ Um) | R^8 → R^16 | Rank 5 | 0% convergence — target M not achievable for arbitrary segments. |
| M-diagonal (canonical) | R^8 → R^8 | Rank 5 | Fails on 73% of segments — canonical gauge K_R=I infeasible. |
| Polynomial gauge-fixing (R[i,j]) | R^8 → R^4 | Rank 3 | No single R[i,j] works universally across faces. |
| **Matrix matching (HS-norm)** | R^24 → R^32 | **Full rank 24** | Correct rank. But 0.5ms dispatch overhead negates advantage. Removed. |

### 2.8 Solver Architecture

| Variant | Outcome |
|---------|---------|
| **Random-restart GN + Weyl early-exit** | **Shipped.** Robust to all ISA types. |
| Two-stage (GN → full LM restart loop) | Stage 2 was dead code. Removed -92/+20 lines. |
| vmap vectorized restarts (256 parallel) | Only 3.7x speedup. 7.5x more work. Net regression. |
| Weyl-prefix init (N damped Weyl steps before GN) | 51→97% hit rate, but +400μs/restart overhead cancels savings. |
| Homotopy/continuation | Solution map is discontinuous at c1=c2. 100-4700ms warm-start. |
| Null-space line search on Makhlin manifold | 0/130 rescues. Curvature κ~70 overwhelms Weyl gradient. |
| Split inner loop (384+128 with mid-Weyl check) | Neutral. XLA graph bloat offsets savings. |
| lax.cond mid-restart Weyl checkpoint | +21% from XLA graph bloat inside while_loop. |
| jacfwd vs jacrev | jacrev is ~10% faster in lax.while_loop (R^8→R^3 has fewer outputs). |
| VJP fusion (explicit vjp to avoid duplicate forward pass) | +13% slower. XLA already CSEs the duplicate. |
| Joint multi-segment optimization | **Untested.** Architecturally correct but likely regresses 99% easy case. |
| Smooth sorting network (Weyl polish) | Differentiable `_smooth_minmax` with ε=1e-10. Zero improvement — `jnp.sort` gradients are fine during optimization. |
| **Makhlin acceptance fallback** | **Shipped.** Accept on `makhlin_res ≤ tol` when Weyl polish fails. Primary path at degenerate faces (~33-67% of good restarts). |
| **GN best-tracking** | **Shipped.** Track `best_norm`/`best_params` across all iterations per restart. P95 reduced 60-76%. |
| **Inner-loop best_norm exit** | **Shipped.** Check `best_norm > tol` instead of `prev_norm > tol`. Exits ~50-100 iters earlier at degenerate faces. |

### 2.9 Dispatch / Rearrangement

| Variant | Outcome |
|---------|---------|
| **Rearranged dispatch (with basis gate face check)** | **Shipped.** isa3 max 25→7.4ms. No regressions when gated on c1=c2 basis gate. |
| Rearranged dispatch (without basis gate check) | **isa4 regression 6→19ms.** False positive: CX-family basis gates don't trigger Makhlin obstruction. |
| Matrix matching as pre-dispatch | Zero benefit over rearranged. +670ms JIT compilation. Removed. |
| Matrix matching as fallback | Not tested — rearranged dispatch already handles all cases where Makhlin struggles. |
| Target perturbation (ε off Weyl wall) | Per-segment speedup real (42-46x). But +150% total as pre-dispatch. Neutral as fallback (standard solve never fails). |
| Analytic Kronecker SVD hint (restart 0 init) | Solves wrong problem (unitary equality vs invariant matching). Zero perf impact. Removed. |

---

## 3. Structural Constants (Not Tunable)

These are mathematical properties of the problem, not free parameters.

| Property | Value | Implication |
|----------|-------|-------------|
| Makhlin Jacobian rank at c1=c2 | 2 (topological, Chevalley) | Cannot be fixed by any polynomial residual augmentation |
| Makhlin→Weyl sensitivity at c1=c2 | Quadratic: `makh ≈ C * weyl²`, C ∈ [6.75, 140] | 1e-9 Makhlin maps to ~1e-5 Weyl |
| Double coset fiber dimension at c1=c2 | 6 (enhanced SO(2) isotropy) | Any double-coset residual has rank ≤ 8-6 = 2 |
| Algebraic degree of Makhlin system | 128 (all non-degenerate ISAs) | Difficulty is differential-geometric, not algebraic |
| Good basin hit rate (hardest segments) | 5.5% (cx^(1/3)+cx^(1/4)) to 54% (iSwap ISAs) | Determines restart budget needed |
| Good restart inner-loop cost | ~512 × 55μs ≈ 28ms | The binding constraint for tail timing |
| Bad restart with stagnation abort | ~8 × 55μs ≈ 0.4ms | Stagnation abort makes failures cheap |
| Python→JAX dispatch overhead | ~0.5ms per JIT'd function call | Fixed cost, not reducible |
| Qiskit TwoQubitWeylDecomposition | ~7.8μs/call | Recovery is not a bottleneck |
| `recover_local_equivalence` | ~55μs/call | Recovery is not a bottleneck |

---

## 4. Optimization Timeline

| Phase | Key change | Effect |
|-------|-----------|--------|
| Pre-highspy | Flat at 8ms median, 150ms+ tail | — |
| Highspy refactor (a1a6ae7) | LP solver upgrade | 3x median speedup, 4x tail reduction |
| JAX primitives (60a5bd6) | Move invariant computation to JAX | Mean 150→5ms, doubled failures |
| Knob-stacking era | +10 config params in one sprint | Config proliferation, correctness regression |
| Revert to 0401d6a | Strip all knobs | Best correctness, worst tail |
| GN best-tracking | Track oscillation minima | P95 reduced 60-76% |
| Weyl early-exit | Check Weyl coords in restart loop | Grand total 4.3→2.1s |
| maxiter 256→512 | Deeper inner loop | Grand total 2.1→0.9s |
| Three-tier stagnation abort | Absolute thresholds at j=8/16/32 | isa1 max 16→8ms, simple_speed isa2 max 53→8ms |
| Narrow init [-0.1, 0.1] | Smaller quaternion init range | -8% total, -42% isa3 max |
| m0-prefer tiebreaker | LP objective eps=1e-12 on m0 | -12% total, -59% isa1 max |
| Rearranged dispatch | Swap target↔prefix when target is degenerate | isa3 max 25→7.4ms |
| Basis gate face check | Gate dispatch on c1=c2 basis | Eliminated isa4 false positive (19→5.5ms) |
| Inner-loop best_norm check | Exit on best (not current) residual | ~50-100 fewer iters at degenerate faces |
| PRNG retry | Diversified key on failure | Catches rare PRNGKey(0) failures |
| Makhlin acceptance fallback | Accept on Makhlin when Weyl polish fails | Primary path at degenerate faces |
| Stage 2 removal | Delete dead LM restart loop | -92 lines, +10-20% median improvement |
