"""Profile tail-case decompositions — what makes the worst cases so slow?

Identifies the slowest decompositions and breaks down where time goes.
"""

import time

import numpy as np
from qiskit.quantum_info.random import random_unitary

from gulps.gulps_decomposer import GulpsDecomposer
from tests.fixtures.isas import get_slim_isas

N = 1000

isas = get_slim_isas()

for isa in isas:
    print(f"\n{'=' * 60}")
    print(f"ISA: {isa}")
    decomposer = GulpsDecomposer(isa=isa)
    decomposer(random_unitary(4, seed=9999))  # warmup

    unitaries = [random_unitary(4, seed=i) for i in range(N)]

    results = []
    for i, u in enumerate(unitaries):
        decomposer(u)
        t = decomposer.last_timing
        results.append(
            {
                "seed": i,
                "total": t.get("total", 0) * 1000,
                "lp": t.get("lp_sentence", 0) * 1000,
                "segments": t.get("segments", 0) * 1000,
            }
        )

    totals = np.array([r["total"] for r in results])
    lps = np.array([r["lp"] for r in results])
    segs = np.array([r["segments"] for r in results])

    print(f"\n  Distribution (ms):")
    for label, arr in [("total", totals), ("lp", lps), ("segments", segs)]:
        print(
            f"    {label:>10s}: p50={np.median(arr):6.1f}  p90={np.percentile(arr, 90):6.1f}  "
            f"p95={np.percentile(arr, 95):6.1f}  p99={np.percentile(arr, 99):6.1f}  "
            f"max={np.max(arr):7.1f}"
        )

    # Analyze the worst 1%
    threshold = np.percentile(totals, 99)
    tail_indices = [i for i, r in enumerate(results) if r["total"] >= threshold]

    print(f"\n  Worst {len(tail_indices)} cases (>= p99 = {threshold:.1f}ms):")
    for idx in sorted(tail_indices, key=lambda i: -results[i]["total"])[:10]:
        r = results[idx]
        print(
            f"    seed={r['seed']:>4d}  total={r['total']:7.1f}  "
            f"lp={r['lp']:7.1f}  segments={r['segments']:7.1f}"
        )

    # What fraction of tail time is LP vs segments?
    tail_lp = np.mean([results[i]["lp"] for i in tail_indices])
    tail_seg = np.mean([results[i]["segments"] for i in tail_indices])
    print(
        f"\n  Tail mean: lp={tail_lp:.1f}ms ({100 * tail_lp / (tail_lp + tail_seg):.0f}%) "
        f"segments={tail_seg:.1f}ms ({100 * tail_seg / (tail_lp + tail_seg):.0f}%)"
    )

    # Also check: are tail cases correlated with sentence length?
    print(f"\n  Checking sentence length correlation...")
    slow_seeds = [results[i]["seed"] for i in tail_indices]
    fast_indices = [i for i, r in enumerate(results) if r["total"] <= np.median(totals)]
    fast_seeds = [results[i]["seed"] for i in fast_indices[:50]]

    # Re-run a few slow and fast to get sentence info
    slow_lens = []
    for seed in slow_seeds[:10]:
        u = random_unitary(4, seed=seed)
        circ = decomposer(u)
        n2q = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
        slow_lens.append(n2q)

    fast_lens = []
    for seed in fast_seeds[:50]:
        u = random_unitary(4, seed=seed)
        circ = decomposer(u)
        n2q = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
        fast_lens.append(n2q)

    print(f"    Slow cases 2q gates: {slow_lens}")
    print(
        f"    Fast cases 2q gates (sample): mean={np.mean(fast_lens):.1f} "
        f"dist={dict(zip(*np.unique(fast_lens, return_counts=True)))}"
    )
