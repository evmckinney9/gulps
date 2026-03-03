"""Profile the time breakdown across the 3 benchmark ISAs."""

import time

import numpy as np
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.quantum_info.random import random_unitary

from gulps import logger
from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.isa_library.fsim import fsim

logger.setLevel("WARNING")

BENCHMARK_ISAS = {
    "sq4iswap": [
        (iSwapGate().power(1 / 4), 1 / 4, "isw^1/4"),
    ],
    "hetero_cx_isw": [
        (CXGate().power(1 / 2), 1 / 2, "cx^1/2"),
        (iSwapGate().power(1 / 3), 1 / 3, "isw^1/3"),
    ],
    "fsim_mix": [
        (fsim(np.pi / 2, np.pi / 6), 1.0, "fsim"),
        (iSwapGate().power(1 / 4), 1 / 4, "isw^1/4"),
    ],
}

N = 200

for name, isa_spec in BENCHMARK_ISAS.items():
    gate_set, costs, names = zip(*isa_spec)
    isa = DiscreteISA(gate_set, costs, names, precompute_polytopes=False)
    decomposer = GulpsDecomposer(isa=isa)

    # Warmup
    decomposer(random_unitary(4, seed=9999))

    lp_times = []
    seg_times = []
    total_times = []
    failures = 0

    for idx in range(N):
        u = random_unitary(4, seed=idx)
        try:
            decomposer(u)
            t = decomposer.last_timing
            lp_times.append(t["lp_sentence"])
            seg_times.append(t["segments"])
            total_times.append(t["total"])
        except Exception:
            failures += 1

    print(f"\n=== {name} ({', '.join(names)}) ===")
    print(f"  N={N}, failures={failures}")
    print(
        f"  total:    mean={np.mean(total_times) * 1000:.1f} ms  median={np.median(total_times) * 1000:.1f} ms"
    )
    print(
        f"  LP:       mean={np.mean(lp_times) * 1000:.1f} ms  median={np.median(lp_times) * 1000:.1f} ms  ({np.mean(lp_times) / np.mean(total_times) * 100:.0f}%)"
    )
    print(
        f"  segments: mean={np.mean(seg_times) * 1000:.1f} ms  median={np.median(seg_times) * 1000:.1f} ms  ({np.mean(seg_times) / np.mean(total_times) * 100:.0f}%)"
    )
