"""Profile the decomposer to find hotspots."""

import cProfile
import pstats
import io
import numpy as np
from qiskit.quantum_info import random_unitary

from gulps.gulps_decomposer import GulpsDecomposer
from tests.fixtures.isas import get_slim_isas


def run_n(decomposer, N=200):
    for idx in range(N):
        u = random_unitary(4, seed=idx)
        decomposer(u)


def main():
    isas = get_slim_isas()
    # Profile just the first ISA for clarity
    isa = isas[0]
    name = "+".join(g.name for g in isa.gate_set)
    decomposer = GulpsDecomposer(isa=isa)
    # warmup
    decomposer(random_unitary(4, seed=9999))

    pr = cProfile.Profile()
    pr.enable()
    run_n(decomposer, N=200)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(60)
    print(f"\n=== CUMULATIVE (ISA: {name}) ===")
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(60)
    print(f"\n=== TOTAL TIME (ISA: {name}) ===")
    print(s2.getvalue())


if __name__ == "__main__":
    main()
