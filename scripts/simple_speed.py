import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    RZXGate,
    SwapGate,
    XXPlusYYGate,
    iSwapGate,
)
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.quantum_info.random import random_unitary
from tqdm import trange

from gulps import logger
from gulps.core.coverage import coverage_report
from gulps.core.invariants import GateInvariants
from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.isa_library.fsim import fsim
from gulps.viz.invariant_viz import plot_decomposition
from gulps.viz.report_viz import report_benchmark_results


def main():
    # build the sq4iswap decomposer
    isa = [
        (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    ]
    gate_set, costs, names = zip(*isa)
    isa = DiscreteISA(gate_set, costs, names, precompute_polytopes=False)
    decomposer = GulpsDecomposer(isa=isa)

    # JIT warmup
    decomposer(random_unitary(4, seed=9999))
    logger.setLevel("INFO")
    N = 1_000

    fidelities = []
    all_timings = []
    failures = 0

    for idx in range(N):
        u = random_unitary(4, seed=idx)
        try:
            fid = average_gate_fidelity(u, Operator(decomposer(u)))
            if fid < 1 - 1e-8:
                raise ValueError(f"Fidelity too low: {fid:.8f}")

            fidelities.append(fid)
            all_timings.append(decomposer.last_timing)

        except Exception as e:
            print(f"[WARN] Failed on unitary {idx}: {e}")
            failures += 1

    fidelities = np.array(fidelities)
    print(f"Benchmark: {len(fidelities)}/{N} successful ({failures} failures)")
    print(
        f"Fidelity: median={np.median(fidelities):.10f}, min={np.min(fidelities):.10f}"
    )
    print(
        f"Avg time: {np.mean([t['total'] for t in all_timings]) * 1000:.1f} ms/decomposition"
    )


if __name__ == "__main__":
    main()
