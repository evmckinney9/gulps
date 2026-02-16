"""ISA configurations for benchmarking.

Three benchmark sets, each with a clear purpose:

1. ISA scaling: LP enumeration cost vs |gate_set| at roughly fixed depth.
2. Sentence length scaling: numeric segment time vs depth using single-gate ISAs.
3. Cross-decomposer comparisons: gulps vs nuop vs bsqkit on representative ISAs.

Usage:
    from gulps.isa_library.benchmark_isas import get_isa_scaling, get_depth_scaling, get_comparison_isas
"""

import numpy as np
from qiskit.circuit.library import CXGate, iSwapGate

from gulps.isa_library.fsim import fsim


def get_isa_scaling():
    """ISAs for measuring LP enumeration cost vs gate set size.

    Start from the strongest gate and progressively add weaker ones.
    Depth stays roughly pinned by the strongest gate. The dominant
    effect is that more gates = more infeasible sentences to reject
    before finding the first feasible one.

    Returns:
        List of (name, gate_set) tuples.
    """
    configs = []

    # Progressive iSwap: {1/2}, {1/2,1/3}, ..., {1/2,...,1/8}
    iswap_denoms = [2, 3, 4, 5, 6, 7, 8]
    for n in range(1, len(iswap_denoms) + 1):
        denoms = iswap_denoms[:n]
        gate_set = [
            (iSwapGate().power(1 / d), 1 / d, f"iswap^1/{d}")
            for d in denoms
        ]
        configs.append((f"iswap_prog_{n}g", gate_set))

    # Progressive CX: {1/2}, {1/2,1/3}, ..., {1/2,...,1/7}
    cx_denoms = [2, 3, 4, 5, 6, 7]
    for n in range(1, len(cx_denoms) + 1):
        denoms = cx_denoms[:n]
        gate_set = [
            (CXGate().power(1 / d), 1 / d, f"cx^1/{d}")
            for d in denoms
        ]
        configs.append((f"cx_prog_{n}g", gate_set))

    # Progressive mixed: pair CX and iSwap at each fraction level
    mixed_denoms = [2, 3, 4, 5, 6]
    for n in range(1, len(mixed_denoms) + 1):
        denoms = mixed_denoms[:n]
        gate_set = []
        for d in denoms:
            gate_set.append((CXGate().power(1 / d), 1 / d, f"cx^1/{d}"))
            gate_set.append((iSwapGate().power(1 / d), 1 / d, f"iswap^1/{d}"))
        configs.append((f"mixed_prog_{2 * n}g", gate_set))

    return configs


def get_depth_scaling():
    """ISAs for measuring numeric segment time vs sentence depth.

    Single-gate ISAs with shrinking fractional powers. As the gate gets
    weaker, the optimal sentence gets longer, so segment synthesis time
    should scale linearly with depth.

    Returns:
        List of (name, gate_set) tuples.
    """
    configs = []

    for d in range(2, 9):
        configs.append((
            f"iswap^1/{d}",
            [(iSwapGate().power(1 / d), 1 / d, f"iswap^1/{d}")],
        ))

    for d in range(2, 9):
        configs.append((
            f"cx^1/{d}",
            [(CXGate().power(1 / d), 1 / d, f"cx^1/{d}")],
        ))

    return configs


def get_comparison_isas():
    """ISAs for cross-decomposer comparison (gulps vs nuop vs bsqkit).

    Three representative cases:
        1. Non-standard discrete gate (iswap^1/4) — tests basic non-CX support.
        2. Heterogeneous discrete ISA — multiple gate families, tests mixed-ISA handling.
        3. Continuous fSim family — tests continuous parameter optimization.

    Returns:
        List of (name, gate_set) tuples.
    """
    configs = []

    # (1) Non-standard fractional gate
    configs.append((
        "iswap^1/4",
        [(iSwapGate().power(1 / 4), 0.25, "iswap^1/4")],
    ))

    # (2) Heterogeneous discrete ISA
    configs.append((
        "hetero_4g",
        [
            (CXGate(), 1.0, "cx"),
            (CXGate().power(1 / 2), 0.5, "cx^1/2"),
            (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
            (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
        ],
    ))

    # (3) Continuous fSim (discrete approximation for nuop/bsqkit comparison;
    #     gulps uses ContinuousISA.from_base_gate for the actual solve)
    configs.append((
        "fsim_continuous",
        [
            (fsim(np.pi / 2, np.pi / 6), 1.0, "fsim"),
            (fsim(np.pi / 2, np.pi / 6).power(1 / 2), 0.5, "fsim^1/2"),
            (fsim(np.pi / 2, np.pi / 6).power(1 / 3), 1 / 3, "fsim^1/3"),
        ],
    ))

    return configs


def get_benchmark_isas():
    """Return all benchmark ISAs (union of all three sets).

    Returns:
        List of (name, gate_set) tuples.
    """
    return get_isa_scaling() + get_depth_scaling() + get_comparison_isas()
