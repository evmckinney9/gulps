# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ISA configurations for benchmarking.

Three benchmark sets, each with a clear purpose:

1. ISA scaling: LP enumeration cost vs |gate_set| at roughly fixed depth.
2. Sentence length scaling: numeric segment time vs depth using single-gate ISAs.
3. Cross-decomposer comparisons: gulps vs nuop vs bsqkit on representative ISAs.

Usage:
    from gulps.comparisons.isa_library.benchmark_isas import get_isa_scaling, get_depth_scaling, get_comparison_isas
"""

import numpy as np
from qiskit.circuit.library import CXGate, iSwapGate

from gulps.comparisons.isa_library.fsim import fsim
from gulps.core.isa import DiscreteISA


def get_isa_scaling(max_sequence_length=18):
    """ISAs for measuring LP enumeration cost vs gate set size.

    Start from the strongest gate and progressively add weaker ones.
    Depth stays roughly pinned by the strongest gate. The dominant
    effect is that more gates = more infeasible sentences to reject
    before finding the first feasible one.

    Returns:
        List of (name, DiscreteISA) tuples.
    """
    configs = []

    # Progressive iSwap: {1/2}, {1/2,1/3}, ..., {1/2,...,1/8}
    iswap_denoms = [2, 3, 4, 5, 6, 7, 8]
    for n in range(1, len(iswap_denoms) + 1):
        denoms = iswap_denoms[:n]
        isa = DiscreteISA(
            gate_set=[iSwapGate().power(1 / d) for d in denoms],
            costs=[1 / d for d in denoms],
            names=[f"iswap^1/{d}" for d in denoms],
            max_sequence_length=max_sequence_length,
        )
        configs.append((f"iswap_prog_{n}g", isa))

    # Progressive CX: {1/2}, {1/2,1/3}, ..., {1/2,...,1/7}
    cx_denoms = [2, 3, 4, 5, 6, 7]
    for n in range(1, len(cx_denoms) + 1):
        denoms = cx_denoms[:n]
        isa = DiscreteISA(
            gate_set=[CXGate().power(1 / d) for d in denoms],
            costs=[1 / d for d in denoms],
            names=[f"cx^1/{d}" for d in denoms],
            max_sequence_length=max_sequence_length,
        )
        configs.append((f"cx_prog_{n}g", isa))

    # Progressive mixed: pair CX and iSwap at each fraction level
    mixed_denoms = [2, 3, 4, 5, 6]
    for n in range(1, len(mixed_denoms) + 1):
        denoms = mixed_denoms[:n]
        gates, costs, names = [], [], []
        for d in denoms:
            gates.extend([CXGate().power(1 / d), iSwapGate().power(1 / d)])
            costs.extend([1 / d, 1 / d])
            names.extend([f"cx^1/{d}", f"iswap^1/{d}"])
        isa = DiscreteISA(
            gate_set=gates,
            costs=costs,
            names=names,
            max_sequence_length=max_sequence_length,
        )
        configs.append((f"mixed_prog_{2 * n}g", isa))

    return configs


def get_depth_scaling(max_sequence_length=18):
    """ISAs for measuring numeric segment time vs sentence depth.

    Single-gate ISAs with shrinking fractional powers. As the gate gets
    weaker, the optimal sentence gets longer, so segment synthesis time
    should scale linearly with depth.

    Returns:
        List of (name, DiscreteISA) tuples.
    """
    configs = []

    for d in range(2, 9):
        isa = DiscreteISA(
            gate_set=[iSwapGate().power(1 / d)],
            costs=[1 / d],
            names=[f"iswap^1/{d}"],
            max_sequence_length=max_sequence_length,
        )
        configs.append((f"iswap^1/{d}", isa))

    for d in range(2, 9):
        isa = DiscreteISA(
            gate_set=[CXGate().power(1 / d)],
            costs=[1 / d],
            names=[f"cx^1/{d}"],
            max_sequence_length=max_sequence_length,
        )
        configs.append((f"cx^1/{d}", isa))

    return configs


def get_comparison_isas(max_sequence_length=8):
    """ISAs for cross-decomposer comparison (gulps vs nuop vs bsqkit).

    Three representative cases:
        1. Non-standard discrete gate (iswap^1/4) - tests basic non-CX support.
        2. Heterogeneous discrete ISA - multiple gate families, tests mixed-ISA handling.
        3. TODO FIXME Continuous fSim family - tests continuous parameter optimization.

    Returns:
        List of (name, DiscreteISA) tuples.
    """
    configs = []

    # (1) Non-standard fractional gate
    configs.append(
        (
            "iswap^1/4",
            DiscreteISA(
                gate_set=[iSwapGate().power(1 / 4)],
                costs=[0.25],
                names=["iswap^1/4"],
                max_sequence_length=max_sequence_length,
            ),
        )
    )

    # (2) Heterogeneous discrete ISA
    configs.append(
        (
            "hetero_4g",
            DiscreteISA(
                gate_set=[
                    CXGate(),
                    CXGate().power(1 / 2),
                    iSwapGate().power(1 / 2),
                    iSwapGate().power(1 / 3),
                ],
                costs=[1.0, 0.5, 0.5, 1 / 3],
                names=["cx", "cx^1/2", "iswap^1/2", "iswap^1/3"],
                max_sequence_length=max_sequence_length,
            ),
        )
    )

    # (3) Continuous fSim (discrete approximation for nuop/bsqkit comparison;
    #     gulps uses ContinuousISA.from_base_gate for the actual solve)
    configs.append(
        (
            "fsim_continuous",
            DiscreteISA(
                gate_set=[
                    fsim(np.pi / 2, np.pi / 6),
                    fsim(np.pi / 2, np.pi / 6).power(1 / 2),
                    fsim(np.pi / 2, np.pi / 6).power(1 / 3),
                ],
                costs=[1.0, 0.5, 1 / 3],
                names=["fsim", "fsim^1/2", "fsim^1/3"],
                max_sequence_length=max_sequence_length,
            ),
        )
    )

    return configs


def get_benchmark_isas():
    """Return all benchmark ISAs (union of all three sets).

    Returns:
        List of (name, DiscreteISA) tuples.
    """
    return get_isa_scaling() + get_depth_scaling() + get_comparison_isas()
