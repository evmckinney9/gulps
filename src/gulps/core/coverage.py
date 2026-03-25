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

"""Weyl-chamber coverage computation and sampling for ISA gate sets.

This module requires the optional ``monodromy`` package.  All heavy imports
are guarded so that the rest of GULPS can be used without it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from numpy import ndarray

from gulps.viz.weyl_chamber import weyl_linspace  # noqa: F401  — re-export for compat

try:
    from monodromy.coordinates import unitary_to_monodromy_coordinate
    from monodromy.coverage import (
        CircuitPolytope,
        build_coverage_set,
        deduce_qlr_consequences,
    )
    from monodromy.haar import distance_polynomial_integrals
    from monodromy.static.examples import (
        everything_polytope,
        exactly,
        identity_polytope,
    )
except ModuleNotFoundError as _exc:
    raise ImportError(
        "Coverage computation requires the 'monodromy' package. "
        "Install with: pip install -r requirements-monodromy.txt"
    ) from _exc

if TYPE_CHECKING:
    from gulps.core.isa import ISAInvariants


def _operation_to_circuit_polytope(
    unitary: ndarray,
    op_name: str,
    cost: float,
    single_qubit_cost: float = 0.0,
) -> CircuitPolytope:
    b_polytope = exactly(
        *(
            Fraction(x).limit_denominator(10_000)
            for x in unitary_to_monodromy_coordinate(unitary)[:-1]
        )
    )
    convex_polytope = deduce_qlr_consequences(
        target="c",
        a_polytope=identity_polytope,
        b_polytope=b_polytope,
        c_polytope=everything_polytope,
    )

    return CircuitPolytope(
        operations=[op_name],
        cost=cost + single_qubit_cost,
        convex_subpolytopes=convex_polytope.convex_subpolytopes,
    )


def isa_to_coverage(
    isa: "ISAInvariants",
    sort=True,
) -> list[CircuitPolytope]:
    """Calculates coverage given a basis gate set."""
    unitaries = [g.matrix for g in isa.gate_set]
    costs = [isa.cost_dict[g] for g in isa.gate_set]
    names = [f"{g.name}_{i}" for i, g in enumerate(isa.gate_set)]
    single_qubit_cost = isa.single_qubit_cost

    operations = [
        _operation_to_circuit_polytope(
            unitary=u, op_name=n, cost=c, single_qubit_cost=single_qubit_cost
        )
        for u, n, c in zip(unitaries, names, costs)
    ]
    coverage_set = build_coverage_set(operations)

    # NOTE slightly hacky modification to avoid modifying build_coverage_set
    name_to_instruction = {n: g for n, g in zip(names, isa.gate_set)}
    for polytope in coverage_set:
        # first, for each polytope, we need to attach instruction metadata
        instructions = [name_to_instruction[op_name] for op_name in polytope.operations]
        polytope.instructions = instructions
        # second, a bit pedantic but we can fix the off by-one in cost here
        polytope.cost += single_qubit_cost

    if sort:
        return sorted(coverage_set, key=lambda k: k.cost)

    return coverage_set


def compute_coverage_statistics(coverage_set, chatty=False):
    """Compute coverage statistics without plotting or printing.

    Args:
        coverage_set: List of CircuitPolytope objects.
        chatty: Whether to print verbose output during integral calculation.

    Returns:
        dict: Coverage analysis containing volume_info, expected_cost,
            expected_depth, expected_index, and total_coverage.
    """
    # prune coverage_set with empty operations
    coverage_set = [p for p in coverage_set if p.operations]

    # Calculate integrals once (expensive operation)
    integrals = distance_polynomial_integrals(coverage_set, chatty=chatty)

    # Initialize accumulators
    expected_cost = 0.0
    expected_depth = 0.0
    expected_index = 0.0
    total_coverage = 0.0
    cumulative_sum = 0.0

    # Build list of (cost, depth, volume, cumulative_vol) for each polytope
    volume_info = []

    # Single pass through coverage_set to compute everything
    for i, polytope in enumerate(coverage_set):
        cost = polytope.cost
        depth = len(polytope.instructions)
        haar_vol = integrals[tuple(polytope.operations)][0]

        # Accumulate for expected values
        expected_cost += polytope.cost * haar_vol
        expected_depth += depth * haar_vol
        expected_index += i * haar_vol
        total_coverage += haar_vol

        # Track cumulative and add to volume_info (skip zero-cost)
        if cost == 0:
            continue
        cumulative_sum += haar_vol
        volume_info.append((cost, depth, haar_vol, cumulative_sum))

    return {
        "volume_info": volume_info,
        "expected_cost": expected_cost,
        "expected_depth": expected_depth,
        "expected_index": expected_index,
        "total_coverage": total_coverage,
    }


def coverage_report(coverage_set, chatty=False):
    """Analyze, plot, and print coverage statistics.

    Args:
        coverage_set: List of CircuitPolytope objects.
        chatty: Whether to print verbose output during integral calculation.

    Returns:
        dict: Coverage analysis containing volume_info, expected_cost,
            expected_depth, expected_index, and total_coverage.
    """
    from gulps.viz.polytope_viz import plot_coverage_set

    report = compute_coverage_statistics(coverage_set, chatty=chatty)

    # Plot
    plot_coverage_set(coverage_set, volume_info=report["volume_info"])

    # Print
    print("=" * 60)
    print("Coverage Set Statistics (Haar-averaged over SU(4))")
    print("=" * 60)
    print(f"Expected Cost:  {report['expected_cost']:.6f}")
    print(f"  Average cost per random 2-qubit unitary")
    print()
    print(f"Expected Depth: {report['expected_depth']:.6f}")
    print(f"  Average number of 2-qubit gates")
    print("=" * 60)

    return report
