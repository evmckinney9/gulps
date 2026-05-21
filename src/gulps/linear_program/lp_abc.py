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

"""Base classes and types for LP constraint solvers."""

from dataclasses import dataclass
from typing import Protocol

from gulps.core.invariants import GateInvariants


@dataclass
class ConstraintSolution:
    """Unified result from any constraint solver (discrete or continuous).

    Attributes:
        success: Whether the LP/MILP found a feasible solution.
        sentence: Ordered list of gate invariants forming the decomposition.
            For discrete: the input sentence (fixed).
            For continuous: gates instantiated from k values.
        intermediates: Intermediate invariants [C_1, ..., C_n] representing
            the path through monodromy space: I -> C_1 -> ... -> C_n = target.
        parameters: Per-gate parameters. For continuous single-family ISAs,
            these are the k values where G_i = k_i * base. For discrete ISAs,
            these are the per-gate costs from the ISA's cost_dict.
        cost: Total decomposition cost (sum of parameters plus single-qubit
            overhead). Populated for both discrete and continuous paths.
    """

    success: bool
    sentence: tuple[GateInvariants, ...] | None = None
    intermediates: tuple[GateInvariants, ...] | None = None
    parameters: tuple[float, ...] | None = None
    cost: float | None = None


class ISAConstraints(Protocol):
    """Protocol for LP/MILP constraint solvers. Single public method: solve."""

    def solve(
        self, target: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Solve the LP/MILP for ``target``; pick the rho orientation internally."""
        ...
