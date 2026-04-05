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

"""CPLEX-based MILP constraints for continuous ISA gate sequences.

Uses semicontinuous k variables (k ∈ {0} ∪ [k_lb, 1]) with binary depth
indicators for exact cost accounting, and a binary d_rho for automatic
rho-orientation selection. Monotonic ordering k₁ ≥ k₂ ≥ ... ensures
left-packing.

Gate parameterization: g_i = B * k_i where B = base.monodromy. This requires the
monodromy path m(k) = k*B to be linear through the origin. When eigenphase sorting
places gate^k on the rho-branch (e.g., SwapGate), the base gate is automatically
replaced with its canonical_matrix equivalent which stays on the principal branch.
"""

try:
    import docplex  # noqa: F401
except ModuleNotFoundError as exc:
    raise ImportError(
        "ContinuousISAConstraints requires the 'cplex' extra. "
        "Install with `pip install gulps[cplex]`."
    ) from exc

from typing import TYPE_CHECKING

import numpy as np
from docplex.mp.model import Model

from gulps import GateInvariants
from gulps.config import GulpsConfig
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

if TYPE_CHECKING:
    from gulps.core.isa import ContinuousISA


def _ensure_linear_monodromy(base: GateInvariants) -> GateInvariants:
    """Ensure base gate has linear monodromy path m(k) = k * B.

    Some gates (e.g., SwapGate) have eigenphase sorting that places gate^k
    on the rho-branch, making monodromy affine (not through origin). When
    detected, substitutes the canonical_matrix which is locally equivalent
    and stays on the principal branch.

    Returns the original base if linear, or a canonical substitute if not.
    """
    B = base.monodromy
    test_k = 0.5

    actual_mono = GateInvariants.from_unitary(base.gate.power(test_k)).monodromy
    if np.max(np.abs(actual_mono - B * test_k)) < 0.01:
        return base

    canon = GateInvariants.from_unitary(base.canonical_matrix, name=base.name)
    canon_mono = GateInvariants.from_unitary(canon.gate.power(test_k)).monodromy
    if np.max(np.abs(canon_mono - canon.monodromy * test_k)) < 0.01:
        return canon

    raise ValueError(
        f"Gate '{base.name}' has non-linear monodromy path and its canonical "
        f"matrix does not fix it. Cannot use with continuous ISA."
    )


class ContinuousISAConstraints(ISAConstraints):
    """MILP constraints for a single continuous ISA gate family.

    Semicontinuous k variables with binary depth indicators and rho selector.
    Gate monodromy g_i = B * k_i is substituted directly into QLR constraints
    (no auxiliary gi variables). Model is built once and reused — only the
    target RHS changes per solve.
    """

    _ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

    def __init__(
        self,
        base: GateInvariants,
        max_sequence_length: int = 3,
        offset: float = 1e-8,
        k_lb: float = 0.01,
        single_qubit_cost: float = 1e-8,
        config: GulpsConfig | None = None,
    ):
        """Build the CPLEX model for a continuous single-family ISA."""
        self.N = max_sequence_length
        if self.N < 2:
            raise ValueError("sequence_length must be at least 2")
        self.base = _ensure_linear_monodromy(base)
        self.B = self.base.monodromy
        self.offset = offset
        self.k_lb = 0.001 if k_lb < 0.0 else k_lb
        self.single_qubit_cost = single_qubit_cost
        self.config = config or GulpsConfig()

        # Precompute QLR dot products with B for gate substitution:
        # _gi_block[r] @ (B * k_i) = (_gi_block[r] @ B) * k_i
        self._gi_dot_B = self._gi_block @ self.B  # shape (72,)
        # First intermediate c₁ = g₁ = B * k₁, so _ci_block[r] @ c₁ = (_ci_block[r] @ B) * k₁
        self._ci_dot_B = self._ci_block @ self.B  # shape (72,)

        self.model = self._create_model()
        self._target_cts: list = []

    def set_target(self, target: GateInvariants) -> None:
        """Set the target gate invariants for the constraint RHS.

        Uses a binary variable d_rho to select between direct and rho-reflected
        orientations in a single solve: c_N = T_rho + d_rho * (T - T_rho).
        When d_rho=1 → c_N = T (direct). When d_rho=0 → c_N = T_rho.
        """
        t_direct = np.asarray(target.monodromy[:3], dtype=float)
        t_rho = np.asarray(target.rho_reflect.monodromy[:3], dtype=float)
        delta = t_direct - t_rho

        if self._target_cts:
            self.model.remove_constraints(self._target_cts)
            self._target_cts.clear()

        self._target_cts = [
            self.model.add_constraint(
                self.ci_nested[-1][j] - self.d_rho * float(delta[j]) == float(t_rho[j])
            )
            for j in range(3)
        ]

    def solve(
        self, target: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Solve MILP with internal rho selection via binary d_rho variable."""
        self.set_target(target)
        return self.solve_single(log_output=log_output)

    def solve_single(self, log_output: bool = False) -> ConstraintSolution:
        """Solve the MILP once and extract results."""
        sol = self.model.solve(log_output=log_output)
        if not sol:
            return ConstraintSolution(success=False)

        ks = [float(sol.get_value(self.k_vars[i])) for i in range(self.N)]
        depth = sum(1 for k in ks if k > self.k_lb / 2)

        if depth == 0:
            return ConstraintSolution(
                success=True,
                sentence=(),
                intermediates=(),
                parameters=(),
                cost=self.single_qubit_cost,
            )

        cis = [
            np.array(
                [sol.get_value(self.ci_nested[i][j]) for j in range(3)], dtype=float
            )
            for i in range(self.N - 1)
        ]

        gi_list = [
            GateInvariants.from_unitary(self.base.gate.power(k), name=self.base.name)
            for k in ks[:depth]
        ]
        intermediate_invariants = (gi_list[0],) + tuple(
            GateInvariants(tuple(c)) for c in cis[: depth - 1]
        )

        return ConstraintSolution(
            success=True,
            sentence=tuple(gi_list),
            intermediates=intermediate_invariants,
            parameters=tuple(ks[:depth]),
            cost=sum(ks[:depth]) + self.single_qubit_cost * (depth + 1),
        )

    def _configure_model_params(self, m: Model) -> None:
        """Configure CPLEX solver parameters for optimal performance.

        Tuned via one-at-a-time sweep + combination validation across
        4 gate types (CX, iSWAP, SWAP, fSim) × 2 sqc values (0.1, 1.0).
        Key finding: presolve=0 is the biggest single win (~35% median
        reduction) — the model is small enough that presolve overhead
        exceeds its benefit. Disabling MIP heuristics, RINS, cut passes,
        and probing removes ~10% more overhead.
        """
        m.parameters.threads = 1
        m.parameters.preprocessing.presolve = 0
        m.parameters.mip.limits.cutpasses = -1
        m.parameters.mip.strategy.heuristicfreq = -1
        m.parameters.mip.strategy.rinsheur = -1
        m.parameters.mip.strategy.probe = -1

    def _create_model(self):
        m = Model("ContinuousISAConstraints", ignore_names=True)
        self._configure_model_params(m)

        # STEP 1. VARIABLES
        # Semicontinuous: k ∈ {0} ∪ [k_lb, 1]. When k=0, gate is off (identity).
        self.k_vars = m.semicontinuous_var_list(self.N, lb=self.k_lb)
        # Binary rho selector: d_rho = 1 → direct, d_rho = 0 → rho-reflected.
        self.d_rho = m.binary_var()
        # Free intermediate monodromy nodes (no gi variables — substituted out).
        self.ci_nested = [
            m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N - 1)
        ]

        # Monotonic k ordering: k₁ ≥ k₂ ≥ ... ≥ k_N.
        for i in range(self.N - 1):
            m.add_constraint(self.k_vars[i] >= self.k_vars[i + 1])

        # STEP 2. QLR CONSTRAINTS with gate monodromy substituted inline.
        # Chain: c₁ = g₁ = B*k₁, then c₂, ..., c_{N-1} are free intermediates,
        # c_N = target (set per-solve). QLR block i (for gate i=2..N) connects
        # (c_{i-1}, g_i, c_i) where g_i = B*k_i.
        self._qlr_constraints(m)

        # STEP 3. OBJECTIVE FUNCTION
        # Per-gate cost is piecewise linear: f(k) = 0 when k=0, f(k) = k + sqc
        # when k ≥ k_lb. The step at k=0 absorbs the single_qubit_cost per
        # active gate, eliminating the need for binary depth indicators (d_vars).
        sqc = self.single_qubit_cost
        pwl = m.piecewise(0, [(0, 0), (0, sqc)], 1)
        # Primary: total cost = Σ f(k_i) + sqc (initial 1q layer).
        obj_cost = m.sum(pwl(self.k_vars[i]) for i in range(self.N)) + sqc
        # Secondary: maximize Σ intermediate monodromy (push toward polytope
        # interior, away from degenerate Weyl faces).
        obj_interior = m.sum(
            self.ci_nested[i][j] for i in range(self.N - 1) for j in range(3)
        )

        m.set_multi_objective(
            sense="min",
            exprs=[obj_cost, -obj_interior],
            priorities=[2, 1],
            abstols=[self.offset, 0],
        )

        return m

    def _qlr_constraints(self, m: Model):
        """Add QLR feasibility constraints with g_i = B*k_i substituted inline.

        For each QLR block i (gate i=2..N):
          _ci_block[r] @ c_{i-1} + _gi_block[r] @ (B*k_i) + _ciplus1_block[r] @ c_i ≤ _bi[r]

        When i=2, c_{i-1} = c₁ = g₁ = B*k₁, so _ci_block[r] @ c₁ = (_ci_block @ B)[r] * k₁.
        The gate term _gi_block[r] @ (B*k_i) = (_gi_block @ B)[r] * k_i.
        Both are scalar × k, eliminating the need for gi auxiliary variables.
        """
        for i in range(2, self.N + 1):
            c_i = self.ci_nested[i - 2]  # current intermediate
            k_gate = self.k_vars[i - 1]  # gate i's parameter

            for r in range(len_qlr):
                # Gate contribution: _gi_block[r] @ (B * k_i) = _gi_dot_B[r] * k_i
                expr = float(self._gi_dot_B[r]) * k_gate

                # Previous node contribution
                if i == 2:
                    # c₁ = g₁ = B*k₁ → _ci_block[r] @ (B*k₁) = _ci_dot_B[r] * k₁
                    expr += float(self._ci_dot_B[r]) * self.k_vars[0]
                else:
                    expr += m.scal_prod(self.ci_nested[i - 3], self._ci_block[r])

                # Current node contribution
                expr += m.scal_prod(c_i, self._ciplus1_block[r])

                m.add_constraint(expr <= float(self._bi[r]))
