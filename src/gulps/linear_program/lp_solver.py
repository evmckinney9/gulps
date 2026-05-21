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

"""LP feasibility for ordered gate sentences via QLR monodromy polytope.

Block-tridiagonal constraint matrix ``A`` stacks QLR blocks (72 rows, 3
columns per intermediate invariant) for each adjacent gate pair. The first
and last invariants are fixed by the first gate and the target; the n-2
interior invariants are LP variables. Gate-and-target data enters via the
RHS ``b``, rebuilt per solve. A custom dual simplex starts from an
identity basis (standard-basis-vector constraints) for dual feasibility +
fast repeated solves.
"""

from __future__ import annotations

import numpy as np

from gulps._accelerate import DualSimplex
from gulps.config import GulpsConfig
from gulps.core.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

_ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities
_d = LEN_GATE_INVARIANTS  # 3, dimension of each stage's invariant vector


def _build_constraint_matrix(n_gates: int) -> np.ndarray:
    """Assemble block-tridiagonal *A* for a sentence of n_gates gates.

    Each of the ``n_gates - 1`` QLR blocks contributes 72 rows.
    Block i places ``_ci_block`` on stage ``i-1`` and ``_ciplus1_block``
    on stage ``i``.
    """
    n_rows = len_qlr * (n_gates - 1)
    n_cols = _d * (n_gates - 2)
    A = np.zeros((n_rows, n_cols))
    for i in range(n_gates - 1):
        rows = slice(len_qlr * i, len_qlr * (i + 1))
        if i > 0:
            A[rows, _d * (i - 1) : _d * i] += _ci_block
        if i < n_gates - 2:
            A[rows, _d * i : _d * (i + 1)] += _ciplus1_block
    return np.ascontiguousarray(A, dtype=np.float64)


def _identity_row_indices(block: np.ndarray) -> np.ndarray:
    """Find rows in block equal to each standard basis vector e_j."""
    d = block.shape[1]
    indices = np.empty(d, dtype=np.intp)
    for j in range(d):
        ej = np.zeros(d)
        ej[j] = 1.0
        (matches,) = np.where(np.all(block == ej, axis=1))
        if len(matches) == 0:
            raise ValueError(f"No identity row e_{j} in QLR block")
        indices[j] = matches[0]
    return indices


# Indices of identity rows in each QLR block (used for cold start basis)
_ci_identity_rows = _identity_row_indices(_ci_block)
_cip_identity_rows = _identity_row_indices(_ciplus1_block)


def _build_cold_start_basis(n_stages: int) -> np.ndarray:
    """Build an initial dual-feasible basis from the staircase structure."""
    basis = np.empty(_d * n_stages, dtype=np.intp)
    basis[:_d] = _cip_identity_rows
    for k in range(1, n_stages):
        block_start = (k + 1) * len_qlr
        basis[k * _d : (k + 1) * _d] = block_start + _ci_identity_rows
    return basis


class LPSolverCache:
    """Cache of dual-simplex solvers keyed by ``(n_gates, tol)``.

    The constraint matrix A depends only on sentence length, so solvers
    are safely shared across ISAs and targets. Per-call data enters via
    the RHS, rebuilt in MinimalOrderedISAConstraints.
    """

    def __init__(self) -> None:
        """Empty cache; entries are built lazily by :meth:`get`."""
        self._cache: dict[tuple[int, float], DualSimplex] = {}

    def get(self, n_gates: int, tol: float) -> DualSimplex:
        """Get or create a cached solver for sentence length *n_gates*."""
        key = (n_gates, tol)
        if key not in self._cache:
            A = _build_constraint_matrix(n_gates)
            n_stages = n_gates - 2
            # Non-zero objective is required: c=0 causes degenerate cycling
            # in the dual simplex (all dual ratios become 0 → arbitrary pivots
            # → exhausts MAX_PIVOTS on the QLR system). Maximizing
            # Σ monodromy also steers intermediates toward polytope interior.
            c = -np.ones(A.shape[1])
            basis = _build_cold_start_basis(n_stages)
            self._cache[key] = DualSimplex(A, c, basis.tolist(), tol)
        return self._cache[key]


_IDENTITY_GATE = GateInvariants((0, 0, 0, 0), name="I")


class MinimalOrderedISAConstraints(ISAConstraints):
    """LP feasibility checker for an ordered, discrete gate sentence."""

    def __init__(
        self,
        isa_sequence: list[GateInvariants],
        config: GulpsConfig | None = None,
        solver_cache: LPSolverCache | None = None,
        *,
        parameters: tuple[float, ...] | None = None,
        single_qubit_cost: float = 0.0,
    ) -> None:
        """Build the constraint matrix and cache cost metadata.

        ``parameters`` and ``single_qubit_cost`` are sentence-fixed and
        written into every successful :class:`ConstraintSolution` returned.
        """
        self._config = config or GulpsConfig()
        self._sentence = tuple(isa_sequence)
        self._solver_cache = solver_cache or LPSolverCache()
        self._parameters = parameters
        self._cost = (
            sum(parameters) + (len(self._sentence) + 1) * single_qubit_cost
            if parameters is not None
            else None
        )

        # Pad 1-gate sentences so the QLR block math stays uniform.
        if len(isa_sequence) == 1:
            isa_sequence = [*isa_sequence, _IDENTITY_GATE]

        n = len(isa_sequence)
        self._sequence = isa_sequence
        tol = self._config.lp_feasibility_tol
        self._solver = self._solver_cache.get(n, tol) if n > 2 else None
        self._b = self._build_base_rhs(isa_sequence)
        self._last_target_contrib = np.zeros(len_qlr)

    @staticmethod
    def _build_base_rhs(sequence: list[GateInvariants]) -> np.ndarray:
        """Build b vector from fixed gate monodromy values (target-independent)."""
        n = len(sequence)
        b = np.empty(len_qlr * (n - 1))
        for i in range(n - 1):
            gi_contrib = _gi_block @ sequence[i + 1].monodromy
            if i == 0:
                gi_contrib += _ci_block @ sequence[0].monodromy
            b[len_qlr * i : len_qlr * (i + 1)] = _bi - gi_contrib
        return b

    def solve(
        self,
        target: GateInvariants,
        log_output: bool = False,
    ) -> ConstraintSolution:
        """Try both rho orientations against this sentence; most-slack first."""
        ct_direct = _ciplus1_block @ target.monodromy
        ct_rho = _ciplus1_block @ target.rho_reflect.monodromy

        base_tail = self._b[-len_qlr:] + self._last_target_contrib
        if (base_tail - ct_direct).min() >= (base_tail - ct_rho).min():
            first, first_ct = target, ct_direct
            second, second_ct = target.rho_reflect, ct_rho
        else:
            first, first_ct = target.rho_reflect, ct_rho
            second, second_ct = target, ct_direct

        self._apply_target(first, first_ct)
        result = self._solve_current()
        if result.success:
            return result
        self._apply_target(second, second_ct)
        return self._solve_current()

    def _apply_target(self, target: GateInvariants, ct: np.ndarray) -> None:
        """Incremental RHS update for a new target.

        ``ct = _ciplus1_block @ target.monodromy`` is passed in by :meth:`solve`,
        which already computed it for the orientation comparison. The
        ``_b[-72:] += _last_target_contrib - ct`` delta reverses the previous
        target's contribution instead of rebuilding ``_b`` from scratch.
        """
        self._target = target
        self._b[-len_qlr:] += self._last_target_contrib - ct
        self._last_target_contrib = ct

    def _solve_current(self) -> ConstraintSolution:
        """Run the LP on the current ``_b``, return the ConstraintSolution."""
        slack_tol = -10 * self._config.lp_feasibility_tol

        # 1- or 2-gate: no free variables, pure feasibility check.
        if self._solver is None:
            if not np.all(slack_tol <= self._b):
                return ConstraintSolution(success=False)
            # 1-gate intermediates are just the gate itself (gate is target);
            # 2-gate intermediates are (first gate, target).
            if len(self._sentence) == 1:
                intermediates = (self._sequence[0],)
            else:
                intermediates = (self._sequence[0], self._target)
            return ConstraintSolution(
                success=True,
                sentence=self._sentence,
                intermediates=intermediates,
                parameters=self._parameters,
                cost=self._cost,
            )

        # 3+ gate: delegate to dual simplex.
        x, feasible = self._solver.solve(self._b)
        if not feasible:
            return ConstraintSolution(success=False)

        lp_intermediates = [
            GateInvariants(tuple(x[i : i + _d])) for i in range(0, len(x), _d)
        ]

        return ConstraintSolution(
            success=True,
            sentence=self._sentence,
            intermediates=(self._sequence[0], *lp_intermediates, self._target),
            parameters=self._parameters,
            cost=self._cost,
        )
