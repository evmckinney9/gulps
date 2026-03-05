"""QLR constraint builder for monodromy polytope feasibility.

Builds and solves the LP for QLR monodromy feasibility.

Given a sequence of gates, this module assembles the block-tridiagonal constraint matrix A and right-hand side b from the QLR inequality tables. The LP variables are the intermediate invariants between gates. The first and last invariants are fixed (by the first gate and the target).

The constraint matrix is built by stacking QLR blocks for each gate pair. Each block has 72 inequalities acting on two adjacent invariants. The resulting matrix is block-tridiagonal.

A custom dual simplex solver is used for speed. The initial basis is chosen by picking constraints that act on individual components (standard basis vectors) so the basis matrix is the identity. This guarantees dual feasibility and makes the solver fast for repeated solves.

Constraint structure:
- For n gates, there are n-2 free intermediate invariants (each a 3-vector).
- Each adjacent gate pair contributes a QLR block (72 rows, 3 columns per invariant).
- The right-hand side b absorbs the fixed gate monodromy contributions.
"""

from __future__ import annotations

import numpy as np

from gulps.config import GulpsConfig
from gulps.core.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.linear_program.dual_simplex import (
    DualRevisedSimplex,
    build_cold_start_basis,
    identity_row_indices,
)
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

_ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities
_d = LEN_GATE_INVARIANTS  # 3 — dimension of each stage's invariant vector


# --- Constraint matrix assembly ---


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


# --- Initial basis selection for simplex ---


# Indices of identity rows in each QLR block (used for cold start basis)
_ci_identity_rows = identity_row_indices(_ci_block)
_cip_identity_rows = identity_row_indices(_ciplus1_block)


def _build_cold_start_basis(n_stages: int) -> np.ndarray:
    return build_cold_start_basis(
        n_stages, _ci_identity_rows, _cip_identity_rows, len_qlr, _d
    )


# --- Solver cache ---

_solver_cache: dict[tuple[int, float], DualRevisedSimplex] = {}


def _get_solver(n_gates: int, tol: float, config: GulpsConfig) -> DualRevisedSimplex:
    """Return the shared dual-simplex solver for *n_gates* gates.

    One solver instance is cached per ``(n_gates, tol, objective)`` triple
    and re-used across all LP solves with the same sentence length.
    Warm-starting from the previous solve's basis is safe because the
    asymmetric objective below breaks LP degeneracy, making the
    optimum unique regardless of starting basis.

    Objective design (per stage, variables ``m0, m1, m2``):

    * **m1/m2 asymmetry** - steers intermediates toward ``m1 > m2``
      (equivalently ``c1 > c2``).  This avoids both the ``c1 = c2``
      Makhlin Jacobian rank-deficiency (Gauss-Newton stalls) and the
      ``c1 + c2 ~= 1`` boundary.
    * **m0 weight** - preserving full vertex-seeking behaviour which
      benefits deep sentences and caching.

    Only the objective vector changes; the constraint matrix is
    untouched, so the LP feasible region and gate-count optimality
    are preserved.
    """
    w0, w1, w2 = config.lp_objective_bias
    key = (n_gates, tol, w0, w1, w2)
    if key not in _solver_cache:
        A = _build_constraint_matrix(n_gates)
        n_cols = A.shape[1]
        n_stages = n_cols // _d
        c = np.empty(n_cols)
        for k in range(n_stages):
            c[_d * k + 0] = w0
            c[_d * k + 1] = w1
            c[_d * k + 2] = w2
        basis = _build_cold_start_basis(n_gates - 2)
        _solver_cache[key] = DualRevisedSimplex(
            A, c, basis, tol=tol, max_pivots=config.lp_max_pivots
        )
    return _solver_cache[key]


# --- Public interface ---

_IDENTITY_GATE = GateInvariants((0, 0, 0, 0), name="I")


class MinimalOrderedISAConstraints(ISAConstraints):
    """LP feasibility checker for ordered, discrete gate sentences.

    Parameters
    ----------
    isa_sequence : list[GateInvariants]
        Ordered gate invariants forming the sentence (length ≥ 1).
    config : GulpsConfig, optional
        Solver configuration (tolerance, etc.).
    """

    def __init__(
        self,
        isa_sequence: list[GateInvariants],
        config: GulpsConfig | None = None,
    ) -> None:
        """Initialise constraints for a fixed gate sentence."""
        self._config = config or GulpsConfig()
        self._sentence = tuple(isa_sequence)

        # Pad 1-gate sentences so the QLR block math stays uniform.
        if len(isa_sequence) == 1:
            isa_sequence = [*isa_sequence, _IDENTITY_GATE]

        n = len(isa_sequence)
        self._sequence = isa_sequence
        tol = self._config.lp_feasibility_tol
        self._solver = _get_solver(n, tol, self._config) if n > 2 else None
        self._b = self._build_base_rhs(isa_sequence)
        self._last_target_contrib = np.zeros(len_qlr)

    # ── RHS construction ──────────────────────────────────────────────

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

    def set_target(self, target: GateInvariants) -> None:
        """Update the last QLR block of b vector for a new target."""
        self._target = target
        ct = _ciplus1_block @ target.monodromy
        self._b[-len_qlr:] += self._last_target_contrib - ct
        self._last_target_contrib = ct

    # ── solve ─────────────────────────────────────────────────────────

    def solve(
        self,
        target: GateInvariants,
        log_output: bool = False,
    ) -> ConstraintSolution:
        """Try both rho orientations, most-slack first."""
        ct_direct = _ciplus1_block @ target.monodromy
        ct_rho = _ciplus1_block @ target.rho_reflect.monodromy

        base_tail = self._b[-len_qlr:] + self._last_target_contrib
        if np.min(base_tail - ct_direct) >= np.min(base_tail - ct_rho):
            first, second = target, target.rho_reflect
        else:
            first, second = target.rho_reflect, target

        self.set_target(first)
        result = self.solve_single()
        if result.success:
            return result

        self.set_target(second)
        return self.solve_single()

    def solve_single(self, log_output: bool = False) -> ConstraintSolution:
        """Solve the LP for the current b vector."""
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
        )
