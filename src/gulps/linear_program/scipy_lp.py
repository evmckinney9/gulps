"""Staircase dual simplex for monodromy polytope feasibility.

Solves  ``min -1'x  s.t. Ax ≤ b``  where *A* has block-tridiagonal
(staircase) structure from the QLR inequalities.  Only the RHS vector *b*
depends on the specific gate sequence; *A* is fixed per sentence length.

Mathematical background
-----------------------
Each pair of adjacent gates ``(g_i, g_{i+1})`` contributes a QLR block::

    C_i · c_i  +  G_i · g_i  +  C_{i+1} · c_{i+1}  ≤  b_i

where ``c_i`` are free 3-D intermediate invariants and ``g_i`` are fixed
gate monodromy values.  Stacking gives a block-tridiagonal *A*::

    Block 0:   [ C_{i+1} |    0    |  ···  ]
    Block 1:   [   C_i   | C_{i+1} |  ···  ]
      ⋮
    Block K:   [  ···     |    0    |  C_i  ]

The solver exploits this via:

1. A **cold-start basis** from ``+e_j`` rows in the end-blocks → upper-
   block-triangular ``A_B`` with ``+I`` diagonal → guaranteed dual-feasible
   without Phase I.
2. **Warm-start** across solves: only *b* changes, so dual feasibility is
   preserved and typically 2–8 pivots suffice.
3. **Sherman–Morrison** rank-1 updates for ``B⁻¹`` (6×6 matrices).
"""

from __future__ import annotations

import numpy as np

from gulps.config import GulpsConfig
from gulps.core.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

_ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

# ── constants ─────────────────────────────────────────────────────────────

_MAX_PIVOTS = 50
"""Hard cap on simplex iterations per solve (should rarely exceed ~10)."""

_DEGEN_TOL = 1e-14
"""Threshold below which a Sherman–Morrison denominator triggers a full inv."""


def _find_plus_ej_rows(block: np.ndarray) -> np.ndarray:
    """Return row indices where ``block[row] == +e_j`` for j = 0 … d-1.

    These rows become the identity-diagonal entries of the staircase
    cold-start basis.
    """
    d = block.shape[1]
    indices = np.empty(d, dtype=np.intp)
    for j in range(d):
        ej = np.zeros(d)
        ej[j] = 1.0
        (matches,) = np.where(np.all(block == ej, axis=1))
        if len(matches) == 0:  # pragma: no cover
            msg = f"No +e_{j} row found in block – QLR tables may have changed"
            raise ValueError(msg)
        indices[j] = matches[0]
    return indices


# Derived once at import time from the QLR tables.
_CI_EJ: np.ndarray = _find_plus_ej_rows(_ci_block)
_CIP_EJ: np.ndarray = _find_plus_ej_rows(_ciplus1_block)

# ── staircase dual simplex ────────────────────────────────────────────────


class _StaircaseDualSimplex:
    """Dual revised simplex for the QLR staircase LP, cached per length.

    Parameters
    ----------
    n_gates : int
        Sentence length (≥ 3; shorter sentences don't need an LP).
    tol : float
        Primal feasibility tolerance.
    """

    _cache: dict[tuple[int, float], _StaircaseDualSimplex] = {}

    @classmethod
    def get(cls, n_gates: int, tol: float) -> _StaircaseDualSimplex:
        """Retrieve or create a solver for *n_gates* at the given tolerance."""
        key = (n_gates, tol)
        if key not in cls._cache:
            cls._cache[key] = cls(n_gates, tol)
        return cls._cache[key]

    def __init__(self, n_gates: int, tol: float) -> None:
        """Build the constraint matrix and cold-start basis."""
        n_stages = n_gates - 2
        n_vars = LEN_GATE_INVARIANTS * n_stages

        self._n_vars = n_vars
        self._tol = tol
        self._neg_c = np.ones(n_vars)  # -c  (c = -1)
        self._A = self._build_constraint_matrix(n_gates)
        self._basis = self._build_cold_start_basis(n_stages)

    # ── matrix assembly ───────────────────────────────────────────────

    @staticmethod
    def _build_constraint_matrix(n_gates: int) -> np.ndarray:
        """Assemble block-tridiagonal *A* from the QLR tables.

        Each of the ``n_gates - 1`` QLR blocks contributes ``len_qlr`` rows.
        Block *i* places ``_ci_block`` in columns for stage ``i-1`` and
        ``_ciplus1_block`` in columns for stage ``i``.
        """
        d = LEN_GATE_INVARIANTS
        n_rows = len_qlr * (n_gates - 1)
        n_cols = d * (n_gates - 2)
        A = np.zeros((n_rows, n_cols))
        for i in range(n_gates - 1):
            rows = slice(len_qlr * i, len_qlr * (i + 1))
            if i > 0:
                col = d * (i - 1)
                A[rows, col : col + d] += _ci_block
            if i < n_gates - 2:
                col = d * i
                A[rows, col : col + d] += _ciplus1_block
        return np.ascontiguousarray(A, dtype=np.float64)

    @staticmethod
    def _build_cold_start_basis(n_stages: int) -> np.ndarray:
        """Build an initial dual-feasible basis from the staircase structure.

        * Stage 0: ``+e_j`` rows from block 0 via ``_ciplus1_block``.
        * Stages 1…K-1: ``+e_j`` rows from block ``k+1`` via ``_ci_block``.

        The resulting ``A_B`` is upper-block-triangular with ``+I`` on the
        diagonal, so ``λ_B = -A_B^{-T} c ≥ 0`` is guaranteed (all dual
        variables non-negative).
        """
        d = LEN_GATE_INVARIANTS
        basis = np.empty(d * n_stages, dtype=np.intp)
        basis[:d] = _CIP_EJ  # stage 0 from block 0
        for k in range(1, n_stages):
            offset = (k + 1) * len_qlr
            basis[k * d : (k + 1) * d] = offset + _CI_EJ
        return basis

    # ── solve ─────────────────────────────────────────────────────────

    def solve(self, b: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Solve ``min c'x  s.t. Ax ≤ b``.  Returns ``(x, feasible)``.

        Warm-starts from the previous call's basis; since only *b* changes,
        dual feasibility carries over.
        """
        A = self._A
        neg_c = self._neg_c
        n = self._n_vars
        tol = self._tol
        basis = self._basis.copy()
        B_inv = np.linalg.inv(A[basis])

        for _ in range(_MAX_PIVOTS):
            x = B_inv @ b[basis]

            # Primal feasibility: max constraint violation
            violations = A @ x - b
            entering = int(np.argmax(violations))
            if violations[entering] <= tol:
                self._basis = basis
                return x, True

            # Entering direction in basis space and current dual multipliers
            tau = B_inv.T @ A[entering]
            lam = B_inv.T @ neg_c  # λ_B = -B⁻ᵀ c  (KKT: A'λ = -c)

            # Dual ratio test – find leaving variable to keep λ ≥ 0
            leaving = _dual_ratio_test(tau, lam, n, tol)
            if leaving < 0:
                self._basis = basis
                return None, False  # dual unbounded → primal infeasible

            # Sherman–Morrison rank-1 update of B⁻¹
            old_row = basis[leaving]
            delta = A[entering] - A[old_row]
            denom = 1.0 + delta @ B_inv[:, leaving]
            if abs(denom) < _DEGEN_TOL:
                # Near-degenerate pivot: fall back to full factorization
                basis[leaving] = entering
                B_inv = np.linalg.inv(A[basis])
            else:
                col_l = B_inv[:, leaving].copy()
                B_inv -= np.outer(col_l, delta @ B_inv) / denom
                basis[leaving] = entering

        self._basis = basis
        return None, False


def _dual_ratio_test(
    tau: np.ndarray, lam: np.ndarray, n: int, tol: float
) -> int:
    """Minimum-ratio rule for the dual simplex.

    Returns the index of the leaving basis element, or ``-1`` when no valid
    pivot exists (dual unbounded → primal infeasible).
    """
    best = -1
    best_ratio = np.inf
    for k in range(n):
        if tau[k] > tol:
            ratio = lam[k] / tau[k]
            if ratio < best_ratio:
                best_ratio = ratio
                best = k
    return best


# ── public LP interface ───────────────────────────────────────────────────


class MinimalOrderedISAConstraints(ISAConstraints):
    """LP feasibility checker for ordered, discrete gate sentences.

    Builds the RHS vector *b* from fixed gate monodromy values and delegates
    the solve to a cached :class:`_StaircaseDualSimplex`.  The constraint
    matrix *A* depends only on sentence length and is shared across all
    sentences of that length.

    Parameters
    ----------
    isa_sequence : list[GateInvariants]
        Ordered gate invariants forming the sentence.
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
        self._orig_len = len(isa_sequence)

        # Pad 1-gate sentences so the QLR block math is uniform.
        if len(isa_sequence) == 1:
            isa_sequence = [*isa_sequence, GateInvariants((0, 0, 0, 0), name="I")]

        self._sequence = isa_sequence
        self._n = len(isa_sequence)
        self._n_ineq = len_qlr * (self._n - 1)
        self._n_params = LEN_GATE_INVARIANTS * (self._n - 2)

        self._solver: _StaircaseDualSimplex | None = None
        if self._n_params > 0:
            self._solver = _StaircaseDualSimplex.get(
                self._n, self._config.lp_feasibility_tol
            )
        self._b = self._build_base_rhs()
        self._last_target_contrib = np.zeros(len_qlr)

    # ── RHS construction ──────────────────────────────────────────────

    def _build_base_rhs(self) -> np.ndarray:
        """Build *b* from fixed gate monodromy values (target-independent)."""
        b = np.zeros(self._n_ineq)
        for i in range(self._n - 1):
            rows = slice(len_qlr * i, len_qlr * (i + 1))
            gi_contrib = _gi_block @ self._sequence[i + 1].monodromy
            if i == 0:
                gi_contrib += _ci_block @ self._sequence[0].monodromy
            b[rows] = _bi - gi_contrib
        return b

    def set_target(self, target: GateInvariants) -> None:
        """Update the last QLR block of *b* for a new target."""
        self._target = target
        ct = _ciplus1_block @ target.monodromy
        self._b[-len_qlr:] += self._last_target_contrib - ct
        self._last_target_contrib = ct

    # ── solve interface ───────────────────────────────────────────────

    def solve(
        self,
        target: GateInvariants,
        log_output: bool = False,
    ) -> ConstraintSolution:
        """Try both rho orientations, most-feasible first.

        Both orientations share *A*; only the last ``len_qlr`` entries of *b*
        differ.  The orientation with larger min-slack is tried first, avoiding
        a redundant solve ~98 % of the time.
        """
        ct_direct = _ciplus1_block @ target.monodromy
        ct_rho = _ciplus1_block @ target.rho_reflect.monodromy

        base_tail = self._b[-len_qlr:] + self._last_target_contrib
        if np.min(base_tail - ct_direct) >= np.min(base_tail - ct_rho):
            first, second = target, target.rho_reflect
        else:
            first, second = target.rho_reflect, target

        self.set_target(first)
        result = self.solve_single(log_output=log_output)
        if result.success:
            return result

        self.set_target(second)
        return self.solve_single(log_output=log_output)

    def solve_single(self, log_output: bool = False) -> ConstraintSolution:
        """Solve the LP for the current *b*.  Dispatches by sentence length."""
        slack_tol = -10 * self._config.lp_feasibility_tol

        # 1-gate: trivially feasible if all constraints satisfied.
        if self._orig_len == 1:
            if np.all(slack_tol <= self._b):
                return ConstraintSolution(
                    success=True,
                    sentence=(self._sequence[0],),
                    intermediates=(self._sequence[0],),
                )
            return ConstraintSolution(success=False)

        # 2-gate: no free variables → pure feasibility check.
        if self._n_params == 0:
            if np.all(slack_tol <= self._b):
                return ConstraintSolution(
                    success=True,
                    sentence=tuple(self._sequence),
                    intermediates=(self._sequence[0], self._target),
                )
            return ConstraintSolution(success=False)

        # 3+ gates: staircase dual simplex.
        assert self._solver is not None
        x, feasible = self._solver.solve(self._b)
        if not feasible:
            return ConstraintSolution(success=False)

        intermediates = [
            GateInvariants(tuple(x[i : i + LEN_GATE_INVARIANTS]))
            for i in range(0, len(x), LEN_GATE_INVARIANTS)
        ]
        return ConstraintSolution(
            success=True,
            sentence=tuple(self._sequence),
            intermediates=(self._sequence[0], *intermediates, self._target),
        )
