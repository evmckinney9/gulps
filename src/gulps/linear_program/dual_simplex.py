"""Warm-startable dual revised simplex for small dense LPs.

Implements a dual revised simplex method for small dense LPs.

Solves problems of the form:
    minimize    c' x
    subject to  A x <= b
where A and c are fixed, and b changes between solves.

Uses warm-starting: the previous basis is reused, which is efficient when only b changes. The initial basis must be dual-feasible (dual multipliers >= 0).

References:
    Nocedal, Wright, "Numerical Optimization", 2nd ed., S13.5 (Springer, 2006).
    Maros, "Computational Techniques of the Simplex Method", S5 (Springer, 2003).

Example:
    solver = DualRevisedSimplex(A, c, initial_basis)
    x, ok = solver.solve(b1)  # cold start
    x, ok = solver.solve(b2)  # warm start
"""

from __future__ import annotations

import numpy as np

# The dual simplex needs an initial basis. Some QLR constraints act only on a single component (standard basis vector).
# Picking these rows for each stage gives a basis matrix that is the identity, which is always dual-feasible.


def identity_row_indices(block: np.ndarray) -> np.ndarray:
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


def build_cold_start_basis(
    n_stages: int,
    ci_identity_rows: np.ndarray,
    cip_identity_rows: np.ndarray,
    len_qlr: int,
    d: int,
) -> np.ndarray:
    """Build an initial dual-feasible basis from the staircase structure.

    Each stage needs d basis rows that act as +I on that stage's columns.
    Stage 0 uses cip_identity_rows; stages 1..K-1 use ci_identity_rows from later blocks.
    """
    basis = np.empty(d * n_stages, dtype=np.intp)
    basis[:d] = cip_identity_rows
    for k in range(1, n_stages):
        block_start = (k + 1) * len_qlr
        basis[k * d : (k + 1) * d] = block_start + ci_identity_rows
    return basis


# Safety cap on iterations
MAX_PIVOTS: int = 64


class DualRevisedSimplex:
    """Dual revised simplex with basis warm-start across solves.

    Parameters:
        A : (m, n) array
            Constraint matrix. Must have full column rank.
        c : (n,) array
            Objective coefficients.
        initial_basis : (n,) int array
            Row indices into A forming an initial dual-feasible basis (dual multipliers >= 0).
        tol : float
            Feasibility tolerance for constraint violations.
    """

    def __init__(
        self,
        A: np.ndarray,
        c: np.ndarray,
        initial_basis: np.ndarray,
        tol: float = 1e-10,
    ) -> None:
        """Initialize the solver with fixed A, c, and initial basis."""
        self._A = np.ascontiguousarray(A, dtype=np.float64)
        self._c = c.astype(np.float64)
        self._n = A.shape[1]
        self._tol = tol
        self._basis = initial_basis.copy()

    def solve(self, b: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Solve for a given b vector.  Returns ``(x, feasible)``."""
        A, c, n, tol = self._A, self._c, self._n, self._tol
        basis = self._basis.copy()
        B_inv = np.linalg.inv(A[basis])

        for _ in range(MAX_PIVOTS):
            x = B_inv @ b[basis]

            # Which constraint is most violated?
            violations = A @ x - b
            entering = int(np.argmax(violations))
            if violations[entering] <= tol:
                self._basis = basis
                return x, True  # all satisfied -- optimal

            # Dual ratio test: pick the leaving variable that keeps lambda >= 0.
            tau = B_inv.T @ A[entering]  # step direction in basis space
            dual = -(B_inv.T @ c)  # dual multipliers  (KKT: A'lam = -c)
            leaving = _min_ratio(tau, dual, n, tol)

            if leaving < 0:
                self._basis = basis
                return None, False  # no valid pivot -- infeasible

            # Swap basis element, update B^-1 via rank-1 (Sherman-Morrison).
            _update_basis(B_inv, basis, A, entering, leaving)

        self._basis = basis
        return None, False  # iteration limit


def _min_ratio(tau: np.ndarray, dual: np.ndarray, n: int, tol: float) -> int:
    """Minimum-ratio test.  Returns leaving index, or -1 if unbounded."""
    best_idx = -1
    best_ratio = np.inf
    for k in range(n):
        if tau[k] > tol:
            r = dual[k] / tau[k]
            if r < best_ratio:
                best_ratio = r
                best_idx = k
    return best_idx


def _update_basis(
    B_inv: np.ndarray,
    basis: np.ndarray,
    A: np.ndarray,
    entering: int,
    leaving: int,
) -> None:
    """Rank-1 Sherman-Morrison update of B^-1, with full-inv fallback."""
    old_row = basis[leaving]
    delta = A[entering] - A[old_row]
    denom = 1.0 + delta @ B_inv[:, leaving]

    if abs(denom) < 1e-14:
        # Near-degenerate: recompute from scratch.
        basis[leaving] = entering
        B_inv[:] = np.linalg.inv(A[basis])
    else:
        col = B_inv[:, leaving].copy()
        B_inv -= np.outer(col, delta @ B_inv) / denom
        basis[leaving] = entering
