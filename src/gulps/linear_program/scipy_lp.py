"""Discrete LP solver with fast numpy vertex solver and HiGHS fallback.

Solves the monodromy polytope feasibility problem for ordered gate sentences.
The constraint matrix (A_ub) depends only on sentence length; gate-specific
monodromy values only affect the RHS (b_ub). This separation is exploited
by _LPScaffolding to avoid redundant CSC conversions across sentences.

For small variable counts (d ≤ 6), a precomputed-vertex numpy solver replaces
HiGHS entirely — exploiting that the 144-row A_ub has only ~14 unique row
directions, yielding ~77 dual-feasible vertex candidates checkable in ~0.03ms.
"""

from itertools import combinations

import highspy
import numpy as np

from gulps.config import GulpsConfig
from gulps.core.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

_ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

# Maximum LP variable count for the numpy vertex solver.
# d=3 (3-gate sentences): ~14 unique dirs → C(14,3)=364 combos, ~77 dual-feasible.
# d=6+ gets combinatorially expensive; fall back to HiGHS for 4+ gate sentences.
_VERTEX_LP_MAX_DIM = 3


class _VertexLP:
    """Vectorized numpy LP for fixed A, varying b.

    Exploits that A_ub has very few unique row directions (14 from 144 for d=3).
    Precomputes all dual-feasible vertex candidates; at runtime evaluates them
    in a single batched einsum + argmin.

    Attributes:
        unique_A: (n_unique, d) deduplicated constraint normals.
        K: number of dual-feasible vertex candidates.
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, tol: float = 1e-10):
        m, d = A.shape
        self.d = d
        self.tol = tol

        # Deduplicate rows of A
        row_dict: dict[tuple, list[int]] = {}
        for i in range(m):
            key = tuple(A[i])
            row_dict.setdefault(key, []).append(i)

        self.unique_A = np.array([np.array(k) for k in row_dict])  # (n_u, d)
        groups = list(row_dict.values())
        n_u = len(groups)

        # Build padded index array for fast b-reduction (min per group)
        max_g = max(len(g) for g in groups)
        self._b_idx = np.zeros((n_u, max_g), dtype=np.intp)
        self._b_mask = np.zeros((n_u, max_g), dtype=bool)
        for j, g in enumerate(groups):
            self._b_idx[j, : len(g)] = g
            self._b_mask[j, : len(g)] = True

        # Enumerate dual-feasible vertex bases
        idx_list, inv_list = [], []
        for combo in combinations(range(n_u), d):
            A_B = self.unique_A[list(combo)]
            det = np.linalg.det(A_B)
            if abs(det) < 1e-12:
                continue
            A_inv = np.linalg.inv(A_B)
            if np.all(A_inv.T @ c >= -tol):
                idx_list.append(combo)
                inv_list.append(A_inv)

        self.K = len(idx_list)
        self._idx = np.array(idx_list, dtype=np.intp)  # (K, d)
        self._Ainv = np.array(inv_list)  # (K, d, d)
        self._c = c  # (d,)

    def solve(self, b_full: np.ndarray):
        """Solve min c'x s.t. Ax <= b.  Returns (x, True) or (None, False)."""
        # Reduce b: take tightest constraint per direction group
        sel = b_full[self._b_idx]
        sel[~self._b_mask] = np.inf
        b_red = np.min(sel, axis=1)  # (n_unique,)

        # Compute all vertex candidates: x_k = A_inv_k @ b_red[idx_k]
        b_basis = b_red[self._idx]  # (K, d)
        x_all = np.einsum("kij,kj->ki", self._Ainv, b_basis)  # (K, d)

        # Feasibility: max over constraints of (A_red @ x - b_red)
        viol = self.unique_A @ x_all.T - b_red[:, None]  # (n_u, K)
        max_viol = np.max(viol, axis=0)  # (K,)
        feasible = max_viol <= 10 * self.tol

        if not np.any(feasible):
            return None, False

        objs = x_all @ self._c  # (K,)
        objs[~feasible] = np.inf
        return x_all[np.argmin(objs)], True


class _LPScaffolding:
    """Length-dependent LP scaffolding: A_ub, CSC, bounds, HiGHS options.

    A_ub depends only on sentence length (the QLR block pattern is fixed);
    gate monodromy values only affect b_ub.  Expensive parts are computed once
    per (length, tolerance) and reused across all sentences of the same length.
    """

    _cache: dict[tuple, "_LPScaffolding"] = {}

    @classmethod
    def get(cls, n: int, config: GulpsConfig) -> "_LPScaffolding":
        key = (n, config.lp_feasibility_tol)
        if key not in cls._cache:
            cls._cache[key] = cls(n, config)
        return cls._cache[key]

    def __init__(self, n: int, config: GulpsConfig):
        num_ineq = len_qlr * (n - 1)
        num_params = LEN_GATE_INVARIANTS * (n - 2)

        self.c = -np.ones(num_params)

        # A_ub: length-dependent block placement from QLR inequalities
        A_ub = np.zeros((num_ineq, num_params))
        for i in range(n - 1):
            rows = slice(len_qlr * i, len_qlr * (i + 1))
            if i > 0:
                offset = LEN_GATE_INVARIANTS * (i - 1)
                A_ub[rows, offset : offset + LEN_GATE_INVARIANTS] += _ci_block
            if i < n - 2:
                offset = LEN_GATE_INVARIANTS * i
                A_ub[rows, offset : offset + LEN_GATE_INVARIANTS] += _ciplus1_block

        if num_params > 0:
            # Fast numpy vertex solver for small LPs
            if num_params <= _VERTEX_LP_MAX_DIM:
                self.vertex_lp = _VertexLP(self.c, A_ub, tol=config.lp_feasibility_tol)
            else:
                self.vertex_lp = None

            # Persistent highspy model — only RHS changes between solves
            h = highspy.Highs()
            h.silent()
            for _ in range(num_params):
                h.addVar(-highspy.kHighsInf, highspy.kHighsInf)
            h.changeColsCost(
                num_params,
                np.arange(num_params, dtype=np.int32),
                self.c,
            )
            for i in range(num_ineq):
                nz_idx = np.nonzero(A_ub[i])[0].astype(np.int32)
                h.addRow(
                    -highspy.kHighsInf,
                    0.0,
                    len(nz_idx),
                    nz_idx,
                    A_ub[i, nz_idx],
                )
            h.setOptionValue("solver", "simplex")
            h.setOptionValue("presolve", "off")
            h.setOptionValue("primal_feasibility_tolerance", config.lp_feasibility_tol)
            h.setOptionValue("dual_feasibility_tolerance", config.lp_feasibility_tol)
            self._highs = h
            self._row_indices = np.arange(num_ineq, dtype=np.int32)
            self._row_lower = np.full(num_ineq, -highspy.kHighsInf)


class MinimalOrderedISAConstraints(ISAConstraints):
    """Minimal LP constraints for ordered, discrete ISA sequences.

    Specialized for fixed gate sentences: gate contributions are folded into b_ub
    and C_1 = G_1.  Length-dependent scaffolding (A_ub, CSC, bounds, HiGHS options)
    is shared via _LPScaffolding; only the gate-dependent b_ub is rebuilt per sentence.
    """

    def __init__(
        self, isa_sequence: list[GateInvariants], config: GulpsConfig | None = None
    ):
        self._orig_len = len(isa_sequence)
        if len(isa_sequence) == 1:
            isa_sequence = isa_sequence + [
                GateInvariants((0.0, 0.0, 0.0, 0.0), name="I")
            ]
        self.isa_sequence = isa_sequence
        self.n = len(isa_sequence)
        self.num_ineq = len_qlr * (self.n - 1)
        self.num_params = LEN_GATE_INVARIANTS * (self.n - 2)
        self.last_iter_ct = np.zeros(len_qlr)
        self.config = config or GulpsConfig()
        self._scaffolding = _LPScaffolding.get(self.n, self.config)
        self.b_ub = self._compute_b_ub()

    def _compute_b_ub(self):
        """Build b_ub from gate monodromy values (sentence-dependent)."""
        b_ub = np.zeros(self.num_ineq)
        for i in range(self.n - 1):
            rows = slice(len_qlr * i, len_qlr * (i + 1))
            gi_contrib = np.dot(_gi_block, self.isa_sequence[i + 1].monodromy)
            if i == 0:
                gi_contrib += np.dot(_ci_block, self.isa_sequence[i].monodromy)
            b_ub[rows] += _bi - gi_contrib
        return b_ub

    def set_target(self, target: GateInvariants) -> None:
        """Set target gate invariants, updating only the last len_qlr rows of b_ub."""
        self._target_def = target
        ct = np.dot(_ciplus1_block, target.monodromy)
        self.b_ub[-len_qlr:] += self.last_iter_ct - ct
        self.last_iter_ct = ct

    def solve(
        self, target: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Solve LP, trying the most-feasible rho orientation first.

        Both orientations share the same A_ub; only the last len_qlr entries of
        b_ub differ.  We pick the orientation with larger min(b_ub) slack first,
        avoiding a redundant HiGHS call ~98% of the time.
        """
        ct_direct = np.dot(_ciplus1_block, target.monodromy)
        ct_rho = np.dot(_ciplus1_block, target.rho_reflect.monodromy)

        base_tail = self.b_ub[-len_qlr:] + self.last_iter_ct
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

    def solve_single(self, log_output=False) -> ConstraintSolution:
        tol = -10 * self.config.lp_feasibility_tol

        # 1-gate sentence (padded to 2 for LP math)
        if self._orig_len == 1:
            if np.all(tol <= self.b_ub):
                return ConstraintSolution(
                    success=True,
                    sentence=(self.isa_sequence[0],),
                    intermediates=(self.isa_sequence[0],),
                )
            return ConstraintSolution(success=False)

        # 2-gate sentence (no free variables, feasibility check only)
        if self.num_params == 0:
            if np.all(tol <= self.b_ub):
                return ConstraintSolution(
                    success=True,
                    sentence=tuple(self.isa_sequence),
                    intermediates=(self.isa_sequence[0], self._target_def),
                )
            return ConstraintSolution(success=False)

        # 3+ gate sentence - call HiGHS via cached scaffolding
        s = self._scaffolding

        # Fast path: numpy vertex solver (precomputed dual-feasible bases)
        if s.vertex_lp is not None:
            x, feasible = s.vertex_lp.solve(self.b_ub)
            if not feasible:
                return ConstraintSolution(success=False)
            lp_invariants = [
                GateInvariants(tuple(x[i : i + LEN_GATE_INVARIANTS]))
                for i in range(0, len(x), LEN_GATE_INVARIANTS)
            ]
            return ConstraintSolution(
                success=True,
                sentence=tuple(self.isa_sequence),
                intermediates=(
                    self.isa_sequence[0],
                    *lp_invariants,
                    self._target_def,
                ),
            )

        # HiGHS via persistent highspy model (only RHS update + warm-start)
        h = s._highs
        h.changeRowsBounds(
            len(self.b_ub),
            s._row_indices,
            s._row_lower,
            self.b_ub,
        )
        h.run()
        status = h.getModelStatus()
        if (
            status != highspy.HighsModelStatus.kOptimal
            and status != highspy.HighsModelStatus.kObjectiveBound
        ):
            return ConstraintSolution(success=False)

        x = np.array(h.getSolution().col_value)
        lp_invariants = [
            GateInvariants(tuple(x[i : i + LEN_GATE_INVARIANTS]))
            for i in range(0, len(x), LEN_GATE_INVARIANTS)
        ]
        return ConstraintSolution(
            success=True,
            sentence=tuple(self.isa_sequence),
            intermediates=(self.isa_sequence[0], *lp_invariants, self._target_def),
        )
