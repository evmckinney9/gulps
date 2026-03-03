"""Discrete LP solver using SciPy's HiGHS backend.

Solves the monodromy polytope feasibility problem for ordered gate sentences.
The constraint matrix (A_ub) depends only on sentence length; gate-specific
monodromy values only affect the RHS (b_ub). This separation is exploited
by _LPScaffolding to avoid redundant CSC conversions across sentences.
"""

import numpy as np
from scipy.optimize._highspy._core import (
    HighsDebugLevel,
    HighsModelStatus,
    ObjSense,
)
from scipy.optimize._highspy._core import (
    simplex_constants as _s_c,
)
from scipy.optimize._highspy._highs_wrapper import _highs_wrapper
from scipy.optimize._linprog_highs import _replace_inf
from scipy.sparse import csc_array

from gulps.config import GulpsConfig
from gulps.core.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities

_ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities


class _LPScaffolding:
    """Length-dependent LP scaffolding: A_ub, CSC, bounds, HiGHS options.

    A_ub depends only on sentence length (the QLR block pattern is fixed);
    gate monodromy values only affect b_ub.  Expensive parts are computed once
    per (length, tolerance) and reused across all sentences of the same length.
    """

    _cache: dict[tuple, "_LPScaffolding"] = {}
    _EMPTY_INTEGRALITY = np.empty(0, dtype=np.uint8)

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

        # CSC, bounds, and HiGHS options — all length-dependent
        if num_params > 0:
            A_csc = csc_array(A_ub)
            self.csc_indptr = A_csc.indptr.copy()
            self.csc_indices = A_csc.indices.copy()
            self.csc_data = A_csc.data.copy()
            self.lhs_ub = _replace_inf(np.full(num_ineq, -np.inf))
            self.lb = _replace_inf(np.full(num_params, -np.inf))
            self.ub = _replace_inf(np.full(num_params, np.inf))
            self.highs_options = {
                "presolve": True,
                "sense": ObjSense.kMinimize,
                "solver": "simplex",
                "time_limit": float("inf"),
                "highs_debug_level": HighsDebugLevel.kHighsDebugLevelNone,
                "dual_feasibility_tolerance": config.lp_feasibility_tol,
                "ipm_optimality_tolerance": None,
                "log_to_console": False,
                "mip_max_nodes": None,
                "output_flag": False,
                "primal_feasibility_tolerance": config.lp_feasibility_tol,
                "simplex_dual_edge_weight_strategy": None,
                "simplex_strategy": _s_c.SimplexStrategy.kSimplexStrategyDual,
                "ipm_iteration_limit": None,
                "simplex_iteration_limit": None,
                "mip_rel_gap": None,
            }


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

        # 2-gate sentence (no free variables — feasibility check only)
        if self.num_params == 0:
            if np.all(tol <= self.b_ub):
                return ConstraintSolution(
                    success=True,
                    sentence=tuple(self.isa_sequence),
                    intermediates=(self.isa_sequence[0], self._target_def),
                )
            return ConstraintSolution(success=False)

        # 3+ gate sentence — call HiGHS via cached scaffolding
        s = self._scaffolding
        res = _highs_wrapper(
            s.c,
            s.csc_indptr,
            s.csc_indices,
            s.csc_data,
            s.lhs_ub,
            _replace_inf(self.b_ub),
            s.lb,
            s.ub,
            s._EMPTY_INTEGRALITY,
            s.highs_options,
        )
        status = res.get("status")
        if (
            status != HighsModelStatus.kOptimal
            and status != HighsModelStatus.kObjectiveBound
        ):
            return ConstraintSolution(success=False)

        x = res["x"]
        lp_invariants = [
            GateInvariants(tuple(x[i : i + LEN_GATE_INVARIANTS]))
            for i in range(0, len(x), LEN_GATE_INVARIANTS)
        ]
        return ConstraintSolution(
            success=True,
            sentence=tuple(self.isa_sequence),
            intermediates=(self.isa_sequence[0], *lp_invariants, self._target_def),
        )
