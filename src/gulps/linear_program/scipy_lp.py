from typing import List

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

# Singleton empty integrality array (avoids repeated allocation)
_EMPTY_INTEGRALITY = np.empty(0, dtype=np.uint8)

# Successful HiGHS statuses
_OPTIMAL_STATUSES = frozenset(
    {HighsModelStatus.kOptimal, HighsModelStatus.kObjectiveBound}
)


class MinimalOrderedISAConstraints(ISAConstraints):
    """Minimal LP constraints for ordered, defined ISA sequences.

    This is specialized assuming a fixed sentence of constant Gs.
    (a) G contributions can be moved to RHS (b_i)
    (b) C_1 is trivially G_1
    """

    _ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

    def __init__(
        self, isa_sequence: List[GateInvariants], config: GulpsConfig | None = None
    ):
        self._orig_len = len(isa_sequence)
        # Pad 1-gate sentences with identity for LP constraint construction
        if len(isa_sequence) == 1:
            identity = GateInvariants((0.0, 0.0, 0.0, 0.0), name="I")
            isa_sequence = isa_sequence + [identity]
        self.isa_sequence = isa_sequence
        self.n = len(isa_sequence)
        self.num_ineq = len_qlr * (self.n - 1)
        self.num_params = LEN_GATE_INVARIANTS * (self.n - 2)
        self.last_iter_ct = np.zeros(len_qlr)
        self.config = config or GulpsConfig()
        self._setup_model()

    def _setup_model(self):
        # maximize total "strength" of intermediates (sum of monodromy coords)
        # This pushes intermediates toward polytope facets, producing more predictable
        # waypoints that improve segment cache hit rates across decompositions.
        # Use -1 coefficients since linprog minimizes.
        self.c = -np.ones(self.num_params)

        self.A_ub, self.b_ub = self._setup_inequalities()

        # Pre-compute CSC representation and HiGHS options dict once.
        # This avoids the per-solve overhead of scipy.linprog option validation
        # which creates a HighsOptionsManager for every option on every call (~0.65s/200 calls).
        if self.num_params > 0:
            A_csc = csc_array(self.A_ub)
            self._csc_indptr = A_csc.indptr.copy()
            self._csc_indices = A_csc.indices.copy()
            self._csc_data = A_csc.data.copy()
            self._lhs_ub = _replace_inf(np.full(self.num_ineq, -np.inf))
            self._lb = _replace_inf(np.full(self.num_params, -np.inf))
            self._ub = _replace_inf(np.full(self.num_params, np.inf))
            self._highs_options = {
                "presolve": True,
                "sense": ObjSense.kMinimize,
                "solver": "simplex",
                "time_limit": float("inf"),
                "highs_debug_level": HighsDebugLevel.kHighsDebugLevelNone,
                "dual_feasibility_tolerance": self.config.lp_feasibility_tol,
                "ipm_optimality_tolerance": None,
                "log_to_console": False,
                "mip_max_nodes": None,
                "output_flag": False,
                "primal_feasibility_tolerance": self.config.lp_feasibility_tol,
                "simplex_dual_edge_weight_strategy": None,
                "simplex_strategy": _s_c.SimplexStrategy.kSimplexStrategyDual,
                "ipm_iteration_limit": None,
                "simplex_iteration_limit": None,
                "mip_rel_gap": None,
            }

    def _setup_inequalities(self):
        A_ub = np.zeros((self.num_ineq, self.num_params))
        b_ub = np.zeros(self.num_ineq)

        # Build segment constraints L(c_{i+1}, g_{i+2}, c_{i+2}) for i=0..n-2
        # i=0: L(c_1, g_2, c_2) where c_1 = g_1 = isa_sequence[0]
        # i=k: L(c_{k+1}, g_{k+2}, c_{k+2})
        for i in range(self.n - 1):
            rows = slice(len_qlr * i, len_qlr * (i + 1))

            # c_{i+1} prefix block (LHS variable for i>0, RHS constant for i=0)
            if i > 0:
                offset = LEN_GATE_INVARIANTS * (i - 1)
                A_ub[rows, offset : offset + LEN_GATE_INVARIANTS] += self._ci_block

            # g_{i+2} gate contribution (always moved to RHS)
            gi_contrib = np.dot(self._gi_block, self.isa_sequence[i + 1].monodromy)
            if i == 0:
                # c_1 = g_1, move to RHS
                gi_contrib += np.dot(self._ci_block, self.isa_sequence[i].monodromy)
            b_ub[rows] += self._bi - gi_contrib

            # c_{i+2} result block (LHS variable, except last segment uses target)
            if i < self.n - 2:
                offset = LEN_GATE_INVARIANTS * i
                A_ub[rows, offset : offset + LEN_GATE_INVARIANTS] += self._ciplus1_block

        return A_ub, b_ub

    def set_target(self, target: GateInvariants) -> None:
        """Set the target gate invariants for the constraint RHS."""
        # NOTE avoid reconstructing all of b_ub by tracking the last set target gate
        # when setting a new target, remove the contribution of the previous target
        self._target_def = target
        ct = np.dot(self._ciplus1_block, self._target_def.monodromy)
        self.b_ub[-len_qlr:] += self.last_iter_ct - ct
        self.last_iter_ct = ct

    def solve_single(self, log_output=False) -> ConstraintSolution:
        # Edge case: 1-gate sentence (padded to 2 for LP math)
        if self._orig_len == 1:
            if np.all(-10 * self.config.lp_feasibility_tol <= self.b_ub):
                return ConstraintSolution(
                    success=True,
                    sentence=(self.isa_sequence[0],),
                    intermediates=(self.isa_sequence[0],),
                )
            return ConstraintSolution(success=False)

        # Edge case: 2-gate sentence (no free variables)
        if len(self.A_ub[0]) == 0:
            # NOTE: try eps here but if causes problems maybe need to go to next enumerated sentence
            if np.all(-10 * self.config.lp_feasibility_tol <= self.b_ub):
                return ConstraintSolution(
                    success=True,
                    sentence=tuple(self.isa_sequence),
                    intermediates=(self.isa_sequence[0], self._target_def),
                )
            return ConstraintSolution(success=False)

        # Call HiGHS directly, bypassing scipy.linprog option validation overhead
        rhs = _replace_inf(self.b_ub)
        res = _highs_wrapper(
            self.c,
            self._csc_indptr,
            self._csc_indices,
            self._csc_data,
            self._lhs_ub,
            rhs,
            self._lb,
            self._ub,
            _EMPTY_INTEGRALITY,
            self._highs_options,
        )
        if res.get("status") not in _OPTIMAL_STATUSES:
            return ConstraintSolution(success=False)

        x = res["x"]

        # Extract intermediate invariants from LP solution vector
        lp_invariants = [
            GateInvariants(tuple(x[i : i + LEN_GATE_INVARIANTS]))
            for i in range(0, len(x), LEN_GATE_INVARIANTS)
        ]
        intermediates = (self.isa_sequence[0], *lp_invariants, self._target_def)

        return ConstraintSolution(
            success=True,
            sentence=tuple(self.isa_sequence),
            intermediates=intermediates,
        )
