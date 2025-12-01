try:
    import docplex  # or the precise submodule you need
except ModuleNotFoundError as exc:
    raise ImportError(
        "ContinuousISAConstraints requires the 'cplex' extra. "
        "Install with `pip install gulps[cplex]`."
    ) from exc

from typing import List, Optional, Tuple

import numpy as np
from docplex.mp.dvar import Var
from docplex.mp.model import Model

from gulps import GateInvariants
from gulps.linear_program.qlr import len_qlr, qlr_inequalities


class ContinuousISAConstraints:
    """LP constraints for a single yet continuous ISA sequence.

    Continuous ordered-ISA constraints with a single base gate:
    g_i = k_i * base.monodromy,  k_i ∈ [0,1].

    This is a variation of the original CPLEX implementation
    https://github.com/ajavadia/hetero_isas/blob/main/src/hetero_isas/monodromy_lp/lp_constraints/docplex_constraints.py

    But this used more indicator variables to select gates out of the ISA. In other words,
    this was used to bypass the enumeration of sentences.

    MinimalOrderedISAConstraints is fairly slimmed down to just the matrix construction,
    which makes it hard to add more complicated constraint relationships. So I will
    in the more general construction use CPLEX instead of Scipy.

    I want to eliminate (for now) the selection of gates from the ISA; rather we will
    consider G to be a single parameterized gate. Example, XX(theta), iswap(theta), or fSim(theta)
    but not {XX(theta1), iSWAP(theta2)}

    Crucially, NOTE we will assume the ISA contains a single gate described by a
    single continuous family: g = B * k, with k in [0,1]. The (duration/cost) is k.

    Further, I think I will drop the binary decision variable for T vs rho(T) since it is not
    a significant speedup and complicates the solution extraction step.

    Finally, this will remain a specialized method because `GateInvariants` does not have
    a proper way of dealing with parameterized gates. I tried to have both defined/parameterized gates
    in the original implementation of from `MonodromyLPGate` but dropped this when refactoring into
    `GateInvariants` in gulps.

    I think we can consider a sufficiently long sequence_length, and rather than pad identities,
    we can rely on the objective function to set theta/k:=0 when sentence is under-constrained and truncate ourselves.

    NOTE I will try to follow same interface as MinimalOrderedISAConstraints, but will avoid
    giving them a shared interface class for the time being. (init(), set(), solve())

    Thus, the following constraints must be made
    (a) standard qlr steps
    (b) boundary conditions (start at I, finish at T)
    (c) each G in (a) constraints by the bespoke parameterization
    (d) monotonic strength ordering
    """

    _ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

    def __init__(
        self,
        base: GateInvariants,
        sequence_length: int = 8,
        offset=1e-8,
    ):
        self.N = sequence_length
        if self.N < 2:
            raise ValueError("sequence_length must be at least 2")
        self.B = base.monodromy
        self.offset = offset
        self.k_lb = 0.01
        self.model = self._create_model()
        self._target_def: Optional[np.ndarray] = None
        self._target_cts: List = []  # holds the 3 equality constraints for c_N

    def set_target(self, target_gate: GateInvariants, rho_bool: bool = False):
        if rho_bool:
            self._target_def = target_gate.rho_reflect.monodromy
        else:
            self._target_def = target_gate.monodromy

        # remove previous target constraints (if any), then add new ones
        if self._target_cts:
            self.model.remove_constraints(self._target_cts)
            self._target_cts.clear()

        # ci_nested represents [c2, ..., cN]; last element is cN
        self._target_cts = [
            self.model.add_constraint(
                self.ci_nested[-1][j] == float(self._target_def[j])
            )
            for j in range(3)
        ]

    def solve(self, log_output: bool = False):
        sol = self.model.solve(log_output=log_output)
        if not sol:
            return None, None, None

        # Extract k, g, c
        ks = [float(sol.get_value(self.k_vars[i])) for i in range(self.N)]
        gis = [
            np.array(
                [sol.get_value(self.gi_nested[i][j]) for j in range(3)], dtype=float
            )
            for i in range(self.N)
        ]
        cis = [
            np.array(
                [sol.get_value(self.ci_nested[i][j]) for j in range(3)], dtype=float
            )
            for i in range(self.N - 1)
        ]

        gi_list = [GateInvariants(tuple(g)) for g in gis]
        intermediate_invariants = (gi_list[0],) + tuple(
            GateInvariants(tuple(c)) for c in cis
        )

        # ---- prune trailing zeros ----
        tol = max(self.offset, 1e-12)
        nz = [i for i, k in enumerate(ks) if k > tol]
        zero_index = (max(nz) + 1) if nz else 0

        gi_list = gi_list[:zero_index]
        intermediate_invariants = intermediate_invariants[:zero_index]

        return gi_list, intermediate_invariants, ks[:zero_index]

    def _create_model(self):
        m = Model("ContinuousISAConstraints", ignore_names=True)
        # Define variables, constraints, and objective here
        # STEP 1. CREATE ALL VARIABLES
        # Variables
        self.y = m.binary_var_list(self.N)
        # self.k_vars = m.continuous_var_list(self.N, lb=0.0, ub=1.0)
        self.k_vars = m.semicontinuous_var_list(self.N, lb=self.k_lb, ub=1.0)

        self.gi_nested = [
            m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N)
        ]
        self.ci_nested = [
            m.continuous_var_list(3, lb=-1.0, ub=1.0) for _ in range(self.N - 1)
        ]
        # used for depth/nz counting
        for i in range(self.N):
            m.add_constraint(self.k_vars[i] >= self.k_lb * self.y[i])
            m.add_constraint(self.k_vars[i] <= 1.0 * self.y[i])

        # STEP 2. GATE PARAMETERIZATION + MONOTONIC
        for i in range(self.N):
            m.add_constraint(self.gi_nested[i][0] == self.B[0] * self.k_vars[i])
            m.add_constraint(self.gi_nested[i][1] == self.B[1] * self.k_vars[i])
            m.add_constraint(self.gi_nested[i][2] == self.B[2] * self.k_vars[i])
        for i in range(self.N - 1):
            m.add_constraint(self.k_vars[i] >= self.k_vars[i + 1])
            m.add_constraint(self.y[i] >= self.y[i + 1])

        # STEP 3. QLR CONSTRAINTS
        self._qlr_constraints(m)

        # STEP 4. OBJECTIVE FUNCTION
        # primary: minimize total fractional gate cost ∑ k_i
        # secondary, shortest depth
        obj_cost = m.sum(self.k_vars)
        obj_depth = m.sum(self.y)
        # obj_leftpack = m.sum((i + 1) * self.y[i] for i in range(self.N))

        m.set_multi_objective(
            sense="min",
            exprs=[obj_cost, obj_depth],
            priorities=[2, 1],
            abstols=[self.offset, 0],
        )

        return m

    def _qlr_constraints(self, m: Model):
        # First block: L(g2, g1, c2)
        for r in range(len_qlr):
            m.add_constraint(
                m.scal_prod(self.gi_nested[1], self._ci_block[r])
                + m.scal_prod(self.gi_nested[0], self._gi_block[r])
                + m.scal_prod(self.ci_nested[0], self._ciplus1_block[r])
                <= float(self._bi[r])
            )

        # Interior blocks: L(c_j, g_{j+1}, c_{j+1})
        # ci_nested indexes: 0..N-2  -> (c2..cN)
        for j in range(1, self.N - 1):
            for r in range(len_qlr):
                m.add_constraint(
                    m.scal_prod(self.ci_nested[j - 1], self._ci_block[r])
                    + m.scal_prod(self.gi_nested[j + 1], self._gi_block[r])
                    + m.scal_prod(self.ci_nested[j], self._ciplus1_block[r])
                    <= float(self._bi[r])
                )
