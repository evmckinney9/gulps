# given a CircuitSentence, returns the LP problem constraints
# resembles lp_constraints/scipy_constraints.py/OrderedDefinedISAConstraints
# write a less generic version of the LP formalism for GULPS
# adapted from lp_constraints/lp_constraints.py
#              lp_constraints/qlr.py
#              lp_constraints/scipy_constraints.py
from typing import List

import numpy as np
from scipy.optimize import linprog

from gulps.config import GulpsConfig
from gulps.core.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.linear_program.lp_abc import ConstraintSolution, ISAConstraints
from gulps.linear_program.qlr import len_qlr, qlr_inequalities


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
        self.isa_sequence = isa_sequence
        self.n = len(isa_sequence)
        self.num_ineq = len_qlr * (self.n - 1)
        self.num_params = LEN_GATE_INVARIANTS * (self.n - 2)
        self.last_iter_ct = np.zeros(len_qlr)
        self.config = config or GulpsConfig()
        self._setup_model()

    def _setup_model(self):
        # Objective: maximize total "strength" of intermediates (sum of monodromy coords)
        # This pushes intermediates toward polytope facets, producing more predictable
        # waypoints that improve segment cache hit rates across decompositions.
        # Use -1 coefficients since linprog minimizes.
        # self.c = np.zeros(self.num_params)
        self.c = -np.ones(self.num_params)
        self.A_ub, self.b_ub = self._setup_inequalities()
        self.A_eq, self.b_eq = None, None  # no equalities

    # XXX this is a bit hard to read
    def _setup_inequalities(self):
        A_ub = np.zeros((self.num_ineq, self.num_params))
        b_ub = np.zeros(self.num_ineq)

        for i in range(self.n - 1):
            rows = slice(len_qlr * i, len_qlr * (i + 1))

            # c_i block
            if i > 0:
                offset = LEN_GATE_INVARIANTS * (i - 1)
                A_ub[rows, offset : offset + LEN_GATE_INVARIANTS] += self._ci_block

            # g_i contribution
            gi_contrib = np.dot(self._gi_block, self.isa_sequence[i + 1].monodromy)
            if i == 0:
                gi_contrib += np.dot(self._ci_block, self.isa_sequence[i].monodromy)
            b_ub[rows] += self._bi - gi_contrib

            # c_{i+1} block
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
        # Edge case: no free variables (2-gate sentence)
        if len(self.A_ub[0]) == 0:
            # NOTE: try eps here but if causes problems maybe need to go to next enumerated sentence
            if np.all(-10 * self.config.lp_feasibility_tol <= self.b_ub):
                return ConstraintSolution(
                    success=True,
                    sentence=tuple(self.isa_sequence),
                    intermediates=(self.isa_sequence[0], self._target_def),
                )
            return ConstraintSolution(success=False)

        result = linprog(
            c=self.c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            method="highs",
            bounds=(None, None),
            options={
                "disp": log_output,
                "presolve": True,
                "primal_feasibility_tolerance": self.config.lp_feasibility_tol,
                "dual_feasibility_tolerance": self.config.lp_feasibility_tol,
            },
        )
        if not result.success:
            return ConstraintSolution(success=False)

        # Extract intermediate invariants from LP solution vector
        lp_invariants = [
            GateInvariants(tuple(result.x[i : i + LEN_GATE_INVARIANTS]))
            for i in range(0, len(result.x), LEN_GATE_INVARIANTS)
        ]
        intermediates = (self.isa_sequence[0], *lp_invariants, self._target_def)

        return ConstraintSolution(
            success=True,
            sentence=tuple(self.isa_sequence),
            intermediates=intermediates,
        )
