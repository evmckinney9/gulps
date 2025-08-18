# given a CircuitSentence, returns the LP problem constraints
# resembles lp_constraints/scipy_constraints.py/OrderedDefinedISAConstraints
# write a less generic version of the LP formalism for GULPS
# adapted from lp_constraints/lp_constraints.py
#              lp_constraints/qlr.py
#              lp_constraints/scipy_constraints.py
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from gulps.utils.invariants import LEN_GATE_INVARIANTS, GateInvariants
from gulps.utils.qlr import len_qlr, qlr_inequalities


class MinimalOrderedISAConstraints:
    """Minimal LP constraints for ordered, defined ISA sequences.

    This is specialized assuming a fixed sentence of constant Gs.
    (a) G contributions can be moved to RHS (b_i)
    (b) C_1 is trivially G_1
    """

    _ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities

    def __init__(self, isa_sequence: List[GateInvariants]):
        self.isa_sequence = isa_sequence
        self.n = len(isa_sequence)
        self.num_ineq = len_qlr * (self.n - 1)
        self.num_params = LEN_GATE_INVARIANTS * (self.n - 2)
        self.last_iter_ct = np.zeros(len_qlr)
        self._setup_model()

    def _setup_model(self):
        self.c = np.zeros(self.num_params)  # objective: minimize zero
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

    def set_target(self, target_gate: GateInvariants, rho_bool=False):
        # NOTE avoid reconstructing all of b_ub by tracking the last set target gate
        # when setting a new target, remove the contribution of the previous target
        if rho_bool:
            self._target_def = target_gate.rho_reflect
        else:
            self._target_def = target_gate
        ct = np.dot(self._ciplus1_block, self._target_def.monodromy)
        self.b_ub[-len_qlr:] += self.last_iter_ct - ct
        self.last_iter_ct = ct

    def solve(self, log_output=False):
        # edge case, if there were no free variables in x_vec
        if len(self.A_ub[0]) == 0:
            # NOTE, try eps here but if causes problems maybe need to go to next enumerated sentence
            if np.all(-5e-7 <= self.b_ub):  # 0<=self.b_ub
                intermediate_invariants = (
                    self.isa_sequence[0],
                    self._target_def,
                )
                return self.isa_sequence, intermediate_invariants
            else:
                return None, None
        result = linprog(
            c=self.c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            method="highs",
            bounds=(None, None),  # no bounds on variables
            options={
                "disp": log_output,
                "presolve": True,
                # "primal_feasibility_tolerance": 1e-7,
                # "dual_feasibility_tolerance": 1e-7,
            },
        )
        if result.success:
            return self._extract_from_blocks(result.x)
        return None, None

    def _extract_from_blocks(self, lp_vec):
        lp_invariants = [
            GateInvariants(tuple(lp_vec[i : i + LEN_GATE_INVARIANTS]))
            for i in range(0, len(lp_vec), LEN_GATE_INVARIANTS)
        ]

        # Build full intermediate sequence
        intermediate_invariants = (
            self.isa_sequence[0],
            *lp_invariants,
            self._target_def,
        )

        return self.isa_sequence, intermediate_invariants
