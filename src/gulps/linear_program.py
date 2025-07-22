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

id_inv = GateInvariants(logspec=(0.0, 0.0, 0.0, 0.0))


class MinimalOrderedISAConstraints:
    """Minimal LP constraints for ordered, defined ISA sequences."""

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

    def set_target(self, target_gate: GateInvariants, rho_reflect=False):
        # NOTE avoid reconstructing all of b_ub by tracking the last set target gate
        # when setting a new target, remove the contribution of the previous target
        self._target_def = target_gate
        # FIXME, actual_target is for debugging, remove later
        if rho_reflect:
            target_monodromy = target_gate.rho_reflect
            self._actual_target = target_gate
        else:
            target_monodromy = target_gate.monodromy
            self._actual_target = GateInvariants(logspec=target_monodromy)  # XXX FIXME
        ct = np.dot(self._ciplus1_block, target_monodromy)
        self.b_ub[-len_qlr:] += self.last_iter_ct - ct
        self.last_iter_ct = ct

    def solve(self, log_output=False):
        # edge case, if there were no free variables in x_vec
        if len(self.A_ub[0]) == 0:
            if np.all(0 <= self.b_ub):
                intermediate_invariants = (
                    id_inv,
                    self.isa_sequence[0],
                    self._actual_target,  # self._target_def, # FIXME
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
            id_inv,
            self.isa_sequence[0],
            *lp_invariants,
            self._actual_target,  # self._target_def, #FIXME
        )

        return self.isa_sequence, intermediate_invariants
