"""Constraint definitions and factory for monodromy linear programming."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from hetero_isas.monodromy_lp.isa import MonodromyLPGate, MonodromyLPISA
from hetero_isas.monodromy_lp.lp_constraints.qlr import len_qlr, qlr_inequalities

GATE_INVARIANTS = 3  # define this constant to avoid magic values
len_gi = GATE_INVARIANTS


class ConstraintsFactory:
    """Factory class for creating appropriate constraint objects.

    This class determines whether to use LPConstraints or ConstantSubLPConstraints
    based on the properties of the given ISA.

    1.discrete ISA, defined gate sequence
    A_ub.x <= b_ub; A_eq.x=b_eq
    x = [c2,...cn]
    minimize 0

    2. continuous ISA, defined gate sequence
    A_ub.x <= b_ub; A_eq.x=b_eq
    x = [f1,..,fn|c2,...cn]
    minimize sum f_i

    for undefined sequence, could iterate N
    or decide to pad N>> with identity gates
    3.discrete ISA, undefined sequence,
    A_ub.x <= b_ub; A_eq.x=b_eq
    x = [kij, ..., knm|f1,...,fn|c2,...cn]
    sum_j' k_ij' = 1
    minimize sum f_i
    where k_ij is {0,1} which selects gi in gate sequence to use gate Gj in ISA.

    4.continuous ISA, undefined sequence
    (4.5 a continuous gate + a discrete gate)
    NotImplementedError()

    Attributes:
        _ci_block, _gi_block, _ciplus1_block, _bi: Constants used in constraint generation.
        constraints_cls: The constraint class to use (LPConstraints or ConstantSubLPConstraints).

    Methods:
        build: Create and return the appropriate constraints object.
    """

    def __init__(
        self,
        isa: MonodromyLPISA,
        use_ordered_sequences: bool,
        include_rho_target: bool = True,
    ):
        """Initialize ConstraintsFactory."""
        # case 1. discrete ISA, defined gate sequence
        if isa.defined and use_ordered_sequences:
            if include_rho_target:
                from hetero_isas.monodromy_lp.lp_constraints.scipy_constraints import (
                    RhoInclusiveOrderedDefinedISAConstraints,
                )

                self.constraints_cls = RhoInclusiveOrderedDefinedISAConstraints
            else:
                from hetero_isas.monodromy_lp.lp_constraints.scipy_constraints import (
                    OrderedDefinedISAConstraints,
                )

                self.constraints_cls = OrderedDefinedISAConstraints
        # case 2. continuous ISA, undefined gate sequence
        elif isa.defined and not use_ordered_sequences:
            from hetero_isas.monodromy_lp.lp_constraints.docplex_constraints import (
                DocplexConstraints,
            )

            self.constraints_cls = DocplexConstraints
        # case 3. discrete ISA, undefined sequence
        elif not isa.defined and use_ordered_sequences:
            raise NotImplementedError()
        # case 4.continuous ISA, undefined sequence
        else:
            raise NotImplementedError()

    def build(self, isa_sequence: List[MonodromyLPGate]) -> "LPConstraints":
        """Build and return the appropriate constraints object.

        Args:
            isa_sequence (List[MonodromyLPGate]): The sequence of gates to use.
            target_gate (MonodromyLPGate): The target gate to decompose.

        Returns:
            LPConstraints: An instance of either LPConstraints or ConstantSubLPConstraints.
        """
        return self.constraints_cls(isa_sequence)


class LPConstraints(ABC):
    """Represents the constraints for the linear programming problem.

    This class sets up the matrices and vectors needed for the LP solver,
    including inequality and equality constraints.

    A_ub.x <= b_ub; A_eq.x=b_eq
    x = [f1,..,fn|g1,...,gn|c2,...cn]
    minimize sum f_i, (costs of basis gates g_i)

    N=0: I ?= T (base case)
    N=1: L.G.L ?= T (base case)
    N=2: L(c1:=G1, g:G2 c2:=T)
    N=3: L(c1:=G1, g:=G2, c2) -> L(c2, g:=G3, c3:=T)
    N=n: L(c1:=G1, g:=G2, c2) -...-> L(c{n-1}, g:=Gn, cn:=T)
    """

    _ci_block, _gi_block, _ciplus1_block, _bi = qlr_inequalities
    # NOTE decided that is easier to rho-rotate the target gate
    # instead of rotating all the inequalities...
    # _rho_ci_block, _rho_gi_block, _rho_ciplus1_block, _rho_bi = rho_qlr_inequalities

    def __init__(self, isa_sequence: List[MonodromyLPGate]):
        """Initialize LPConstraints."""
        self.n = len(isa_sequence)
        self.isa_sequence = isa_sequence
        self.bounds = (None, None)
        self.integrality = None
        self.num_ineq = None
        self.num_params = None
        self.g1_index = None
        self.c2_index = None
        # self.rho_reflect = False

    def attempt_solve(self, target, log_output=False):
        self._set_target(target)
        self._last_result = self._lp_solve(log_output)
        if self._last_result and self._last_result.success:
            return self._extract_solution()
        return None

    @abstractmethod
    def _lp_solve(self):
        raise NotImplementedError

    @abstractmethod
    def _create_model(self):
        raise NotImplementedError

    @abstractmethod
    def _set_target(self, target_gate: MonodromyLPGate):
        raise NotImplementedError

    @abstractmethod
    def _extract_solution(self):
        # (basis gate sequence, intermediate mono coords)
        raise NotImplementedError

    # FIXME?
    def rho_rotate(self) -> "LPConstraints":
        """Rho-reflect.

        NOTE I have found only rotated target wokrs bests but might make
        more sense (or equivalent? maybe not) to keep the target fixed
        and rotate qlr instead?
        """
        # we need to update all (in)equalities
        # ret = LPConstraints(self.isa_sequence, self.target_gate)
        # ret.rho_reflect = not self.rho_reflect  # use rotated qlr_poly
        # see https://github.com/Qiskit-Extensions/monodromy/blob/main/monodromy/coordinates.py#L271
        return self.__class__(self.isa_sequence, self.target_gate.rho_reflect())
