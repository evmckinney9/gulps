from typing import List, Optional, Tuple

import numpy as np
from qiskit.circuit import Gate

from hetero_isas.monodromy_lp.invariants import unitary_to_mono_coordinates

GATE_INVARIANTS = 3  # define this constant to avoid magic values


# TODO
# can a gate have dynamic cost depending on parameterization?
class MonodromyLPGate:
    """Represents a gate in the monodromy linear programming framework.

    This class encapsulates the properties and constraints of a quantum gate,
    including its cost, equality and inequality constraints, and whether it's
    fully defined or part of a continuous family of gates.

    Attributes:
        cost (float): The cost associated with applying this gate.
        equalities (List[Tuple[float]]): Equality constraints for the gate.
        defined (bool): Whether the gate is fully defined or part of a continuous family.
        definition (Optional[Tuple[float, float, float]]): The monodromy coordinates of the gate if defined.
        name (Optional[str]): A string representation of the gate.

    Methods:
        from_unitary: Create a MonodromyLPGate instance from a Qiskit Gate object.
        local_equiv: Check if this gate is locally equivalent to another gate.
    """

    def __init__(
        self,
        equalities: Optional[List[Tuple[float]]],
        cost: float = 1.0,
        logspec: Optional[Tuple[float, float, float]] = None,
        name: Optional[str] = None,
    ):
        """Construct a gate.

        A*g_a + B*g_b + C*g_b = D
        write list of tuples as (A,B,C,D)

        similarly for
        A*g_a + B*g_b + C*g_b <= D

        Example:
        CXGate has coordinates (.25, .25, -.25),
        thus equalities written as
            [(1, 0, 0, 0.25000000000000006),
            (0, 1, 0, 0.25000000000000006),
            (0, 0, 1, -0.25)]

        in comparison to CX^alpha family has coordinates (A,A,-A)
        equalities written as
            [(1, 1, -2, 0),
            (1, -1, 0, 0)].

        NOTE working on a self.defined attribute. This indicates that
        there is only a single gate defined by the set of equations given.
        Example, CNOT is defined by CNOT-family is not. (discrete vs continuous)
        The reason I want this is because I have variables in LP that are redundant
        I don't need to have these be variables that get solved for if they are already constants.
        More importantly, if they are defined to be constants I can setup mixed-integer
        where I introduce indicator variabes to select the gate sequence in one step.
        """
        # TODO if we wanted to, could check if the (in)equalities have a single solution
        # if only has a single solution then we know it is defined.

        self.cost = cost  # TODO convert to (in)fidelity?

        # TODO, need to modify inequality_matrices
        # in order to have both qlr and some additional gate definitions
        # currently all gate definitions are equalities only
        # self._inequality_terms = inequalities or []
        self._equality_terms = equalities
        self.name = name
        self.logspec = logspec
        self.defined = logspec is not None
        if self.defined:
            self._definition = logspec[:GATE_INVARIANTS]
        self.rho_reflect_invariant = (
            False  # FIXME I don't have confidence in my rho-reflect definitions
            and self.defined
            and (np.isclose(self._definition[2] + 0.5, self._definition[0]))
        )
        assert (
            len(self._equality_terms) <= GATE_INVARIANTS
        ), "Too many equality constraints"

    @classmethod
    def from_unitary(
        cls, gate: Gate, cost=1.0, name=None, alcove_norm=False
    ) -> "MonodromyLPGate":
        """Create a MonodromyLPGate instance from a Qiskit Gate object.

        Args:
            gate (Gate): The Qiskit Gate to convert.
            cost (float): The cost associated with the gate.
            name (Optional[str]): A string representation of the gate.
            alcove_norm (bool): Uniquely select gate up to quotient set degeneracy. XXX not tested.

        Returns:
            MonodromyLPGate: The created instance.
        """
        coords = unitary_to_mono_coordinates(gate)

        # uniquely select {LogSpecU, LogSpec-U}
        if alcove_norm and (coords[2] + 0.5 <= coords[0]):
            coords = [
                coords[2] + 0.5,
                coords[3] + 0.5,
                coords[0] - 0.5,
                coords[1] - 0.5,
            ]

        equality_terms = [
            (1, 0, 0, coords[0]),
            (0, 1, 0, coords[1]),
            (0, 0, 1, coords[2]),
        ]
        return cls(
            equalities=equality_terms,
            cost=cost,
            logspec=coords,
            name=name,
        )

    def __str__(self) -> str:
        """Override to_string."""
        return self.name if self.name is not None else repr(self)

    def __lt__(self, other: "MonodromyLPGate") -> bool:
        """Lexicographic ordering by cost."""
        return self.cost < other.cost

    def rho_reflect(self) -> "MonodromyLPGate":
        """Rho-reflection."""
        if not self.defined:
            raise NotImplementedError()

        if self.rho_reflect_invariant:
            return self

        # XXX pretty sure this is wrong?
        # if we are using the new alcove definition
        rho_coords = [
            self.logspec[2] + 0.5,
            self.logspec[3] + 0.5,
            self.logspec[0] - 0.5,
            self.logspec[1] - 0.5,
        ]

        equality_terms = [
            (1, 0, 0, rho_coords[0]),
            (0, 1, 0, rho_coords[1]),
            (0, 0, 1, rho_coords[2]),
        ]

        return MonodromyLPGate(
            equalities=equality_terms,
            cost=self.cost,
            logspec=rho_coords,
            name=f"*{self.name}",
        )
        # qcontrol_coords = list(c1c2c3(gate))
        # if qcontrol_coords[0] > 0.5:
        #     qcontrol_coords[0] = 1 - qcontrol_coords[0]

        #     coords = positive_canonical_to_monodromy_coordinate(
        #         *np.array(qcontrol_coords) * (np.pi / 2)
        #     )

    def local_equiv(self, other: "MonodromyLPGate") -> bool:
        """Check if this gate is locally equivalent to another gate.

        Args:
            other (MonodromyLPGate): The gate to compare with.

        Returns:
            bool: True if the gates are locally equivalent, False otherwise.
        """
        if not self.defined or not other.defined:
            raise NotImplementedError(
                "Local equivalence check not implemented for undefined gates"
            )
        return self.definition == other.definition

    # FIXME still thinking about the best way to do this
    @property
    def definition(self) -> Tuple[float, float, float]:
        """The monodromy coordinates of the gate, if defined."""
        if self.defined:
            return self._definition
        raise ValueError("Not a well-defined gate yet, instead use (in)equalities")

    @property
    def strength(self) -> float:
        """Heuristic for gate pulse cost via Weyl chamber vector length."""
        if self.defined:
            return min(sum(self._definition), sum(self.rho_reflect()._definition))
        raise ValueError("Not a well-defined gate yet.")

    @property
    def equality_terms(self) -> List[Tuple[float]]:
        """Equality constraints for the gate."""
        return self._equality_terms
