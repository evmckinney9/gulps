"""Base classes and types for LP constraint solvers."""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from gulps.core.invariants import GateInvariants


@dataclass
class ConstraintSolution:
    """Unified result from any constraint solver (discrete or continuous).

    Attributes:
        success: Whether the LP/MILP found a feasible solution.
        sentence: Ordered list of gate invariants forming the decomposition.
            For discrete: the input sentence (fixed).
            For continuous: gates instantiated from k values.
        intermediates: Intermediate invariants [C_1, ..., C_n] representing
            the path through monodromy space: I -> C_1 -> ... -> C_n = target.
        parameters: Gate parameters (continuous only). For single-family
            continuous ISA, these are the k values where G_i = k_i * base.
        cost: Total cost of the decomposition (if computed by solver).
    """

    success: bool
    sentence: Optional[Tuple[GateInvariants, ...]] = None
    intermediates: Optional[Tuple[GateInvariants, ...]] = None
    parameters: Optional[Tuple[float, ...]] = None
    cost: Optional[float] = None


class ISAConstraints(Protocol):
    """Protocol for LP/MILP constraint solvers."""

    def set_target(self, target: GateInvariants) -> None:
        """Set the target gate invariants for the constraint RHS."""
        ...

    def solve_single(self, log_output: bool = False) -> "ConstraintSolution":
        """Solve the LP/MILP with the current target."""
        ...

    def solve(self, target: GateInvariants) -> ConstraintSolution:
        """Solve LP, automatically trying both rho orientations.

        Args:
            target: Alcove-normalized target gate invariants.

        Returns:
            ConstraintSolution with success=True if either orientation works.
        """
        for t in [target, target.rho_reflect]:
            self.set_target(t)
            result = self.solve_single()
            if result.success:
                return result
        return ConstraintSolution(success=False)
