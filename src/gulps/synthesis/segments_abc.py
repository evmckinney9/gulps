"""Abstract base class and data types for segment solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gulps.core.invariants import GateInvariants


@dataclass
class SegmentSolution:
    """Result of a single segment solve (local unitaries and diagnostics)."""

    u0: np.ndarray | None  # 2x2 or None if failure
    u1: np.ndarray | None  # 2x2 or None if failure
    max_residual: float  # worst-case residual component (Linf norm)
    success: bool
    metadata: dict[str, Any]  # e.g. {"nfev": nfev, "label": "easy", "attempt": 3}


class SegmentSolver(ABC):
    """Interface for solving a single segment's local unitaries.

    Given prefix_op C, basis_gate G, and a target canonical representative,
    solves for single-qubit unitaries u0, u1 such that

        U ~= G * (u1 kron u0) * C

    is locally equivalent to the target (up to invariants).

    This interface can be implemented by:
    - Cache lookups (exact match on invariants)
    - Analytic solvers (pattern-specific closed forms)
    - Numeric solvers (general optimization, always matches)
    """

    @abstractmethod
    def try_solve(
        self,
        prefix_inv: "GateInvariants",
        basis_inv: "GateInvariants",
        target_inv: "GateInvariants",
        *,
        step: int | None = 0,
        rng_seed: int | None = None,
    ) -> SegmentSolution | None:
        """Try to solve this segment.

        Returns:
            SegmentSolution if this solver can handle the segment, None otherwise.

        The solver should:
        1. Check if it can handle this segment (pattern matching)
        2. If yes, compute and return the solution
        3. If no, return None to try the next solver
        """
        pass
