"""Specialized analytic segment solvers for known patterns."""

from typing import Optional

import numpy as np

from gulps.core.invariants import GateInvariants
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver


class LinearWeylSolver(SegmentSolver):
    """Handles segments where Weyl coordinates add linearly: (a,b,c) + (x,y,z) = (a+x, b+y, c+z).

    When the composite gate's Weyl coordinates are exactly the sum of the
    prefix and basis gate coordinates, the local unitaries are identity.
    This occurs for certain gate combinations in the Weyl chamber.
    """

    def __init__(self, tolerance: float = 1e-6):
        """Set match tolerance for linear-Weyl detection."""
        self.tolerance = tolerance

    def try_solve(
        self,
        step: int,
        prefix_inv: GateInvariants,
        basis_inv: GateInvariants,
        target_inv: GateInvariants,
        *,
        rng_seed: int | None = None,
    ) -> Optional[SegmentSolution]:
        """Check if Weyl coords add linearly, return identity solution if so."""
        # Get Weyl coordinates
        prefix_weyl = prefix_inv.weyl
        basis_weyl = basis_inv.weyl
        target_weyl = target_inv.weyl

        # Check if target = prefix + basis (element-wise)
        expected = prefix_weyl + basis_weyl
        diff = np.abs(target_weyl - expected)

        if np.max(diff) > self.tolerance:
            return None

        # Pattern matches - return identity solution
        return SegmentSolution(
            u0=np.eye(2, dtype=np.complex128),
            u1=np.eye(2, dtype=np.complex128),
            max_residual=float(np.max(diff)),
            success=True,
            metadata={"solver": "linear_weyl", "type": "analytic"},
        )
