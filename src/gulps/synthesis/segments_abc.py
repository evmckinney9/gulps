from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class SegmentSolution:
    u0: Optional[np.ndarray]  # 2x2 or None if failure
    u1: Optional[np.ndarray]  # 2x2 or None if failure
    residual_norm: float
    success: bool
    metadata: dict[str, Any]  # e.g. {"nfev": nfev, "label": "easy", "attempt": 3}


class SegmentSolver(ABC):
    """Interface for solving a single segment's local unitaries.

    Given prefix_op C, basis_gate G, and a target canonical representative,
    solves for single-qubit unitaries u0, u1 such that

        U ≈ G · (u1 ⊗ u0) · C

    is locally equivalent to the target (up to invariants).
    """

    @abstractmethod
    def solve_segment(
        self,
        prefix_op: np.ndarray,  # 4x4 complex np.ndarray
        basis_gate: np.ndarray,  # 4x4 complex np.ndarray
        target: np.ndarray,  # 4x4 complex np.ndarray
        *,
        rng_seed: int | None = None,
    ) -> SegmentSolution:
        pass
