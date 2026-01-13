import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gulps.core.invariants import GateInvariants
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

logger = logging.getLogger(__name__)


@dataclass
class SegmentCacheEntry:
    """Cache entry storing keys and solution with hit count for LFU eviction."""

    prefix_key: tuple
    basis_key: tuple
    target_key: tuple
    solution: SegmentSolution
    hit_count: int = 0


class SegmentCache(SegmentSolver):
    """Cache-based segment solver with LFU eviction."""

    def __init__(self, max_entries_per_step: int = 2):
        self._entries: dict[int, list[SegmentCacheEntry]] = {}
        self._max_entries = max_entries_per_step
        self.hits = 0
        self.misses = 0

    def try_solve(
        self,
        step: int,
        prefix_inv: GateInvariants,
        basis_inv: GateInvariants,
        target_inv: GateInvariants,
        *,
        rng_seed: int | None = None,
    ) -> Optional[SegmentSolution]:
        """Return cached solution if available, None otherwise."""
        return self._lookup(step, prefix_inv, basis_inv, target_inv)

    def _lookup(
        self,
        step: int,
        prefix_inv: GateInvariants,
        basis_inv: GateInvariants,
        target_inv: GateInvariants,
    ) -> Optional[SegmentSolution]:
        """Lookup cached solution using fast _key comparison."""
        entries = self._entries.get(step)
        if entries is None:
            self.misses += 1
            return None

        pk, bk, tk = prefix_inv._key, basis_inv._key, target_inv._key
        for entry in entries:
            if (
                entry.target_key == tk
                and entry.prefix_key == pk
                and entry.basis_key == bk
            ):
                entry.hit_count += 1
                self.hits += 1
                return entry.solution

        self.misses += 1
        return None

    def put(
        self,
        step: int,
        prefix_inv: GateInvariants,
        basis_inv: GateInvariants,
        target_inv: GateInvariants,
        solution: SegmentSolution,
    ) -> None:
        """Store solution with LFU eviction when at capacity."""
        if step not in self._entries:
            self._entries[step] = []

        entries = self._entries[step]
        pk, bk, tk = prefix_inv._key, basis_inv._key, target_inv._key

        # Check if already cached
        for entry in entries:
            if (
                entry.target_key == tk
                and entry.prefix_key == pk
                and entry.basis_key == bk
            ):
                return

        new_entry = SegmentCacheEntry(
            prefix_key=pk, basis_key=bk, target_key=tk, solution=solution, hit_count=0
        )

        if len(entries) < self._max_entries:
            entries.append(new_entry)
        else:
            min_idx = min(range(len(entries)), key=lambda i: entries[i].hit_count)
            entries[min_idx] = new_entry

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        total_entries = sum(len(e) for e in self._entries.values())
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "entries": total_entries,
            "steps": len(self._entries),
        }
