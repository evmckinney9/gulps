"""Shared JIT-compiled solver fixture for test suite performance."""

import pytest

from gulps.config import GulpsConfig
from gulps.synthesis.jax_lm import JaxLMSegmentSolver
from gulps.synthesis.segments_cache import SegmentCache
from gulps.synthesis.segments_solver import SegmentSynthesizer


@pytest.fixture(autouse=True, scope="session")
def _shared_jax_solver():
    """Patch SegmentSynthesizer to reuse one JIT-compiled solver."""
    solver = JaxLMSegmentSolver()
    orig_init = SegmentSynthesizer.__init__

    def _patched_init(self, config=None):
        self.config = config or GulpsConfig()
        self._cache = SegmentCache(max_entries_per_step=self.config.segment_cache_size)
        self._jax_lm_solver = solver

    SegmentSynthesizer.__init__ = _patched_init
    yield
    SegmentSynthesizer.__init__ = orig_init
