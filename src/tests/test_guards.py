"""Input-validation guards on user-facing constructors."""

import numpy as np
import pytest
from qiskit.circuit.library import iSwapGate

from gulps import GulpsDecomposer
from gulps.core.isa import DiscreteISA


def test_invalid_target_raises():
    """Malformed target raises ValueError, not a Rust panic or silent wrong answer."""
    dec = GulpsDecomposer(isa=DiscreteISA([iSwapGate().power(1.0)], [1.0], ["iswap"]))
    with pytest.raises(ValueError):
        dec(2 * np.eye(4, dtype=complex))


def test_invalid_isa_cost_raises():
    """Non-finite or negative gate costs raise ValueError at ISA construction."""
    with pytest.raises(ValueError):
        DiscreteISA([iSwapGate().power(1.0)], [-1.0], ["iswap"])
