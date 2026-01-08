"""GULPS python module."""

# Configure JAX to use 64-bit precision before importing any JAX code
import jax

jax.config.update("jax_enable_x64", True)

from ._internal.logging_config import logger
from .core.invariants import GateInvariants
from .gulps_decomposer import GulpsDecomposer
from .qiskit_ext.synthesis_pass import GulpsDecompositionPass
from .qiskit_ext.synthesis_plugin import GulpsSynthesisPlugin
