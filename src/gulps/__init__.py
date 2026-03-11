"""GULPS python module."""

# Silence JAX's GPU/TPU probe warning before importing it
import logging as _logging
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _n in ("jax", "jaxlib", "absl", "jax._src.xla_bridge"):
    _logging.getLogger(_n).setLevel(_logging.ERROR)

import jax

jax.config.update("jax_enable_x64", True)

from ._internal.logging_config import logger
from .core.invariants import GateInvariants
from .gulps_decomposer import GulpsDecomposer
from .qiskit_ext.decomposer_pass import GulpsDecompositionPass
from .qiskit_ext.translation_plugin import GulpsTranslationPlugin
