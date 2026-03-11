# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
