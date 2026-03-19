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

"""JAX primitives for two-qubit gate invariants and residual evaluation.

Constants, differentiable invariant functions, parameterization helpers,
and residual functions used by the Gauss-Newton solver in jax_lm.py.
"""

import jax.numpy as jnp
import numpy as np
from jax import jit

from gulps.core.coordinates import M_WEYL as _M_WEYL_NP
from gulps.core.coordinates import MAGIC as _MAGIC_NP
from gulps.core.coordinates import MAGIC_DAG as _MAGIC_DAG_NP
from gulps.core.coordinates import SYSY as _SYSY_NP

NUM_PARAMS = 8

# ---------------------------------------------------------------------------
# Constants -- JAX versions of the numpy originals in coordinates.py
# ---------------------------------------------------------------------------

MAGIC = jnp.asarray(_MAGIC_NP)
MAGIC_DAG = jnp.asarray(_MAGIC_DAG_NP)

_SYSY = jnp.asarray(_SYSY_NP)  # sy kron sy

_M = jnp.asarray(_M_WEYL_NP)  # monodromy -> Weyl linear transform


# ---------------------------------------------------------------------------
# Differentiable invariant functions
# ---------------------------------------------------------------------------


@jit
def makhlin_invariants(U):
    """Makhlin invariants [Re(G1), Im(G1), Re(G2)] of a 4x4 unitary."""
    Um = MAGIC_DAG @ U @ MAGIC
    det_um = jnp.linalg.det(Um)
    det_um = det_um / jnp.abs(det_um)
    M = Um.T @ Um
    t1 = jnp.trace(M)
    t1s = t1 * t1
    g1 = t1s / (16.0 * det_um)
    g2 = (t1s - jnp.trace(M @ M)) / (4.0 * det_um)
    return jnp.array([jnp.real(g1), jnp.imag(g1), jnp.real(g2)], dtype=jnp.float64)


@jit
def weyl_coordinates(U):
    """Weyl chamber coordinates (c1, c2, c3) in units of pi.

    Folded to canonical Weyl chamber: c1 >= c2 >= c3 >= 0, c1 <= 0.5.
    """
    U = U * jnp.exp(-0.25 * jnp.log(jnp.linalg.det(U)))  # SU(4) project
    U_tilde = _SYSY @ U.T @ _SYSY
    ev = jnp.linalg.eigvals(U @ U_tilde)
    two_S = jnp.angle(ev) / jnp.pi
    two_S = jnp.where(two_S < -0.5 + 1e-12, two_S + 2.0, two_S)
    S = jnp.sort(two_S / 2.0)[::-1]
    n = jnp.round(jnp.sum(S)).astype(jnp.int32)
    S = S - (jnp.arange(4, dtype=jnp.int32) < n).astype(jnp.float64)
    S = jnp.roll(S, -n)
    c = _M @ S[:3]
    c1, c2, c3 = c[0], c[1], c[2]
    c1 = jnp.where(c3 < 0, 1.0 - c1, c1)
    c3 = jnp.where(c3 < 0, -c3, c3)
    return jnp.array([c1, jnp.maximum(c2, 0), jnp.maximum(c3, 0)], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# JAX matrix helper
# ---------------------------------------------------------------------------


def _get_jax_matrix(inv):
    """Return a 4x4 unitary for a GateInvariants, caching on first call.

    ISA gates (have ``_unitary``) are cached as DeviceArrays — they are reused
    across many decompositions, so paying the ``jnp.asarray`` conversion once is
    worth it.  LP intermediates (no ``_unitary``) are cached as numpy arrays
    because they are used once; the JIT boundary converts at negligible cost.
    """
    try:
        return inv._jax_matrix
    except AttributeError:
        if inv._unitary is not None:
            # ISA gate: cache as DeviceArray (reused across decompositions)
            mat = np.asarray(inv._unitary, dtype=np.complex128)
            inv._jax_matrix = jnp.asarray(mat, dtype=jnp.complex128)
        else:
            # LP intermediate: cache as numpy (single-use, JIT converts)
            mat = inv.canonical_matrix
            if mat.dtype != np.complex128:
                mat = mat.astype(np.complex128)
            inv._jax_matrix = mat
        return inv._jax_matrix


# ---------------------------------------------------------------------------
# Parameterization: 8D quaternion pairs -> SU(2) x SU(2)
# ---------------------------------------------------------------------------


def _params_to_unitaries(params):
    """8D quaternion params -> two SU(2) matrices (u0, u1)."""
    eps = 1e-12

    def quat_to_su2(q):
        q = q / jnp.maximum(jnp.linalg.norm(q), eps)
        w, x, y, z = q
        a = w + 1j * z
        b = x + 1j * y
        return jnp.array([[a, b], [-jnp.conj(b), jnp.conj(a)]], dtype=jnp.complex128)

    return quat_to_su2(params[:4]), quat_to_su2(params[4:])


def _kron_2x2(u0, u1):
    """Kronecker product of two 2x2 matrices."""
    return jnp.block([[u0[0, 0] * u1, u0[0, 1] * u1], [u0[1, 0] * u1, u0[1, 1] * u1]])


# ---------------------------------------------------------------------------
# Residual functions
# ---------------------------------------------------------------------------


@jit
def _weyl_residual(x, prefix_op, basis_gate, target_inv):
    """Weyl-coordinate residual (used for polish stage)."""
    u0, u1 = _params_to_unitaries(x)
    U = basis_gate @ _kron_2x2(u1, u0) @ prefix_op
    return target_inv - weyl_coordinates(U)


@jit
def _makhlin_residual_fused(x, prefix_magic, magic_basis, target_packed):
    """Fused Makhlin residual with precomputed magic-basis transforms.

    Eliminates 2 matmuls + 1 det per eval by absorbing MAGIC into
    the constant prefix/basis and precomputing the determinant phase.
    target_packed = [Re(G1), Im(G1), Re(G2), Re(det_phase), Im(det_phase)].
    """
    target_inv = target_packed[:3]
    det_phase = target_packed[3] + 1j * target_packed[4]
    u0, u1 = _params_to_unitaries(x)
    Um = magic_basis @ _kron_2x2(u1, u0) @ prefix_magic
    M = Um.T @ Um
    t1 = jnp.trace(M)
    t1s = t1 * t1
    g1 = t1s / (16.0 * det_phase)
    g2 = (t1s - jnp.trace(M @ M)) / (4.0 * det_phase)
    inv = jnp.array([jnp.real(g1), jnp.imag(g1), jnp.real(g2)], dtype=jnp.float64)
    return target_inv - inv


@jit
def _precompute_makhlin_args(prefix_op, basis_gate, target_makhlin):
    """Precompute fused matrices + constant det phase for Makhlin solver."""
    magic_basis = MAGIC_DAG @ basis_gate
    prefix_magic = prefix_op @ MAGIC
    det_um = jnp.linalg.det(magic_basis @ prefix_magic)
    det_phase = det_um / jnp.maximum(jnp.abs(det_um), 1e-30)
    return (
        prefix_magic,
        magic_basis,
        jnp.concatenate(
            [target_makhlin, jnp.array([jnp.real(det_phase), jnp.imag(det_phase)])]
        ),
    )


@jit
def _matrix_residual(x, prefix_op, basis_gate, target_mat):
    """Matrix matching residual: G@L1@C - L2@T@L3, 32 reals.

    Solves for three local equivalences (L1, L2, L3) such that
    basis_gate @ L1 @ prefix_op = L2 @ target_mat @ L3.
    Full-rank Jacobian at c1=c2, bypassing the Makhlin rank-2 obstruction.
    """
    u0_1, u1_1 = _params_to_unitaries(x[:8])
    u0_2, u1_2 = _params_to_unitaries(x[8:16])
    u0_3, u1_3 = _params_to_unitaries(x[16:24])
    LHS = basis_gate @ _kron_2x2(u1_1, u0_1) @ prefix_op
    RHS = _kron_2x2(u1_2, u0_2) @ target_mat @ _kron_2x2(u1_3, u0_3)
    diff = LHS - RHS
    return jnp.concatenate([diff.real.ravel(), diff.imag.ravel()])
