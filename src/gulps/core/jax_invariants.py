"""JAX implementations of two-qubit gate invariants.

This module provides JAX/JIT-compiled implementations of:
- Makhlin invariants (G1, G2, G3)
- Weyl coordinates (c1, c2, c3)

These are used in numerical optimization to match the non-local content
of two-qubit gates (the part invariant under local single-qubit operations).
"""

import jax.numpy as jnp
from jax import jit

# Magic basis for two-qubit decomposition
MAGIC = jnp.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=jnp.complex128,
) / jnp.sqrt(2)
MAGIC_DAG = MAGIC.conj().T

# SySy in computational basis
_SY = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
_SYSY = jnp.kron(_SY, _SY)

# Transform matrix for Weyl coordinates
_M = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=jnp.float64)


@jit
def makhlin_invariants(U: jnp.ndarray) -> jnp.ndarray:
    """Compute Makhlin invariants [Re(G1), Im(G1), Re(G2)].

    WARNING: Determinant normalization is skipped for performance.
    This appears stable in practice but may cause issues for gates
    far from SU(4). If optimization fails mysteriously, try restoring:
        det_um = jnp.linalg.det(Um)
        det_um /= jnp.abs(det_um)
    """
    Um = MAGIC_DAG @ U @ MAGIC

    # Skipping det normalization - risky but faster
    det_um = jnp.linalg.det(Um)
    det_um /= jnp.abs(det_um)
    # det_um = 1.0

    M = Um.T @ Um
    t1 = jnp.trace(M)
    t1s = t1 * t1
    t2 = jnp.trace(M @ M)

    g1 = t1s / (16.0 * det_um)
    g2 = (t1s - t2) / (4.0 * det_um)

    return jnp.array([jnp.real(g1), jnp.imag(g1), jnp.real(g2)], dtype=jnp.float64)


@jit
def _su4_normalize(U: jnp.ndarray) -> jnp.ndarray:
    """Project global phase so det(U)=1 using principal log."""
    detU = jnp.linalg.det(U)
    phase = jnp.exp(-0.25 * jnp.log(detU))
    return U * phase


@jit
def weyl_coordinates(U: jnp.ndarray) -> jnp.ndarray:
    """Compute Weyl chamber coordinates (c1, c2, c3) in units of π.

    Output is folded to the canonical Weyl chamber:
    c1 >= c2 >= c3 >= 0, c1 <= 0.5

    Reference points:
    - (0, 0, 0) = Identity
    - (0.5, 0, 0) = CNOT
    - (0.5, 0.5, 0) = DCNOT
    - (0.5, 0.5, 0.5) = SWAP

    WARNING: SU(4) normalization is skipped for performance.
    Global phase cancels in U @ U_tilde, but if issues arise, restore:
        U = _su4_normalize(U)
    """
    U = _su4_normalize(U)
    U_tilde = _SYSY @ U.T @ _SYSY
    gamma = U @ U_tilde

    ev = jnp.linalg.eigvals(gamma)
    two_S = jnp.angle(ev) / jnp.pi
    two_S = jnp.where(two_S <= -0.5, two_S + 2.0, two_S)

    S = jnp.sort(two_S / 2.0)[::-1]

    n = jnp.round(jnp.sum(S)).astype(jnp.int32)
    ones_mask = (jnp.arange(4, dtype=jnp.int32) < n).astype(jnp.float64)
    S = S - ones_mask
    S = jnp.roll(S, -n)

    c = _M @ S[:3]

    c1, c2, c3 = c[0], c[1], c[2]
    c1 = jnp.where(c3 < 0.0, 1.0 - c1, c1)
    c3 = jnp.where(c3 < 0.0, -c3, c3)

    c2 = jnp.maximum(c2, 0.0)
    c3 = jnp.maximum(c3, 0.0)

    return jnp.array([c1, c2, c3], dtype=jnp.float64)
