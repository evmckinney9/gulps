"""JAX implementations of two-qubit gate invariants.

This module provides JAX/JIT-compiled implementations of various invariant
coordinate systems for two-qubit gates:
- Makhlin invariants (G1, G2, G3)
- Weyl coordinates (a, b, c)
- Monodromy coordinates

These implementations are designed for use in numerical optimization and can
also be used by other parts of the codebase to avoid duplication.

The module also provides numpy-compatible wrappers (using device_get) that
can be used outside of JAX contexts (e.g., in GateInvariants).
"""

import jax.numpy as jnp
import numpy as np
from jax import device_get, jit

# Magic basis for two-qubit decomposition
MAGIC = jnp.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=jnp.complex128,
) / jnp.sqrt(2)
MAGIC_DAG = MAGIC.conj().T


@jit
def makhlin_invariants(U: jnp.ndarray) -> jnp.ndarray:
    """This should be the same as crates/synthesis/src/two_qubit_decompose.rs

    Note: qiskit does not normalize the determinant /= abs(det)

    However, I need it defined in jax to use autodiff for optimization.
    """
    # Transform to magic basis
    Um = MAGIC_DAG @ (U @ MAGIC)

    # Normalize determinant to remove global phase
    det_um = jnp.linalg.det(Um)
    det_um /= jnp.abs(det_um)

    # Compute invariants from M = Um^T @ Um
    M = Um.T @ Um
    t1 = jnp.trace(M)
    t1s = t1 * t1
    t2 = jnp.trace(M @ M)

    g1 = t1s / (16.0 * det_um)
    g2 = (t1s - t2) / (4.0 * det_um)

    return jnp.array([jnp.real(g1), jnp.imag(g1), jnp.real(g2)], dtype=jnp.float64)


# SySy in computational basis (matches common conventions used in Weyl extraction)
# If your convention differs, keep it consistent with your target canonical matrices.
_SY = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
_SYSY = jnp.kron(_SY, _SY)  # 4x4

_M = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=jnp.float64)


@jit
def _su4_normalize(U: jnp.ndarray) -> jnp.ndarray:
    """Project global phase so det(U)=1 using principal log."""
    detU = jnp.linalg.det(U)
    # detU^(-1/4) = exp(-(1/4) log(detU)) (principal branch)
    phase = jnp.exp(-0.25 * jnp.log(detU))
    return U * phase


@jit
def weyl_c1c2c3_pi(U: jnp.ndarray) -> jnp.ndarray:
    """Compute Weyl chamber coordinates (c1,c2,c3) in units of π.
    Port of the standard Childs et al. / weylchamber algorithm.

    Output is folded to the Weyl chamber with c1 >= c2 >= c3 >= 0 and c1 <= 1/2.
    """
    U = _su4_normalize(U)

    # U_tilde = (Sy⊗Sy) U^T (Sy⊗Sy)
    U_tilde = _SYSY @ (U.T) @ _SYSY

    # gamma = U U_tilde  (already SU(4)-normalized by _su4_normalize)
    gamma = U @ U_tilde

    ev = jnp.linalg.eigvals(gamma)  # 4 eigenvalues on/near unit circle

    # two_S = angle(ev)/π  (matches your reference code)
    two_S = jnp.angle(ev) / jnp.pi

    # Wrap: if <= -0.5, add 2.0 (branch fix)
    two_S = jnp.where(two_S <= -0.5, two_S + 2.0, two_S)

    # S = sort(two_S/2) decreasing
    S = jnp.sort(two_S / 2.0)[::-1]  # shape (4,)

    # n = round(sum(S)) in {0,1,2,3,4}
    n = jnp.round(jnp.sum(S)).astype(jnp.int32)

    # subtract [1,1,...,0,0] (n ones) from S
    ones_mask = (jnp.arange(4, dtype=jnp.int32) < n).astype(jnp.float64)
    S = S - ones_mask

    # roll by -n
    S = jnp.roll(S, -n)

    # canonical coords in π units: (c1,c2,c3) = M @ S[:3]
    c = _M @ S[:3]  # shape (3,)

    # final reflection if c3 < 0
    c1, c2, c3 = c[0], c[1], c[2]
    c1 = jnp.where(c3 < 0.0, 1.0 - c1, c1)
    c3 = jnp.where(c3 < 0.0, -c3, c3)

    # Optional: clamp tiny negatives due to numeric noise
    c2 = jnp.maximum(c2, 0.0)
    c3 = jnp.maximum(c3, 0.0)

    return jnp.array([c1, c2, c3], dtype=jnp.float64)


# Registry of invariant functions for easy lookup
INVARIANT_FUNCTIONS = {
    "makhlin": makhlin_invariants,
    "weyl": weyl_c1c2c3_pi,
    # "monodromy": monodromy_coordinates,
}


def get_invariant_function(invariant_type: str):
    """Get the invariant function for a given type.

    Args:
        invariant_type: One of "makhlin", "weyl", or "monodromy".

    Returns:
        JIT-compiled function that computes the invariants.

    Raises:
        ValueError: If invariant_type is not recognized.
    """
    if invariant_type not in INVARIANT_FUNCTIONS:
        raise ValueError(
            f"Unknown invariant_type: {invariant_type}. "
            f"Must be one of {list(INVARIANT_FUNCTIONS.keys())}"
        )
    return INVARIANT_FUNCTIONS[invariant_type]


# Numpy-compatible wrappers for non-JAX contexts
def makhlin_invariants_numpy(U: np.ndarray) -> np.ndarray:
    """Compute Makhlin invariants using JAX but return numpy array.

    Args:
        U: A 4x4 numpy array representing a two-qubit gate.

    Returns:
        Numpy array of shape (3,) containing [Re(G1), Im(G1), Re(G2)].
    """
    U_jax = jnp.array(U, dtype=jnp.complex128)
    result = makhlin_invariants(U_jax)
    return np.array(result)


def weyl_coordinates_numpy(U: np.ndarray) -> np.ndarray:
    """Compute Weyl coordinates using JAX but return numpy array.

    Args:
        U: A 4x4 numpy array representing a two-qubit gate.

    Returns:
        Numpy array of shape (3,) containing [a, b, c].
    """
    U_jax = jnp.array(U, dtype=jnp.complex128)
    result = weyl_coordinates(U_jax)
    return np.array(device_get(result))


def monodromy_coordinates_numpy(U: np.ndarray) -> np.ndarray:
    """Compute monodromy coordinates using JAX but return numpy array.

    Args:
        U: A 4x4 numpy array representing a two-qubit gate.

    Returns:
        Numpy array of shape (3,) containing monodromy coordinates.
    """
    U_jax = jnp.array(U, dtype=jnp.complex128)
    result = monodromy_coordinates(U_jax)
    return np.array(device_get(result))
