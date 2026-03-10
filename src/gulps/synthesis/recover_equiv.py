"""Local equivalence recovery via Weyl/KAK decomposition."""

import logging

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.synthesis.two_qubit import TwoQubitWeylDecomposition

from gulps.core.coordinates import PAULI_X, PAULI_Y, PAULI_Z

logger = logging.getLogger(__name__)

# Tolerance for Weyl coordinate matching in branch selection.
# This is a mathematical constant (not user-tunable): the solver's
# weyl_conv_tol is 1e-5 and accumulated segment error stays well
# below 2e-5 for the direct/rho branches to be distinguishable.
_WEYL_MATCH_TOL = 2e-5


def _closest_unitary(A: np.ndarray) -> np.ndarray:
    """Closest unitary to A via polar decomposition (SVD)."""
    V, _, Wh = np.linalg.svd(A)
    return V @ Wh


def recover_local_equivalence(
    U_target: np.ndarray,
    U_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Promote local equivalence to unitary equality by finding local corrections.

    Find single-qubit corrections k1, k2, k3, k4 and a global phase so that:

    .. code-block:: none

        U_target ~= (k4 x k3) . U_basis . (k2 x k1) . exp(i * global_phase)

    Uses Qiskit's ``TwoQubitWeylDecomposition`` (KAK).  Three cases:

      1. Direct Weyl match: corrections are K-factor ratios.
      2. Rho-reflected match: (a,b,c) vs (pi/2-a, b, -c), inserts Paulis.
      3. Frobenius fallback: computes both branches, picks lowest error.
         Needed when a ~ pi/4 makes the branches indistinguishable.
    """
    tol = _WEYL_MATCH_TOL

    # Weyl decompose both unitaries
    T = TwoQubitWeylDecomposition(U_target, fidelity=1.0)
    try:
        B = TwoQubitWeylDecomposition(U_basis, fidelity=1.0)
    except QiskitError:
        U_basis_closest = _closest_unitary(U_basis)
        B = TwoQubitWeylDecomposition(U_basis_closest, fidelity=1.0)

    a1, b1, c1 = T.a, T.b, T.c
    a2, b2, c2 = B.a, B.b, B.c
    diffs = np.abs([a1 - a2, b1 - b2, c1 - c2])

    # 1) exact Weyl match?
    if np.allclose([a1, b1, c1], [a2, b2, c2], atol=tol):
        k4 = T.K1l @ B.K1l.conj().T
        k3 = T.K1r @ B.K1r.conj().T
        k2 = B.K2l.conj().T @ T.K2l
        k1 = B.K2r.conj().T @ T.K2r
        return k1, k2, k3, k4, (T.global_phase - B.global_phase)

    # 2) rho reflection case: target=(a,b,c), basis=(pi/2-a,b,-c)?
    if np.allclose([a1, b1, c1], [np.pi / 2 - a2, b2, -c2], atol=tol):
        logger.debug("Detected rho reflect; inserting Pauli corrections.")

        k4 = T.K1l @ PAULI_Y @ B.K1l.conj().T
        k3 = T.K1r @ B.K1r.conj().T
        k2 = B.K2l.conj().T @ PAULI_Z @ T.K2l
        k1 = B.K2r.conj().T @ PAULI_X @ T.K2r
        return k1, k2, k3, k4, (T.global_phase - B.global_phase)

    # 3) Frobenius-error fallback: try both branches, pick lowest
    #    reconstruction error. Needed when a ~= pi/4 makes the branches
    #    indistinguishable from Weyl coordinates alone.
    def _direct_corrections():
        k4 = T.K1l @ B.K1l.conj().T
        k3 = T.K1r @ B.K1r.conj().T
        k2 = B.K2l.conj().T @ T.K2l
        k1 = B.K2r.conj().T @ T.K2r
        return k1, k2, k3, k4, (T.global_phase - B.global_phase)

    def _rho_corrections():
        k4 = T.K1l @ PAULI_Y @ B.K1l.conj().T
        k3 = T.K1r @ B.K1r.conj().T
        k2 = B.K2l.conj().T @ PAULI_Z @ T.K2l
        k1 = B.K2r.conj().T @ PAULI_X @ T.K2r
        return k1, k2, k3, k4, (T.global_phase - B.global_phase)

    def _recon_error(k1, k2, k3, k4, gphase):
        recon = np.exp(1j * gphase) * np.kron(k4, k3) @ U_basis @ np.kron(k2, k1)
        return np.linalg.norm(U_target - recon, ord="fro")

    direct = _direct_corrections()
    rho = _rho_corrections()
    err_d = _recon_error(*direct)
    err_r = _recon_error(*rho)

    best = direct if err_d <= err_r else rho
    best_err = min(err_d, err_r)

    # Only accept if the best branch actually reconstructs well
    if best_err < 0.1:
        logger.debug(
            "Fallback branch selection: direct_err=%.2e, rho_err=%.2e (chose %s)",
            err_d,
            err_r,
            "direct" if err_d <= err_r else "rho",
        )
        return best

    raise ValueError(f"Cannot recover local equivalence; Weyl differences {diffs}")
