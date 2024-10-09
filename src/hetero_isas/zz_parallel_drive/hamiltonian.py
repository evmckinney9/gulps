"""Utilities for initial parallel-drive testing."""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib.pyplot import cm
from qiskit.circuit.library import UnitaryGate
from qutip import qeye, sigmax, sigmaz, tensor


class Hamiltonian(ABC):
    """Abstract qutip Hamiltonian."""

    def to_unitary(self, t=1):
        """U = e^{-i H t} for time-independent pulses."""
        return UnitaryGate((-1j * self.H * t).expm().full())


class GateCost(ABC):
    """Something that defines a unit cost over interaction terms."""

    # FIXME, messy pattern??
    @staticmethod
    @abstractmethod
    def unit_gate_cost(*args):
        """Cost definde by abstract static method."""
        raise NotImplementedError


# gate cost we know in new system CZ gates are much faster than before
# e.g. X gate is 40ns, CZ is 80ns
# use these gate types to inform contributions to speedlimits
# really will depend on how the hardware works....


class ZZ_ParallelDrive(Hamiltonian):
    r"""Implements a basic parallel-drive.

    H = \gamma_1 Z_1 Z_2 + \gamma_2 I_1 Z_2 + \gamma_3 Z_1 I_2 +
    \gamma_4 X_1 I_2 + \gamma_5 I_1 X_2
    """

    zz = tensor(sigmaz(), sigmaz())
    iz = tensor(qeye(2), sigmaz())
    zi = tensor(sigmaz(), qeye(2))
    xi = tensor(sigmax(), qeye(2))
    ix = tensor(qeye(2), sigmax())
    g_labels = ["ZZ", "IZ", "ZI", "XI", "IX"]

    def __init__(self, g_zz, g_iz=0, g_zi=0, g_xi=0, g_ix=0):
        """Constructor using interaction term coefficients."""
        self.g_terms = [g_zz, g_iz, g_zi, g_xi, g_ix]
        # by qiskit convention, include a 1/2 factor
        self.g_terms = (1 / 2) * np.array(self.g_terms)
        self.h_terms = [self.zz, self.iz, self.zi, self.xi, self.ix]
        self.H = sum([g * h_term for g, h_term in zip(self.g_terms, self.h_terms)])
        self.cost = NaiveCost.unit_gate_cost(*self.g_terms)
        self.color = NaiveCost.to_color(self.cost)


class NaiveCost(GateCost):
    """Naive cost sums coefficient, excluding virtual-Z."""

    @staticmethod
    def unit_gate_cost(*args):
        """Cost proprotional to ZZ + IX + XI."""
        s = sum(np.abs(args)) - sum(np.abs(args[1:3]))
        return s / (np.pi / 4)

    @staticmethod
    def to_color(cost, max_cost=4.0):
        """Convert cost to color.

        Requires some maxmimum cost in order to interpolate the color
        space.
        """
        # if cost > max_cost:
        # raise ValueError(f"exceeded max with {cost}")
        cost = min(cost, max_cost)
        return cm.viridis(cost / max_cost)
