"""Numerical decomposition using scipy.minimize.

Primary implementation: https://github.com/evmckinney9/transpile_benchy/blob/main/src/transpile_benchy/utilities/numerical_decomp.py
I am recoding it here to grab some of the core functionality.
The main difference is in transpile_benchy I allow the ansatz to be extended. Here, I will treat the ansatz as fixed.
If the ansatz does not converge, fail rather than adding more gates and trying again.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from monodromy.coordinates import (
    positive_canonical_to_monodromy_coordinate,
    unitary_to_monodromy_coordinate,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit_algorithms.optimizers import NELDER_MEAD, P_BFGS, Optimizer, OptimizerResult
from scipy.optimize import OptimizeResult
from weylchamber import J_T_LI, c1c2c3, canonical_gate, g1g2g3, g1g2g3_from_c1c2c3

from hetero_isas.zz_parallel_drive.ansatz import BasicCircuitAnsatz, CircuitAnsatz


class ObjectiveFunction(ABC):
    """Similarity between two operators (smaller output is more similiar)."""

    def __call__(input: Gate | Operator, target: Gate | Operator) -> float:
        """Evaluate the cost function of the decomposition.

        Args:
        input (Gate): unitary given by current iteration of parameter optimization
        target (Gate): the target unitary to be decomposed
        """
        return NotImplementedError


class HilbertSchmidt(ObjectiveFunction):
    """Use Hilbert-Schmidt inner product as the cost function."""

    def __call__(self, input, target) -> float:  # noqa: D102
        input, target = input.to_matrix(), target.to_matrix()
        dim = target.shape[0]
        return 1 - np.abs(np.trace(input @ np.matrix.getH(target))) / dim


class Infidelity(ObjectiveFunction):
    """Use process-fidelity as the cost function."""

    def __call__(self, input, target) -> float:  # noqa: D102
        if isinstance(target, np.ndarray):
            target = Operator(target)
        return 1 - average_gate_fidelity(input, target)


class MakhlinFunctional(ObjectiveFunction):
    """Use Makhlin functional as the cost function."""

    def __call__(self, input, target) -> float:  # noqa: D102
        # return J_T_LI(input.to_matrix(), target.to_matrix())
        input = input.to_matrix()
        if isinstance(target, (Gate, Operator)):
            target = target.to_matrix()
            target = g1g2g3(target)
        return np.sum(np.abs(np.array(g1g2g3(input)) - np.array(target)) ** 2)


class NumericalDecomposer(ABC):
    """Circuit decomposition using numerical optimization."""

    def __init__(
        self,
        ansatz: CircuitAnsatz,
        obj_fun: ObjectiveFunction,
        optimizer: Optimizer | str,
        restart_attempts: int,
        convergence_threshold: float,
        allow_fail: bool = False,
    ):
        """Initialize by passing an ansatz (parameterized qc)."""
        self.ansatz = ansatz
        self.num_params = ansatz.num_params
        self.obj_fun = obj_fun
        self.restart_attempts = restart_attempts
        self.convergence_threshold = convergence_threshold
        self._final_inplace = False
        self.allow_fail = allow_fail

        if isinstance(optimizer, str):
            if optimizer == "bfgs":
                self.optimizer = P_BFGS()
            elif optimizer == "nelder-mead":
                self.optimizer = NELDER_MEAD()
            else:
                raise ValueError("invalid optimzer str")
        else:
            self.optimizer = optimizer

        if isinstance(self.obj_fun, MakhlinFunctional) and not isinstance(
            self.optimizer, NELDER_MEAD
        ):
            raise warnings.warn("makhlin works best with nelder-mead method")
        if (
            isinstance(self.obj_fun, Infidelity)
            or isinstance(self.obj_fun, HilbertSchmidt)
        ) and not isinstance(self.optimizer, P_BFGS):
            raise warnings.warn("infdelity works best with bfgs method")

    def _construct_obj(self, target) -> Callable:
        _ansatz = self.ansatz
        _obj_fun = self.obj_fun

        def _obj_wrapper(x0):
            input = Operator(_ansatz._assign_parameters(x0))
            return _obj_fun(input, target)

        return _obj_wrapper

    def _run(self, target: Gate | Operator) -> QuantumCircuit:
        """Return circuit with fit parameters for the decomposition task."""
        # construct a new objective function for the given target
        obj_fun = self._construct_obj(target)
        x0 = np.random.uniform(-1.0 * np.pi, 1.0 * np.pi, self.num_params)

        for _ in range(self.restart_attempts):
            self._ret = self.optimizer.minimize(obj_fun, x0)
            if self._ret.fun <= self.convergence_threshold:
                return self.ansatz._assign_parameters(
                    self._ret.x, inplace=self._final_inplace
                )
        if self.allow_fail:
            return self.ansatz._assign_parameters(
                self._ret.x, inplace=self._final_inplace
            )
        raise RuntimeError(
            "Optimizer failed to meet required threshold! Try again with a more expressive PQC"
        )

    @abstractmethod
    def __call__(self, target: Gate | Operator) -> QuantumCircuit:
        """Invoke the decomposition routine on target."""
        raise NotImplementedError


class CircuitAnsatzDecomposer(NumericalDecomposer):
    """Numerical optimization for circuit decompositoin.

    Uses hilbert-schmidt norm and bfgs.
    """

    def __init__(self, ansatz: CircuitAnsatz):
        """Instantiate a decomposer instance."""
        warnings.warn(
            "using this decomposer is pointless :) \
                      that is because you can get the exterior 1Q gates using KAK instead."
        )
        obj_fun = Infidelity()
        optimizer = P_BFGS()
        if isinstance(ansatz, BasicCircuitAnsatz):
            if ansatz.outside_layer_u != (3, 3):
                warnings.warn(
                    "Not using fully expressive 1Qs, but cost function tests for exact unitary equivalence."
                )
        super().__init__(
            ansatz=ansatz,
            obj_fun=obj_fun,
            optimizer=optimizer,
            restart_attempts=1,
            convergence_threshold=1e-10,
        )

    def __call__(self, target: Gate | Operator) -> QuantumCircuit:  # noqa: D102
        try:
            return super()._run(target)
        except RuntimeError as e:
            raise e


class InvariantConvergenceWarning(UserWarning):  # noqa: D101
    pass


class ClosestInvariantSearch(NumericalDecomposer):
    """Numerical Optimization up to local equivalence.

    Uses Makhlin invariants and Nelder-Mead.
    """

    def __init__(
        self,
        ansatz: CircuitAnsatz,
        warn_on_fail=True,
        use_callback=True,
        coord_form="c",
    ):
        """Instantiate a decomposer instance."""
        obj_fun = MakhlinFunctional()
        optimizer = NELDER_MEAD()
        self.use_callback = use_callback
        self.warn_on_fail = warn_on_fail
        self.coord_form = coord_form
        super().__init__(
            ansatz=ansatz,
            obj_fun=obj_fun,
            optimizer=optimizer,
            restart_attempts=1,
            convergence_threshold=1e-10,
        )

        # we want to track all intermediate points as valid in the point cloud
        # XXX generally bad practice to set this attribute like this,
        # but the qiskit.Optimizer does not expose all the parameters to minimize()
        def callback(intermediate_result: OptimizeResult):
            # found_coord = c1c2c3(
            #     Operator(self.ansatz._assign_parameters(intermediate_result.x))
            # )
            found_coord = unitary_to_monodromy_coordinate(
                Operator(self.ansatz._assign_parameters(intermediate_result.x)).data
            )[:3]

            self.found_coords.append(found_coord)
            # self.ansatz.point_cloud.append(found_coord)

        if self.use_callback:
            self.optimizer._kwargs = {"callback": callback}

    def __call__(self, target: Gate | Operator | tuple[float]) -> QuantumCircuit:
        """Invoke the decomposition routine on target.

        Input can be a coordinate, using positive canonical convention.
        (form='c') Or can be a gate invariant, using Makhlin invariants
        (form='g') Coordinate normalized using qucontrol conventions.
        """
        if self.use_callback:
            self.found_coords = []

        if not isinstance(target, (Gate, Operator)):
            assert len(target) == 3
            # FIXME, handling multiple coordinate conventions poorly
            if self.coord_form == "c":
                target = g1g2g3_from_c1c2c3(target)
            else:
                assert self.coord_form == "g"
        try:
            return super()._run(target)
        except RuntimeError as e:
            if self.warn_on_fail:
                warnings.warn(
                    "Failed to find solution, returning closest found point.",
                    InvariantConvergenceWarning,
                )
                return self.ansatz._assign_parameters(self._ret.x)
            else:
                raise e
        finally:
            if self.use_callback:
                # nested coordinate list
                self.ansatz.point_cloud.append(self.found_coords)
