"""Numerical decomposition using scipy.minimize.

Fit 1Q gate parameters to gate decomposiiton given a solution to
monodromy linear programming.
"""

import jax.numpy as jnp
import numpy as np
from jax import config, jacfwd, jit
from jax.random import PRNGKey, uniform
from jaxopt import GaussNewton, LevenbergMarquardt, ScipyLeastSquares
from jaxopt.linear_solve import solve_cg
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import RVGate, U3Gate, UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

# from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
from scipy.optimize import least_squares

from gulps.invariants import mono_coordinates_to_CAN, recover_local_equivalence
from hetero_isas.monodromy_lp.mono_lp_result import MonodromyLPDecomposerResult

config.update("jax_enable_x64", True)

MAGIC = jnp.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=jnp.cdouble,
) / jnp.sqrt(2)

ATOL = 4e-14  # arbitrary, a little bit less than float
CONV_TOL = 0  # forces to use all alloted iterations


@jit
def two_qubit_local_invariants(U):
    # same as qiskit.synthesis.local_invariance.two_qubit_local_invariants
    # but with jax :)
    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    det_um = jnp.complex128(jnp.linalg.det(Um))
    M = jnp.dot(Um.T, Um)
    m_tr2 = jnp.trace(M)
    m_tr2 *= m_tr2
    G1 = m_tr2 / (16 * det_um)
    G2 = (m_tr2 - jnp.trace(M.dot(M))) / (4 * det_um)
    return jnp.array([G1.real, G1.imag, G2.real], dtype=jnp.double)


@jit
def _rv(v):
    # taken from qiskit.library.RV
    # half_angle = jnp.sqrt(jnp.dot(v, v)) / 2
    half_angle = jnp.double(jnp.linalg.norm(v) / 2)
    nx, ny, nz = v / (2 * half_angle)
    sin = jnp.sin(half_angle)
    cos = jnp.cos(half_angle)
    return jnp.array(
        [
            [cos - 1j * nz * sin, (-ny - 1j * nx) * sin],
            [(ny - 1j * nx) * sin, cos + 1j * nz * sin],
        ],
        dtype=jnp.cdouble,
    )


@jit
def _u3(v):
    theta, phi, lam = v
    cos = jnp.cos(theta / 2)
    sin = jnp.sin(theta / 2)
    return jnp.array(
        [
            [cos, -jnp.exp(1j * lam) * sin],
            [jnp.exp(1j * phi) * sin, jnp.exp(1j * (phi + lam)) * cos],
        ],
        dtype=jnp.cdouble,
    )


@jit
def _construct_op(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
):
    # extended parameter vector
    return (
        basis_gate
        @ jnp.kron(_rv(x[:3]), _rv(x[3:6]))
        @ jnp.kron(_rv(x[6:]), _rv(x[6:]))
        @ prefix_op
    )


@jit
def _construct_op_simple(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
):
    # extended parameter vector
    return basis_gate @ jnp.kron(_rv(x[:3]), _rv(x[3:])) @ prefix_op


@jit
def _construct_op_u3(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
):
    # extended parameter vector
    return basis_gate @ jnp.kron(_u3(x[:3]), _u3(x[3:])) @ prefix_op


@jit
def _objective_function(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
    target_inv: jnp.ndarray,
):
    """Use Makhlin functional as the cost function."""
    construct_op = _construct_op(x, prefix_op, basis_gate)
    construct_inv = two_qubit_local_invariants(construct_op)
    diff = target_inv - construct_inv
    return diff


@jit
def _objective_function_simple(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
    target_inv: jnp.ndarray,
):
    """Use Makhlin functional as the cost function."""
    construct_op = _construct_op_simple(x, prefix_op, basis_gate)
    construct_inv = two_qubit_local_invariants(construct_op)
    diff = target_inv - construct_inv
    return diff


@jit
def _objective_function_u3(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
    target_inv: jnp.ndarray,
):
    """Use Makhlin functional as the cost function."""
    construct_op = _construct_op_u3(x, prefix_op, basis_gate)
    construct_inv = two_qubit_local_invariants(construct_op)
    diff = target_inv - construct_inv
    return diff


j_lm = LevenbergMarquardt(
    residual_fun=_objective_function,
    maxiter=256,  # 128
    solver=solve_cg,  # {'qr'}
    implicit_diff=False,
    materialize_jac=True,  #  NOTE way faster!!
    jit=True,
    tol=CONV_TOL,
    xtol=CONV_TOL,
    gtol=CONV_TOL,
)

j_lm_simple = LevenbergMarquardt(
    residual_fun=_objective_function_simple,
    maxiter=256,  # 128
    solver=solve_cg,  # {'qr'}
    implicit_diff=False,
    materialize_jac=True,  #  NOTE way faster!!
    jit=True,
    tol=CONV_TOL,
    xtol=CONV_TOL,
    gtol=CONV_TOL,
)

j_lm_u3 = LevenbergMarquardt(
    residual_fun=_objective_function_u3,
    maxiter=32,  # 128
    solver=solve_cg,  # {'qr'}
    implicit_diff=False,
    materialize_jac=True,
    jit=True,
    tol=CONV_TOL,
    xtol=CONV_TOL,
    gtol=CONV_TOL,
)


class NumericDecompConvergenceWarning(UserWarning):  # noqa: D101
    def __init__(self):  # noqa: D107
        message = "Failed to converge to solution within precision tolerance. If allowed, \
            by NumericalDecomposer.allow_fail, will return best solution instead."
        super().__init__(message)


class MonodromyLPNumericalDecomposer:
    """Circuit decomposition using numerical optimization.

    NOTE refactor: previously this was an abstract class that was subclassed by a decomposer
    that used unitary equivalence + BFGS versus local invariant equivalence + Nelder-Mead.

    Maybe we have clsmethod that constructs new ansatz from CircuitAnsatz?
    The reason is I want to change this class to only deal with matrices never Operator,QuantumCircuit

    So I think we should get rid of NumericalDecomposer.ansatz attribute. Instead the ansatz
    if the objective_function...
    """

    def __init__(
        self,
        allow_fail: bool = False,
    ):
        """Initialize by passing an ansatz (parameterized qc)."""
        # XXX should never get close to using this many
        self.max_restart_attempts = 16
        self.allow_fail = allow_fail
        self._use_simple = True
        self._use_u3 = False

    def _optimize_single_qubit_gates(
        self,
        prefix_op,
        basis_gate,
        target,
        seed=0,
        debugging=False,
        lp_result: "MonodromyLPDecomposerResult" = None,
    ) -> list[Instruction]:
        """Return circuit with fit parameters for the decomposition task."""
        # parse target input into jax types
        j_target_inv = two_qubit_local_invariants(target)
        j_prefix_op = jnp.array(prefix_op, dtype=jnp.complex128)
        j_basis_gate = jnp.array(basis_gate, dtype=jnp.complex128)

        restart_attempts, total_nfev, success_nfev, success = 0, 0, 0, False
        while not success and restart_attempts < self.max_restart_attempts:
            if self._use_simple or self._use_u3:
                x0 = uniform(
                    PRNGKey(seed + restart_attempts),
                    (6,),
                    minval=-2 * jnp.pi,
                    maxval=2 * jnp.pi,
                )
            else:
                x0 = uniform(
                    PRNGKey(seed + restart_attempts),
                    (9,),  # extended parameter vector (otherwise 6)
                    minval=-2 * jnp.pi,
                    maxval=2 * jnp.pi,
                )

            if self._use_u3:
                _ret = j_lm_u3.run(x0, j_prefix_op, j_basis_gate, j_target_inv)
            if self._use_simple:
                _ret = j_lm_simple.run(x0, j_prefix_op, j_basis_gate, j_target_inv)
            else:
                _ret = j_lm.run(x0, j_prefix_op, j_basis_gate, j_target_inv)

            # NOTE retrieve result data from j_lm
            (success, params, num_fun_eval, fun_val) = (
                _ret.state.value <= ATOL,
                _ret.params,
                _ret.state.iter_num,
                _ret.state.residual,
            )
            restart_attempts += 1
            success_nfev = int(num_fun_eval)
            total_nfev += success_nfev

        # save back metadata to result object
        if debugging and lp_result:
            lp_result._setter_1q_numerics(
                success, restart_attempts, total_nfev, success_nfev, fun_val
            )

        if not (success or self.allow_fail):
            raise NumericDecompConvergenceWarning()

        # convert solution into list of qiskit.Instructions
        np_params = np.array(params)
        if self._use_simple or self._use_u3:
            new_prefix = _construct_op_simple(params, j_prefix_op, j_basis_gate)
            if self._use_u3:
                inner_local_gates = [
                    U3Gate(*np_params[:3]),
                    U3Gate(*np_params[3:]),
                ]
            else:
                inner_local_gates = [
                    RVGate(*np_params[:3]),
                    RVGate(*np_params[3:]),
                ]
        else:
            new_prefix = _construct_op(params, j_prefix_op, j_basis_gate)
            inner_local_gates = [
                RVGate(*np_params[:3]),
                RVGate(*np_params[3:6]),
                RVGate(*np_params[6:]),
            ]

        try:
            k_vec = recover_local_equivalence(target, new_prefix)
            global_phase = k_vec[-1]
            outer_local_gates = [UnitaryGate(k) for k in k_vec[:-1]]
        except ValueError as e:
            print(e)
            if not self.allow_fail:
                raise e
            # XXX meaningless if has already failed
            # but need to give it values so rest of code doesn't break
            outer_local_gates = [RVGate(0, 0, 0) for _ in range(4)]
            global_phase = 0

        return outer_local_gates + inner_local_gates, global_phase

    def _segments_solve(
        self,
        target_op: np.ndarray,
        lp_result: "MonodromyLPDecomposerResult",
        debugging=False,
    ):
        """Iterates using CAN as prefix placeholders.

        Therefore, doesn't rely on any previous iteration results, since we should know
        to what unitary it was hopefully going to converge to anyway (and that a solution exists).
        Example: the subsequent ansatz is replaced by the expected intermediate Unitary.

        c2:
             ┌──────────┐┌─────────────┐
        q_0: ┤0         ├┤ U(x0,x1,x2) ├──■──
             │  C1      │├─────────────┤┌─┴─┐
        q_1: ┤1         ├┤ U(x3,x4,x5) ├┤ X ├
             └──────────┘└─────────────┘└───┘

        c3:
             ┌──────────┐ ┌─────────────┐
        q_0: ┤0         ├─┤ U(x6,x7,x8) ├───■──
             │  CAN~C2  │┌┴─────────────┴┐┌─┴─┐
        q_1: ┤1         ├┤ U(x9,x10,x11) ├┤ X ├
             └──────────┘└───────────────┘└───┘

        NOTE We don't know exactly what unitary is created by previous iterations, only the invariants.
        Thus, this class relies on using CAN() as a substitute to evaluate the ansatz. Hopefully,
        this discprenacy is then corrected away for in the final exterior 1Q gates.

        NOTE 2, since the ansatz is created from scratch each iteration we need some other place
        to store the found parameters before they are discarded. So build temp_result along the way.
        """
        basis_gate_sequence = lp_result.basis_gate_sequence
        prefix_can_placeholders = [
            mono_coordinates_to_CAN(*c_args) for c_args in lp_result.mono_points[2:]
        ]

        instructions_list = []
        global_phase = 0

        # [2:] because skip trivial cases (identity, first basis gate)
        for idx, (placeholder_target, basis_gate) in enumerate(
            zip(prefix_can_placeholders, basis_gate_sequence[1:])
        ):
            last_layer_bool = idx == len(basis_gate_sequence) - 2
            instructions, phase = self._optimize_single_qubit_gates(
                prefix_op=(
                    prefix_can_placeholders[idx - 1]
                    if idx > 0
                    else basis_gate_sequence[0]
                ),
                basis_gate=basis_gate,
                target=placeholder_target if not last_layer_bool else target_op,
                seed=np.random.randint(-1e16, 1e16),
                debugging=debugging,
                lp_result=lp_result,
            )
            instructions_list.append(instructions)
            global_phase += phase

        return instructions_list, global_phase, lp_result

    def run(
        self,
        target_op: np.ndarray,
        lp_result: "MonodromyLPDecomposerResult",
        use_dag=False,
        debugging=False,
    ) -> QuantumCircuit:
        segment_solutions, global_phase, lp_result = self._segments_solve(
            target_op, lp_result, debugging
        )
        if debugging and lp_result:
            lp_result._segment_solutions = segment_solutions
            lp_result._global_phase = global_phase

        decomposed_qc = QuantumCircuit(2, global_phase=global_phase)
        qreg = decomposed_qc.qregs[0]

        decomposed_dag = circuit_to_dag(decomposed_qc)

        # piece together into a QuantumCircuit
        basis_gate_sequence = lp_result.basis_gate_sequence
        decomposed_dag.apply_operation_back(basis_gate_sequence[0], qreg[0:2])

        ## XXX
        # sometimes this messes up but its an off by -1.0* error
        # I think it has to do with whether or not the KAK was (A,B,C) vs. (A,B,-C)
        # and then one of the wires needs its angles rotated
        # requires more investigation.
        # XXX

        for basis_gate, segment in zip(basis_gate_sequence[1:], segment_solutions):
            # before KAK gates
            decomposed_dag.apply_operation_front(segment[0], qreg[0:1])
            decomposed_dag.apply_operation_front(segment[1], qreg[1:2])

            if not (self._use_simple or self._use_u3):
                # extended RVs come first in the circuit
                decomposed_dag.apply_operation_back(segment[6], qreg[1:2])
                decomposed_dag.apply_operation_back(segment[6], qreg[0:1])

            # first two gates are RVs
            decomposed_dag.apply_operation_back(segment[5], qreg[0:1])
            decomposed_dag.apply_operation_back(segment[4], qreg[1:2])
            decomposed_dag.apply_operation_back(basis_gate, qreg[0:2])

            # after KAK gates
            decomposed_dag.apply_operation_back(segment[2], qreg[0:1])
            decomposed_dag.apply_operation_back(segment[3], qreg[1:2])

        if use_dag:
            lp_result._set_final_qc(decomposed_dag)

        decomposed_qc = dag_to_circuit(decomposed_dag)
        lp_result._set_final_qc(decomposed_qc)

        output_qc = decomposed_dag if use_dag else decomposed_qc
        if debugging:
            return output_qc, lp_result
        return output_qc


# # NOTE not a bad option
# j_gn = GaussNewton(
#     residual_fun=_objective_function,
#     maxiter=256,  # 128
#     implicit_diff=False,
#     jit=True,
#     tol=CONV_TOL,
# )


# @jit
# def scalar_obj_fun(x, *args):
#     diff = _objective_function(x, *args)
#     return jnp.linalg.norm(diff)


# XXX doesn't compile
# @jit
# def scipy_lsq(x0, *args):
#     return least_squares(
#         vector_obj_fun,
#         x0,
#         args=args,
#         jac=jacfwd(vector_obj_fun),
#         method="dogbox",
#         xtol=4e-16,
#         max_nfev=64,
#     )
# _ret = scipy_lsq(x0, j_prefix_op, j_basis_gate, j_target_inv)
# (success, params, num_fun_eval, fun_val) = (
#     _ret.success,
#     _ret.x,
#     _ret.nfev,
#     _ret.cost,
# )


# NOTE this method seems slow
# from jaxopt import LevenbergMarquardt
# from jax import jacrev
# j_opt = LevenbergMarquardt(
#     residual_fun=vector_obj_fun,
#     maxiter=48,
#     tol=4e-16,
#     jit=True,
#     jac_fun=jacrev(vector_obj_fun),
# )

# @jit
# def vector_obj_fun(x, *args):
#     # NOTE need abs() because lsq takes **2
#     # diff = _objective_function(x, *args)
#     # return jnp.concatenate([diff, diff*2]) # seems unnecessary now
#     return _objective_function(x, *args)


# NOTE dogbox is faster but gives worse solutions
# XXX not working?, still calls implicit_diff.py
# j_lsq = ScipyLeastSquares(
#     method="lm",  # {‘trf’, ‘dogbox’, ‘lm’}
#     fun=_objective_function,
#     options={
#         "xtol": ATOL,
#         "max_nfev": 128,
#     },
#     use_jacrev=True,
# )
# NOTE retrieve result data from j_lsq
# (success, params, num_fun_eval, fun_val) = (
#     _ret.state.success,
#     _ret.params,
#     _ret.state.num_fun_eval,
#     _ret.state.cost_val,
# )
# )
