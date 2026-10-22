//! Qiskit C API calls: writing native gates, reading two-qubit unitaries from
//! a DAG, and substituting their replacements. DAG node ids never leave this
//! module.

use indexmap::map::Entry;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use qiskit_numerics::IndexMap;
use qiskit_pyo3_ffi as qk;
use qk::{QkDag, QkExitCode, QkGate, QkOperationKind};

use gulps_core::{C64, CompiledTarget, Gate, Mat4, Op, Unitary, map_batch};

use crate::decomposer::compile_error;
use crate::matrix;
use crate::transpiler::Edge;

/// How the C API writes one native gate.
#[derive(Clone)]
pub struct NativeGate(Writer);

#[derive(Clone)]
enum Writer {
    /// `qk_dag_apply_gate` with these parameters: a two-qubit gate taking
    /// exactly this many.
    Standard(QkGate, Vec<f64>),
    /// `qk_dag_apply_unitary` with the gate's own matrix, row-major.
    Unitary(Box<[C64; 16]>),
}

impl NativeGate {
    /// A gate written as its matrix.
    pub fn unitary(matrix: &Mat4) -> Self {
        Self(Writer::Unitary(Box::new(std::array::from_fn(|i| {
            matrix[(i / 4, i % 4)]
        }))))
    }

    /// Qiskit's synthesis convention (as `TwoQubitBasisDecomposer(gate: Gate)`):
    /// a standard gate under its own name is written by enum and a
    /// `UnitaryGate` by matrix. Any other gate is rejected: the C API writes
    /// only those two. Returns the gate's class and frame with it.
    pub fn from_python(gate: &Bound<'_, PyAny>) -> PyResult<(Gate, Self)> {
        let py = gate.py();
        if !gate.is_instance(&py.import("qiskit.circuit")?.getattr("Gate")?)? {
            return Err(PyTypeError::new_err(
                "a gate must be a Qiskit Gate or LocalEquivalenceClass; wrap a matrix in UnitaryGate",
            ));
        }
        let name = gate.getattr("name")?.extract::<String>()?;
        if gate.call_method0("is_parameterized")?.extract::<bool>()? {
            return Err(PyNotImplementedError::new_err(format!(
                "{name} has unbound parameters; see https://github.com/evmckinney9/gulps/issues/2"
            )));
        }
        let standard = gate.getattr("_standard_gate")?;
        let standard_gate = if standard
            .getattr("name")
            .and_then(|n| n.extract::<String>())
            .is_ok_and(|n| n == name)
        {
            let params: Vec<f64> = gate.getattr("params")?.extract()?;
            two_qubit_gate(standard.call_method0("__int__")?.extract()?, params.len())
                .map(|kind| Self(Writer::Standard(kind, params)))
        } else {
            None
        };
        let unitary_gate = py
            .import("qiskit.circuit.library")?
            .getattr("UnitaryGate")?;
        if standard_gate.is_none() && !(name == "unitary" && gate.get_type().is(&unitary_gate)) {
            return Err(PyNotImplementedError::new_err(format!(
                "{name} is neither a Qiskit standard gate under its own name nor a UnitaryGate; \
                 gulps emits only those through Qiskit's C API. Wrap the matrix in UnitaryGate; \
                 see https://github.com/evmckinney9/gulps/issues/18"
            )));
        }
        let matrix = matrix(gate)?;
        let native = standard_gate.unwrap_or_else(|| Self::unitary(&matrix));
        let gate =
            Gate::new(&Unitary::Matrix(matrix)).map_err(|err| compile_error(py, err, None))?;
        Ok((gate, native))
    }
}

/// The `QkGate` for a `StandardGate.__int__()` value that names a two-qubit
/// gate taking `num_params` parameters.
fn two_qubit_gate(kind: i32, num_params: usize) -> Option<QkGate> {
    let kind = u8::try_from(kind)
        .ok()
        .filter(|&k| k <= QkGate::RC3X as u8)?;
    // SAFETY: `QkGate` is `#[repr(u8)]` with contiguous discriminants `0..=RC3X`.
    let gate: QkGate = unsafe { std::mem::transmute(kind) };
    // SAFETY: pure predicates on a valid gate value.
    let fits = unsafe {
        qk::qk_gate_num_qubits(gate) == 2 && qk::qk_gate_num_params(gate) as usize == num_params
    };
    fits.then_some(gate)
}

/// One distinct compilation job: an edge and a matrix, and every node, on any
/// qubits, that it replaces.
struct Job<'a> {
    edge: &'a Edge,
    /// The nodes' matrix, in the edge's qubit order.
    unitary: Unitary,
    /// Whether the nodes' operands are in the opposite order to the edge's.
    flip: bool,
    nodes: Vec<u32>,
}

/// Replace every two-qubit unitary node of `dag` in place. `edge` gives each
/// operand pair's edge, and whether the edge's qubit order is the opposite;
/// nodes with the same edge, order, and a bit-identical matrix form one job.
/// Every replacement is built before the DAG changes.
///
/// # Safety
///
/// Compilation runs detached from Python, holding a raw handle and node ids
/// of `dag`. As for Qiskit's own passes, no other thread may use `dag` until
/// the call returns.
pub unsafe fn replace<'a>(
    dag: &Bound<'_, PyAny>,
    edge: impl Fn((u32, u32)) -> PyResult<(&'a Edge, bool)>,
) -> PyResult<()> {
    let py = dag.py();
    let raw = borrow(dag)?;
    // SAFETY: `raw` is borrowed from `dag`, which the caller holds and, per
    // this function's contract, no other thread uses during the call.
    let jobs = unsafe { read(raw, edge) }?;
    // `build` calls the C API only on the replacement it creates. At the pinned
    // revision those calls attach to Python only for control-flow and `store`
    // instructions, which are never written, so they can run detached.
    // About 4 to 13 µs per job: compiling, then building.
    let replacements = py
        .detach(|| {
            map_batch(&jobs, 6_000, |job| {
                let target = job.edge.binding.compile(&job.unitary)?;
                Ok(Replacement::build(&job.edge.natives, &target, job.flip))
            })
        })
        .map_err(|(i, err)| compile_error(py, err, jobs[i].unitary.class().ok()))?;
    for (replacement, job) in replacements.iter().zip(&jobs) {
        // SAFETY: as for `read`; `job.nodes` came from `read` of this DAG.
        unsafe { replacement.substitute(raw, &job.nodes) };
    }
    Ok(())
}

/// One compiled target as a Python `QuantumCircuit`, or a `DAGCircuit` when `as_dag`.
pub fn circuit(
    py: Python<'_>,
    natives: &[NativeGate],
    compiled: &CompiledTarget,
    as_dag: bool,
) -> PyResult<Py<PyAny>> {
    Replacement::build(natives, compiled, false).into_python(py, as_dag)
}

fn borrow(dag: &Bound<'_, PyAny>) -> PyResult<*mut QkDag> {
    // SAFETY: `dag` is a live Python object for the duration of the call.
    let raw = unsafe { qk::qk_dag_borrow_from_python(dag.as_ptr()) };
    if raw.is_null() {
        // Qiskit sets the Python error, e.g. a TypeError for a non-DAG.
        return Err(PyErr::take(dag.py()).unwrap_or_else(|| {
            PyRuntimeError::new_err("qk_dag_borrow_from_python returned null")
        }));
    }
    Ok(raw)
}

/// The two operand qubits of `node`, or `None` if it does not act on two.
///
/// # Safety
///
/// `dag` must be a live DAG holding `node`.
unsafe fn operand_pair(dag: *const QkDag, node: u32) -> Option<(u32, u32)> {
    // SAFETY: per the contract; two qubits are read only from a two-qubit node.
    unsafe {
        if qk::qk_dag_op_node_num_qubits(dag, node) != 2 {
            return None;
        }
        let qubits = qk::qk_dag_op_node_qubits(dag, node);
        Some((*qubits, *qubits.add(1)))
    }
}

/// The jobs of a DAG, in one topological traversal.
///
/// # Safety
///
/// `dag` must be a live DAG that no one else uses during the call.
unsafe fn read<'a>(
    dag: *mut QkDag,
    edge: impl Fn((u32, u32)) -> PyResult<(&'a Edge, bool)>,
) -> PyResult<Vec<Job<'a>>> {
    // SAFETY: per the contract; `order` has exactly `qk_dag_num_op_nodes` slots.
    let mut order = vec![0; unsafe { qk::qk_dag_num_op_nodes(dag) }];
    // SAFETY: as above.
    unsafe { qk::qk_dag_topological_op_nodes(dag, order.as_mut_ptr()) };
    let mut jobs = IndexMap::<(usize, bool, [u64; 32]), Job<'a>>::default();
    for node in order {
        // SAFETY: `node` came from `qk_dag_topological_op_nodes` of this DAG.
        let kind = unsafe { qk::qk_dag_op_node_kind(dag, node) };
        match kind {
            // The C API reads control-flow blocks only from circuits, and cannot
            // replace them, so their bodies are out of reach from a DAG.
            QkOperationKind::ControlFlow => {
                return Err(PyNotImplementedError::new_err(
                    "circuits with control flow are not supported; see \
                     https://github.com/evmckinney9/gulps/issues/23",
                ));
            }
            QkOperationKind::Unitary
            | QkOperationKind::Gate
            | QkOperationKind::Unknown
            | QkOperationKind::PauliProductRotation => {}
            _ => continue,
        }
        // SAFETY: as above.
        let Some(pair) = (unsafe { operand_pair(dag, node) }) else {
            continue;
        };
        if kind != QkOperationKind::Unitary {
            return Err(PyNotImplementedError::new_err(format!(
                "DAG node {node} is a two-qubit gate without a matrix: GULPS compiles the \
                 unitaries ConsolidateBlocks makes (the pass requires it), and a gate with \
                 unbound parameters has none; see https://github.com/evmckinney9/gulps/issues/2"
            )));
        }
        let (edge, flip) = edge(pair)?;
        let mut entries = [C64::new(0.0, 0.0); 16];
        // SAFETY: a two-qubit unitary node of this DAG; the buffer has 16 entries.
        unsafe { qk::qk_dag_op_node_unitary(dag, node, entries.as_mut_ptr()) };
        let bits = std::array::from_fn(|i| {
            let z = entries[i / 2];
            (if i % 2 == 0 { z.re } else { z.im }).to_bits()
        });
        match jobs.entry((std::ptr::from_ref(edge).addr(), flip, bits)) {
            Entry::Occupied(entry) => entry.into_mut().nodes.push(node),
            Entry::Vacant(entry) => {
                // Swapping the qubits exchanges basis states 1 and 2.
                let swap = |i: usize| if flip { [0, 2, 1, 3][i] } else { i };
                let matrix = Mat4::from_fn(|r, c| entries[4 * swap(r) + swap(c)]);
                entry.insert(Job {
                    edge,
                    unitary: Unitary::Matrix(matrix),
                    flip,
                    nodes: vec![node],
                });
            }
        }
    }
    Ok(jobs.into_values().collect())
}

/// An owned two-qubit replacement DAG.
struct Replacement(*mut QkDag);

// SAFETY: the handle is the DAG's only owner, and the DAG is Qiskit's
// `DAGCircuit`, a `Send` pyclass that holds no Python references here.
unsafe impl Send for Replacement {}

impl Drop for Replacement {
    fn drop(&mut self) {
        // SAFETY: this handle owns the DAG and is its only owner.
        unsafe { qk::qk_dag_free(self.0) };
    }
}

impl Replacement {
    /// A two-qubit DAG of the compiled target's operations, each native gate
    /// written as `natives[slot]` says, with qubits 0 and 1 exchanged if `flip`.
    /// The C API allocations here fail only when memory is exhausted, which
    /// panics.
    fn build(natives: &[NativeGate], target: &CompiledTarget, flip: bool) -> Self {
        // SAFETY: `qk_dag_new` has no preconditions.
        let raw = unsafe { qk::qk_dag_new() };
        assert!(!raw.is_null(), "qk_dag_new returned null");
        let dag = Self(raw);
        // SAFETY: the name is a NUL-terminated literal.
        let qreg = unsafe { qk::qk_quantum_register_new(2, c"q".as_ptr()) };
        assert!(!qreg.is_null(), "qk_quantum_register_new returned null");
        // SAFETY: the DAG copies the live register before it is freed.
        unsafe {
            qk::qk_dag_add_quantum_register(dag.0, qreg);
            qk::qk_quantum_register_free(qreg);
        }
        let qubits = if flip { [1, 0] } else { [0, 1] };
        for op in target.ops() {
            match *op {
                Op::Local { qubit, matrix } => {
                    let entries: [C64; 4] = std::array::from_fn(|i| matrix[(i / 2, i % 2)]);
                    let qubit = qubits[qubit as usize];
                    // SAFETY: the DAG is owned; a 4-entry matrix for one qubit.
                    unsafe {
                        qk::qk_dag_apply_unitary(dag.0, entries.as_ptr(), &qubit, 1, false);
                    }
                }
                Op::Gate(slot) => dag.native(&natives[slot], &qubits),
            }
        }
        // SAFETY: no preconditions; the parameter is freed after use.
        let param = unsafe { qk::qk_param_from_double(target.phase()) };
        assert!(!param.is_null(), "qk_param_from_double returned null");
        // SAFETY: the DAG and the parameter are live; the DAG copies the value.
        let code = unsafe { qk::qk_dag_set_global_phase(dag.0, param) };
        // SAFETY: allocated above and not used after this.
        unsafe { qk::qk_param_free(param) };
        assert!(
            code == QkExitCode::Success,
            "a finite global phase is always assignable"
        );
        dag
    }

    fn native(&self, gate: &NativeGate, qubits: &[u32; 2]) {
        match &gate.0 {
            // SAFETY: the DAG is owned. `Writer::Standard` is built only in
            // `from_python`, after `two_qubit_gate` checked that `kind` acts on two
            // qubits and takes `params.len()` parameters, so the C API reads
            // exactly the two qubits and the parameters passed here.
            Writer::Standard(kind, params) => unsafe {
                qk::qk_dag_apply_gate(self.0, *kind, qubits.as_ptr(), params.as_ptr(), false);
            },
            // SAFETY: the DAG is owned; a 16-entry matrix for two qubits.
            Writer::Unitary(matrix) => unsafe {
                qk::qk_dag_apply_unitary(self.0, matrix.as_ptr(), qubits.as_ptr(), 2, false);
            },
        }
    }

    /// Substitute this replacement at each of `nodes`.
    ///
    /// # Safety
    ///
    /// `dag` must be live and unshared, and each node an unreplaced two-qubit
    /// unitary of it.
    unsafe fn substitute(&self, dag: *mut QkDag, nodes: &[u32]) {
        for &node in nodes {
            // SAFETY: per the contract; the replacement owns two qubits and no
            // classical bits or variables.
            unsafe { qk::qk_dag_substitute_node_with_dag(dag, node, self.0) };
        }
    }

    fn into_python(self, py: Python<'_>, as_dag: bool) -> PyResult<Py<PyAny>> {
        if as_dag {
            // `qk_dag_to_python` takes ownership, so the guard must not free the DAG.
            let raw = std::mem::ManuallyDrop::new(self).0;
            // SAFETY: `py` proves attachment; `raw` is an owned, live DAG that nothing else frees.
            let dag = unsafe { Bound::from_owned_ptr_or_err(py, qk::qk_dag_to_python(raw)) }?;
            return Ok(dag.unbind());
        }
        // SAFETY: the guard owns a live DAG; conversion returns an owned circuit.
        let raw = unsafe { qk::qk_dag_to_circuit(self.0) };
        assert!(!raw.is_null(), "qk_dag_to_circuit returned null");
        // SAFETY: transfer consumes the owned circuit and returns a new Python reference.
        let circuit =
            unsafe { Bound::from_owned_ptr_or_err(py, qk::qk_circuit_to_python_full(raw)) }?;
        Ok(circuit.unbind())
    }
}
