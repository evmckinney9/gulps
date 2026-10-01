//! `GulpsDecomposer`: one set of gates and costs as one `Edge`, and the
//! decomposition errors.

use std::sync::Arc;

use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use gulps_core::{CompileError, Gate, compile_all};

use crate::invariants::LocalEquivalenceClass;
use crate::qiskit::{self, NativeGate};
use crate::transpiler::Edge;
use crate::{checked_mat4, mat4_to_numpy};

pyo3::create_exception!(
    gulps.decomposition,
    DecompositionError,
    pyo3::exceptions::PyRuntimeError,
    "A target could not be selected or synthesized, or a gate's class could not be computed. ``target`` is the target's ``LocalEquivalenceClass``, or None when no single target applies."
);
pyo3::create_exception!(
    gulps.decomposition,
    SearchDepthError,
    DecompositionError,
    "No sentence within ``max_depth`` reaches the target."
);

/// The Python exception for a compile error, with the failing target's class,
/// when known, as its `target` attribute.
pub fn compile_error(
    py: Python<'_>,
    err: CompileError,
    target: Option<LocalEquivalenceClass>,
) -> PyErr {
    let exception = match err {
        CompileError::DepthLimit(_) => SearchDepthError::new_err(err.to_string()),
        CompileError::Synthesis(_) => DecompositionError::new_err(err.to_string()),
    };
    if let Some(target) = target {
        // Setting an attribute on a fresh exception instance cannot fail.
        let _ = exception.value(py).setattr("target", target);
    }
    exception
}

/// Synthesizes a two-qubit unitary exactly, as the sequence of the given gates
/// with the least total cost.
///
/// Calling the decomposer on a two-qubit ``unitary`` (a Qiskit gate, an
/// ``Operator``, or a 4-by-4 array) returns a ``QuantumCircuit`` of the native
/// gates between single-qubit unitaries, or a ``DAGCircuit`` with
/// ``use_dag=True``.
///
/// Of gates with exactly the same computed class, only the cheapest is kept,
/// the first on ties, with a warning. A class is emitted as its canonical gate
/// in a ``UnitaryGate``.
///
/// Args:
///     gates (Sequence[qiskit.circuit.Gate | ``LocalEquivalenceClass``]): Each
///         gate is a Qiskit standard gate under its own name or a
///         ``UnitaryGate``, with no unbound parameters.
///     costs (Sequence[float]): Finite and nonnegative, one per gate.
///     ``local_layer_cost`` (float): Cost of each layer of single-qubit gates
///         before, between, and after the two-qubit gates.
///     ``max_depth`` (int): Maximum number of two-qubit gates in a sequence.
///
/// Example:
///     >>> decomposer = GulpsDecomposer([CXGate(), iSwapGate()], [1.0, 1.0])
///     >>> circuit = decomposer(random_unitary(4))
#[allow(clippy::doc_markdown)] // the example is Python, rendered by Sphinx
#[pyclass(frozen, module = "gulps.decomposition")]
pub struct GulpsDecomposer {
    pub edge: Arc<Edge>,
    /// The kept Qiskit gate objects, one per slot.
    pub gates: Py<PyTuple>,
}

#[pymethods]
impl GulpsDecomposer {
    #[new]
    #[pyo3(signature = (gates, costs, local_layer_cost=0.0, max_depth=crate::MAX_DEPTH))]
    fn new(
        py: Python<'_>,
        gates: Vec<Bound<'_, PyAny>>,
        costs: Vec<f64>,
        local_layer_cost: f64,
        max_depth: usize,
    ) -> PyResult<Self> {
        if gates.len() != costs.len() {
            return Err(PyValueError::new_err(
                "gates and costs must have the same length",
            ));
        }
        let unitary_gate = py
            .import("qiskit.circuit.library")?
            .getattr("UnitaryGate")?;
        let mut objects = Vec::with_capacity(gates.len());
        let mut natives = Vec::with_capacity(gates.len());
        for gate in &gates {
            if let Ok(class) = gate.cast::<LocalEquivalenceClass>() {
                let matrix = class.get().mat4();
                objects.push(unitary_gate.call1((mat4_to_numpy(py, &matrix),))?);
                let core = Gate::new(&matrix).map_err(|err| compile_error(py, err, None))?;
                natives.push((core, NativeGate::unitary(&matrix)));
            } else {
                natives.push(NativeGate::from_python(gate)?);
                objects.push(gate.call_method0("copy")?);
            }
        }
        let natives: Vec<_> = natives.iter().zip(costs).collect();
        let (edge, order) =
            Edge::new(&natives, local_layer_cost, max_depth, []).map_err(PyValueError::new_err)?;
        if order.len() < gates.len() {
            let warning =
                "Identical computed gate classes were removed; cheapest wins, first on ties.";
            let warn = py.import("warnings")?.getattr("warn")?;
            warn.call1((warning, py.get_type::<PyUserWarning>(), 2))?;
        }
        Ok(Self {
            edge: Arc::new(edge),
            gates: PyTuple::new(py, order.iter().map(|&i| &objects[i]))?.unbind(),
        })
    }

    /// int: Maximum number of two-qubit gates in a sequence.
    #[getter]
    fn max_depth(&self) -> usize {
        self.edge.binding.decomposer().max_depth()
    }

    /// float: Cost of each layer of single-qubit gates.
    #[getter]
    fn local_layer_cost(&self) -> f64 {
        self.edge.binding.decomposer().local_layer_cost()
    }

    /// tuple[qiskit.circuit.Gate, ...]: The kept gates, as the decomposer's own
    /// copies, which ``select`` also returns. Do not mutate them.
    #[getter]
    fn gates<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.gates.bind(py).clone()
    }

    /// tuple[float, ...]: The cost of each gate in ``gates``.
    #[getter]
    fn costs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.edge
                .binding
                .decomposer()
                .slots()
                .iter()
                .map(|&(_, cost)| cost),
        )
    }

    /// The cost and gates of the least-cost sequence for each target, without
    /// synthesizing a circuit.
    ///
    /// Args:
    ///     targets (qiskit.circuit.Gate | ``qiskit.quantum_info.Operator`` | np.ndarray | ``LocalEquivalenceClass`` | list):
    ///         One target, or a list, tuple, or ``(N, 4, 4)`` array of targets.
    ///
    /// Returns:
    ///     tuple[float, tuple[qiskit.circuit.Gate, ...]] | list[...]: The
    ///     sequence's cost and gates, or a list with one such pair per target.
    fn select(&self, py: Python<'_>, targets: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let batch = targets.is_instance_of::<PyList>()
            || targets.is_instance_of::<PyTuple>()
            || (targets.getattr("ndim").and_then(|ndim| ndim.eq(3))).unwrap_or(false);
        let inputs = if batch {
            targets.extract()?
        } else {
            vec![targets.clone()]
        };
        let classes = LocalEquivalenceClass::from_unitaries(py, inputs)?;
        let monos: Vec<_> = classes.iter().map(|c| c.mono).collect();
        let hits = py
            .detach(|| self.edge.binding.decomposer().select(&monos))
            .map_err(|err| compile_error(py, err, (!batch).then(|| classes[0])))?;
        let gates = self.gates.bind(py);
        let mut selected = hits
            .into_iter()
            .map(|hit| {
                let slots = hit
                    .gates
                    .iter()
                    .map(|&slot| gates.get_borrowed_item(slot).expect("each slot has a gate"));
                (hit.cost, PyTuple::new(py, slots)?).into_pyobject(py)
            })
            .collect::<PyResult<Vec<_>>>()?;
        if batch {
            Ok(PyList::new(py, selected)?.into_any().unbind())
        } else {
            let single = selected.pop().expect("one target, one selection");
            Ok(single.into_any().unbind())
        }
    }

    // PyO3 does not publish slot-method docs; the class docstring documents
    // calling the decomposer.
    #[pyo3(signature = (unitary, use_dag=false))]
    fn __call__(
        &self,
        py: Python<'_>,
        unitary: &Bound<'_, PyAny>,
        use_dag: bool,
    ) -> PyResult<Py<PyAny>> {
        let matrix = checked_mat4(unitary)?;
        let compiled = py
            .detach(|| compile_all(&[(&self.edge.binding, &matrix)]))
            .map_err(|(_, err)| compile_error(py, err, LocalEquivalenceClass::of(&matrix)))?;
        qiskit::circuit(py, &self.edge.natives, &compiled[0], use_dag)
    }

    /// Pickle as the constructor arguments; the search restarts after unpickling.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let args = (
            this.gates.bind(py),
            this.costs(py)?,
            this.local_layer_cost(),
            this.max_depth(),
        );
        (slf.get_type(), args).into_pyobject(py)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    // Without a known target class, `target` reads as None.
    py.get_type::<DecompositionError>()
        .setattr("target", py.None())?;
    m.add("DecompositionError", py.get_type::<DecompositionError>())?;
    m.add("SearchDepthError", py.get_type::<SearchDepthError>())?;
    m.add_class::<GulpsDecomposer>()
}
