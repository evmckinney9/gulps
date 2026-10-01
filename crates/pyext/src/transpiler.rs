//! `BlockSynthesizer`, the compiler behind the transpiler pass: the physical
//! gates offered on each Target edge, bound to shared canonical decomposers.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use qiskit_numerics::IndexMap;

use gulps_core::{Binding, Decomposer, Gate, compile_all};

use crate::decomposer::{GulpsDecomposer, compile_error};
use crate::invariants::LocalEquivalenceClass;
use crate::qiskit::{self, NativeGate};

/// The gates offered on one edge: their binding to a decomposer, and how each
/// slot's gate is written.
pub struct Edge {
    pub binding: Binding,
    pub natives: Vec<NativeGate>,
}

impl Edge {
    /// Bind gates, each with its cost, as [`Binding::new`]. Returns the edge
    /// and, for each slot, the input position of the gate filling it.
    pub fn new<'a>(
        gates: &[(&(Gate, NativeGate), f64)],
        local_layer_cost: f64,
        max_depth: usize,
        existing: impl IntoIterator<Item = &'a Arc<Decomposer>>,
    ) -> Result<(Self, Vec<usize>), String> {
        let classes: Vec<_> = gates
            .iter()
            .map(|&((gate, _), cost)| (gate, cost))
            .collect();
        let (binding, order) = Binding::new(&classes, local_layer_cost, max_depth, existing)?;
        let natives = order.iter().map(|&g| gates[g].0.1.clone()).collect();
        Ok((Self { binding, natives }, order))
    }
}

/// The edge for each qubit pair, and for any pair not listed.
#[pyclass(frozen, skip_from_py_object, module = "gulps._accelerate")]
pub struct BlockSynthesizer {
    edges: IndexMap<(u32, u32), Arc<Edge>>,
    default: Option<Arc<Edge>>,
}

impl BlockSynthesizer {
    /// The edge for an ordered pair of physical qubits, and whether its qubit
    /// order is the opposite: a pair uses its own edge, else its reverse
    /// edge, else the default.
    fn get(&self, (q0, q1): (u32, u32)) -> PyResult<(&Edge, bool)> {
        let edge = (self.edges.get(&(q0, q1)).map(|e| (e, false)))
            .or_else(|| self.edges.get(&(q1, q0)).map(|e| (e, true)))
            .or_else(|| self.default.as_ref().map(|e| (e, false)));
        edge.map(|(e, flip)| (e.as_ref(), flip)).ok_or_else(|| {
            PyValueError::new_err(format!("the Target has no two-qubit gate on ({q0}, {q1})"))
        })
    }
}

/// A Target's two-qubit gates and their properties: those offered on every
/// pair, and those listed per edge.
struct TargetGates<'py> {
    natives: BTreeMap<String, (Gate, NativeGate)>,
    global: BTreeMap<String, Bound<'py, PyAny>>,
    explicit: BTreeMap<(u32, u32), BTreeMap<String, Bound<'py, PyAny>>>,
}

fn read_target<'py>(target: &Bound<'py, PyAny>) -> PyResult<TargetGates<'py>> {
    let mut gates = TargetGates {
        natives: BTreeMap::new(),
        global: BTreeMap::new(),
        explicit: BTreeMap::new(),
    };
    for name in target.getattr("operation_names")?.try_iter()? {
        let name: String = name?.extract()?;
        let gate = target.call_method1("operation_from_name", (&name,))?;
        if !gate.getattr("num_qubits")?.eq(2)? {
            continue;
        }
        let own_name: String = gate.getattr("name")?.extract()?;
        if own_name != name {
            return Err(PyNotImplementedError::new_err(format!(
                "Target instruction '{name}' is an alias of '{own_name}', which the Target \
                 does not support under that name; see \
                 https://github.com/evmckinney9/gulps/issues/18"
            )));
        }
        gates
            .natives
            .insert(name.clone(), NativeGate::from_python(&gate)?);
        for item in target.get_item(&name)?.call_method0("items")?.try_iter()? {
            let (qargs, props): (Option<(u32, u32)>, Bound<'py, PyAny>) = item?.extract()?;
            match qargs {
                Some(pair) => gates
                    .explicit
                    .entry(pair)
                    .or_default()
                    .insert(name.clone(), props),
                None => gates.global.insert(name.clone(), props),
            };
        }
    }
    Ok(gates)
}

/// Each gate's cost on one edge: the requested Target property, or 1.0 for
/// every gate (so the search minimizes depth) when no gate has it. One binding
/// cannot mix cost models, so only some gates having it is an error.
fn edge_costs(
    props: &[&Bound<'_, PyAny>],
    field: &str,
    edge: Option<(u32, u32)>,
) -> PyResult<Vec<f64>> {
    let values = props
        .iter()
        .map(|p| {
            if p.is_none() {
                Ok(None)
            } else {
                p.getattr(field)?.extract::<Option<f64>>()
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    if values.iter().all(Option::is_none) {
        return Ok(vec![1.0; values.len()]);
    }
    (values.into_iter().collect::<Option<_>>()).ok_or_else(|| {
        edge_error(
            edge,
            &format!("some gates have a {field} and others do not"),
        )
    })
}

fn edge_error(edge: Option<(u32, u32)>, message: &str) -> PyErr {
    let on = edge.map_or("every pair".into(), |(a, b)| format!("({a}, {b})"));
    PyValueError::new_err(format!("Target edge {on}: {message}"))
}

#[pymethods]
impl BlockSynthesizer {
    /// One edge per Target edge (its gates and the gates offered on every
    /// pair), also used in reverse where the reverse pair has none, and a
    /// default for the gates offered on every pair. Local layers cost nothing: see
    /// <https://github.com/evmckinney9/gulps/issues/21>.
    #[staticmethod]
    fn from_target(target: &Bound<'_, PyAny>, cost: &str, max_depth: usize) -> PyResult<Self> {
        if cost != "duration" && cost != "error" {
            return Err(PyValueError::new_err(format!(
                "cost must be 'duration' or 'error', got {cost:?}"
            )));
        }
        let local_layer_cost = 0.0;
        let TargetGates {
            natives,
            global,
            explicit,
        } = read_target(target)?;
        let gates_on = |props: &BTreeMap<String, Bound<'_, PyAny>>, edge| -> PyResult<_> {
            let costs = edge_costs(&props.values().collect::<Vec<_>>(), cost, edge)?;
            Ok((props.keys().map(|name| &natives[name]))
                .zip(costs)
                .collect::<Vec<_>>())
        };
        // Pairs with the same gates and costs share one edge.
        let mut groups = IndexMap::<Vec<(String, u64)>, (Vec<_>, Vec<_>)>::default();
        for (&pair, props) in &explicit {
            let mut props = props.clone();
            for (name, p) in &global {
                props.entry(name.clone()).or_insert_with(|| p.clone());
            }
            let gates = gates_on(&props, Some(pair))?;
            let key = (props.keys().cloned())
                .zip(gates.iter().map(|&(_, cost)| cost.to_bits()))
                .collect();
            groups
                .entry(key)
                .or_insert((gates, Vec::new()))
                .1
                .push(pair);
        }
        // Each new edge may reuse the decomposer of one built before it.
        let mut built = Vec::<Arc<Edge>>::new();
        let mut bind = |gates: &[(&(Gate, NativeGate), f64)], on| -> PyResult<Arc<Edge>> {
            let existing = built.iter().map(|e| e.binding.decomposer());
            let (edge, _) = Edge::new(gates, local_layer_cost, max_depth, existing)
                .map_err(|message| edge_error(on, &message))?;
            let edge = Arc::new(edge);
            built.push(Arc::clone(&edge));
            Ok(edge)
        };
        let mut edges = IndexMap::default();
        for (gates, pairs) in groups.into_values() {
            let edge = bind(&gates, Some(pairs[0]))?;
            for pair in pairs {
                edges.insert(pair, Arc::clone(&edge));
            }
        }
        let default = if global.is_empty() {
            None
        } else {
            Some(bind(&gates_on(&global, None)?, None)?)
        };
        Ok(Self { edges, default })
    }

    /// Every pair uses the decomposer's gates, sharing its search.
    #[staticmethod]
    fn from_decomposer(decomposer: &GulpsDecomposer) -> Self {
        Self {
            edges: IndexMap::default(),
            default: Some(Arc::clone(&decomposer.edge)),
        }
    }

    /// Replace every two-qubit unitary of `dag` in place, and return it. No
    /// other thread may use `dag` until this returns.
    fn run(&self, py: Python<'_>, dag: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if dag.getattr("num_vars")?.extract::<usize>()? != 0 {
            return Err(PyNotImplementedError::new_err(
                "DAGs with classical variables are not supported; see \
                 https://github.com/evmckinney9/gulps/issues/23",
            ));
        }
        qiskit::replace(
            dag,
            |pair| self.get(pair),
            |jobs| {
                let batch: Vec<_> = jobs
                    .iter()
                    .map(|job| (&job.edge.binding, &job.matrix))
                    .collect();
                py.detach(|| compile_all(&batch)).map_err(|(i, err)| {
                    compile_error(py, err, LocalEquivalenceClass::of(&jobs[i].matrix))
                })
            },
        )?;
        Ok(dag.clone().unbind())
    }
}
