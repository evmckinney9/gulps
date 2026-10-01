//! Python bindings for gate classes, decomposers, and the transpiler pass,
//! over the Qiskit C API. The synthesis algorithm lives in gulps-core.

use gulps_core::Mat4;
use numpy::{AllowTypeChange, IntoPyArray, PyArrayLike2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod analysis;
mod decomposer;
mod invariants;
mod qiskit;
mod transpiler;

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    qiskit_pyo3_ffi::qk_import(m.py())?;
    decomposer::register(m)?;
    m.add_class::<transpiler::BlockSynthesizer>()?;
    m.add_class::<invariants::LocalEquivalenceClass>()?;
    m.add("MAX_DEPTH", MAX_DEPTH)?;
    analysis::register(m)
}

/// Default bound on the number of two-qubit gates a search considers. It stops
/// a search that would otherwise not end, and no search is expected to reach it.
const MAX_DEPTH: usize = 64;

/// Entrywise error accepted in U†U − I at the Python input boundary. Class
/// projection and factorization both decompose the closest unitary when needed.
const INPUT_ATOL: f64 = 1e-8;

/// Whether `U†U − I` is entrywise within [`INPUT_ATOL`]. Nonfinite entries fail
/// the comparison, so they are rejected.
fn is_unitary(matrix: &Mat4) -> bool {
    (matrix.adjoint() * matrix - Mat4::identity())
        .iter()
        .all(|value| value.norm() <= INPUT_ATOL)
}

/// A 4x4 matrix from an array, or from anything `Operator` accepts, checked
/// unitary within [`INPUT_ATOL`].
fn checked_mat4(ob: &Bound<'_, PyAny>) -> PyResult<Mat4> {
    let array = ob
        .extract::<PyArrayLike2<gulps_core::C64, AllowTypeChange>>()
        .or_else(|_| {
            ob.py()
                .import("qiskit.quantum_info")?
                .getattr("Operator")?
                .call1((ob,))?
                .getattr("data")?
                .extract()
        })?;
    if array.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!(
            "expected a (4, 4) matrix, got {:?}",
            array.shape()
        )));
    }
    let array = array.as_array();
    let matrix = Mat4::from_fn(|r, c| array[(r, c)]);
    if !is_unitary(&matrix) {
        return Err(PyValueError::new_err(format!(
            "the {} is not unitary within atol={INPUT_ATOL:e}",
            ob.get_type().name()?
        )));
    }
    Ok(matrix)
}

fn mat4_to_numpy<'py>(py: Python<'py>, m: &Mat4) -> Bound<'py, PyAny> {
    numpy::ndarray::Array2::from_shape_fn((4, 4), |(r, c)| m[(r, c)])
        .into_pyarray(py)
        .into_any()
}
