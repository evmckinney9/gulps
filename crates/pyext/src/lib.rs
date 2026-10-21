//! Python bindings for gate classes, decomposers, and the transpiler pass,
//! over the Qiskit C API.

use gulps_core::{Mat4, Unitary};
use numpy::{AllowTypeChange, IntoPyArray, PyArrayLikeDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyImportError, PyValueError};
use pyo3::prelude::*;

mod analysis;
mod decomposer;
mod invariants;
mod qiskit;
mod transpiler;

/// The Qiskit major and minor version that `qiskit-pyo3-ffi` was generated
/// from. The C API may change between minor versions, so the loaded Qiskit
/// must match.
const QISKIT_VERSION: (u32, u32) = (2, 6);

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    qiskit_pyo3_ffi::qk_import(m.py())?;
    // SAFETY: `qk_import` succeeded.
    let version = unsafe { qiskit_pyo3_ffi::qk_api_version() };
    let loaded = (version >> 24, (version >> 16) & 0xff);
    if loaded != QISKIT_VERSION {
        return Err(PyImportError::new_err(format!(
            "gulps was built for Qiskit {}.{}, but Qiskit {}.{} is installed",
            QISKIT_VERSION.0, QISKIT_VERSION.1, loaded.0, loaded.1
        )));
    }
    decomposer::register(m)?;
    m.add_class::<transpiler::BlockSynthesizer>()?;
    m.add_class::<invariants::LocalEquivalenceClass>()?;
    m.add("MAX_DEPTH", MAX_DEPTH)?;
    analysis::register(m)
}

/// Default bound on the number of two-qubit gates a search considers. It stops
/// a search that would otherwise not end, and no search is expected to reach it.
const MAX_DEPTH: usize = 64;

/// A 4x4 matrix from an array, or from anything `Operator` accepts.
fn matrix(ob: &Bound<'_, PyAny>) -> PyResult<Mat4> {
    let array = ob
        .extract::<PyArrayLikeDyn<gulps_core::C64, AllowTypeChange>>()
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
    Ok(Mat4::from_fn(|r, c| array[[r, c]]))
}

/// A two-qubit unitary: the canonical gate of a `LocalEquivalenceClass`, or
/// the matrix that [`matrix`] reads.
fn unitary(ob: &Bound<'_, PyAny>) -> PyResult<Unitary> {
    if let Ok(class) = ob.cast::<invariants::LocalEquivalenceClass>() {
        return Ok(Unitary::Canonical(class.get().mono));
    }
    Ok(Unitary::Matrix(matrix(ob)?))
}

fn mat4_to_numpy<'py>(py: Python<'py>, m: &Mat4) -> Bound<'py, PyAny> {
    numpy::ndarray::Array2::from_shape_fn((4, 4), |(r, c)| m[(r, c)])
        .into_pyarray(py)
        .into_any()
}
