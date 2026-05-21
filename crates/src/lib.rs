//! PyO3 bindings for gulps_accelerate.
//!
//! Exposes Rust-accelerated gate primitives and solver as `gulps._accelerate`.

use nalgebra::{Complex, Matrix2, Matrix4};
use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

pub(crate) type C64 = Complex<f64>;
pub(crate) type Mat2 = Matrix2<C64>;
pub(crate) type Mat4 = Matrix4<C64>;

pub(crate) const C0: C64 = C64::new(0.0, 0.0);
pub(crate) const C1: C64 = C64::new(1.0, 0.0);
pub(crate) const CI: C64 = C64::new(0.0, 1.0);
pub(crate) const CM1: C64 = C64::new(-1.0, 0.0);

mod gate_primitives;
mod jacobian;
mod linalg;
mod lp;
mod pipeline;
mod recover;
mod solver;

fn numpy_to_mat4(u: &PyReadonlyArray2<Complex64>) -> PyResult<Mat4> {
    let slice = u.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be C-contiguous: {e}"))
    })?;
    Ok(Mat4::from_row_slice(slice))
}

fn mat2_to_numpy<'py>(py: Python<'py>, m: &Mat2) -> Bound<'py, numpy::PyArray2<Complex64>> {
    Array2::from_shape_fn((2, 2), |(r, c)| m[(r, c)]).into_pyarray(py)
}

fn mat2_list_to_numpy(py: Python<'_>, ms: &[Mat2]) -> Vec<Py<PyAny>> {
    ms.iter()
        .map(|m| mat2_to_numpy(py, m).into_any().unbind())
        .collect()
}

#[pyfunction]
fn weyl_coordinates<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<Py<PyAny>> {
    let mat = numpy_to_mat4(&u)?;
    let result = gate_primitives::weyl_coordinates(&mat);
    Ok(result.to_vec().into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn canonical_matrix<'py>(py: Python<'py>, c1: f64, c2: f64, c3: f64) -> PyResult<Py<PyAny>> {
    let mat = gate_primitives::canonical_matrix(c1, c2, c3);
    let arr = Array2::from_shape_fn((4, 4), |(r, c)| mat[(r, c)]);
    Ok(arr.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn weyl_from_monodromy<'py>(py: Python<'py>, m: PyReadonlyArray1<'py, f64>) -> PyResult<Py<PyAny>> {
    let s = m.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("m must be contiguous: {e}"))
    })?;
    let result = gate_primitives::weyl_from_monodromy(&[s[0], s[1], s[2]]);
    Ok(result.to_vec().into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn monodromy_from_weyl<'py>(py: Python<'py>, c1: f64, c2: f64, c3: f64) -> PyResult<Py<PyAny>> {
    let result = gate_primitives::monodromy_from_weyl(c1, c2, c3);
    Ok(result.to_vec().into_pyarray(py).into_any().unbind())
}

/// Check if a 4x4 matrix is unitary.
#[pyfunction]
#[pyo3(signature = (u, atol=1e-8, rtol=1e-5))]
fn is_unitary<'py>(u: PyReadonlyArray2<'py, Complex64>, atol: f64, rtol: f64) -> PyResult<bool> {
    let mat = numpy_to_mat4(&u)?;
    let prod = mat.adjoint() * mat;
    for i in 0..4 {
        for j in 0..4 {
            let id_val = if i == j { 1.0 } else { 0.0 };
            let diff = (prod[(i, j)] - C64::new(id_val, 0.0)).norm();
            if diff > atol + rtol * id_val {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Recover the local corrections between two 4×4 unitaries; returns `(k1, k2, k3, k4, global_phase)`.
#[pyfunction]
fn recover_local_equiv<'py>(
    py: Python<'py>,
    u_target: PyReadonlyArray2<'py, Complex64>,
    u_basis: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, f64)> {
    let target = numpy_to_mat4(&u_target)?;
    let basis = numpy_to_mat4(&u_basis)?;

    let result = py
        .detach(|| recover::recover_local_equivalence(&target, &basis))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok((
        mat2_to_numpy(py, &result.k1).into_any().unbind(),
        mat2_to_numpy(py, &result.k2).into_any().unbind(),
        mat2_to_numpy(py, &result.k3).into_any().unbind(),
        mat2_to_numpy(py, &result.k4).into_any().unbind(),
        result.global_phase,
    ))
}

/// Solve and stitch all segments; parallel above `min_batch_size`. Returns
/// `(u0s, u1s, T)`: the fused locals and the chain's phase-absorbed product
/// matrix. Caller runs `recover_local_equiv(target, T)` to wrap it to a
/// specific target.
#[pyfunction]
#[pyo3(signature = (initial_matrix, basis_matrices, target_matrices, makhlin_tol, weyl_tol, min_batch_size, target_weyl_coords, identity_warmstart=false))]
fn solve_and_stitch<'py>(
    py: Python<'py>,
    initial_matrix: PyReadonlyArray2<'py, Complex64>,
    basis_matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
    target_matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
    makhlin_tol: f64,
    weyl_tol: f64,
    min_batch_size: usize,
    target_weyl_coords: Vec<[f64; 3]>,
    identity_warmstart: bool,
) -> PyResult<(
    Vec<Py<PyAny>>, // u0s (fused with intermediate corrections)
    Vec<Py<PyAny>>, // u1s
    Py<PyAny>,      // T (4x4 chain product, intermediate phase absorbed)
)> {
    let init = numpy_to_mat4(&initial_matrix)?;
    let basis_mats: Vec<Mat4> = basis_matrices
        .iter()
        .map(numpy_to_mat4)
        .collect::<PyResult<_>>()?;
    let target_mats: Vec<Mat4> = target_matrices
        .iter()
        .map(numpy_to_mat4)
        .collect::<PyResult<_>>()?;
    let (u0s, u1s, t) = py
        .detach(|| {
            pipeline::solve_segments(
                &init,
                &basis_mats,
                &target_mats,
                &target_weyl_coords,
                makhlin_tol,
                weyl_tol,
                min_batch_size,
                identity_warmstart,
            )
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let t_np = Array2::from_shape_fn((4, 4), |(r, c)| t[(r, c)]);
    Ok((
        mat2_list_to_numpy(py, &u0s),
        mat2_list_to_numpy(py, &u1s),
        t_np.into_pyarray(py).into_any().unbind(),
    ))
}

/// Dual revised simplex with a warm-startable basis. A and c are fixed;
/// only b varies per solve().
#[pyclass]
struct DualSimplex {
    inner: lp::DualSimplexSolver,
}

#[pymethods]
impl DualSimplex {
    #[new]
    fn new(
        a: PyReadonlyArray2<'_, f64>,
        c: PyReadonlyArray1<'_, f64>,
        initial_basis: Vec<usize>,
        tol: f64,
    ) -> PyResult<Self> {
        let a_shape = a.shape();
        let (m, n) = (a_shape[0], a_shape[1]);
        let a_slice = a.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("A must be C-contiguous: {e}"))
        })?;
        // numpy is row-major, nalgebra column-major
        let a_mat = nalgebra::DMatrix::from_fn(m, n, |r, c| a_slice[r * n + c]);
        let c_slice = c.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("c must be contiguous: {e}"))
        })?;
        let c_vec = nalgebra::DVector::from_column_slice(c_slice);

        Ok(Self {
            inner: lp::DualSimplexSolver::new(a_mat, c_vec, initial_basis, tol),
        })
    }

    /// Solve for `b`. Returns `(x, feasible)`; basis is warm-started across calls.
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Option<Py<PyAny>>, bool)> {
        let b_slice = b.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("b must be contiguous: {e}"))
        })?;

        match self.inner.solve(b_slice) {
            Some(x) => {
                let arr = numpy::PyArray1::from_vec(py, x).into_any().unbind();
                Ok((Some(arr), true))
            }
            None => Ok((None, false)),
        }
    }
}

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(weyl_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(weyl_from_monodromy, m)?)?;
    m.add_function(wrap_pyfunction!(monodromy_from_weyl, m)?)?;
    m.add_function(wrap_pyfunction!(is_unitary, m)?)?;
    m.add_function(wrap_pyfunction!(recover_local_equiv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_and_stitch, m)?)?;
    m.add_class::<DualSimplex>()?;
    Ok(())
}
