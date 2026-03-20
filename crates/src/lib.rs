//! PyO3 bindings for gulps_accelerate.
//!
//! Exposes Rust-accelerated gate primitives and solver as `gulps._accelerate`.

use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

mod gate_primitives;
mod lp;
mod recover;
mod solver;

use gate_primitives::{Mat2, Mat4};

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn numpy_to_mat4(u: &PyReadonlyArray2<Complex64>) -> Mat4 {
    let slice = u.as_slice().expect("Array must be C-contiguous");
    Mat4::from_row_slice(slice)
}

fn mat2_to_numpy<'py>(
    py: Python<'py>,
    m: &Mat2,
) -> Bound<'py, numpy::PyArray2<Complex64>> {
    Array2::from_shape_fn((2, 2), |(r, c)| m[(r, c)]).into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Gate primitives
// ---------------------------------------------------------------------------

#[pyfunction]
fn weyl_coordinates<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<Py<PyAny>> {
    let mat = numpy_to_mat4(&u);
    let result = gate_primitives::weyl_coordinates(&mat);
    Ok(result.to_vec().into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn canonical_matrix<'py>(
    py: Python<'py>,
    c1: f64,
    c2: f64,
    c3: f64,
) -> PyResult<Py<PyAny>> {
    let mat = gate_primitives::canonical_matrix(c1, c2, c3);
    let arr = Array2::from_shape_fn((4, 4), |(r, c)| mat[(r, c)]);
    Ok(arr.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn weyl_from_monodromy<'py>(
    py: Python<'py>,
    m: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let s = m.as_slice().expect("m not contiguous");
    let result = gate_primitives::weyl_from_monodromy(&[s[0], s[1], s[2]]);
    Ok(result.to_vec().into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn monodromy_from_weyl<'py>(
    py: Python<'py>,
    c1: f64,
    c2: f64,
    c3: f64,
) -> PyResult<Py<PyAny>> {
    let result = gate_primitives::monodromy_from_weyl(c1, c2, c3);
    Ok(result.to_vec().into_pyarray(py).into_any().unbind())
}

// ---------------------------------------------------------------------------
// Full solver
// ---------------------------------------------------------------------------

/// Solve multiple segments, using Rayon parallelism when beneficial.
///
/// For n < min_batch_size, solves sequentially (avoids Rayon dispatch overhead).
/// For n >= min_batch_size, uses Rayon work-stealing threadpool.
///
/// Returns list of (u0, u1, weyl_res, makhlin_res) tuples.
#[pyfunction]
fn solve_batch<'py>(
    py: Python<'py>,
    prefixes: Vec<PyReadonlyArray2<'py, Complex64>>,
    bases: Vec<PyReadonlyArray2<'py, Complex64>>,
    targets: Vec<PyReadonlyArray2<'py, Complex64>>,
    makhlin_tol: f64,
    weyl_tol: f64,
    min_batch_size: usize,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, f64, f64)>> {
    // Extract all matrices while we hold the GIL
    let inputs: Vec<_> = prefixes
        .iter()
        .zip(bases.iter())
        .zip(targets.iter())
        .map(|((p, b), t)| {
            (numpy_to_mat4(p), numpy_to_mat4(b), numpy_to_mat4(t))
        })
        .collect();

    // Solve with GIL released - sequential below threshold, parallel above.
    // Rayon par_iter wakes the entire thread pool (~140μs overhead).
    // For small N this exceeds the solve time itself, so we go sequential.
    let results: Vec<_> = py.detach(|| {
        if inputs.len() < min_batch_size {
            inputs
                .iter()
                .map(|(prefix, basis, target)| {
                    solver::solve(0, prefix, basis, target, makhlin_tol, weyl_tol)
                })
                .collect()
        } else {
            use rayon::prelude::*;
            inputs
                .par_iter()
                .map(|(prefix, basis, target)| {
                    solver::solve(0, prefix, basis, target, makhlin_tol, weyl_tol)
                })
                .collect()
        }
    });

    // Convert results back to numpy (need GIL)
    results
        .into_iter()
        .map(|(u0, u1, weyl_res, makhlin_res)| {
            Ok((
                mat2_to_numpy(py, &u0).into_any().unbind(),
                mat2_to_numpy(py, &u1).into_any().unbind(),
                weyl_res,
                makhlin_res,
            ))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Local equivalence recovery (uses Qiskit's TwoQubitWeylDecomposition)
// ---------------------------------------------------------------------------

/// Recover local equivalence corrections between two 4×4 unitaries.
///
/// Returns (k1, k2, k3, k4, global_phase) where k_i are 2×2 complex such that:
///   U_target ≈ exp(i·phase) · kron(k4, k3) · U_basis · kron(k2, k1)
#[pyfunction]
fn recover_local_equiv<'py>(
    py: Python<'py>,
    u_target: PyReadonlyArray2<'py, Complex64>,
    u_basis: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, f64)> {
    let target = numpy_to_mat4(&u_target);
    let basis = numpy_to_mat4(&u_basis);

    let result = py.detach(|| {
        recover::recover_local_equivalence(&target, &basis)
    }).map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        mat2_to_numpy(py, &result.k1).into_any().unbind(),
        mat2_to_numpy(py, &result.k2).into_any().unbind(),
        mat2_to_numpy(py, &result.k3).into_any().unbind(),
        mat2_to_numpy(py, &result.k4).into_any().unbind(),
        result.global_phase,
    ))
}

// ---------------------------------------------------------------------------
// Stitch: P accumulation + intermediate recovery in one Rust call
// ---------------------------------------------------------------------------

/// Stitch solved segments: accumulate P, compute intermediate recoveries.
///
/// Takes the initial gate matrix, all segment solutions (u0, u1),
/// basis gate matrices, and target canonical matrices for intermediates.
///
/// Returns (corrections, final_P) where corrections is a list of
/// (k3, k4, global_phase) for each intermediate recovery.
#[pyfunction]
fn stitch_segments<'py>(
    py: Python<'py>,
    initial_matrix: PyReadonlyArray2<'py, Complex64>,
    u0s: Vec<PyReadonlyArray2<'py, Complex64>>,
    u1s: Vec<PyReadonlyArray2<'py, Complex64>>,
    basis_matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
    target_canonical_matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
) -> PyResult<(Vec<(Py<PyAny>, Py<PyAny>, f64)>, Py<PyAny>)> {
    let init = numpy_to_mat4(&initial_matrix);

    // Convert all numpy arrays to nalgebra while holding GIL
    let u0_mats: Vec<Mat2> = u0s.iter().map(|u| {
        let s = u.as_slice().expect("u0 not contiguous");
        Mat2::from_row_slice(s)
    }).collect();
    let u1_mats: Vec<Mat2> = u1s.iter().map(|u| {
        let s = u.as_slice().expect("u1 not contiguous");
        Mat2::from_row_slice(s)
    }).collect();
    let basis_mats: Vec<Mat4> = basis_matrices.iter().map(|b| numpy_to_mat4(b)).collect();
    let target_mats: Vec<Mat4> = target_canonical_matrices.iter().map(|t| numpy_to_mat4(t)).collect();

    // Stitch with GIL released
    let result = py.detach(|| {
        recover::stitch_segments(&init, &u0_mats, &u1_mats, &basis_mats, &target_mats)
    }).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    // Convert results back to numpy
    let corrections: Vec<(Py<PyAny>, Py<PyAny>, f64)> = result.corrections.iter().map(|c| {
        (
            mat2_to_numpy(py, &c.k3).into_any().unbind(),
            mat2_to_numpy(py, &c.k4).into_any().unbind(),
            c.global_phase,
        )
    }).collect();

    let final_p = Array2::from_shape_fn((4, 4), |(r, c)| result.final_p[(r, c)])
        .into_pyarray(py).into_any().unbind();

    Ok((corrections, final_p))
}

// ---------------------------------------------------------------------------
// LP solver (dual revised simplex)
// ---------------------------------------------------------------------------

/// Dual revised simplex solver with warm-startable basis.
///
/// Solves: minimize c'x subject to Ax ≤ b
/// where A and c are fixed at construction, and b changes per solve().
#[pyclass]
struct DualSimplex {
    inner: lp::DualSimplexSolver,
}

#[pymethods]
impl DualSimplex {
    /// Create a new solver with fixed constraint matrix A, objective c,
    /// initial dual-feasible basis, and feasibility tolerance.
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
        // numpy row-major → nalgebra column-major
        let a_mat = nalgebra::DMatrix::from_fn(m, n, |r, c| a_slice[r * n + c]);
        let c_slice = c.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("c must be contiguous: {e}"))
        })?;
        let c_vec = nalgebra::DVector::from_column_slice(c_slice);

        Ok(Self {
            inner: lp::DualSimplexSolver::new(a_mat, c_vec, initial_basis, tol),
        })
    }

    /// Solve for a given b vector. Returns (x, feasible).
    ///
    /// x is a 1D numpy array (or None if infeasible).
    /// The basis is warm-started across calls.
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

    /// Reset the basis to a cold-start state.
    fn reset_basis(&mut self, basis: Vec<usize>) {
        self.inner.reset_basis(basis);
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(weyl_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(weyl_from_monodromy, m)?)?;
    m.add_function(wrap_pyfunction!(monodromy_from_weyl, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    m.add_function(wrap_pyfunction!(recover_local_equiv, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_segments, m)?)?;
    m.add_class::<DualSimplex>()?;
    Ok(())
}
