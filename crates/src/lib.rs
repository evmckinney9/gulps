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
// Local equivalence recovery
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
// Segment solver (diagnostic / per-segment access)
// ---------------------------------------------------------------------------

/// Solve one or more segments independently, returning per-segment results.
///
/// For production decomposition, use `solve_and_stitch` instead (handles
/// stitching and recovery). This function is useful for diagnostics that
/// need per-segment residuals.
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
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, f64, f64)>> {
    let inputs: Vec<_> = prefixes
        .iter()
        .zip(bases.iter())
        .zip(targets.iter())
        .map(|((p, b), t)| (numpy_to_mat4(p), numpy_to_mat4(b), numpy_to_mat4(t)))
        .collect();

    let results: Vec<_> = py.detach(|| {
        inputs
            .iter()
            .map(|(prefix, basis, target)| {
                solver::solve(0, prefix, basis, target, makhlin_tol, weyl_tol)
            })
            .collect()
    });

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
// Solve + stitch in one call
// ---------------------------------------------------------------------------

/// Solve all segments and stitch into circuit data.
///
/// Solves with canonical prefixes (Rayon above min_batch_size, sequential
/// below), then stitches with intermediate KAK recoveries. Consolidates
/// what was previously two separate FFI calls (solve_batch + stitch_segments).
///
/// Returns (u0s, u1s, k1, k2, k3, k4, global_phase).
#[pyfunction]
fn solve_and_stitch<'py>(
    py: Python<'py>,
    initial_matrix: PyReadonlyArray2<'py, Complex64>,
    basis_matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
    target_matrices: Vec<PyReadonlyArray2<'py, Complex64>>,
    final_target: PyReadonlyArray2<'py, Complex64>,
    makhlin_tol: f64,
    weyl_tol: f64,
    min_batch_size: usize,
    target_weyl_coords: Vec<[f64; 3]>,
) -> PyResult<(
    Vec<Py<PyAny>>,  // u0s (fused with intermediate corrections)
    Vec<Py<PyAny>>,  // u1s
    Py<PyAny>,       // k1 (final recovery)
    Py<PyAny>,       // k2
    Py<PyAny>,       // k3
    Py<PyAny>,       // k4
    f64,             // global_phase
)> {
    let init = numpy_to_mat4(&initial_matrix);
    let target = numpy_to_mat4(&final_target);
    let basis_mats: Vec<Mat4> = basis_matrices.iter().map(|b| numpy_to_mat4(b)).collect();
    let target_mats: Vec<Mat4> = target_matrices.iter().map(|t| numpy_to_mat4(t)).collect();
    let n = basis_mats.len();

    let result = py.detach(|| -> Result<_, String> {
        // Solve: canonical prefixes, Rayon above threshold
        let solve_one = |i: usize| {
            let prefix = if i == 0 { &init } else { &target_mats[i - 1] };
            solver::solve(0, prefix, &basis_mats[i], &target_mats[i], makhlin_tol, weyl_tol)
        };

        let results: Vec<_> = if n >= min_batch_size {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(solve_one).collect()
        } else {
            (0..n).map(solve_one).collect()
        };

        for (i, &(_, _, weyl_res, makhlin_res)) in results.iter().enumerate() {
            if weyl_res > weyl_tol && makhlin_res > makhlin_tol {
                return Err(format!(
                    "Segment {} synthesis failed (weyl={:.2e}, makhlin={:.2e})",
                    i + 1, weyl_res, makhlin_res
                ));
            }
        }

        let u0s: Vec<Mat2> = results.iter().map(|(u0, _, _, _)| *u0).collect();
        let u1s: Vec<Mat2> = results.iter().map(|(_, u1, _, _)| *u1).collect();

        // Stitch: P accumulation + intermediate KAK recovery + final recovery
        let stitch = recover::stitch_segments(
            &init, &u0s, &u1s, &basis_mats, &target_weyl_coords, &target,
        )?;

        let gphase = stitch.intermediate_gphase + stitch.final_recovery.global_phase;
        Ok((stitch.fused_u0s, stitch.fused_u1s, stitch.final_recovery, gphase))
    }).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let (u0s, u1s, recovery, gphase) = result;

    let u0_numpy: Vec<Py<PyAny>> = u0s.iter()
        .map(|u| mat2_to_numpy(py, u).into_any().unbind())
        .collect();
    let u1_numpy: Vec<Py<PyAny>> = u1s.iter()
        .map(|u| mat2_to_numpy(py, u).into_any().unbind())
        .collect();

    Ok((
        u0_numpy,
        u1_numpy,
        mat2_to_numpy(py, &recovery.k1).into_any().unbind(),
        mat2_to_numpy(py, &recovery.k2).into_any().unbind(),
        mat2_to_numpy(py, &recovery.k3).into_any().unbind(),
        mat2_to_numpy(py, &recovery.k4).into_any().unbind(),
        gphase,
    ))
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
    m.add_function(wrap_pyfunction!(solve_and_stitch, m)?)?;
    m.add_class::<DualSimplex>()?;
    Ok(())
}
