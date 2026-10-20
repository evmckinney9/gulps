//! `LocalEquivalenceClass`, a view over one core `Mono`.

use std::hash::{Hash, Hasher};

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::decomposer::compile_error;
use crate::{mat4_to_numpy, unitary};

use gulps_core::class::{self, Mono};
use gulps_core::{Unitary, canonical_matrix, map_batch};

/// The local-equivalence class of a two-qubit gate.
///
/// Construct from Weyl coordinates, or with :meth:`from_unitary`.
///
/// Two classes are equal, and hash equal, when their monodromy coordinates
/// round to the same point of a fixed grid, far finer than any gate's
/// precision.
///
/// Args:
///     weyl (Sequence[float]): Weyl coordinates ``(c1, c2, c3)`` in units of
///         pi, with ``c2 <= c1``, ``c1 + c2 <= 1``, and ``|c3| <= c2``, each
///         within the class tolerance. A point with ``c3 < 0``, or with
///         ``c3`` zero within the class tolerance and ``c1 > 1/2``, is
///         replaced by ``(1 - c1, c2, -c3)``, the same class.
///
/// Raises:
///     ``ValueError``: If a coordinate is not finite or the point is outside
///         that region.
#[pyclass(frozen, eq, hash, from_py_object, module = "gulps.invariants")]
#[derive(Clone, Copy)]
pub struct LocalEquivalenceClass {
    pub mono: Mono,
}

impl PartialEq for LocalEquivalenceClass {
    fn eq(&self, other: &Self) -> bool {
        self.cell() == other.cell()
    }
}

impl Hash for LocalEquivalenceClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cell().hash(state);
    }
}

impl LocalEquivalenceClass {
    /// The `CLASS_TOL` cell of each coordinate.
    #[allow(clippy::cast_possible_truncation)] // |c| <= 2, so the scaled cell is far inside i64
    fn cell(&self) -> [i64; 3] {
        self.mono
            .0
            .map(|c| (c / gulps_core::CLASS_TOL).round_ties_even() as i64)
    }
}

#[pymethods]
impl LocalEquivalenceClass {
    // PyO3 does not publish `#[new]` docs; the class docstring documents `weyl`.
    #[new]
    fn new(weyl: [f64; 3]) -> PyResult<Self> {
        if !weyl.iter().all(|c| c.is_finite()) {
            return Err(PyValueError::new_err("Weyl coordinates must be finite"));
        }
        // Either chart representative is accepted, up to the class tolerance.
        let [c1, c2, c3] = weyl;
        let tol = gulps_core::CLASS_TOL;
        if !(c2 <= c1 + tol && c1 + c2 <= 1.0 + tol && c3.abs() <= c2 + tol) {
            return Err(PyValueError::new_err(format!(
                "Weyl coordinates {weyl:?} are outside the canonical chamber: \
                 need c2 <= c1, c1 + c2 <= 1, and |c3| <= c2"
            )));
        }
        Ok(Self {
            mono: Mono::from_weyl(weyl),
        })
    }

    /// The class of a two-qubit gate or matrix.
    ///
    /// Args:
    ///     gate (qiskit.circuit.Gate | ``qiskit.quantum_info.Operator`` | np.ndarray):
    ///         Unitary to within the input tolerance; a nearly unitary matrix
    ///         is projected through its closest unitary.
    ///
    /// Returns:
    ///     ``LocalEquivalenceClass``
    #[staticmethod]
    fn from_unitary(py: Python<'_>, gate: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_unitaries(py, vec![gate])?
            .pop()
            .expect("one input"))
    }

    /// The classes of many two-qubit gates or matrices, computed in parallel.
    ///
    /// Args:
    ///     matrices (Sequence[qiskit.circuit.Gate | ``qiskit.quantum_info.Operator`` | np.ndarray | ``LocalEquivalenceClass``]):
    ///         Classes in the input are returned unchanged.
    ///
    /// Returns:
    ///     list[``LocalEquivalenceClass``]: In input order.
    ///
    /// Raises:
    ///     ``ValueError``: If a matrix is not a 4-by-4 unitary within the
    ///         input tolerance.
    ///     ``DecompositionError``: If the KAK of a matrix fails.
    #[staticmethod]
    pub fn from_unitaries(py: Python<'_>, matrices: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Self>> {
        let unitaries = matrices.iter().map(unitary).collect::<PyResult<Vec<_>>>()?;
        let monos = py
            .detach(|| map_batch(&unitaries, 1_500, Unitary::class))
            .map_err(|(_, err)| compile_error(py, err, None))?;
        Ok(monos.into_iter().map(|mono| Self { mono }).collect())
    }

    /// np.ndarray: Weyl coordinates ``(c1, c2, c3)`` in units of pi.
    #[getter]
    fn weyl<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &class::weyl_from_monodromy(&self.mono.0))
    }

    /// np.ndarray: The Makhlin invariants ``(Re G1, Im G1, Re G2)``.
    #[getter]
    fn makhlin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.mono.makhlin())
    }

    /// np.ndarray: The canonical gate ``exp(iπ/2 (c1 XX + c2 YY + c3 ZZ))`` of
    /// this class.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat4_to_numpy(py, &canonical_matrix(self.mono))
    }

    fn __repr__(&self) -> String {
        let [a, b, c] = class::weyl_from_monodromy(&self.mono.0);
        format!("LocalEquivalenceClass(({a}, {b}, {c}))")
    }

    /// Pickle the exact monodromy coordinates: Weyl coordinates round.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, ([f64; 3],))> {
        Ok((
            slf.get_type().getattr("_from_monodromy")?,
            (slf.get().mono.0,),
        ))
    }

    #[staticmethod]
    fn _from_monodromy(m: [f64; 3]) -> Self {
        Self { mono: Mono(m) }
    }
}
