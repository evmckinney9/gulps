//! `LocalEquivalenceClass`: the Python-facing local-equivalence class, a frozen view
//! over one core `Mono`; every numeric map delegates to gulps-core.

use std::hash::{Hash, Hasher};

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::decomposer::compile_error;
use crate::{checked_mat4, mat4_to_numpy};

use gulps_core::class::{self, Mono};
use gulps_core::{Mat4, canonical_matrix, monodromies};

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
        self.mono.key() == other.mono.key()
    }
}

impl Hash for LocalEquivalenceClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mono.key().hash(state);
    }
}

impl LocalEquivalenceClass {
    /// The class of a checked matrix, or `None` if its KAK fails.
    pub fn of(matrix: &Mat4) -> Option<Self> {
        let mono = monodromies(&[*matrix]).ok()?[0];
        Some(Self { mono })
    }

    /// The canonical gate of this class.
    pub fn mat4(&self) -> Mat4 {
        let [c1, c2, c3] = class::weyl_from_monodromy(&self.mono.0);
        canonical_matrix(c1, c2, c3)
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
        let ([c1, c2, c3], _) = class::fold_weyl(weyl);
        let mono = Mono(class::monodromy_from_weyl(c1, c2, c3));
        Ok(Self { mono })
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
        let coerced = (matrices.iter())
            .filter(|ob| !ob.is_instance_of::<Self>())
            .map(checked_mat4)
            .collect::<PyResult<Vec<_>>>()?;
        let mut monos = py
            .detach(|| monodromies(&coerced))
            .map_err(|err| compile_error(py, err, None))?
            .into_iter();
        Ok(matrices
            .into_iter()
            .map(|ob| {
                if let Ok(existing) = ob.cast::<Self>() {
                    return *existing.get();
                }
                Self {
                    mono: monos.next().expect("one class per matrix"),
                }
            })
            .collect())
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
        mat4_to_numpy(py, &self.mat4())
    }

    fn __repr__(&self) -> String {
        let [a, b, c] = class::weyl_from_monodromy(&self.mono.0);
        format!("LocalEquivalenceClass(({a}, {b}, {c}))")
    }
}
