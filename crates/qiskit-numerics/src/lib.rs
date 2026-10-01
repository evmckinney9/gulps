//! Qiskit's two-qubit numerical routines, vendored until Qiskit's C API
//! exposes them. See README.md for provenance and local changes.

// Removing Python and circuit items leaves unused imports and helpers. Upstream
// code follows Qiskit's lint configuration, not this workspace's.
#![allow(unused_imports, dead_code, clippy::all)]

// Upstream modules name their utility crate `qiskit_util`; here it is `util`.
extern crate self as qiskit_util;

pub mod linalg;
pub mod two_qubit_decompose {
    pub mod common;
}
mod util;
pub use util::*;

/// Upstream's Python error types, reduced to messages: this crate does not use Python.
pub type PyErr = String;
/// Result type used by the vendored routines.
pub type PyResult<T> = Result<T, PyErr>;

/// Upstream's Python exception, reduced to its message.
pub struct QiskitError;

impl QiskitError {
    pub fn new_err(message: impl Into<String>) -> PyErr {
        message.into()
    }
}
