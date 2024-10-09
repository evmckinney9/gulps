// crates/src/basic_functions/basic_strings.rs
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
pub fn concat_in_rust(a: String, b: String) -> PyResult<String> {
    Ok(a + &b)
}
