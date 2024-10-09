// crates/src/basic_functions/mod.rs
pub mod basic_math;
pub mod basic_strings;

use pyo3::prelude::*;

#[pymodule]
pub fn basic_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(basic_math::add_in_rust))?;
    m.add_wrapped(wrap_pyfunction!(basic_math::subtract_in_rust))?;
    m.add_wrapped(wrap_pyfunction!(basic_strings::concat_in_rust))?;
    Ok(())
}
