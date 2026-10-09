use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (*_args, **_kwargs))]
pub fn moeaplot<'py>(_py: Python<'py>, _args: &Bound<'py, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'py, pyo3::types::PyDict>>) -> PyResult<()> {
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (*_args, **_kwargs))]
pub fn soeaplot<'py>(_py: Python<'py>, _args: &Bound<'py, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'py, pyo3::types::PyDict>>) -> PyResult<()> {
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (*_args, **_kwargs))]
pub fn trcplot<'py>(_py: Python<'py>, _args: &Bound<'py, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'py, pyo3::types::PyDict>>) -> PyResult<()> {
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (*_args, **_kwargs))]
pub fn varplot<'py>(_py: Python<'py>, _args: &Bound<'py, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'py, pyo3::types::PyDict>>) -> PyResult<()> {
    Ok(())
}
