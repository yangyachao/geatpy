use numpy::ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn to_f64_array2<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(ro.as_array().to_owned());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<i64>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<i32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f64>>() {
        let n = ro.as_array().len();
        return Ok(ro.as_array().to_owned().into_shape_with_order((n, 1)).unwrap());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i64>>() {
        let n = ro.as_array().len();
        return Ok(ro.as_array().mapv(|x| x as f64).into_shape_with_order((n, 1)).unwrap());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i32>>() {
        let n = ro.as_array().len();
        return Ok(ro.as_array().mapv(|x| x as f64).into_shape_with_order((n, 1)).unwrap());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f32>>() {
        let n = ro.as_array().len();
        return Ok(ro.as_array().mapv(|x| x as f64).into_shape_with_order((n, 1)).unwrap());
    }
    if let Ok(arr) = obj.call_method1("astype", ("float64",)) {
        if let Ok(ro) = arr.extract::<PyReadonlyArray2<f64>>() {
            return Ok(ro.as_array().to_owned());
        }
        if let Ok(ro) = arr.extract::<PyReadonlyArray1<f64>>() {
            let n = ro.as_array().len();
            return Ok(ro.as_array().to_owned().into_shape_with_order((n, 1)).unwrap());
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err("Expected 1D or 2D numeric numpy array"))
}

pub fn to_f64_array1<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array1<f64>> {
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(ro.as_array().to_owned());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(ro.as_array().mapv(|x| x as f64));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f64>>() {
        let arr = ro.as_array();
        if arr.shape()[1] == 1 {
            return Ok(arr.column(0).to_owned());
        } else if arr.shape()[0] == 1 {
            return Ok(arr.row(0).to_owned());
        }
    }
    if let Ok(arr) = obj.call_method1("astype", ("float64",)) {
        if let Ok(ro) = arr.extract::<PyReadonlyArray1<f64>>() {
            return Ok(ro.as_array().to_owned());
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err("Expected 1D numeric numpy array"))
}
