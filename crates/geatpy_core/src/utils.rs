use numpy::ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// numpy.asarray(obj, dtype=float64) for lists, tuples, scalars and other array-likes.
fn as_float_array<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = obj.py().import("numpy")?;
    np.call_method1(
        "ascontiguousarray",
        (np.call_method1("asarray", (obj, "float64"))?,),
    )
}

pub fn to_f64_array2<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    if obj.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a numeric array, got None",
        ));
    }
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
        return Ok(ro
            .as_array()
            .to_owned()
            .into_shape_with_order((n, 1))
            .unwrap());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i64>>() {
        let n = ro.as_array().len();
        return Ok(ro
            .as_array()
            .mapv(|x| x as f64)
            .into_shape_with_order((n, 1))
            .unwrap());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<i32>>() {
        let n = ro.as_array().len();
        return Ok(ro
            .as_array()
            .mapv(|x| x as f64)
            .into_shape_with_order((n, 1))
            .unwrap());
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f32>>() {
        let n = ro.as_array().len();
        return Ok(ro
            .as_array()
            .mapv(|x| x as f64)
            .into_shape_with_order((n, 1))
            .unwrap());
    }
    if let Ok(arr) = as_float_array(obj) {
        if let Ok(ro) = arr.extract::<PyReadonlyArray2<f64>>() {
            return Ok(ro.as_array().to_owned());
        }
        if let Ok(ro) = arr.extract::<PyReadonlyArray1<f64>>() {
            let n = ro.as_array().len();
            return Ok(ro
                .as_array()
                .to_owned()
                .into_shape_with_order((n, 1))
                .unwrap());
        }
        if let Ok(v) = arr.extract::<f64>() {
            return Ok(Array2::from_elem((1, 1), v));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected 1D or 2D numeric numpy array",
    ))
}

pub fn to_f64_array1<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Array1<f64>> {
    if obj.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a numeric array, got None",
        ));
    }
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
    if let Ok(arr) = as_float_array(obj) {
        if let Ok(ro) = arr.extract::<PyReadonlyArray1<f64>>() {
            return Ok(ro.as_array().to_owned());
        }
        if let Ok(ro) = arr.extract::<PyReadonlyArray2<f64>>() {
            let a = ro.as_array();
            if a.shape()[0] == 1 || a.shape()[1] == 1 {
                return Ok(a.iter().cloned().collect());
            }
        }
        if let Ok(v) = arr.extract::<f64>() {
            return Ok(Array1::from_elem(1, v));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected 1D numeric numpy array",
    ))
}

/// True when `obj` is a numpy array (or Python sequence) holding integers or booleans.
pub fn is_integer_like(obj: &Bound<'_, PyAny>) -> bool {
    obj.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "i" || k == "u" || k == "b")
        .unwrap_or(false)
}

/// Return a chromosome matrix, as int64 when the caller's input was an integer matrix.
pub fn chrom_out(py: Python<'_>, chrom: Array2<f64>, as_int: bool) -> PyResult<PyObject> {
    use numpy::IntoPyArray;
    if as_int {
        Ok(chrom
            .mapv(|x| x.round() as i64)
            .into_pyarray(py)
            .into_any()
            .unbind())
    } else {
        Ok(chrom.into_pyarray(py).into_any().unbind())
    }
}

/// Output of an 'RI' operator: like geatpy 2.7.0, an int32 matrix when every variable is an integer
/// (varTypes all 1), a float64 matrix otherwise.
pub fn ri_out(py: Python<'_>, chrom: Array2<f64>, discrete: &[bool]) -> PyResult<PyObject> {
    use numpy::IntoPyArray;
    if !discrete.is_empty() && discrete.iter().all(|&d| d) {
        Ok(chrom
            .mapv(|x| x.round() as i32)
            .into_pyarray(py)
            .into_any()
            .unbind())
    } else {
        Ok(chrom.into_pyarray(py).into_any().unbind())
    }
}

/// None / missing -> None; Python bool -> Some(Bool); number -> Scalar; 1-D array -> Array.
pub enum Knob {
    Bool(bool),
    Scalar(f64),
    Array(Vec<f64>),
}

pub fn knob(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Knob>> {
    let Some(o) = obj else { return Ok(None) };
    if o.is_none() {
        return Ok(None);
    }
    let type_name = o.get_type().name()?.to_string();
    if o.is_instance_of::<pyo3::types::PyBool>() || type_name == "bool_" || type_name == "bool" {
        return Ok(Some(Knob::Bool(o.is_truthy()?)));
    }
    if let Ok(v) = o.extract::<f64>() {
        return Ok(Some(Knob::Scalar(v)));
    }
    let a = to_f64_array2(o)?;
    Ok(Some(Knob::Array(a.iter().cloned().collect())))
}

/// Per-gene parameter: scalar broadcast, 1-D array of length `d`, or `default` when absent.
pub fn per_gene(
    obj: Option<&Bound<'_, PyAny>>,
    d: usize,
    default: f64,
    name: &str,
) -> PyResult<Vec<f64>> {
    match knob(obj)? {
        None => Ok(vec![default; d]),
        Some(Knob::Bool(b)) => Ok(vec![if b { 1.0 } else { 0.0 }; d]),
        Some(Knob::Scalar(v)) => Ok(vec![v; d]),
        Some(Knob::Array(v)) => {
            if v.len() != d {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "error: the length of {} must equal the chromosome length ({}).",
                    name, d
                )));
            }
            Ok(v)
        }
    }
}

/// Optional integer argument that may arrive as None.
pub fn opt_int(obj: Option<&Bound<'_, PyAny>>, default: i64) -> PyResult<i64> {
    match obj {
        Some(o) if !o.is_none() => Ok(o.extract::<f64>()? as i64),
        _ => Ok(default),
    }
}

/// FixType argument: 1 (default), 2, 3 or 4; anything else is rejected like in geatpy 2.7.0.
pub fn fix_type(obj: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<i64> {
    let t = opt_int(obj, 1)?;
    if !(1..=4).contains(&t) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: FixType must be an integer that is 1, 2, 3 or 4. (传入参数FixType必须是1,2,3或4。)",
            name
        )));
    }
    Ok(t)
}

/// Lower/upper bounds and spans as used by the real-valued operators: integer variables are
/// widened by 0.499999 on both sides so that rounding maps back uniformly onto the integers.
pub struct Bounds {
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub span: Vec<f64>,
    pub discrete: Vec<bool>,
}

pub fn bounds(field: &Array2<f64>, d: usize, widen_discrete: bool) -> PyResult<Bounds> {
    if field.shape()[0] < 2 || field.shape()[1] != d {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error: FieldDR must have 3 rows and as many columns as the chromosome length.",
        ));
    }
    let mut b = Bounds {
        lb: vec![0.0; d],
        ub: vec![0.0; d],
        span: vec![0.0; d],
        discrete: vec![false; d],
    };
    for j in 0..d {
        let discrete = field.shape()[0] > 2 && field[[2, j]] == 1.0;
        let (mut l, mut u) = (field[[0, j]], field[[1, j]]);
        if discrete && widen_discrete {
            l -= 0.499999;
            u += 0.499999;
        }
        b.lb[j] = l;
        b.ub[j] = u;
        b.span[j] = (u - l).abs();
        b.discrete[j] = discrete;
    }
    Ok(b)
}

/// Out-of-range repair. 1: truncate, 2: wrap around, 3: reflect, 4: uniform random, other: none.
/// A degenerate range (span <= 1e-15) collapses to lb, as in the reference implementation.
pub fn fix_value<R: rand::Rng>(
    x: f64,
    lb: f64,
    ub: f64,
    span: f64,
    fix_type: i64,
    rng: &mut R,
) -> f64 {
    if (1..=4).contains(&fix_type) && span <= 1e-15 {
        return lb;
    }
    match fix_type {
        1 => x.clamp(lb, ub),
        2 => {
            if x > ub {
                lb + (x - ub) % span
            } else if x < lb {
                ub - (lb - x) % span
            } else {
                x
            }
        }
        3 => {
            if x > ub {
                ub - (x - ub) % span
            } else if x < lb {
                lb + (lb - x) % span
            } else {
                x
            }
        }
        4 => {
            if x > ub || x < lb {
                lb + rng.gen::<f64>() * span
            } else {
                x
            }
        }
        _ => x,
    }
}

/// Multiply each objective by its maxormins flag (1: minimise, -1: maximise).
pub fn unify_objectives(
    obj: &mut Array2<f64>,
    maxormins: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if let Some(mom) = maxormins {
        if !mom.is_none() {
            let v = to_f64_array1(mom)?;
            if v.len() != obj.shape()[1] {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: The length of maxormins must equal the number of objectives.",
                ));
            }
            for mut row in obj.rows_mut() {
                for (x, m) in row.iter_mut().zip(v.iter()) {
                    *x *= m;
                }
            }
        }
    }
    Ok(())
}

/// Total positive constraint violation of each row, if CV is given.
pub fn violation(cv: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Option<Vec<f64>>> {
    match cv {
        Some(c) if !c.is_none() => {
            let a = to_f64_array2(c)?;
            if a.shape()[0] != n {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: CV and ObjV disagree in number of rows.",
                ));
            }
            Ok(Some(
                a.rows()
                    .into_iter()
                    .map(|r| r.iter().filter(|&&x| x > 0.0).sum())
                    .collect(),
            ))
        }
        _ => Ok(None),
    }
}

/// Feasibility rule used across the core: an infeasible row gets, on every objective,
/// the column maximum of the population plus its total violation.
pub fn penalise_infeasible(obj: &mut Array2<f64>, viol: &[f64]) {
    let m = obj.shape()[1];
    let col_max: Vec<f64> = (0..m)
        .map(|j| {
            obj.column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    for (i, &v) in viol.iter().enumerate() {
        if v > 0.0 {
            for j in 0..m {
                obj[[i, j]] = col_max[j] + v;
            }
        }
    }
}

/// ObjV converted to "minimise everything" with infeasible rows penalised.
pub fn prepared_objectives(
    obj_v: &Bound<'_, PyAny>,
    cv: Option<&Bound<'_, PyAny>>,
    maxormins: Option<&Bound<'_, PyAny>>,
) -> PyResult<Array2<f64>> {
    let mut obj = to_f64_array2(obj_v)?;
    unify_objectives(&mut obj, maxormins)?;
    if let Some(v) = violation(cv, obj.shape()[0])? {
        penalise_infeasible(&mut obj, &v);
    }
    Ok(obj)
}
