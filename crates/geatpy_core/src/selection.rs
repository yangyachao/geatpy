use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

/// Duplicate selection: select top `sel_num` individuals sorted by fitness descending
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num))]
pub fn dup<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    if n == 0 {
        let empty = Array1::<i64>::zeros(0);
        return Ok(empty.into_pyarray(py));
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| fit[b].partial_cmp(&fit[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Array1::<i64>::zeros(sel_num);
    for i in 0..sel_num {
        out[i] = indices[i % n] as i64;
    }
    Ok(out.into_pyarray(py))
}

/// Tournament selection
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num, tour_size=2))]
pub fn tour<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
    tour_size: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    if n == 0 {
        return Ok(Array1::<i64>::zeros(0).into_pyarray(py));
    }
    let k = tour_size.max(1);

    let mut rng = rand::thread_rng();
    let mut out = Array1::<i64>::zeros(sel_num);

    for i in 0..sel_num {
        let mut best_idx = rng.gen_range(0..n);
        let mut best_val = fit[best_idx];

        for _ in 1..k {
            let candidate = rng.gen_range(0..n);
            if fit[candidate] > best_val {
                best_val = fit[candidate];
                best_idx = candidate;
            }
        }
        out[i] = best_idx as i64;
    }
    Ok(out.into_pyarray(py))
}

/// Elitist tournament selection
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num, tour_size=2))]
pub fn etour<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
    tour_size: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    if n == 0 || sel_num == 0 {
        return Ok(Array1::<i64>::zeros(0).into_pyarray(py));
    }

    // Best individual is always included as first elite
    let best_overall = fit
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut out = Array1::<i64>::zeros(sel_num);
    out[0] = best_overall as i64;

    let k = tour_size.max(1);
    let mut rng = rand::thread_rng();

    for i in 1..sel_num {
        let mut best_idx = rng.gen_range(0..n);
        let mut best_val = fit[best_idx];

        for _ in 1..k {
            let candidate = rng.gen_range(0..n);
            if fit[candidate] > best_val {
                best_val = fit[candidate];
                best_idx = candidate;
            }
        }
        out[i] = best_idx as i64;
    }
    Ok(out.into_pyarray(py))
}

/// Roulette Wheel Selection (RWS)
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num))]
pub fn rws<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    if n == 0 {
        return Ok(Array1::<i64>::zeros(0).into_pyarray(py));
    }

    let min_val = fit.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = if min_val < 0.0 { -min_val } else { 0.0 };

    let mut cumsum = Vec::with_capacity(n);
    let mut total = 0.0;
    for &v in &fit {
        total += v + shift;
        cumsum.push(total);
    }

    let mut rng = rand::thread_rng();
    let mut out = Array1::<i64>::zeros(sel_num);

    if total <= 1e-12 {
        for i in 0..sel_num {
            out[i] = rng.gen_range(0..n) as i64;
        }
        return Ok(out.into_pyarray(py));
    }

    for i in 0..sel_num {
        let r = rng.gen_range(0.0..total);
        let idx = match cumsum.binary_search_by(|v| v.partial_cmp(&r).unwrap()) {
            Ok(idx) => (idx + 1).min(n - 1),
            Err(idx) => idx.min(n - 1),
        };
        out[i] = idx as i64;
    }
    Ok(out.into_pyarray(py))
}

/// Stochastic Universal Sampling (SUS)
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num))]
pub fn sus<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    if n == 0 || sel_num == 0 {
        return Ok(Array1::<i64>::zeros(0).into_pyarray(py));
    }

    let min_val = fit.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = if min_val < 0.0 { -min_val } else { 0.0 };

    let mut cumsum = Vec::with_capacity(n);
    let mut total = 0.0;
    for &v in &fit {
        total += v + shift;
        cumsum.push(total);
    }

    let mut out = Array1::<i64>::zeros(sel_num);
    if total <= 1e-12 {
        let mut rng = rand::thread_rng();
        for i in 0..sel_num {
            out[i] = rng.gen_range(0..n) as i64;
        }
        return Ok(out.into_pyarray(py));
    }

    let step = total / (sel_num as f64);
    let mut rng = rand::thread_rng();
    let start = rng.gen_range(0.0..step);

    let mut cur_idx = 0;
    for i in 0..sel_num {
        let ptr = start + (i as f64) * step;
        while cur_idx < n - 1 && cumsum[cur_idx] < ptr {
            cur_idx += 1;
        }
        out[i] = cur_idx as i64;
    }
    Ok(out.into_pyarray(py))
}

fn get_n_ind(val: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(n) = val.extract::<usize>() {
        return Ok(n);
    }
    if let Ok(n) = val.extract::<i64>() {
        return Ok(n.max(0) as usize);
    }
    let fit = extract_1d(val)?;
    Ok(fit.len())
}

/// Uniform Random Selection (URS)
#[pyfunction]
#[pyo3(signature = (n_ind, sel_num))]
pub fn urs<'py>(py: Python<'py>, n_ind: &Bound<'py, PyAny>, sel_num: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let n = get_n_ind(n_ind)?;
    let mut rng = rand::thread_rng();
    let mut out = Array1::<i64>::zeros(sel_num);
    if n > 0 {
        for i in 0..sel_num {
            out[i] = rng.gen_range(0..n) as i64;
        }
    }
    Ok(out.into_pyarray(py))
}

/// Random Candidate Selection (RCS)
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num))]
pub fn rcs<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    urs(py, fitn_v, sel_num)
}

/// Random Permutation Selection (RPS)
#[pyfunction]
#[pyo3(signature = (n_ind, sel_num))]
pub fn rps<'py>(py: Python<'py>, n_ind: &Bound<'py, PyAny>, sel_num: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let n = get_n_ind(n_ind)?;
    let mut rng = rand::thread_rng();
    let mut base: Vec<i64> = (0..n as i64).collect();
    base.shuffle(&mut rng);
    let mut out = Array1::<i64>::zeros(sel_num);
    if n > 0 {
        for i in 0..sel_num {
            out[i] = base[i % n];
        }
    }
    Ok(out.into_pyarray(py))
}

/// Selection Dispatcher
#[pyfunction]
#[pyo3(signature = (sel_func, fitn_v, sel_num))]
pub fn selecting<'py>(
    py: Python<'py>,
    sel_func: &str,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    match sel_func.to_lowercase().as_str() {
        "tour" => tour(py, fitn_v, sel_num, 2),
        "dup" => dup(py, fitn_v, sel_num),
        "rws" => rws(py, fitn_v, sel_num),
        "sus" => sus(py, fitn_v, sel_num),
        "etour" => etour(py, fitn_v, sel_num, 2),
        "urs" => urs(py, fitn_v, sel_num),
        "rcs" => rcs(py, fitn_v, sel_num),
        "rps" => rps(py, fitn_v, sel_num),
        "otos" => otos(py, fitn_v, sel_num),
        "ecs" => ecs(py, fitn_v, sel_num),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported selection function: {}",
            sel_func
        ))),
    }
}

/// One-to-one selection
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num))]
pub fn otos<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    let mut out = Array1::<i64>::zeros(sel_num);
    if n >= 2 * sel_num {
        for i in 0..sel_num {
            if fit[i] >= fit[i + sel_num] {
                out[i] = i as i64;
            } else {
                out[i] = (i + sel_num) as i64;
            }
        }
    } else {
        let mut idxs: Vec<usize> = (0..n).collect();
        idxs.sort_by(|&a, &b| fit[b].partial_cmp(&fit[a]).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..sel_num {
            out[i] = idxs[i % n] as i64;
        }
    }
    Ok(out.into_pyarray(py))
}

/// Elite candidate selection
#[pyfunction]
#[pyo3(signature = (fitn_v, sel_num))]
pub fn ecs<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    if n == 0 {
        return Ok(Array1::<i64>::zeros(0).into_pyarray(py));
    }
    let mut best_idx = 0;
    let mut best_val = fit[0];
    for (i, &val) in fit.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    let mut out = Array1::<i64>::zeros(sel_num);
    for i in 0..sel_num {
        out[i] = best_idx as i64;
    }
    Ok(out.into_pyarray(py))
}

/// Multi-selection
#[pyfunction]
#[pyo3(signature = (sel_func, fitn_v, sel_num))]
pub fn mselecting<'py>(
    py: Python<'py>,
    sel_func: &str,
    fitn_v: &Bound<'py, PyAny>,
    sel_num: usize,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let list = pyo3::types::PyList::empty(py);
    if let Ok(seq) = fitn_v.downcast::<pyo3::types::PySequence>() {
        let len = seq.len().unwrap_or(0);
        let mut is_multi = false;
        if len > 0 {
            if let Ok(first) = seq.get_item(0) {
                if first.is_instance_of::<pyo3::types::PyList>() || first.hasattr("__array__").unwrap_or(false) {
                    is_multi = true;
                }
            }
        }
        if is_multi {
            for i in 0..len {
                let item = seq.get_item(i)?;
                let res = selecting(py, sel_func, &item, sel_num)?;
                list.append(res)?;
            }
            return Ok(list);
        }
    }
    let res = selecting(py, sel_func, fitn_v, sel_num)?;
    list.append(res)?;
    Ok(list)
}

/// Rank-based fitness assignment
#[pyfunction]
#[pyo3(signature = (obj_v, leg_v=None, sp=2.0))]
pub fn ranking<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    leg_v: Option<&Bound<'py, PyAny>>,
    sp: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = leg_v;
    let obj = extract_1d(obj_v)?;
    let n = obj.len();
    if n == 0 {
        return Ok(Array2::<f64>::zeros((0, 1)).into_pyarray(py));
    }
    if n == 1 {
        let mut fit = Array2::<f64>::zeros((1, 1));
        fit[[0, 0]] = 1.0;
        return Ok(fit.into_pyarray(py));
    }

    // Sort order: smaller obj value is better (higher rank)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| obj[a].partial_cmp(&obj[b]).unwrap_or(std::cmp::Ordering::Equal));

    // rank from worst (0) to best (n - 1)
    let mut fit = Array2::<f64>::zeros((n, 1));
    let n_minus_1 = (n - 1) as f64;

    for (rank_from_best, &idx) in order.iter().enumerate() {
        let rank = (n - 1 - rank_from_best) as f64;
        let fitness = 2.0 - sp + 2.0 * (sp - 1.0) * rank / n_minus_1;
        fit[[idx, 0]] = fitness.max(0.0);
    }

    Ok(fit.into_pyarray(py))
}

/// Single-objective scaling with constraint handling
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None))]
pub fn scaling<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raw_obj = extract_1d(obj_v)?;
    let n = raw_obj.len();
    if n == 0 {
        return Ok(Array2::<f64>::zeros((0, 1)).into_pyarray(py));
    }

    let mult = if let Some(m) = maxormins {
        if let Ok(vec) = m.extract::<Vec<f64>>() {
            if !vec.is_empty() && vec[0] < 0.0 { -1.0 } else { 1.0 }
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut obj: Vec<f64> = raw_obj.iter().map(|&v| v * mult).collect();

    // Constraint violation
    if let Some(cv_obj) = cv {
        if !cv_obj.is_none() {
            if let Ok(cv_2d) = cv_obj.extract::<PyReadonlyArray2<f64>>() {
                let cv_arr = cv_2d.as_array();
                let mut cv_sum = vec![0.0; n];
                let mut has_feasible = false;
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..cv_arr.shape()[1] {
                        let c = cv_arr[[i, j]];
                        if c > 0.0 {
                            s += c;
                        }
                    }
                    cv_sum[i] = s;
                    if s <= 1e-12 {
                        has_feasible = true;
                    }
                }

                if has_feasible {
                    let max_feasible = (0..n)
                        .filter(|&i| cv_sum[i] <= 1e-12)
                        .map(|i| obj[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                    for i in 0..n {
                        if cv_sum[i] > 1e-12 {
                            obj[i] = max_feasible + cv_sum[i];
                        }
                    }
                } else {
                    for i in 0..n {
                        obj[i] = cv_sum[i];
                    }
                }
            }
        }
    }

    // Rank individuals: best obj gets highest fitness
    let dummy_obj = Array1::from_vec(obj);
    let bound_arr = dummy_obj.into_pyarray(py);
    ranking(py, &bound_arr.as_any(), None, 2.0)
}

/// Power scaling
#[pyfunction]
#[pyo3(signature = (fitn_v, factor=2.0))]
pub fn powing<'py>(
    py: Python<'py>,
    fitn_v: &Bound<'py, PyAny>,
    factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fit = extract_1d(fitn_v)?;
    let n = fit.len();
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = fit[i].max(0.0).powf(factor);
    }
    Ok(out.into_pyarray(py))
}

fn extract_1d(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr2) = obj.extract::<PyReadonlyArray2<f64>>() {
        let arr = arr2.as_array();
        let n = arr.shape()[0];
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(arr[[i, 0]]);
        }
        Ok(v)
    } else if let Ok(arr1) = obj.extract::<PyReadonlyArray1<f64>>() {
        Ok(arr1.as_slice()?.to_vec())
    } else if let Ok(vec) = obj.extract::<Vec<f64>>() {
        Ok(vec)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected 1D or 2D NumPy array or float list for fitness values",
        ))
    }
}
