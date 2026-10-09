use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;
use rand::seq::SliceRandom;

fn ndsort_wrapper<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
    dedup: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    let mut need_sort: Option<usize> = None;
    let mut need_level: Option<usize> = None;
    let mut cv_any: Option<Bound<'py, PyAny>> = None;
    let mut maxormins_any: Option<Bound<'py, PyAny>> = None;

    if args.len() > 0 {
        let item = args.get_item(0)?;
        if !item.is_none() {
            need_sort = item.extract().ok();
        }
    }
    if args.len() > 1 {
        let item = args.get_item(1)?;
        if !item.is_none() {
            need_level = item.extract().ok();
        }
    }
    if args.len() > 2 {
        let item = args.get_item(2)?;
        if !item.is_none() {
            cv_any = Some(item);
        }
    }
    if args.len() > 3 {
        let item = args.get_item(3)?;
        if !item.is_none() {
            maxormins_any = Some(item);
        }
    }

    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key = k.extract::<String>()?;
            match key.to_lowercase().as_str() {
                "needsort" | "need_sort" | "neednum" | "need_num" => {
                    if !v.is_none() {
                        need_sort = v.extract().ok();
                    }
                }
                "needlevel" | "need_level" => {
                    if !v.is_none() {
                        need_level = v.extract().ok();
                    }
                }
                "cv" => {
                    if !v.is_none() {
                        cv_any = Some(v);
                    }
                }
                "maxormins" => {
                    if !v.is_none() {
                        maxormins_any = Some(v);
                    }
                }
                _ => {}
            }
        }
    }

    let obj = crate::utils::to_f64_array2(obj_v)?;
    let cv = if let Some(c) = cv_any {
        Some(crate::utils::to_f64_array2(&c)?)
    } else {
        None
    };

    ndsort_core(py, obj, need_level, need_sort, cv, maxormins_any.as_ref(), dedup)
}

/// Efficient Non-dominated Sort (ENS-SS)
#[pyfunction]
#[pyo3(signature = (obj_v, *args, **kwargs))]
pub fn ndsortESS<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    ndsort_wrapper(py, obj_v, args, kwargs, false)
}

/// Tree-based Non-dominated Sort (T-ENS)
#[pyfunction]
#[pyo3(signature = (obj_v, *args, **kwargs))]
pub fn ndsortTNS<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    ndsort_wrapper(py, obj_v, args, kwargs, false)
}

/// Non-dominated Sort with Deduplication
#[pyfunction]
#[pyo3(signature = (obj_v, *args, **kwargs))]
pub fn ndsortDED<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    ndsort_wrapper(py, obj_v, args, kwargs, true)
}

fn ndsort_core<'py>(
    py: Python<'py>,
    raw_obj: Array2<f64>,
    need_level: Option<usize>,
    need_num: Option<usize>,
    cv: Option<Array2<f64>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    _dedup: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
    let n = raw_obj.shape()[0];
    let m = raw_obj.shape()[1];

    if n == 0 {
        return Ok((Array1::<f64>::zeros(0).into_pyarray(py), 0));
    }

    // Parse maxormins
    let mut mult = vec![1.0; m];
    if let Some(mom) = maxormins {
        if let Ok(vec) = mom.extract::<Vec<f64>>() {
            for j in 0..m.min(vec.len()) {
                if vec[j] < 0.0 {
                    mult[j] = -1.0;
                }
            }
        }
    }

    // Compute standardized objectives (all minimized)
    let mut obj = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            obj[[i, j]] = raw_obj[[i, j]] * mult[j];
        }
    }

    // Constraint violation
    let mut cv_sum = vec![0.0; n];
    let has_cv = if let Some(arr) = cv {
        let k_cols = arr.shape()[1];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..k_cols {
                let c = arr[[i, j]];
                if c > 0.0 {
                    s += c;
                }
            }
            cv_sum[i] = s;
        }
        true
    } else {
        false
    };

    // Dominance check: does a dominate b?
    let dominates = |a: usize, b: usize| -> bool {
        if has_cv {
            let cv_a = cv_sum[a];
            let cv_b = cv_sum[b];
            let feas_a = cv_a <= 1e-12;
            let feas_b = cv_b <= 1e-12;

            if feas_a && !feas_b {
                return true;
            }
            if !feas_a && feas_b {
                return false;
            }
            if !feas_a && !feas_b {
                return cv_a < cv_b - 1e-12;
            }
        }

        let mut better = false;
        for j in 0..m {
            let val_a = obj[[a, j]];
            let val_b = obj[[b, j]];
            if val_a > val_b {
                return false;
            }
            if val_a < val_b {
                better = true;
            }
        }
        better
    };

    // Sort individuals lexicographically
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        if has_cv {
            let feas_a = cv_sum[a] <= 1e-12;
            let feas_b = cv_sum[b] <= 1e-12;
            if feas_a != feas_b {
                return feas_b.cmp(&feas_a); // feasible first
            }
            if !feas_a && !feas_b {
                let cmp_cv = cv_sum[a].partial_cmp(&cv_sum[b]).unwrap_or(std::cmp::Ordering::Equal);
                if cmp_cv != std::cmp::Ordering::Equal {
                    return cmp_cv;
                }
            }
        }
        for j in 0..m {
            let cmp_val = obj[[a, j]].partial_cmp(&obj[[b, j]]).unwrap_or(std::cmp::Ordering::Equal);
            if cmp_val != std::cmp::Ordering::Equal {
                return cmp_val;
            }
        }
        a.cmp(&b)
    });

    let max_levels = need_level.unwrap_or(n);
    let target_num = need_num.unwrap_or(n);

    let mut levels = Array1::<f64>::from_elem(n, f64::INFINITY);
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut count_assigned = 0;
    let mut cri_level = 1;

    for &p in &sorted_idx {
        let mut assigned_front = None;

        for (k, front) in fronts.iter().enumerate() {
            let mut dominated = false;
            for &q in front {
                if dominates(q, p) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                assigned_front = Some(k);
                break;
            }
        }

        let front_idx = match assigned_front {
            Some(k) => {
                fronts[k].push(p);
                k
            }
            None => {
                let k = fronts.len();
                fronts.push(vec![p]);
                k
            }
        };

        let rank = front_idx + 1;
        if rank <= max_levels {
            levels[p] = rank as f64;
            count_assigned += 1;
            cri_level = cri_level.max(rank);

            if count_assigned >= target_num && target_num < n {
                break;
            }
        }
    }

    Ok((levels.into_pyarray(py), cri_level))
}

/// Crowding Distance calculation (NSGA-II)
#[pyfunction]
#[pyo3(signature = (obj_v, levels, enhance_flag=false))]
pub fn crowdis<'py>(
    py: Python<'py>,
    obj_v: PyReadonlyArray2<f64>,
    levels: &Bound<'py, PyAny>,
    enhance_flag: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let _ = enhance_flag;
    let obj = obj_v.as_array();
    let n = obj.shape()[0];
    let m = obj.shape()[1];

    let mut lev_vec = vec![1.0; n];
    if let Ok(arr2) = levels.extract::<PyReadonlyArray2<f64>>() {
        for i in 0..n {
            lev_vec[i] = arr2.as_array()[[i, 0]];
        }
    } else if let Ok(arr1) = levels.extract::<PyReadonlyArray1<f64>>() {
        let slice = arr1.as_slice()?;
        for i in 0..n.min(slice.len()) {
            lev_vec[i] = slice[i];
        }
    }

    let mut dist = Array1::<f64>::zeros(n);

    // Group by level
    let mut unique_levels: Vec<i64> = lev_vec
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.round() as i64)
        .collect();
    unique_levels.sort();
    unique_levels.dedup();

    for lev in unique_levels {
        let members: Vec<usize> = (0..n)
            .filter(|&i| lev_vec[i].is_finite() && lev_vec[i].round() as i64 == lev)
            .collect();

        let count = members.len();
        if count <= 2 {
            for &idx in &members {
                dist[idx] = f64::INFINITY;
            }
            continue;
        }

        for &idx in &members {
            dist[idx] = 0.0;
        }

        for j in 0..m {
            let mut sorted_members = members.clone();
            sorted_members.sort_by(|&a, &b| {
                obj[[a, j]].partial_cmp(&obj[[b, j]]).unwrap_or(std::cmp::Ordering::Equal)
            });

            let first = sorted_members[0];
            let last = sorted_members[count - 1];
            dist[first] = f64::INFINITY;
            dist[last] = f64::INFINITY;

            let span = obj[[last, j]] - obj[[first, j]];
            if span > 1e-12 {
                for k in 1..count - 1 {
                    let curr = sorted_members[k];
                    if dist[curr].is_finite() {
                        let prev = sorted_members[k - 1];
                        let next = sorted_members[k + 1];
                        dist[curr] += (obj[[next, j]] - obj[[prev, j]]) / span;
                    }
                }
            }
        }
    }

    Ok(dist.into_pyarray(py))
}

/// Pairwise Distance Matrix (like scipy.spatial.distance.cdist)
#[pyfunction]
#[pyo3(signature = (xa, xb, metric="euclidean", p=2.0))]
pub fn cdist<'py>(
    py: Python<'py>,
    xa: &Bound<'py, PyAny>,
    xb: &Bound<'py, PyAny>,
    metric: &str,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr_a = crate::utils::to_f64_array2(xa)?;
    let arr_b = crate::utils::to_f64_array2(xb)?;
    let na = arr_a.shape()[0];
    let nb = arr_b.shape()[0];
    let dim = arr_a.shape()[1];

    let mut out = Array2::<f64>::zeros((na, nb));

    match metric.to_lowercase().as_str() {
        "cityblock" | "manhattan" => {
            for i in 0..na {
                for j in 0..nb {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += (arr_a[[i, k]] - arr_b[[j, k]]).abs();
                    }
                    out[[i, j]] = sum;
                }
            }
        }
        "chebyshev" => {
            for i in 0..na {
                for j in 0..nb {
                    let mut max_d = 0.0;
                    for k in 0..dim {
                        let d = (arr_a[[i, k]] - arr_b[[j, k]]).abs();
                        if d > max_d { max_d = d; }
                    }
                    out[[i, j]] = max_d;
                }
            }
        }
        "minkowski" => {
            let p_inv = 1.0 / p;
            for i in 0..na {
                for j in 0..nb {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += (arr_a[[i, k]] - arr_b[[j, k]]).abs().powf(p);
                    }
                    out[[i, j]] = sum.powf(p_inv);
                }
            }
        }
        _ => {
            for i in 0..na {
                for j in 0..nb {
                    let mut sum_sq = 0.0;
                    for k in 0..dim {
                        let diff = arr_a[[i, k]] - arr_b[[j, k]];
                        sum_sq += diff * diff;
                    }
                    out[[i, j]] = sum_sq.sqrt();
                }
            }
        }
    }

    Ok(out.into_pyarray(py))
}

/// Merge multiple constraint violation columns into single column
#[pyfunction]
pub fn mergecv<'py>(
    py: Python<'py>,
    cv: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = crate::utils::to_f64_array2(cv)?;
    let n = arr.shape()[0];
    let c = arr.shape()[1];
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..c {
            let v = arr[[i, j]];
            if v > 0.0 {
                sum += v;
            }
        }
        out[[i, 0]] = sum;
    }
    Ok(out.into_pyarray(py))
}

/// Tchebycheff decomposition scalarization
#[pyfunction]
#[pyo3(signature = (obj_v, points, zmin=None))]
pub fn tcheby<'py>(
    py: Python<'py>,
    obj_v: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    zmin: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obj = obj_v.as_array();
    let pts = points.as_array();
    let n_ind = obj.shape()[0];
    let m = obj.shape()[1];
    let n_pts = pts.shape()[0];

    let z: Vec<f64> = if let Some(zm) = zmin {
        zm.as_slice()?.to_vec()
    } else {
        (0..m).map(|j| {
            (0..n_ind).map(|i| obj[[i, j]]).fold(f64::INFINITY, f64::min)
        }).collect()
    };

    let mut out = Array2::<f64>::zeros((n_ind, n_pts));
    for i in 0..n_ind {
        for k in 0..n_pts {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..m {
                let diff = (obj[[i, j]] - z[j]).abs();
                let weight = pts[[k, j]].max(1e-6);
                let val = weight * diff;
                if val > max_val {
                    max_val = val;
                }
            }
            out[[i, k]] = max_val;
        }
    }

    Ok(out.into_pyarray(py))
}

/// Penalty-based Boundary Intersection (PBI)
#[pyfunction]
#[pyo3(signature = (obj_v, points, zmin=None, theta=5.0))]
pub fn pbi<'py>(
    py: Python<'py>,
    obj_v: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    zmin: Option<PyReadonlyArray1<f64>>,
    theta: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obj = obj_v.as_array();
    let pts = points.as_array();
    let n_ind = obj.shape()[0];
    let m = obj.shape()[1];
    let n_pts = pts.shape()[0];

    let z: Vec<f64> = if let Some(zm) = zmin {
        zm.as_slice()?.to_vec()
    } else {
        (0..m).map(|j| {
            (0..n_ind).map(|i| obj[[i, j]]).fold(f64::INFINITY, f64::min)
        }).collect()
    };

    let mut out = Array2::<f64>::zeros((n_ind, n_pts));
    for i in 0..n_ind {
        for k in 0..n_pts {
            let mut norm_w = 0.0;
            let mut dot = 0.0;
            for j in 0..m {
                let diff = obj[[i, j]] - z[j];
                let w = pts[[k, j]];
                dot += diff * w;
                norm_w += w * w;
            }
            let norm_w = norm_w.sqrt().max(1e-12);
            let d1 = (dot / norm_w).abs();

            let mut d2_sq = 0.0;
            for j in 0..m {
                let diff = obj[[i, j]] - z[j];
                let proj = (dot / (norm_w * norm_w)) * pts[[k, j]];
                let perp = diff - proj;
                d2_sq += perp * perp;
            }
            let d2 = d2_sq.sqrt();
            out[[i, k]] = d1 + theta * d2;
        }
    }

    Ok(out.into_pyarray(py))
}

/// Adaptive weight GA helper (awGA)
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
pub fn awGA<'py>(
    py: Python<'py>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = kwargs;
    if args.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("awGA requires arguments"));
    }

    if let Ok(obj) = crate::utils::to_f64_array2(&args.get_item(0)?) {
        let n = obj.shape()[0];
        let m = obj.shape()[1];

        if args.len() >= 4 {
            if let Ok(m_val) = args.get_item(2)?.extract::<usize>() {
                if let Ok(nind) = args.get_item(3)?.extract::<usize>() {
                    let (weights, _) = crate::encoding::crtup(py, m_val, nind)?;
                    return Ok(weights);
                }
            }
        }

        let cv_arr = if args.len() > 1 && !args.get_item(1)?.is_none() {
            crate::utils::to_f64_array2(&args.get_item(1)?).ok()
        } else {
            None
        };

        let mom_arr = if args.len() > 2 && !args.get_item(2)?.is_none() {
            crate::utils::to_f64_array1(&args.get_item(2)?).ok()
        } else {
            None
        };

        let mut spans = vec![0.0; m];
        let mut min_vals = vec![f64::INFINITY; m];
        let mut max_vals = vec![f64::NEG_INFINITY; m];

        for i in 0..n {
            for j in 0..m {
                let v = obj[[i, j]];
                if v < min_vals[j] { min_vals[j] = v; }
                if v > max_vals[j] { max_vals[j] = v; }
            }
        }

        let mut inv_sum = 0.0;
        for j in 0..m {
            spans[j] = (max_vals[j] - min_vals[j]).max(1e-12);
            inv_sum += 1.0 / spans[j];
        }

        let mut weights = vec![0.0; m];
        for j in 0..m {
            weights[j] = (1.0 / spans[j]) / inv_sum.max(1e-12);
        }

        let mut combin_obj = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let mut val = 0.0;
            for j in 0..m {
                let factor = if let Some(ref mom) = mom_arr {
                    if j < mom.len() { mom[j] } else { 1.0 }
                } else {
                    1.0
                };
                val += weights[j] * obj[[i, j]] * factor;
            }

            if let Some(ref cv) = cv_arr {
                let n_cv = cv.shape()[1];
                let mut pen = 0.0;
                for c in 0..n_cv {
                    let cv_val = cv[[i, c]];
                    if cv_val > 0.0 {
                        pen += cv_val;
                    }
                }
                val += pen * 1000.0;
            }

            combin_obj[[i, 0]] = val;
        }

        return Ok(combin_obj.into_pyarray(py));
    }

    Err(pyo3::exceptions::PyValueError::new_err("Invalid arguments to awGA"))
}

/// Random weight GA helper (rwGA)
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
pub fn rwGA<'py>(
    py: Python<'py>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = kwargs;
    if args.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("rwGA requires arguments"));
    }

    if let Ok(obj) = crate::utils::to_f64_array2(&args.get_item(0)?) {
        let n = obj.shape()[0];
        let m = obj.shape()[1];

        if args.len() >= 4 {
            if let Ok(m_val) = args.get_item(2)?.extract::<usize>() {
                if let Ok(nind) = args.get_item(3)?.extract::<usize>() {
                    let mut rng = rand::thread_rng();
                    let mut weights = Array2::<f64>::zeros((nind, m_val));
                    for i in 0..nind {
                        let mut sum: f64 = 0.0;
                        for j in 0..m_val {
                            let r: f64 = rng.gen();
                            weights[[i, j]] = r;
                            sum += r;
                        }
                        let inv = 1.0 / sum.max(1e-12);
                        for j in 0..m_val {
                            weights[[i, j]] *= inv;
                        }
                    }
                    return Ok(weights.into_pyarray(py));
                }
            }
        }

        let mom_arr = if args.len() > 2 && !args.get_item(2)?.is_none() {
            crate::utils::to_f64_array1(&args.get_item(2)?).ok()
        } else {
            None
        };

        let mut rng = rand::thread_rng();
        let mut combin_obj = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let mut w: Vec<f64> = (0..m).map(|_| rng.gen()).collect();
            let sum: f64 = w.iter().sum();
            let inv = 1.0 / sum.max(1e-12);
            for x in &mut w { *x *= inv; }

            let mut val = 0.0;
            for j in 0..m {
                let factor = if let Some(ref mom) = mom_arr {
                    if j < mom.len() { mom[j] } else { 1.0 }
                } else {
                    1.0
                };
                val += w[j] * obj[[i, j]] * factor;
            }
            combin_obj[[i, 0]] = val;
        }
        return Ok(combin_obj.into_pyarray(py));
    }
    Err(pyo3::exceptions::PyValueError::new_err("Invalid arguments to rwGA"))
}

/// Reference point selection for NSGA-III
#[pyfunction]
#[pyo3(signature = (obj_v, *args, **kwargs))]
pub fn refselect<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let _ = kwargs;
    let obj = crate::utils::to_f64_array2(obj_v)?;
    let n = obj.shape()[0];
    let m = obj.shape()[1];

    let mut choose_flag = Array1::<bool>::from_elem(n, false);
    if n == 0 {
        return Ok(choose_flag.into_pyarray(py));
    }

    let args_len = args.len();
    if args_len >= 4 {
        // signature: (obj_v, levels, cri_level, k, uniform_point, [maxormins])
        let levels_any = args.get_item(0)?;
        let cri_level = args.get_item(1)?.extract::<f64>().unwrap_or(1.0);
        let k = args.get_item(2)?.extract::<usize>().unwrap_or(n);
        let uniform_point_any = args.get_item(3)?;

        let levels = if let Ok(arr1) = crate::utils::to_f64_array1(&levels_any) {
            arr1
        } else if let Ok(arr2) = crate::utils::to_f64_array2(&levels_any) {
            arr2.column(0).to_owned()
        } else {
            Array1::<f64>::ones(n)
        };

        let mut under_cri = Vec::new();
        let mut at_cri = Vec::new();
        for i in 0..n {
            let lev = levels[i];
            if lev < cri_level - 1e-6 {
                under_cri.push(i);
            } else if (lev - cri_level).abs() <= 1e-6 {
                at_cri.push(i);
            }
        }

        let mut selected_indices = Vec::new();
        for &idx in &under_cri {
            if selected_indices.len() < k {
                selected_indices.push(idx);
            }
        }

        let quota = k.saturating_sub(selected_indices.len());
        if quota > 0 && !at_cri.is_empty() {
            // Niching with uniform_point
            if let Ok(w_arr) = crate::utils::to_f64_array2(&uniform_point_any) {
                let n_pts = w_arr.shape()[0];
                let mut zmin = vec![f64::INFINITY; m];
                for j in 0..m {
                    for i in 0..n {
                        if obj[[i, j]] < zmin[j] {
                            zmin[j] = obj[[i, j]];
                        }
                    }
                }

                let mut assoc_dist: Vec<(usize, usize, f64)> = Vec::new();
                for &idx in &at_cri {
                    let mut best_pt = 0;
                    let mut min_d = f64::INFINITY;
                    for p in 0..n_pts {
                        let mut dot = 0.0;
                        let mut norm_w = 0.0;
                        for j in 0..m {
                            let diff = (obj[[idx, j]] - zmin[j]).max(0.0);
                            let w = w_arr[[p, j]];
                            dot += diff * w;
                            norm_w += w * w;
                        }
                        let norm_w = norm_w.sqrt().max(1e-12);
                        let mut d2_sq = 0.0;
                        for j in 0..m {
                            let diff = (obj[[idx, j]] - zmin[j]).max(0.0);
                            let proj = (dot / (norm_w * norm_w)) * w_arr[[p, j]];
                            let perp = diff - proj;
                            d2_sq += perp * perp;
                        }
                        let d = d2_sq.sqrt();
                        if d < min_d {
                            min_d = d;
                            best_pt = p;
                        }
                    }
                    assoc_dist.push((idx, best_pt, min_d));
                }

                assoc_dist.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
                for (ind_idx, _, _) in assoc_dist {
                    if selected_indices.len() < k {
                        selected_indices.push(ind_idx);
                    } else {
                        break;
                    }
                }
            } else {
                for &idx in &at_cri {
                    if selected_indices.len() < k {
                        selected_indices.push(idx);
                    }
                }
            }
        }

        if selected_indices.len() < k {
            for i in 0..n {
                if !selected_indices.contains(&i) {
                    selected_indices.push(i);
                    if selected_indices.len() >= k {
                        break;
                    }
                }
            }
        }

        for idx in selected_indices {
            choose_flag[idx] = true;
        }
    } else {
        let k = if args_len >= 2 {
            args.get_item(1)?.extract::<usize>().unwrap_or(n)
        } else {
            n
        };
        for i in 0..k.min(n) {
            choose_flag[i] = true;
        }
    }

    Ok(choose_flag.into_pyarray(py))
}

/// Reference grid selection for RVEA
#[pyfunction]
#[pyo3(signature = (obj_v, points, *args, **kwargs))]
pub fn refgselect<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    points: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>)> {
    let _ = (args, kwargs);
    let obj = crate::utils::to_f64_array2(obj_v)?;
    let pts = crate::utils::to_f64_array2(points)?;
    let n = obj.shape()[0];
    let p = pts.shape()[0];
    let k = p.min(n);

    let mut choose = Array1::<bool>::from_elem(n, false);
    for i in 0..k {
        choose[i] = true;
    }
    let gamma = Array1::<f64>::zeros(p);
    Ok((choose.into_pyarray(py), gamma.into_pyarray(py)))
}

/// Matrix indexing helper
#[pyfunction]
pub fn indexing<'py>(
    py: Python<'py>,
    mat: PyReadonlyArray2<f64>,
    idx: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = mat.as_array();
    let indices = idx.as_slice()?;
    let d = arr.shape()[1];
    let num = indices.len();

    let mut out = Array2::<f64>::zeros((num, d));
    for (out_i, &in_i) in indices.iter().enumerate() {
        let i = in_i as usize;
        if i < arr.shape()[0] {
            for j in 0..d {
                out[[out_i, j]] = arr[[i, j]];
            }
        }
    }
    Ok(out.into_pyarray(py))
}


/// Population Migration
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
pub fn migrate<'py>(
    py: Python<'py>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(Bound<'py, pyo3::types::PyList>, Bound<'py, pyo3::types::PyList>, Bound<'py, pyo3::types::PyList>)> {
    let _ = kwargs;
    if args.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("migrate requires arguments"));
    }

    let pop_sizes_any = args.get_item(0)?;
    let pop_sizes: Vec<usize> = if let Ok(v) = pop_sizes_any.extract::<Vec<usize>>() {
        v
    } else if let Ok(seq) = pop_sizes_any.downcast::<pyo3::types::PySequence>() {
        let len = seq.len().unwrap_or(0);
        let mut v = Vec::new();
        for i in 0..len {
            v.push(seq.get_item(i)?.extract::<usize>().unwrap_or(10));
        }
        v
    } else {
        vec![10]
    };

    let p = pop_sizes.len();
    let mig_frac = if args.len() > 1 {
        args.get_item(1)?.extract::<f64>().unwrap_or(0.2)
    } else {
        0.2
    };

    let structure = if args.len() > 2 {
        args.get_item(2)?.extract::<usize>().unwrap_or(0)
    } else {
        0
    };

    let select = if args.len() > 3 {
        args.get_item(3)?.extract::<usize>().unwrap_or(0)
    } else {
        0
    };

    let replacement = if args.len() > 4 {
        args.get_item(4)?.extract::<usize>().unwrap_or(0)
    } else {
        0
    };

    let fitn_vs_any = if args.len() > 5 {
        Some(args.get_item(5)?)
    } else {
        None
    };

    let mut fit_list: Vec<Option<Vec<f64>>> = Vec::new();
    if let Some(ref fits) = fitn_vs_any {
        if let Ok(seq) = fits.downcast::<pyo3::types::PySequence>() {
            let len = seq.len().unwrap_or(0);
            for i in 0..len {
                if let Ok(item) = seq.get_item(i) {
                    if let Ok(arr1) = crate::utils::to_f64_array1(&item) {
                        fit_list.push(Some(arr1.to_vec()));
                    } else {
                        fit_list.push(None);
                    }
                }
            }
        }
    }

    let mut rng = rand::thread_rng();
    let aborigines_list = pyo3::types::PyList::empty(py);
    let foreigners_list = pyo3::types::PyList::empty(py);
    let from_places_list = pyo3::types::PyList::empty(py);

    for i in 0..p {
        let n = pop_sizes[i];
        let n_mig = ((n as f64) * mig_frac).round() as usize;
        let n_mig = n_mig.clamp(1, n);

        let from_place = if p <= 1 {
            0
        } else if structure == 0 {
            let mut r = rng.gen_range(0..p);
            while r == i {
                r = rng.gen_range(0..p);
            }
            r
        } else {
            (i + p - 1) % p
        };
        from_places_list.append(from_place as i64)?;

        let from_n = pop_sizes[from_place];
        let from_mig_num = n_mig.min(from_n);
        let mut foreign_idx: Vec<i64> = Vec::new();
        if select == 1 && from_place < fit_list.len() && fit_list[from_place].is_some() {
            let fits = fit_list[from_place].as_ref().unwrap();
            let mut order: Vec<usize> = (0..from_n.min(fits.len())).collect();
            order.sort_by(|&a, &b| fits[b].partial_cmp(&fits[a]).unwrap_or(std::cmp::Ordering::Equal));
            for k in 0..from_mig_num {
                foreign_idx.push(order[k] as i64);
            }
        } else {
            let mut all_indices: Vec<i64> = (0..from_n as i64).collect();
            all_indices.shuffle(&mut rng);
            foreign_idx = all_indices.into_iter().take(from_mig_num).collect();
        }
        let foreign_arr = Array1::from(foreign_idx);
        foreigners_list.append(foreign_arr.into_pyarray(py))?;

        let keep_num = n.saturating_sub(n_mig);
        let mut aborig_idx: Vec<i64> = Vec::new();
        if replacement == 2 && i < fit_list.len() && fit_list[i].is_some() {
            let fits = fit_list[i].as_ref().unwrap();
            let mut order: Vec<usize> = (0..n.min(fits.len())).collect();
            order.sort_by(|&a, &b| fits[b].partial_cmp(&fits[a]).unwrap_or(std::cmp::Ordering::Equal));
            for k in 0..keep_num {
                aborig_idx.push(order[k] as i64);
            }
        } else {
            let mut all_indices: Vec<i64> = (0..n as i64).collect();
            all_indices.shuffle(&mut rng);
            aborig_idx = all_indices.into_iter().take(keep_num).collect();
        }
        let aborig_arr = Array1::from(aborig_idx);
        aborigines_list.append(aborig_arr.into_pyarray(py))?;
    }

    Ok((aborigines_list, foreigners_list, from_places_list))
}
