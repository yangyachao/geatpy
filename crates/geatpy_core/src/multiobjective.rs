use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

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

    ndsort_core(
        py,
        obj,
        need_level,
        need_sort,
        cv,
        maxormins_any.as_ref(),
        dedup,
    )
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
    if let Some(ref c) = cv {
        if c.shape()[0] != n {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in ndsort: CV and ObjV disagree in number of rows.",
            ));
        }
    }

    // Unify to minimisation.
    let mut mult = vec![1.0; m];
    if let Some(mom) = maxormins {
        if !mom.is_none() {
            let v = crate::utils::to_f64_array1(mom)?;
            if v.len() != m {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in ndsort: maxormins and ObjV disagree in number of objectives.",
                ));
            }
            for j in 0..m {
                mult[j] = v[j];
            }
        }
    }
    let mut obj: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| raw_obj[[i, j]] * mult[j]).collect())
        .collect();

    // Constraint domination: an infeasible individual gets (column maximum + total violation)
    // on every objective, so feasible ones dominate it and infeasible ones compare by violation.
    if let Some(c) = cv {
        let col_max: Vec<f64> = (0..m)
            .map(|j| obj.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        for (row, viol) in obj.iter_mut().zip(c.rows()) {
            let s: f64 = viol.iter().filter(|&&x| x > 0.0).sum();
            if s > 0.0 {
                for (x, mx) in row.iter_mut().zip(&col_max) {
                    *x = mx + s;
                }
            }
        }
    }

    // Unique rows in lexicographic order, with multiplicities.
    let mut order: Vec<usize> = (0..n).collect();
    let lex = |a: &Vec<f64>, b: &Vec<f64>| {
        for j in 0..m {
            match a[j].partial_cmp(&b[j]) {
                Some(std::cmp::Ordering::Equal) | None => continue,
                Some(o) => return o,
            }
        }
        std::cmp::Ordering::Equal
    };
    order.sort_by(|&a, &b| lex(&obj[a], &obj[b]));
    let mut uni: Vec<usize> = Vec::new(); // representative row of each unique vector
    let mut rep: Vec<usize> = Vec::new();
    let mut inverse = vec![0usize; n];
    for &i in &order {
        match uni.last() {
            Some(&u) if lex(&obj[u], &obj[i]) == std::cmp::Ordering::Equal => {
                *rep.last_mut().unwrap() += 1;
            }
            _ => {
                uni.push(i);
                rep.push(1);
            }
        }
        inverse[i] = uni.len() - 1;
    }
    let u = uni.len();

    // Efficient non-dominated sort, one complete front per pass.
    let nums = need_num.unwrap_or(n);
    let max_level = need_level.unwrap_or(n);
    let mut lv = vec![0usize; u];
    let mut cri_level = 0usize;
    let mut sums = 0usize;
    if nums > 0 {
        loop {
            cri_level += 1;
            if cri_level > max_level {
                break;
            }
            for i in 0..u {
                if lv[i] != 0 {
                    continue;
                }
                // Rows are sorted on the first objective, so only earlier rows of this front can
                // dominate row i; uniqueness turns weak dominance into dominance.
                let dominated = (0..i).rev().any(|j| {
                    lv[j] == cri_level && (1..m).all(|k| obj[uni[j]][k] <= obj[uni[i]][k])
                });
                if !dominated {
                    lv[i] = cri_level;
                    sums += rep[i];
                }
            }
            if sums >= nums || sums >= n {
                break;
            }
        }
    }

    let levels = Array1::from_iter((0..n).map(|i| {
        let l = lv[inverse[i]];
        if l == 0 {
            f64::INFINITY
        } else {
            l as f64
        }
    }));
    Ok((levels.into_pyarray(py), cri_level))
}

/// Crowding distance (NSGA-II). Unranked individuals (level Inf) get 0; boundary points get Inf.
#[pyfunction]
#[pyo3(signature = (obj_v, levels, enhance_flag=None, parallel=None))]
pub fn crowdis<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
    enhance_flag: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let _ = (enhance_flag, parallel);
    let obj = crate::utils::to_f64_array2(obj_v)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let lev = if levels.is_none() {
        Array1::<f64>::ones(n)
    } else {
        crate::utils::to_f64_array1(levels)?
    };
    if lev.len() != n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crowdis: The length of levels must equal the number of rows of ObjV.",
        ));
    }
    let mut dist = Array1::<f64>::zeros(n);
    let mut unique_levels: Vec<f64> = lev.iter().cloned().filter(|v| v.is_finite()).collect();
    unique_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_levels.dedup();
    for l in unique_levels {
        let members: Vec<usize> = (0..n).filter(|&i| lev[i] == l).collect();
        let count = members.len();
        if count <= 2 {
            for &i in &members {
                dist[i] = f64::INFINITY;
            }
            continue;
        }
        for j in 0..m {
            let mut sorted = members.clone();
            sorted.sort_by(|&a, &b| {
                obj[[a, j]]
                    .partial_cmp(&obj[[b, j]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let (first, last) = (sorted[0], sorted[count - 1]);
            let span = obj[[last, j]] - obj[[first, j]] + 1e-6;
            for k in 1..count - 1 {
                let cur = sorted[k];
                dist[cur] += (obj[[sorted[k + 1], j]] - obj[[sorted[k - 1], j]]) / span;
            }
            dist[first] = f64::INFINITY;
            dist[last] = f64::INFINITY;
        }
    }
    Ok(dist.into_pyarray(py))
}

/// Pairwise distances between the rows of A and B: 'euclidean', 'cityblock',
/// 'cosine' (1 - cosine similarity) or 'cosine_similarity'.
#[pyfunction]
#[pyo3(signature = (a, b=None, metric=None, parallel=None))]
pub fn cdist<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: Option<&Bound<'py, PyAny>>,
    metric: Option<&str>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let arr_a = crate::utils::to_f64_array2(a)?;
    let arr_b = match b {
        Some(x) if !x.is_none() => crate::utils::to_f64_array2(x)?,
        _ => arr_a.clone(),
    };
    let (na, nb, dim) = (arr_a.shape()[0], arr_b.shape()[0], arr_a.shape()[1]);
    if arr_b.shape()[1] != dim {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in cdist: A and B must have the same number of columns.",
        ));
    }
    let metric = metric.unwrap_or("euclidean");
    let unit = |x: &Array2<f64>| -> Array2<f64> {
        let mut y = x.clone();
        for mut r in y.rows_mut() {
            let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            r.mapv_inplace(|v| v / norm);
        }
        y
    };
    let mut out = Array2::<f64>::zeros((na, nb));
    match metric {
        "euclidean" => {
            for i in 0..na {
                for j in 0..nb {
                    out[[i, j]] = (0..dim).map(|k| (arr_a[[i, k]] - arr_b[[j, k]]).powi(2)).sum::<f64>().sqrt();
                }
            }
        }
        "cityblock" => {
            for i in 0..na {
                for j in 0..nb {
                    out[[i, j]] = (0..dim).map(|k| (arr_a[[i, k]] - arr_b[[j, k]]).abs()).sum();
                }
            }
        }
        "cosine" | "cosine_similarity" => {
            let (ua, ub) = (unit(&arr_a), unit(&arr_b));
            let similarity = metric == "cosine_similarity";
            for i in 0..na {
                for j in 0..nb {
                    let dot: f64 = (0..dim).map(|k| ua[[i, k]] * ub[[j, k]]).sum();
                    out[[i, j]] = if similarity { dot } else { 1.0 - dot };
                }
            }
        }
        other => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "error in cdist: unsupported metric '{}' (支持 'euclidean', 'cityblock', 'cosine', 'cosine_similarity')。",
                other
            )))
        }
    }
    Ok(out.into_pyarray(py))
}

/// Merge the columns of CV into one violation column: the sum of the entries above the threshold
/// (0 for a scalar threshold, element-wise for an array threshold). The original 2.7.0 binary
/// accumulates this sum across rows by mistake; here every row is summed on its own, as documented.
#[pyfunction]
#[pyo3(signature = (cv, threshold=None, return_count=None, parallel=None))]
pub fn mergecv<'py>(
    py: Python<'py>,
    cv: &Bound<'py, PyAny>,
    threshold: Option<&Bound<'py, PyAny>>,
    return_count: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let arr = crate::utils::to_f64_array2(cv)?;
    let (n, c) = (arr.shape()[0], arr.shape()[1]);
    let (per_column, scalar) = match crate::utils::knob(threshold)? {
        Some(crate::utils::Knob::Array(v)) => {
            if v.len() != c {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in mergecv: The length of Threshold must equal the number of columns of CV.",
                ));
            }
            (Some(v), 0.0)
        }
        Some(crate::utils::Knob::Scalar(t)) => (None, t),
        _ => (None, 0.0),
    };
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = (0..c)
            .filter(|&j| match &per_column {
                Some(t) => arr[[i, j]] > t[j],
                None => arr[[i, j]] > 0.0,
            })
            .map(|j| arr[[i, j]])
            .sum();
    }
    let want_count = return_count
        .map(|o| o.is_truthy())
        .transpose()?
        .unwrap_or(false);
    if want_count {
        let limit = if per_column.is_some() { 0.0 } else { scalar };
        let count = out.iter().filter(|&&v| v <= limit).count();
        let tuple = (out.into_pyarray(py), count);
        return Ok(tuple.into_pyobject(py)?.into_any().unbind());
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// Pairs every row of ObjV with its weight vector; a single ObjV row is broadcast over all weights.
fn decomposition_rows(n_obj: usize, n_w: usize) -> PyResult<usize> {
    if n_obj == n_w || n_obj == 1 {
        Ok(n_w)
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error: The number of rows of weights must equal that of ObjV (or ObjV must have one row).",
        ))
    }
}

fn ideal_or_min(obj: &Array2<f64>, ideal: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
    let m = obj.shape()[1];
    match ideal {
        Some(z) if !z.is_none() => {
            let v = crate::utils::to_f64_array1(z)?;
            if v.len() != m {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: idealPoint must be a 1-D array with one entry per objective.",
                ));
            }
            Ok(v.to_vec())
        }
        _ => Ok((0..m)
            .map(|j| obj.column(j).iter().cloned().fold(f64::INFINITY, f64::min))
            .collect()),
    }
}

/// Tchebycheff aggregation (MOEA/D): max_j w_j |f_j - z_j|.
#[pyfunction]
#[pyo3(signature = (obj_v, weights, ideal_point=None, cv=None, maxormins=None, parallel=None))]
pub fn tcheby<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    ideal_point: Option<&Bound<'py, PyAny>>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins)?;
    let w = crate::utils::to_f64_array2(weights)?;
    let rows = decomposition_rows(obj.shape()[0], w.shape()[0])?;
    let z = ideal_or_min(&obj, ideal_point)?;
    let m = obj.shape()[1];
    let mut out = Array2::<f64>::zeros((rows, 1));
    for i in 0..rows {
        let r = if obj.shape()[0] == 1 { 0 } else { i };
        out[[i, 0]] = (0..m)
            .map(|j| w[[i, j]] * (obj[[r, j]] - z[j]).abs())
            .fold(f64::NEG_INFINITY, f64::max);
    }
    Ok(out.into_pyarray(py))
}

/// Penalty-based boundary intersection (MOEA/D): d1 + Theta * d2.
#[pyfunction]
#[pyo3(signature = (obj_v, weights, ideal_point=None, cv=None, maxormins=None, theta=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn pbi<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    ideal_point: Option<&Bound<'py, PyAny>>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    theta: Option<f64>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let theta = theta.unwrap_or(5.0);
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins)?;
    let w = crate::utils::to_f64_array2(weights)?;
    let rows = decomposition_rows(obj.shape()[0], w.shape()[0])?;
    let z = ideal_or_min(&obj, ideal_point)?;
    let m = obj.shape()[1];
    let mut out = Array2::<f64>::zeros((rows, 1));
    for i in 0..rows {
        let r = if obj.shape()[0] == 1 { 0 } else { i };
        let norm = (0..m).map(|j| w[[i, j]] * w[[i, j]]).sum::<f64>().sqrt();
        let d1 = (0..m)
            .map(|j| (obj[[r, j]] - z[j]) * w[[i, j]])
            .sum::<f64>()
            / norm;
        let d2 = (0..m)
            .map(|j| (obj[[r, j]] - z[j] - d1 * w[[i, j]] / norm).powi(2))
            .sum::<f64>()
            .sqrt();
        out[[i, 0]] = d1 + theta * d2;
    }
    Ok(out.into_pyarray(py))
}

/// Ideal point of ObjV, optionally merged with an older one. With CV the infeasible rows are first
/// penalised (column maximum + violation, as in every other core operator) and the extreme is then
/// taken over all rows, exactly like geatpy 2.7.0. `reverse` flips the convention to
/// "larger is better" (column maxima).
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, old_ideal_point=None, reverse=None))]
pub fn crtidp<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    old_ideal_point: Option<&Bound<'py, PyAny>>,
    reverse: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let reverse = reverse.map(|r| r.is_truthy()).transpose()?.unwrap_or(false);
    let rows: Vec<usize> = (0..n).collect();
    let pick = |a: f64, b: f64| if reverse { a.max(b) } else { a.min(b) };
    let start = if reverse {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut z: Vec<f64> = (0..m)
        .map(|j| rows.iter().map(|&i| obj[[i, j]]).fold(start, pick))
        .collect();
    if let Some(old) = old_ideal_point {
        if !old.is_none() {
            let o = crate::utils::to_f64_array1(old)?;
            for j in 0..m.min(o.len()) {
                z[j] = pick(z[j], o[j]);
            }
        }
    }
    Ok(Array1::from(z).into_pyarray(py))
}

/// Adaptive-weight aggregation (Gen & Cheng awGA), smaller is better.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None))]
pub fn awGA<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let zp: Vec<f64> = (0..m)
        .map(|j| obj.column(j).iter().cloned().fold(f64::INFINITY, f64::min))
        .collect();
    let zm: Vec<f64> = (0..m)
        .map(|j| {
            obj.column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let weight: Vec<f64> = (0..m)
        .map(|j| {
            (0..n)
                .map(|i| (obj[[i, j]] - zp[j]) / (zm[j] - zp[j] + 0.1))
                .sum()
        })
        .collect();
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = (0..m).map(|j| weight[j] * obj[[i, j]]).sum();
    }
    Ok(out.into_pyarray(py))
}

/// Random-weight aggregation: one random weight vector (summing to 1) shared by the population.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None))]
pub fn rwGA<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    let mut rng = rand::thread_rng();
    let r: Vec<f64> = (0..m).map(|_| rng.gen::<f64>()).collect();
    let total: f64 = r.iter().sum();
    let mut out = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        out[[i, 0]] = (0..m).map(|j| r[j] / total * obj[[i, j]]).sum();
    }
    Ok(out.into_pyarray(py))
}

fn cosine_matrix(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let norm = |x: &Array2<f64>| -> Vec<f64> {
        x.rows()
            .into_iter()
            .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect()
    };
    let (na, nb) = (norm(a), norm(b));
    let mut c = Array2::<f64>::zeros((a.shape()[0], b.shape()[0]));
    for i in 0..a.shape()[0] {
        for j in 0..b.shape()[0] {
            let dot: f64 = a
                .row(i)
                .iter()
                .zip(b.row(j).iter())
                .map(|(x, y)| x * y)
                .sum();
            c[[i, j]] = dot / (na[i] * nb[j]);
        }
    }
    c
}

/// Solve A x = rhs by Gaussian elimination with partial pivoting; None when A is singular.
fn solve_linear(mut a: Array2<f64>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let m = rhs.len();
    for col in 0..m {
        let piv =
            (col..m).max_by(|&x, &y| a[[x, col]].abs().partial_cmp(&a[[y, col]].abs()).unwrap())?;
        if a[[piv, col]] == 0.0 {
            return None;
        }
        if piv != col {
            for k in 0..m {
                a.swap([piv, k], [col, k]);
            }
            rhs.swap(piv, col);
        }
        for r in col + 1..m {
            let f = a[[r, col]] / a[[col, col]];
            for k in col..m {
                a[[r, k]] -= f * a[[col, k]];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0; m];
    for r in (0..m).rev() {
        let s: f64 = (r + 1..m).map(|k| a[[r, k]] * x[k]).sum();
        x[r] = (rhs[r] - s) / a[[r, r]];
    }
    Some(x)
}

/// NSGA-III environmental selection on the critical front (Deb & Jain). Individuals of the fronts
/// before criLevel are all kept; the remaining quota is filled niche by niche: the least crowded
/// reference point first, the closest individual for an empty niche, otherwise the first candidate.
#[pyfunction]
#[pyo3(signature = (obj_v, levels, cri_level, need_num, ref_point, maxormins=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn refselect<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
    cri_level: f64,
    need_num: usize,
    ref_point: &Bound<'py, PyAny>,
    maxormins: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let _ = parallel;
    let mut obj = crate::utils::to_f64_array2(obj_v)?;
    crate::utils::unify_objectives(&mut obj, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    if n <= need_num {
        return Ok(Array1::from_elem(n, true).into_pyarray(py));
    }
    let lev = crate::utils::to_f64_array1(levels)?;
    let z = crate::utils::to_f64_array2(ref_point)?;
    let nz = z.shape()[0];
    let idx1: Vec<usize> = (0..n).filter(|&i| lev[i] < cri_level).collect();
    let idx2: Vec<usize> = (0..n).filter(|&i| lev[i] == cri_level).collect();
    let (n1, n2) = (idx1.len(), idx2.len());
    let mut flag = Array1::from_elem(n, false);
    for &i in &idx1 {
        flag[i] = true;
    }
    let quota = need_num.saturating_sub(n1);
    if quota == 0 || n2 == 0 {
        return Ok(flag.into_pyarray(py));
    }
    // candidates translated by their ideal point
    let rows: Vec<usize> = idx1.iter().chain(idx2.iter()).cloned().collect();
    let mut c = Array2::<f64>::zeros((rows.len(), m));
    for (r, &i) in rows.iter().enumerate() {
        c.row_mut(r).assign(&obj.row(i));
    }
    for j in 0..m {
        let mn = c.column(j).iter().cloned().fold(f64::INFINITY, f64::min);
        c.column_mut(j).mapv_inplace(|v| v - mn);
    }
    // extreme points (achievement scalarising function with weights 1e-6 off the diagonal)
    let extreme: Vec<usize> = (0..m)
        .map(|i| {
            (0..rows.len())
                .map(|r| {
                    (0..m)
                        .map(|j| if j == i { c[[r, j]] } else { c[[r, j]] / 1e-6 })
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .enumerate()
                .fold(
                    (0, f64::INFINITY),
                    |best, (r, v)| if v < best.1 { (r, v) } else { best },
                )
                .0
        })
        .collect();
    let mut ext = Array2::<f64>::zeros((m, m));
    for (k, &r) in extreme.iter().enumerate() {
        ext.row_mut(k).assign(&c.row(r));
    }
    // Like the reference implementation, fall back to the column maxima only for a singular system.
    let intercept_scale: Vec<f64> = match solve_linear(ext, vec![1.0; m]) {
        Some(h) => h,
        None => (0..m)
            .map(|j| {
                let mx = c
                    .column(j)
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                1.0 / if mx <= 1e-6 { mx + 1e-6 } else { mx }
            })
            .collect(),
    };
    for (j, &scale) in intercept_scale.iter().enumerate() {
        c.column_mut(j).mapv_inplace(|v| v * scale);
    }
    // association with the reference points
    let cos = cosine_matrix(&c, &z);
    let mut pi = vec![0usize; rows.len()];
    let mut d = vec![0.0; rows.len()];
    for r in 0..rows.len() {
        let norm2: f64 = c.row(r).iter().map(|v| v * v).sum();
        let (best, dist) = (0..nz)
            .map(|k| (norm2 * (1.0 - cos[[r, k]] * cos[[r, k]])).max(0.0).sqrt())
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |b, (k, v)| if v < b.1 { (k, v) } else { b },
            );
        pi[r] = best;
        d[r] = dist;
    }
    let mut rho = vec![0usize; nz];
    for &p in &pi[..n1] {
        rho[p] += 1;
    }
    let mut excluded = vec![false; nz];
    let mut chosen = vec![false; n2];
    let mut picked = 0;
    while picked < quota {
        let Some(&min_rho) = (0..nz).filter(|&k| !excluded[k]).map(|k| &rho[k]).min() else {
            break;
        };
        let j = (0..nz)
            .find(|&k| !excluded[k] && rho[k] == min_rho)
            .unwrap();
        let cands: Vec<usize> = (0..n2).filter(|&t| pi[n1 + t] == j && !chosen[t]).collect();
        if cands.is_empty() {
            excluded[j] = true;
            continue;
        }
        let s = if rho[j] == 0 {
            *cands
                .iter()
                .min_by(|&&a, &&b| d[n1 + a].partial_cmp(&d[n1 + b]).unwrap())
                .unwrap()
        } else {
            cands[0]
        };
        chosen[s] = true;
        rho[j] += 1;
        picked += 1;
    }
    for (t, &i) in idx2.iter().enumerate() {
        if chosen[t] {
            flag[i] = true;
        }
    }
    Ok(flag.into_pyarray(py))
}

type FlagsAndGamma<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>);

/// RVEA reference-vector guided selection (angle-penalised distance). Returns the selection flags
/// and Gamma (smallest angle between each reference vector and the others), reusing Gamma if given.
#[pyfunction]
#[pyo3(signature = (obj_v, ref_point, p_theta, cv=None, gamma=None, maxormins=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn refgselect<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    ref_point: &Bound<'py, PyAny>,
    p_theta: f64,
    cv: Option<&Bound<'py, PyAny>>,
    gamma: Option<&Bound<'py, PyAny>>,
    maxormins: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<FlagsAndGamma<'py>> {
    let _ = parallel;
    let mut obj = crate::utils::to_f64_array2(obj_v)?;
    crate::utils::unify_objectives(&mut obj, maxormins)?;
    let (n, m) = (obj.shape()[0], obj.shape()[1]);
    for j in 0..m {
        let mn = obj.column(j).iter().cloned().fold(f64::INFINITY, f64::min);
        obj.column_mut(j).mapv_inplace(|v| v - mn);
    }
    let v = crate::utils::to_f64_array2(ref_point)?;
    let nv = v.shape()[0];
    let viol = crate::utils::violation(cv, n)?.unwrap_or_else(|| vec![0.0; n]);
    let gamma: Vec<f64> = match gamma {
        Some(g) if !g.is_none() => crate::utils::to_f64_array1(g)?.to_vec(),
        _ => {
            let mut cvv = cosine_matrix(&v, &v);
            for i in 0..nv {
                cvv[[i, i]] = 0.0;
            }
            (0..nv)
                .map(|i| {
                    cvv.row(i)
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .clamp(-1.0, 1.0)
                        .acos()
                })
                .collect()
        }
    };
    let cos = cosine_matrix(&obj, &v);
    let norm: Vec<f64> = obj
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    let assoc: Vec<usize> = (0..n)
        .map(|i| {
            (0..nv)
                .fold((0, f64::NEG_INFINITY), |b, k| {
                    if cos[[i, k]] > b.1 {
                        (k, cos[[i, k]])
                    } else {
                        b
                    }
                })
                .0
        })
        .collect();
    let mut flag = Array1::from_elem(n, false);
    let mut refs: Vec<usize> = assoc.clone();
    refs.sort();
    refs.dedup();
    for k in refs {
        let members: Vec<usize> = (0..n).filter(|&i| assoc[i] == k).collect();
        let feasible: Vec<usize> = members
            .iter()
            .cloned()
            .filter(|&i| viol[i] <= 1e-12)
            .collect();
        let best = if !feasible.is_empty() {
            let w = p_theta / gamma[k];
            let apd = |i: usize| (w * cos[[i, k]].clamp(-1.0, 1.0).acos() + 1.0) * norm[i];
            feasible
                .into_iter()
                .fold((usize::MAX, f64::INFINITY), |b, i| {
                    let a = apd(i);
                    if a < b.1 {
                        (i, a)
                    } else {
                        b
                    }
                })
                .0
        } else {
            members
                .into_iter()
                .fold((usize::MAX, f64::INFINITY), |b, i| {
                    if viol[i] < b.1 {
                        (i, viol[i])
                    } else {
                        b
                    }
                })
                .0
        };
        if best != usize::MAX {
            flag[best] = true;
        }
    }
    Ok((flag.into_pyarray(py), Array1::from(gamma).into_pyarray(py)))
}

/// Migration between sub-populations. Each population sends max(1, floor(MIGR * size)) individuals
/// (random, or the fittest with Select=1) to the population determined by Structure (0: all
/// populations shifted by one random offset, 1: a random neighbour, 2: ring). Replacement 0 moves
/// the emigrants out; 1 and 2 copy them and drop as many residents, randomly (1) or the least fit (2).
#[pyfunction]
#[pyo3(signature = (pop_sizes, migr=None, structure=None, select=None, replacement=None, fitn_vs=None, separate=None))]
#[allow(clippy::too_many_arguments)]
pub fn migrate<'py>(
    py: Python<'py>,
    pop_sizes: Vec<usize>,
    migr: Option<f64>,
    structure: Option<i64>,
    select: Option<i64>,
    replacement: Option<i64>,
    fitn_vs: Option<&Bound<'py, PyAny>>,
    separate: Option<bool>,
) -> PyResult<PyObject> {
    let p = pop_sizes.len();
    let migr = migr.unwrap_or(0.2);
    let (structure, select, replacement) = (
        structure.unwrap_or(0),
        select.unwrap_or(0),
        replacement.unwrap_or(0),
    );
    let fits: Option<Vec<Vec<f64>>> = match fitn_vs {
        Some(f) if !f.is_none() => Some(
            f.try_iter()?
                .map(|x| Ok(crate::utils::to_f64_array2(&x?)?.column(0).to_vec()))
                .collect::<PyResult<_>>()?,
        ),
        _ => None,
    };
    if (select == 1 || replacement == 2) && fits.is_none() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in migrate: FitnVs is required when Select=1 or Replacement=2. (此时必须传入FitnVs。)",
        ));
    }
    let mut rng = rand::thread_rng();
    let by_fitness_desc = |i: usize| -> Vec<usize> {
        let f = &fits.as_ref().unwrap()[i];
        let mut o: Vec<usize> = (0..pop_sizes[i]).collect();
        o.sort_by(|&a, &b| f[b].partial_cmp(&f[a]).unwrap_or(std::cmp::Ordering::Equal));
        o
    };
    let n_emig: Vec<usize> = pop_sizes
        .iter()
        .map(|&s| ((migr * s as f64).floor() as usize).max(1).min(s))
        .collect();
    // emigrants of every population
    let emigrants: Vec<Vec<usize>> = (0..p)
        .map(|i| {
            if select == 1 {
                by_fitness_desc(i).into_iter().take(n_emig[i]).collect()
            } else {
                let mut v: Vec<usize> = (0..pop_sizes[i]).collect();
                v.shuffle(&mut rng);
                v.truncate(n_emig[i]);
                v
            }
        })
        .collect();
    let shift = if p > 1 { rng.gen_range(1..p) } else { 0 };
    let from: Vec<usize> = (0..p)
        .map(|i| match structure {
            1 if p > 1 => {
                if rng.gen::<bool>() {
                    (i + p - 1) % p
                } else {
                    (i + 1) % p
                }
            }
            2 => (i + p - 1) % p,
            _ => (i + p - shift) % p,
        })
        .collect();
    let mut aborigines: Vec<Vec<usize>> = Vec::with_capacity(p);
    for i in 0..p {
        let keep = match replacement {
            0 => (0..pop_sizes[i])
                .filter(|x| !emigrants[i].contains(x))
                .collect(),
            2 => by_fitness_desc(i)
                .into_iter()
                .take(pop_sizes[i].saturating_sub(n_emig[from[i]]))
                .collect(),
            _ => {
                let mut v: Vec<usize> = (0..pop_sizes[i]).collect();
                v.shuffle(&mut rng);
                v.truncate(pop_sizes[i].saturating_sub(n_emig[from[i]]));
                v
            }
        };
        aborigines.push(keep);
    }
    let foreigners: Vec<Vec<usize>> = (0..p).map(|i| emigrants[from[i]].clone()).collect();
    // Python lists of indices, like geatpy 2.7.0.
    let to_arr = |v: Vec<usize>| pyo3::types::PyList::new(py, v.into_iter().map(|x| x as i64));
    if separate.unwrap_or(true) {
        let ab = pyo3::types::PyList::empty(py);
        let fo = pyo3::types::PyList::empty(py);
        for i in 0..p {
            ab.append(to_arr(aborigines[i].clone())?)?;
            fo.append(to_arr(foreigners[i].clone())?)?;
        }
        let fp = Array1::from_iter(from.iter().map(|&x| x as i32)).into_pyarray(py);
        return Ok((ab, fo, fp).into_pyobject(py)?.into_any().unbind());
    }
    let offsets: Vec<usize> = pop_sizes
        .iter()
        .scan(0, |acc, &s| {
            let o = *acc;
            *acc += s;
            Some(o)
        })
        .collect();
    let out = pyo3::types::PyList::empty(py);
    for i in 0..p {
        let mut v: Vec<usize> = aborigines[i].iter().map(|&x| x + offsets[i]).collect();
        v.extend(foreigners[i].iter().map(|&x| x + offsets[from[i]]));
        out.append(to_arr(v)?)?;
    }
    Ok(out.into_any().unbind())
}
