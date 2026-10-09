use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rand::Rng;

/// Polynomial Mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field, pm=None, dis_i=20.0, fix_type=1, parallel=false))]
pub fn mutpolyn<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    pm: Option<f64>,
    dis_i: f64,
    fix_type: i32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, fix_type, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let f_arr = crate::utils::to_f64_array2(field)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let lb = f_arr.row(0);
    let ub = f_arr.row(1);
    let var_types = f_arr.row(2);

    let prob = pm.unwrap_or(1.0 / (d as f64).max(1.0)).clamp(0.0, 1.0);
    let eta = dis_i.max(0.0);
    let eta_plus_1 = eta + 1.0;
    let inv_eta = 1.0 / eta_plus_1;

    let mut rng = rand::thread_rng();

    for i in 0..n_ind {
        for j in 0..d {
            if rng.gen_bool(prob) {
                let y = chrom[[i, j]];
                let yl = lb[j];
                let yu = ub[j];
                let span = yu - yl;

                if span > 1e-12 {
                    let delta1 = ((y - yl) / span).clamp(0.0, 1.0);
                    let delta2 = ((yu - y) / span).clamp(0.0, 1.0);
                    let u: f64 = rng.gen();

                    let delta_q = if u <= 0.5 {
                        let xy = 1.0 - delta1;
                        let val = 2.0 * u + (1.0 - 2.0 * u) * xy.powf(eta_plus_1);
                        val.max(0.0).powf(inv_eta) - 1.0
                    } else {
                        let xy = 1.0 - delta2;
                        let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy.powf(eta_plus_1);
                        1.0 - val.max(0.0).powf(inv_eta)
                    };

                    let mut mutated = (y + delta_q * span).clamp(yl, yu);
                    if var_types[j] == 1.0 {
                        mutated = mutated.round();
                    }
                    chrom[[i, j]] = mutated;
                }
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Gaussian Mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field, pm=None, sigma=None, middle=None, fix_type=1, parallel=false))]
pub fn mutgau<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    pm: Option<f64>,
    sigma: Option<f64>,
    middle: Option<f64>,
    fix_type: i32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, middle, fix_type, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let f_arr = crate::utils::to_f64_array2(field)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let lb = f_arr.row(0);
    let ub = f_arr.row(1);
    let var_types = f_arr.row(2);

    let prob = pm.unwrap_or(1.0 / (d as f64).max(1.0)).clamp(0.0, 1.0);
    let sig = sigma.unwrap_or(0.1);
    let mut rng = rand::thread_rng();

    for i in 0..n_ind {
        for j in 0..d {
            if rng.gen_bool(prob) {
                let y = chrom[[i, j]];
                let span = ub[j] - lb[j];
                let std = sig * span;
                let norm: f64 = rng.sample(rand_distr::StandardNormal);
                let mut mutated = (y + norm * std).clamp(lb[j], ub[j]);
                if var_types[j] == 1.0 {
                    mutated = mutated.round();
                }
                chrom[[i, j]] = mutated;
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Breeder Genetic Algorithm Mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field, pm=None, mut_shrink=0.5, gradient=20, fix_type=1, parallel=false))]
pub fn mutbga<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    pm: Option<f64>,
    mut_shrink: f64,
    gradient: usize,
    fix_type: i32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, gradient, fix_type, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let f_arr = crate::utils::to_f64_array2(field)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let lb = f_arr.row(0);
    let ub = f_arr.row(1);
    let var_types = f_arr.row(2);

    let prob = pm.unwrap_or(1.0 / (d as f64).max(1.0)).clamp(0.0, 1.0);
    let mut rng = rand::thread_rng();

    for i in 0..n_ind {
        for j in 0..d {
            if rng.gen_bool(prob) {
                let y = chrom[[i, j]];
                let span = ub[j] - lb[j];
                let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                let mut sum_bits = 0.0;
                for k in 0..16 {
                    if rng.gen_bool(1.0 / 16.0) {
                        sum_bits += 2.0_f64.powi(-(k as i32));
                    }
                }
                let delta = mut_shrink * span * sum_bits;
                let mut mutated = (y + sign * delta).clamp(lb[j], ub[j]);
                if var_types[j] == 1.0 {
                    mutated = mutated.round();
                }
                chrom[[i, j]] = mutated;
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Binary bit-flip mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, pm=None, parallel=false))]
pub fn mutbin<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    pm: Option<f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let prob = pm.unwrap_or(1.0 / (d as f64).max(1.0)).clamp(0.0, 1.0);
    let mut rng = rand::thread_rng();

    for i in 0..n_ind {
        for j in 0..d {
            if rng.gen_bool(prob) {
                chrom[[i, j]] = if chrom[[i, j]] > 0.5 { 0.0 } else { 1.0 };
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Differential Evolution Mutation
fn extract_indices(elem: &Bound<'_, PyAny>, n_ind: usize) -> Option<Vec<usize>> {
    if elem.is_none() {
        return None;
    }
    if let Ok(arr) = elem.extract::<Vec<usize>>() {
        return Some(arr);
    }
    if let Ok(arr) = elem.extract::<Vec<i64>>() {
        return Some(arr.into_iter().map(|x| (x.rem_euclid(n_ind as i64)) as usize).collect());
    }
    if let Ok(arr) = elem.extract::<Vec<f64>>() {
        return Some(arr.into_iter().map(|x| ((x as i64).rem_euclid(n_ind as i64)) as usize).collect());
    }
    if let Ok(ro) = elem.extract::<numpy::PyReadonlyArray1<i64>>() {
        if let Ok(s) = ro.as_slice() {
            return Some(s.iter().map(|&x| (x.rem_euclid(n_ind as i64)) as usize).collect());
        }
    }
    if let Ok(ro) = elem.extract::<numpy::PyReadonlyArray1<f64>>() {
        if let Ok(s) = ro.as_slice() {
            return Some(s.iter().map(|&x| ((x as i64).rem_euclid(n_ind as i64)) as usize).collect());
        }
    }
    if let Ok(ro) = elem.extract::<numpy::PyReadonlyArray2<i64>>() {
        let arr = ro.as_array();
        return Some(arr.iter().map(|&x| (x.rem_euclid(n_ind as i64)) as usize).collect());
    }
    if let Ok(ro) = elem.extract::<numpy::PyReadonlyArray2<f64>>() {
        let arr = ro.as_array();
        return Some(arr.iter().map(|&x| ((x as i64).rem_euclid(n_ind as i64)) as usize).collect());
    }
    None
}

/// Differential Evolution Mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field, xr_list=None, f=0.5, fix_type=1, parallel=false))]
pub fn mutde<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    xr_list: Option<&Bound<'py, PyAny>>,
    f: f64,
    fix_type: i32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let f_arr = crate::utils::to_f64_array2(field)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let lb = f_arr.row(0);
    let ub = f_arr.row(1);
    let var_types = f_arr.row(2);

    let mut new_chrom = chrom.clone();
    let mut rng = rand::thread_rng();

    if n_ind < 2 {
        return Ok(new_chrom.into_pyarray(py));
    }

    let mut parsed_xr: Vec<Option<Vec<usize>>> = Vec::new();
    if let Some(xr_any) = xr_list {
        if let Ok(seq) = xr_any.downcast::<pyo3::types::PySequence>() {
            let len = seq.len().unwrap_or(0);
            for idx in 0..len {
                if let Ok(item) = seq.get_item(idx) {
                    parsed_xr.push(extract_indices(&item, n_ind));
                } else {
                    parsed_xr.push(None);
                }
            }
        }
    }

    for i in 0..n_ind {
        let base_idx = if !parsed_xr.is_empty() && parsed_xr[0].as_ref().and_then(|v| v.get(i)).is_some() {
            parsed_xr[0].as_ref().unwrap()[i]
        } else {
            let mut r = rng.gen_range(0..n_ind);
            while r == i && n_ind > 1 {
                r = rng.gen_range(0..n_ind);
            }
            r
        };

        let r1 = if parsed_xr.len() > 1 && parsed_xr[1].as_ref().and_then(|v| v.get(i)).is_some() {
            parsed_xr[1].as_ref().unwrap()[i]
        } else {
            let mut r = rng.gen_range(0..n_ind);
            while (r == i || r == base_idx) && n_ind > 2 {
                r = rng.gen_range(0..n_ind);
            }
            r
        };

        let r2 = if parsed_xr.len() > 2 && parsed_xr[2].as_ref().and_then(|v| v.get(i)).is_some() {
            parsed_xr[2].as_ref().unwrap()[i]
        } else {
            let mut r = rng.gen_range(0..n_ind);
            while (r == i || r == base_idx || r == r1) && n_ind > 3 {
                r = rng.gen_range(0..n_ind);
            }
            r
        };

        let has_pair2 = parsed_xr.len() >= 5;
        let r3 = if has_pair2 && parsed_xr[3].as_ref().and_then(|v| v.get(i)).is_some() {
            Some(parsed_xr[3].as_ref().unwrap()[i])
        } else {
            None
        };
        let r4 = if has_pair2 && parsed_xr[4].as_ref().and_then(|v| v.get(i)).is_some() {
            Some(parsed_xr[4].as_ref().unwrap()[i])
        } else {
            None
        };

        for j in 0..d {
            let mut v = chrom[[base_idx, j]] + f * (chrom[[r1, j]] - chrom[[r2, j]]);
            if let (Some(idx3), Some(idx4)) = (r3, r4) {
                v += f * (chrom[[idx3, j]] - chrom[[idx4, j]]);
            }
            let mut val = match fix_type {
                2 => {
                    if v < lb[j] || v > ub[j] {
                        rng.gen_range(lb[j]..=ub[j])
                    } else {
                        v
                    }
                }
                3 => {
                    let mut x = v;
                    if x < lb[j] {
                        x = 2.0 * lb[j] - x;
                    }
                    if x > ub[j] {
                        x = 2.0 * ub[j] - x;
                    }
                    x.clamp(lb[j], ub[j])
                }
                4 => {
                    let span = ub[j] - lb[j];
                    if span > 1e-12 {
                        lb[j] + ((v - lb[j]) % span + span) % span
                    } else {
                        lb[j]
                    }
                }
                _ => v.clamp(lb[j], ub[j]),
            };
            if var_types[j] == 1.0 {
                val = val.round();
            }
            new_chrom[[i, j]] = val;
        }
    }

    Ok(new_chrom.into_pyarray(py))
}

/// Inversion mutation for permutations
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field=None, pm=None, invert_len=None, parallel=false))]
pub fn mutinv<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: Option<&Bound<'py, PyAny>>,
    pm: Option<f64>,
    invert_len: Option<usize>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, field, invert_len, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let prob = pm.unwrap_or(0.1).clamp(0.0, 1.0);
    let mut rng = rand::thread_rng();

    if d < 2 {
        return Ok(chrom.into_pyarray(py));
    }

    for i in 0..n_ind {
        if rng.gen_bool(prob) {
            let mut p1 = rng.gen_range(0..d);
            let mut p2 = rng.gen_range(0..d);
            if p1 > p2 {
                std::mem::swap(&mut p1, &mut p2);
            }
            let mut l = p1;
            let mut r = p2;
            while l < r {
                let tmp = chrom[[i, l]];
                chrom[[i, l]] = chrom[[i, r]];
                chrom[[i, r]] = tmp;
                l += 1;
                r -= 1;
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Move / insertion mutation for permutations
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field=None, pm=None, move_len=None, pr=None, parallel=false))]
pub fn mutmove<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: Option<&Bound<'py, PyAny>>,
    pm: Option<f64>,
    move_len: Option<usize>,
    pr: Option<f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, field, move_len, pr, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let prob = pm.unwrap_or(0.1).clamp(0.0, 1.0);
    let mut rng = rand::thread_rng();

    if d < 2 {
        return Ok(chrom.into_pyarray(py));
    }

    for i in 0..n_ind {
        if rng.gen_bool(prob) {
            let src = rng.gen_range(0..d);
            let dst = rng.gen_range(0..d);
            if src != dst {
                let mut row: Vec<f64> = (0..d).map(|j| chrom[[i, j]]).collect();
                let val = row.remove(src);
                row.insert(dst, val);
                for j in 0..d {
                    chrom[[i, j]] = row[j];
                }
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Swap mutation for permutations
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field=None, pm=None, parallel=false))]
pub fn mutswap<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: Option<&Bound<'py, PyAny>>,
    pm: Option<f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, field, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let prob = pm.unwrap_or(0.1).clamp(0.0, 1.0);
    let mut rng = rand::thread_rng();

    if d < 2 {
        return Ok(chrom.into_pyarray(py));
    }

    for i in 0..n_ind {
        if rng.gen_bool(prob) {
            let p1 = rng.gen_range(0..d);
            let mut p2 = rng.gen_range(0..d);
            while p2 == p1 {
                p2 = rng.gen_range(0..d);
            }
            let tmp = chrom[[i, p1]];
            chrom[[i, p1]] = chrom[[i, p2]];
            chrom[[i, p2]] = tmp;
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Uniform Mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field, pm=None, alpha=None, middle=None, fix_type=1, parallel=false))]
pub fn mutuni<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    pm: Option<f64>,
    alpha: Option<f64>,
    middle: Option<f64>,
    fix_type: i32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (encoding, alpha, middle, fix_type, parallel);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let f_arr = crate::utils::to_f64_array2(field)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    let lb = f_arr.row(0);
    let ub = f_arr.row(1);
    let var_types = f_arr.row(2);

    let prob = pm.unwrap_or(1.0 / (d as f64).max(1.0)).clamp(0.0, 1.0);
    let mut rng = rand::thread_rng();

    for i in 0..n_ind {
        for j in 0..d {
            if rng.gen_bool(prob) {
                let l = lb[j];
                let u = ub[j];
                let is_discrete = var_types[j] == 1.0;
                let val = if is_discrete {
                    let low = l.round() as i64;
                    let high = u.round() as i64;
                    if low <= high { rng.gen_range(low..=high) as f64 } else { low as f64 }
                } else {
                    if l < u { rng.gen_range(l..=u) } else { l }
                };
                chrom[[i, j]] = val;
            }
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Permutation Mutation
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field=None, pm=None, mut_n=None, parallel=false))]
pub fn mutpp<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: Option<&Bound<'py, PyAny>>,
    pm: Option<f64>,
    mut_n: Option<usize>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = mut_n;
    mutswap(py, encoding, old_chrom, field, pm, parallel)
}

/// High-level Mutation Dispatcher
#[pyfunction]
#[pyo3(signature = (mut_oper, encoding, old_chrom, field=None, **kwargs))]
pub fn mutate<'py>(
    py: Python<'py>,
    mut_oper: &str,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut pm = None;
    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("Pm")? {
            pm = Some(val.extract::<f64>()?);
        }
    }

    match mut_oper.to_lowercase().as_str() {
        "mutpolyn" => {
            let f = field.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Field required for mutpolyn"))?;
            mutpolyn(py, encoding, old_chrom, f, pm, 20.0, 1, false)
        }
        "mutgau" => {
            let f = field.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Field required for mutgau"))?;
            mutgau(py, encoding, old_chrom, f, pm, None, None, 1, false)
        }
        "mutbga" => {
            let f = field.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Field required for mutbga"))?;
            mutbga(py, encoding, old_chrom, f, pm, 0.5, 20, 1, false)
        }
        "mutbin" => mutbin(py, encoding, old_chrom, pm, false),
        "mutde" => {
            let f = field.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Field required for mutde"))?;
            mutde(py, encoding, old_chrom, f, None, 0.5, 1, false)
        }
        "mutinv" => mutinv(py, encoding, old_chrom, None, pm, None, false),
        "mutmove" => mutmove(py, encoding, old_chrom, None, pm, None, None, false),
        "mutswap" => mutswap(py, encoding, old_chrom, None, pm, false),
        "mutuni" => {
            let f = field.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Field required for mutuni"))?;
            mutuni(py, encoding, old_chrom, f, pm, None, None, 1, false)
        }
        "mutpp" => mutpp(py, encoding, old_chrom, None, pm, None, false),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported mutation operator: {}",
            mut_oper
        ))),
    }
}
