use numpy::ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::Rng;

use crate::utils::{bounds, chrom_out, fix_value, is_integer_like, knob, opt_int, per_gene, Knob};

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

fn require_ri(encoding: &str, name: &str) -> PyResult<()> {
    if encoding != "RI" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: The encoding must be 'RI'. (编码方式必须为'RI'。)",
            name
        )));
    }
    Ok(())
}

fn default_pm(pm: Opt, d: usize) -> PyResult<Vec<f64>> {
    per_gene(pm, d, 1.0 / (d.max(1) as f64), "Pm")
}

/// Polynomial mutation (Deb). Inputs are repaired first; the mutated value stays in range by construction.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, dis_i=None, fix_type=None, parallel=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutpolyn<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    dis_i: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params7);
    require_ri(encoding, "mutpolyn")?;
    // A 1-D OldChrom is a single chromosome (the original has a dedicated 1-D path).
    let one_d = old_chrom
        .getattr("ndim")
        .and_then(|v| v.extract::<usize>())
        .map(|nd| nd == 1)
        .unwrap_or(false);
    let mut chrom = if one_d {
        let v = crate::utils::to_f64_array1(old_chrom)?;
        let l = v.len();
        v.into_shape_with_order((1, l)).unwrap()
    } else {
        crate::utils::to_f64_array2(old_chrom)?
    };
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d)?;
    let eta = per_gene(dis_i, d, 20.0, "DisI")?;
    let fix = crate::utils::fix_type(fix_type, "mutpolyn")?;
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in 0..d {
            let mut x = if b.span[j] > 1e-15 {
                fix_value(chrom[[i, j]], b.lb[j], b.ub[j], b.span[j], fix, &mut rng)
            } else {
                b.lb[j]
            };
            if b.span[j] > 1e-15 && rng.gen::<f64>() < pm[j] {
                let e1 = eta[j] + 1.0;
                let u: f64 = rng.gen();
                if u > 0.5 {
                    let dv = 2.0 * u - 1.0;
                    let t = (1.0 - (b.ub[j] - x) / b.span[j]).powf(e1);
                    x += (1.0 - ((1.0 - dv) + dv * t).powf(1.0 / e1)) * b.span[j];
                } else {
                    let t = (1.0 - (x - b.lb[j]) / b.span[j]).powf(e1);
                    x += (((1.0 - 2.0 * u) * t + 2.0 * u).powf(1.0 / e1) - 1.0) * b.span[j];
                }
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    if one_d {
        let out = crate::utils::ri_out(py, chrom, &b.discrete)?;
        return Ok(out.bind(py).call_method0("ravel")?.unbind());
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    rng.sample(rand_distr::StandardNormal)
}

/// Gaussian mutation. Sigma3 is three standard deviations: scalar, per-gene array,
/// True (min distance to a bound) or False/None (0.5 * (ub - lb)).
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, sigma3=None, middle=None, fix_type=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutgau<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    sigma3: Opt<'_, 'py>,
    middle: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutgau")?;
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d)?;
    let fix = crate::utils::fix_type(fix_type, "mutgau")?;
    let middle = matches!(knob(middle)?, Some(Knob::Bool(true)));
    // None => per-gene sigma derived from bounds; Some(true) => adaptive; per-gene fixed sigma otherwise.
    let (adaptive, sigma): (bool, Option<Vec<f64>>) = match knob(sigma3)? {
        None | Some(Knob::Bool(false)) => (false, None),
        Some(Knob::Bool(true)) => (true, None),
        Some(Knob::Scalar(s)) => (
            false,
            Some(
                (0..d)
                    .map(|j| {
                        if b.discrete[j] {
                            (s + 0.499999) / 3.0
                        } else {
                            s / 3.0
                        }
                    })
                    .collect(),
            ),
        ),
        Some(Knob::Array(v)) => {
            if v.len() != d {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in mutgau: The length of Sigma3 must equal the chromosome length.",
                ));
            }
            (
                false,
                Some(
                    (0..d)
                        .map(|j| {
                            if b.discrete[j] {
                                (v[j] + 0.499999) / 3.0
                            } else {
                                v[j] / 3.0
                            }
                        })
                        .collect(),
                ),
            )
        }
    };
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in 0..d {
            let mut x;
            if b.span[j] <= 1e-15 {
                x = b.lb[j];
            } else {
                x = chrom[[i, j]];
                if rng.gen::<f64>() < pm[j] {
                    let s = if adaptive {
                        (b.ub[j] - x).abs().min((x - b.lb[j]).abs()) / 3.0
                    } else {
                        match &sigma {
                            Some(v) => v[j],
                            None => (b.ub[j] - b.lb[j]) / 6.0,
                        }
                    };
                    if middle {
                        x = (b.lb[j] + b.ub[j]) * 0.5;
                    }
                    if s > 1e-15 {
                        x += s * standard_normal(&mut rng);
                    }
                }
                x = fix_value(x, b.lb[j], b.ub[j], b.span[j], fix, &mut rng);
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Breeder GA mutation (Mühlenbein & Schlierkamp-Voosen).
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, mut_shrink=None, gradient=None, fix_type=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutbga<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    mut_shrink: Opt<'_, 'py>,
    gradient: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutbga")?;
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d)?;
    let shrink = per_gene(mut_shrink, d, 0.5, "MutShrink")?;
    let grad: Vec<i64> = per_gene(gradient, d, 20.0, "Gradient")?
        .iter()
        .map(|&g| g as i64)
        .collect();
    let fix = crate::utils::fix_type(fix_type, "mutbga")?;
    // The mutation range doubles as the repair span, as in the reference implementation.
    let range: Vec<f64> = (0..d).map(|j| b.span[j] * shrink[j]).collect();
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in 0..d {
            let mut x;
            if range[j] <= 1e-15 {
                x = b.lb[j];
            } else {
                x = chrom[[i, j]];
                if rng.gen::<f64>() < pm[j] {
                    let g = grad[j].max(1);
                    let p = 1.0 / g as f64;
                    let mut delta: f64 = (0..g)
                        .filter(|_| rng.gen::<f64>() < p)
                        .map(|k| 0.5f64.powi(k as i32))
                        .sum();
                    delta = delta.max(0.5f64.powi((g - 1) as i32));
                    if rng.gen::<f64>() < 0.5 {
                        delta = -delta;
                    }
                    x += delta * range[j];
                }
                x = fix_value(x, b.lb[j], b.ub[j], range[j], fix, &mut rng);
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Uniform mutation within a radius Alpha around the current value (or the domain centre).
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, pm=None, alpha=None, middle=None, fix_type=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutuni<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    pm: Opt<'_, 'py>,
    alpha: Opt<'_, 'py>,
    middle: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutuni")?;
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let pm = default_pm(pm, d)?;
    let fix = crate::utils::fix_type(fix_type, "mutuni")?;
    let middle = matches!(knob(middle)?, Some(Knob::Bool(true)));
    let (adaptive, radius): (bool, Option<Vec<f64>>) = match knob(alpha)? {
        None | Some(Knob::Bool(false)) => (false, None),
        Some(Knob::Bool(true)) => (true, None),
        Some(Knob::Scalar(a)) => (
            false,
            Some(
                (0..d)
                    .map(|j| if b.discrete[j] { a + 0.5 } else { a })
                    .collect(),
            ),
        ),
        Some(Knob::Array(v)) => {
            if v.len() != d {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in mutuni: The length of Alpha must equal the chromosome length.",
                ));
            }
            (
                false,
                Some(
                    (0..d)
                        .map(|j| if b.discrete[j] { v[j] + 0.5 } else { v[j] })
                        .collect(),
                ),
            )
        }
    };
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in 0..d {
            let mut x;
            if b.span[j] <= 1e-15 {
                x = b.lb[j];
            } else {
                x = chrom[[i, j]];
                if rng.gen::<f64>() < pm[j] {
                    let r = if adaptive {
                        (b.ub[j] - x).abs().min((x - b.lb[j]).abs())
                    } else {
                        match &radius {
                            Some(v) => v[j],
                            None => (b.ub[j] - b.lb[j]) * 0.5,
                        }
                    };
                    if middle {
                        x = (b.lb[j] + b.ub[j]) * 0.5;
                    }
                    x = x - r + 2.0 * r * rng.gen::<f64>();
                }
                x = fix_value(x, b.lb[j], b.ub[j], b.span[j], fix, &mut rng);
            }
            chrom[[i, j]] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Bit-flip mutation for binary chromosomes.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, params2=None, pm=None, parallel=None, params5=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutbin<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    params2: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params5: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (params2, parallel, params5, params6, params7);
    if encoding != "BG" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in mutbin: The encoding must be 'BG'. (编码方式必须为'BG'。)",
        ));
    }
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let pm = default_pm(pm, d)?;
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in 0..d {
            if rng.gen::<f64>() < pm[j] {
                chrom[[i, j]] = 1.0 - chrom[[i, j]];
            }
        }
    }
    chrom_out(py, chrom, true)
}

/// Draw without replacement from `pool`; falls back to any index when the pool is exhausted.
fn draw<R: Rng>(pool: &mut Vec<usize>, n: usize, rng: &mut R) -> usize {
    if pool.is_empty() {
        return rng.gen_range(0..n);
    }
    let k = rng.gen_range(0..pool.len());
    pool.swap_remove(k)
}

enum Xr {
    Random,
    Index(Vec<usize>),
    Vector(Array2<f64>),
}

/// Differential mutation: Xr0 + F1 (Xr1 - Xr2) [+ F2 (Xr3 - Xr4)].
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, xr_list=None, f=None, fix_type=None, mask_n=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutde<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    xr_list: Opt<'_, 'py>,
    f: Opt<'_, 'py>,
    fix_type: Opt<'_, 'py>,
    mask_n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    require_ri(encoding, "mutde")?;
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, d) = (chrom.shape()[0], chrom.shape()[1]);
    let b = bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let fix = crate::utils::fix_type(fix_type, "mutde")?;

    // Output rows, and an optional element mask.
    let mut rows = n;
    let mut mask: Option<Array2<bool>> = None;
    if let Some(m) = mask_n {
        if !m.is_none() {
            if let Ok(k) = m.extract::<usize>() {
                rows = k;
            } else {
                let arr = crate::utils::to_f64_array2(m)?;
                if arr.shape()[1] != d {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "error in mutde: The number of columns of Mask_N must equal that of OldChrom.",
                    ));
                }
                rows = arr.shape()[0];
                mask = Some(arr.mapv(|x| x != 0.0));
            }
        }
    }

    let mut xrs: Vec<Xr> = Vec::new();
    if let Some(list) = xr_list {
        if !list.is_none() {
            for item in list.try_iter()? {
                let item = item?;
                if item.is_none() {
                    xrs.push(Xr::Random);
                    continue;
                }
                let ndim: usize = item.getattr("ndim").and_then(|v| v.extract()).unwrap_or(1);
                if ndim >= 2 {
                    let v = crate::utils::to_f64_array2(&item)?;
                    if v.shape()[0] < rows || v.shape()[1] != d {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "error in mutde: The shape of a vector in XrList does not match.",
                        ));
                    }
                    xrs.push(Xr::Vector(v));
                } else {
                    let v = crate::utils::to_f64_array1(&item)?;
                    let idx: Vec<usize> = v
                        .iter()
                        .map(|&x| x as i64)
                        .map(|x| {
                            if x < 0 || x as usize >= n {
                                usize::MAX
                            } else {
                                x as usize
                            }
                        })
                        .collect();
                    if idx.len() < rows || idx.iter().take(rows).any(|&x| x == usize::MAX) {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "error in mutde: The index in XrList is out of range. (XrList中的索引越界。)",
                        ));
                    }
                    xrs.push(Xr::Index(idx));
                }
            }
        }
    }
    let pairs = if xrs.len() > 3 { 2 } else { 1 };
    while xrs.len() < 1 + 2 * pairs {
        xrs.push(Xr::Random);
    }

    // F: scalar, [F1, F2] with None meaning "random in (0, 1) per individual", or None.
    let parse_f = |o: &Bound<'py, PyAny>| -> PyResult<f64> {
        if o.is_none() {
            Ok(-1.0)
        } else {
            o.extract::<f64>()
        }
    };
    let (f1, f2) = match f {
        None => (-1.0, -1.0),
        Some(o) if o.is_none() => (-1.0, -1.0),
        Some(o) => {
            if let Ok(v) = o.extract::<f64>() {
                (v, v)
            } else {
                let items: Vec<Bound<'py, PyAny>> = o.try_iter()?.collect::<PyResult<_>>()?;
                let a = items.first().map(&parse_f).transpose()?.unwrap_or(-1.0);
                let c = items.get(1).map(parse_f).transpose()?.unwrap_or(a);
                (a, c)
            }
        }
    };

    let mut rng = rand::thread_rng();
    // Trial values of row i on `cols`, with freshly drawn donors and F (the target row is excluded
    // from the donors when one offspring is produced per individual).
    let trial = |i: usize, cols: &[usize], rng: &mut rand::rngs::ThreadRng| -> Vec<f64> {
        let mut pool: Vec<usize> = (0..n).filter(|&k| !(rows == n && k == i)).collect();
        let vecs: Vec<Vec<f64>> = xrs
            .iter()
            .take(1 + 2 * pairs)
            .map(|x| match x {
                Xr::Random => chrom.row(draw(&mut pool, n, rng)).to_vec(),
                Xr::Index(idx) => chrom.row(idx[i]).to_vec(),
                Xr::Vector(v) => v.row(i).to_vec(),
            })
            .collect();
        let fa = if f1 < 0.0 { rng.gen::<f64>() } else { f1 };
        let fb = if f2 < 0.0 { rng.gen::<f64>() } else { f2 };
        cols.iter()
            .map(|&j| {
                let x = if b.span[j] > 1e-15 {
                    let mut v = vecs[0][j] + fa * (vecs[1][j] - vecs[2][j]);
                    if pairs == 2 {
                        v += fb * (vecs[3][j] - vecs[4][j]);
                    }
                    fix_value(v, b.lb[j], b.ub[j], b.span[j], fix, rng)
                } else {
                    b.lb[j]
                };
                if b.discrete[j] {
                    x.round()
                } else {
                    x
                }
            })
            .collect()
    };
    if let Some(m) = mask {
        // Mask_N matrix: like geatpy 2.7.0, every selected element gets its own donors and F.
        let vals: Vec<f64> = m
            .indexed_iter()
            .filter(|(_, &on)| on)
            .map(|((i, j), _)| trial(i, &[j], &mut rng)[0])
            .collect();
        return Ok(Array1::from(vals).into_pyarray(py).into_any().unbind());
    }
    let all_cols: Vec<usize> = (0..d).collect();
    let mut out = Array2::<f64>::zeros((rows, d));
    for i in 0..rows {
        let row = trial(i, &all_cols, &mut rng);
        out.row_mut(i).assign(&Array1::from(row));
    }
    crate::utils::ri_out(py, out, &b.discrete)
}

/// Segment length in [lo, max_len] such that every (start, length) pair is equally likely.
fn segment_len<R: Rng>(l: usize, max_len: usize, lo: usize, rng: &mut R) -> usize {
    let max_len = max_len.clamp(lo, l.max(lo));
    let weights: Vec<usize> = (lo..=max_len).map(|k| l + 1 - k.min(l)).collect();
    let total: usize = weights.iter().sum();
    if total == 0 {
        return lo;
    }
    let mut r = rng.gen_range(0..total);
    for (k, w) in (lo..=max_len).zip(weights) {
        if r < w {
            return k;
        }
        r -= w;
    }
    max_len
}

/// Parses InvertLen / MoveLen: int => maximum length, [k] => fixed length, None => default.
fn parse_len(obj: Opt, default: usize) -> PyResult<(usize, bool)> {
    match obj {
        None => Ok((default, false)),
        Some(o) if o.is_none() => Ok((default, false)),
        Some(o) => {
            if let Ok(v) = o.extract::<f64>() {
                Ok((v as usize, false))
            } else {
                let v: Vec<f64> = o.extract()?;
                Ok((v.first().copied().unwrap_or(default as f64) as usize, true))
            }
        }
    }
}

/// "Random repair" of the order mutations on 'RI' chromosomes (geatpy 2.7.0 semantics): every gene is
/// checked against the bounds (integer variables widened by 0.499999), an out-of-range gene is redrawn
/// uniformly in the range, a degenerate range collapses to lb, and integer genes are rounded.
/// Returns the varTypes flags of FieldDR, or None when no FieldDR was given.
fn random_repair_rows<R: Rng>(
    chrom: &mut Array2<f64>,
    field: Opt,
    rng: &mut R,
) -> PyResult<Option<Vec<bool>>> {
    let Some(f) = field.filter(|f| !f.is_none()) else {
        return Ok(None);
    };
    let d = chrom.shape()[1];
    let b = bounds(&crate::utils::to_f64_array2(f)?, d, true)?;
    for mut row in chrom.rows_mut() {
        for j in 0..d {
            let x = row[j];
            let v = if b.span[j] <= 1e-15 {
                b.lb[j]
            } else if x < b.lb[j] || x > b.ub[j] {
                b.lb[j] + rng.gen::<f64>() * b.span[j]
            } else {
                x
            };
            row[j] = if b.discrete[j] { v.round() } else { v };
        }
    }
    Ok(Some(b.discrete))
}

/// Output of the order mutations: 'RI' follows varTypes (int32 when all integer), 'P' keeps the input type.
fn order_out(
    py: Python<'_>,
    chrom: Array2<f64>,
    discrete: Option<Vec<bool>>,
    as_int: bool,
) -> PyResult<PyObject> {
    match discrete {
        Some(d) => crate::utils::ri_out(py, chrom, &d),
        None => chrom_out(py, chrom, as_int),
    }
}

fn check_order_encoding(encoding: &str, name: &str) -> PyResult<()> {
    if encoding != "P" && encoding != "RI" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: The encoding must be 'P' or 'RI'.",
            name
        )));
    }
    Ok(())
}

/// Inversion mutation: reverse one segment of each chromosome with probability Pm.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, invert_len=None, parallel=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutinv<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    invert_len: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params6, params7);
    check_order_encoding(encoding, "mutinv")?;
    let as_int = is_integer_like(old_chrom);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let pm = crate::utils::per_gene(pm, 1, 1.0, "Pm")?[0];
    let (len, fixed) = parse_len(invert_len, l)?;
    let len = len.min(l);
    let mut rng = rand::thread_rng();
    if l >= 2 {
        for i in 0..n {
            if rng.gen::<f64>() >= pm {
                continue;
            }
            let k = if fixed {
                len.max(1)
            } else {
                segment_len(l, len, 2, &mut rng)
            };
            let start = rng.gen_range(0..=(l - k));
            let mut row = chrom.row_mut(i);
            let seg: Vec<f64> = (start..start + k).map(|j| row[j]).collect();
            for (t, v) in seg.into_iter().rev().enumerate() {
                row[start + t] = v;
            }
        }
    }
    let discrete = if encoding == "RI" {
        random_repair_rows(&mut chrom, field_dr, &mut rng)?
    } else {
        None
    };
    order_out(py, chrom, discrete, as_int)
}

/// Shift mutation: move one segment to another position, optionally reversing it with probability Pr.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, move_len=None, pr=None, parallel=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutmove<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    move_len: Opt<'_, 'py>,
    pr: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params7);
    check_order_encoding(encoding, "mutmove")?;
    let as_int = is_integer_like(old_chrom);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let pm = crate::utils::per_gene(pm, 1, 1.0, "Pm")?[0];
    let pr = crate::utils::per_gene(pr, 1, 0.0, "Pr")?[0];
    let (len, fixed) = parse_len(move_len, l)?;
    let len = len.min(l.saturating_sub(1)).max(1);
    let mut rng = rand::thread_rng();
    if l >= 2 {
        for i in 0..n {
            if rng.gen::<f64>() >= pm {
                continue;
            }
            let k = if fixed {
                len
            } else {
                segment_len(l, len, 1, &mut rng)
            };
            let start = rng.gen_range(0..=(l - k));
            let slots = l - k + 1;
            let dest = (start + 1 + rng.gen_range(0..(l - k).max(1))) % slots;
            let mut row: Vec<f64> = chrom.row(i).to_vec();
            let mut seg: Vec<f64> = row.drain(start..start + k).collect();
            if rng.gen::<f64>() < pr {
                seg.reverse();
            }
            for (t, v) in seg.into_iter().enumerate() {
                row.insert(dest + t, v);
            }
            for (j, v) in row.into_iter().enumerate() {
                chrom[[i, j]] = v;
            }
        }
    }
    let discrete = if encoding == "RI" {
        random_repair_rows(&mut chrom, field_dr, &mut rng)?
    } else {
        None
    };
    order_out(py, chrom, discrete, as_int)
}

/// Swap mutation: exchange two distinct genes of each chromosome with probability Pm.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, parallel=None, params5=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutswap<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params5: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params5, params6, params7);
    check_order_encoding(encoding, "mutswap")?;
    let as_int = is_integer_like(old_chrom);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let pm = crate::utils::per_gene(pm, 1, 1.0, "Pm")?[0];
    let mut rng = rand::thread_rng();
    if l >= 2 {
        for i in 0..n {
            if rng.gen::<f64>() < pm {
                let a = rng.gen_range(0..l);
                let b = (a + 1 + rng.gen_range(0..l - 1)) % l;
                chrom.swap([i, a], [i, b]);
            }
        }
    }
    let discrete = if encoding == "RI" {
        random_repair_rows(&mut chrom, field_dr, &mut rng)?
    } else {
        None
    };
    order_out(py, chrom, discrete, as_int)
}

/// Permutation-point mutation: replace MutN genes by other values of [lb, ub],
/// swapping with the gene that already holds the new value.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr=None, pm=None, mut_n=None, parallel=None, params6=None, params7=None))]
#[allow(clippy::too_many_arguments)]
pub fn mutpp<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: Opt<'_, 'py>,
    pm: Opt<'_, 'py>,
    mut_n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params6: Opt<'_, 'py>,
    params7: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params6, params7);
    if encoding != "P" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in mutpp: The encoding must be 'P'.",
        ));
    }
    let as_int = is_integer_like(old_chrom);
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let (lo, hi) = match field_dr {
        Some(f) if !f.is_none() => {
            let a = crate::utils::to_f64_array2(f)?;
            (a[[0, 0]].round() as i64, a[[1, 0]].round() as i64)
        }
        _ => {
            let mn = chrom.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = chrom.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mn as i64, mx as i64)
        }
    };
    let range = (hi - lo + 1).max(1) as usize;
    let pm = crate::utils::per_gene(pm, 1, 1.0 / l.max(1) as f64, "Pm")?[0];
    let times = (opt_int(mut_n, 1)?.max(0) as usize).min(l);
    let mut rng = rand::thread_rng();
    if l >= 1 && range >= 2 {
        for i in 0..n {
            if rng.gen::<f64>() >= pm {
                continue;
            }
            // position of each value in the chromosome, or None when absent
            let mut pos: Vec<Option<usize>> = vec![None; range];
            for j in 0..l {
                let v = (chrom[[i, j]] as i64 - lo) as usize;
                if v < range {
                    pos[v] = Some(j);
                }
            }
            for _ in 0..times {
                let a = rng.gen_range(0..l);
                let old = (chrom[[i, a]] as i64 - lo) as usize;
                let new = (old + 1 + rng.gen_range(0..range - 1)) % range;
                match pos[new] {
                    None => {
                        chrom[[i, a]] = (lo + new as i64) as f64;
                        pos[new] = Some(a);
                        pos[old] = None;
                    }
                    Some(b) => {
                        chrom.swap([i, a], [i, b]);
                        pos.swap(old, new);
                    }
                }
            }
        }
    }
    chrom_out(py, chrom, as_int)
}

/// Dispatcher kept for API compatibility: mutate(MUT_F, Encoding, OldChrom, ...) forwards the
/// remaining arguments to the named operator.
#[pyfunction]
#[pyo3(signature = (mut_f, *args, **kwargs))]
pub fn mutate<'py>(
    py: Python<'py>,
    mut_f: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<PyObject> {
    let name = mut_f.to_lowercase();
    let allowed = [
        "mutpolyn", "mutgau", "mutbga", "mutuni", "mutbin", "mutde", "mutinv", "mutmove",
        "mutswap", "mutpp",
    ];
    if !allowed.contains(&name.as_str()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in mutate: unknown operator {}.",
            mut_f
        )));
    }
    let module = py
        .import("_geatpy_core")
        .or_else(|_| py.import("geatpy._geatpy_core"))?;
    Ok(module.getattr(name.as_str())?.call(args, kwargs)?.unbind())
}
