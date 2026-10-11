use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::seq::SliceRandom;
use rand::Rng;

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

/// FitnV_N is either a fitness column vector or the population size (all fitness equal to 1).
fn fitness_or_n(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if !obj.hasattr("__len__")? {
        if let Ok(n) = obj.extract::<f64>() {
            return Ok(vec![1.0; n.max(0.0) as usize]);
        }
    }
    let a = crate::utils::to_f64_array2(obj)?;
    Ok(a.column(0).to_vec())
}

/// Number of individuals to select: None => n, (0, 1) => ceil(n * x), otherwise ceil(x).
fn sel_count(nsel: Opt, n: usize) -> PyResult<usize> {
    match nsel {
        Some(o) if !o.is_none() => {
            let x: f64 = o.extract()?;
            Ok(if x > 0.0 && x < 1.0 {
                (n as f64 * x).ceil() as usize
            } else {
                x.ceil().max(0.0) as usize
            })
        }
        _ => Ok(n),
    }
}

fn idx_out(py: Python<'_>, v: Vec<usize>) -> Bound<'_, PyArray1<i64>> {
    Array1::from_iter(v.into_iter().map(|x| x as i64)).into_pyarray(py)
}

fn order_desc(fit: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..fit.len()).collect();
    o.sort_by(|&a, &b| {
        fit[b]
            .partial_cmp(&fit[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    o
}

fn dup_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    let n = fit.len();
    if n == 0 {
        return vec![];
    }
    let order = order_desc(fit);
    let mut out: Vec<usize> = (0..nsel).map(|i| order[i % n]).collect();
    out.shuffle(&mut rand::thread_rng());
    out
}

fn tour_idx(fit: &[f64], nsel: usize, k: usize, elitist: bool) -> Vec<usize> {
    let n = fit.len();
    if n == 0 {
        return vec![];
    }
    let k = if k >= 1 && k <= n { k } else { 2.min(n) };
    let mut rng = rand::thread_rng();
    // A later contestant wins only when strictly fitter (ties keep the first one drawn).
    let mut out: Vec<usize> = (0..nsel)
        .map(|_| {
            let mut win = rng.gen_range(0..n);
            for _ in 1..k {
                let c = rng.gen_range(0..n);
                if fit[c] > fit[win] {
                    win = c;
                }
            }
            win
        })
        .collect();
    if elitist && nsel > 0 {
        // The (first) best individual survives: if no tournament picked it, it replaces the
        // fittest pick (geatpy 2.7.0 behaviour).
        let best = (1..n).fold(0, |b, i| if fit[i] > fit[b] { i } else { b });
        if !out.contains(&best) {
            let slot = (1..nsel).fold(0, |s, t| if fit[out[t]] > fit[out[s]] { t } else { s });
            out[slot] = best;
        }
    }
    out
}

fn cumulative(fit: &[f64]) -> (Vec<f64>, f64) {
    let shift = fit.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let mut total = 0.0;
    let cum = fit
        .iter()
        .map(|&v| {
            total += v - shift;
            total
        })
        .collect();
    (cum, total)
}

fn rws_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    let n = fit.len();
    let mut rng = rand::thread_rng();
    let (cum, total) = cumulative(fit);
    (0..nsel)
        .map(|_| {
            if n == 0 {
                return 0;
            }
            if total <= 0.0 {
                return rng.gen_range(0..n);
            }
            let r = rng.gen::<f64>() * total;
            cum.partition_point(|&c| c <= r).min(n - 1)
        })
        .collect()
}

fn sus_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    let n = fit.len();
    let mut rng = rand::thread_rng();
    let (cum, total) = cumulative(fit);
    if n == 0 || nsel == 0 {
        return vec![];
    }
    if total <= 0.0 {
        return (0..nsel).map(|_| rng.gen_range(0..n)).collect();
    }
    let step = total / nsel as f64;
    let start = rng.gen::<f64>() * step;
    // Pointers are returned in ascending order, as in geatpy 2.7.0 (no shuffle).
    (0..nsel)
        .map(|i| {
            cum.partition_point(|&c| c <= start + i as f64 * step)
                .min(n - 1)
        })
        .collect()
}

fn urs_idx(n: usize, nsel: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    if n == 0 {
        return vec![];
    }
    (0..nsel).map(|_| rng.gen_range(0..n)).collect()
}

fn rps_idx(n: usize, nsel: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    v.shuffle(&mut rand::thread_rng());
    v.truncate(nsel.min(n));
    v
}

fn rcs_idx(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let rg = if n > 1 {
        rand::thread_rng().gen_range(1..n)
    } else {
        0
    };
    (0..n).map(|i| (i + rg) % n).collect()
}

/// One-to-one survivor selection: the population is cut into chunks of `nsel`; position j of the
/// result is the fittest among the j-th members of all chunks.
fn otos_idx(fit: &[f64], nsel: usize) -> PyResult<Vec<usize>> {
    let n = fit.len();
    if nsel == 0 || !n.is_multiple_of(nsel) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in otos: Nsel must divide the population size. (Nsel必须能够整除种群个体数。)",
        ));
    }
    let mut out: Vec<usize> = (0..nsel)
        .map(|j| {
            (0..n / nsel)
                .map(|s| s * nsel + j)
                .fold(j, |best, i| if fit[i] > fit[best] { i } else { best })
        })
        .collect();
    out.shuffle(&mut rand::thread_rng());
    Ok(out)
}

fn ecs_idx(fit: &[f64], nsel: usize) -> Vec<usize> {
    if fit.is_empty() {
        return vec![];
    }
    vec![order_desc(fit)[0]; nsel]
}

fn select_by_name(name: &str, fit: &[f64], nsel: usize, tour: usize) -> PyResult<Vec<usize>> {
    Ok(match name {
        "dup" => dup_idx(fit, nsel),
        "ecs" => ecs_idx(fit, nsel),
        "etour" => tour_idx(fit, nsel, tour, true),
        "tour" => tour_idx(fit, nsel, tour, false),
        "otos" => otos_idx(fit, nsel)?,
        "rcs" => rcs_idx(fit.len()),
        "rps" => rps_idx(fit.len(), nsel),
        "rws" => rws_idx(fit, nsel),
        "sus" => sus_idx(fit, nsel),
        "urs" => urs_idx(fit.len(), nsel),
        other => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "error in selecting: unknown selection operator '{}'.",
                other
            )))
        }
    })
}

/// Truncation selection: the Nsel fittest individuals (cycling when Nsel > N), in random order.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, params3=None))]
pub fn dup<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, params3);
    Ok(idx_out(py, dup_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Elite copy selection: the best individual Nsel times.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, params3=None))]
pub fn ecs<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, params3);
    Ok(idx_out(py, ecs_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Tournament selection with replacement.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, tour=None, parallel=None))]
pub fn tour<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    tour: Option<usize>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = parallel;
    Ok(idx_out(
        py,
        tour_idx(&fitness_or_n(fitn_v_n)?, nsel, tour.unwrap_or(2), false),
    ))
}

/// Tournament selection in which the best individual always takes part (and therefore survives).
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, tour=None, parallel=None))]
pub fn etour<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    tour: Option<usize>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = parallel;
    Ok(idx_out(
        py,
        tour_idx(&fitness_or_n(fitn_v_n)?, nsel, tour.unwrap_or(2), true),
    ))
}

/// Roulette-wheel selection.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, parallel=None))]
pub fn rws<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, parallel);
    Ok(idx_out(py, rws_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Stochastic universal sampling.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, parallel=None))]
pub fn sus<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, parallel);
    Ok(idx_out(py, sus_idx(&fitness_or_n(fitn_v_n)?, nsel)))
}

/// Uniform random selection with replacement.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, params3=None))]
pub fn urs<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, params3);
    Ok(idx_out(py, urs_idx(fitness_or_n(fitn_v_n)?.len(), nsel)))
}

/// Random compensation selection: r0 = (i + rg) % N with one random rg in [1, N).
#[pyfunction]
#[pyo3(signature = (fitn_v_n, params1=None, params2=None, parallel=None))]
pub fn rcs<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    params1: Opt<'_, 'py>,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params1, params2, parallel);
    Ok(idx_out(py, rcs_idx(fitness_or_n(fitn_v_n)?.len())))
}

/// Random permutation selection without replacement.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel=None, params2=None, params3=None))]
pub fn rps<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: Opt<'_, 'py>,
    params2: Opt<'_, 'py>,
    params3: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, params3);
    let n = fitness_or_n(fitn_v_n)?.len();
    let k = match nsel {
        Some(o) if !o.is_none() => o.extract::<usize>()?,
        _ => n,
    };
    if k > n {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in rps: Nsel must not exceed the population size.",
        ));
    }
    Ok(idx_out(py, rps_idx(n, k)))
}

/// One-to-one survivor selection.
#[pyfunction]
#[pyo3(signature = (fitn_v_n, nsel, params2=None, parallel=None))]
pub fn otos<'py>(
    py: Python<'py>,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: usize,
    params2: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (params2, parallel);
    Ok(idx_out(py, otos_idx(&fitness_or_n(fitn_v_n)?, nsel)?))
}

/// Dispatcher: selecting(SEL_F, FitnV_N, NSel, Parallel).
#[pyfunction]
#[pyo3(signature = (sel_f, fitn_v_n, nsel=None, parallel=None))]
pub fn selecting<'py>(
    py: Python<'py>,
    sel_f: &str,
    fitn_v_n: &Bound<'py, PyAny>,
    nsel: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = parallel;
    let fit = fitness_or_n(fitn_v_n)?;
    let k = sel_count(nsel, fit.len())?;
    Ok(idx_out(py, select_by_name(sel_f, &fit, k, 2)?))
}

/// Multi-population selection. Selection runs over the merged populations; a population that gets
/// nothing then receives its minimum quota (MSel) chosen inside it.
#[pyfunction]
#[pyo3(signature = (sel_f, fitn_vs, nsel=None, msel=None, separate=None))]
pub fn mselecting<'py>(
    py: Python<'py>,
    sel_f: &str,
    fitn_vs: &Bound<'py, PyAny>,
    nsel: Opt<'_, 'py>,
    msel: Option<f64>,
    separate: Option<bool>,
) -> PyResult<Bound<'py, PyList>> {
    let fits: Vec<Vec<f64>> = fitn_vs
        .try_iter()?
        .map(|f| fitness_or_n(&f?))
        .collect::<PyResult<_>>()?;
    let sizes: Vec<usize> = fits.iter().map(|f| f.len()).collect();
    let total: usize = sizes.iter().sum();
    let msel = msel.unwrap_or(0.1);
    let mins: Vec<usize> = sizes
        .iter()
        .map(|&s| {
            if msel < 1.0 {
                (s as f64 * msel).ceil() as usize
            } else {
                (msel.ceil() as usize).min(s)
            }
        })
        .collect();
    let k = sel_count(nsel, total)?.max(mins.iter().sum());
    let all: Vec<f64> = fits.iter().flatten().cloned().collect();
    let offsets: Vec<usize> = sizes
        .iter()
        .scan(0, |acc, &s| {
            let o = *acc;
            *acc += s;
            Some(o)
        })
        .collect();
    let mut per_pop: Vec<Vec<usize>> = vec![Vec::new(); fits.len()];
    for g in select_by_name(sel_f, &all, k, 2)? {
        let p = offsets.iter().rposition(|&o| o <= g).unwrap();
        per_pop[p].push(g - offsets[p]);
    }
    for (p, chosen) in per_pop.iter_mut().enumerate() {
        if chosen.is_empty() && sizes[p] > 0 {
            *chosen = select_by_name(sel_f, &fits[p], mins[p], 2)?;
        }
    }
    let separate = separate.unwrap_or(true);
    let out = PyList::empty(py);
    for (p, chosen) in per_pop.into_iter().enumerate() {
        let shift = if separate { 0 } else { offsets[p] };
        // Python lists of indices, like geatpy 2.7.0.
        out.append(PyList::new(
            py,
            chosen.into_iter().map(|i| (i + shift) as i64),
        )?)?;
    }
    Ok(out)
}

/// Single-objective column prepared for fitness assignment (minimisation, feasibility rule applied).
fn prepared_column(obj_v: &Bound<'_, PyAny>, cv: Opt, maxormins: Opt) -> PyResult<Vec<f64>> {
    let obj = crate::utils::prepared_objectives(obj_v, cv, maxormins)?;
    Ok(obj.column(0).to_vec())
}

fn column(v: Vec<f64>) -> Array2<f64> {
    let n = v.len();
    Array2::from_shape_vec((n, 1), v).unwrap()
}

/// Positive real root of (SP - N) x^(N-1) + SP x^(N-2) + ... + SP x + SP = 0 (Pohlheim).
fn nonlinear_root(sp: f64, n: usize) -> f64 {
    let f = |x: f64| {
        (sp - n as f64) * x.powi(n as i32 - 1)
            + (0..n - 1).map(|i| sp * x.powi(i as i32)).sum::<f64>()
    };
    let (mut lo, mut hi) = (1.0, 2.0);
    while f(hi) > 0.0 && hi < 1e6 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

/// Rank-based fitness. Ties share the rank of the worst member of the tie group.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, rm=None, sp=None, mask=None))]
pub fn ranking<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    rm: Option<i64>,
    sp: Option<f64>,
    mask: Opt<'_, 'py>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(obj_v, cv, maxormins)?;
    let n = f.len();
    // position counted from the worst individual (0) to the best (n - 1)
    let pos: Vec<usize> = f
        .iter()
        .map(|&x| f.iter().filter(|&&y| y > x).count())
        .collect();
    let fit: Vec<f64> = if let Some(m) = mask.filter(|m| !m.is_none()) {
        let m = crate::utils::to_f64_array1(m)?;
        if m.len() != n {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in ranking: The length of Mask must equal Nind.",
            ));
        }
        pos.iter().map(|&p| m[p]).collect()
    } else if n <= 1 {
        vec![1.0; n]
    } else if rm.unwrap_or(0) == 1 {
        let sp = sp.unwrap_or(2.0);
        let x = nonlinear_root(sp, n);
        let denom: f64 = (0..n).map(|i| x.powi(i as i32)).sum();
        pos.iter()
            .map(|&p| n as f64 * x.powi(p as i32) / denom)
            .collect()
    } else {
        let sp = sp.unwrap_or(2.0);
        pos.iter()
            .map(|&p| 2.0 - sp + 2.0 * (sp - 1.0) * p as f64 / (n - 1) as f64)
            .collect()
    };
    Ok(column(fit).into_pyarray(py))
}

/// Linear scaling (Goldberg) of max(F) - F; the minimum fitness is kept non-negative.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, smul=None))]
pub fn scaling<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    smul: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(obj_v, cv, maxormins)?;
    let n = f.len();
    let smul = smul.unwrap_or(2.0);
    let fmax = f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let g: Vec<f64> = f.iter().map(|&x| fmax - x).collect();
    if n == 0 || g.iter().all(|&x| x == g[0]) {
        return Ok(column(vec![1.0; n]).into_pyarray(py));
    }
    let avg = g.iter().sum::<f64>() / n as f64;
    let gmin = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let gmax = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (a, b) = if gmin < (avg * smul - gmax) / (smul - 1.0) {
        let a = avg / (avg - gmin);
        (a, -gmin * a)
    } else {
        (
            (smul - 1.0) * avg / (gmax - avg),
            (gmax - avg * smul) * avg / (gmax - avg),
        )
    };
    Ok(column(g.iter().map(|&x| a * x + b).collect()).into_pyarray(py))
}

/// Power scaling: ((max - F) / (max - min)) ^ k.
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, k=None))]
pub fn powing<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    k: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(obj_v, cv, maxormins)?;
    let k = k.unwrap_or(1.0);
    let (mn, mx) = f
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let fit = if mx - mn <= 0.0 {
        vec![1.0; f.len()]
    } else {
        f.iter().map(|&x| ((mx - x) / (mx - mn)).powf(k)).collect()
    };
    Ok(column(fit).into_pyarray(py))
}

/// Exponential scaling: exp(-Beta * (F - min) / (max - min)).
#[pyfunction]
#[pyo3(signature = (obj_v, cv=None, maxormins=None, beta=None))]
pub fn indexing<'py>(
    py: Python<'py>,
    obj_v: &Bound<'py, PyAny>,
    cv: Opt<'_, 'py>,
    maxormins: Opt<'_, 'py>,
    beta: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = prepared_column(obj_v, cv, maxormins)?;
    let beta = beta.unwrap_or(1.0);
    let (mn, mx) = f
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let fit = if mx - mn <= 0.0 {
        vec![1.0; f.len()]
    } else {
        f.iter()
            .map(|&x| (-beta * (x - mn) / (mx - mn)).exp())
            .collect()
    };
    Ok(column(fit).into_pyarray(py))
}
