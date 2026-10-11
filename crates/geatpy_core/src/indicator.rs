use numpy::ndarray::Array2;
use pyo3::prelude::*;
use rand::Rng;

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

fn pair(
    obj_v: &Bound<'_, PyAny>,
    pf: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Array2<f64>, Array2<f64>)> {
    let p = crate::utils::to_f64_array2(obj_v)?;
    let r = crate::utils::to_f64_array2(pf)?;
    if p.shape()[1] != r.shape()[1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in indicator.{}: ObjV and PF must have the same number of columns.",
            name
        )));
    }
    Ok((p, r))
}

fn sq_dist(a: &Array2<f64>, i: usize, b: &Array2<f64>, k: usize) -> f64 {
    (0..a.shape()[1])
        .map(|j| (a[[i, j]] - b[[k, j]]).powi(2))
        .sum()
}

/// Generational distance: sqrt(sum of squared nearest distances to PF) / N.
#[pyfunction]
#[pyo3(signature = (obj_v, pf, parallel=None))]
pub fn GD(obj_v: &Bound<'_, PyAny>, pf: &Bound<'_, PyAny>, parallel: Opt) -> PyResult<f64> {
    let _ = parallel;
    let (p, r) = pair(obj_v, pf, "GD")?;
    let (n, nr) = (p.shape()[0], r.shape()[0]);
    if n == 0 || nr == 0 {
        return Ok(0.0);
    }
    let s: f64 = (0..n)
        .map(|i| {
            (0..nr)
                .map(|k| sq_dist(&p, i, &r, k))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    Ok(s.sqrt() / n as f64)
}

/// Inverted generational distance: mean distance from each PF point to its nearest solution.
#[pyfunction]
#[pyo3(signature = (obj_v, pf, parallel=None))]
pub fn IGD(obj_v: &Bound<'_, PyAny>, pf: &Bound<'_, PyAny>, parallel: Opt) -> PyResult<f64> {
    let _ = parallel;
    let (p, r) = pair(obj_v, pf, "IGD")?;
    let (n, nr) = (p.shape()[0], r.shape()[0]);
    if n == 0 || nr == 0 {
        return Ok(0.0);
    }
    let s: f64 = (0..nr)
        .map(|k| {
            (0..n)
                .map(|i| sq_dist(&p, i, &r, k))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .sum();
    Ok(s / nr as f64)
}

/// Schott's spacing: sample standard deviation of nearest-neighbour Manhattan distances.
#[pyfunction]
#[pyo3(signature = (obj_v, parallel=None))]
pub fn Spacing(obj_v: &Bound<'_, PyAny>, parallel: Opt) -> PyResult<f64> {
    let _ = parallel;
    let p = crate::utils::to_f64_array2(obj_v)?;
    let (n, m) = (p.shape()[0], p.shape()[1]);
    if n <= 1 {
        return Ok(0.0);
    }
    let d: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&k| k != i)
                .map(|k| (0..m).map(|j| (p[[i, j]] - p[[k, j]]).abs()).sum::<f64>())
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mean = d.iter().sum::<f64>() / n as f64;
    Ok((d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt())
}

/// Exact hypervolume (minimisation) of `pts` against `refp` by slicing along the last objective.
fn hv_exact(mut pts: Vec<Vec<f64>>, refp: &[f64]) -> f64 {
    let m = refp.len();
    pts.retain(|p| p.iter().zip(refp).all(|(a, r)| a < r));
    if pts.is_empty() {
        return 0.0;
    }
    if m == 1 {
        return refp[0] - pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    }
    if m == 2 {
        pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        let mut vol = 0.0;
        let mut y = refp[1];
        for p in &pts {
            if p[1] < y {
                vol += (refp[0] - p[0]) * (y - p[1]);
                y = p[1];
            }
        }
        return vol;
    }
    pts.sort_by(|a, b| a[m - 1].partial_cmp(&b[m - 1]).unwrap());
    let mut vol = 0.0;
    for i in 0..pts.len() {
        let upper = if i + 1 < pts.len() {
            pts[i + 1][m - 1]
        } else {
            refp[m - 1]
        };
        let depth = upper - pts[i][m - 1];
        if depth > 0.0 {
            let slice: Vec<Vec<f64>> = pts[..=i].iter().map(|p| p[..m - 1].to_vec()).collect();
            vol += depth * hv_exact(slice, &refp[..m - 1]);
        }
    }
    vol
}

fn hv_monte_carlo(pts: &[Vec<f64>], refp: &[f64], samples: usize) -> f64 {
    let m = refp.len();
    let lo: Vec<f64> = (0..m)
        .map(|j| pts.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min))
        .collect();
    let box_vol: f64 = (0..m).map(|j| refp[j] - lo[j]).product();
    let mut rng = rand::thread_rng();
    let hits = (0..samples)
        .filter(|_| {
            let s: Vec<f64> = (0..m)
                .map(|j| lo[j] + rng.gen::<f64>() * (refp[j] - lo[j]))
                .collect();
            pts.iter().any(|p| p.iter().zip(&s).all(|(a, b)| a <= b))
        })
        .count();
    box_vol * hits as f64 / samples as f64
}

/// Hypervolume of ObjV after dividing by the column maxima of PF (PF defaults to ObjV), with the
/// reference point 1.1 on every axis, normalised by 1.1^M. Exact for M <= 4, Monte Carlo above.
#[pyfunction]
#[pyo3(signature = (obj_v, pf=None, parallel=None))]
pub fn HV(obj_v: &Bound<'_, PyAny>, pf: Opt, parallel: Opt) -> PyResult<f64> {
    let _ = parallel;
    let p = crate::utils::to_f64_array2(obj_v)?;
    let r = match pf {
        Some(x) if !x.is_none() => crate::utils::to_f64_array2(x)?,
        _ => p.clone(),
    };
    let (n, m) = (p.shape()[0], p.shape()[1]);
    if r.shape()[1] != m {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in indicator.HV: PF must be a matrix with as many columns as ObjV.",
        ));
    }
    if n == 0 || m == 0 {
        return Ok(0.0);
    }
    let scale: Vec<f64> = (0..m)
        .map(|j| {
            r.column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let pts: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| p[[i, j]] / scale[j]).collect())
        .collect();
    let refp = vec![1.1; m];
    let inside: Vec<Vec<f64>> = pts
        .into_iter()
        .filter(|q| q.iter().all(|&x| x < 1.1))
        .collect();
    if inside.is_empty() {
        return Ok(0.0);
    }
    let vol = if m <= 4 {
        hv_exact(inside, &refp)
    } else {
        hv_monte_carlo(&inside, &refp, 1_000_000)
    };
    Ok(vol / 1.1f64.powi(m as i32))
}
