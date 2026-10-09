use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::Rng;

/// Generational Distance (GD)
#[pyfunction]
pub fn GD(nd_set: PyReadonlyArray2<f64>, refer_obj_v: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let p = nd_set.as_array();
    let r = refer_obj_v.as_array();
    let n = p.shape()[0];
    let n_ref = r.shape()[0];
    let m = p.shape()[1];

    if n == 0 || n_ref == 0 {
        return Ok(0.0);
    }

    let mut sum_sq = 0.0;
    for i in 0..n {
        let mut min_d_sq = f64::INFINITY;
        for k in 0..n_ref {
            let mut d_sq = 0.0;
            for j in 0..m {
                let diff = p[[i, j]] - r[[k, j]];
                d_sq += diff * diff;
            }
            if d_sq < min_d_sq {
                min_d_sq = d_sq;
            }
        }
        sum_sq += min_d_sq;
    }

    Ok((sum_sq.sqrt()) / (n as f64))
}

/// Inverted Generational Distance (IGD)
#[pyfunction]
pub fn IGD(nd_set: PyReadonlyArray2<f64>, refer_obj_v: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let p = nd_set.as_array();
    let r = refer_obj_v.as_array();
    let n = p.shape()[0];
    let n_ref = r.shape()[0];
    let m = p.shape()[1];

    if n == 0 || n_ref == 0 {
        return Ok(0.0);
    }

    let mut sum_d = 0.0;
    for k in 0..n_ref {
        let mut min_d = f64::INFINITY;
        for i in 0..n {
            let mut d_sq = 0.0;
            for j in 0..m {
                let diff = r[[k, j]] - p[[i, j]];
                d_sq += diff * diff;
            }
            let d = d_sq.sqrt();
            if d < min_d {
                min_d = d;
            }
        }
        sum_d += min_d;
    }

    Ok(sum_d / (n_ref as f64))
}

/// Spacing metric
#[pyfunction]
pub fn Spacing(nd_set: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let p = nd_set.as_array();
    let n = p.shape()[0];
    let m = p.shape()[1];

    if n <= 1 {
        return Ok(0.0);
    }

    let mut d_vec = vec![0.0; n];
    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        for k in 0..n {
            if i != k {
                let mut dist = 0.0;
                for j in 0..m {
                    dist += (p[[i, j]] - p[[k, j]]).abs();
                }
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        d_vec[i] = min_dist;
    }

    let mean_d = d_vec.iter().sum::<f64>() / (n as f64);
    let var: f64 = d_vec.iter().map(|&d| (d - mean_d).powi(2)).sum::<f64>() / ((n - 1) as f64);

    Ok(var.sqrt())
}

/// Hypervolume (HV) indicator
#[pyfunction]
#[pyo3(signature = (nd_set, refer_obj_v=None))]
pub fn HV(nd_set: PyReadonlyArray2<f64>, refer_obj_v: Option<PyReadonlyArray2<f64>>) -> PyResult<f64> {
    let p = nd_set.as_array();
    let n = p.shape()[0];
    let m = p.shape()[1];

    if n == 0 || m == 0 {
        return Ok(0.0);
    }

    // Determine reference point (nadir point)
    let ref_pt: Vec<f64> = if let Some(r_arr) = refer_obj_v {
        let r = r_arr.as_array();
        if r.shape()[0] > 0 {
            (0..m).map(|j| {
                (0..r.shape()[0]).map(|i| r[[i, j]]).fold(f64::NEG_INFINITY, f64::max)
            }).collect()
        } else {
            (0..m).map(|j| {
                (0..n).map(|i| p[[i, j]]).fold(f64::NEG_INFINITY, f64::max) * 1.1 + 1e-3
            }).collect()
        }
    } else {
        (0..m).map(|j| {
            (0..n).map(|i| p[[i, j]]).fold(f64::NEG_INFINITY, f64::max) * 1.1 + 1e-3
        }).collect()
    };

    // Filter points strictly dominating ref_pt
    let valid_points: Vec<Vec<f64>> = (0..n)
        .filter_map(|i| {
            let pt: Vec<f64> = (0..m).map(|j| p[[i, j]]).collect();
            if pt.iter().zip(&ref_pt).all(|(&pj, &rj)| pj <= rj) {
                Some(pt)
            } else {
                None
            }
        })
        .collect();

    if valid_points.is_empty() {
        return Ok(0.0);
    }

    // Exact 2D calculation
    if m == 2 {
        let mut pts = valid_points;
        // Sort by f1 ascending
        pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

        let mut hv = 0.0;
        let mut last_y = ref_pt[1];

        for pt in pts {
            if pt[1] < last_y {
                hv += (ref_pt[0] - pt[0]) * (last_y - pt[1]);
                last_y = pt[1];
            }
        }
        return Ok(hv.max(0.0));
    }

    // For M > 2: Fast Monte Carlo approximation with adaptive samples
    let min_pt: Vec<f64> = (0..m).map(|j| {
        valid_points.iter().map(|pt| pt[j]).fold(f64::INFINITY, f64::min)
    }).collect();

    let total_box_vol: f64 = (0..m).map(|j| (ref_pt[j] - min_pt[j]).max(0.0)).product();
    if total_box_vol <= 1e-12 {
        return Ok(0.0);
    }

    let n_samples = 20_000usize;
    let mut rng = rand::thread_rng();
    let mut hits = 0usize;

    for _ in 0..n_samples {
        let sample: Vec<f64> = (0..m).map(|j| rng.gen_range(min_pt[j]..ref_pt[j])).collect();
        let dominated = valid_points.iter().any(|pt| {
            pt.iter().zip(&sample).all(|(&pj, &sj)| pj <= sj)
        });
        if dominated {
            hits += 1;
        }
    }

    let hv = total_box_vol * (hits as f64) / (n_samples as f64);
    Ok(hv)
}
