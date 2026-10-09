use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rand::Rng;

fn parse_half_n(half_n: Option<&Bound<'_, PyAny>>, n_ind: usize) -> (bool, usize) {
    if let Some(obj) = half_n {
        if let Ok(b) = obj.extract::<bool>() {
            if b {
                return (true, n_ind / 2);
            } else {
                return (false, n_ind);
            }
        }
        if let Ok(k) = obj.extract::<usize>() {
            if k > 0 {
                return (true, k.min(n_ind));
            } else {
                return (false, n_ind);
            }
        }
    }
    (false, n_ind)
}

fn get_mating_pairs(n_ind: usize, is_half: bool, num_offspring: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    if is_half {
        let half = n_ind / 2;
        for i in 0..num_offspring {
            let p2 = if half > 0 { (i + half) % n_ind } else { i };
            pairs.push((i, p2));
        }
    } else {
        for i in (0..n_ind.saturating_sub(1)).step_by(2) {
            pairs.push((i, i + 1));
        }
    }
    pairs
}

/// Simulated Binary Crossover (SBX)
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=1.0, half_n=None, n=20.0, parallel=false))]
pub fn recsbx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    n: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];

    if n_ind < 2 || d == 0 {
        return Ok(chrom.to_owned().into_pyarray(py));
    }

    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut new_chrom = chrom.to_owned();

    let mut rng = rand::thread_rng();
    let eta = n.max(0.0);
    let exponent = 1.0 / (eta + 1.0);

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            for j in 0..d {
                let y1 = chrom[[p1_idx, j]];
                let y2 = chrom[[p2_idx, j]];

                if (y1 - y2).abs() > 1e-12 {
                    let u: f64 = rng.gen();
                    let beta = if u <= 0.5 {
                        (2.0 * u).powf(exponent)
                    } else {
                        (1.0 / (2.0 * (1.0 - u))).powf(exponent)
                    };

                    let mut c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1).abs());
                    let mut c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1).abs());

                    if rng.gen_bool(0.5) {
                        std::mem::swap(&mut c1, &mut c2);
                    }

                    new_chrom[[p1_idx, j]] = c1;
                    if !is_half {
                        new_chrom[[p2_idx, j]] = c2;
                    }
                }
            }
        }
    }

    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Discrete Recombination
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=0.0, half_n=None, gene_id=None, parallel=false))]
pub fn recdis<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (rec_opt, gene_id, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        for j in 0..d {
            if rng.gen_bool(0.5) {
                new_chrom[[p1_idx, j]] = chrom[[p2_idx, j]];
            }
            if !is_half && rng.gen_bool(0.5) {
                new_chrom[[p2_idx, j]] = chrom[[p1_idx, j]];
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Intermediate Recombination
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=0.0, half_n=None, alpha=None, parallel=false))]
pub fn recint<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    alpha: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (rec_opt, alpha, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        for j in 0..d {
            let a1 = rng.gen_range(-0.25..1.25);
            new_chrom[[p1_idx, j]] = chrom[[p1_idx, j]] + a1 * (chrom[[p2_idx, j]] - chrom[[p1_idx, j]]);
            if !is_half {
                let a2 = rng.gen_range(-0.25..1.25);
                new_chrom[[p2_idx, j]] = chrom[[p2_idx, j]] + a2 * (chrom[[p1_idx, j]] - chrom[[p2_idx, j]]);
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Linear Recombination
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=0.0, half_n=None, alpha=None, parallel=false))]
pub fn reclin<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    alpha: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (rec_opt, alpha, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        let a1 = rng.gen_range(-0.25..1.25);
        for j in 0..d {
            new_chrom[[p1_idx, j]] = chrom[[p1_idx, j]] + a1 * (chrom[[p2_idx, j]] - chrom[[p1_idx, j]]);
        }
        if !is_half {
            let a2 = rng.gen_range(-0.25..1.25);
            for j in 0..d {
                new_chrom[[p2_idx, j]] = chrom[[p2_idx, j]] + a2 * (chrom[[p1_idx, j]] - chrom[[p2_idx, j]]);
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Normal Distribution Crossover
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, a=None, parallel=false))]
pub fn recndx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    a: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (a, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            for j in 0..d {
                let diff = (chrom[[p1_idx, j]] - chrom[[p2_idx, j]]).abs();
                let normal: f64 = rng.sample(rand_distr::StandardNormal);
                let shift = normal * (diff / 6.0).max(1e-6);
                new_chrom[[p1_idx, j]] = chrom[[p1_idx, j]] + shift;
                if !is_half {
                    new_chrom[[p2_idx, j]] = chrom[[p2_idx, j]] - shift;
                }
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Single-point Crossover
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovsp<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (gene_id, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    if d <= 1 {
        return Ok(new_chrom.into_pyarray(py));
    }
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            let point = rng.gen_range(1..d);
            for j in point..d {
                new_chrom[[p1_idx, j]] = chrom[[p2_idx, j]];
                if !is_half {
                    new_chrom[[p2_idx, j]] = chrom[[p1_idx, j]];
                }
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Double-point Crossover
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovdp<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    if d <= 2 {
        return xovsp(py, old_chrom, xovr, half_n, gene_id, parallel);
    }
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            let mut pt1 = rng.gen_range(1..d);
            let mut pt2 = rng.gen_range(1..d);
            if pt1 > pt2 {
                std::mem::swap(&mut pt1, &mut pt2);
            }
            for j in pt1..pt2 {
                new_chrom[[p1_idx, j]] = chrom[[p2_idx, j]];
                if !is_half {
                    new_chrom[[p2_idx, j]] = chrom[[p1_idx, j]];
                }
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Uniform Crossover
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovud<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (gene_id, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            for j in 0..d {
                if rng.gen_bool(0.5) {
                    new_chrom[[p1_idx, j]] = chrom[[p2_idx, j]];
                    if !is_half {
                        new_chrom[[p2_idx, j]] = chrom[[p1_idx, j]];
                    }
                }
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Shuffle Crossover
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovsh<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    xovsp(py, old_chrom, xovr, half_n, gene_id, parallel)
}

/// Partially Mapped Crossover (PMX) for permutations
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, method=1, parallel=false))]
pub fn xovpmx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    method: i32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (method, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    if d <= 2 {
        return Ok(new_chrom.into_pyarray(py));
    }
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            let mut pt1 = rng.gen_range(0..d);
            let mut pt2 = rng.gen_range(0..d);
            if pt1 > pt2 {
                std::mem::swap(&mut pt1, &mut pt2);
            }

            let mut c1 = chrom.row(p1_idx).to_vec();
            let mut c2 = chrom.row(p2_idx).to_vec();

            for i in pt1..=pt2 {
                let v1 = chrom[[p1_idx, i]];
                let v2 = chrom[[p2_idx, i]];
                if (v1 - v2).abs() > 1e-12 {
                    if let Some(pos1) = c1.iter().position(|&x| (x - v2).abs() < 1e-12) {
                        c1[pos1] = c1[i];
                        c1[i] = v2;
                    }
                    if let Some(pos2) = c2.iter().position(|&x| (x - v1).abs() < 1e-12) {
                        c2[pos2] = c2[i];
                        c2[i] = v1;
                    }
                }
            }

            for j in 0..d {
                new_chrom[[p1_idx, j]] = c1[j];
                if !is_half {
                    new_chrom[[p2_idx, j]] = c2[j];
                }
            }
        }
    }

    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Order Crossover (OX) for permutations
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, parallel=false))]
pub fn xovox<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = parallel;
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    if d <= 2 {
        return Ok(new_chrom.into_pyarray(py));
    }
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        if rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            let mut pt1 = rng.gen_range(0..d);
            let mut pt2 = rng.gen_range(0..d);
            if pt1 > pt2 {
                std::mem::swap(&mut pt1, &mut pt2);
            }

            let mut c1 = vec![-1.0; d];
            for i in pt1..=pt2 {
                c1[i] = chrom[[p1_idx, i]];
            }
            let mut cur = (pt2 + 1) % d;
            for i in 0..d {
                let idx = (pt2 + 1 + i) % d;
                let val = chrom[[p2_idx, idx]];
                if !c1[pt1..=pt2].iter().any(|&x| (x - val).abs() < 1e-12) {
                    c1[cur] = val;
                    cur = (cur + 1) % d;
                }
            }

            for j in 0..d {
                new_chrom[[p1_idx, j]] = c1[j];
            }
        }
    }

    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Section Crossover
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovsec<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    xovdp(py, old_chrom, xovr, half_n, gene_id, parallel)
}

/// Binomial crossover for Differential Evolution
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovbd<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (gene_id, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        let j_rand = rng.gen_range(0..d);
        for j in 0..d {
            if rng.gen_bool(xovr.clamp(0.0, 1.0)) || j == j_rand {
                new_chrom[[p1_idx, j]] = chrom[[p2_idx, j]];
            }
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// Exponential crossover for Differential Evolution
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=0.7, half_n=None, gene_id=None, parallel=false))]
pub fn xovexp<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: f64,
    half_n: Option<&Bound<'py, PyAny>>,
    gene_id: Option<&Bound<'py, PyAny>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = (gene_id, parallel);
    let chrom = crate::utils::to_f64_array2(old_chrom)?;
    let n_ind = chrom.shape()[0];
    let d = chrom.shape()[1];
    let mut new_chrom = chrom.to_owned();
    let (is_half, num_offspring) = parse_half_n(half_n, n_ind);
    let pairs = get_mating_pairs(n_ind, is_half, num_offspring);
    let mut rng = rand::thread_rng();

    for (p1_idx, p2_idx) in pairs {
        let start = rng.gen_range(0..d);
        let mut l = 0;
        while l < d && rng.gen_bool(xovr.clamp(0.0, 1.0)) {
            new_chrom[[p1_idx, (start + l) % d]] = chrom[[p2_idx, (start + l) % d]];
            l += 1;
        }
    }
    let final_chrom = if is_half {
        new_chrom.slice_move(ndarray::s![0..num_offspring, ..])
    } else {
        new_chrom
    };
    Ok(final_chrom.into_pyarray(py))
}

/// High-level Recombination Dispatcher
#[pyfunction]
#[pyo3(signature = (rec_oper, old_chrom, **kwargs))]
pub fn recombin<'py>(
    py: Python<'py>,
    rec_oper: &str,
    old_chrom: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut xovr = 1.0;
    let mut half_n: Option<Bound<'py, PyAny>> = None;
    let mut n = 20.0;

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("XOVR")? {
            xovr = val.extract::<f64>()?;
        }
        if let Some(val) = dict.get_item("Half_N")? {
            half_n = Some(val);
        }
        if let Some(val) = dict.get_item("n")? {
            n = val.extract::<f64>()?;
        }
    }

    let hn_ref = half_n.as_ref();
    match rec_oper.to_lowercase().as_str() {
        "recsbx" => recsbx(py, old_chrom, xovr, hn_ref, n, false),
        "recdis" => recdis(py, old_chrom, xovr, hn_ref, None, false),
        "recint" => recint(py, old_chrom, xovr, hn_ref, None, false),
        "reclin" => reclin(py, old_chrom, xovr, hn_ref, None, false),
        "recndx" => recndx(py, old_chrom, xovr, hn_ref, None, false),
        "xovsp" => xovsp(py, old_chrom, xovr, hn_ref, None, false),
        "xovdp" => xovdp(py, old_chrom, xovr, hn_ref, None, false),
        "xovud" => xovud(py, old_chrom, xovr, hn_ref, None, false),
        "xovsh" => xovsh(py, old_chrom, xovr, hn_ref, None, false),
        "xovpmx" => xovpmx(py, old_chrom, xovr, hn_ref, 1, false),
        "xovox" => xovox(py, old_chrom, xovr, hn_ref, false),
        "xovsec" => xovsec(py, old_chrom, xovr, hn_ref, None, false),
        "xovbd" => xovbd(py, old_chrom, xovr, hn_ref, None, false),
        "xovexp" => xovexp(py, old_chrom, xovr, hn_ref, None, false),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported recombination operator: {}",
            rec_oper
        ))),
    }
}
