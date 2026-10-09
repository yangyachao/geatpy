use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

/// Create FieldD or FieldDR descriptor matrix
#[pyfunction]
#[pyo3(signature = (encoding, var_types, ranges, borders, precisions_or_contraction=None, codes=None, scales=None))]
pub fn crtfld<'py>(
    py: Python<'py>,
    encoding: &str,
    var_types: &Bound<'py, PyAny>,
    ranges: &Bound<'py, PyAny>,
    borders: &Bound<'py, PyAny>,
    precisions_or_contraction: Option<&Bound<'py, PyAny>>,
    codes: Option<&Bound<'py, PyAny>>,
    scales: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let var_types = crate::utils::to_f64_array1(var_types)?;
    let ranges = crate::utils::to_f64_array2(ranges)?;
    let borders = crate::utils::to_f64_array2(borders)?;
    let d = var_types.len();

    let mut lb = ranges.row(0).to_vec();
    let mut ub = ranges.row(1).to_vec();
    let mut lbin = borders.row(0).to_vec();
    let mut ubin = borders.row(1).to_vec();

    if encoding == "RI" || encoding == "P" {
        // Parse contraction if given
        let contraction: Vec<f64> = if let Some(obj) = precisions_or_contraction {
            if let Ok(seq) = obj.extract::<Vec<f64>>() {
                seq
            } else {
                vec![0.0; d]
            }
        } else {
            vec![0.0; d]
        };

        for j in 0..d {
            let is_discrete = var_types[j] == 1.0;
            if is_discrete {
                if lbin[j] == 0.0 {
                    lb[j] = (lb[j] + 1.0).floor();
                } else {
                    lb[j] = lb[j].ceil();
                }
                if ubin[j] == 0.0 {
                    ub[j] = (ub[j] - 1.0).ceil();
                } else {
                    ub[j] = ub[j].floor();
                }
                lbin[j] = 1.0;
                ubin[j] = 1.0;
            } else {
                let shrink = if j < contraction.len() && contraction[j] > 0.0 {
                    10.0_f64.powf(-contraction[j])
                } else {
                    0.0
                };
                if lbin[j] == 0.0 {
                    lb[j] += shrink;
                }
                if ubin[j] == 0.0 {
                    ub[j] -= shrink;
                }
            }
        }

        let mut res = Array2::<f64>::zeros((3, d));
        for j in 0..d {
            res[[0, j]] = lb[j];
            res[[1, j]] = ub[j];
            res[[2, j]] = var_types[j];
        }
        Ok(res.into_pyarray(py))
    } else if encoding == "BG" {
        let precisions: Vec<f64> = if let Some(obj) = precisions_or_contraction {
            if let Ok(seq) = obj.extract::<Vec<f64>>() {
                seq
            } else {
                vec![0.0; d]
            }
        } else {
            vec![0.0; d]
        };

        let codes_vec: Vec<f64> = if let Some(obj) = codes {
            if let Ok(seq) = obj.extract::<Vec<f64>>() {
                seq
            } else {
                vec![0.0; d]
            }
        } else {
            vec![0.0; d]
        };

        let scales_vec: Vec<f64> = if let Some(obj) = scales {
            if let Ok(seq) = obj.extract::<Vec<f64>>() {
                seq
            } else {
                vec![0.0; d]
            }
        } else {
            vec![0.0; d]
        };

        let mut lens = vec![0.0; d];
        for j in 0..d {
            let p = if j < precisions.len() { precisions[j] } else { 0.0 };
            let span = if var_types[j] == 1.0 {
                (ub[j] - lb[j]).abs()
            } else {
                (ub[j] - lb[j]).abs() * 10.0_f64.powf(p)
            };
            let mut l = (span + 1.0).log2().ceil();
            if l < 1.0 {
                l = 1.0;
            }
            lens[j] = l;
        }

        let mut res = Array2::<f64>::zeros((8, d));
        for j in 0..d {
            res[[0, j]] = lens[j];
            res[[1, j]] = lb[j];
            res[[2, j]] = ub[j];
            res[[3, j]] = if j < codes_vec.len() { codes_vec[j] } else { 0.0 };
            res[[4, j]] = if j < scales_vec.len() { scales_vec[j] } else { 0.0 };
            res[[5, j]] = lbin[j];
            res[[6, j]] = ubin[j];
            res[[7, j]] = var_types[j];
        }
        Ok(res.into_pyarray(py))
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported encoding: {}. Must be 'BG', 'RI' or 'P'.",
            encoding
        )))
    }
}

/// Decode binary/gray chromosomes into real/integer phenotype matrix
#[pyfunction]
pub fn bs2ri<'py>(
    py: Python<'py>,
    chrom: PyReadonlyArray2<f64>,
    field_d: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let chrom_arr = chrom.as_array();
    let field = field_d.as_array();
    let n_ind = chrom_arr.shape()[0];
    let d = field.shape()[1];

    let lens: Vec<usize> = (0..d).map(|j| field[[0, j]] as usize).collect();
    let lb: Vec<f64> = (0..d).map(|j| field[[1, j]]).collect();
    let ub: Vec<f64> = (0..d).map(|j| field[[2, j]]).collect();
    let codes: Vec<u8> = (0..d).map(|j| field[[3, j]] as u8).collect();
    let lbin: Vec<u8> = (0..d).map(|j| field[[5, j]] as u8).collect();
    let ubin: Vec<u8> = (0..d).map(|j| field[[6, j]] as u8).collect();
    let var_types: Vec<u8> = (0..d).map(|j| field[[7, j]] as u8).collect();

    let mut phen = Array2::<f64>::zeros((n_ind, d));

    for i in 0..n_ind {
        let mut col_start = 0;
        for j in 0..d {
            let l = lens[j];
            let is_gray = codes[j] == 1;

            let mut bits = Vec::with_capacity(l);
            for k in 0..l {
                bits.push(if chrom_arr[[i, col_start + k]] > 0.5 { 1u64 } else { 0u64 });
            }

            // If Gray code, convert to binary
            let binary_bits = if is_gray {
                let mut bin = vec![0u64; l];
                if l > 0 {
                    bin[0] = bits[0];
                    for k in 1..l {
                        bin[k] = bin[k - 1] ^ bits[k];
                    }
                }
                bin
            } else {
                bits
            };

            // Calculate decimal integer value
            let mut val_int = 0u64;
            for &b in &binary_bits {
                val_int = (val_int << 1) | b;
            }

            let denom = if lbin[j] == 1 && ubin[j] == 1 {
                if l > 0 { (1u64 << l) - 1 } else { 1 }
            } else {
                1u64 << l
            };

            let v_float = if denom == 0 {
                lb[j]
            } else {
                lb[j] + (val_int as f64) * (ub[j] - lb[j]) / (denom as f64)
            };

            phen[[i, j]] = if var_types[j] == 1 {
                v_float.round()
            } else {
                v_float
            };

            col_start += l;
        }
    }

    Ok(phen.into_pyarray(py))
}

#[pyfunction]
pub fn bs2int<'py>(
    py: Python<'py>,
    chrom: PyReadonlyArray2<f64>,
    field_d: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut field = field_d.as_array().to_owned();
    let d = field.shape()[1];
    for j in 0..d {
        field[[7, j]] = 1.0;
    }
    let dummy = field.into_pyarray(py);
    bs2ri(py, chrom, dummy.readonly())
}

#[pyfunction]
pub fn bs2real<'py>(
    py: Python<'py>,
    chrom: PyReadonlyArray2<f64>,
    field_d: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut field = field_d.as_array().to_owned();
    let d = field.shape()[1];
    for j in 0..d {
        field[[7, j]] = 0.0;
    }
    let dummy = field.into_pyarray(py);
    bs2ri(py, chrom, dummy.readonly())
}

/// Encode real/integer phenotype matrix back into binary/gray chromosome matrix
#[pyfunction]
pub fn ri2bs<'py>(
    py: Python<'py>,
    phen: PyReadonlyArray2<f64>,
    field_d: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let phen_arr = phen.as_array();
    let field = field_d.as_array();
    let n_ind = phen_arr.shape()[0];
    let d = field.shape()[1];

    let lens: Vec<usize> = (0..d).map(|j| field[[0, j]] as usize).collect();
    let total_bits: usize = lens.iter().sum();
    let lb: Vec<f64> = (0..d).map(|j| field[[1, j]]).collect();
    let ub: Vec<f64> = (0..d).map(|j| field[[2, j]]).collect();
    let codes: Vec<u8> = (0..d).map(|j| field[[3, j]] as u8).collect();
    let lbin: Vec<u8> = (0..d).map(|j| field[[5, j]] as u8).collect();
    let ubin: Vec<u8> = (0..d).map(|j| field[[6, j]] as u8).collect();

    let mut chrom = Array2::<f64>::zeros((n_ind, total_bits));

    for i in 0..n_ind {
        let mut col_start = 0;
        for j in 0..d {
            let l = lens[j];
            let span = ub[j] - lb[j];
            let denom = if lbin[j] == 1 && ubin[j] == 1 {
                if l > 0 { (1u64 << l) - 1 } else { 1 }
            } else {
                1u64 << l
            };

            let normalized = if span.abs() > 1e-12 {
                ((phen_arr[[i, j]] - lb[j]) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let max_val = if l > 0 { (1u64 << l) - 1 } else { 0 };
            let val_int = ((normalized * denom as f64).round() as u64).min(max_val);

            // Binary bits
            let mut bin_bits = vec![0u64; l];
            for k in 0..l {
                bin_bits[l - 1 - k] = (val_int >> k) & 1;
            }

            // Convert to Gray if needed: g = b ^ (b >> 1)
            let out_bits = if codes[j] == 1 {
                let mut gray = vec![0u64; l];
                if l > 0 {
                    gray[0] = bin_bits[0];
                    for k in 1..l {
                        gray[k] = bin_bits[k] ^ bin_bits[k - 1];
                    }
                }
                gray
            } else {
                bin_bits
            };

            for k in 0..l {
                chrom[[i, col_start + k]] = out_bits[k] as f64;
            }

            col_start += l;
        }
    }

    Ok(chrom.into_pyarray(py))
}

/// Boundary fix
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, loop_fix=false))]
pub fn boundfix<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    loop_fix: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if encoding != "RI" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boundfix only supports 'RI' encoding.",
        ));
    }
    let mut chrom_arr = crate::utils::to_f64_array2(old_chrom)?;
    let field = crate::utils::to_f64_array2(field_dr)?;
    let n_ind = chrom_arr.shape()[0];
    let d = field.shape()[1];

    let lb = field.row(0);
    let ub = field.row(1);
    let var_types = field.row(2);

    for i in 0..n_ind {
        for j in 0..d {
            let l = lb[j];
            let u = ub[j];
            let is_discrete = var_types[j] == 1.0;
            let mut val = chrom_arr[[i, j]];

            if loop_fix {
                let span = u - l;
                if span > 0.0 {
                    let mut offset = (val - l) % span;
                    if offset < 0.0 {
                        offset += span;
                    }
                    val = l + offset;
                } else {
                    val = l;
                }
            } else {
                val = val.clamp(l, u);
            }

            if is_discrete {
                val = val.round();
            }
            chrom_arr[[i, j]] = val;
        }
    }

    Ok(chrom_arr.into_pyarray(py))
}

/// Create random binary population
#[pyfunction]
pub fn crtbp<'py>(py: Python<'py>, n_ind: usize, lind: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut rng = rand::thread_rng();
    let mut chrom = Array2::<f64>::zeros((n_ind, lind));
    for i in 0..n_ind {
        for j in 0..lind {
            chrom[[i, j]] = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
        }
    }
    Ok(chrom.into_pyarray(py))
}

/// Create random integer population
#[pyfunction]
pub fn crtip<'py>(
    py: Python<'py>,
    n_ind: usize,
    field_dr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut rng = rand::thread_rng();
    let field = crate::utils::to_f64_array2(field_dr)?;
    let d = field.shape()[1];
    let lb = field.row(0);
    let ub = field.row(1);

    let mut chrom = Array2::<f64>::zeros((n_ind, d));
    for i in 0..n_ind {
        for j in 0..d {
            let low = lb[j].round() as i64;
            let high = ub[j].round() as i64;
            let val = if low <= high {
                rng.gen_range(low..=high)
            } else {
                low
            };
            chrom[[i, j]] = val as f64;
        }
    }
    Ok(chrom.into_pyarray(py))
}

/// Create random real/integer population
#[pyfunction]
pub fn crtrp<'py>(
    py: Python<'py>,
    n_ind: usize,
    field_dr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut rng = rand::thread_rng();
    let field = crate::utils::to_f64_array2(field_dr)?;
    let d = field.shape()[1];
    let lb = field.row(0);
    let ub = field.row(1);
    let var_types = field.row(2);

    let mut chrom = Array2::<f64>::zeros((n_ind, d));
    for i in 0..n_ind {
        for j in 0..d {
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
    Ok(chrom.into_pyarray(py))
}

/// Create random permutation population
#[pyfunction]
pub fn crtpp<'py>(
    py: Python<'py>,
    n_ind: usize,
    field_dr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut rng = rand::thread_rng();
    let field = crate::utils::to_f64_array2(field_dr)?;
    let d = field.shape()[1];
    let low = field[[0, 0]].round() as i64;
    let high = field[[1, 0]].round() as i64;

    let base: Vec<f64> = (low..=high).map(|v| v as f64).collect();
    if base.len() < d {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Permutation range (ub - lb + 1) must be >= chromosome length.",
        ));
    }

    let mut chrom = Array2::<f64>::zeros((n_ind, d));
    for i in 0..n_ind {
        let mut perm = base.clone();
        perm.shuffle(&mut rng);
        for j in 0..d {
            chrom[[i, j]] = perm[j];
        }
    }
    Ok(chrom.into_pyarray(py))
}

/// Dispatcher to create population chromosome according to encoding
#[pyfunction]
pub fn crtpc<'py>(
    py: Python<'py>,
    encoding: &str,
    n_ind: usize,
    field: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    match encoding {
        "BG" => {
            let arr = field.extract::<PyReadonlyArray2<f64>>()?;
            let lens_sum: f64 = arr.as_array().row(0).iter().sum();
            let res = crtbp(py, n_ind, lens_sum.round() as usize)?;
            Ok(res.into_any().unbind())
        }
        "RI" => {
            let res = crtrp(py, n_ind, field)?;
            Ok(res.into_any().unbind())
        }
        "P" => {
            let res = crtpp(py, n_ind, field)?;
            Ok(res.into_any().unbind())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Encoding must be 'BG', 'RI', or 'P', got {}",
            encoding
        ))),
    }
}

#[pyfunction]
pub fn crtri<'py>(
    py: Python<'py>,
    n_ind: usize,
    field_dr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    crtrp(py, n_ind, field_dr)
}

/// Create uniform reference points on unit simplex using Das & Dennis recursion
#[pyfunction]
pub fn crtup<'py>(
    py: Python<'py>,
    m: usize,
    n: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    if m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("M must be > 0"));
    }
    if m == 1 {
        let arr = Array2::from_elem((1, 1), 1.0);
        return Ok((arr.into_pyarray(py), 1));
    }

    // Find partition number H such that nCr(H + M - 1, M - 1) is closest to N
    let n_cr = |n_val: usize, r: usize| -> usize {
        if r > n_val { return 0; }
        let mut res = 1usize;
        for i in 1..=r {
            res = res * (n_val - r + i) / i;
        }
        res
    };

    let mut h = 1usize;
    while n_cr(h + m - 1, m - 1) < n {
        h += 1;
    }
    // Check if h or h-1 is closer
    if h > 1 {
        let count_h = n_cr(h + m - 1, m - 1);
        let count_prev = n_cr(h + m - 2, m - 1);
        if (n as isize - count_prev as isize).abs() < (n as isize - count_h as isize).abs() {
            h -= 1;
        }
    }

    fn generate_simplex_points(m: usize, remaining: usize) -> Vec<Vec<usize>> {
        if m == 1 {
            return vec![vec![remaining]];
        }
        let mut points = Vec::new();
        for i in 0..=remaining {
            let sub = generate_simplex_points(m - 1, remaining - i);
            for mut s in sub {
                s.insert(0, i);
                points.push(s);
            }
        }
        points
    }

    let partitions = generate_simplex_points(m, h);
    let total_points = partitions.len();
    let mut out = Array2::<f64>::zeros((total_points, m));
    let inv_h = 1.0 / (h as f64);

    for (i, p) in partitions.iter().enumerate() {
        for j in 0..m {
            out[[i, j]] = (p[j] as f64) * inv_h;
        }
    }

    Ok((out.into_pyarray(py), total_points))
}

/// Create grid points
#[pyfunction]
pub fn crtgp<'py>(py: Python<'py>, m: usize, n: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (pts, _) = crtup(py, m, n)?;
    Ok(pts)
}

/// Create inverted / boundary points
#[pyfunction]
pub fn crtidp<'py>(py: Python<'py>, m: usize, n: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (pts, actual_n) = crtup(py, m, n)?;
    let pts_ro = pts.readonly();
    let pts_arr = pts_ro.as_array();
    let mut inv = Array2::<f64>::zeros((actual_n, m));
    for i in 0..actual_n {
        for j in 0..m {
            inv[[i, j]] = (1.0 - pts_arr[[i, j]]).max(0.0);
        }
    }
    Ok(inv.into_pyarray(py))
}

