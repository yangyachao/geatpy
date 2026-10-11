use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

/// Per-variable list argument (precisions, codes, scales) with a default for missing entries.
fn list_arg(obj: Option<&Bound<'_, PyAny>>, d: usize, default: f64) -> PyResult<Vec<f64>> {
    match obj {
        Some(o) if !o.is_none() => {
            let v = crate::utils::to_f64_array2(o)?
                .iter()
                .cloned()
                .collect::<Vec<f64>>();
            Ok((0..d)
                .map(|j| v.get(j).copied().unwrap_or(default))
                .collect())
        }
        _ => Ok(vec![default; d]),
    }
}

/// Build FieldDR ('RI', 'P') or FieldD ('BG'). Integer variables are shrunk inwards and rounded,
/// excluded continuous bounds of 'RI' are shrunk by 0.1^precision (precision defaults to 4).
#[pyfunction]
#[pyo3(signature = (encoding, var_types, ranges, borders=None, precisions=None, codes=None, scales=None))]
#[allow(clippy::too_many_arguments)]
pub fn crtfld<'py>(
    py: Python<'py>,
    encoding: &str,
    var_types: &Bound<'py, PyAny>,
    ranges: &Bound<'py, PyAny>,
    borders: Option<&Bound<'py, PyAny>>,
    precisions: Option<&Bound<'py, PyAny>>,
    codes: Option<&Bound<'py, PyAny>>,
    scales: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !["BG", "RI", "P"].contains(&encoding) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in crtfld: Encoding must be 'BG', 'RI' or 'P', got '{}'.",
            encoding
        )));
    }
    let ranges = crate::utils::to_f64_array2(ranges)?;
    let d = ranges.shape()[1];
    let mut vt: Vec<f64> = crate::utils::to_f64_array2(var_types)?
        .iter()
        .cloned()
        .collect();
    if vt.len() != d || ranges.shape()[0] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtfld: ranges must be a 2 x Dim matrix matching varTypes.",
        ));
    }
    if encoding == "P" {
        vt = vec![1.0; d];
    }
    let (mut lbin, mut ubin) = match borders {
        Some(b) if !b.is_none() => {
            let b = crate::utils::to_f64_array2(b)?;
            (b.row(0).to_vec(), b.row(1).to_vec())
        }
        _ => (vec![1.0; d], vec![1.0; d]),
    };
    let prec = list_arg(precisions, d, 4.0)?;
    let mut lb = ranges.row(0).to_vec();
    let mut ub = ranges.row(1).to_vec();
    for j in 0..d {
        if lb[j] > ub[j] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in crtfld: The upper bound must not be smaller than the lower bound. (决策变量范围的上界必须不小于下界。)",
            ));
        }
        if vt[j] == 1.0 {
            lb[j] = if lbin[j] == 0.0 {
                lb[j].floor() + 1.0
            } else {
                lb[j].ceil()
            };
            ub[j] = if ubin[j] == 0.0 {
                ub[j].ceil() - 1.0
            } else {
                ub[j].floor()
            };
            lbin[j] = 1.0;
            ubin[j] = 1.0;
        } else if encoding != "BG" {
            let shrink = 0.1f64.powf(prec[j]);
            if lbin[j] == 0.0 {
                lb[j] += shrink;
            }
            if ubin[j] == 0.0 {
                ub[j] -= shrink;
            }
        }
    }
    // After the open borders have been shrunk the range must still be valid (geatpy 2.7.0 checks):
    // non-empty for 'RI'/'P', at least two values for 'BG'.
    for j in 0..d {
        if (encoding == "BG" && ub[j] <= lb[j]) || ub[j] < lb[j] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                if encoding == "BG" {
                    "error in crtfld: The upper bound must be bigger than the lower bound when using binary/gray coding. (使用二进制/格雷编码时，变量范围的上界必须大于下界。)"
                } else {
                    "error in crtfld: The upper bound must be or bigger than the lower bound after handling the borders. (处理边界后，变量范围的上界必须不小于下界。)"
                },
            ));
        }
    }
    if encoding != "BG" {
        let mut res = Array2::<f64>::zeros((3, d));
        for j in 0..d {
            res[[0, j]] = lb[j];
            res[[1, j]] = ub[j];
            res[[2, j]] = vt[j];
        }
        return Ok(res.into_pyarray(py));
    }
    let codes = list_arg(codes, d, 0.0)?;
    let scales = list_arg(scales, d, 0.0)?;
    let mut res = Array2::<f64>::zeros((8, d));
    for j in 0..d {
        let count = if vt[j] == 1.0 {
            ub[j] - lb[j] + 1.0
        } else {
            (ub[j] - lb[j]) * 10f64.powf(prec[j]) + 1.0
        };
        res[[0, j]] = count.log2().ceil().max(1.0);
        if res[[0, j]] > 31.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in crtfld: The length of the code should not be bigger than 31; lower the precision or narrow the range. (编码长度不能超过31，请降低精度或缩小范围。)",
            ));
        }
        res[[1, j]] = lb[j];
        res[[2, j]] = ub[j];
        res[[3, j]] = codes[j];
        res[[4, j]] = scales[j];
        res[[5, j]] = lbin[j];
        res[[6, j]] = ubin[j];
        res[[7, j]] = vt[j];
    }
    Ok(res.into_pyarray(py))
}

struct FieldD {
    lens: Vec<usize>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    gray: Vec<bool>,
    lbin: Vec<bool>,
    ubin: Vec<bool>,
    discrete: Vec<bool>,
}

fn parse_field_d(obj: &Bound<'_, PyAny>, lind: Option<usize>) -> PyResult<FieldD> {
    let f = crate::utils::to_f64_array2(obj)?;
    if f.shape()[0] < 8 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error: FieldD must have 8 rows.",
        ));
    }
    let d = f.shape()[1];
    let fd = FieldD {
        lens: (0..d).map(|j| f[[0, j]] as usize).collect(),
        lb: f.row(1).to_vec(),
        ub: f.row(2).to_vec(),
        gray: (0..d).map(|j| f[[3, j]] == 1.0).collect(),
        lbin: (0..d).map(|j| f[[5, j]] == 1.0).collect(),
        ubin: (0..d).map(|j| f[[6, j]] == 1.0).collect(),
        discrete: (0..d).map(|j| f[[7, j]] == 1.0).collect(),
    };
    if let Some(l) = lind {
        if fd.lens.iter().sum::<usize>() != l {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error: The chromosome length does not match the sum of the lens in FieldD. (染色体长度与FieldD不匹配。)",
            ));
        }
    }
    Ok(fd)
}

/// Integer value of one binary / Gray-coded gene segment.
fn segment_value(bits: impl Iterator<Item = f64>, gray: bool) -> u64 {
    let mut v = 0u64;
    let mut parity = 0u64;
    for b in bits {
        let bit = (b > 0.5) as u64;
        let bin = if gray {
            parity ^= bit;
            parity
        } else {
            bit
        };
        v = (v << 1) | bin;
    }
    v
}

enum Decode {
    Mixed,
    Real,
    Int,
}

fn decode_bg<'py>(
    py: Python<'py>,
    chrom: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    mode: Decode,
) -> PyResult<PyObject> {
    let c = crate::utils::to_f64_array2(chrom)?;
    let (n, lind) = (c.shape()[0], c.shape()[1]);
    let f = parse_field_d(field, Some(lind))?;
    let d = f.lens.len();
    let mut phen = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let mut start = 0;
        for j in 0..d {
            let l = f.lens[j];
            let k = segment_value((start..start + l).map(|t| c[[i, t]]), f.gray[j]) as f64;
            start += l;
            let full = 2f64.powi(l as i32);
            let span = (f.ub[j] - f.lb[j]).abs();
            phen[[i, j]] = match mode {
                Decode::Int => {
                    let (lb, ub) = (f.lb[j] as i64, f.ub[j] as i64);
                    (lb + (k as i64 * (ub - lb).abs()) / (full as i64 - 1)) as f64
                }
                Decode::Mixed if f.discrete[j] => f.lb[j] + (k * span / (full - 1.0) + 0.5).trunc(),
                _ => {
                    let offset = if f.lbin[j] { 0.0 } else { 1.0 };
                    let denom = full + 1.0 - f.ubin[j] as u8 as f64 - f.lbin[j] as u8 as f64;
                    f.lb[j] + (k + offset) * span / denom
                }
            };
        }
    }
    match mode {
        Decode::Mixed => crate::utils::ri_out(py, phen, &f.discrete),
        _ => crate::utils::chrom_out(py, phen, matches!(mode, Decode::Int)),
    }
}

/// Decode binary/Gray chromosomes into real and integer variables.
#[pyfunction]
#[pyo3(signature = (chrom, field_d, parallel=None))]
pub fn bs2ri<'py>(
    py: Python<'py>,
    chrom: &Bound<'py, PyAny>,
    field_d: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    decode_bg(py, chrom, field_d, Decode::Mixed)
}

/// Decode every variable as a real number.
#[pyfunction]
#[pyo3(signature = (chrom, field_d, parallel=None))]
pub fn bs2real<'py>(
    py: Python<'py>,
    chrom: &Bound<'py, PyAny>,
    field_d: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    decode_bg(py, chrom, field_d, Decode::Real)
}

/// Decode every variable as an integer (integer arithmetic on truncated bounds).
#[pyfunction]
#[pyo3(signature = (chrom, field_d, parallel=None))]
pub fn bs2int<'py>(
    py: Python<'py>,
    chrom: &Bound<'py, PyAny>,
    field_d: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    decode_bg(py, chrom, field_d, Decode::Int)
}

/// Encode real/integer variables into binary/Gray chromosomes (inverse of bs2ri).
#[pyfunction]
#[pyo3(signature = (phen, field_d, parallel=None))]
pub fn ri2bs<'py>(
    py: Python<'py>,
    phen: &Bound<'py, PyAny>,
    field_d: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let p = crate::utils::to_f64_array2(phen)?;
    let f = parse_field_d(field_d, None)?;
    let (n, d) = (p.shape()[0], f.lens.len());
    if p.shape()[1] != d {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in ri2bs: Phen and FieldD disagree in number of variables.",
        ));
    }
    let lind: usize = f.lens.iter().sum();
    let mut chrom = Array2::<f64>::zeros((n, lind));
    for i in 0..n {
        let mut start = 0;
        for j in 0..d {
            let l = f.lens[j];
            let full = 2f64.powi(l as i32);
            let span = (f.ub[j] - f.lb[j]).abs();
            let x = p[[i, j]] - f.lb[j];
            let k = if f.discrete[j] {
                ((full - 1.0) * x / span + 0.5) as i64
            } else {
                let offset = if f.lbin[j] { 0.0 } else { 1.0 };
                let denom = full + 1.0 - f.ubin[j] as u8 as f64 - f.lbin[j] as u8 as f64;
                (denom * x / span - offset) as i64
            };
            let mut prev = 0i64;
            for t in 0..l {
                let bit = (k / (1i64 << (l - 1 - t))) % 2;
                chrom[[i, start + t]] = if f.gray[j] {
                    (prev ^ bit) as f64
                } else {
                    bit as f64
                };
                prev = bit;
            }
            start += l;
        }
    }
    crate::utils::chrom_out(py, chrom, true)
}

/// Repair out-of-range genes of a real/integer population (FixType 1: truncate, 2: wrap, 3: reflect,
/// 4: random); integer variables are rounded afterwards.
#[pyfunction]
#[pyo3(signature = (encoding, old_chrom, field_dr, fix_type=None, parallel=None))]
pub fn boundfix<'py>(
    py: Python<'py>,
    encoding: &str,
    old_chrom: &Bound<'py, PyAny>,
    field_dr: &Bound<'py, PyAny>,
    fix_type: Option<&Bound<'py, PyAny>>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    if encoding != "RI" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in boundfix: The encoding must be 'RI'.",
        ));
    }
    let mut chrom = crate::utils::to_f64_array2(old_chrom)?;
    let d = chrom.shape()[1];
    let b = crate::utils::bounds(&crate::utils::to_f64_array2(field_dr)?, d, true)?;
    let fix = crate::utils::fix_type(fix_type, "boundfix")?;
    let mut rng = rand::thread_rng();
    for mut row in chrom.rows_mut() {
        for j in 0..d {
            let x = crate::utils::fix_value(row[j], b.lb[j], b.ub[j], b.span[j], fix, &mut rng);
            row[j] = if b.discrete[j] { x.round() } else { x };
        }
    }
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Random binary population (int64 matrix of 0/1).
#[pyfunction]
#[pyo3(signature = (nind, lind, params2=None))]
pub fn crtbp<'py>(
    py: Python<'py>,
    nind: usize,
    lind: usize,
    params2: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = params2;
    let mut rng = rand::thread_rng();
    let chrom = Array2::from_shape_fn((nind, lind), |_| if rng.gen::<bool>() { 1.0 } else { 0.0 });
    crate::utils::chrom_out(py, chrom, true)
}

/// FieldDR bounds (validated); `discrete` honours the varTypes row.
fn field_dr(field: &Bound<'_, PyAny>) -> PyResult<crate::utils::Bounds> {
    let f = crate::utils::to_f64_array2(field)?;
    let d = f.shape()[1];
    if f.shape()[0] < 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error: FieldDR must be a matrix whose first two rows are the lower and upper bounds.",
        ));
    }
    crate::utils::bounds(&f, d, false)
}

fn uniform_int<R: Rng>(lb: f64, ub: f64, rng: &mut R) -> f64 {
    let (lo, hi) = (lb.ceil() as i64, ub.floor() as i64);
    if lo <= hi {
        rng.gen_range(lo..=hi) as f64
    } else {
        lo as f64
    }
}

fn uniform_real<R: Rng>(lb: f64, ub: f64, rng: &mut R) -> f64 {
    if lb < ub {
        rng.gen_range(lb..ub)
    } else {
        lb
    }
}

/// Random integer population within [lb, ub] (varTypes ignored).
#[pyfunction]
#[pyo3(signature = (nind, field_dr, parallel=None))]
pub fn crtip<'py>(
    py: Python<'py>,
    nind: usize,
    field_dr: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let b = self::field_dr(field_dr)?;
    let mut rng = rand::thread_rng();
    let chrom = Array2::from_shape_fn((nind, b.lb.len()), |(_, j)| {
        uniform_int(b.lb[j], b.ub[j], &mut rng)
    });
    crate::utils::chrom_out(py, chrom, true)
}

/// Random real population, uniform in [lb, ub) (varTypes ignored).
#[pyfunction]
#[pyo3(signature = (nind, field_dr, parallel=None))]
pub fn crtrp<'py>(
    py: Python<'py>,
    nind: usize,
    field_dr: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let b = self::field_dr(field_dr)?;
    let mut rng = rand::thread_rng();
    let chrom = Array2::from_shape_fn((nind, b.lb.len()), |(_, j)| {
        uniform_real(b.lb[j], b.ub[j], &mut rng)
    });
    crate::utils::chrom_out(py, chrom, false)
}

/// Random real/integer population: integer variables (varTypes 1) uniform over the integers in range.
#[pyfunction]
#[pyo3(signature = (nind, field_dr, parallel=None))]
pub fn crtri<'py>(
    py: Python<'py>,
    nind: usize,
    field_dr: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let b = self::field_dr(field_dr)?;
    let mut rng = rand::thread_rng();
    let chrom = Array2::from_shape_fn((nind, b.lb.len()), |(_, j)| {
        if b.discrete[j] {
            uniform_int(b.lb[j], b.ub[j], &mut rng)
        } else {
            uniform_real(b.lb[j], b.ub[j], &mut rng)
        }
    });
    crate::utils::ri_out(py, chrom, &b.discrete)
}

/// Random permutation population: each row holds Lind distinct values from {Lb, ..., Ub}.
#[pyfunction]
#[pyo3(signature = (nind, field_dr, parallel=None))]
pub fn crtpp<'py>(
    py: Python<'py>,
    nind: usize,
    field_dr: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let b = self::field_dr(field_dr)?;
    let lind = b.lb.len();
    if lind == 0 {
        return crate::utils::chrom_out(py, Array2::<f64>::zeros((nind, 0)), true);
    }
    if b.lb.iter().any(|&v| v != b.lb[0]) || b.ub.iter().any(|&v| v != b.ub[0]) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtpp: Invalid syntax of FieldDR: every column must share the same lower and upper bound. (FieldDR的格式错误。)",
        ));
    }
    let (lo, hi) = (b.lb[0].round() as i64, b.ub[0].round() as i64);
    let base: Vec<f64> = (lo..=hi).map(|v| v as f64).collect();
    if base.len() < lind {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtpp: Ub - Lb + 1 must not be smaller than the chromosome length.",
        ));
    }
    let mut rng = rand::thread_rng();
    let mut chrom = Array2::<f64>::zeros((nind, lind));
    for mut row in chrom.rows_mut() {
        let picked: Vec<f64> = base.choose_multiple(&mut rng, lind).cloned().collect();
        for (x, v) in row.iter_mut().zip(picked) {
            *x = v;
        }
    }
    crate::utils::chrom_out(py, chrom, true)
}

/// Create a population chromosome matrix for the given encoding ('BG', 'RI' or 'P').
#[pyfunction]
#[pyo3(signature = (encoding, nind, field, parallel=None))]
pub fn crtpc<'py>(
    py: Python<'py>,
    encoding: &str,
    nind: usize,
    field: &Bound<'py, PyAny>,
    parallel: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    match encoding {
        "BG" => {
            let f = crate::utils::to_f64_array2(field)?;
            if f.shape()[0] < 8 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error in crtpc: FieldD must have 8 rows for 'BG'.",
                ));
            }
            let lind = f.row(0).iter().sum::<f64>().round() as usize;
            crtbp(py, nind, lind, None)
        }
        "RI" => crtri(py, nind, field, parallel),
        "P" => crtpp(py, nind, field, parallel),
        _ => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in crtpc: Encoding must be 'BG', 'RI' or 'P', got '{}'.",
            encoding
        ))),
    }
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (1..=k).fold(1usize, |acc, i| acc * (n - k + i) / i)
}

/// All compositions of `h` into `m` non-negative parts, divided by h (Das & Dennis lattice).
fn simplex_lattice(m: usize, h: usize) -> Vec<Vec<f64>> {
    fn rec(m: usize, remaining: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if m == 1 {
            prefix.push(remaining);
            out.push(prefix.clone());
            prefix.pop();
            return;
        }
        for i in 0..=remaining {
            prefix.push(i);
            rec(m - 1, remaining - i, prefix, out);
            prefix.pop();
        }
    }
    let mut out = Vec::new();
    rec(m, h, &mut Vec::new(), &mut out);
    out.into_iter()
        .map(|p| p.into_iter().map(|x| x as f64 / h.max(1) as f64).collect())
        .collect()
}

/// Uniformly distributed reference points on the unit simplex (two-layer for many objectives).
/// With NUM the result never exceeds NUM points; with Div every axis is split into Div parts.
#[pyfunction]
#[pyo3(signature = (dim, num=None, div=None))]
pub fn crtup<'py>(
    py: Python<'py>,
    dim: usize,
    num: Option<usize>,
    div: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    if dim == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtup: Dim must be positive.",
        ));
    }
    if num.is_some() == div.is_some() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtup: exactly one of NUM and Div must be given. (NUM和Div必须且只能传入一个。)",
        ));
    }
    if dim == 1 {
        // A single objective: NUM (or Div + 1) evenly spaced points on [0, 1], as in geatpy 2.7.0.
        let count = num.unwrap_or_else(|| div.unwrap() + 1);
        let pts: Vec<f64> = (0..count)
            .map(|i| {
                if count > 1 {
                    i as f64 / (count - 1) as f64
                } else {
                    0.0
                }
            })
            .collect();
        let arr = Array2::from_shape_vec((count, 1), pts).unwrap();
        return Ok((arr.into_pyarray(py), count));
    }
    if let Some(n) = num {
        if n < dim {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "error in crtup: NUM must not be smaller than Dim. (NUM必须不小于Dim。)",
            ));
        }
    }
    let mut pts: Vec<Vec<f64>> = match (num, div) {
        (None, Some(h)) => simplex_lattice(dim, h),
        (Some(n), None) => {
            let mut h1 = 1;
            while n_choose_k(h1 + dim, dim - 1) <= n {
                h1 += 1;
            }
            let mut w = simplex_lattice(dim, h1);
            if h1 < dim {
                let base = n_choose_k(h1 + dim - 1, dim - 1);
                let mut h2 = 0;
                while base + n_choose_k(h2 + dim, dim - 1) <= n {
                    h2 += 1;
                }
                if h2 > 0 {
                    let shift = 1.0 / (2.0 * dim as f64);
                    w.extend(
                        simplex_lattice(dim, h2)
                            .into_iter()
                            .map(|p| p.into_iter().map(|x| x / 2.0 + shift).collect()),
                    );
                }
            }
            w
        }
        _ => unreachable!(),
    };
    for p in pts.iter_mut() {
        for x in p.iter_mut() {
            *x = x.max(1e-6);
        }
    }
    let n = pts.len();
    let arr = Array2::from_shape_vec((n, dim), pts.into_iter().flatten().collect()).unwrap();
    Ok((arr.into_pyarray(py), n))
}

/// Uniform grid in the unit hypercube with floor(NUM^(1/Dim))^Dim points (first axis fastest).
#[pyfunction]
pub fn crtgp<'py>(
    py: Python<'py>,
    dim: usize,
    num: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    if dim == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtgp: Dim must be positive.",
        ));
    }
    let mut k = (num as f64).powf(1.0 / dim as f64).floor() as usize;
    while (k + 1).pow(dim as u32) <= num {
        k += 1;
    }
    while k > 0 && k.pow(dim as u32) > num {
        k -= 1;
    }
    if k < 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in crtgp: NUM is too small to place at least two grid points on every axis. (NUM过小，每一维至少需要2个网格点。)",
        ));
    }
    let total = k.pow(dim as u32);
    let mut arr = Array2::<f64>::zeros((total, dim));
    // Same order as np.meshgrid(*axes) (indexing='xy') flattened: axes ordered (1, 0, 2, 3, ...),
    // the last one varying fastest.
    let mut axis_order: Vec<usize> = (0..dim).collect();
    if dim >= 2 {
        axis_order.swap(0, 1);
    }
    for idx in 0..total {
        let mut rest = idx;
        for &axis in axis_order.iter().rev() {
            let c = rest % k;
            rest /= k;
            arr[[idx, axis]] = if k > 1 {
                c as f64 / (k - 1) as f64
            } else {
                0.0
            };
        }
    }
    Ok((arr.into_pyarray(py), total))
}
