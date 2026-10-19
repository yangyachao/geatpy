//! Recombination and crossover operators.
//!
//! Pairing follows geatpy: with Half_N False/None the first half of the population is paired with
//! the second half (individual i with i + N/2) and both offspring are returned in place (an odd last
//! individual is copied); Half_N True returns only the first offspring of every pair; a positive
//! integer Half_N switches to global mode, producing that many offspring from random parents.

use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::utils::{chrom_out, is_integer_like, to_f64_array1, to_f64_array2};

type Opt<'a, 'py> = Option<&'a Bound<'py, PyAny>>;

#[derive(Clone, Copy)]
enum Pairing {
    Full,
    Half,
    Global(usize),
}

fn pairing(half_n: Opt, n: usize, name: &str) -> PyResult<Pairing> {
    match half_n {
        None => Ok(Pairing::Full),
        Some(o) if o.is_none() => Ok(Pairing::Full),
        Some(o) => {
            let type_name = o.get_type().name()?.to_string();
            if o.is_instance_of::<pyo3::types::PyBool>()
                || type_name == "bool_"
                || type_name == "bool"
            {
                return Ok(if o.is_truthy()? {
                    Pairing::Half
                } else {
                    Pairing::Full
                });
            }
            // a positive int (geatpy 2.7.0 rejects floats such as 3.0)
            let k: f64 = if o.is_instance_of::<pyo3::types::PyFloat>() {
                0.0
            } else {
                o.extract()?
            };
            if k < 1.0 || k.fract() != 0.0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "error in {}: Half_N error. (传入参数Half_N有误，详见API文档。)",
                    name
                )));
            }
            Ok(Pairing::Global((k as usize).min(n)))
        }
    }
}

/// Crossover probability `XOVR` (or `RecOpt` for recint/reclin/recdis), which must lie in [0, 1].
fn prob(obj: Opt, default: f64, name: &str) -> PyResult<f64> {
    let p: f64 = match obj {
        Some(o) if !o.is_none() => o.extract()?,
        _ => default,
    };
    if !(0.0..=1.0).contains(&p) {
        let msg = if matches!(name, "recint" | "reclin" | "recdis") {
            "RecOpt must be in [0,1]. (重组概率RecOpt必须在[0, 1]之间。)"
        } else {
            "XOVR must be in [0,1]. (交叉概率XOVR必须在[0, 1]之间。)"
        };
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: {}",
            name, msg
        )));
    }
    Ok(p)
}

/// Non-negative scalar or per-gene parameter (recsbx `n`, recndx `A`).
fn non_negative(obj: Opt, l: usize, default: f64, op: &str, name: &str) -> PyResult<Vec<f64>> {
    let v = crate::utils::per_gene(obj, l, default, name)?;
    if v.iter().any(|&x| x < 0.0 || x.is_nan()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: {} must be or bigger than 0. (当输入参数{}为一个数时，它必须不小于0。)",
            op, name, name
        )));
    }
    Ok(v)
}

/// Element type of a permutation chromosome: integer arrays are processed as i32 without a round
/// trip through f64.
trait Gene: Copy + PartialEq + PartialOrd {
    /// The gene as an integer, or None when it is not a whole number.
    fn whole(self) -> Option<i64>;
}

impl Gene for f64 {
    fn whole(self) -> Option<i64> {
        (self.is_finite() && self.fract() == 0.0 && self.abs() < 9e15).then_some(self as i64)
    }
}

impl Gene for i32 {
    fn whole(self) -> Option<i64> {
        Some(self as i64)
    }
}

/// geatpy 2.7.0 checks that every row of a permutation chromosome holds distinct genes.
fn check_permutation<T: Gene>(chrom: &Array2<T>, name: &str) -> PyResult<()> {
    let err = || {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {}: Chromosome error. (染色体不满足排列编码规则。)",
            name
        ))
    };
    let mut seen = Vec::new();
    for row in chrom.rows() {
        let owned;
        let row: &[T] = match row.as_slice() {
            Some(r) => r,
            None => {
                owned = row.to_vec();
                &owned
            }
        };
        if let Some(index) = DenseIndex::new(row, &[]) {
            // small integer values (the usual 0..L-1 / 1..L): a bitmap finds duplicates in O(L)
            seen.clear();
            seen.resize(index.size, false);
            for &v in row {
                let k = index.key(v);
                if seen[k] {
                    return Err(err());
                }
                seen[k] = true;
            }
        } else {
            let mut v = row.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if v.windows(2).any(|w| w[0] == w[1]) {
                return Err(err());
            }
        }
    }
    Ok(())
}

/// Maps the values of permutation rows to dense slots when they are all integers within a small
/// range, so positions can be looked up in O(1) instead of searching the row.
struct DenseIndex {
    min: i64,
    size: usize,
}

impl DenseIndex {
    fn new<T: Gene>(a: &[T], b: &[T]) -> Option<Self> {
        let (mut min, mut max) = (i64::MAX, i64::MIN);
        for &v in a.iter().chain(b) {
            let v = v.whole()?;
            min = min.min(v);
            max = max.max(v);
        }
        let size = if a.is_empty() && b.is_empty() {
            0
        } else {
            (max - min) as usize + 1
        };
        (size <= 4 * (a.len() + b.len()) + 16).then_some(DenseIndex { min, size })
    }

    /// Slot of a gene seen by `new` (always a whole number in range).
    fn key<T: Gene>(&self, v: T) -> usize {
        (v.whole().unwrap_or(self.min) - self.min) as usize
    }
}

fn random_pair<R: Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    let a = rng.gen_range(0..n);
    let b = if n > 1 {
        (a + 1 + rng.gen_range(0..n - 1)) % n
    } else {
        a
    };
    (a, b)
}

/// Length of an exponential-crossover section: geometric with ratio pc, at least 1, at most l.
fn exp_len<R: Rng>(pc: f64, l: usize, rng: &mut R) -> usize {
    if pc >= 1.0 {
        return l;
    }
    if pc <= 0.0 {
        return 1;
    }
    let u: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    ((u.ln() / pc.ln()).ceil().max(1.0) as usize).min(l)
}

#[derive(Clone, Copy, PartialEq)]
enum CopyOp {
    SinglePoint,
    DoublePoint,
    Uniform,
    Shuffle,
    ShuffleExp,
    Exponential,
    Binomial,
    Discrete,
}

impl CopyOp {
    /// Whether the operator first decides, with probability pc, if a pair crosses at all.
    fn pair_level(self) -> bool {
        matches!(
            self,
            CopyOp::SinglePoint | CopyOp::DoublePoint | CopyOp::Shuffle
        )
    }

    /// The gene range a single- or two-point crossover exchanges, drawn after the pair-level
    /// probability check (None for other operators, or when the chromosome is too short).
    fn point_range<R: Rng>(self, l: usize, rng: &mut R) -> Option<std::ops::Range<usize>> {
        match self {
            CopyOp::SinglePoint if l >= 2 => Some(rng.gen_range(0..l - 1) + 1..l),
            CopyOp::DoublePoint if l >= 1 => {
                let (mut a, mut b) = (rng.gen_range(0..l), rng.gen_range(0..l));
                if b < a {
                    std::mem::swap(&mut a, &mut b);
                }
                Some(a..b + 1)
            }
            _ => None,
        }
    }

    /// Genes the first offspring takes from the partner (None: the pair does not cross).
    fn mask<R: Rng>(self, l: usize, pc: f64, rng: &mut R, forced: bool) -> Option<Vec<bool>> {
        if l == 0 {
            return None;
        }
        if self.pair_level() && !forced && rng.gen::<f64>() >= pc {
            return None;
        }
        let mut m = vec![false; l];
        match self {
            CopyOp::SinglePoint | CopyOp::DoublePoint => {
                let r = self.point_range(l, rng)?;
                m[r].iter_mut().for_each(|x| *x = true);
            }
            CopyOp::Uniform | CopyOp::Discrete => {
                m.iter_mut().for_each(|x| *x = rng.gen::<f64>() < pc)
            }
            CopyOp::Shuffle => {
                if l < 2 {
                    return None;
                }
                let mut perm: Vec<usize> = (0..l).collect();
                perm.shuffle(rng);
                let point = rng.gen_range(0..l - 1);
                perm[point + 1..].iter().for_each(|&j| m[j] = true);
            }
            CopyOp::ShuffleExp => {
                let mut perm: Vec<usize> = (0..l).collect();
                perm.shuffle(rng);
                let start = rng.gen_range(0..l);
                let len = exp_len(pc, l, rng);
                (0..len).for_each(|t| m[perm[(start + t) % l]] = true);
            }
            CopyOp::Exponential => {
                let start = rng.gen_range(0..l);
                let len = exp_len(pc, l, rng);
                (0..len).for_each(|t| m[(start + t) % l] = true);
            }
            CopyOp::Binomial => {
                let jrand = rng.gen_range(0..l);
                for (j, x) in m.iter_mut().enumerate() {
                    *x = j == jrand || rng.gen::<f64>() < pc;
                }
            }
        }
        Some(m)
    }
}

/// For every offspring row and every (virtual) gene, the parent row it is copied from.
fn copy_sources(op: CopyOp, n: usize, l: usize, pc: f64, mode: Pairing) -> Vec<Vec<usize>> {
    let mut rng = crate::utils::rng();
    let half = n / 2;
    match mode {
        // Global uniform/discrete recombination: each (virtual) gene comes from a random individual
        // of the whole population, and the probability is not used (geatpy 2.7.0 `*_gb`).
        Pairing::Global(k) if matches!(op, CopyOp::Uniform | CopyOp::Discrete) => (0..k)
            .map(|_| (0..l).map(|_| rng.gen_range(0..n)).collect())
            .collect(),
        Pairing::Global(k) => (0..k)
            .map(|_| {
                let (a, b) = random_pair(n, &mut rng);
                match op.mask(l, pc, &mut rng, true) {
                    Some(m) => m.iter().map(|&t| if t { b } else { a }).collect(),
                    None => vec![a; l],
                }
            })
            .collect(),
        Pairing::Half => (0..half)
            .map(|i| match op.mask(l, pc, &mut rng, false) {
                Some(m) => m.iter().map(|&t| if t { i + half } else { i }).collect(),
                None => vec![i; l],
            })
            .collect(),
        Pairing::Full => {
            let mut out: Vec<Vec<usize>> = (0..n).map(|i| vec![i; l]).collect();
            for i in 0..half {
                let j = i + half;
                if let Some(m) = op.mask(l, pc, &mut rng, false) {
                    // discrete recombination draws an independent mask for the second offspring
                    let m2 = if op == CopyOp::Discrete {
                        op.mask(l, pc, &mut rng, false).unwrap()
                    } else {
                        m.clone()
                    };
                    out[i] = m.iter().map(|&t| if t { j } else { i }).collect();
                    out[j] = m2.iter().map(|&t| if t { i } else { j }).collect();
                }
            }
            out
        }
    }
}

/// Pairwise crossover (individual i with i + N/2) applied directly to the chromosome matrix: the
/// same masks as `copy_sources` for `Pairing::Full` / `Pairing::Half`, without building the
/// per-gene source table. With `half_only` only the first offspring of every pair is returned.
fn paired_crossover<T: Copy>(
    chrom: Array2<T>,
    op: CopyOp,
    pc: f64,
    half_only: bool,
    seg: &[usize],
    virtual_len: usize,
) -> Array2<T> {
    let mut rng = crate::utils::rng();
    let (n, l) = chrom.dim();
    let half = n / 2;
    // rows are crossed in place: row i only ever reads row i + N/2 and vice versa
    let mut buf = if chrom.is_standard_layout() {
        chrom.into_raw_vec_and_offset().0
    } else {
        chrom.iter().copied().collect()
    };
    let identity = seg.len() == virtual_len && seg.iter().enumerate().all(|(g, &s)| g == s);
    // per-gene mask from the (virtual) segment mask
    let per_gene = |m: Vec<bool>| -> Vec<bool> {
        if identity {
            m
        } else {
            seg.iter().map(|&s| m[s]).collect()
        }
    };
    let point_op = identity && matches!(op, CopyOp::SinglePoint | CopyOp::DoublePoint);
    for i in 0..half {
        let j = i + half;
        let (lo, hi) = buf.split_at_mut(j * l);
        let (ri, rj) = (&mut lo[i * l..(i + 1) * l], &mut hi[..l]);
        if point_op {
            // same draws as `mask`, but the exchanged genes are one contiguous slice
            if l == 0 || rng.gen::<f64>() >= pc {
                continue;
            }
            let Some(r) = op.point_range(l, &mut rng) else {
                continue;
            };
            if half_only {
                ri[r.clone()].copy_from_slice(&rj[r]);
            } else {
                ri[r.clone()].swap_with_slice(&mut rj[r]);
            }
            continue;
        }
        let Some(m) = op.mask(virtual_len, pc, &mut rng, false) else {
            continue;
        };
        // discrete recombination draws an independent mask for the second offspring
        let m2 = if !half_only && op == CopyOp::Discrete {
            op.mask(virtual_len, pc, &mut rng, false).map(per_gene)
        } else {
            None
        };
        let m = per_gene(m);
        match (half_only, m2) {
            (true, _) => {
                for ((x, &y), &t) in ri.iter_mut().zip(rj.iter()).zip(&m) {
                    *x = if t { y } else { *x };
                }
            }
            (false, None) => {
                for ((x, y), &t) in ri.iter_mut().zip(rj.iter_mut()).zip(&m) {
                    let (a, b) = if t { (*y, *x) } else { (*x, *y) };
                    (*x, *y) = (a, b);
                }
            }
            (false, Some(m2)) => {
                for (((x, y), &t), &t2) in ri.iter_mut().zip(rj.iter_mut()).zip(&m).zip(&m2) {
                    let (a, b) = (*x, *y);
                    *x = if t { b } else { a };
                    *y = if t2 { a } else { b };
                }
            }
        }
    }
    let rows = if half_only { half } else { n };
    buf.truncate(rows * l);
    Array2::from_shape_vec((rows, l), buf).expect("shape matches")
}

/// Segment index of every gene for GeneID (same ID => crossed as a whole).
fn gene_segments(gene_id: Opt, l: usize) -> PyResult<Option<(Vec<usize>, usize)>> {
    match gene_id {
        Some(g) if !g.is_none() => {
            let ids = to_f64_array1(g)?;
            if ids.len() != l {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: The length of GeneID must equal the chromosome length.",
                ));
            }
            // Segments are ordered by ID value (geatpy 2.7.0 sorts the genes by GeneID first), so
            // position-based crossovers cut between consecutive IDs, not in order of appearance.
            // geatpy 2.7.0 casts GeneID to int32 (NaN becomes 0, as numpy does on arm64)
            let ids: Vec<f64> = ids.iter().map(|&v| f64::from(v as i32)).collect();
            let mut uniq: Vec<f64> = ids.clone();
            uniq.sort_by(f64::total_cmp);
            uniq.dedup();
            let seg = ids
                .iter()
                .map(|&v| uniq.iter().position(|&s| s == v).unwrap())
                .collect();
            Ok(Some((seg, uniq.len())))
        }
        _ => Ok(None),
    }
}

/// Runs a value-copying crossover on a chromosome matrix or on a list of chromosome matrices
/// (in which case whole chromosomes are exchanged).
fn copy_crossover<'py>(
    py: Python<'py>,
    name: &str,
    op: CopyOp,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    gene_id: Opt,
) -> PyResult<PyObject> {
    if let Ok(list) = old.downcast::<PyList>() {
        let mats: Vec<(Array2<f64>, bool)> = list
            .iter()
            .map(|m| Ok((to_f64_array2(&m)?, is_integer_like(&m))))
            .collect::<PyResult<_>>()?;
        let n = mats.first().map(|m| m.0.shape()[0]).unwrap_or(0);
        let src = copy_sources(op, n, mats.len(), pc, pairing(half_n, n, name)?);
        let out = PyList::empty(py);
        for (c, (m, as_int)) in mats.iter().enumerate() {
            let mut new = Array2::<f64>::zeros((src.len(), m.shape()[1]));
            for (r, s) in src.iter().enumerate() {
                new.row_mut(r).assign(&m.row(s[c]));
            }
            out.append(chrom_out(py, new, *as_int)?)?;
        }
        return Ok(out.into_any().unbind());
    }
    let as_int = is_integer_like(old);
    let int_chrom = if as_int {
        crate::utils::to_i32_array2(old)
    } else {
        None
    };
    let chrom = match int_chrom {
        Some(_) => None,
        None => Some(to_f64_array2(old)?),
    };
    let shape = match (&int_chrom, &chrom) {
        (Some(c), _) => c.dim(),
        (_, Some(c)) => c.dim(),
        _ => unreachable!(),
    };
    let (n, l) = shape;
    let mode = pairing(half_n, n, name)?;
    let (seg, virtual_len) = match gene_segments(gene_id, l)? {
        Some((s, k)) => (s, k),
        None => ((0..l).collect(), l),
    };
    if !matches!(mode, Pairing::Global(_)) {
        let half_only = matches!(mode, Pairing::Half);
        return match (int_chrom, chrom) {
            (Some(c), _) => {
                let out = paired_crossover(c, op, pc, half_only, &seg, virtual_len);
                Ok(numpy::IntoPyArray::into_pyarray(out, py)
                    .into_any()
                    .unbind())
            }
            (_, Some(c)) => {
                let out = paired_crossover(c, op, pc, half_only, &seg, virtual_len);
                chrom_out(py, out, as_int)
            }
            _ => unreachable!(),
        };
    }
    let chrom = match (int_chrom, chrom) {
        (Some(c), _) => c.mapv(f64::from),
        (_, Some(c)) => c,
        _ => unreachable!(),
    };
    let src = copy_sources(op, n, virtual_len, pc, mode);
    let mut out = Array2::<f64>::zeros((src.len(), l));
    for (r, s) in src.iter().enumerate() {
        for j in 0..l {
            out[[r, j]] = chrom[[s[seg[j]], j]];
        }
    }
    chrom_out(py, out, as_int)
}

macro_rules! copy_operator {
    ($name:ident, $op:expr, $default_pc:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (old_chrom, xovr=None, half_n=None, gene_id=None, parallel=None))]
        pub fn $name<'py>(
            py: Python<'py>,
            old_chrom: &Bound<'py, PyAny>,
            xovr: Opt<'_, 'py>,
            half_n: Opt<'_, 'py>,
            gene_id: Opt<'_, 'py>,
            parallel: Opt<'_, 'py>,
        ) -> PyResult<PyObject> {
            let _ = parallel;
            let name = stringify!($name);
            copy_crossover(
                py,
                name,
                $op,
                old_chrom,
                prob(xovr, $default_pc, name)?,
                half_n,
                gene_id,
            )
        }
    };
}

copy_operator!(xovsp, CopyOp::SinglePoint, 0.7, "Single-point crossover.");
copy_operator!(
    xovdp,
    CopyOp::DoublePoint,
    0.7,
    "Two-point crossover (inclusive segment)."
);
copy_operator!(
    xovud,
    CopyOp::Uniform,
    0.5,
    "Uniform crossover: every gene is exchanged with probability XOVR."
);
copy_operator!(
    xovsh,
    CopyOp::Shuffle,
    0.7,
    "Shuffle crossover: single-point crossover on a random gene order."
);
copy_operator!(
    xovsec,
    CopyOp::ShuffleExp,
    0.7,
    "Shuffle exponential crossover (Tanabe & Fukunaga)."
);
copy_operator!(
    xovexp,
    CopyOp::Exponential,
    0.7,
    "Exponential crossover (DE): a circular section of geometric length."
);
copy_operator!(
    xovbd,
    CopyOp::Binomial,
    0.7,
    "Binomial crossover (DE): one forced gene plus each other gene with probability XOVR."
);

/// Discrete recombination: every gene of each offspring comes from the partner with probability RecOpt.
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=None, half_n=None, gene_id=None, parallel=None))]
pub fn recdis<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    gene_id: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    copy_crossover(
        py,
        "recdis",
        CopyOp::Discrete,
        old_chrom,
        prob(rec_opt, 0.5, "recdis")?,
        half_n,
        gene_id,
    )
}

/// Real-valued recombination framework: `pair(p1, p2, rng)` returns both offspring of one pair,
/// `global(pa, pb, rng)` one gene of a global-mode offspring.
fn real_recombination<'py, F, G>(
    py: Python<'py>,
    name: &str,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    mut pair: F,
    mut global: G,
) -> PyResult<PyObject>
where
    F: FnMut(usize, &[f64], &[f64], &mut rand::rngs::SmallRng) -> (Vec<f64>, Vec<f64>),
    G: FnMut(usize, f64, f64, &mut rand::rngs::SmallRng) -> f64,
{
    let chrom = to_f64_array2(old)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let half = n / 2;
    let mut rng = crate::utils::rng();
    let out = match pairing(half_n, n, name)? {
        Pairing::Global(k) => {
            let mut out = Array2::<f64>::zeros((k, l));
            for r in 0..k {
                for j in 0..l {
                    let (a, b) = random_pair(n, &mut rng);
                    out[[r, j]] = global(j, chrom[[a, j]], chrom[[b, j]], &mut rng);
                }
            }
            out
        }
        mode => {
            let rows = if matches!(mode, Pairing::Half) {
                half
            } else {
                n
            };
            let mut out = Array2::<f64>::zeros((rows, l));
            if matches!(mode, Pairing::Full) {
                out.assign(&chrom);
            }
            for i in 0..half {
                let (p1, p2) = (chrom.row(i).to_vec(), chrom.row(i + half).to_vec());
                let (c1, c2) = if rng.gen::<f64>() < pc {
                    pair(i, &p1, &p2, &mut rng)
                } else {
                    (p1, p2)
                };
                out.row_mut(i).assign(&numpy::ndarray::Array1::from(c1));
                if matches!(mode, Pairing::Full) {
                    out.row_mut(i + half)
                        .assign(&numpy::ndarray::Array1::from(c2));
                }
            }
            out
        }
    };
    chrom_out(py, out, false)
}

fn sbx_beta<R: Rng>(eta: f64, rng: &mut R) -> f64 {
    let u: f64 = rng.gen();
    let b = if u > 0.5 {
        1.0 / (2.0 * (1.0 - u))
    } else {
        2.0 * u
    };
    let b = b.powf(1.0 / (eta + 1.0));
    if rng.gen::<f64>() < 0.5 {
        -b
    } else {
        b
    }
}

/// Simulated binary crossover: each variable of a crossing pair is recombined with probability 0.5.
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, n=None, parallel=None))]
pub fn recsbx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let l = to_f64_array2(old_chrom)?.shape()[1];
    let eta = non_negative(n, l, 20.0, "recsbx", "n")?;
    let eta2 = eta.clone();
    real_recombination(
        py,
        "recsbx",
        old_chrom,
        prob(xovr, 0.7, "recsbx")?,
        half_n,
        move |_, p1, p2, rng| {
            let mut c1 = p1.to_vec();
            let mut c2 = p2.to_vec();
            for j in 0..p1.len() {
                if rng.gen::<f64>() < 0.5 {
                    let b = sbx_beta(eta[j], rng);
                    c1[j] = 0.5 * ((1.0 + b) * p1[j] + (1.0 - b) * p2[j]);
                    c2[j] = 0.5 * ((1.0 - b) * p1[j] + (1.0 + b) * p2[j]);
                }
            }
            (c1, c2)
        },
        move |j, a, b, rng| {
            if rng.gen::<f64>() < 0.5 {
                let beta = sbx_beta(eta2[j], rng);
                0.5 * ((1.0 + beta) * a + (1.0 - beta) * b)
            } else {
                a
            }
        },
    )
}

enum Alpha {
    Random,
    Scalar(f64),
    Matrix(Array2<f64>),
}

/// Alpha: a scalar or a matrix of factors, all in [-0.25, 1.25) as required by geatpy 2.7.0.
fn parse_alpha(alpha: Opt, name: &str) -> PyResult<Alpha> {
    let ok = |v: f64| (-0.25..1.25).contains(&v);
    let illegal = || {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in {0}: The value of Alpha is illegal. (输入参数Alpha的值不合法，详见{0}帮助文档。)",
            name
        ))
    };
    match alpha {
        Some(a) if !a.is_none() => {
            if let Ok(v) = a.extract::<f64>() {
                if !ok(v) {
                    return Err(illegal());
                }
                Ok(Alpha::Scalar(v))
            } else {
                let m = to_f64_array2(a)?;
                if !m.iter().all(|&v| ok(v)) {
                    return Err(illegal());
                }
                Ok(Alpha::Matrix(m))
            }
        }
        _ => Ok(Alpha::Random),
    }
}

fn intermediate<'py>(
    py: Python<'py>,
    name: &str,
    old: &Bound<'py, PyAny>,
    rec_opt: Opt,
    half_n: Opt,
    alpha: Opt,
    per_gene: bool,
) -> PyResult<PyObject> {
    let alpha = parse_alpha(alpha, name)?;
    let draw = |rng: &mut rand::rngs::SmallRng| rng.gen::<f64>() * 1.5 - 0.25;
    let factor = |row: usize, j: usize, rng: &mut rand::rngs::SmallRng, shared: f64| -> f64 {
        match &alpha {
            Alpha::Scalar(a) => *a,
            Alpha::Matrix(m) => m[[row.min(m.shape()[0] - 1), j.min(m.shape()[1] - 1)]],
            Alpha::Random => {
                if per_gene {
                    draw(rng)
                } else {
                    shared
                }
            }
        }
    };
    let chrom = to_f64_array2(old)?;
    let half = chrom.shape()[0] / 2;
    real_recombination(
        py,
        name,
        old,
        prob(rec_opt, 0.7, name)?,
        half_n,
        |i, p1, p2, rng| {
            let (s1, s2) = (draw(rng), draw(rng));
            let c1 = (0..p1.len())
                .map(|j| p1[j] + factor(i, j, rng, s1) * (p2[j] - p1[j]))
                .collect();
            let c2 = (0..p1.len())
                .map(|j| p1[j] + factor(i + half, j, rng, s2) * (p2[j] - p1[j]))
                .collect();
            (c1, c2)
        },
        |j, a, b, rng| {
            let s = draw(rng);
            a + factor(0, j, rng, s) * (b - a)
        },
    )
}

/// Intermediate recombination: offspring = parent1 + a (parent2 - parent1), a in [-0.25, 1.25) per gene.
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=None, half_n=None, alpha=None, parallel=None))]
pub fn recint<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    alpha: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    intermediate(py, "recint", old_chrom, rec_opt, half_n, alpha, true)
}

/// Line recombination: like intermediate recombination with one factor a per offspring.
#[pyfunction]
#[pyo3(signature = (old_chrom, rec_opt=None, half_n=None, alpha=None, parallel=None))]
pub fn reclin<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    rec_opt: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    alpha: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    intermediate(py, "reclin", old_chrom, rec_opt, half_n, alpha, false)
}

fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    rng.sample(rand_distr::StandardNormal)
}

/// Normal distribution crossover. As in geatpy 2.7.0, a recombined gene is spread around the
/// parents' mean only when parent1 > parent2; otherwise both offspring take the mean.
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, a=None, parallel=None))]
pub fn recndx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    a: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let l = to_f64_array2(old_chrom)?.shape()[1];
    let aa = non_negative(a, l, 1.4826, "recndx", "A")?;
    let aa2 = aa.clone();
    let gene = |aj: f64, x: f64, y: f64, rng: &mut rand::rngs::SmallRng| -> (f64, f64) {
        let mean = (x + y) * 0.5;
        let d = (x - y) * 0.5;
        let s = if rng.gen::<f64>() > 0.5 { -aj } else { aj };
        let g = standard_normal(rng);
        if d > 1e-15 {
            (mean + g * s * d, mean - g * s * d)
        } else {
            (mean, mean)
        }
    };
    real_recombination(
        py,
        "recndx",
        old_chrom,
        prob(xovr, 0.7, "recndx")?,
        half_n,
        move |_, p1, p2, rng| {
            let mut c1 = p1.to_vec();
            let mut c2 = p2.to_vec();
            for j in 0..p1.len() {
                if rng.gen::<f64>() >= 0.5 {
                    let (x, y) = gene(aa[j], p1[j], p2[j], rng);
                    c1[j] = x;
                    c2[j] = y;
                }
            }
            (c1, c2)
        },
        move |j, x, y, rng| {
            if rng.gen::<f64>() >= 0.5 {
                gene(aa2[j], x, y, rng).0
            } else {
                x
            }
        },
    )
}

fn distinct_cuts<R: Rng>(l: usize, rng: &mut R) -> (usize, usize) {
    let a = rng.gen_range(0..l);
    let b = (a + 1 + rng.gen_range(0..l - 1)) % l;
    (a.min(b), a.max(b))
}

/// PMX on the positions where `take` is true, in geatpy 2.7.0's sequential form: at every crossing
/// position the two offspring exchange their *current* genes, and each offspring swaps the incoming
/// gene with the position that held it. Both offspring stay permutations.
fn pmx_pair<T: Gene>(p1: &[T], p2: &[T], take: &[bool]) -> (Vec<T>, Vec<T>) {
    let mut c1 = p1.to_vec();
    let mut c2 = p2.to_vec();
    let Some(index) = DenseIndex::new(p1, p2) else {
        for (k, &t) in take.iter().enumerate() {
            if !t {
                continue;
            }
            let (v1, v2) = (c1[k], c2[k]);
            if let Some(pos) = c1.iter().position(|&x| x == v2) {
                c1.swap(k, pos);
            }
            if let Some(pos) = c2.iter().position(|&x| x == v1) {
                c2.swap(k, pos);
            }
        }
        return (c1, c2);
    };
    // same swaps as above, with value -> position tables kept up to date
    const NONE: usize = usize::MAX;
    let mut pos1 = vec![NONE; index.size];
    let mut pos2 = vec![NONE; index.size];
    for (k, (&a, &b)) in c1.iter().zip(&c2).enumerate() {
        pos1[index.key(a)] = k;
        pos2[index.key(b)] = k;
    }
    let swap = |c: &mut Vec<T>, pos: &mut Vec<usize>, k: usize, v: T| {
        let p = pos[index.key(v)];
        if p != NONE {
            c.swap(k, p);
            pos[index.key(c[k])] = k;
            pos[index.key(c[p])] = p;
        }
    };
    for (k, &t) in take.iter().enumerate() {
        if t {
            let (v1, v2) = (c1[k], c2[k]);
            swap(&mut c1, &mut pos1, k, v2);
            swap(&mut c2, &mut pos2, k, v1);
        }
    }
    (c1, c2)
}

/// Order crossover: each offspring keeps its own [lo, hi] section and fills the remaining positions
/// from position 0 onwards with the partner's genes in the partner's order.
fn ox_pair<T: Gene>(p1: &[T], p2: &[T], lo: usize, hi: usize) -> (Vec<T>, Vec<T>) {
    let index = DenseIndex::new(p1, p2);
    let mut kept_set = Vec::new();
    let mut fill = |own: &[T], other: &[T]| -> Vec<T> {
        let kept = &own[lo..=hi];
        // membership in the kept section: a bitmap for small integer genes, a scan otherwise
        let is_kept: Box<dyn Fn(&T) -> bool + '_> = match &index {
            Some(ix) => {
                kept_set.clear();
                kept_set.resize(ix.size, false);
                for &v in kept {
                    kept_set[ix.key(v)] = true;
                }
                let set = &kept_set;
                Box::new(move |v: &T| set[ix.key(*v)])
            }
            None => Box::new(|v: &T| kept.contains(v)),
        };
        let mut rest = other.iter().filter(|v| !is_kept(v));
        (0..own.len())
            .map(|j| {
                if (lo..=hi).contains(&j) {
                    own[j]
                } else {
                    *rest.next().unwrap()
                }
            })
            .collect()
    };
    let c1 = fill(p1, p2);
    let c2 = fill(p2, p1);
    (c1, c2)
}

/// The permutation crossovers that share the pairing logic below.
#[derive(Clone, Copy)]
enum PermOp {
    /// PMX; `uniform` is Method 2 (every position crosses with probability `pc`).
    Pmx {
        uniform: bool,
        pc: f64,
    },
    Ox,
}

impl PermOp {
    fn cross<T: Gene, R: Rng>(
        self,
        a: &[T],
        b: &[T],
        global: bool,
        rng: &mut R,
    ) -> (Vec<T>, Vec<T>) {
        let l = a.len();
        if l < 2 {
            return (b.to_vec(), a.to_vec());
        }
        match self {
            PermOp::Pmx { uniform, pc } => {
                let take: Vec<bool> = if uniform {
                    (0..l).map(|_| global || rng.gen::<f64>() < pc).collect()
                } else {
                    let (lo, hi) = distinct_cuts(l, rng);
                    (0..l).map(|j| j >= lo && j <= hi).collect()
                };
                pmx_pair(a, b, &take)
            }
            PermOp::Ox => {
                let (lo, hi) = distinct_cuts(l, rng);
                ox_pair(a, b, lo, hi)
            }
        }
    }
}

/// Pairs parents like the other crossovers. With `pair_gate` a pair crosses with probability `pc`;
/// without it the operator always runs and handles `pc` itself (global mode: Half_N an integer).
fn permutation_rows<T: Gene>(
    chrom: &Array2<T>,
    op: PermOp,
    pc: f64,
    mode: Pairing,
    pair_gate: bool,
) -> Array2<T> {
    let (n, l) = chrom.dim();
    let half = n / 2;
    let mut rng = crate::utils::rng();
    let row = |i: usize| chrom.row(i).to_vec();
    match mode {
        Pairing::Global(k) => {
            let mut out = Vec::with_capacity(k * l);
            for _ in 0..k {
                let (a, b) = random_pair(n, &mut rng);
                out.extend(op.cross(&row(a), &row(b), true, &mut rng).0);
            }
            Array2::from_shape_vec((k, l), out).expect("shape matches")
        }
        mode => {
            let full = matches!(mode, Pairing::Full);
            let mut out = if full {
                chrom.clone()
            } else {
                chrom.slice(numpy::ndarray::s![..half, ..]).to_owned()
            };
            for i in 0..half {
                if pair_gate && rng.gen::<f64>() >= pc {
                    continue;
                }
                let (c1, c2) = op.cross(&row(i), &row(i + half), false, &mut rng);
                out.row_mut(i)
                    .assign(&numpy::ndarray::ArrayView1::from(&c1[..]));
                if full {
                    out.row_mut(i + half)
                        .assign(&numpy::ndarray::ArrayView1::from(&c2[..]));
                }
            }
            out
        }
    }
}

fn permutation_crossover<'py>(
    py: Python<'py>,
    name: &str,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    pair_gate: bool,
    op: PermOp,
) -> PyResult<PyObject> {
    let as_int = is_integer_like(old);
    if as_int {
        if let Some(chrom) = crate::utils::to_i32_array2(old) {
            check_permutation(&chrom, name)?;
            let mode = pairing(half_n, chrom.nrows(), name)?;
            let out = permutation_rows(&chrom, op, pc, mode, pair_gate);
            return Ok(numpy::IntoPyArray::into_pyarray(out, py)
                .into_any()
                .unbind());
        }
    }
    let chrom = to_f64_array2(old)?;
    // geatpy 2.7.0 reads the gene range with int(np.min(Chrom)) and int(np.max(Chrom)) first.
    if chrom.iter().any(|v| v.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cannot convert float NaN to integer",
        ));
    }
    if chrom.iter().any(|v| v.is_infinite()) {
        return Err(pyo3::exceptions::PyOverflowError::new_err(
            "cannot convert float infinity to integer",
        ));
    }
    check_permutation(&chrom, name)?;
    let mode = pairing(half_n, chrom.nrows(), name)?;
    chrom_out(
        py,
        permutation_rows(&chrom, op, pc, mode, pair_gate),
        as_int,
    )
}

/// Partially mapped crossover. Method 1: a pair crosses with probability XOVR on a two-point section;
/// Method 2 (uniform PMX): every position crosses with probability XOVR (all positions in global mode).
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, method=None, parallel=None))]
pub fn xovpmx<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    method: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = parallel;
    let method = crate::utils::opt_int(method, 1)?;
    if method != 1 && method != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "error in xovpmx: Method must be 1 or 2. (传入参数Method必须是1或者2。)",
        ));
    }
    let pc = prob(xovr, 0.7, "xovpmx")?;
    permutation_crossover(
        py,
        "xovpmx",
        old_chrom,
        pc,
        half_n,
        method != 2,
        PermOp::Pmx {
            uniform: method == 2,
            pc,
        },
    )
}

/// Order crossover (OX) for permutations.
#[pyfunction]
#[pyo3(signature = (old_chrom, xovr=None, half_n=None, parallel=None, params4=None))]
pub fn xovox<'py>(
    py: Python<'py>,
    old_chrom: &Bound<'py, PyAny>,
    xovr: Opt<'_, 'py>,
    half_n: Opt<'_, 'py>,
    parallel: Opt<'_, 'py>,
    params4: Opt<'_, 'py>,
) -> PyResult<PyObject> {
    let _ = (parallel, params4);
    permutation_crossover(
        py,
        "xovox",
        old_chrom,
        prob(xovr, 0.7, "xovox")?,
        half_n,
        true,
        PermOp::Ox,
    )
}

/// Dispatcher kept for API compatibility: recombin(REC_F, OldChrom, ...) forwards the remaining
/// arguments to the named operator.
#[pyfunction]
#[pyo3(signature = (rec_f, *args, **kwargs))]
pub fn recombin<'py>(
    py: Python<'py>,
    rec_f: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<PyObject> {
    let name = rec_f.to_lowercase();
    let allowed = [
        "recsbx", "recdis", "recint", "reclin", "recndx", "xovsp", "xovdp", "xovud", "xovsh",
        "xovpmx", "xovox", "xovsec", "xovbd", "xovexp",
    ];
    if !allowed.contains(&name.as_str()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "error in recombin: unknown operator {}.",
            rec_f
        )));
    }
    let module = py
        .import("_geatpy_core")
        .or_else(|_| py.import("geatpy._geatpy_core"))?;
    Ok(module.getattr(name.as_str())?.call(args, kwargs)?.unbind())
}
