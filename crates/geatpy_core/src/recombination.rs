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

fn pairing(half_n: Opt, n: usize) -> PyResult<Pairing> {
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
            let k: f64 = o.extract()?;
            if k <= 0.0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "error: Half_N must be a bool or a positive integer.",
                ));
            }
            Ok(Pairing::Global((k as usize).min(n)))
        }
    }
}

fn prob(obj: Opt, default: f64) -> PyResult<f64> {
    match obj {
        Some(o) if !o.is_none() => o.extract(),
        _ => Ok(default),
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
            CopyOp::SinglePoint => {
                if l < 2 {
                    return None;
                }
                let point = rng.gen_range(0..l - 1);
                m.iter_mut().skip(point + 1).for_each(|x| *x = true);
            }
            CopyOp::DoublePoint => {
                let (mut a, mut b) = (rng.gen_range(0..l), rng.gen_range(0..l));
                if b < a {
                    std::mem::swap(&mut a, &mut b);
                }
                m[a..=b].iter_mut().for_each(|x| *x = true);
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
    let mut rng = rand::thread_rng();
    let half = n / 2;
    match mode {
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
            let mut uniq: Vec<f64> = ids.to_vec();
            uniq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let src = copy_sources(op, n, mats.len(), pc, pairing(half_n, n)?);
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
    let chrom = to_f64_array2(old)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let mode = pairing(half_n, n)?;
    let (seg, virtual_len) = match gene_segments(gene_id, l)? {
        Some((s, k)) => (s, k),
        None => ((0..l).collect(), l),
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
            copy_crossover(
                py,
                $op,
                old_chrom,
                prob(xovr, $default_pc)?,
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
    0.7,
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
        CopyOp::Discrete,
        old_chrom,
        prob(rec_opt, 0.5)?,
        half_n,
        gene_id,
    )
}

/// Real-valued recombination framework: `pair(p1, p2, rng)` returns both offspring of one pair,
/// `global(pa, pb, rng)` one gene of a global-mode offspring.
fn real_recombination<'py, F, G>(
    py: Python<'py>,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    mut pair: F,
    mut global: G,
) -> PyResult<PyObject>
where
    F: FnMut(usize, &[f64], &[f64], &mut rand::rngs::ThreadRng) -> (Vec<f64>, Vec<f64>),
    G: FnMut(usize, f64, f64, &mut rand::rngs::ThreadRng) -> f64,
{
    let chrom = to_f64_array2(old)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let half = n / 2;
    let mut rng = rand::thread_rng();
    let out = match pairing(half_n, n)? {
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

fn per_gene_vec(obj: Opt, l: usize, default: f64, name: &str) -> PyResult<Vec<f64>> {
    crate::utils::per_gene(obj, l, default, name)
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
    let eta = per_gene_vec(n, l, 20.0, "n")?;
    let eta2 = eta.clone();
    real_recombination(
        py,
        old_chrom,
        prob(xovr, 0.7)?,
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

fn parse_alpha(alpha: Opt) -> PyResult<Alpha> {
    match alpha {
        Some(a) if !a.is_none() => {
            if let Ok(v) = a.extract::<f64>() {
                Ok(Alpha::Scalar(v))
            } else {
                Ok(Alpha::Matrix(to_f64_array2(a)?))
            }
        }
        _ => Ok(Alpha::Random),
    }
}

fn intermediate<'py>(
    py: Python<'py>,
    old: &Bound<'py, PyAny>,
    rec_opt: Opt,
    half_n: Opt,
    alpha: Opt,
    per_gene: bool,
) -> PyResult<PyObject> {
    let alpha = parse_alpha(alpha)?;
    let draw = |rng: &mut rand::rngs::ThreadRng| rng.gen::<f64>() * 1.5 - 0.25;
    let factor = |row: usize, j: usize, rng: &mut rand::rngs::ThreadRng, shared: f64| -> f64 {
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
        old,
        prob(rec_opt, 0.7)?,
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
    intermediate(py, old_chrom, rec_opt, half_n, alpha, true)
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
    intermediate(py, old_chrom, rec_opt, half_n, alpha, false)
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
    let aa = per_gene_vec(a, l, 1.4826, "A")?;
    let aa2 = aa.clone();
    let gene = |aj: f64, x: f64, y: f64, rng: &mut rand::rngs::ThreadRng| -> (f64, f64) {
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
        old_chrom,
        prob(xovr, 0.7)?,
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

/// Swap-based PMX on the positions where `take` is true; both offspring stay permutations.
fn pmx_pair(p1: &[f64], p2: &[f64], take: &[bool]) -> (Vec<f64>, Vec<f64>) {
    let mut c1 = p1.to_vec();
    let mut c2 = p2.to_vec();
    for (k, &t) in take.iter().enumerate() {
        if !t {
            continue;
        }
        let (v1, v2) = (p1[k], p2[k]);
        if let Some(pos) = c1.iter().position(|&x| x == v2) {
            c1.swap(k, pos);
        }
        if let Some(pos) = c2.iter().position(|&x| x == v1) {
            c2.swap(k, pos);
        }
    }
    (c1, c2)
}

/// Order crossover: each offspring keeps its own [lo, hi] section and fills the remaining positions
/// from position 0 onwards with the partner's genes in the partner's order.
fn ox_pair(p1: &[f64], p2: &[f64], lo: usize, hi: usize) -> (Vec<f64>, Vec<f64>) {
    let fill = |own: &[f64], other: &[f64]| -> Vec<f64> {
        let kept = &own[lo..=hi];
        let mut rest = other.iter().filter(|v| !kept.contains(v));
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
    (fill(p1, p2), fill(p2, p1))
}

fn permutation_crossover<'py, F>(
    py: Python<'py>,
    old: &Bound<'py, PyAny>,
    pc: f64,
    half_n: Opt,
    mut op: F,
) -> PyResult<PyObject>
where
    F: FnMut(&[f64], &[f64], &mut rand::rngs::ThreadRng) -> (Vec<f64>, Vec<f64>),
{
    let as_int = is_integer_like(old);
    let chrom = to_f64_array2(old)?;
    let (n, l) = (chrom.shape()[0], chrom.shape()[1]);
    let half = n / 2;
    let mut rng = rand::thread_rng();
    let cross = |a: &[f64],
                 b: &[f64],
                 rng: &mut rand::rngs::ThreadRng,
                 op: &mut F|
     -> (Vec<f64>, Vec<f64>) {
        if l < 2 {
            (b.to_vec(), a.to_vec())
        } else {
            op(a, b, rng)
        }
    };
    let out = match pairing(half_n, n)? {
        Pairing::Global(k) => {
            let mut out = Array2::<f64>::zeros((k, l));
            for r in 0..k {
                let (a, b) = random_pair(n, &mut rng);
                let (c, _) = cross(
                    &chrom.row(a).to_vec(),
                    &chrom.row(b).to_vec(),
                    &mut rng,
                    &mut op,
                );
                out.row_mut(r).assign(&numpy::ndarray::Array1::from(c));
            }
            out
        }
        mode => {
            let full = matches!(mode, Pairing::Full);
            let mut out = if full {
                chrom.clone()
            } else {
                Array2::<f64>::zeros((half, l))
            };
            for i in 0..half {
                let (p1, p2) = (chrom.row(i).to_vec(), chrom.row(i + half).to_vec());
                let (c1, c2) = if rng.gen::<f64>() < pc {
                    cross(&p1, &p2, &mut rng, &mut op)
                } else {
                    (p1, p2)
                };
                out.row_mut(i).assign(&numpy::ndarray::Array1::from(c1));
                if full {
                    out.row_mut(i + half)
                        .assign(&numpy::ndarray::Array1::from(c2));
                }
            }
            out
        }
    };
    chrom_out(py, out, as_int)
}

/// Partially mapped crossover (Method 1: two-point section, 2: uniform positions).
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
    permutation_crossover(py, old_chrom, prob(xovr, 0.7)?, half_n, move |a, b, rng| {
        let l = a.len();
        let take: Vec<bool> = if method == 2 {
            (0..l).map(|_| rng.gen::<f64>() < 0.5).collect()
        } else {
            let (lo, hi) = distinct_cuts(l, rng);
            (0..l).map(|j| j >= lo && j <= hi).collect()
        };
        pmx_pair(a, b, &take)
    })
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
    permutation_crossover(py, old_chrom, prob(xovr, 0.7)?, half_n, |a, b, rng| {
        let (lo, hi) = distinct_cuts(a.len(), rng);
        ox_pair(a, b, lo, hi)
    })
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
