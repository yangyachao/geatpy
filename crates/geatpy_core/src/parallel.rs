//! Thread pool for the operators that run in parallel.
//!
//! Parallel paths only split work whose result does not depend on how it is split (one output row
//! per task, one candidate per task), so the result is bit-identical to the serial one for every
//! thread count. Small inputs stay serial: below `min_work` the hand-off costs more than it saves.

use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Threads set by `set_num_threads`; 0 means the default.
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
/// The pool, with the process id and size it was built for.
static POOL: Mutex<Option<(u32, usize, Arc<ThreadPool>)>> = Mutex::new(None);

/// GEATPY_NUM_THREADS when it is a positive integer, otherwise the number of logical CPUs.
fn default_threads() -> usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("GEATPY_NUM_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    })
}

pub fn num_threads() -> usize {
    match REQUESTED.load(Ordering::Relaxed) {
        0 => default_threads(),
        n => n,
    }
}

/// The pool to run `work` units on, or None to run serially.
pub fn pool_for(work: usize, min_work: usize) -> Option<Arc<ThreadPool>> {
    let threads = num_threads();
    if threads <= 1 || work < min_work {
        return None;
    }
    let pid = std::process::id();
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, t, pool)) = slot.as_ref() {
        if *p == pid && *t == threads {
            return Some(pool.clone());
        }
    }
    // A forked child (multiprocessing on Linux) inherits the pool without its worker threads, so
    // it builds its own; the inherited one is leaked rather than torn down.
    if let Some((p, _, old)) = slot.take() {
        if p != pid {
            std::mem::forget(old);
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("geatpy-{i}"))
            .build()
            .ok()?,
    );
    *slot = Some((pid, threads, pool.clone()));
    Some(pool)
}

/// (0..n).map(f).collect(), on `pool` when there is one.
pub fn map_collect<T, F>(pool: Option<&ThreadPool>, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    match pool {
        Some(p) => p.install(|| (0..n).into_par_iter().map(f).collect()),
        None => (0..n).map(f).collect(),
    }
}

/// Calls f(row index, row) for every `width`-long row of `data`, on `pool` when there is one.
pub fn for_each_row<T, F>(pool: Option<&ThreadPool>, data: &mut [T], width: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if width == 0 {
        return;
    }
    match pool {
        Some(p) => p.install(|| {
            data.par_chunks_mut(width)
                .enumerate()
                .for_each(|(i, row)| f(i, row))
        }),
        None => data
            .chunks_mut(width)
            .enumerate()
            .for_each(|(i, row)| f(i, row)),
    }
}

/// Sets the number of threads the operators may use; None or 0 restores the default
/// (GEATPY_NUM_THREADS, or the number of logical CPUs). 1 runs everything serially.
#[pyfunction]
#[pyo3(signature = (n=None))]
pub fn set_num_threads(n: Option<usize>) {
    REQUESTED.store(n.unwrap_or(0), Ordering::Relaxed);
}

/// The number of threads the operators may use.
#[pyfunction]
pub fn get_num_threads() -> usize {
    num_threads()
}
