//! Thread pool for the operators that run in parallel.
//!
//! Parallel paths only split work whose result does not depend on how it is split (one output row
//! per task, one candidate per task), so in the default floating-point environment the result is
//! bit-identical to the serial one for every thread count. (Worker threads keep the
//! floating-point control bits of the thread that built the pool, so a caller that turns on
//! flush-to-zero afterwards may see subnormal results differ.) Every parallel call has a fixed
//! hand-off cost of tens of microseconds, so each operator stays serial below a work threshold
//! measured where the parallel path starts to pay.

use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Parallel calls made so far (read by the tests to check that their inputs go parallel).
static PARALLEL_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Threads set by `set_num_threads`; 0 means the default.
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

/// The pool and the thread count it was built for; `None` as the pool remembers a failed build.
type Slot = Option<(usize, Option<Arc<ThreadPool>>)>;
static POOL: Mutex<Slot> = Mutex::new(None);

fn lock() -> MutexGuard<'static, Slot> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Logical CPUs available to this process (respecting CPU affinity): more threads than this only
/// slow the work down, so it caps every request.
fn cpus() -> usize {
    static CPUS: OnceLock<usize> = OnceLock::new();
    *CPUS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// GEATPY_NUM_THREADS when it is a positive integer, otherwise the number of logical CPUs.
fn default_threads() -> usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("GEATPY_NUM_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(cpus)
    })
}

/// Threads the operators use: the request (or the default), at most the number of logical CPUs.
pub fn num_threads() -> usize {
    let n = match REQUESTED.load(Ordering::Relaxed) {
        0 => default_threads(),
        n => n,
    };
    n.clamp(1, cpus())
}

/// The pool to run `work` units on, or None to run serially. The pool is built without holding
/// the lock (building starts OS threads), so a fork in another thread never inherits it locked.
pub fn pool_for(work: usize, min_work: usize) -> Option<Arc<ThreadPool>> {
    let threads = num_threads();
    if threads <= 1 || work < min_work {
        return None;
    }
    if let Some((t, pool)) = lock().as_ref() {
        if *t == threads {
            return pool.clone();
        }
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("geatpy-{i}"))
        .build()
        .ok()
        .map(Arc::new);
    let mut slot = lock();
    match slot.as_ref() {
        // another thread installed a pool of this size meanwhile: use that one
        Some((t, pool)) if *t == threads => pool.clone(),
        _ => {
            *slot = Some((threads, built.clone()));
            built
        }
    }
}

thread_local! {
    /// The pool lock held by the thread calling `os.fork`, from just before the fork until after it.
    static FORK_GUARD: RefCell<Option<MutexGuard<'static, Slot>>> = const { RefCell::new(None) };
}

/// Registers `os.register_at_fork` handlers: the pool lock is taken before a fork and released in
/// both processes afterwards, and the child forgets the pool, whose worker threads exist only in
/// the parent (it is leaked, as tearing it down would wait for those threads).
pub fn register_fork_handlers(py: Python<'_>) -> PyResult<()> {
    use pyo3::types::{PyCFunction, PyDict, PyTuple};
    let os = py.import("os")?;
    if !os.hasattr("register_at_fork")? {
        return Ok(()); // Windows: no fork
    }
    let before = PyCFunction::new_closure(
        py,
        None,
        None,
        |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            FORK_GUARD.with(|g| *g.borrow_mut() = Some(lock()));
        },
    )?;
    let parent = PyCFunction::new_closure(
        py,
        None,
        None,
        |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            FORK_GUARD.with(|g| g.borrow_mut().take());
        },
    )?;
    let child = PyCFunction::new_closure(
        py,
        None,
        None,
        |_: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            FORK_GUARD.with(|g| {
                if let Some(mut slot) = g.borrow_mut().take() {
                    if let Some((_, Some(pool))) = slot.take() {
                        std::mem::forget(pool);
                    }
                }
            });
        },
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("before", before)?;
    kwargs.set_item("after_in_parent", parent)?;
    kwargs.set_item("after_in_child", child)?;
    os.getattr("register_at_fork")?.call((), Some(&kwargs))?;
    Ok(())
}

/// (0..n).map(f).collect(), on `pool` when there is one.
pub fn map_collect<T, F>(pool: Option<&ThreadPool>, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    match pool {
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| (0..n).into_par_iter().map(f).collect())
        }
        None => (0..n).map(f).collect(),
    }
}

/// Like `map_collect` with a scratch value made by `init` once per piece of work a thread takes
/// (once in all when serial) instead of once per item.
pub fn map_collect_init<T, S, I, F>(pool: Option<&ThreadPool>, n: usize, init: I, f: F) -> Vec<T>
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> T + Sync + Send,
{
    match pool {
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| (0..n).into_par_iter().map_init(&init, &f).collect())
        }
        None => {
            let mut scratch = init();
            (0..n).map(|i| f(&mut scratch, i)).collect()
        }
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
        Some(p) => {
            PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
            p.install(|| {
                data.par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(i, row)| f(i, row))
            })
        }
        None => data
            .chunks_mut(width)
            .enumerate()
            .for_each(|(i, row)| f(i, row)),
    }
}

/// Sets the number of threads the operators may use; None or 0 restores the default
/// (GEATPY_NUM_THREADS, or the number of logical CPUs). 1 runs everything serially. More threads
/// than logical CPUs are not used.
#[pyfunction]
#[pyo3(signature = (n=None))]
pub fn set_num_threads(n: Option<usize>) {
    REQUESTED.store(n.unwrap_or(0), Ordering::Relaxed);
}

/// The number of threads the operators use.
#[pyfunction]
pub fn get_num_threads() -> usize {
    num_threads()
}

/// Number of parallel calls so far; for the tests only.
#[pyfunction]
pub fn _parallel_calls() -> usize {
    PARALLEL_CALLS.load(Ordering::Relaxed)
}
