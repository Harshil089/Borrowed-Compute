//! A fixed set of worker threads for the matrix products. A token runs ~170 products in a
//! 24-layer model; spawning threads for each costs more than the product itself.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Pool {
    workers: Vec<Mutex<Sender<Job>>>,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        // BC_INFER_THREADS caps the cores the engine uses, so a lender can keep some.
        let all = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1);
        let n = std::env::var("BC_INFER_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(all).clamp(1, all).saturating_sub(1);
        let workers = (0..n)
            .map(|i| {
                let (tx, rx) = channel::<Job>();
                std::thread::Builder::new()
                    .name(format!("bc-infer-{i}"))
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            job();
                        }
                    })
                    .expect("worker thread");
                Mutex::new(tx)
            })
            .collect();
        Pool { workers }
    })
}

/// Threads a product can use, the calling one included.
pub fn threads() -> usize {
    pool().workers.len() + 1
}

/// Runs `f(0)…f(n-1)` on the workers and this thread; returns when all are done.
///
/// Indexes are handed out one at a time from a shared counter, so a thread the system
/// pauses (a lender's owner is using the machine) holds up at most one small piece
/// instead of a fixed share.
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n <= 1 {
        if n == 1 {
            f(0);
        }
        return;
    }
    let p = pool();
    let helpers = p.workers.len().min(n - 1);
    let next = Arc::new(AtomicUsize::new(0));
    let latch = Arc::new((Mutex::new((helpers, false)), Condvar::new()));
    // SAFETY: every job sent below finishes (or panics, caught) before the latch reaches
    // zero, and this function does not return until it does, so `f` outlives every use.
    let f_static: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
    for w in &p.workers[..helpers] {
        let (latch, next) = (latch.clone(), next.clone());
        let job: Job = Box::new(move || {
            let ok = catch_unwind(AssertUnwindSafe(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                f_static(i);
            }))
            .is_ok();
            let (m, c) = &*latch;
            let mut g = m.lock().unwrap();
            g.0 -= 1;
            g.1 |= !ok;
            if g.0 == 0 {
                c.notify_one();
            }
        });
        w.lock().unwrap().send(job).expect("worker thread gone");
    }
    let mine = catch_unwind(AssertUnwindSafe(|| loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        f(i);
    }));
    let (m, c) = &*latch;
    let mut g = m.lock().unwrap();
    while g.0 > 0 {
        g = c.wait(g).unwrap();
    }
    if g.1 || mine.is_err() {
        panic!("a matrix product panicked");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn runs_every_index_once() {
        let hits: Vec<std::sync::atomic::AtomicUsize> = (0..37).map(|_| Default::default()).collect();
        super::run(37, &|i| {
            hits[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(std::sync::atomic::Ordering::Relaxed) == 1));
    }
}
