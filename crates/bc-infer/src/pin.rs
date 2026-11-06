//! Keeps model weights in RAM. A computer short of memory otherwise swaps them out, and
//! then every token reads them back from disk: on an 8 GB Mac that made a stage ten times
//! slower, and it would also put weights on disk, which the owner ruled out. Locking is
//! best effort: when the system refuses (its limit on locked memory), the weights stay
//! unlocked and it says so once.

use std::sync::atomic::{AtomicBool, Ordering};

static WARNED: AtomicBool = AtomicBool::new(false);

fn refused() {
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("bc-infer: the system would not keep model weights in RAM; under memory pressure they may be swapped out");
    }
}

/// Locks `len` bytes at `ptr` in RAM. True if they are locked.
pub fn lock(ptr: *const u8, len: usize) -> bool {
    let locked = sys::lock(ptr, len);
    if !locked {
        refused();
    }
    locked
}

/// Undoes [`lock`] before the memory is freed.
pub fn unlock(ptr: *const u8, len: usize) {
    sys::unlock(ptr, len)
}

#[cfg(unix)]
mod sys {
    pub fn lock(ptr: *const u8, len: usize) -> bool {
        // SAFETY: the caller owns the allocation for `len` bytes.
        unsafe { libc::mlock(ptr as *const libc::c_void, len) == 0 }
    }
    pub fn unlock(ptr: *const u8, len: usize) {
        // SAFETY: as above.
        unsafe { libc::munlock(ptr as *const libc::c_void, len) };
    }
}

#[cfg(windows)]
mod sys {
    use std::sync::Mutex;
    use windows_sys::Win32::System::Memory::{GetProcessWorkingSetSizeEx, SetProcessWorkingSetSizeEx, VirtualLock, VirtualUnlock, QUOTA_LIMITS_HARDWS_MAX_DISABLE, QUOTA_LIMITS_HARDWS_MIN_DISABLE};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// Windows locks only within the process's minimum working set, so each lock grows it
    /// by its own size first.
    static WORKING_SET: Mutex<()> = Mutex::new(());

    pub fn lock(ptr: *const u8, len: usize) -> bool {
        let _one = WORKING_SET.lock().unwrap();
        // SAFETY: plain calls on this process's own handle and memory it owns.
        unsafe {
            let process = GetCurrentProcess();
            let (mut min, mut max, mut flags) = (0usize, 0usize, 0u32);
            if GetProcessWorkingSetSizeEx(process, &mut min, &mut max, &mut flags) == 0 {
                return false;
            }
            let soft = QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE;
            if SetProcessWorkingSetSizeEx(process, min + len, max.max(min + len) + len, soft) == 0 {
                return false;
            }
            VirtualLock(ptr as *const core::ffi::c_void, len) != 0
        }
    }

    pub fn unlock(ptr: *const u8, len: usize) {
        // SAFETY: memory this process owns and locked.
        unsafe { VirtualUnlock(ptr as *const core::ffi::c_void, len) };
    }
}

#[cfg(not(any(unix, windows)))]
mod sys {
    pub fn lock(_: *const u8, _: usize) -> bool {
        false
    }
    pub fn unlock(_: *const u8, _: usize) {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn locks_and_unlocks_memory() {
        let b = crate::quant::Bytes::from_slice(&[7u8; 100_000]);
        assert!(b.locked(), "a small allocation can be kept in RAM");
        assert_eq!(b[99_999], 7);
    }
}
