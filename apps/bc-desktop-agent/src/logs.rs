//! Log files on the Mac, in the `logs` folder of the app's home: `agent.log` gets what the
//! agent prints about itself (startup, cloud errors), and `jobs/` gets one file per job with
//! its command, result and usage. Job output is never written: nothing from a job stays on
//! the Mac. A separate history file brings the recent-jobs list back after a restart.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::de::DeserializeOwned;
use serde::Serialize;

/// A log over this size is renamed to `<name>.1` (replacing the previous one) at startup.
const MAX_LOG_BYTES: u64 = 5 << 20;

/// Job files beyond this many are deleted, oldest first.
const MAX_JOB_FILES: usize = 500;

pub struct Logs {
    pub dir: PathBuf,
    /// One JSON line per job, read back at startup.
    history: PathBuf,
}

impl Logs {
    pub fn new(dir: PathBuf, history: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(dir.join("jobs"));
        rotate(&history);
        Logs { dir, history }
    }

    /// Sends stderr to `agent.log`. Skipped when stderr is a terminal, as under
    /// `cargo run`, where printing there is more useful.
    pub fn capture_stderr(&self) -> std::io::Result<()> {
        use std::io::IsTerminal;
        if std::io::stderr().is_terminal() {
            return Ok(());
        }
        let path = self.dir.join("agent.log");
        rotate(&path);
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: dup2 only swaps file descriptors; both are valid here.
            if unsafe { libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        #[cfg(not(unix))]
        drop(file);
        Ok(())
    }

    /// Writes one job's file. `name` starts with the job's start time, so names sort
    /// oldest first. A failure to write never affects the job.
    pub fn write_job_file(&self, name: &str, text: &str) {
        let dir = self.dir.join("jobs");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(name), text);
        prune(&dir, MAX_JOB_FILES);
    }

    /// Adds one job to the history. A failure to write never affects the job.
    pub fn append_history(&self, job: &impl Serialize) {
        let Ok(line) = serde_json::to_string(job) else { return };
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.history) {
            let _ = writeln!(f, "{line}");
        }
    }

    /// The last `n` jobs, newest first. Lines that no longer parse are skipped.
    pub fn recent_jobs<T: DeserializeOwned>(&self, n: usize) -> Vec<T> {
        let Ok(file) = std::fs::File::open(&self.history) else {
            return Vec::new();
        };
        let mut jobs: Vec<T> = BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str(&line).ok())
            .collect();
        jobs.reverse();
        jobs.truncate(n);
        jobs
    }
}

/// Writes a timestamped line to the agent log.
pub fn log(message: impl std::fmt::Display) {
    eprintln!("{} {message}", humantime::format_rfc3339_seconds(SystemTime::now()));
}

fn prune(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    let excess = files.len().saturating_sub(keep);
    for old in &files[..excess] {
        let _ = std::fs::remove_file(old);
    }
}

fn rotate(path: &Path) {
    if std::fs::metadata(path).map(|m| m.len() > MAX_LOG_BYTES).unwrap_or(false) {
        let mut old = path.as_os_str().to_owned();
        old.push(".1");
        let _ = std::fs::rename(path, old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_recent_jobs_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let logs = Logs::new(dir.path().join("logs"), dir.path().join("history.jsonl"));
        for i in 0..5 {
            logs.append_history(&i);
        }
        std::fs::OpenOptions::new()
            .append(true)
            .open(&logs.history)
            .unwrap()
            .write_all(b"not json\n")
            .unwrap();
        assert_eq!(logs.recent_jobs::<i32>(3), [4, 3, 2]);
    }
}

#[cfg(test)]
mod job_file_tests {
    use super::*;

    #[test]
    fn one_file_per_job_and_oldest_pruned() {
        let dir = tempfile::tempdir().unwrap();
        let logs = Logs::new(dir.path().join("logs"), dir.path().join("history.jsonl"));
        for i in 0..MAX_JOB_FILES + 3 {
            logs.write_job_file(&format!("{i:04}-job.log"), &format!("job {i}\n"));
        }
        let jobs = logs.dir.join("jobs");
        assert_eq!(std::fs::read_dir(&jobs).unwrap().count(), MAX_JOB_FILES);
        assert!(!jobs.join("0002-job.log").exists());
        assert_eq!(std::fs::read_to_string(jobs.join("0003-job.log")).unwrap(), "job 3\n");
    }
}
