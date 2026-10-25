//! Log files on the Mac, in the `logs` folder of the app's home: `agent.log` gets
//! everything the agent prints, and `jobs.jsonl` gets one line per job, which also brings
//! the recent-jobs list back after a restart. Job output is never written here: nothing
//! from a job stays on the Mac.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::de::DeserializeOwned;
use serde::Serialize;

/// A log over this size is renamed to `<name>.1` (replacing the previous one) at startup.
const MAX_LOG_BYTES: u64 = 5 << 20;

pub struct Logs {
    pub dir: PathBuf,
}

impl Logs {
    pub fn new(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        rotate(&dir.join("jobs.jsonl"));
        Logs { dir }
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

    /// Appends one job. A failure to write never affects the job.
    pub fn append_job(&self, job: &impl Serialize) {
        let Ok(line) = serde_json::to_string(job) else { return };
        if let Ok(mut f) =
            std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("jobs.jsonl"))
        {
            let _ = writeln!(f, "{line}");
        }
    }

    /// The last `n` jobs, newest first. Lines that no longer parse are skipped.
    pub fn recent_jobs<T: DeserializeOwned>(&self, n: usize) -> Vec<T> {
        let Ok(file) = std::fs::File::open(self.dir.join("jobs.jsonl")) else {
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
        let logs = Logs::new(dir.path().join("logs"));
        for i in 0..5 {
            logs.append_job(&i);
        }
        std::fs::OpenOptions::new()
            .append(true)
            .open(logs.dir.join("jobs.jsonl"))
            .unwrap()
            .write_all(b"not json\n")
            .unwrap();
        assert_eq!(logs.recent_jobs::<i32>(3), [4, 3, 2]);
    }
}
