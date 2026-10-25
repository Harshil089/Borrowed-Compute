//! Runs one job in a fresh workspace and streams its output.
//!
//! Phase one: the workspace is a temporary directory that is deleted after the job, and
//! every process the job started is killed when it ends. The dedicated `_bcjob` user and
//! the Seatbelt profile come in Phase two.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_proto::wire::{AgentMsg, JobRequest, OutputStream};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::mpsc;

const READ_CHUNK: usize = 16 * 1024;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Parent directory for job workspaces.
    pub work_root: PathBuf,
    /// Used when the job sets no timeout.
    pub default_timeout: Duration,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig {
            work_root: std::env::temp_dir().join("borrowed-compute"),
            default_timeout: Duration::from_secs(60 * 60),
        }
    }
}

/// Runs the job and sends `Started`, `Output` and a final `Exited` or `Failed` to `tx`.
/// Returns the workspace path, which no longer exists when this returns.
pub async fn run_job(
    config: &RunnerConfig,
    req: &JobRequest,
    archive: Vec<u8>,
    tx: mpsc::Sender<AgentMsg>,
) -> PathBuf {
    let workspace = match prepare(config, archive).await {
        Ok(ws) => ws,
        Err(e) => {
            let _ = tx
                .send(AgentMsg::Failed {
                    reason: format!("preparing workspace: {e:#}"),
                })
                .await;
            return PathBuf::new();
        }
    };
    let path = workspace.path().to_path_buf();
    let last = match execute(config, req, workspace.path(), &tx).await {
        Ok(exited) => exited,
        Err(e) => AgentMsg::Failed {
            reason: format!("{e:#}"),
        },
    };
    let _ = tx.send(last).await;
    // Deleting the workspace keeps the promise that nothing stays on the Mac.
    drop(workspace);
    path
}

async fn prepare(config: &RunnerConfig, archive: Vec<u8>) -> Result<tempfile::TempDir> {
    tokio::fs::create_dir_all(&config.work_root)
        .await
        .with_context(|| format!("creating {}", config.work_root.display()))?;
    let workspace = tempfile::Builder::new()
        .prefix("job-")
        .tempdir_in(&config.work_root)?;
    let root = workspace.path().to_path_buf();
    tokio::task::spawn_blocking(move || unpack(&archive, &root)).await??;
    Ok(workspace)
}

/// Unpacks a tar archive, refusing any entry that would land outside `root`.
fn unpack(archive: &[u8], root: &Path) -> Result<()> {
    let mut tar = tar::Archive::new(archive);
    tar.set_preserve_permissions(true);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if !is_inside(&path) {
            bail!("archive entry {} escapes the workspace", path.display());
        }
        if !entry.unpack_in(root)? {
            bail!("archive entry {} escapes the workspace", path.display());
        }
    }
    Ok(())
}

fn is_inside(path: &Path) -> bool {
    path.components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

async fn execute(
    config: &RunnerConfig,
    req: &JobRequest,
    root: &Path,
    tx: &mpsc::Sender<AgentMsg>,
) -> Result<AgentMsg> {
    let workdir_rel = Path::new(&req.workdir);
    if !is_inside(workdir_rel) {
        bail!("workdir {} is outside the project", req.workdir);
    }
    let workdir = root.join(workdir_rel);
    if !workdir.is_dir() {
        bail!("workdir {} does not exist in the uploaded project", req.workdir);
    }
    let (program, args) = req.argv.split_first().context("empty argv")?;

    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(&workdir)
        .envs(&req.env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Own process group, so the job and everything it starts can be killed together.
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("starting {program:?}"))?;
    let pid = child.id();
    let _ = tx.send(AgentMsg::Started).await;

    let seq = Arc::new(AtomicU64::new(0));
    let stdout = tokio::spawn(pump(
        child.stdout.take().context("no stdout")?,
        OutputStream::Stdout,
        seq.clone(),
        tx.clone(),
    ));
    let stderr = tokio::spawn(pump(
        child.stderr.take().context("no stderr")?,
        OutputStream::Stderr,
        seq,
        tx.clone(),
    ));

    let limit = req
        .timeout_secs
        .map(Duration::from_secs)
        .unwrap_or(config.default_timeout);
    let (status, timed_out) = match tokio::time::timeout(limit, child.wait()).await {
        Ok(status) => (status?, false),
        Err(_) => {
            kill_group(pid);
            (child.wait().await?, true)
        }
    };
    // Background processes left by the job would keep its output pipes open.
    kill_group(pid);
    let _ = stdout.await;
    let _ = stderr.await;

    Ok(AgentMsg::Exited {
        code: if timed_out { None } else { status.code() },
        timed_out,
    })
}

async fn pump(
    mut reader: impl AsyncRead + Unpin,
    stream: OutputStream,
    seq: Arc<AtomicU64>,
    tx: mpsc::Sender<AgentMsg>,
) {
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let msg = AgentMsg::Output {
                    stream,
                    seq: seq.fetch_add(1, Ordering::Relaxed),
                    data: buf[..n].to_vec(),
                };
                if tx.send(msg).await.is_err() {
                    return;
                }
            }
        }
    }
}

fn kill_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        // SAFETY: killpg only sends a signal; a stale group id fails harmlessly with ESRCH.
        unsafe {
            libc::killpg(pid as libc::pid_t, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn archive(files: &[(&str, &str)]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (path, body) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, path, body.as_bytes())
                .unwrap();
        }
        builder.into_inner().unwrap()
    }

    fn request(argv: &[&str], timeout_secs: Option<u64>) -> JobRequest {
        JobRequest {
            job_id: "test".into(),
            argv: argv.iter().map(|s| s.to_string()).collect(),
            workdir: ".".into(),
            env: BTreeMap::from([("BC_TEST".into(), "yes".into())]),
            timeout_secs,
            archive_bytes: 0,
        }
    }

    async fn run(req: JobRequest, archive: Vec<u8>) -> (Vec<AgentMsg>, PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let config = RunnerConfig {
            work_root: root.path().to_path_buf(),
            default_timeout: Duration::from_secs(30),
        };
        let (tx, mut rx) = mpsc::channel(64);
        let ws = run_job(&config, &req, archive, tx).await;
        let mut msgs = Vec::new();
        while let Some(m) = rx.recv().await {
            msgs.push(m);
        }
        (msgs, ws, root.keep())
    }

    fn output(msgs: &[AgentMsg], want: OutputStream) -> String {
        let mut out = Vec::new();
        for m in msgs {
            if let AgentMsg::Output { stream, data, .. } = m {
                if *stream == want {
                    out.extend_from_slice(data);
                }
            }
        }
        String::from_utf8(out).unwrap()
    }

    #[tokio::test]
    async fn runs_job_in_uploaded_project() {
        let (msgs, ws, root) = run(
            request(
                &["sh", "-c", "cat hello.txt; echo $BC_TEST; echo oops >&2; exit 3"],
                None,
            ),
            archive(&[("hello.txt", "hi from windows\n")]),
        )
        .await;
        assert_eq!(msgs.first(), Some(&AgentMsg::Started));
        assert_eq!(output(&msgs, OutputStream::Stdout), "hi from windows\nyes\n");
        assert_eq!(output(&msgs, OutputStream::Stderr), "oops\n");
        assert_eq!(
            msgs.last(),
            Some(&AgentMsg::Exited {
                code: Some(3),
                timed_out: false
            })
        );
        assert!(!ws.exists(), "workspace must be deleted after the job");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn times_out_and_kills_background_processes() {
        let (msgs, _ws, root) = run(
            request(&["sh", "-c", "sleep 30 & sleep 30"], Some(1)),
            archive(&[]),
        )
        .await;
        assert_eq!(
            msgs.last(),
            Some(&AgentMsg::Exited {
                code: None,
                timed_out: true
            })
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn reports_missing_program() {
        let (msgs, _ws, root) = run(request(&["no-such-program-bc"], None), archive(&[])).await;
        assert!(matches!(msgs.last(), Some(AgentMsg::Failed { .. })));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn refuses_archive_escape() {
        // tar::Builder refuses `..` paths, so write the raw header name directly.
        let mut header = tar::Header::new_gnu();
        header.as_gnu_mut().unwrap().name[..9].copy_from_slice(b"../evil.t");
        header.set_size(0);
        header.set_mode(0o644);
        header.set_cksum();
        let mut builder = tar::Builder::new(Vec::new());
        builder.append(&header, &[][..]).unwrap();
        let bytes = builder.into_inner().unwrap();

        let root = tempfile::tempdir().unwrap();
        assert!(unpack(&bytes, root.path()).is_err());
        assert!(!root.path().parent().unwrap().join("evil.t").exists());
    }
}
