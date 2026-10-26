//! Runs one job in a fresh workspace and streams its output.
//!
//! Phase one: the workspace is a temporary directory that is deleted after the job, and
//! every process the job started is killed when it ends. The dedicated `_bcjob` user and
//! the Seatbelt profile come in Phase two.

use std::path::{Component, Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_proto::wire::{AgentMsg, JobRequest, JobUsage, OutputStream};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{ChildStderr, ChildStdout};
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
    let last = match execute(config, req, &path.join(PROJECT_DIR), &path.join(CACHE_DIR), &tx).await {
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

/// The uploaded project, inside the job's workspace.
const PROJECT_DIR: &str = "project";
/// Every cache and temporary file the job's tools write, also inside the workspace, so
/// deleting the workspace deletes them too.
const CACHE_DIR: &str = "cache";

async fn prepare(config: &RunnerConfig, archive: Vec<u8>) -> Result<tempfile::TempDir> {
    tokio::fs::create_dir_all(&config.work_root)
        .await
        .with_context(|| format!("creating {}", config.work_root.display()))?;
    let workspace = tempfile::Builder::new()
        .prefix("job-")
        .tempdir_in(&config.work_root)?;
    let root = workspace.path().join(PROJECT_DIR);
    tokio::fs::create_dir_all(workspace.path().join(CACHE_DIR).join("tmp")).await?;
    tokio::fs::create_dir_all(&root).await?;
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

/// Points every common tool's dependency cache and temporary files into `cache`, which
/// is deleted with the workspace. Nothing a job downloads (crates, npm packages, wheels,
/// Go modules, Maven artifacts) outlives it: each run starts from empty caches, by the
/// owner's choice of zero data on the Mac. Installed toolchains are left where they are.
fn isolated_env(cache: &Path) -> Vec<(&'static str, std::ffi::OsString)> {
    let dir = |name: &str| cache.join(name).into_os_string();
    let mut env = vec![
        ("TMPDIR", dir("tmp")),
        ("XDG_CACHE_HOME", dir("xdg")),
        // Rust: downloaded crates and git checkouts. Toolchains stay in RUSTUP_HOME.
        ("CARGO_HOME", dir("cargo")),
        // JavaScript package managers.
        ("npm_config_cache", dir("npm")),
        ("npm_config_store_dir", dir("pnpm")),
        ("YARN_CACHE_FOLDER", dir("yarn")),
        ("BUN_INSTALL_CACHE_DIR", dir("bun")),
        // Python.
        ("PIP_CACHE_DIR", dir("pip")),
        ("UV_CACHE_DIR", dir("uv")),
        ("POETRY_CACHE_DIR", dir("poetry")),
        ("PYTHONPYCACHEPREFIX", dir("pycache")),
        // Go: modules and the build cache.
        ("GOMODCACHE", dir("go-mod")),
        ("GOCACHE", dir("go-build")),
        ("GOPATH", dir("go")),
        // JVM and .NET.
        ("GRADLE_USER_HOME", dir("gradle")),
        ("MAVEN_OPTS", format!("-Dmaven.repo.local={}", cache.join("m2").display()).into()),
        ("NUGET_PACKAGES", dir("nuget")),
        ("DOTNET_CLI_HOME", dir("dotnet")),
    ];
    // rustup normally finds its toolchains through CARGO_HOME's sibling; moving CARGO_HOME
    // must not hide them, so name RUSTUP_HOME explicitly.
    if std::env::var_os("RUSTUP_HOME").is_none() {
        if let Some(home) = std::env::var_os("HOME") {
            let rustup = Path::new(&home).join(".rustup");
            if rustup.is_dir() {
                env.push(("RUSTUP_HOME", rustup.into_os_string()));
            }
        }
    }
    env
}

/// The PATH a job runs with. An app started from Finder gets only the system folders
/// (`/usr/bin:/bin:/usr/sbin:/sbin`), which hides tools installed by Homebrew, rustup,
/// nvm and the like. Jobs get the PATH of the user's login shell instead, the one their
/// terminal uses, asked once and then reused.
async fn job_path() -> Option<std::ffi::OsString> {
    static PATH: tokio::sync::OnceCell<Option<std::ffi::OsString>> = tokio::sync::OnceCell::const_new();
    PATH.get_or_init(login_shell_path).await.clone()
}

#[cfg(unix)]
async fn login_shell_path() -> Option<std::ffi::OsString> {
    const MARK: &str = "__BC_PATH__";
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    // Interactive and login, so both .zprofile and .zshrc (where nvm and the like live)
    // have run. The marks separate PATH from anything those files print.
    let mut cmd = tokio::process::Command::new(shell);
    cmd.args(["-l", "-i", "-c", &format!("printf '{MARK}%s{MARK}' \"$PATH\"")])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let out = tokio::time::timeout(Duration::from_secs(10), cmd.output()).await.ok()?.ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let shell_path = text.split(MARK).nth(1).filter(|p| !p.is_empty())?;
    // Keep anything the agent itself had that the shell does not.
    let mut dirs: Vec<std::path::PathBuf> = std::env::split_paths(shell_path).collect();
    if let Some(own) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&own) {
            if !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    std::env::join_paths(dirs).ok()
}

#[cfg(not(unix))]
async fn login_shell_path() -> Option<std::ffi::OsString> {
    None
}

async fn execute(
    config: &RunnerConfig,
    req: &JobRequest,
    root: &Path,
    cache: &Path,
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

    let path = job_path().await;
    // Resolve the program on the job's PATH; Command looks it up on the agent's own.
    let lookup = path.clone().or_else(|| std::env::var_os("PATH"));
    let program_path = lookup
        .as_ref()
        .and_then(|p| std::env::split_paths(p).map(|d| d.join(program)).find(|c| c.is_file()))
        .filter(|_| !program.contains('/'))
        .unwrap_or_else(|| PathBuf::from(program));
    // The workspace holds both the project and the job's caches.
    let workspace = root.parent().unwrap_or(root);
    let wrapper = crate::sandbox::wrapper(workspace, path.as_deref());
    // Under the sandbox, sandbox-exec itself always starts, so check the program here to
    // keep a clear "not found" instead of a bare exit code.
    if wrapper.is_some() {
        let found = if program.contains('/') {
            workdir.join(program).is_file()
        } else {
            program_path.is_absolute() && program_path.is_file()
        };
        if !found {
            bail!("starting {program:?}: not found on this Mac");
        }
    }
    let mut cmd = match wrapper {
        Some(wrapper) => {
            let mut cmd = std::process::Command::new(&wrapper[0]);
            cmd.args(&wrapper[1..]).arg(&program_path);
            cmd
        }
        None => std::process::Command::new(&program_path),
    };
    if let Some(path) = &path {
        cmd.env("PATH", path);
    }
    cmd.args(args)
        .current_dir(&workdir)
        .envs(&req.env)
        // After the job's own env, so a job cannot point a cache back outside its workspace.
        .envs(isolated_env(cache))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group, so the job and everything it starts can be killed together.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let t0 = Instant::now();
    let mut child = cmd
        .spawn()
        .with_context(|| format!("starting {program:?}"))?;
    let pid = child.id();
    let group = GroupGuard(Some(pid));
    let _ = tx.send(AgentMsg::Started).await;

    let seq = Arc::new(AtomicU64::new(0));
    let stdout = tokio::spawn(pump(
        ChildStdout::from_std(child.stdout.take().context("no stdout")?)?,
        OutputStream::Stdout,
        seq.clone(),
        tx.clone(),
    ));
    let stderr = tokio::spawn(pump(
        ChildStderr::from_std(child.stderr.take().context("no stderr")?)?,
        OutputStream::Stderr,
        seq,
        tx.clone(),
    ));

    // Services run until stopped, so they have no limit.
    let limit = match req.timeout_secs {
        _ if req.service => None,
        Some(secs) => Some(Duration::from_secs(secs)),
        None => Some(config.default_timeout),
    };
    // The wait runs on its own thread and reaps the process even if this job is abandoned:
    // the group guard kills the job, and the wait then returns.
    let mut wait = tokio::task::spawn_blocking(move || wait_with_usage(child));
    let waited = match limit {
        Some(limit) => tokio::time::timeout(limit, &mut wait).await.ok(),
        None => Some((&mut wait).await),
    };
    let ((status, usage), timed_out) = match waited {
        Some(waited) => (waited??, false),
        None => {
            kill_group(Some(pid));
            (wait.await??, true)
        }
    };
    let wall = t0.elapsed();
    // Background processes left by the job would keep its output pipes open.
    drop(group);
    let _ = stdout.await;
    let _ = stderr.await;

    Ok(AgentMsg::Exited {
        code: if timed_out { None } else { status.code() },
        timed_out,
        usage: usage.map(|u| JobUsage { wall_ms: wall.as_millis() as u64, ..u }),
    })
}

/// Waits for the job's process and reads what it and the processes it waited for used.
#[cfg(unix)]
fn wait_with_usage(child: std::process::Child) -> std::io::Result<(ExitStatus, Option<JobUsage>)> {
    use std::os::unix::process::ExitStatusExt;

    let pid = child.id() as libc::pid_t;
    // Waiting by pid reaps the process; `child` is never waited on, so dropping it is safe.
    drop(child);
    let mut status = 0;
    // SAFETY: rusage is plain data that wait4 fills in.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    loop {
        // SAFETY: valid pointers to locals; pid is our own unreaped child.
        if unsafe { libc::wait4(pid, &mut status, 0, &mut ru) } == pid {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
    let ms = |t: libc::timeval| t.tv_sec as u64 * 1000 + t.tv_usec as u64 / 1000;
    // macOS reports peak memory in bytes, Linux in kilobytes.
    let rss_unit = if cfg!(target_os = "macos") { 1 } else { 1024 };
    let usage = JobUsage {
        wall_ms: 0,
        user_ms: ms(ru.ru_utime),
        system_ms: ms(ru.ru_stime),
        max_rss_bytes: ru.ru_maxrss as u64 * rss_unit,
    };
    Ok((ExitStatus::from_raw(status), Some(usage)))
}

#[cfg(not(unix))]
fn wait_with_usage(mut child: std::process::Child) -> std::io::Result<(ExitStatus, Option<JobUsage>)> {
    Ok((child.wait()?, None))
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

/// Kills the job's process group when dropped: at the normal end of a job, and also when
/// the job is abandoned mid-run (cancelled, or the client went away), so processes it
/// started never outlive it.
struct GroupGuard(Option<u32>);

impl Drop for GroupGuard {
    fn drop(&mut self) {
        kill_group(self.0.take());
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
    // Phase one targets Mac agents; this keeps a Windows agent from leaking the job.
    #[cfg(not(unix))]
    if let Some(pid) = pid {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }
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
            service: false,
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
        match msgs.last() {
            Some(AgentMsg::Exited { code: Some(3), timed_out: false, usage: Some(u) }) => {
                assert!(u.wall_ms < 10_000, "{u:?}");
                assert!(u.max_rss_bytes > 100_000, "peak memory looks wrong: {u:?}");
            }
            other => panic!("unexpected end {other:?}"),
        }
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
        assert!(matches!(
            msgs.last(),
            Some(AgentMsg::Exited { code: None, timed_out: true, usage: Some(_) })
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn abandoned_job_kills_its_process_group() {
        let root = tempfile::tempdir().unwrap();
        let config = RunnerConfig {
            work_root: root.path().to_path_buf(),
            default_timeout: Duration::from_secs(30),
        };
        let req = request(&["sh", "-c", "sleep 4711 & sleep 4712"], None);
        let (tx, mut rx) = mpsc::channel(64);
        let job = tokio::spawn(async move { run_job(&config, &req, archive(&[]), tx).await });
        assert_eq!(rx.recv().await, Some(AgentMsg::Started));
        tokio::time::sleep(Duration::from_millis(300)).await;

        // What the server does when the client cancels or disconnects.
        job.abort();
        let _ = job.await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        let left = std::process::Command::new("pgrep")
            .args(["-f", "sleep 471[12]"])
            .output()
            .unwrap();
        assert!(
            left.stdout.is_empty(),
            "processes outlived the job: {}",
            String::from_utf8_lossy(&left.stdout)
        );
    }

    #[tokio::test]
    async fn counts_cpu_of_processes_the_job_waits_for() {
        // The busy loop runs in a child of `sh`, which waits for it.
        let (msgs, _ws, root) = run(
            request(&["sh", "-c", "sh -c 'i=0; while [ $i -lt 300000 ]; do i=$((i+1)); done'"], None),
            archive(&[]),
        )
        .await;
        match msgs.last() {
            Some(AgentMsg::Exited { code: Some(0), usage: Some(u), .. }) => {
                assert!(u.user_ms + u.system_ms >= 50, "child CPU not counted: {u:?}");
                assert!(u.wall_ms >= u.user_ms / 2, "{u:?}");
            }
            other => panic!("unexpected end {other:?}"),
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn caches_live_in_the_workspace_and_go_with_it() {
        // The job writes into every cache it is given and prints where they are.
        let script = "for d in \"$CARGO_HOME\" \"$npm_config_cache\" \"$PIP_CACHE_DIR\" \"$GOMODCACHE\" \"$TMPDIR\"; \
                      do mkdir -p \"$d\" && echo x > \"$d/written\" && echo \"$d\"; done";
        let mut req = request(&["sh", "-c", script], None);
        // A job trying to keep its cache elsewhere is overridden.
        req.env.insert("CARGO_HOME".into(), "/tmp/bc-escape".into());
        let (msgs, ws, root) = run(req, archive(&[])).await;
        let printed = output(&msgs, OutputStream::Stdout);
        let dirs: Vec<&str> = printed.lines().collect();
        assert_eq!(dirs.len(), 5, "{printed}");
        for dir in &dirs {
            assert!(Path::new(dir).starts_with(&ws), "{dir} is outside the workspace {}", ws.display());
        }
        assert!(!ws.exists(), "the workspace and its caches must be deleted");
        assert!(!Path::new("/tmp/bc-escape/written").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn jobs_get_the_login_shell_path() {
        let path = job_path().await.expect("a login shell PATH");
        let dirs: Vec<_> = std::env::split_paths(&path).collect();
        assert!(dirs.iter().any(|d| d == Path::new("/usr/bin")), "{path:?}");
    }

    /// A job reads and writes its workspace but nothing outside it.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn sandbox_confines_jobs_to_their_workspace() {
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "do not read").unwrap();
        let script = format!(
            "cat hello.txt && echo inside > made.txt && cat made.txt; \
             cat '{0}' && echo READ-OUTSIDE; \
             echo x > '{1}' && echo WROTE-OUTSIDE; \
             ls ~ >/dev/null 2>&1 && echo LISTED-HOME; true",
            secret.display(),
            outside.path().join("escaped.txt").display()
        );
        let (msgs, _ws, root) = run(request(&["sh", "-c", &script], None), archive(&[("hello.txt", "hi\n")])).await;
        let out = output(&msgs, OutputStream::Stdout);
        assert!(out.contains("hi\ninside\n"), "the workspace works: {out}");
        assert!(!out.contains("READ-OUTSIDE"), "read a file outside the workspace: {out}");
        assert!(!out.contains("WROTE-OUTSIDE"), "wrote outside the workspace: {out}");
        assert!(!out.contains("LISTED-HOME"), "listed the home folder: {out}");
        assert!(!outside.path().join("escaped.txt").exists());
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
