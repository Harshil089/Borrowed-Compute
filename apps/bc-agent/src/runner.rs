//! Runs one job in a fresh workspace and streams its output.
//!
//! Phase one: the workspace is a temporary directory that is deleted after the job, and
//! every process the job started is killed when it ends. The dedicated `_bcjob` user and
//! the Seatbelt profile come in Phase two.

use std::path::{Component, Path, PathBuf};
use std::process::ExitStatus;
#[cfg(unix)]
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_proto::wire::{AgentMsg, JobRequest, JobUsage, OutputStream};
use tokio::io::{AsyncRead, AsyncReadExt};
#[cfg(unix)]
use tokio::process::{ChildStderr, ChildStdout};
use tokio::sync::mpsc;

const READ_CHUNK: usize = 16 * 1024;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Parent directory for job workspaces.
    pub work_root: PathBuf,
    /// Used when the job sets no timeout.
    pub default_timeout: Duration,
    /// Localhost ports jobs may not connect to, beyond the sandbox's own list: the
    /// built-in cloud's, when this Mac hosts it.
    pub denied_ports: Vec<u16>,
    /// The job is a guest's: someone on another account, trusted less than the owner's own
    /// devices, so the sandbox lets it reach only web ports (see `sandbox::profile`).
    pub guest: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig {
            work_root: std::env::temp_dir().join("borrowed-compute"),
            default_timeout: Duration::from_secs(60 * 60),
            denied_ports: Vec::new(), guest: false,
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
    run_job_with(config, req, archive, tx, None).await
}

/// A running job, as services see it: enough to find the ports it listens on and to
/// connect to them.
#[derive(Clone)]
pub struct StartedJob {
    /// The job's process, which leads its process group on a Mac.
    pub pid: u32,
    #[cfg(windows)]
    pub job: crate::sandbox_windows::JobHandle,
}

impl StartedJob {
    /// TCP ports the job's processes listen on. Forwarding is limited to these, so a paired
    /// device reaches its own service and nothing else on this device's localhost.
    pub fn listening_ports(&self) -> Vec<u16> {
        #[cfg(windows)]
        return self.job.listening_ports();
        #[cfg(not(windows))]
        return crate::services::listening_ports(self.pid);
    }
}

/// [`run_job`], also sending the started job to `started` once it runs, so services can
/// find and reach the ports the job listens on.
pub async fn run_job_with(
    config: &RunnerConfig,
    req: &JobRequest,
    archive: Vec<u8>,
    tx: mpsc::Sender<AgentMsg>,
    pid: Option<tokio::sync::oneshot::Sender<StartedJob>>,
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
    let last = match execute(config, req, &path.join(PROJECT_DIR), &path.join(CACHE_DIR), &tx, pid).await {
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
        // git fails outright when it cannot read ~/.gitconfig, which the sandbox hides.
        ("GIT_CONFIG_GLOBAL", if cfg!(windows) { "NUL" } else { "/dev/null" }.into()),
    ];
    // Windows tools take their temporary and settings folders from these instead.
    if cfg!(windows) {
        env.extend([("TEMP", dir("tmp")), ("TMP", dir("tmp")), ("APPDATA", dir("appdata")), ("LOCALAPPDATA", dir("localappdata"))]);
    }
    // rustup normally finds its toolchains through CARGO_HOME's sibling; moving CARGO_HOME
    // must not hide them, so name RUSTUP_HOME explicitly.
    if std::env::var_os("RUSTUP_HOME").is_none() {
        if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
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
    pid_sink: Option<tokio::sync::oneshot::Sender<StartedJob>>,
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
    let program_path = find_program(program, lookup.as_deref(), &workdir);
    // The workspace holds both the project and the job's caches.
    let workspace = root.parent().unwrap_or(root);
    let mut envs: Vec<(std::ffi::OsString, std::ffi::OsString)> = Vec::new();
    if let Some(path) = &path {
        envs.push(("PATH".into(), path.clone()));
    }
    envs.extend(req.env.iter().map(|(k, v)| (k.into(), v.into())));
    // After the job's own env, so a job cannot point a cache back outside its workspace.
    envs.extend(isolated_env(cache).into_iter().map(|(k, v)| (k.into(), v)));

    #[cfg(unix)]
    let (t0, pid, group, mut wait, stdout, stderr) = {
        let wrapper = crate::sandbox::wrapper(workspace, path.as_deref(), &config.denied_ports, config.guest);
        // Under the sandbox, sandbox-exec itself always starts, so check the program here to
        // keep a clear "not found" instead of a bare exit code.
        if wrapper.is_some() && !(program_path.is_absolute() && program_path.is_file()) {
            bail!("starting {program:?}: not found on this Mac");
        }
        let mut cmd = match wrapper {
            Some(wrapper) => {
                let mut cmd = std::process::Command::new(&wrapper[0]);
                cmd.args(&wrapper[1..]).arg(&program_path);
                cmd
            }
            None => std::process::Command::new(&program_path),
        };
        cmd.args(args)
            .current_dir(&workdir)
            .envs(envs)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Own process group, so the job and everything it starts can be killed together.
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);

        let t0 = Instant::now();
        let mut child = cmd
            .spawn()
            .with_context(|| format!("starting {program:?}"))?;
        let pid = child.id();
        let stdout = ChildStdout::from_std(child.stdout.take().context("no stdout")?)?;
        let stderr = ChildStderr::from_std(child.stderr.take().context("no stderr")?)?;
        // The wait runs on its own thread and reaps the process even if this job is
        // abandoned: the group guard kills the job, and the wait then returns.
        let wait = tokio::task::spawn_blocking(move || wait_with_usage(child));
        (t0, pid, GroupGuard(Some(pid)), wait, stdout, stderr)
    };
    #[cfg(windows)]
    let (t0, pid, group, mut wait, stdout, stderr) = {
        // The AppContainer starts the program itself, so a missing one fails at start.
        if !(program_path.is_absolute() && program_path.is_file()) {
            bail!("starting {program:?}: not found on this PC");
        }
        let confinement = crate::sandbox_windows::Confinement::for_job(config.guest);
        let t0 = Instant::now();
        let spawned = crate::sandbox_windows::spawn(&program_path, args, &workdir, &envs, workspace, confinement)
            .with_context(|| format!("starting {program:?}"))?;
        let process = spawned.process;
        let wait = tokio::task::spawn_blocking(move || {
            crate::sandbox_windows::wait(&process).map(|code| (std::os::windows::process::ExitStatusExt::from_raw(code), None))
        });
        let stdout = tokio::fs::File::from_std(spawned.stdout);
        let stderr = tokio::fs::File::from_std(spawned.stderr);
        (t0, spawned.pid, spawned.group, wait, stdout, stderr)
    };

    if let Some(sink) = pid_sink {
        #[cfg(windows)]
        let _ = sink.send(StartedJob { pid, job: group.handle() });
        #[cfg(not(windows))]
        let _ = sink.send(StartedJob { pid });
    }
    let _ = tx.send(AgentMsg::Started).await;

    let seq = Arc::new(AtomicU64::new(0));
    let stdout = tokio::spawn(pump(stdout, OutputStream::Stdout, seq.clone(), tx.clone()));
    let stderr = tokio::spawn(pump(stderr, OutputStream::Stderr, seq, tx.clone()));

    // Services run until stopped, so they have no limit.
    let limit = match req.timeout_secs {
        _ if req.service => None,
        Some(secs) => Some(Duration::from_secs(secs)),
        None => Some(config.default_timeout),
    };
    let waited = match limit {
        Some(limit) => tokio::time::timeout(limit, &mut wait).await.ok(),
        None => Some((&mut wait).await),
    };
    let ((status, usage), timed_out): ((ExitStatus, Option<JobUsage>), bool) = match waited {
        Some(waited) => (waited??, false),
        None => {
            group.kill();
            (wait.await??, true)
        }
    };
    let wall = t0.elapsed();
    // The Job Object counts every process the job ran, so read it before it goes.
    #[cfg(windows)]
    let usage = usage.or_else(|| {
        group.usage().map(|(user_ms, system_ms, max_rss_bytes)| JobUsage { wall_ms: 0, user_ms, system_ms, max_rss_bytes })
    });
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
/// started never outlive it. On Windows `sandbox_windows::Group` does the same.
#[cfg(unix)]
struct GroupGuard(Option<u32>);

#[cfg(unix)]
impl GroupGuard {
    fn kill(&self) {
        kill_group(self.0);
    }
}

#[cfg(unix)]
impl Drop for GroupGuard {
    fn drop(&mut self) {
        kill_group(self.0.take());
    }
}

#[cfg(unix)]
fn kill_group(pid: Option<u32>) {
    if let Some(pid) = pid {
        // SAFETY: killpg only sends a signal; a stale group id fails harmlessly with ESRCH.
        unsafe {
            libc::killpg(pid as libc::pid_t, libc::SIGKILL);
        }
    }
}

/// The program to start: a path relative to the job's folder when it names one, else the
/// first match on `path`. Windows also tries each `PATHEXT` extension, so `npm` finds
/// `npm.cmd`.
fn find_program(program: &str, path: Option<&std::ffi::OsStr>, workdir: &Path) -> PathBuf {
    let extensions: Vec<String> = if cfg!(windows) {
        let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
        let pathext = pathext.split(';').filter(|e| !e.is_empty()).map(str::to_owned);
        // A bare name only when it already has an extension: Node puts an extensionless
        // shell script `npm` beside `npm.cmd`, which Windows cannot start.
        let bare = Path::new(program).extension().is_some().then(String::new);
        bare.into_iter().chain(pathext).collect()
    } else {
        vec![String::new()]
    };
    let with_extension = |base: PathBuf| {
        extensions.iter().map(move |e| {
            let mut name = base.clone().into_os_string();
            name.push(e);
            PathBuf::from(name)
        })
    };
    let names_a_path = program.contains('/') || (cfg!(windows) && program.contains('\\'));
    if names_a_path {
        // Without `.` parts and with native separators: cmd.exe reads `/name` in
        // `project\./name.cmd` as a switch and tries to run the folder.
        let relative: PathBuf = Path::new(program).components().filter(|c| !matches!(c, Component::CurDir)).collect();
        let path = workdir.join(relative);
        return with_extension(path.clone()).find(|c| c.is_file()).unwrap_or(path);
    }
    path.into_iter()
        .flat_map(std::env::split_paths)
        .flat_map(|dir| with_extension(dir.join(program)))
        .find(|c| c.is_file())
        .unwrap_or_else(|| PathBuf::from(program))
}

#[cfg(test)]
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
            denied_ports: Vec::new(), guest: false,
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

    #[cfg(unix)]
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

    #[cfg(unix)]
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

    #[cfg(unix)]
    #[tokio::test]
    async fn abandoned_job_kills_its_process_group() {
        let root = tempfile::tempdir().unwrap();
        let config = RunnerConfig {
            work_root: root.path().to_path_buf(),
            default_timeout: Duration::from_secs(30),
            denied_ports: Vec::new(), guest: false,
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

    #[cfg(unix)]
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

    #[cfg(unix)]
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

    #[cfg(unix)]
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

    /// The Windows runner: an AppContainer and a Job Object around each job.
    #[cfg(windows)]
    async fn run_within(req: JobRequest, archive: Vec<u8>) -> Vec<AgentMsg> {
        tokio::time::timeout(Duration::from_secs(60), run(req, archive))
            .await
            .expect("the job's output closed once it ended")
            .0
    }

    #[cfg(windows)]
    fn exited(msgs: &[AgentMsg]) -> (Option<i32>, bool, Option<JobUsage>) {
        match msgs.last() {
            Some(AgentMsg::Exited { code, timed_out, usage }) => (*code, *timed_out, *usage),
            other => panic!("expected Exited, got {other:?}"),
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_runs_job_in_uploaded_project() {
        let msgs = run_within(
            request(&["cmd", "/c", "type hello.txt & echo %BC_TEST% & echo oops 1>&2 & exit 3"], None),
            archive(&[("hello.txt", "hi\n")]),
        )
        .await;
        let out = output(&msgs, OutputStream::Stdout);
        assert!(out.contains("hi") && out.contains("yes"), "{out}");
        assert!(output(&msgs, OutputStream::Stderr).contains("oops"));
        let (code, timed_out, usage) = exited(&msgs);
        assert_eq!((code, timed_out), (Some(3), false));
        assert!(usage.is_some(), "the Job Object reports usage");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_runs_batch_files_from_the_project() {
        let msgs = run_within(request(&["./hello.cmd"], None), archive(&[("hello.cmd", "@echo from batch\r\n")])).await;
        assert!(output(&msgs, OutputStream::Stdout).contains("from batch"), "{msgs:?}");
        assert_eq!(exited(&msgs).0, Some(0));
    }

    /// npm is a batch file in Program Files on Windows; the runner starts the script it
    /// wraps with Node directly.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_runs_npm_from_program_files() {
        if !find_program("npm", std::env::var_os("PATH").as_deref(), Path::new(".")).is_absolute() {
            return eprintln!("npm is not installed; skipping");
        }
        // A sandboxed job reads only what grants "ALL APPLICATION PACKAGES" access, as a
        // normal Program Files install does. Where npm's files do not, that is this
        // machine's setup, not batch handling.
        let npm = find_program("npm", std::env::var_os("PATH").as_deref(), Path::new("."));
        let cli = npm.with_file_name("node_modules").join("npm").join("bin").join("npm-cli.js");
        let read = format!("require('fs').readFileSync({:?})", cli.display().to_string());
        let node = run_within(request(&["node", "-e", &read], None), Vec::new()).await;
        if exited(&node).0 != Some(0) {
            return eprintln!("npm's files are not open to AppContainers on this machine; skipping: {}", output(&node, OutputStream::Stderr));
        }
        let msgs = run_within(request(&["npm", "--version"], None), Vec::new()).await;
        assert_eq!(exited(&msgs).0, Some(0), "{}", output(&msgs, OutputStream::Stderr));
    }

    /// Node loads the project's own scripts inside the AppContainer.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_runs_node_scripts_from_the_project() {
        if !find_program("node", std::env::var_os("PATH").as_deref(), Path::new(".")).is_absolute() {
            return eprintln!("node is not installed; skipping");
        }
        let files = [("main.js", "console.log(require('./lib').word)"), ("lib.js", "exports.word = 'from a module'")];
        let msgs = run_within(request(&["node", "main.js"], None), archive(&files)).await;
        assert!(output(&msgs, OutputStream::Stdout).contains("from a module"), "{}", output(&msgs, OutputStream::Stderr));
    }

    /// A job reads and writes its workspace but nothing else of the owner's.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_sandbox_confines_jobs_to_their_workspace() {
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "do not read").unwrap();
        let script = format!(
            "type hello.txt & echo inside> made.txt & type made.txt & \
             (type \"{0}\" && echo READ-OUTSIDE) & \
             (echo x> \"{1}\\made.txt\" && echo WROTE-OUTSIDE) & \
             (dir \"%USERPROFILE%\" >NUL 2>&1 && echo LISTED-HOME) & exit 0",
            secret.display(),
            outside.path().display()
        );
        let msgs = run_within(request(&["cmd", "/c", &script], None), archive(&[("hello.txt", "hi\n")])).await;
        let out = output(&msgs, OutputStream::Stdout);
        assert!(out.contains("hi") && out.contains("inside"), "{out}");
        for leak in ["do not read", "READ-OUTSIDE", "WROTE-OUTSIDE", "LISTED-HOME"] {
            assert!(!out.contains(leak), "{leak} in {out}");
        }
        assert!(!outside.path().join("made.txt").exists());
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_times_out_and_counts_cpu() {
        let msgs = run_within(request(&["cmd", "/c", "for /l %i in (0,0,1) do @rem"], Some(2)), Vec::new()).await;
        let (code, timed_out, usage) = exited(&msgs);
        assert_eq!((code, timed_out), (None, true));
        assert!(usage.expect("usage").user_ms > 0);
    }

    /// A process the job left running in the background ends with the job, so the job's
    /// output closes and the run returns.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_background_processes_end_with_the_job() {
        let msgs = run_within(
            request(&["cmd", "/c", "start", "/b", "cmd", "/c", "for /l %i in (0,0,1) do @rem"], None),
            Vec::new(),
        )
        .await;
        assert!(!exited(&msgs).1);
    }

    /// pnpm links each package into `node_modules` from its store, and the package finds
    /// its own dependencies beside the link's target. That only works when Node resolves
    /// the link, which it can inside the AppContainer once the folders above can be read.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_node_resolves_pnpm_links() {
        if !find_program("node", std::env::var_os("PATH").as_deref(), Path::new(".")).is_absolute() {
            return eprintln!("node is not installed; skipping");
        }
        let files = [
            ("main.js", "console.log(require('a'))"),
            ("node_modules/.pnpm/a@1/node_modules/a/index.js", "module.exports = 'a needs ' + require('b')"),
            ("node_modules/.pnpm/a@1/node_modules/b/index.js", "module.exports = 'b'"),
        ];
        let script = "mklink /J node_modules\\a node_modules\\.pnpm\\a@1\\node_modules\\a >NUL && node main.js";
        let msgs = run_within(request(&["cmd", "/c", script], None), archive(&files)).await;
        assert!(output(&msgs, OutputStream::Stdout).contains("a needs b"), "{}", output(&msgs, OutputStream::Stderr));
    }

    /// Global packages npm installs (yarn, pnpm) come with a batch file that only starts
    /// their script with Node; the runner starts the script directly.
    #[cfg(windows)]
    #[test]
    fn windows_runs_npm_shims_with_node() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("node_modules").join("yarn").join("bin");
        std::fs::create_dir_all(&script).unwrap();
        std::fs::write(script.join("yarn.js"), "").unwrap();
        std::fs::write(dir.path().join("node.exe"), "").unwrap();
        let shim = dir.path().join("yarn.cmd");
        std::fs::write(&shim, "@ECHO off\r\nendLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & \"%_prog%\"  \"%dp0%\\node_modules\\yarn\\bin\\yarn.js\" %*\r\n").unwrap();
        let (application, line) = crate::sandbox_windows::command_line_for_tests(&shim, &["install".into()], Path::new("C:\\work"));
        assert_eq!(application, dir.path().join("node.exe").display().to_string());
        assert!(line.ends_with("yarn.js install"), "{line}");
    }

    #[cfg(windows)]
    #[test]
    fn windows_finds_programs_through_pathext() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tool"), "#!/bin/sh").unwrap();
        std::fs::write(dir.path().join("tool.cmd"), "@echo hi").unwrap();
        let found = find_program("tool", Some(dir.path().as_os_str()), dir.path());
        assert!(found.to_string_lossy().eq_ignore_ascii_case(&dir.path().join("tool.cmd").to_string_lossy()), "{found:?}");
    }

}
