//! Runs each job on Windows inside its own AppContainer, the Windows counterpart of the
//! Seatbelt profile in `sandbox.rs`, and inside a Job Object.
//!
//! An AppContainer process is denied by default: it reads only what grants access to its
//! own SID, to one of its capabilities, or to "ALL APPLICATION PACKAGES". So a job:
//! - reads and writes its own workspace, which grants its SID full access. Each job gets
//!   a fresh AppContainer, so jobs cannot read each other's workspaces either;
//! - reads and runs the system folders and Program Files, which Windows opens to all
//!   application packages;
//! - reads and runs the toolchains inside the user's profile (rustup, nvm, Scoop and
//!   similar, plus every PATH folder inside the profile), which grant read and execute to
//!   the `borrowedComputeToolchains` capability every job carries;
//! - cannot read anything else of the owner's: Documents, .ssh, browser profiles.
//!
//! Network: owner jobs reach the internet and the local network; a guest's job (someone
//! on another account) reaches the internet only. AppContainers cannot reach this PC's
//! localhost at all, so the owner's local services, and the built-in cloud, are out of
//! reach. The same rule keeps the agent from reaching a job's own server, so a forwarded
//! connection goes through a relay process started inside the job's AppContainer
//! ([`JobHandle::relay`]), which talks to the server over loopback inside the container
//! and to the agent over pipes.
//!
//! The Job Object kills every process the job started when it ends, and reports the CPU
//! time and peak memory the job used. `BC_NO_SANDBOX=1` on the agent turns the
//! AppContainer off for debugging; the Job Object stays.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::mem::{size_of, size_of_val, zeroed};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use windows_sys::core::BOOL;
use windows_sys::Win32::Foundation::{LocalFree, SetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT};
use windows_sys::Win32::Security::Authorization::{
    GetNamedSecurityInfoW, SetEntriesInAclW, SetNamedSecurityInfoW, EXPLICIT_ACCESS_W, GRANT_ACCESS,
    NO_MULTIPLE_TRUSTEE, SE_FILE_OBJECT, TRUSTEE_IS_SID, TRUSTEE_IS_WELL_KNOWN_GROUP, TRUSTEE_W,
};
use windows_sys::Win32::Security::Isolation::{CreateAppContainerProfile, DeleteAppContainerProfile};
use windows_sys::Win32::NetworkManagement::IpHelper::{
    GetExtendedTcpTable, MIB_TCP6TABLE_OWNER_PID, MIB_TCPTABLE_OWNER_PID, TCP_TABLE_OWNER_PID_LISTENER,
};
use windows_sys::Win32::Networking::WinSock::{AF_INET, AF_INET6};
use windows_sys::Win32::Security::{
    CreateWellKnownSid, DeriveCapabilitySidsFromName, EqualSid, FreeSid, GetAce, GetSecurityDescriptorControl,
    InitializeSecurityDescriptor, SetFileSecurityW, SetSecurityDescriptorControl, SetSecurityDescriptorDacl,
    WinBuiltinAnyPackageSid, WinCapabilityInternetClientSid, WinCapabilityPrivateNetworkClientServerSid,
    ACCESS_ALLOWED_ACE, ACL, DACL_SECURITY_INFORMATION, NO_INHERITANCE, PSID, SECURITY_CAPABILITIES,
    SECURITY_DESCRIPTOR, SECURITY_MAX_SID_SIZE, SID_AND_ATTRIBUTES, SUB_CONTAINERS_AND_OBJECTS_INHERIT,
    WELL_KNOWN_SID_TYPE,
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_ALL_ACCESS, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ, FILE_LIST_DIRECTORY, FILE_READ_ATTRIBUTES,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectBasicAccountingInformation, JobObjectBasicProcessIdList,
    JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_BASIC_PROCESS_ID_LIST, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess, InitializeProcThreadAttributeList,
    ResumeThread, TerminateProcess, UpdateProcThreadAttribute, WaitForSingleObject, CREATE_NO_WINDOW,
    CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, EXTENDED_STARTUPINFO_PRESENT, INFINITE,
    LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

/// `SE_GROUP_ENABLED`, which windows-sys keeps in a module of its own.
const SE_GROUP_ENABLED: u32 = 4;
// ACE and security descriptor constants windows-sys spreads over other modules.
const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
const INHERIT_ONLY_ACE: u8 = 8;
const SECURITY_DESCRIPTOR_REVISION: u32 = 1;
const SE_DACL_AUTO_INHERITED: u16 = 0x0400;
const SE_DACL_PROTECTED: u16 = 0x1000;
const GENERIC_ALL: u32 = 0x1000_0000;
const GENERIC_EXECUTE: u32 = 0x2000_0000;
const GENERIC_READ: u32 = 0x8000_0000;

/// The argument that makes the agent's own program a relay ([`relay_main`]).
pub const RELAY_ARG: &str = "--bc-relay";
/// The arguments for the one-time setup that needs an administrator ([`setup`]).
pub const SETUP_ARG: &str = "--bc-setup";
pub const SETUP_CHECK_ARG: &str = "--bc-setup-check";
/// The firewall rule [`setup`] adds for the app.
const FIREWALL_RULE: &str = "Borrowed Compute";

/// The capability every job carries, which the toolchain folders grant read and execute.
const TOOLCHAIN_CAPABILITY: &str = "borrowedComputeToolchains";

/// Toolchain folders inside the user's profile, relative to `%USERPROFILE%`.
const PROFILE_TOOLCHAINS: &[&str] = &[".rustup", ".cargo\\bin", ".bun", ".deno", "go\\bin", "scoop"];
/// Relative to `%LOCALAPPDATA%`: per-user installs of Python, Node and the like.
const LOCAL_TOOLCHAINS: &[&str] = &["Programs", "Volta", "fnm", "uv\\python"];
/// Relative to `%APPDATA%`: nvm-windows and npm's global commands.
const ROAMING_TOOLCHAINS: &[&str] = &["nvm", "npm"];

/// How confined the job is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confinement {
    /// No AppContainer (`BC_NO_SANDBOX=1`).
    Off,
    /// The owner's own devices: internet and local network.
    Owner,
    /// A guest: internet only.
    Guest,
}

impl Confinement {
    pub fn for_job(guest: bool) -> Self {
        if std::env::var_os("BC_NO_SANDBOX").is_some() {
            Confinement::Off
        } else if guest {
            Confinement::Guest
        } else {
            Confinement::Owner
        }
    }
}

/// A started job: its main process, its output, and the group that holds it.
pub struct Spawned {
    pub pid: u32,
    pub process: OwnedHandle,
    pub stdout: File,
    pub stderr: File,
    pub group: Group,
}

/// The job's Job Object and AppContainer. Dropping it kills every process of the job;
/// the AppContainer goes once no [`JobHandle`] needs it either.
pub struct Group {
    shared: Arc<Shared>,
}

struct Shared {
    job: OwnedHandle,
    container: Option<Container>,
    confinement: Confinement,
    ended: AtomicBool,
}

impl Group {
    pub fn kill(&self) {
        self.shared.ended.store(true, Ordering::Relaxed);
        // SAFETY: a valid job handle; terminating an empty job is harmless.
        unsafe { TerminateJobObject(self.shared.job.as_raw_handle(), 1) };
    }

    /// CPU time in user and kernel mode, in milliseconds, and peak committed memory in
    /// bytes, over every process the job ran.
    pub fn usage(&self) -> Option<(u64, u64, u64)> {
        // SAFETY: plain-data structs that the query fills in, with their exact sizes.
        unsafe {
            let mut times: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = zeroed();
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
            let job = self.shared.job.as_raw_handle();
            let ok = QueryInformationJobObject(
                job,
                JobObjectBasicAccountingInformation,
                &mut times as *mut _ as _,
                size_of_val(&times) as u32,
                null_mut(),
            ) != 0
                && QueryInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &mut limits as *mut _ as _,
                    size_of_val(&limits) as u32,
                    null_mut(),
                ) != 0;
            // Job times count 100-nanosecond ticks.
            let ms = |ticks: i64| ticks.max(0) as u64 / 10_000;
            ok.then(|| (ms(times.TotalUserTime), ms(times.TotalKernelTime), limits.PeakJobMemoryUsed as u64))
        }
    }

    /// A handle for reaching into the running job: its ports, and relays to them.
    pub fn handle(&self) -> JobHandle {
        JobHandle(self.shared.clone())
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Reaches into a running job from outside: which ports it listens on, and connections to
/// them through a relay inside its AppContainer.
#[derive(Clone)]
pub struct JobHandle(Arc<Shared>);

/// A relay process connected to one of the job's ports: write to `input` to send to the
/// server, read `output` for its replies. Closing `input` closes the connection's sending
/// side.
pub struct Relayed {
    pub input: File,
    pub output: File,
    pub process: OwnedHandle,
}

impl JobHandle {
    /// TCP ports any process of the job listens on.
    pub fn listening_ports(&self) -> Vec<u16> {
        let pids = self.pids();
        let mut ports: Vec<u16> = listeners().into_iter().filter(|(pid, _)| pids.contains(pid)).map(|(_, port)| port).collect();
        ports.sort_unstable();
        ports.dedup();
        ports
    }

    fn pids(&self) -> Vec<u32> {
        // Room for 4096 processes after the two counts.
        let mut buffer = vec![0usize; 4097];
        // SAFETY: the buffer is pointer-aligned and as large as the size passed.
        let ok = unsafe {
            QueryInformationJobObject(
                self.0.job.as_raw_handle(),
                JobObjectBasicProcessIdList,
                buffer.as_mut_ptr() as _,
                (buffer.len() * size_of::<usize>()) as u32,
                null_mut(),
            )
        };
        if ok == 0 {
            return Vec::new();
        }
        // SAFETY: the query filled in the list header and that many ids after it.
        unsafe {
            let list = buffer.as_ptr() as *const JOBOBJECT_BASIC_PROCESS_ID_LIST;
            let count = (*list).NumberOfProcessIdsInList as usize;
            std::slice::from_raw_parts((*list).ProcessIdList.as_ptr(), count).iter().map(|&p| p as u32).collect()
        }
    }

    /// Starts a relay to `port` inside the job's AppContainer. It answers one byte once it
    /// has connected, which this waits for, so a refused connection is an error here.
    pub fn relay(&self, port: u16) -> io::Result<Relayed> {
        if self.0.ended.load(Ordering::Relaxed) {
            return Err(io::Error::other("the job has ended"));
        }
        let program = relay_program()?;
        let (input_read, input_write) = pipe()?;
        let (output_read, output_write) = pipe()?;
        let errors = File::open("NUL")?;
        let mut line = Vec::new();
        quote(program.as_os_str(), &mut line);
        line.extend(format!(" {RELAY_ARG} {port}").encode_utf16());
        line.push(0);
        let cwd = std::env::temp_dir();
        let (_, process) = launch(
            &self.0,
            &wide(program.as_os_str()),
            line,
            &cwd,
            &environment_block(&[]),
            [input_read.as_raw_handle(), output_write.as_raw_handle(), errors.as_raw_handle()],
        )?;
        // Only the relay holds these ends now, so its exit closes the pipes.
        drop((input_read, output_write, errors));
        let mut output = File::from(output_read);
        let mut ready = [0u8];
        io::Read::read_exact(&mut output, &mut ready)
            .map_err(|_| io::Error::new(io::ErrorKind::ConnectionRefused, format!("nothing in the job accepts connections on port {port}")))?;
        Ok(Relayed { input: File::from(input_write), output, process })
    }
}

/// The program started as a relay: this program unless [`set_relay_program`] named
/// another. It must handle [`RELAY_ARG`] by calling [`relay_main`].
fn relay_program() -> io::Result<PathBuf> {
    let program = match RELAY_PROGRAM.lock().unwrap().clone() {
        Some(program) => program,
        None => std::env::current_exe()?,
    };
    // The AppContainer reads and runs only what is open to it.
    if let Some(capability) = toolchain_capability() {
        ensure_access(&program, capability, FILE_GENERIC_READ | FILE_GENERIC_EXECUTE)?;
    }
    Ok(program)
}

static RELAY_PROGRAM: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Uses `program` as the relay instead of this program, for tests whose own program
/// cannot be one.
pub fn set_relay_program(program: PathBuf) {
    *RELAY_PROGRAM.lock().unwrap() = Some(program);
}

/// The relay's side, run inside the job's AppContainer: connects to `port` on loopback,
/// says so with one byte, then copies stdin to the connection and the connection to
/// stdout until both sides are done.
pub fn relay_main(port: u16) -> i32 {
    use std::io::{Read, Write};
    let Ok(tcp) = std::net::TcpStream::connect(("127.0.0.1", port)).or_else(|_| std::net::TcpStream::connect(("::1", port))) else {
        return 1;
    };
    let _ = tcp.set_nodelay(true);
    let mut stdout = io::stdout().lock();
    if stdout.write_all(&[1]).and_then(|_| stdout.flush()).is_err() {
        return 1;
    }
    let Ok(mut sending) = tcp.try_clone() else { return 1 };
    let upstream = std::thread::spawn(move || {
        let _ = io::copy(&mut io::stdin().lock(), &mut sending);
        let _ = sending.shutdown(std::net::Shutdown::Write);
    });
    let mut buffer = [0u8; 16 * 1024];
    let mut tcp = tcp;
    loop {
        match tcp.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if stdout.write_all(&buffer[..n]).and_then(|_| stdout.flush()).is_err() {
                    break;
                }
            }
        }
    }
    drop(stdout);
    let _ = upstream.join();
    0
}

/// Every listening TCP socket on this PC, IPv4 and IPv6, as (process id, port).
fn listeners() -> Vec<(u32, u16)> {
    let mut found = Vec::new();
    for family in [AF_INET, AF_INET6] {
        let mut size = 0u32;
        let mut buffer: Vec<u32> = Vec::new();
        // The table can grow between asking its size and reading it.
        for _ in 0..4 {
            // SAFETY: the buffer holds `size` bytes, as passed.
            let err = unsafe {
                GetExtendedTcpTable(buffer.as_mut_ptr() as _, &mut size, 0, family as u32, TCP_TABLE_OWNER_PID_LISTENER, 0)
            };
            if err == 0 {
                break;
            }
            buffer = vec![0u32; (size as usize).div_ceil(4)];
            size = (buffer.len() * 4) as u32;
        }
        if buffer.is_empty() {
            continue;
        }
        let port = |raw: u32| u16::from_be(raw as u16);
        // SAFETY: a table the call filled in, with as many rows as its count says.
        unsafe {
            if family == AF_INET {
                let table = buffer.as_ptr() as *const MIB_TCPTABLE_OWNER_PID;
                let rows = std::slice::from_raw_parts((*table).table.as_ptr(), (*table).dwNumEntries as usize);
                found.extend(rows.iter().map(|r| (r.dwOwningPid, port(r.dwLocalPort))));
            } else {
                let table = buffer.as_ptr() as *const MIB_TCP6TABLE_OWNER_PID;
                let rows = std::slice::from_raw_parts((*table).table.as_ptr(), (*table).dwNumEntries as usize);
                found.extend(rows.iter().map(|r| (r.dwOwningPid, port(r.dwLocalPort))));
            }
        }
    }
    found
}

/// Waits for the process to end and returns its exit code.
pub fn wait(process: &OwnedHandle) -> io::Result<u32> {
    // SAFETY: a valid process handle we own.
    unsafe {
        WaitForSingleObject(process.as_raw_handle(), INFINITE);
        let mut code = 0;
        check(GetExitCodeProcess(process.as_raw_handle(), &mut code))?;
        Ok(code)
    }
}

/// Starts `program` with `args` in `cwd`, confined to `workspace`. `env` adds to and
/// overrides the agent's own environment.
pub fn spawn(
    program: &Path,
    args: &[String],
    cwd: &Path,
    env: &[(OsString, OsString)],
    workspace: &Path,
    confinement: Confinement,
) -> io::Result<Spawned> {
    let job = new_job()?;
    let container = match confinement {
        Confinement::Off => None,
        _ => {
            let container = Container::create()?;
            grant(workspace, container.sid, FILE_ALL_ACCESS)?;
            Some(container)
        }
    };
    let shared = Arc::new(Shared { job, container, confinement, ended: AtomicBool::new(false) });

    let path = env.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case("PATH")).map(|(_, v)| v.clone()).or_else(|| std::env::var_os("PATH"));
    let mut paths_resolve = true;
    if let Some(toolchains) = toolchain_capability() {
        grant_toolchains(toolchains, workspace.parent().unwrap_or(workspace));
        paths_resolve = open_folder_attributes(toolchains, &resolved_folders(workspace, path.as_deref())).is_empty();
    }

    // Output pipes, and NUL for input.
    let (stdout_read, stdout_write) = pipe()?;
    let (stderr_read, stderr_write) = pipe()?;
    let stdin = File::open("NUL")?;

    let (application, command_line) = command_line(program, args, cwd, path.as_deref());
    let mut env = env.to_vec();
    if confinement != Confinement::Off {
        env.push(("NODE_OPTIONS".into(), node_options(&env, paths_resolve)));
        // Windows does not let an AppContainer create junctions, which pnpm links every
        // package with; its hoisted layout copies packages into place like npm instead.
        env.push(("npm_config_node_linker".into(), "hoisted".into()));
    }
    let (pid, process) = launch(
        &shared,
        &application,
        command_line,
        cwd,
        &environment_block(&env),
        [stdin.as_raw_handle(), stdout_write.as_raw_handle(), stderr_write.as_raw_handle()],
    )?;
    Ok(Spawned {
        pid,
        process,
        stdout: File::from(stdout_read),
        stderr: File::from(stderr_read),
        group: Group { shared },
    })
}

/// Starts a process inside the job: in its AppContainer with its capabilities, and in its
/// Job Object before it runs a single instruction, so nothing it starts escapes. `stdio`
/// is stdin, stdout and stderr; only these handles are inherited.
fn launch(
    shared: &Shared,
    application: &[u16],
    mut command_line: Vec<u16>,
    cwd: &Path,
    environment: &[u16],
    stdio: [HANDLE; 3],
) -> io::Result<(u32, OwnedHandle)> {
    // Capabilities, as SIDs in buffers that live until the process has started.
    let mut sid_buffers: Vec<[u8; SECURITY_MAX_SID_SIZE as usize]> = Vec::new();
    sid_buffers.push(well_known_sid(WinCapabilityInternetClientSid)?);
    if shared.confinement == Confinement::Owner {
        sid_buffers.push(well_known_sid(WinCapabilityPrivateNetworkClientServerSid)?);
    }
    let mut capabilities: Vec<SID_AND_ATTRIBUTES> = sid_buffers
        .iter_mut()
        .map(|b| SID_AND_ATTRIBUTES { Sid: b.as_mut_ptr() as PSID, Attributes: SE_GROUP_ENABLED })
        .collect();
    if let Some(toolchains) = toolchain_capability() {
        capabilities.push(SID_AND_ATTRIBUTES { Sid: toolchains, Attributes: SE_GROUP_ENABLED });
    }
    let container = shared.container.as_ref();
    let security = SECURITY_CAPABILITIES {
        AppContainerSid: container.map_or(null_mut(), |c| c.sid),
        Capabilities: capabilities.as_mut_ptr(),
        CapabilityCount: capabilities.len() as u32,
        Reserved: 0,
    };

    for h in stdio {
        // SAFETY: valid handles the caller owns.
        check(unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) })?;
    }
    let attributes = AttributeList::new(if container.is_some() { 2 } else { 1 })?;
    attributes.set(PROC_THREAD_ATTRIBUTE_HANDLE_LIST, stdio.as_ptr() as _, size_of_val(&stdio))?;
    if container.is_some() {
        attributes.set(PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, &security as *const _ as _, size_of_val(&security))?;
    }

    // SAFETY: STARTUPINFOEXW is plain data; every pointer in it outlives CreateProcessW.
    let mut startup: STARTUPINFOEXW = unsafe { zeroed() };
    startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = stdio[0];
    startup.StartupInfo.hStdOutput = stdio[1];
    startup.StartupInfo.hStdError = stdio[2];
    startup.lpAttributeList = attributes.ptr();

    let cwd = wide(cwd.as_os_str());
    // SAFETY: zeroed PROCESS_INFORMATION is valid output space.
    let mut info: PROCESS_INFORMATION = unsafe { zeroed() };
    // SAFETY: all strings are NUL-terminated wide strings, `command_line` is mutable as
    // CreateProcessW requires, and the environment block is double-NUL-terminated.
    check(unsafe {
        CreateProcessW(
            application.as_ptr(),
            command_line.as_mut_ptr(),
            null(),
            null(),
            1,
            CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
            environment.as_ptr() as _,
            cwd.as_ptr(),
            &startup.StartupInfo,
            &mut info,
        )
    })?;
    // SAFETY: CreateProcessW returned fresh handles that we now own.
    let process = unsafe { OwnedHandle::from_raw_handle(info.hProcess) };
    let thread = unsafe { OwnedHandle::from_raw_handle(info.hThread) };
    // SAFETY: valid handles.
    if let Err(e) = check(unsafe { AssignProcessToJobObject(shared.job.as_raw_handle(), process.as_raw_handle()) }) {
        unsafe { TerminateProcess(process.as_raw_handle(), 1) };
        return Err(e);
    }
    // SAFETY: the suspended main thread of the process we just made.
    unsafe { ResumeThread(thread.as_raw_handle()) };
    Ok((info.dwProcessId, process))
}

/// A Job Object that kills its processes when its last handle closes.
fn new_job() -> io::Result<OwnedHandle> {
    // SAFETY: no security attributes and no name; the result is checked.
    let raw = unsafe { CreateJobObjectW(null(), null()) };
    if raw.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a fresh handle we own.
    let job = unsafe { OwnedHandle::from_raw_handle(raw) };
    // SAFETY: plain data, passed with its exact size.
    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    check(unsafe {
        SetInformationJobObject(
            job.as_raw_handle(),
            JobObjectExtendedLimitInformation,
            &limits as *const _ as _,
            size_of_val(&limits) as u32,
        )
    })?;
    Ok(job)
}

/// One job's AppContainer profile, deleted when dropped.
struct Container {
    name: Vec<u16>,
    sid: PSID,
}

// SAFETY: the SID is owned memory that nothing else frees or mutates.
unsafe impl Send for Container {}
unsafe impl Sync for Container {}

impl Container {
    fn create() -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!("BorrowedCompute.Job.{}.{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
        let name = wide(OsStr::new(&name));
        let mut sid = null_mut();
        // A profile left behind by a crashed agent with the same name is replaced.
        for _ in 0..2 {
            // SAFETY: NUL-terminated names, no capabilities; the SID comes back in `sid`.
            let hr = unsafe { CreateAppContainerProfile(name.as_ptr(), name.as_ptr(), name.as_ptr(), null(), 0, &mut sid) };
            if hr >= 0 {
                return Ok(Container { name, sid });
            }
            // HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)
            if hr as u32 != 0x8007_00B7 {
                return Err(io::Error::other(format!("creating the job's AppContainer failed (0x{:08x})", hr as u32)));
            }
            // SAFETY: a NUL-terminated name.
            unsafe { DeleteAppContainerProfile(name.as_ptr()) };
        }
        Err(io::Error::other("creating the job's AppContainer failed: the name is taken"))
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        // SAFETY: the SID came from CreateAppContainerProfile, which documents FreeSid.
        unsafe {
            FreeSid(self.sid);
            DeleteAppContainerProfile(self.name.as_ptr());
        }
    }
}

/// The toolchain capability's SID, derived once and kept for the agent's lifetime.
fn toolchain_capability() -> Option<PSID> {
    static SID: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let sid = SID.get_or_init(|| {
        let name = wide(OsStr::new(TOOLCHAIN_CAPABILITY));
        let (mut groups, mut group_count, mut sids, mut count) = (null_mut(), 0, null_mut(), 0);
        // SAFETY: output pointers to locals; the arrays and SIDs are LocalAlloc'd and kept.
        let ok = unsafe { DeriveCapabilitySidsFromName(name.as_ptr(), &mut groups, &mut group_count, &mut sids, &mut count) };
        if ok == 0 || count == 0 {
            eprintln!("could not derive the toolchain capability: {}", io::Error::last_os_error());
            return None;
        }
        // SAFETY: `sids` holds `count` SIDs.
        Some(unsafe { *sids } as usize)
    });
    sid.map(|s| s as PSID)
}

/// The toolchain folders that exist on this PC: the fixed list, and every PATH folder
/// inside the user's profile.
fn toolchain_dirs() -> Vec<PathBuf> {
    let under = |var: &str, dirs: &[&str]| -> Vec<PathBuf> {
        std::env::var_os(var).map_or_else(Vec::new, |base| dirs.iter().map(|d| Path::new(&base).join(d)).collect())
    };
    let mut all = under("USERPROFILE", PROFILE_TOOLCHAINS);
    all.extend(under("LOCALAPPDATA", LOCAL_TOOLCHAINS));
    all.extend(under("APPDATA", ROAMING_TOOLCHAINS));
    if let (Some(profile), Some(path)) = (std::env::var_os("USERPROFILE"), std::env::var_os("PATH")) {
        all.extend(std::env::split_paths(&path).filter(|d| d.starts_with(&profile)));
    }
    all.retain(|d| d.is_dir());
    all.sort();
    all.dedup();
    all
}

/// Grants the toolchain capability read and execute on each toolchain folder, once:
/// granting a large folder such as `.rustup` walks every file in it, so the folders done
/// are listed in `.toolchains` under `work_root`, which jobs cannot read.
fn grant_toolchains(capability: PSID, work_root: &Path) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _held = LOCK.lock();
    let marker = work_root.join(".toolchains");
    let done = std::fs::read_to_string(&marker).unwrap_or_default();
    let mut list = done.clone();
    for dir in toolchain_dirs() {
        let line = dir.to_string_lossy();
        if done.lines().any(|l| l == line) {
            continue;
        }
        match grant(&dir, capability, FILE_GENERIC_READ | FILE_GENERIC_EXECUTE) {
            Ok(()) => {
                list.push_str(&line);
                list.push('\n');
            }
            Err(e) => eprintln!("could not open {} to jobs: {e}", dir.display()),
        }
    }
    if list != done {
        let _ = std::fs::write(&marker, list);
    }
}

/// Adds an inheritable ACE granting `access` to `sid` on `path` and everything in it.
fn grant(path: &Path, sid: PSID, access: u32) -> io::Result<()> {
    let name = wide(path.as_os_str());
    let mut old: *mut ACL = null_mut();
    let mut descriptor = null_mut();
    // SAFETY: NUL-terminated path, output pointers to locals; the descriptor is freed below
    // and `old` points into it.
    let err = unsafe {
        GetNamedSecurityInfoW(name.as_ptr(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, null_mut(), null_mut(), &mut old, null_mut(), &mut descriptor)
    };
    if err != 0 {
        return Err(io::Error::from_raw_os_error(err as i32));
    }
    let entry = EXPLICIT_ACCESS_W {
        grfAccessPermissions: access,
        grfAccessMode: GRANT_ACCESS,
        grfInheritance: SUB_CONTAINERS_AND_OBJECTS_INHERIT,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: null_mut(),
            MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_WELL_KNOWN_GROUP,
            ptstrName: sid as _,
        },
    };
    let mut new: *mut ACL = null_mut();
    // SAFETY: one entry, the old ACL from above; the new ACL is LocalAlloc'd and freed below.
    let err = unsafe { SetEntriesInAclW(1, &entry, old, &mut new) };
    unsafe { LocalFree(descriptor) };
    if err != 0 {
        return Err(io::Error::from_raw_os_error(err as i32));
    }
    // SAFETY: NUL-terminated path and the ACL just built.
    let err = unsafe {
        SetNamedSecurityInfoW(name.as_ptr(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, null_mut(), null_mut(), new, null())
    };
    unsafe { LocalFree(new as _) };
    if err != 0 {
        return Err(io::Error::from_raw_os_error(err as i32));
    }
    Ok(())
}

/// The folders above which Node resolves paths in a job: the workspace, the toolchains,
/// and every PATH folder jobs can list.
fn resolved_folders(workspace: &Path, path: Option<&OsStr>) -> Vec<PathBuf> {
    let mut folders = vec![workspace.to_path_buf()];
    folders.extend(toolchain_dirs());
    if let (Some(path), Some(capability)) = (path, toolchain_capability()) {
        folders.extend(std::env::split_paths(path).filter(|d| rights(d, capability) & FILE_LIST_DIRECTORY != 0));
    }
    folders
}

/// Lets jobs read the attributes of every folder above `folders`, which Node's module
/// loader reads on its way to each file it loads (on Windows from `C:\` down). Only the
/// attributes: not a listing, not the files inside. Returns the folders this process may
/// not change, such as `C:\` and `C:\Users`, which need [`setup`] as an administrator.
fn open_folder_attributes(capability: PSID, folders: &[PathBuf]) -> Vec<PathBuf> {
    // Folders already open, so each job checks only what is new to it.
    static OPEN: std::sync::Mutex<std::collections::BTreeSet<PathBuf>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
    let mut above: Vec<&Path> = folders.iter().flat_map(|f| f.ancestors().skip(1)).collect();
    above.retain(|d| d.exists());
    above.sort();
    above.dedup();
    let mut open = OPEN.lock().unwrap();
    let mut missing = Vec::new();
    for dir in above {
        if open.contains(dir) {
            continue;
        }
        match ensure_access(dir, capability, FILE_READ_ATTRIBUTES) {
            Ok(()) => {
                open.insert(dir.to_path_buf());
            }
            Err(_) => missing.push(dir.to_path_buf()),
        }
    }
    missing
}

/// Makes sure `capability` (or every application package) has `access` on `path` itself,
/// adding a non-inherited ACE if it lacks it. SetFileSecurityW changes this one object:
/// SetNamedSecurityInfoW would walk everything below, the whole drive for `C:\`.
fn ensure_access(path: &Path, capability: PSID, access: u32) -> io::Result<()> {
    if rights(path, capability) & access == access {
        return Ok(());
    }
    let name = wide(path.as_os_str());
    let mut old: *mut ACL = null_mut();
    let mut descriptor = null_mut();
    // SAFETY: NUL-terminated path, output pointers to locals; the descriptor is freed below
    // and `old` points into it.
    let err = unsafe {
        GetNamedSecurityInfoW(name.as_ptr(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, null_mut(), null_mut(), &mut old, null_mut(), &mut descriptor)
    };
    if err != 0 {
        return Err(io::Error::from_raw_os_error(err as i32));
    }
    let (mut control, mut revision) = (0u16, 0u32);
    // SAFETY: the descriptor from above.
    unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) };
    let entry = EXPLICIT_ACCESS_W {
        grfAccessPermissions: access,
        grfAccessMode: GRANT_ACCESS,
        grfInheritance: NO_INHERITANCE,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: null_mut(),
            MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_WELL_KNOWN_GROUP,
            ptstrName: capability as _,
        },
    };
    let mut new: *mut ACL = null_mut();
    // SAFETY: one entry and the old ACL; the new ACL is LocalAlloc'd and freed below.
    let err = unsafe { SetEntriesInAclW(1, &entry, old, &mut new) };
    unsafe { LocalFree(descriptor) };
    if err != 0 {
        return Err(io::Error::from_raw_os_error(err as i32));
    }
    // SAFETY: an absolute descriptor on the stack pointing at the new ACL, both alive
    // until SetFileSecurityW returns.
    let result = unsafe {
        let mut sd: SECURITY_DESCRIPTOR = zeroed();
        let sd_ptr = &mut sd as *mut _ as _;
        let keep = SE_DACL_AUTO_INHERITED | SE_DACL_PROTECTED;
        let ok = InitializeSecurityDescriptor(sd_ptr, SECURITY_DESCRIPTOR_REVISION) != 0
            && SetSecurityDescriptorDacl(sd_ptr, 1, new, 0) != 0
            && SetSecurityDescriptorControl(sd_ptr, keep, control & keep) != 0
            && SetFileSecurityW(name.as_ptr(), DACL_SECURITY_INFORMATION, sd_ptr) != 0;
        let result = if ok { Ok(()) } else { Err(io::Error::last_os_error()) };
        LocalFree(new as _);
        result
    };
    result
}

/// What `capability` and "ALL APPLICATION PACKAGES" are allowed on `path` itself, from its
/// allow entries. AppContainer jobs get nothing from entries for Everyone or Users, so
/// only these two count.
fn rights(path: &Path, capability: PSID) -> u32 {
    let Ok(mut any_package) = well_known_sid(WinBuiltinAnyPackageSid) else { return 0 };
    let name = wide(path.as_os_str());
    let mut acl: *mut ACL = null_mut();
    let mut descriptor = null_mut();
    // SAFETY: as in `grant`; `acl` points into the descriptor, freed at the end.
    let err = unsafe {
        GetNamedSecurityInfoW(name.as_ptr(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, null_mut(), null_mut(), &mut acl, null_mut(), &mut descriptor)
    };
    if err != 0 {
        return 0;
    }
    let mut mask = 0;
    // A null DACL allows everyone everything.
    if acl.is_null() {
        mask = FILE_ALL_ACCESS;
    } else {
        // SAFETY: each index is below the ACL's count; allow ACEs carry a SID at SidStart.
        unsafe {
            for i in 0..(*acl).AceCount as u32 {
                let mut ace = null_mut();
                if GetAce(acl, i, &mut ace) == 0 {
                    continue;
                }
                let ace = &*(ace as *const ACCESS_ALLOWED_ACE);
                if ace.Header.AceType != ACCESS_ALLOWED_ACE_TYPE || ace.Header.AceFlags & INHERIT_ONLY_ACE != 0 {
                    continue;
                }
                let sid = &ace.SidStart as *const u32 as PSID;
                if EqualSid(sid, capability) != 0 || EqualSid(sid, any_package.as_mut_ptr() as PSID) != 0 {
                    mask |= ace.Mask;
                }
            }
        }
    }
    unsafe { LocalFree(descriptor) };
    if mask & GENERIC_ALL != 0 {
        mask |= FILE_ALL_ACCESS;
    }
    if mask & GENERIC_READ != 0 {
        mask |= FILE_GENERIC_READ;
    }
    if mask & GENERIC_EXECUTE != 0 {
        mask |= FILE_GENERIC_EXECUTE;
    }
    mask
}

/// Whether [`setup`] still has something to do: folders only an administrator can open
/// to jobs, or the app's firewall rule.
pub fn setup_needed() -> bool {
    let Some(capability) = toolchain_capability() else { return false };
    !open_folder_attributes(capability, &setup_folders()).is_empty() || !firewall_rule_exists()
}

/// The one-time setup that needs an administrator, run elevated from the installer or when
/// lending is turned on: lets jobs read the attributes of the folders above the user's
/// profile and toolchains (`C:\`, `C:\Users`), so Node resolves paths in jobs the way it
/// does outside, symlinked packages included; and lets the app through the firewall on
/// private networks, so lending and nearby discovery work without a firewall prompt.
pub fn setup() -> io::Result<()> {
    let mut failed = Vec::new();
    if let Some(capability) = toolchain_capability() {
        failed = open_folder_attributes(capability, &setup_folders());
    }
    let program = std::env::current_exe()?;
    let netsh = |args: &[&str]| {
        std::process::Command::new("netsh").args(["advfirewall", "firewall"]).args(args).creation_flags(CREATE_NO_WINDOW).output()
    };
    let name = format!("name={FIREWALL_RULE}");
    // Replaced, so a rule for an older install path does not linger.
    let _ = netsh(&["delete", "rule", &name]);
    let added = netsh(&["add", "rule", &name, "dir=in", "action=allow", &format!("program={}", program.display()), "enable=yes", "profile=private,domain"])?;
    if !added.status.success() {
        return Err(io::Error::other(format!("adding the firewall rule failed: {}", String::from_utf8_lossy(&added.stdout).trim())));
    }
    if !failed.is_empty() {
        return Err(io::Error::other(format!("could not open {} to jobs", failed[0].display())));
    }
    Ok(())
}

/// The folders whose parents [`setup`] opens: the profile, where workspaces and per-user
/// toolchains live, and the toolchains wherever they are.
fn setup_folders() -> Vec<PathBuf> {
    let mut folders = vec![std::env::temp_dir().join("borrowed-compute").join("job")];
    folders.extend(std::env::var_os("USERPROFILE").map(|p| PathBuf::from(p).join("x")));
    folders.extend(toolchain_dirs());
    folders
}

fn firewall_rule_exists() -> bool {
    std::process::Command::new("netsh")
        .args(["advfirewall", "firewall", "show", "rule", &format!("name={FIREWALL_RULE}")])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Runs [`setup`] elevated: asks Windows to start this program as an administrator with
/// [`SETUP_ARG`], which shows the UAC prompt, and waits for it. Declining is an error.
pub fn request_setup() -> io::Result<()> {
    use windows_sys::Win32::UI::Shell::{ShellExecuteExW, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW};
    let program = wide(std::env::current_exe()?.as_os_str());
    let verb = wide(OsStr::new("runas"));
    let args = wide(OsStr::new(SETUP_ARG));
    // SAFETY: plain data; every string outlives the call.
    let mut info: SHELLEXECUTEINFOW = unsafe { zeroed() };
    info.cbSize = size_of::<SHELLEXECUTEINFOW>() as u32;
    info.fMask = SEE_MASK_NOCLOSEPROCESS;
    info.lpVerb = verb.as_ptr();
    info.lpFile = program.as_ptr();
    info.lpParameters = args.as_ptr();
    check(unsafe { ShellExecuteExW(&mut info) })?;
    if info.hProcess.is_null() {
        return Err(io::Error::other("setup did not start"));
    }
    // SAFETY: the process handle ShellExecuteExW returned, now ours.
    let process = unsafe { OwnedHandle::from_raw_handle(info.hProcess) };
    match wait(&process)? {
        0 => Ok(()),
        code => Err(io::Error::other(format!("setup ended with code {code}"))),
    }
}

/// Handles the agent program's helper modes: a relay ([`RELAY_ARG`] port) inside a job's
/// AppContainer, or the elevated setup. `None` for a normal start.
pub fn helper_main() -> Option<i32> {
    let args: Vec<String> = std::env::args().skip(1).take(2).collect();
    match args.first().map(String::as_str) {
        Some(RELAY_ARG) => Some(args.get(1).and_then(|p| p.parse().ok()).map_or(2, relay_main)),
        Some(SETUP_ARG) => Some(match setup() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("{e}");
                1
            }
        }),
        Some(SETUP_CHECK_ARG) => Some(setup_needed() as i32),
        _ => None,
    }
}

fn well_known_sid(kind: WELL_KNOWN_SID_TYPE) -> io::Result<[u8; SECURITY_MAX_SID_SIZE as usize]> {
    let mut buffer = [0u8; SECURITY_MAX_SID_SIZE as usize];
    let mut size = buffer.len() as u32;
    // SAFETY: the buffer is SECURITY_MAX_SID_SIZE bytes, as `size` says.
    check(unsafe { CreateWellKnownSid(kind, null_mut(), buffer.as_mut_ptr() as PSID, &mut size) })?;
    Ok(buffer)
}

fn pipe() -> io::Result<(OwnedHandle, OwnedHandle)> {
    let (mut read, mut write) = (null_mut(), null_mut());
    // SAFETY: output pointers to locals; both handles are ours on success.
    check(unsafe { CreatePipe(&mut read, &mut write, null(), 0) })?;
    Ok(unsafe { (OwnedHandle::from_raw_handle(read), OwnedHandle::from_raw_handle(write)) })
}

/// A process-thread attribute list, deleted when dropped.
struct AttributeList(Vec<usize>);

impl AttributeList {
    fn new(count: u32) -> io::Result<Self> {
        let mut size = 0;
        // SAFETY: a size query; it fails by design and fills in `size`.
        unsafe { InitializeProcThreadAttributeList(null_mut(), count, 0, &mut size) };
        let mut buffer = vec![0usize; size.div_ceil(size_of::<usize>())];
        // SAFETY: a buffer of at least `size` bytes, aligned for pointers.
        check(unsafe { InitializeProcThreadAttributeList(buffer.as_mut_ptr() as _, count, 0, &mut size) })?;
        Ok(AttributeList(buffer))
    }

    fn ptr(&self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.0.as_ptr() as _
    }

    /// The value must outlive the process creation that uses this list.
    fn set(&self, attribute: u32, value: *const core::ffi::c_void, size: usize) -> io::Result<()> {
        // SAFETY: an initialized list; the caller keeps `value` alive.
        check(unsafe { UpdateProcThreadAttribute(self.ptr(), 0, attribute as usize, value, size, null_mut(), null()) })
    }
}

impl Drop for AttributeList {
    fn drop(&mut self) {
        // SAFETY: an initialized list.
        unsafe { DeleteProcThreadAttributeList(self.ptr()) };
    }
}

/// The application and command line for CreateProcessW. Batch files (`npm.cmd`) only run
/// through cmd.exe. Its metacharacters in arguments are not escaped: the job runs code of
/// its sender's choosing anyway, confined the same either way.
///
/// Inside the AppContainer cmd.exe runs a batch file only by a path relative to the job's
/// folder: given a full path it says "Access is denied", and it finds nothing on PATH. So a
/// batch file in the job's folder is named relative to it; batch files that only start a
/// Node script (`npm.cmd`, `yarn.cmd`, `pnpm.cmd`) run that script with Node directly
/// ([`node_shim`]); any other batch file outside the job's folder is named by its name
/// alone when on PATH, which works only outside the AppContainer.
fn command_line(program: &Path, args: &[String], cwd: &Path, path: Option<&OsStr>) -> (Vec<u16>, Vec<u16>) {
    let is_batch = program
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"));
    if let (true, Err(_), Some((node, script))) = (is_batch, program.strip_prefix(cwd), node_shim(program, path)) {
        let mut line = Vec::new();
        quote(node.as_os_str(), &mut line);
        line.push(' ' as u16);
        quote(script.as_os_str(), &mut line);
        for arg in args {
            line.push(' ' as u16);
            quote(OsStr::new(arg), &mut line);
        }
        line.push(0);
        return (wide(node.as_os_str()), line);
    }
    let mut line = Vec::new();
    if is_batch {
        let cmd = std::env::var_os("ComSpec").unwrap_or_else(|| "C:\\Windows\\System32\\cmd.exe".into());
        line.extend("cmd.exe /d /s /c \"".encode_utf16());
        let on_path = |dir: &Path| path.is_some_and(|p| std::env::split_paths(p).any(|d| d == dir));
        let named = match (program.strip_prefix(cwd), program.parent(), program.file_name()) {
            (Ok(relative), _, _) => Path::new(".").join(relative),
            (_, Some(dir), Some(name)) if on_path(dir) => PathBuf::from(name),
            _ => program.to_path_buf(),
        };
        quote(named.as_os_str(), &mut line);
        for arg in args {
            line.push(' ' as u16);
            quote(OsStr::new(arg), &mut line);
        }
        line.push('"' as u16);
        line.push(0);
        return (wide(&cmd), line);
    }
    quote(program.as_os_str(), &mut line);
    for arg in args {
        line.push(' ' as u16);
        quote(OsStr::new(arg), &mut line);
    }
    line.push(0);
    (wide(program.as_os_str()), line)
}

/// The job's `NODE_OPTIONS`, plus the switches Node needs inside the AppContainer when
/// `paths_resolve` is false. Node's module loader resolves every file it loads through
/// each folder above it, starting with `C:\`, whose attributes an AppContainer may not
/// read until [`setup`] has run ("EPERM: operation not permitted, lstat 'C:\'"), so no
/// script would load. Keeping paths as given skips that walk; the cost is that a
/// symlinked package (pnpm's layout) loads by its link path and misses its dependencies.
fn node_options(env: &[(OsString, OsString)], paths_resolve: bool) -> OsString {
    let own = env
        .iter()
        .rev()
        .find(|(k, _)| k.eq_ignore_ascii_case("NODE_OPTIONS"))
        .map(|(_, v)| v.clone())
        .or_else(|| std::env::var_os("NODE_OPTIONS"))
        .unwrap_or_default();
    if paths_resolve {
        return own;
    }
    let mut options = OsString::from("--preserve-symlinks --preserve-symlinks-main");
    if !own.is_empty() {
        options.push(" ");
        options.push(own);
    }
    options
}

/// For a batch file that only starts a Node script: `node.exe` and the script. That is
/// Node's own `npm.cmd` and `npx.cmd`, and every shim npm writes for a global package
/// (`yarn.cmd`, `pnpm.cmd` in `%APPDATA%\npm`) or Corepack's in Node's folder. Node is
/// the one beside the batch file, else the first on `path`.
fn node_shim(program: &Path, path: Option<&OsStr>) -> Option<(PathBuf, PathBuf)> {
    let dir = program.parent()?;
    let name = program.file_stem()?.to_str()?.to_ascii_lowercase();
    let script = if name == "npm" || name == "npx" {
        dir.join("node_modules").join("npm").join("bin").join(format!("{name}-cli.js"))
    } else {
        let text = std::fs::read(program).ok().filter(|t| t.len() < 64 << 10)?;
        dir.join(shim_script(&String::from_utf8_lossy(&text))?)
    };
    let node = Some(dir.join("node.exe"))
        .filter(|n| n.is_file())
        .or_else(|| std::env::split_paths(path?).map(|d| d.join("node.exe")).find(|n| n.is_file()))?;
    script.is_file().then_some((node, script))
}

/// The script a cmd-shim batch file runs, relative to its folder. Its last line ends in
/// `"%_prog%"  "%dp0%\node_modules\yarn\bin\yarn.js" %*`.
fn shim_script(text: &str) -> Option<String> {
    text.lines().rev().find_map(|line| {
        let start = line.find("\"%dp0%\\").or_else(|| line.find("\"%~dp0\\"))?;
        let rest = &line[start + 1..];
        let end = rest.find('"')?;
        let (relative, after) = (rest[..end].split_once('\\')?.1, &rest[end + 1..]);
        let script = [".js", ".cjs", ".mjs"].iter().any(|e| relative.to_ascii_lowercase().ends_with(e));
        (script && after.trim_start().starts_with("%*")).then(|| relative.to_string())
    })
}

/// Quotes one argument the way the C runtime splits command lines.
fn quote(arg: &OsStr, out: &mut Vec<u16>) {
    let arg: Vec<u16> = arg.encode_wide().collect();
    let plain = !arg.is_empty() && !arg.iter().any(|&c| c == ' ' as u16 || c == '\t' as u16 || c == '"' as u16);
    if plain {
        out.extend(arg);
        return;
    }
    out.push('"' as u16);
    let mut backslashes = 0;
    for c in arg {
        if c == '\\' as u16 {
            backslashes += 1;
        } else {
            if c == '"' as u16 {
                // Double the backslashes before a quote, then escape the quote.
                out.extend(std::iter::repeat_n('\\' as u16, backslashes + 1));
            }
            backslashes = 0;
        }
        out.push(c);
    }
    // Backslashes before the closing quote are doubled too.
    out.extend(std::iter::repeat_n('\\' as u16, backslashes));
    out.push('"' as u16);
}

/// The agent's environment with `overrides` applied, as a sorted, double-NUL-terminated
/// block. Windows names are case-insensitive.
fn environment_block(overrides: &[(OsString, OsString)]) -> Vec<u16> {
    let mut vars: std::collections::BTreeMap<String, (OsString, OsString)> = std::collections::BTreeMap::new();
    for (k, v) in std::env::vars_os().chain(overrides.iter().cloned()) {
        vars.insert(k.to_string_lossy().to_uppercase(), (k, v));
    }
    let mut block = Vec::new();
    for (k, v) in vars.into_values() {
        block.extend(k.encode_wide());
        block.push('=' as u16);
        block.extend(v.encode_wide());
        block.push(0);
    }
    block.push(0);
    block
}

fn wide(s: &OsStr) -> Vec<u16> {
    s.encode_wide().chain(Some(0)).collect()
}

fn check(ok: BOOL) -> io::Result<()> {
    if ok == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// The script a Node shim runs, for the runner's tests.
#[cfg(test)]
pub(crate) fn shim_script_for_tests(program: &Path) -> Option<PathBuf> {
    node_shim(program, std::env::var_os("PATH").as_deref()).map(|(_, script)| script)
}

/// [`command_line`] as strings, for the runner's tests.
#[cfg(test)]
pub(crate) fn command_line_for_tests(program: &Path, args: &[String], cwd: &Path) -> (String, String) {
    let (application, line) = command_line(program, args, cwd, None);
    let text = |w: &[u16]| String::from_utf16_lossy(w).trim_end_matches('\0').to_string();
    (text(&application), text(&line))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quoted(arg: &str) -> String {
        let mut out = Vec::new();
        quote(OsStr::new(arg), &mut out);
        String::from_utf16(&out).unwrap()
    }

    #[test]
    fn quotes_like_the_c_runtime() {
        assert_eq!(quoted("plain"), "plain");
        assert_eq!(quoted(""), "\"\"");
        assert_eq!(quoted("two words"), "\"two words\"");
        assert_eq!(quoted("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(quoted("C:\\dir with space\\"), "\"C:\\dir with space\\\\\"");
        assert_eq!(quoted("a\\\\\"b"), "\"a\\\\\\\\\\\"b\"");
    }

    #[test]
    fn batch_files_run_through_cmd() {
        let (app, line) = command_line(Path::new("C:\\nodejs\\npm.cmd"), &["run".into(), "build it".into()], Path::new("C:\\work"), None);
        let app = String::from_utf16_lossy(&app);
        assert!(app.to_lowercase().ends_with("cmd.exe\0"), "{app}");
        assert_eq!(String::from_utf16_lossy(&line), "cmd.exe /d /s /c \"C:\\nodejs\\npm.cmd run \"build it\"\"\0");
        let (_, line) = command_line(Path::new("C:\\work\\scripts\\go.cmd"), &[], Path::new("C:\\work"), None);
        assert_eq!(String::from_utf16_lossy(&line), "cmd.exe /d /s /c \".\\scripts\\go.cmd\"\0");
        let path = OsString::from("C:\\Windows;C:\\nodejs");
        let (_, line) = command_line(Path::new("C:\\nodejs\\npm.cmd"), &["test".into()], Path::new("C:\\work"), Some(&path));
        assert_eq!(String::from_utf16_lossy(&line), "cmd.exe /d /s /c \"npm.cmd test\"\0");
    }

    #[test]
    fn node_keeps_the_job_s_own_options() {
        let own = [("node_options".into(), "--max-old-space-size=4096".into())];
        assert_eq!(node_options(&own, false), "--preserve-symlinks --preserve-symlinks-main --max-old-space-size=4096");
        assert_eq!(node_options(&own, true), "--max-old-space-size=4096");
    }

    #[test]
    fn finds_the_script_a_shim_runs() {
        let yarn = "@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n:start\r\nSETLOCAL\r\nCALL :find_dp0\r\n\r\n\
            IF EXIST \"%dp0%\\node.exe\" (\r\n  SET \"_prog=%dp0%\\node.exe\"\r\n) ELSE (\r\n  SET \"_prog=node\"\r\n)\r\n\r\n\
            endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & \"%_prog%\"  \"%dp0%\\node_modules\\yarn\\bin\\yarn.js\" %*\r\n";
        assert_eq!(shim_script(yarn).as_deref(), Some("node_modules\\yarn\\bin\\yarn.js"));
        assert_eq!(shim_script("@echo off\r\n\"%dp0%\\tool.exe\" %*\r\n"), None);
    }

    #[test]
    fn environment_overrides_are_case_insensitive() {
        let block = environment_block(&[("path".into(), "X".into())]);
        let text = String::from_utf16_lossy(&block);
        let paths: Vec<_> = text.split('\0').filter(|v| v.to_uppercase().starts_with("PATH=")).collect();
        assert_eq!(paths, ["path=X"]);
        assert!(text.ends_with("\0\0"));
    }
}
