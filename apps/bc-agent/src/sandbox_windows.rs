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
//! reach.
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
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::core::BOOL;
use windows_sys::Win32::Foundation::{LocalFree, SetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT};
use windows_sys::Win32::Security::Authorization::{
    GetNamedSecurityInfoW, SetEntriesInAclW, SetNamedSecurityInfoW, EXPLICIT_ACCESS_W, GRANT_ACCESS,
    NO_MULTIPLE_TRUSTEE, SE_FILE_OBJECT, TRUSTEE_IS_SID, TRUSTEE_IS_WELL_KNOWN_GROUP, TRUSTEE_W,
};
use windows_sys::Win32::Security::Isolation::{CreateAppContainerProfile, DeleteAppContainerProfile};
use windows_sys::Win32::Security::{
    CreateWellKnownSid, DeriveCapabilitySidsFromName, FreeSid, WinCapabilityInternetClientSid,
    WinCapabilityPrivateNetworkClientServerSid, ACL, DACL_SECURITY_INFORMATION, PSID, SECURITY_CAPABILITIES,
    SECURITY_MAX_SID_SIZE, SID_AND_ATTRIBUTES, SUB_CONTAINERS_AND_OBJECTS_INHERIT,
    WELL_KNOWN_SID_TYPE,
};
use windows_sys::Win32::Storage::FileSystem::{FILE_ALL_ACCESS, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectBasicAccountingInformation,
    JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
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

/// The job's Job Object and AppContainer. Dropping it kills every process of the job,
/// then deletes the AppContainer.
pub struct Group {
    job: OwnedHandle,
    _container: Option<Container>,
}

impl Group {
    pub fn kill(&self) {
        // SAFETY: a valid job handle; terminating an empty job is harmless.
        unsafe { TerminateJobObject(self.job.as_raw_handle(), 1) };
    }

    /// CPU time in user and kernel mode, in milliseconds, and peak committed memory in
    /// bytes, over every process the job ran.
    pub fn usage(&self) -> Option<(u64, u64, u64)> {
        // SAFETY: plain-data structs that the query fills in, with their exact sizes.
        unsafe {
            let mut times: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = zeroed();
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
            let job = self.job.as_raw_handle();
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
}

impl Drop for Group {
    fn drop(&mut self) {
        self.kill();
    }
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

    // Capabilities, as SIDs in buffers that live until the process has started.
    let mut sid_buffers: Vec<[u8; SECURITY_MAX_SID_SIZE as usize]> = Vec::new();
    sid_buffers.push(well_known_sid(WinCapabilityInternetClientSid)?);
    if confinement == Confinement::Owner {
        sid_buffers.push(well_known_sid(WinCapabilityPrivateNetworkClientServerSid)?);
    }
    let mut capabilities: Vec<SID_AND_ATTRIBUTES> = sid_buffers
        .iter_mut()
        .map(|b| SID_AND_ATTRIBUTES { Sid: b.as_mut_ptr() as PSID, Attributes: SE_GROUP_ENABLED })
        .collect();
    if let Some(toolchains) = toolchain_capability() {
        grant_toolchains(toolchains, workspace.parent().unwrap_or(workspace));
        capabilities.push(SID_AND_ATTRIBUTES { Sid: toolchains, Attributes: SE_GROUP_ENABLED });
    }
    let security = SECURITY_CAPABILITIES {
        AppContainerSid: container.as_ref().map_or(null_mut(), |c| c.sid),
        Capabilities: capabilities.as_mut_ptr(),
        CapabilityCount: capabilities.len() as u32,
        Reserved: 0,
    };

    // Output pipes, and NUL for input. Only the child's ends are inheritable, and the
    // handle list below limits inheritance to exactly these.
    let (stdout_read, stdout_write) = pipe()?;
    let (stderr_read, stderr_write) = pipe()?;
    let stdin = File::open("NUL")?;
    for h in [stdout_write.as_raw_handle(), stderr_write.as_raw_handle(), stdin.as_raw_handle()] {
        // SAFETY: valid handles we own.
        check(unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) })?;
    }
    let inherit: [HANDLE; 3] = [stdin.as_raw_handle(), stdout_write.as_raw_handle(), stderr_write.as_raw_handle()];

    let attributes = AttributeList::new(if container.is_some() { 2 } else { 1 })?;
    attributes.set(PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit.as_ptr() as _, size_of_val(&inherit))?;
    if container.is_some() {
        attributes.set(PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, &security as *const _ as _, size_of_val(&security))?;
    }

    // SAFETY: STARTUPINFOEXW is plain data; every pointer in it outlives CreateProcessW.
    let mut startup: STARTUPINFOEXW = unsafe { zeroed() };
    startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = inherit[0];
    startup.StartupInfo.hStdOutput = inherit[1];
    startup.StartupInfo.hStdError = inherit[2];
    startup.lpAttributeList = attributes.ptr();

    let path = env.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case("PATH")).map(|(_, v)| v.clone()).or_else(|| std::env::var_os("PATH"));
    let (application, mut command_line) = command_line(program, args, cwd, path.as_deref());
    let environment = environment_block(env);
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
    // In the job before it runs a single instruction, so nothing it starts escapes.
    // SAFETY: valid handles.
    if let Err(e) = check(unsafe { AssignProcessToJobObject(job.as_raw_handle(), process.as_raw_handle()) }) {
        unsafe { TerminateProcess(process.as_raw_handle(), 1) };
        return Err(e);
    }
    // SAFETY: the suspended main thread of the process we just made.
    unsafe { ResumeThread(thread.as_raw_handle()) };

    Ok(Spawned {
        pid: info.dwProcessId,
        process,
        stdout: File::from(stdout_read),
        stderr: File::from(stderr_read),
        group: Group { job, _container: container },
    })
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
/// cmd.exe given a batch file's full path says "Access is denied" inside the AppContainer
/// (it looks through folders above it that the job may not list), so a batch file inside
/// the job's folder is named relative to it, and one in a PATH folder (`npm.cmd`) by its
/// name alone, for cmd.exe to find on PATH.
fn command_line(program: &Path, args: &[String], cwd: &Path, path: Option<&OsStr>) -> (Vec<u16>, Vec<u16>) {
    let is_batch = program
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"));
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
    fn environment_overrides_are_case_insensitive() {
        let block = environment_block(&[("path".into(), "X".into())]);
        let text = String::from_utf16_lossy(&block);
        let paths: Vec<_> = text.split('\0').filter(|v| v.to_uppercase().starts_with("PATH=")).collect();
        assert_eq!(paths, ["path=X"]);
        assert!(text.ends_with("\0\0"));
    }
}
