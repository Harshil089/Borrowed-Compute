//! Runs each job inside a macOS Seatbelt sandbox, so a job reads and writes its own
//! workspace and nothing else of the owner's.
//!
//! The profile builds on Apple's `bsd.sb` (deny by default, plus what system libraries
//! need) and allows:
//! - read and write: the job's workspace (project, caches, temporary files);
//! - read: system folders, Homebrew, Xcode, and the user's toolchains (rustup, nvm,
//!   pyenv and similar, plus every PATH folder inside the home folder);
//! - network: outbound TCP and UDP anywhere, since builds download dependencies, except
//!   well-known services on this Mac's localhost (databases, Redis, Docker, Ollama, ssh);
//!   inbound only on localhost; Unix sockets only inside the workspace (plus DNS), so the
//!   owner's ssh agent and other local sockets are out of reach;
//! - system services, except the clipboard, the keychain, LaunchServices (`open`) and
//!   Apple Events, and preferences of the system and developer tools only.
//!
//! Everything else in the home folder (~/Documents, ~/.ssh, ~/.aws, ~/.cargo except its
//! bin folder) is unreadable. Jobs still run as the owner's user: the separate `_bcjob`
//! user needs a root helper and is Phase two. `BC_NO_SANDBOX=1` on the agent turns this
//! off for debugging.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Folders under the home folder that hold toolchains, read-only to jobs. Not ~/.cargo:
/// besides `bin` it holds crates.io credentials; `bin` comes in through PATH.
const HOME_TOOLCHAINS: &[&str] = &[
    ".rustup", ".nvm", ".volta", ".bun", ".deno", ".pyenv", ".rbenv", ".sdkman", ".asdf",
    ".local/share/mise", ".local/share/uv/python", "go/bin", "Library/Python",
];

/// Mach services jobs may not look up: pasteboard (clipboard), keychain, LaunchServices
/// (`open`, whose apps run unsandboxed) and Apple Events.
const DENIED_SERVICES: &[&str] = &[
    "com.apple.pasteboard.1",
    "com.apple.SecurityServer",
    "com.apple.securityd",
    "com.apple.secd",
    "com.apple.security.agent",
    "com.apple.coreservices.launchservicesd",
    "com.apple.coreservices.quarantine-resolver",
    "com.apple.coreservices.appleevents",
    "com.apple.lsd.mapdb",
    "com.apple.lsd.modifydb",
    "com.apple.lsd.open",
];

/// Preference domains jobs may read: the global one, which every program reads, and the
/// developer tools'.
const PREFERENCE_DOMAINS: &[&str] = &[
    "kCFPreferencesAnyApplication",
    "com.apple.dt.Xcode",
    "com.apple.dt.xcodebuild",
    "com.apple.dt.SourceKit",
];

/// Ports on this Mac's localhost that jobs may not connect to: ssh, MySQL, PostgreSQL,
/// Redis, Memcached, MongoDB, SQL Server, CouchDB, Elasticsearch, InfluxDB, Docker and
/// Ollama.
const DENIED_LOCAL_PORTS: &[u16] = &[22, 3306, 5432, 6379, 11211, 27017, 1433, 5984, 9200, 8086, 2375, 2376, 11434];

/// System folders jobs may read.
const SYSTEM_READ: &[&str] = &[
    "/usr", "/bin", "/sbin", "/System", "/Library", "/opt", "/Applications/Xcode.app",
    "/private/etc", "/private/var/db", "/private/var/select", "/dev",
];

/// The sandbox-exec argv prefix for a job in `workspace`, or `None` when sandboxing is
/// not available here (not macOS) or turned off.
pub fn wrapper(workspace: &Path, path_env: Option<&OsStr>, denied_ports: &[u16]) -> Option<Vec<OsString>> {
    if !cfg!(target_os = "macos") || std::env::var_os("BC_NO_SANDBOX").is_some() {
        return None;
    }
    if !Path::new(SANDBOX_EXEC).is_file() {
        return None;
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    Some(vec![SANDBOX_EXEC.into(), "-p".into(), profile(workspace, home.as_deref(), path_env, denied_ports).into()])
}

/// The Seatbelt profile. Paths are resolved first: Seatbelt matches real paths, and on
/// macOS /var and /tmp are links into /private.
pub fn profile(workspace: &Path, home: Option<&Path>, path_env: Option<&OsStr>, denied_ports: &[u16]) -> String {
    let mut read: Vec<PathBuf> = SYSTEM_READ.iter().map(PathBuf::from).collect();
    if let Some(home) = home {
        read.extend(HOME_TOOLCHAINS.iter().map(|d| home.join(d)));
        // Every PATH folder inside the home folder, such as ~/.cargo/bin or an nvm node's
        // bin. Folders outside home are system ones, covered above or harmless to read.
        if let Some(path_env) = path_env {
            read.extend(std::env::split_paths(path_env).filter(|d| d.starts_with(home)));
        }
    }
    let given = workspace.to_path_buf();
    let workspace = real(workspace);
    let mut sb = String::from("(version 1)\n(import \"bsd.sb\")\n");
    sb += "(allow process-fork process-exec)\n";
    sb += "(allow signal (target same-sandbox))\n";
    sb += "(allow sysctl-read mach-lookup ipc-posix-shm ipc-posix-sem iokit-open)\n";
    // System services that would reach past the workspace: the clipboard, the keychain,
    // opening apps (which run outside the sandbox) and Apple Events to other apps.
    sb += "(deny mach-lookup";
    for name in DENIED_SERVICES {
        sb += &format!(" (global-name \"{name}\")");
    }
    sb += ")\n";
    // Settings of the system and developer tools only, not every app's (they can hold
    // account names and tokens).
    sb += "(allow user-preference-read";
    for domain in PREFERENCE_DOMAINS {
        sb += &format!(" (preference-domain \"{domain}\")");
    }
    sb += ")\n";
    sb += "(allow file-read-metadata)\n";
    sb += "(allow file-read*";
    for dir in read.iter().map(|d| real(d)).filter(|d| d.exists()) {
        sb += &format!("\n  (subpath {})", quote(&dir));
    }
    sb += ")\n";
    sb += &format!("(allow file-read* file-write* (subpath {}))\n", quote(&workspace));
    sb += "(allow file-write* (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/dtracehelper\") (regex #\"^/dev/tty\") (subpath \"/dev/fd\"))\n";
    sb += "(allow file-ioctl (regex #\"^/dev/tty\"))\n";
    // xcrun keeps a lookup cache in the system temp folder whatever TMPDIR says.
    sb += "(allow file-read* file-write* (regex #\"^/private/var/folders/[^/]+/[^/]+/T/xcrun_db\"))\n";
    // Outbound TCP and UDP anywhere, for downloads. Unix sockets only in the workspace
    // (below) and DNS, so the owner's ssh agent and other local sockets stay out of reach.
    sb += "(allow network-outbound (remote ip))\n";
    sb += "(allow network-outbound (literal \"/private/var/run/mDNSResponder\"))\n";
    // Well-known local services (databases, caches, Docker, model servers, ssh) that the
    // owner may run without a password. A job's own servers on other ports still work.
    sb += "(deny network-outbound";
    for port in DENIED_LOCAL_PORTS.iter().chain(denied_ports) {
        sb += &format!(" (remote ip \"localhost:{port}\")");
    }
    sb += ")\n";
    sb += "(allow network-bind network-inbound (local ip \"localhost:*\"))\n";
    // Unix sockets inside the workspace (tsx and test runners talk to themselves this
    // way). Socket paths are matched as given, so allow both spellings of the path.
    sb += &format!(
        "(allow network-bind network-inbound network-outbound (local unix-socket (subpath {}) (subpath {})))\n",
        quote(&workspace),
        quote(&given)
    );
    sb
}

fn real(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// A Seatbelt string literal.
fn quote(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{s}\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_allows_workspace_and_toolchains_but_not_home() {
        let home = tempfile::tempdir().unwrap();
        let ws = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join(".rustup")).unwrap();
        std::fs::create_dir_all(home.path().join(".cargo/bin")).unwrap();
        let path_env = std::env::join_paths([home.path().join(".cargo/bin"), "/usr/bin".into()]).unwrap();
        let sb = profile(ws.path(), Some(home.path()), Some(&path_env), &[]);
        let home_real = real(home.path());
        assert!(sb.contains(&format!("(subpath {})", quote(&home_real.join(".rustup")))));
        assert!(sb.contains(&format!("(subpath {})", quote(&home_real.join(".cargo/bin")))));
        assert!(!sb.contains(&format!("(subpath {})", quote(&home_real))), "the home folder itself is not readable");
        assert!(!sb.contains(&format!("(subpath {})", quote(&home_real.join(".cargo")))), "~/.cargo holds credentials");
        assert!(sb.contains(&format!("(allow file-read* file-write* (subpath {}))", quote(&real(ws.path())))));
    }

    #[test]
    fn profile_shuts_out_clipboard_keychain_and_local_services() {
        let ws = tempfile::tempdir().unwrap();
        let sb = profile(ws.path(), None, None, &[8080]);
        assert!(sb.contains("(global-name \"com.apple.pasteboard.1\")"));
        assert!(sb.contains("(global-name \"com.apple.SecurityServer\")"));
        assert!(sb.contains("(global-name \"com.apple.coreservices.launchservicesd\")"));
        assert!(sb.contains("(remote ip \"localhost:5432\")"));
        assert!(sb.contains("(remote ip \"localhost:8080\")"), "the cloud's port when this Mac hosts it");
        assert!(!sb.contains("(allow network-outbound)\n"), "no blanket outbound, which includes every Unix socket");
        assert!(!sb.contains(" user-preference-read)"), "no blanket preference reading");
    }

    #[test]
    fn quotes_paths() {
        assert_eq!(quote(Path::new("/a \"b\"/c")), "\"/a \\\"b\\\"/c\"");
    }
}
