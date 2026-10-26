//! Runs each job inside a macOS Seatbelt sandbox, so a job reads and writes its own
//! workspace and nothing else of the owner's.
//!
//! The profile builds on Apple's `bsd.sb` (deny by default, plus what system libraries
//! need) and allows:
//! - read and write: the job's workspace (project, caches, temporary files);
//! - read: system folders, Homebrew, Xcode, and the user's toolchains (rustup, nvm,
//!   pyenv and similar, plus every PATH folder inside the home folder);
//! - network: outbound anywhere, since builds download dependencies; inbound only on
//!   localhost and on Unix sockets inside the workspace, so test servers and tools that
//!   talk to themselves work but nothing outside can connect in.
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

/// System folders jobs may read.
const SYSTEM_READ: &[&str] = &[
    "/usr", "/bin", "/sbin", "/System", "/Library", "/opt", "/Applications/Xcode.app",
    "/private/etc", "/private/var/db", "/private/var/select", "/dev",
];

/// The sandbox-exec argv prefix for a job in `workspace`, or `None` when sandboxing is
/// not available here (not macOS) or turned off.
pub fn wrapper(workspace: &Path, path_env: Option<&OsStr>) -> Option<Vec<OsString>> {
    if !cfg!(target_os = "macos") || std::env::var_os("BC_NO_SANDBOX").is_some() {
        return None;
    }
    if !Path::new(SANDBOX_EXEC).is_file() {
        return None;
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    Some(vec![SANDBOX_EXEC.into(), "-p".into(), profile(workspace, home.as_deref(), path_env).into()])
}

/// The Seatbelt profile. Paths are resolved first: Seatbelt matches real paths, and on
/// macOS /var and /tmp are links into /private.
pub fn profile(workspace: &Path, home: Option<&Path>, path_env: Option<&OsStr>) -> String {
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
    sb += "(allow sysctl-read mach-lookup ipc-posix-shm ipc-posix-sem iokit-open user-preference-read)\n";
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
    sb += "(allow network-outbound)\n";
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
        let sb = profile(ws.path(), Some(home.path()), Some(&path_env));
        let home_real = real(home.path());
        assert!(sb.contains(&format!("(subpath {})", quote(&home_real.join(".rustup")))));
        assert!(sb.contains(&format!("(subpath {})", quote(&home_real.join(".cargo/bin")))));
        assert!(!sb.contains(&format!("(subpath {})", quote(&home_real))), "the home folder itself is not readable");
        assert!(!sb.contains(&format!("(subpath {})", quote(&home_real.join(".cargo")))), "~/.cargo holds credentials");
        assert!(sb.contains(&format!("(allow file-read* file-write* (subpath {}))", quote(&real(ws.path())))));
    }

    #[test]
    fn quotes_paths() {
        assert_eq!(quote(Path::new("/a \"b\"/c")), "\"/a \\\"b\\\"/c\"");
    }
}
