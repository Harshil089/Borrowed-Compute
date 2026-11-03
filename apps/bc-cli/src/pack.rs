//! Packs the project into a tar archive.
//!
//! Phase one sends the whole archive on every run. Chunked, cached sync replaces this in
//! Phase two; the file selection rules stay the same.
//!
//! Files written on Windows carry no Unix permissions and often have CRLF line endings,
//! and both break scripts on the Mac. So the archive decides each file's mode from the
//! file system, the git index and the shebang, and converts CRLF to LF in scripts.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bc_proto::wire::MAX_ARCHIVE_BYTES;
use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;

const EXEC_MODE: u32 = 0o755;
const FILE_MODE: u32 = 0o644;
/// How much of a file is checked for NUL bytes to decide it is binary.
const BINARY_PROBE: usize = 8 * 1024;
const SCRIPT_EXTENSIONS: [&str; 3] = ["sh", "bash", "zsh"];

/// A packed project. It lives in a temporary file, so its size is not bounded by memory.
#[derive(Clone)]
pub struct Archive {
    file: Arc<tempfile::NamedTempFile>,
    pub len: u64,
}

impl Archive {
    /// A new handle on the archive, read from the start.
    pub fn open(&self) -> std::io::Result<File> {
        self.file.reopen()
    }
}

/// Files above this size are streamed into the archive and never treated as scripts.
const MAX_SCRIPT_BYTES: u64 = 1 << 20;

/// Archives `root`, honoring `.gitignore` and `excludes`, and always skipping `.git` and
/// `node_modules`: install commands rebuild it on the Mac, and it is often most of the
/// project. Paths in the archive use `/` separators so they unpack the same way on the Mac.
/// A project over [`MAX_ARCHIVE_BYTES`] fails before anything is read.
pub fn pack(root: &Path, excludes: &[String]) -> Result<Archive> {
    let mut globs = GlobSetBuilder::new();
    for pattern in excludes {
        globs.add(Glob::new(pattern).with_context(|| format!("bad exclude {pattern:?}"))?);
    }
    let globs = globs.build()?;
    let git_modes = git_exec_modes(root);

    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(false)
        .require_git(false)
        .filter_entry(|e| {
            e.file_name() != ".git" && !(e.file_name() == "node_modules" && e.file_type().is_some_and(|t| t.is_dir()))
        })
        .build();

    let mut files = Vec::new();
    for entry in walker {
        let entry = entry?;
        let Some(file_type) = entry.file_type() else { continue };
        if !file_type.is_file() && !file_type.is_symlink() {
            continue;
        }
        let rel = entry.path().strip_prefix(root)?;
        let name = rel.to_string_lossy().replace('\\', "/");
        if globs.is_match(&name) {
            continue;
        }
        let size = if file_type.is_file() { entry.metadata().with_context(|| format!("reading {name}"))?.len() } else { 0 };
        files.push((entry, name, size));
    }
    let total: u64 = files.iter().map(|(_, _, size)| size).sum();
    if total > MAX_ARCHIVE_BYTES {
        bail!("{}", too_big(files.iter().map(|(_, name, size)| (name.as_str(), *size)), total));
    }

    let mut tar = tar::Builder::new(BufWriter::new(tempfile::NamedTempFile::new().context("creating the archive")?));
    tar.follow_symlinks(false);
    for (entry, name, size) in files {
        if entry.file_type().is_some_and(|t| t.is_symlink()) {
            tar.append_path_with_name(entry.path(), &name)
                .with_context(|| format!("adding {name}"))?;
            continue;
        }

        let metadata = entry.metadata().with_context(|| format!("reading {name}"))?;
        let git_exec = git_modes.as_ref().map(|modes| modes.get(&name).copied());
        let mut header = tar::Header::new_gnu();
        header.set_mtime(mtime_secs(&metadata));
        header.set_entry_type(tar::EntryType::Regular);
        if size > MAX_SCRIPT_BYTES {
            let mut file = File::open(entry.path()).with_context(|| format!("reading {name}"))?;
            let mut head = [0u8; 2];
            let read = file.read(&mut head)?;
            file.rewind()?;
            let exec = is_executable(fs_executable(&metadata), git_exec, &head[..read]);
            header.set_size(size);
            header.set_mode(if exec { EXEC_MODE } else { FILE_MODE });
            tar.append_data(&mut header, &name, file.take(size))
                .with_context(|| format!("adding {name}"))?;
            continue;
        }
        let mut data = std::fs::read(entry.path()).with_context(|| format!("reading {name}"))?;
        let exec = is_executable(fs_executable(&metadata), git_exec, &data);
        if is_script(&name, exec, &data) {
            data = crlf_to_lf(data);
        }
        header.set_size(data.len() as u64);
        header.set_mode(if exec { EXEC_MODE } else { FILE_MODE });
        tar.append_data(&mut header, &name, data.as_slice())
            .with_context(|| format!("adding {name}"))?;
    }
    let file = tar.into_inner()?.into_inner().map_err(|e| e.into_error())?;
    let len = file.as_file().metadata()?.len();
    Ok(Archive { file: Arc::new(file), len })
}

/// Says the project is too big and names its three biggest top-level files or folders.
fn too_big<'a>(files: impl Iterator<Item = (&'a str, u64)>, total: u64) -> String {
    let gb = |bytes: u64| format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64);
    let mut by_top: HashMap<&str, u64> = HashMap::new();
    for (name, size) in files {
        *by_top.entry(name.split('/').next().unwrap_or(name)).or_default() += size;
    }
    let mut top: Vec<_> = by_top.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let biggest: Vec<_> = top.iter().take(3).map(|(name, size)| format!("{name} ({})", gb(*size))).collect();
    format!(
        "the project is {} and the limit is {}. Biggest: {}. Add what the run does not need to .gitignore.",
        gb(total),
        gb(MAX_ARCHIVE_BYTES),
        biggest.join(", ")
    )
}

/// Executable if the file system says so, or git records it as `100755`. A file git
/// does not track falls back to its shebang, since Windows cannot mark it executable.
///
/// `git_exec` is `None` outside a git repo, `Some(None)` for an untracked file, and
/// `Some(Some(exec))` for a tracked one.
pub fn is_executable(fs_exec: bool, git_exec: Option<Option<bool>>, data: &[u8]) -> bool {
    match git_exec {
        Some(Some(git)) => fs_exec || git,
        Some(None) | None => fs_exec || data.starts_with(b"#!"),
    }
}

/// Scripts get LF line endings: a CR left on the shebang line or in a shell script breaks
/// it on the Mac. Other text files (`.bat`, data, fixtures) are sent untouched.
pub fn is_script(name: &str, exec: bool, data: &[u8]) -> bool {
    if looks_binary(data) {
        return false;
    }
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    exec || data.starts_with(b"#!") || ext.is_some_and(|e| SCRIPT_EXTENSIONS.contains(&e.as_str()))
}

fn looks_binary(data: &[u8]) -> bool {
    data[..data.len().min(BINARY_PROBE)].contains(&0)
}

/// Replaces each CRLF pair with LF. A lone CR is kept.
pub fn crlf_to_lf(data: Vec<u8>) -> Vec<u8> {
    if !data.windows(2).any(|w| w == b"\r\n") {
        return data;
    }
    let mut out = Vec::with_capacity(data.len());
    let mut bytes = data.iter().peekable();
    while let Some(&b) = bytes.next() {
        if b == b'\r' && bytes.peek() == Some(&&b'\n') {
            continue;
        }
        out.push(b);
    }
    out
}

#[cfg(unix)]
fn fs_executable(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn fs_executable(_metadata: &std::fs::Metadata) -> bool {
    false
}

fn mtime_secs(metadata: &std::fs::Metadata) -> u64 {
    metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Tracked files under `root` and whether git records them as executable. `None` when
/// git is not installed or `root` is not in a repository.
fn git_exec_modes(root: &Path) -> Option<HashMap<String, bool>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-s", "-z"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_ls_files(&output.stdout))
}

/// Parses `git ls-files -s -z` output: `<mode> <sha> <stage>\t<path>\0` per entry.
fn parse_ls_files(out: &[u8]) -> HashMap<String, bool> {
    out.split(|&b| b == 0)
        .filter_map(|record| {
            let record = std::str::from_utf8(record).ok()?;
            let (meta, path) = record.split_once('\t')?;
            let mode = meta.split(' ').next()?;
            Some((path.to_string(), mode == "100755"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Entry {
        mode: u32,
        data: Vec<u8>,
    }

    fn entries(archive: &Archive) -> HashMap<String, Entry> {
        tar::Archive::new(archive.open().unwrap())
            .entries()
            .unwrap()
            .map(|e| {
                let mut e = e.unwrap();
                let name = e.path().unwrap().to_string_lossy().into_owned();
                let mode = e.header().mode().unwrap();
                let mut data = Vec::new();
                std::io::Read::read_to_end(&mut e, &mut data).unwrap();
                (name, Entry { mode, data })
            })
            .collect()
    }

    fn write(root: &Path, path: &str, body: &[u8]) {
        let p = root.join(path);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
        }
    }

    fn git(root: &Path, args: &[&str]) -> bool {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn honors_gitignore_excludes_and_skips_git() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for (path, body) in [
            ("src/main.rs", "fn main() {}"),
            ("Cargo.toml", "[package]"),
            (".gitignore", "target/\n*.log\n"),
            ("target/debug/app", "binary"),
            ("run.log", "noise"),
            (".git/HEAD", "ref"),
            ("data/big.bin", "data"),
            (".env.example", "A=1"),
            ("node_modules/left-pad/index.js", "module.exports = 1"),
            ("web/node_modules/x.js", "1"),
        ] {
            write(root, path, body.as_bytes());
        }

        let archive = pack(root, &["data/**".into()]).unwrap();
        let mut names: Vec<_> = entries(&archive).into_keys().collect();
        names.sort();
        assert_eq!(names, [".env.example", ".gitignore", "Cargo.toml", "src/main.rs"]);
    }

    #[test]
    fn streams_big_files_and_keeps_their_mode() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!("#!/bin/sh\r\n{}", "x".repeat(MAX_SCRIPT_BYTES as usize));
        write(dir.path(), "big.sh", body.as_bytes());
        let archive = pack(dir.path(), &[]).unwrap();
        let big = &entries(&archive)["big.sh"];
        // Too big to be a script: sent as is, with its mode from the shebang.
        assert_eq!(big.data, body.as_bytes());
        assert_eq!(big.mode, EXEC_MODE);
    }

    #[test]
    fn too_big_names_the_biggest_folders() {
        let gb = 1u64 << 30;
        let files = [("node_modules/a", 5 * gb), ("dist/b", gb), ("src/c", 1), ("dist/d", gb), ("x", 2)];
        let msg = too_big(files.into_iter(), 7 * gb);
        assert!(msg.starts_with("the project is 7.0 GB and the limit is 20.0 GB."), "{msg}");
        assert!(msg.contains("Biggest: node_modules (5.0 GB), dist (2.0 GB), x (0.0 GB)."), "{msg}");
    }

    #[test]
    fn rejects_bad_glob() {
        let dir = tempfile::tempdir().unwrap();
        assert!(pack(dir.path(), &["[".into()]).is_err());
    }

    #[test]
    fn executable_rule() {
        let script = b"#!/bin/sh\n";
        let plain = b"echo hi\n";
        // No git: file system, then shebang.
        assert!(is_executable(true, None, plain));
        assert!(is_executable(false, None, script));
        assert!(!is_executable(false, None, plain));
        // Untracked file in a repo: same as no git.
        assert!(is_executable(false, Some(None), script));
        assert!(!is_executable(false, Some(None), plain));
        // Tracked file: git's mode wins over the shebang.
        assert!(is_executable(false, Some(Some(true)), plain));
        assert!(!is_executable(false, Some(Some(false)), script));
        assert!(is_executable(true, Some(Some(false)), plain));
    }

    #[test]
    fn script_rule() {
        assert!(is_script("build.sh", false, b"echo\r\n"));
        assert!(is_script("tools/RUN.BASH", false, b"echo\r\n"));
        assert!(is_script("gradlew", true, b"echo\r\n"));
        assert!(is_script("tool", false, b"#!/usr/bin/env python3\r\n"));
        assert!(!is_script("run.bat", false, b"@echo off\r\n"));
        assert!(!is_script("data.csv", false, b"a,b\r\n"));
        assert!(!is_script("blob.sh", true, b"\x00\x01\r\n"));
    }

    #[test]
    fn converts_only_crlf_pairs() {
        assert_eq!(crlf_to_lf(b"a\r\nb\r\n".to_vec()), b"a\nb\n");
        assert_eq!(crlf_to_lf(b"a\rb\n".to_vec()), b"a\rb\n");
        assert_eq!(crlf_to_lf(b"\r\r\n".to_vec()), b"\r\n");
        assert_eq!(crlf_to_lf(b"no endings".to_vec()), b"no endings");
        assert_eq!(crlf_to_lf(Vec::new()), b"");
    }

    #[test]
    fn parses_git_ls_files() {
        let out = b"100755 abc 0\tgradlew\x00100644 def 0\tsrc/a b.rs\x00120000 fed 0\tlink\x00";
        let modes = parse_ls_files(out);
        assert_eq!(modes.get("gradlew"), Some(&true));
        assert_eq!(modes.get("src/a b.rs"), Some(&false));
        assert_eq!(modes.get("link"), Some(&false));
        assert_eq!(modes.len(), 3);
    }

    #[test]
    fn fixes_windows_scripts_and_leaves_other_files_alone() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(root, "build.sh", b"#!/bin/sh\r\necho hi\r\n");
        write(root, "data.bat", b"@echo off\r\necho hi\r\n");
        write(root, "blob.bin", b"\x00\x01\r\n\x02");
        write(root, "notes.txt", b"line\r\n");

        let files = entries(&pack(root, &[]).unwrap());
        assert_eq!(files["build.sh"].mode, EXEC_MODE);
        assert_eq!(files["build.sh"].data, b"#!/bin/sh\necho hi\n");
        assert_eq!(files["data.bat"].mode, FILE_MODE);
        assert_eq!(files["data.bat"].data, b"@echo off\r\necho hi\r\n");
        assert_eq!(files["blob.bin"].data, b"\x00\x01\r\n\x02");
        assert_eq!(files["notes.txt"].data, b"line\r\n");
    }

    #[test]
    fn uses_git_executable_bit() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        if !git(root, &["init", "-q"]) {
            eprintln!("git not available; skipping");
            return;
        }
        write(root, "gradlew", b"exec java -jar wrapper.jar \"$@\"\r\n");
        write(root, "tool.sh", b"#!/bin/sh\necho tracked but not executable\n");
        write(root, "untracked", b"#!/bin/sh\necho hi\n");
        assert!(git(root, &["add", "gradlew", "tool.sh"]));
        assert!(git(root, &["update-index", "--chmod=+x", "gradlew"]));

        let files = entries(&pack(root, &[]).unwrap());
        assert_eq!(files["gradlew"].mode, EXEC_MODE, "git records +x");
        assert_eq!(files["gradlew"].data, b"exec java -jar wrapper.jar \"$@\"\n");
        assert_eq!(files["tool.sh"].mode, FILE_MODE, "git records -x despite the shebang");
        assert_eq!(files["untracked"].mode, EXEC_MODE, "untracked falls back to the shebang");
    }
}
