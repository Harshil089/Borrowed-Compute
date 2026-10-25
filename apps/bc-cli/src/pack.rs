//! Packs the project into a tar archive.
//!
//! Phase one sends the whole archive on every run. Chunked, cached sync replaces this in
//! Phase two; the file selection rules stay the same.

use std::path::Path;

use anyhow::{Context, Result};
use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;

/// Archives `root`, honoring `.gitignore` and `excludes`, and always skipping `.git`.
/// Paths in the archive use `/` separators so they unpack the same way on the Mac.
pub fn pack(root: &Path, excludes: &[String]) -> Result<Vec<u8>> {
    let mut globs = GlobSetBuilder::new();
    for pattern in excludes {
        globs.add(Glob::new(pattern).with_context(|| format!("bad exclude {pattern:?}"))?);
    }
    let globs = globs.build()?;

    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(false)
        .require_git(false)
        .filter_entry(|e| e.file_name() != ".git")
        .build();

    let mut tar = tar::Builder::new(Vec::new());
    tar.follow_symlinks(false);
    for entry in walker {
        let entry = entry?;
        let is_file_or_link = entry
            .file_type()
            .is_some_and(|t| t.is_file() || t.is_symlink());
        if !is_file_or_link {
            continue;
        }
        let rel = entry.path().strip_prefix(root)?;
        let name = rel.to_string_lossy().replace('\\', "/");
        if globs.is_match(&name) {
            continue;
        }
        tar.append_path_with_name(entry.path(), &name)
            .with_context(|| format!("adding {name}"))?;
    }
    Ok(tar.into_inner()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn names(archive: &[u8]) -> Vec<String> {
        let mut names: Vec<String> = tar::Archive::new(archive)
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
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
        ] {
            let p = root.join(path);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, body).unwrap();
        }

        let archive = pack(root, &["data/**".into()]).unwrap();
        assert_eq!(
            names(&archive),
            [".env.example", ".gitignore", "Cargo.toml", "src/main.rs"]
        );
    }

    #[test]
    fn rejects_bad_glob() {
        let dir = tempfile::tempdir().unwrap();
        assert!(pack(dir.path(), &["[".into()]).is_err());
    }
}
