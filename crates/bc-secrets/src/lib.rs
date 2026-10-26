//! Where a device keeps its secrets: its private key and its cloud session token.
//!
//! The apps and CLIs call [`use_keychain`] at startup, which puts secrets in the OS
//! keychain (macOS Keychain, Windows Credential Manager) instead of files. A secret
//! still in its old file moves into the keychain on first read and the file is deleted.
//! When the keychain cannot be used (no keychain, access refused), secrets stay in
//! files readable only by their owner, so a device never gets locked out.
//!
//! Libraries and tests leave the keychain off and use files only, so test runs never
//! touch the developer's real keychain.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

/// Keychain service name shared by all entries.
const SERVICE: &str = "dev.borrowedcompute";

static KEYCHAIN: AtomicBool = AtomicBool::new(false);

/// Stores secrets in the OS keychain from now on. Call once at program start.
pub fn use_keychain() {
    KEYCHAIN.store(true, Ordering::Relaxed);
}

/// A secret is identified by its file, which also names its keychain entry: the same
/// file always means the same entry, and different homes (the Mac agent, the CLI, test
/// folders) never share one.
fn entry(path: &Path) -> Option<keyring::Entry> {
    if !KEYCHAIN.load(Ordering::Relaxed) {
        return None;
    }
    keyring::Entry::new(SERVICE, &path.to_string_lossy()).ok()
}

/// Reads the secret stored for `path`, moving it from the file into the keychain when
/// the keychain is on.
pub fn read(path: &Path) -> Option<Vec<u8>> {
    if let Some(entry) = entry(path) {
        if let Ok(secret) = entry.get_secret() {
            return Some(secret);
        }
        let from_file = std::fs::read(path).ok()?;
        // Delete the file only once the keychain has a verified copy.
        if entry.set_secret(&from_file).is_ok() && entry.get_secret().ok().as_deref() == Some(&from_file[..]) {
            let _ = std::fs::remove_file(path);
        }
        return Some(from_file);
    }
    std::fs::read(path).ok()
}

/// Stores `secret` for `path`: in the keychain when it is on and works, else in an
/// owner-only file.
pub fn write(path: &Path, secret: &[u8]) -> Result<()> {
    if let Some(entry) = entry(path) {
        if entry.set_secret(secret).is_ok() {
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
    }
    write_file(path, secret)
}

/// Removes the secret for `path` from both the keychain and the file.
pub fn delete(path: &Path) {
    if let Some(entry) = entry(path) {
        let _ = entry.delete_credential();
    }
    let _ = std::fs::remove_file(path);
}

fn write_file(path: &Path, secret: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path).with_context(|| format!("writing {}", path.display()))?;
    std::io::Write::write_all(&mut file, secret)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_without_keychain() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("session");
        assert_eq!(read(&path), None);
        write(&path, b"bcs_123").unwrap();
        assert_eq!(read(&path).as_deref(), Some(&b"bcs_123"[..]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "owner-only file");
        }
        delete(&path);
        assert_eq!(read(&path), None);
    }
}
