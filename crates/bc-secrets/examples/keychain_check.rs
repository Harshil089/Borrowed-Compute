//! Manual check against the real OS keychain: migrates a file-held secret into it, reads
//! it back, and cleans up. Uses a temporary path, so real secrets are never touched.
//! Run with `cargo run -p bc-secrets --example keychain_check`.

fn main() {
    bc_secrets::use_keychain();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session");
    std::fs::write(&path, b"bcs_migrate_me").unwrap();

    let read = bc_secrets::read(&path);
    println!("read back: {:?}", read.as_deref().map(String::from_utf8_lossy));
    println!("file after migration exists: {}", path.exists());
    println!("read again (keychain only): {:?}", bc_secrets::read(&path).as_deref().map(String::from_utf8_lossy));

    bc_secrets::write(&path, b"bcs_updated").unwrap();
    println!("after write, file exists: {}", path.exists());
    println!("updated: {:?}", bc_secrets::read(&path).as_deref().map(String::from_utf8_lossy));

    bc_secrets::delete(&path);
    println!("after delete: {:?}", bc_secrets::read(&path));
}
