//! Client steps shared by the `bc` command and the desktop app.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_auth::CloudClient;
use bc_net::{home_dir, EndpointAddr, EndpointId};
use bc_proto::cloud::PairingStatus;
use bc_proto::JobSpec;

/// Where this device keeps its key, session, default Mac and timing log. The CLI and the
/// desktop app share it, so they are the same device.
pub fn client_home() -> Result<PathBuf> {
    Ok(home_dir()?.join("cli"))
}

pub fn key_path(home: &Path) -> PathBuf {
    home.join("device.key")
}

pub fn default_mac(home: &Path) -> Option<String> {
    std::fs::read_to_string(home.join("default-mac"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn set_default_mac(home: &Path, mac: &str) -> Result<()> {
    std::fs::create_dir_all(home)?;
    std::fs::write(home.join("default-mac"), format!("{mac}\n"))?;
    Ok(())
}

pub struct ResolvedMac {
    pub addr: EndpointAddr,
    /// Set when the cloud has not heard from the Mac recently.
    pub offline_warning: Option<String>,
}

/// Finds the Mac's address through the cloud when signed in, else treats `target` as a
/// device id and relies on public address lookup (development only).
pub async fn resolve_mac(cloud: &CloudClient, target: &str) -> Result<ResolvedMac> {
    let raw_id = target.parse::<EndpointId>().ok();
    if cloud.token().is_some() {
        match cloud.find_device(target).await {
            Ok(device) => {
                let id: EndpointId = device.endpoint_id.parse()?;
                let offline_warning = (!device.online).then(|| {
                    format!("{:?} has not checked in recently; it may be offline", device.name)
                });
                // Trust the cloud's address only for the key it claims to belong to.
                let addr = device
                    .addr
                    .and_then(|a| serde_json::from_value::<EndpointAddr>(a).ok())
                    .filter(|a| a.id == id)
                    .unwrap_or_else(|| id.into());
                return Ok(ResolvedMac { addr, offline_warning });
            }
            Err(e) if raw_id.is_none() => return Err(e),
            Err(_) => {}
        }
    }
    let id = raw_id
        .with_context(|| format!("{target:?} is not a device id; sign in to use device names"))?;
    Ok(ResolvedMac { addr: id.into(), offline_warning: None })
}

/// The job to run in `project`: `spec` if given, else `bc.toml` in the project if present.
/// A non-empty `argv` replaces the spec's command.
pub fn load_spec(project: &Path, spec: Option<PathBuf>, argv: Vec<String>) -> Result<JobSpec> {
    let default = project.join("bc.toml");
    let path = spec.or_else(|| default.exists().then_some(default));
    let mut spec = match path {
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            JobSpec::from_toml(&text)?
        }
        None if argv.is_empty() => bail!("nothing to run; give a command or add bc.toml"),
        None => return Ok(JobSpec::from_argv(argv)?),
    };
    if !argv.is_empty() {
        spec.job.argv = argv;
        spec.validate()?;
    }
    Ok(spec)
}

/// Waits until the Mac owner approves pairing `id`. Returns false if the code expired.
pub async fn wait_for_pairing(
    cloud: &CloudClient,
    me: &str,
    id: i64,
    expires_in: Duration,
    mut on_tick: impl FnMut(),
) -> Result<bool> {
    let started = Instant::now();
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let status = cloud
            .pairings(me)
            .await?
            .into_iter()
            .find(|p| p.id == id)
            .map(|p| p.status);
        match status {
            Some(PairingStatus::Approved) => return Ok(true),
            Some(PairingStatus::Pending) if started.elapsed() < expires_in => on_tick(),
            _ => return Ok(false),
        }
    }
}

/// Splits a typed command line into argv. Quotes group words; backslashes are kept
/// literally, because Windows users type paths like `scripts\run.py`. No shell runs the
/// result: the Mac executes argv directly.
pub fn split_command(line: &str) -> Result<Vec<String>> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    for c in line.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => current.push(c),
            (None, '"' | '\'') => {
                quote = Some(c);
                in_word = true;
            }
            (None, c) if c.is_whitespace() => {
                if in_word {
                    args.push(std::mem::take(&mut current));
                    in_word = false;
                }
            }
            (None, c) => {
                current.push(c);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        bail!("unclosed quote in command");
    }
    if in_word {
        args.push(current);
    }
    Ok(args)
}

/// The timing log feeds the milestone's latency numbers. A failure to write it never
/// fails the job.
pub fn append_timing(home: &Path, line: &str) {
    let _ = std::fs::create_dir_all(home);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(home.join("timings.jsonl"))
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Network settings for reaching a Mac: its own relay when its cloud advertises one, so
/// connection metadata stays off public relays; iroh's public relays otherwise.
pub async fn mac_net(cloud: &CloudClient, relay_only: bool) -> bc_net::NetConfig {
    let relays = cloud.relay().await.and_then(|url| bc_net::relay_map(&url).ok());
    bc_net::NetConfig { relay_only, relays, ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_overrides_spec_and_missing_both_fails() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_spec(dir.path(), None, vec![]).is_err());
        let spec = load_spec(dir.path(), None, vec!["make".into()]).unwrap();
        assert_eq!(spec.job.argv, ["make"]);

        std::fs::write(
            dir.path().join("bc.toml"),
            "version = 1\n[job]\nargv = [\"cargo\", \"test\"]\ntimeout = \"5m\"\n",
        )
        .unwrap();
        let from_file = load_spec(dir.path(), None, vec![]).unwrap();
        assert_eq!(from_file.job.argv, ["cargo", "test"]);
        let overridden = load_spec(dir.path(), None, vec!["make".into()]).unwrap();
        assert_eq!(overridden.job.argv, ["make"]);
        assert_eq!(overridden.job.timeout.as_deref(), Some("5m"));
    }

    #[test]
    fn splits_commands_like_a_user_expects() {
        assert_eq!(split_command("cargo test --release").unwrap(), ["cargo", "test", "--release"]);
        assert_eq!(split_command("  python  scripts\\run.py ").unwrap(), ["python", "scripts\\run.py"]);
        assert_eq!(
            split_command(r#"sh -c "echo hi; exit 3""#).unwrap(),
            ["sh", "-c", "echo hi; exit 3"]
        );
        assert_eq!(split_command("echo '' x").unwrap(), ["echo", "", "x"]);
        assert_eq!(split_command("a\"b c\"d").unwrap(), ["ab cd"]);
        assert!(split_command("").unwrap().is_empty());
        assert!(split_command("echo \"oops").is_err());
    }

    #[test]
    fn default_mac_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(default_mac(dir.path()), None);
        set_default_mac(dir.path(), "studio").unwrap();
        assert_eq!(default_mac(dir.path()).as_deref(), Some("studio"));
    }
}
