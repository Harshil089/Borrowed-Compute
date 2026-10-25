use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_auth::{cloud_url, default_device_name, save_session, CloudClient};
use bc_cli::pack::pack;
use bc_cli::run::{run_remote, Exit};
use bc_net::{bind, home_dir, load_or_create_key, EndpointAddr, EndpointId, NetConfig};
use bc_proto::cloud::{DeviceKind, PairingStatus};
use bc_proto::JobSpec;
use clap::{Parser, Subcommand};

/// Run code jobs on a paired Mac.
#[derive(Parser)]
#[command(name = "bc", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sign in with GitHub and register this device.
    Login {
        /// Name for this device. Defaults to the computer name.
        #[arg(long)]
        name: Option<String>,
    },
    /// Sign out on this device.
    Logout,
    /// List the devices on your account.
    Devices,
    /// Ask a Mac for permission to send it jobs, then wait for approval on the Mac.
    Pair { mac: String },
    /// Print this device's id.
    Id,
    /// Set the Mac that `bc run` uses by default (name or device id).
    Use { mac: String },
    /// Run a job on the Mac, for example `bc run -- cargo test`.
    Run {
        /// Mac name or device id. Defaults to the one set with `bc use`.
        #[arg(long)]
        to: Option<String>,
        /// Job spec file. Defaults to `bc.toml` in the current directory, if present.
        #[arg(long)]
        spec: Option<PathBuf>,
        /// Disable direct paths and send all traffic through the relay.
        #[arg(long)]
        relay_only: bool,
        /// Print phase timings when the job ends.
        #[arg(long)]
        timings: bool,
        /// Program and arguments. Overrides `job.argv` in the spec.
        #[arg(last = true)]
        argv: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let home = home_dir()?.join("cli");
    let key_path = home.join("device.key");
    let mut cloud = CloudClient::from_home(cloud_url(), &home);

    match cli.command {
        Command::Login { name } => {
            let key = load_or_create_key(&key_path)?;
            let (token, login) = cloud
                .login(|s| {
                    println!("Open {} and enter code {}", s.verification_uri, s.user_code);
                })
                .await?;
            save_session(&home, &token)?;
            let name = name.unwrap_or_else(default_device_name);
            let device = cloud.register(&key, &name, DeviceKind::current()).await?;
            println!("Signed in as {login}. This device is registered as {:?}.", device.name);
        }
        Command::Logout => {
            if cloud.token().is_some() {
                let _ = cloud.logout().await;
            }
            bc_auth::clear_session(&home);
            println!("Signed out.");
        }
        Command::Devices => {
            let me = load_or_create_key(&key_path)?.public().to_string();
            for d in signed_in(&cloud)?.devices().await? {
                let state = match (d.online, d.last_seen_secs_ago) {
                    (true, _) => "online".to_string(),
                    (false, Some(s)) => format!("last seen {}", ago(s)),
                    (false, None) => "never seen".to_string(),
                };
                let this = if d.endpoint_id == me { "  (this device)" } else { "" };
                println!("{:<20} {:<8} {:<20}{this}", d.name, d.kind.as_str(), state);
            }
        }
        Command::Pair { mac } => {
            let me = load_or_create_key(&key_path)?.public().to_string();
            let cloud = signed_in(&cloud)?;
            let request = cloud.request_pairing(&me, &mac).await?;
            if request.status == PairingStatus::Approved {
                println!("Already paired with {:?}.", request.mac_name);
                return Ok(());
            }
            println!(
                "Pairing code: {}\nOn the Mac {:?}, run:  bc-agent approve {}",
                request.code.as_deref().unwrap_or("?"),
                request.mac_name,
                request.code.as_deref().unwrap_or("<code>")
            );
            print!("Waiting for approval");
            std::io::stdout().flush()?;
            let deadline = Duration::from_secs(request.expires_in_secs.unwrap_or(600).max(0) as u64);
            let started = std::time::Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let now = cloud.pairings(&me).await?;
                match now.iter().find(|p| p.id == request.id).map(|p| p.status) {
                    Some(PairingStatus::Approved) => {
                        println!("\nPaired with {:?}.", request.mac_name);
                        break;
                    }
                    Some(PairingStatus::Pending) if started.elapsed() < deadline => {
                        print!(".");
                        std::io::stdout().flush()?;
                    }
                    _ => bail!("\nthe pairing code expired; run `bc pair {mac}` again"),
                }
            }
        }
        Command::Id => {
            let key = load_or_create_key(&key_path)?;
            println!("{}", key.public());
        }
        Command::Use { mac } => {
            std::fs::create_dir_all(&home)?;
            std::fs::write(home.join("default-mac"), format!("{mac}\n"))?;
            println!("bc run will use {mac}");
        }
        Command::Run { to, spec, relay_only, timings, argv } => {
            let target = match to {
                Some(mac) => mac,
                None => std::fs::read_to_string(home.join("default-mac"))
                    .context("no Mac chosen; pass --to <name> or run `bc use <name>`")?
                    .trim()
                    .to_string(),
            };
            let mac = resolve_mac(&cloud, &target).await?;
            let spec = load_spec(spec, argv)?;
            let root = std::env::current_dir()?;
            let archive = pack(&root, &spec.files.exclude)?;

            let key = load_or_create_key(&key_path)?;
            let endpoint = bind(key, NetConfig { relay_only, ..Default::default() }).await?;
            let cancel = async {
                let _ = tokio::signal::ctrl_c().await;
            };
            let outcome = run_remote(
                &endpoint,
                mac,
                &spec,
                archive,
                &mut std::io::stdout(),
                &mut std::io::stderr(),
                cancel,
            )
            .await?;
            endpoint.close().await;

            let line = outcome.timings.to_json(&outcome.job_id, &outcome.exit);
            append_timing(&home, &line);
            if timings {
                eprintln!("bc: {line}");
            }
            match &outcome.exit {
                Exit::Rejected(reason) => {
                    eprintln!("bc: the Mac rejected the job: {reason}");
                    eprintln!("bc: if you have not paired yet, run `bc pair {target}`");
                }
                Exit::Failed(reason) => eprintln!("bc: job failed on the Mac: {reason}"),
                Exit::TimedOut => eprintln!("bc: job timed out"),
                _ => {}
            }
            std::process::exit(outcome.exit.status());
        }
    }
    Ok(())
}

fn signed_in(cloud: &CloudClient) -> Result<&CloudClient> {
    if cloud.token().is_none() {
        bail!("not signed in; run `bc login`");
    }
    Ok(cloud)
}

/// Finds the Mac's address through the cloud when signed in, else treats `target` as a
/// device id and relies on public address lookup (development only).
async fn resolve_mac(cloud: &CloudClient, target: &str) -> Result<EndpointAddr> {
    let raw_id = target.parse::<EndpointId>().ok();
    if cloud.token().is_some() {
        match cloud.find_device(target).await {
            Ok(device) => {
                let id: EndpointId = device.endpoint_id.parse()?;
                if !device.online {
                    eprintln!("bc: warning: {:?} has not checked in recently; it may be offline", device.name);
                }
                // Trust the cloud's address only for the key it claims to belong to.
                let addr = device
                    .addr
                    .and_then(|a| serde_json::from_value::<EndpointAddr>(a).ok())
                    .filter(|a| a.id == id);
                return Ok(addr.unwrap_or_else(|| id.into()));
            }
            Err(e) if raw_id.is_none() => return Err(e),
            Err(_) => {}
        }
    }
    raw_id
        .map(Into::into)
        .with_context(|| format!("{target:?} is not a device id; run `bc login` to use device names"))
}

fn ago(secs: i64) -> String {
    match secs {
        s if s < 120 => format!("{s}s ago"),
        s if s < 7200 => format!("{}m ago", s / 60),
        s if s < 172_800 => format!("{}h ago", s / 3600),
        s => format!("{}d ago", s / 86_400),
    }
}

fn load_spec(path: Option<PathBuf>, argv: Vec<String>) -> Result<JobSpec> {
    let default = PathBuf::from("bc.toml");
    let path = path.or_else(|| default.exists().then_some(default));
    let mut spec = match path {
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            JobSpec::from_toml(&text)?
        }
        None if argv.is_empty() => bail!("nothing to run; pass a command after `--` or add bc.toml"),
        None => return Ok(JobSpec::from_argv(argv)?),
    };
    if !argv.is_empty() {
        spec.job.argv = argv;
        spec.validate()?;
    }
    Ok(spec)
}

/// The timing log feeds the milestone's latency numbers. A failure to write it never
/// fails the job.
fn append_timing(home: &Path, line: &str) {
    let _ = std::fs::create_dir_all(home);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(home.join("timings.jsonl"))
    {
        let _ = writeln!(f, "{line}");
    }
}
