use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_auth::{cloud_url_for, default_device_name, save_session, CloudClient};
use bc_cli::flows::{
    append_timing, client_home, default_mac, key_path, load_spec, resolve_mac, set_default_mac,
    wait_for_pairing,
};
use bc_cli::pack::pack;
use bc_cli::route::RouteMemory;
use bc_cli::run::{attach_service, list_services, run_on_mac, stop_service, Exit};
use bc_net::{load_or_create_key, NetConfig};
use bc_proto::cloud::{DeviceKind, PairingStatus, Provider};
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
        /// Sign in with Google instead of GitHub. Both reach the same account when they
        /// share a verified email address.
        #[arg(long)]
        google: bool,
    },
    /// Sign out on this device.
    Logout,
    /// List the devices on your account.
    Devices,
    /// Ask a Mac for permission to send it jobs, then wait for approval on the Mac.
    Pair { mac: String },
    /// Show the cloud address, or set it (for example your Mac's Tailscale address).
    /// An empty address returns to the default.
    Cloud { url: Option<String> },
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
        /// Run as a service, such as a dev server: no timeout, and it keeps running on
        /// the Mac (which stays awake) after Ctrl+C or a disconnect, until `bc stop`.
        #[arg(long)]
        service: bool,
        /// Program and arguments. Overrides `job.argv` in the spec.
        #[arg(last = true)]
        argv: Vec<String>,
    },
    /// List the services this device started on the Mac.
    Services {
        #[arg(long)]
        to: Option<String>,
    },
    /// Watch a running service's output. Ctrl+C stops watching, not the service.
    Attach {
        id: String,
        #[arg(long)]
        to: Option<String>,
    },
    /// Forward your services' ports on the Mac to this computer's localhost until Ctrl+C.
    Forward {
        #[arg(long)]
        to: Option<String>,
    },
    /// Stop a running service.
    Stop {
        id: String,
        #[arg(long)]
        to: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Keys and the session token belong in the OS keychain, not in files.
    bc_net::use_keychain();
    let cli = Cli::parse();
    let home = client_home()?;
    let key_path = key_path(&home);
    let mut cloud = CloudClient::from_home(cloud_url_for(&home), &home);

    match cli.command {
        Command::Login { name, google } => {
            let key = load_or_create_key(&key_path)?;
            let provider = if google { Provider::Google } else { Provider::Github };
            let (token, login) = cloud
                .login_with(provider, |s| {
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
                "Pairing code: {}\nOn the Mac {:?}, type this code into Borrowed Compute Agent, or run:  bc-agent approve {}",
                request.code.as_deref().unwrap_or("?"),
                request.mac_name,
                request.code.as_deref().unwrap_or("<code>")
            );
            print!("Waiting for approval");
            std::io::stdout().flush()?;
            let expires = Duration::from_secs(request.expires_in_secs.unwrap_or(600).max(0) as u64);
            let approved = wait_for_pairing(cloud, &me, request.id, expires, || {
                print!(".");
                let _ = std::io::stdout().flush();
            })
            .await?;
            if !approved {
                bail!("\nthe pairing code expired; run `bc pair {mac}` again");
            }
            println!("\nPaired with {:?}.", request.mac_name);
        }
        Command::Cloud { url } => match url {
            None => println!("{}", cloud.base()),
            Some(url) => {
                let url = bc_auth::save_cloud_url(&home, &url)?;
                CloudClient::new(&url, None)
                    .health()
                    .await
                    .with_context(|| format!("saved {url}, but it does not answer yet"))?;
                println!("Using the cloud at {url}. Sign in there with `bc login`.");
            }
        },
        Command::Id => {
            let key = load_or_create_key(&key_path)?;
            println!("{}", key.public());
        }
        Command::Use { mac } => {
            set_default_mac(&home, &mac)?;
            println!("bc run will use {mac}");
        }
        Command::Services { to } => {
            let (_, mac) = target_mac(&home, &cloud, to).await?;
            let key = load_or_create_key(&key_path)?;
            let list = list_services(&key, NetConfig::default(), mac.addr, &RouteMemory::at(&home)).await?;
            if list.is_empty() {
                println!("No services running.");
            }
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs() as i64;
            for s in list {
                println!("{:<28} up {:<10} {}", s.job_id, ago(now - s.started_at as i64).trim_end_matches(" ago"), s.argv.join(" "));
            }
        }
        Command::Attach { id, to } => {
            let (_, mac) = target_mac(&home, &cloud, to).await?;
            let key = load_or_create_key(&key_path)?;
            let cancel = async {
                let _ = tokio::signal::ctrl_c().await;
            };
            let outcome = attach_service(
                &key,
                NetConfig::default(),
                mac.addr,
                &RouteMemory::at(&home),
                &id,
                &mut std::io::stdout(),
                &mut std::io::stderr(),
                cancel,
            )
            .await?;
            report_end(&outcome.job_id, &outcome.exit);
            std::process::exit(outcome.exit.status());
        }
        Command::Forward { to } => {
            let (_, mac) = target_mac(&home, &cloud, to).await?;
            let key = load_or_create_key(&key_path)?;
            eprintln!("bc: forwarding your services' ports; Ctrl+C to stop");
            bc_cli::forward::forward_services(&key, NetConfig::default(), mac.addr, &RouteMemory::at(&home), print_forward, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        }
        Command::Stop { id, to } => {
            let (_, mac) = target_mac(&home, &cloud, to).await?;
            let key = load_or_create_key(&key_path)?;
            stop_service(&key, NetConfig::default(), mac.addr, &RouteMemory::at(&home), &id).await?;
            println!("Stopped {id}.");
        }
        Command::Run { to, spec, relay_only, timings, mut service, mut argv } => {
            let (target, mac) = target_mac(&home, &cloud, to).await?;
            let root = std::env::current_dir()?;
            // With no command and no bc.toml, run what the project looks like it wants.
            if argv.is_empty() && spec.is_none() && !root.join("bc.toml").exists() {
                if let Some(found) = bc_cli::detect::detect(&root) {
                    argv = found.suggestions[0].argv.clone();
                    service |= found.suggestions[0].service;
                    eprintln!("bc: {} project ({}); running: {}", found.kind, found.reason, argv.join(" "));
                }
            }
            let mut spec = load_spec(&root, spec, argv)?;
            spec.job.service |= service;
            let archive = pack(&root, &spec.files.exclude)?;

            let key = load_or_create_key(&key_path)?;
            let cancel = async {
                let _ = tokio::signal::ctrl_c().await;
            };
            let net = NetConfig { relay_only, ..Default::default() };
            let memory = RouteMemory::at(&home);
            // A service's ports reach this computer's localhost while it is watched.
            let forwarding = async {
                if spec.job.service {
                    let _ = bc_cli::forward::forward_services(&key, net.clone(), mac.addr.clone(), &memory, print_forward, std::future::pending()).await;
                }
                std::future::pending::<()>().await
            };
            let (mut stdout, mut stderr) = (std::io::stdout(), std::io::stderr());
            let run = run_on_mac(
                &key,
                net.clone(),
                mac.addr.clone(),
                &spec,
                archive,
                &mut stdout,
                &mut stderr,
                cancel,
                Some(&root),
                &memory,
                |text| eprintln!("bc: {text}"),
            );
            let outcome = tokio::select! {
                outcome = run => outcome?,
                () = forwarding => unreachable!(),
            };

            let line = outcome.timings.to_json(&outcome.job_id, &outcome.exit, outcome.usage.as_ref());
            append_timing(&home, &line);
            if timings {
                eprintln!("bc: {line}");
                if let Some(usage) = &outcome.usage {
                    eprintln!("bc: {}", usage.summary());
                }
            }
            match &outcome.exit {
                Exit::Rejected(reason) => {
                    eprintln!("bc: the Mac rejected the job: {reason}");
                    eprintln!("bc: if you have not paired yet, run `bc pair {target}`");
                }
                Exit::Failed(reason) => eprintln!("bc: job failed on the Mac: {reason}"),
                Exit::TimedOut => eprintln!("bc: job timed out"),
                exit => report_end(&outcome.job_id, exit),
            }
            std::process::exit(outcome.exit.status());
        }
    }
    Ok(())
}

/// The Mac named by `--to`, else the one set with `bc use`.
async fn target_mac(home: &std::path::Path, cloud: &CloudClient, to: Option<String>) -> Result<(String, bc_cli::flows::ResolvedMac)> {
    let target = match to {
        Some(mac) => mac,
        None => default_mac(home).context("no Mac chosen; pass --to <name> or run `bc use <name>`")?,
    };
    let mac = resolve_mac(cloud, &target).await?;
    if let Some(warning) = &mac.offline_warning {
        eprintln!("bc: warning: {warning}");
    }
    Ok((target, mac))
}

fn print_forward(event: bc_cli::forward::ForwardEvent) {
    use bc_cli::forward::ForwardEvent;
    match event {
        ForwardEvent::Opened(f) if f.local == f.remote => eprintln!("bc: http://localhost:{} now reaches the service on the Mac", f.local),
        ForwardEvent::Opened(f) => eprintln!("bc: http://localhost:{} now reaches port {} on the Mac (port {} is taken here)", f.local, f.remote, f.remote),
        ForwardEvent::Closed { job_id } => eprintln!("bc: stopped forwarding {job_id}"),
    }
}

fn report_end(job_id: &str, exit: &Exit) {
    if *exit == Exit::Detached {
        eprintln!("bc: service {job_id} keeps running on the Mac. Watch it with `bc attach {job_id}`, reach its ports with `bc forward`, stop it with `bc stop {job_id}`.");
    }
}

fn signed_in(cloud: &CloudClient) -> Result<&CloudClient> {
    if cloud.token().is_none() {
        bail!("not signed in; run `bc login`");
    }
    Ok(cloud)
}

fn ago(secs: i64) -> String {
    match secs {
        s if s < 120 => format!("{s}s ago"),
        s if s < 7200 => format!("{}m ago", s / 60),
        s if s < 172_800 => format!("{}h ago", s / 3600),
        s => format!("{}d ago", s / 86_400),
    }
}
