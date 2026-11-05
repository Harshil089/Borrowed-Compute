use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_auth::{cloud_url_for, default_device_name, save_session, CloudClient};
use bc_cli::flows::{
    append_timing, client_home, default_mac, key_path, load_spec, mac_net, resolve_mac,
    set_default_mac, wait_for_pairing,
};
use bc_cli::pack::pack;
use bc_cli::route::RouteMemory;
use bc_cli::run::{attach_service, list_services, run_on_mac, stop_service, Exit};
use bc_net::load_or_create_key;
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
    /// Show the cloud address, or set it (your Mac's code from Borrowed Compute Agent, or a web address).
    /// An empty address returns to the default.
    Cloud { url: Option<String> },
    /// Print this device's id.
    Id,
    /// Set the Mac that `bc run` uses by default (name or device id).
    Use { mac: String },
    /// Join a Mac someone shared with you, using their invite. Needs no sign-in.
    Join {
        /// The invite, starting with bc-invite:
        invite: String,
        /// Name the Mac's owner sees for this device. Defaults to the computer name.
        #[arg(long)]
        name: Option<String>,
    },
    /// List the Macs shared with this device.
    Shared,
    /// Forget a shared Mac on this device.
    Leave { mac: String },
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
    /// Use several paired devices at once: measure them, and plan a model across them.
    Mesh {
        #[command(subcommand)]
        action: MeshAction,
    },
}

#[derive(Subcommand)]
enum MeshAction {
    /// Measure every paired device at once: path, round trip, upload rate, memory.
    Probe,
    /// Show where each part of a model would run on your paired devices, and how fast.
    Plan {
        /// A model name; run without one to list them.
        model: Option<String>,
        /// Serve many requests at once instead of one conversation as fast as possible.
        #[arg(long)]
        throughput: bool,
        /// Tokens accepted per pipeline pass with speculative decoding (1 = off).
        #[arg(long, default_value_t = 1.0)]
        speculate: f64,
    },
    /// Run a model across your paired devices and answer one prompt.
    Run {
        /// A model name from `bc mesh plan`, or the https address of a .gguf file.
        model: String,
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// A small model of the same family on this computer for speculative decoding:
        /// a name, an address or a file. `none` turns it off.
        #[arg(long)]
        draft: Option<String>,
        #[arg(long)]
        throughput: bool,
    },
    /// Run a model across your paired devices behind an OpenAI-compatible endpoint on
    /// this computer, until Ctrl+C.
    Serve {
        model: String,
        #[arg(long, default_value_t = 8484)]
        port: u16,
        #[arg(long)]
        draft: Option<String>,
        /// Plan for many requests at once (replicas, balanced stages).
        #[arg(long)]
        throughput: bool,
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
        Command::Join { invite, name } => {
            let key = load_or_create_key(&key_path)?;
            let name = name.unwrap_or_else(default_device_name);
            println!("Asking the Mac to let this device in…");
            let mac = bc_cli::shared::join(&home, key, mac_net(&cloud, false).await, &invite, &name, |code| {
                println!("Code: {code}\nRead this code to the Mac's owner; they type it into Borrowed Compute Agent.");
                print!("Waiting for approval");
                let _ = std::io::stdout().flush();
            })
            .await?;
            if default_mac(&home).is_none() {
                set_default_mac(&home, &mac.name)?;
            }
            println!("\nYou can now run jobs on {:?}: bc run --to {:?} -- <command>", mac.name, mac.name);
        }
        Command::Shared => {
            let macs = bc_cli::shared::list(&home);
            if macs.is_empty() {
                println!("No shared Macs. Join one with `bc join <invite>`.");
            }
            for m in macs {
                println!("{:<24} {}", m.name, m.id);
            }
        }
        Command::Leave { mac } => {
            if !bc_cli::shared::remove(&home, &mac)? {
                bail!("no shared Mac called {mac:?}; see `bc shared`");
            }
            println!("Forgot {mac:?} on this device.");
        }
        Command::Use { mac } => {
            set_default_mac(&home, &mac)?;
            println!("bc run will use {mac}");
        }
        Command::Services { to } => {
            let (_, mac) = target_mac(&home, &cloud, to).await?;
            let key = load_or_create_key(&key_path)?;
            let list = list_services(&key, mac_net(&cloud, false).await, mac.addr, &RouteMemory::at(&home)).await?;
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
                mac_net(&cloud, false).await,
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
            bc_cli::forward::forward_services(&key, mac_net(&cloud, false).await, mac.addr, &RouteMemory::at(&home), print_forward, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        }
        Command::Stop { id, to } => {
            let (_, mac) = target_mac(&home, &cloud, to).await?;
            let key = load_or_create_key(&key_path)?;
            stop_service(&key, mac_net(&cloud, false).await, mac.addr, &RouteMemory::at(&home), &id).await?;
            println!("Stopped {id}.");
        }
        Command::Run { to, spec, relay_only, timings, mut service, mut argv } => {
            let (target, mac) = target_mac(&home, &cloud, to).await?;
            let root = std::env::current_dir()?;
            // With no command and no bc.toml, run what the project looks like it wants.
            if argv.is_empty() && spec.is_none() && !root.join("bc.toml").exists() {
                if let Some(found) = bc_cli::detect::detect(&root, bc_cli::detect::Os::of(mac.kind)) {
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
            let net = mac_net(&cloud, relay_only).await;
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
                bc_cli::detect::Os::of(mac.kind),
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
        Command::Mesh { action } => mesh(&cloud, &key_path, action).await?,
    }
    Ok(())
}

async fn mesh(cloud: &CloudClient, key_path: &std::path::Path, action: MeshAction) -> Result<()> {
    use bc_mesh::plan::Goal;
    if let MeshAction::Run { .. } | MeshAction::Serve { .. } = action {
        return mesh_run(cloud, key_path, action).await;
    }
    let model = match &action {
        MeshAction::Plan { model: None, .. } => {
            for m in bc_mesh::plan::catalog() {
                let gb = (m.layer_bytes * m.layers as u64 + m.extra_bytes) as f64 / (1u64 << 30) as f64;
                println!("{:<16} {:>4} layers  {gb:>6.1} GB (4-bit)", m.name, m.layers);
            }
            return Ok(());
        }
        MeshAction::Plan { model: Some(name), .. } => {
            Some(bc_cli::mesh::find_model(name).with_context(|| format!("unknown model {name:?}; run `bc mesh plan` for the list"))?)
        }
        MeshAction::Probe => None,
        MeshAction::Run { .. } | MeshAction::Serve { .. } => unreachable!(),
    };
    let key = load_or_create_key(key_path)?;
    let me = key.public().to_string();
    let targets = bc_cli::mesh::paired_devices(signed_in(cloud)?, &me).await?;
    if targets.is_empty() {
        bail!("this device is not paired with any device yet; pair with `bc pair <device>`");
    }
    let endpoint = bc_net::bind(key, mac_net(cloud, false).await).await?;
    println!("Measuring {} devices at once…", targets.len());
    let devices = bc_cli::mesh::probe_all(&endpoint, targets).await;
    bc_net::release(endpoint).await;
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    for d in &devices {
        match &d.report {
            Ok(r) => println!(
                "{:<20} {:<6} {:>7.1} ms  {:>8.1} Mbit/s up  {:>5.1}/{:<5.1} GB free  {:>5.1} GB/s memory  {}",
                d.name,
                if r.via_relay { "relay" } else { "direct" },
                r.rtt_ms,
                r.up_bytes_s * 8.0 / 1e6,
                gb(r.device.mem_available),
                gb(r.device.mem_total),
                r.device.mem_bw / 1e9,
                r.device.os,
            ),
            Err(e) => println!("{:<20} unreachable: {e}", d.name),
        }
    }
    let (Some(model), MeshAction::Plan { throughput, speculate, .. }) = (model, action) else { return Ok(()) };
    let goal = if throughput { Goal::Throughput } else { Goal::Latency };
    let (plan, usable) = bc_cli::mesh::plan_for(&model, &devices, goal, speculate);
    if plan.pipelines.is_empty() {
        bail!("{} does not fit in the free memory of these devices together", model.name);
    }
    println!();
    for (i, p) in plan.pipelines.iter().enumerate() {
        println!("Pipeline {}:", i + 1);
        for s in &p.stages {
            println!("  {:<20} layers {:>3}..{:<3}", usable[s.device].name, s.layers.start, s.layers.end);
        }
        println!(
            "  {:.0} ms a token for one conversation ({:.0} ms of it network), {:.1} tokens/s with every stage busy, {:.0} ms network per 1000 prompt tokens",
            p.token_ms, p.network_ms, p.tokens_per_s, p.prefill_net_ms_per_1k
        );
    }
    if plan.pipelines.len() > 1 {
        println!("{:.1} tokens/s across {} replicas", plan.tokens_per_s(), plan.pipelines.len());
    }
    for &d in &plan.unused {
        println!("Not needed: {}", usable[d].name);
    }
    Ok(())
}

/// The Mac named by `--to`, else the one set with `bc use`.
async fn target_mac(home: &std::path::Path, cloud: &CloudClient, to: Option<String>) -> Result<(String, bc_cli::flows::ResolvedMac)> {
    let target = match to {
        Some(mac) => mac,
        None => default_mac(home).context("no Mac chosen; pass --to <name> or run `bc use <name>`")?,
    };
    let mac = resolve_mac(cloud, home, &target).await?;
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
        ForwardEvent::Services(_) => {}
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

async fn mesh_run(cloud: &CloudClient, key_path: &std::path::Path, action: MeshAction) -> Result<()> {
    use bc_mesh::plan::Goal;
    let key = load_or_create_key(key_path)?;
    let me = key.public().to_string();
    let signed = signed_in(cloud)?.clone();
    let targets = bc_cli::mesh::paired_devices(&signed, &me).await?;
    if targets.is_empty() {
        bail!("this device is not paired with any device yet; pair with `bc pair <device>`");
    }
    let endpoint = bc_net::bind(key, mac_net(cloud, false).await).await?;
    let (model, draft, throughput) = match &action {
        MeshAction::Run { model, draft, throughput, .. } | MeshAction::Serve { model, draft, throughput, .. } => (model.clone(), draft.clone(), *throughput),
        _ => unreachable!(),
    };
    let goal = if throughput { Goal::Throughput } else { Goal::Latency };
    let session = std::sync::Arc::new(
        bc_cli::mesh::Session::start(endpoint.clone(), targets, &model, goal, draft.as_deref(), |s| eprintln!("bc: {s}")).await?,
    );
    for (i, p) in session.layout().await.iter().enumerate() {
        let stages: Vec<String> = p.iter().map(|(n, l)| format!("{n} {}..{}", l.start, l.end)).collect();
        eprintln!("bc: pipeline {}: {}", i + 1, stages.join(" → "));
    }
    match action {
        MeshAction::Run { prompt, max_tokens, temperature, .. } => {
            let llm = session.llm.clone();
            let ids = llm.tokenizer.encode(&llm.tokenizer.chat_prompt(&[("user".into(), prompt)]));
            let sampling = bc_infer::Sampling { temperature, top_p: 0.95, seed: rand_seed() };
            let (_, stats) = llm
                .generate(&ids, max_tokens, sampling, |t| {
                    print!("{t}");
                    let _ = std::io::stdout().flush();
                })
                .await?;
            println!();
            let decode_s = (stats.total_ms - stats.first_token_ms).max(1.0) / 1000.0;
            eprintln!(
                "bc: {} prompt tokens, first token after {:.0} ms; {} tokens in {} passes, {:.1} tokens/s",
                stats.prompt_tokens, stats.first_token_ms, stats.generated, stats.passes, stats.generated.saturating_sub(1) as f64 / decode_s
            );
        }
        MeshAction::Serve { port, .. } => {
            let (cloud_c, me_c) = (signed.clone(), me.clone());
            let served = std::sync::Arc::new(bc_cli::serve::Served {
                session: session.clone(),
                targets: std::sync::Arc::new(move || {
                    let (c, m) = (cloud_c.clone(), me_c.clone());
                    Box::pin(async move { bc_cli::mesh::paired_devices(&c, &m).await })
                }),
            });
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.with_context(|| format!("port {port} is taken"))?;
            eprintln!("bc: serving {} at http://127.0.0.1:{port}/v1 (OpenAI-compatible). Ctrl+C stops it.", session.name);
            let balancer = session.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    match balancer.rebalance().await {
                        Ok(true) => eprintln!("bc: moved layers to match how busy each device is"),
                        Ok(false) => {}
                        Err(e) => eprintln!("bc: rebalancing failed: {e:#}"),
                    }
                }
            });
            axum::serve(listener, bc_cli::serve::router(served))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
        }
        _ => unreachable!(),
    }
    bc_net::release(endpoint).await;
    Ok(())
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(7)
}
