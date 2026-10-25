use anyhow::{bail, Context, Result};
use bc_agent::paired::PairedDevices;
use bc_agent::service::{Agent, AgentPaths};
use bc_auth::{cloud_url_for, default_device_name, save_session, CloudClient};
use bc_net::{home_dir, load_or_create_key, EndpointId};
use bc_proto::cloud::{DeviceKind, PairingStatus};
use clap::{Parser, Subcommand};

/// Borrowed Compute agent for the Mac.
#[derive(Parser)]
#[command(name = "bc-agent", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sign in with GitHub and register this Mac.
    Login {
        /// Name other devices use for this Mac. Defaults to the host name.
        #[arg(long)]
        name: Option<String>,
    },
    /// Sign out on this Mac.
    Logout,
    /// Accept and run jobs from paired devices.
    Serve {
        /// Disable direct paths and send all traffic through the relay.
        #[arg(long)]
        relay_only: bool,
    },
    /// Print this Mac's device id.
    Id,
    /// List pairing requests and paired devices.
    Pairings,
    /// Approve a pairing request using the code shown on the other device.
    Approve { code: String },
    /// Remove a paired device, by name or id.
    Unpair { device: String },
    /// Development without the cloud: allow a device id directly.
    Allow { device: EndpointId },
    /// Development without the cloud: remove a device id.
    Revoke { device: EndpointId },
    /// Show the local paired list the agent checks.
    Paired,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let home = home_dir()?.join("agent");
    let paired = PairedDevices::new(home.join("paired"));
    let key_path = home.join("device.key");
    let mut cloud = CloudClient::from_home(cloud_url_for(&home), &home);

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
            let device = cloud.register(&key, &name, DeviceKind::Mac).await?;
            println!("Signed in as {login}. This Mac is registered as {:?}.", device.name);
        }
        Command::Logout => {
            if cloud.token().is_some() {
                let _ = cloud.logout().await;
            }
            bc_auth::clear_session(&home);
            println!("Signed out.");
        }
        Command::Id => {
            let key = load_or_create_key(&key_path)?;
            println!("{}", key.public());
        }
        Command::Pairings => {
            let me = load_or_create_key(&key_path)?.public().to_string();
            let list = signed_in(&cloud)?.pairings(&me).await?;
            if list.is_empty() {
                println!("No pairing requests or paired devices.");
            }
            for p in list {
                match p.status {
                    PairingStatus::Pending => println!(
                        "request from {:<20} code {}  (approve with: bc-agent approve {})",
                        p.client_name,
                        p.code.as_deref().unwrap_or("?"),
                        p.code.as_deref().unwrap_or("<code>")
                    ),
                    PairingStatus::Approved => println!("paired      {:<20} {}", p.client_name, p.client_id),
                }
            }
        }
        Command::Approve { code } => {
            let me = load_or_create_key(&key_path)?.public().to_string();
            let cloud = signed_in(&cloud)?;
            let code = code.trim().to_string();
            let request = cloud
                .pairings(&me)
                .await?
                .into_iter()
                .find(|p| p.status == PairingStatus::Pending && p.code.as_deref() == Some(&code))
                .context("no pending request with that code; check `bc-agent pairings`")?;
            cloud.approve(request.id, &code).await?;
            println!("Paired {:?}. It can send jobs within a few seconds.", request.client_name);
        }
        Command::Unpair { device } => {
            let me = load_or_create_key(&key_path)?.public().to_string();
            let cloud = signed_in(&cloud)?;
            let pairing = cloud
                .pairings(&me)
                .await?
                .into_iter()
                .find(|p| p.mac_id == me && (p.client_name == device || p.client_id == device))
                .with_context(|| format!("{device:?} is not paired with this Mac"))?;
            cloud.unpair(pairing.id).await?;
            println!("Unpaired {:?}.", pairing.client_name);
        }
        Command::Allow { device } => {
            paired.add(device)?;
            println!("allowed {device}");
        }
        Command::Revoke { device } => {
            let removed = paired.remove(&device)?;
            println!("{} {device}", if removed { "revoked" } else { "was not paired:" });
        }
        Command::Paired => {
            for id in paired.load()? {
                println!("{id}");
            }
        }
        Command::Serve { relay_only } => {
            let agent = Agent::start(&AgentPaths::new(home.clone()), relay_only, None).await?;
            eprintln!("bc-agent ready, device id {}", agent.endpoint.id());
            if cloud.token().is_some() {
                eprintln!("signed in: syncing presence and pairings with {}", cloud_url_for(&home));
                agent.spawn_cloud_sync(cloud);
            } else {
                eprintln!("not signed in: using the local paired list only");
            }
            agent.accept_loop().await;
        }
    }
    Ok(())
}

fn signed_in(cloud: &CloudClient) -> Result<&CloudClient> {
    if cloud.token().is_none() {
        bail!("not signed in; run `bc-agent login`");
    }
    Ok(cloud)
}
