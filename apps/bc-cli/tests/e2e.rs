//! End to end: the `bc` client runs a job on an in-process agent, through a local relay
//! with direct paths disabled.
//!
//! Unix only: the agent runs jobs with `sh` and Unix process groups.
#![cfg(unix)]

use std::time::Duration;

use bc_agent::paired::PairedDevices;
use bc_agent::runner::RunnerConfig;
use bc_agent::server::{serve_connection, AgentConfig};
use bc_cli::pack::pack;
use bc_cli::run::{run_remote, Exit};
use bc_net::{bind, NetConfig};
use bc_proto::JobSpec;
use iroh::{Endpoint, SecretKey};

struct Setup {
    agent: Endpoint,
    client: Endpoint,
    paired: PairedDevices,
    work_root: tempfile::TempDir,
    _state: tempfile::TempDir,
    _relay: iroh_relay::server::Server,
}

async fn setup() -> Setup {
    setup_with(None).await
}

async fn setup_with(events: Option<bc_agent::events::EventSender>) -> Setup {
    let (relay_map, _url, relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let config = NetConfig {
        relays: Some(relay_map),
        relay_only: true,
        insecure_relay_tls: true,
    };
    let agent = bind(SecretKey::generate(), config.clone()).await.unwrap();
    let client = bind(SecretKey::generate(), config).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), agent.online())
        .await
        .unwrap();

    let state = tempfile::tempdir().unwrap();
    let paired = PairedDevices::new(state.path().join("paired"));
    let work_root = tempfile::tempdir().unwrap();
    let agent_config = AgentConfig::new(
        RunnerConfig {
            work_root: work_root.path().to_path_buf(),
            default_timeout: Duration::from_secs(30),
        },
        paired.clone(),
    );
    let mut agent_config = agent_config;
    agent_config.events = events;
    let accept = agent.clone();
    tokio::spawn(async move {
        while let Some(incoming) = accept.accept().await {
            let config = agent_config.clone();
            tokio::spawn(async move {
                if let Ok(conn) = incoming.await {
                    let _ = serve_connection(&config, conn).await;
                }
            });
        }
    });

    Setup { agent, client, paired, work_root, _state: state, _relay: relay }
}

fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/input.txt"), "from windows\n").unwrap();
    std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
    std::fs::create_dir_all(dir.path().join("target")).unwrap();
    std::fs::write(dir.path().join("target/skip.txt"), "must not upload\n").unwrap();
    dir
}

async fn run(s: &Setup, argv: &[&str]) -> (Exit, String, String, bool) {
    let proj = project();
    let spec = JobSpec::from_argv(argv.iter().map(|a| a.to_string()).collect()).unwrap();
    let archive = pack(proj.path(), &spec.files.exclude).unwrap();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let outcome = run_remote(
        &s.client,
        s.agent.addr(),
        &spec,
        archive,
        &mut out,
        &mut err,
        std::future::pending(),
    )
    .await
    .unwrap();
    (
        outcome.exit,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
        outcome.timings.via_relay,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn paired_device_runs_job_with_live_output_and_exit_code() {
    let s = setup().await;
    s.paired.add(s.client.id()).unwrap();

    let (exit, out, err, via_relay) = run(
        &s,
        &["sh", "-c", "cat src/input.txt; test -e target || echo no-target; echo warn >&2; exit 7"],
    )
    .await;

    assert_eq!(exit, Exit::Code(7));
    assert_eq!(out, "from windows\nno-target\n");
    assert_eq!(err, "warn\n");
    assert!(via_relay);
    // Workspaces are deleted after each job.
    assert_eq!(std::fs::read_dir(s.work_root.path()).unwrap().count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn unpaired_device_is_rejected() {
    let s = setup().await;
    let (exit, out, _, _) = run(&s, &["sh", "-c", "echo should-not-run"]).await;
    assert!(matches!(exit, Exit::Rejected(ref r) if r.contains("not paired")), "{exit:?}");
    assert!(out.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_reports_job_events() {
    use bc_agent::events::{AgentEvent, JobEnd};

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let s = setup_with(Some(tx)).await;

    let (exit, ..) = run(&s, &["sh", "-c", "echo nope"]).await;
    assert!(matches!(exit, Exit::Rejected(_)));
    let client = s.client.id().to_string();
    match rx.recv().await.unwrap() {
        AgentEvent::JobRejected { client: c, .. } => assert_eq!(c, client),
        other => panic!("expected JobRejected, got {other:?}"),
    }

    s.paired.add(s.client.id()).unwrap();
    let (exit, ..) = run(&s, &["sh", "-c", "echo hi; exit 4"]).await;
    assert_eq!(exit, Exit::Code(4));
    let AgentEvent::JobStarted { job_id, client: c, argv } = rx.recv().await.unwrap() else {
        panic!("expected JobStarted");
    };
    assert_eq!(c, client);
    assert_eq!(argv, ["sh", "-c", "echo hi; exit 4"]);
    match rx.recv().await.unwrap() {
        AgentEvent::JobOutput { job_id: id, data } => assert_eq!((id, data), (job_id.clone(), b"hi\n".to_vec())),
        other => panic!("expected JobOutput, got {other:?}"),
    }
    // The finish event is sent when the connection handler returns, just after the client
    // sees the exit code.
    let finished = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
    assert!(
        matches!(
            &finished,
            AgentEvent::JobFinished {
                job_id: id,
                end: JobEnd::Exited { code: Some(4), timed_out: false, usage: Some(_) },
            } if *id == job_id
        ),
        "{finished:?}"
    );
}

/// A run that fails because a tool is missing gets fixed and run once more: here the
/// first attempt reports `vite: command not found`, auto-fix installs dependencies
/// (an empty package.json installs offline), and the second attempt succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn auto_fix_installs_dependencies_and_runs_again() {
    if std::process::Command::new("npm").arg("--version").output().is_err() {
        eprintln!("npm not installed; skipping");
        return;
    }
    let (relay_map, _url, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let net = NetConfig { relays: Some(relay_map.clone()), relay_only: true, insecure_relay_tls: true };
    let agent = bind(SecretKey::generate(), net.clone()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), agent.online()).await.unwrap();
    let state = tempfile::tempdir().unwrap();
    let paired = PairedDevices::new(state.path().join("paired"));
    let client_key = SecretKey::generate();
    paired.add(client_key.public()).unwrap();
    let work_root = tempfile::tempdir().unwrap();
    let config = AgentConfig::new(
        RunnerConfig { work_root: work_root.path().to_path_buf(), default_timeout: Duration::from_secs(120) },
        paired,
    );
    let accept = agent.clone();
    tokio::spawn(async move {
        while let Some(incoming) = accept.accept().await {
            let config = config.clone();
            tokio::spawn(async move {
                if let Ok(conn) = incoming.await {
                    let _ = serve_connection(&config, conn).await;
                }
            });
        }
    });

    let proj = tempfile::tempdir().unwrap();
    std::fs::write(proj.path().join("package.json"), r#"{"name":"autofix","version":"1.0.0"}"#).unwrap();
    // Stands in for a dev tool that only exists after an install.
    let argv = ["sh", "-c", "test -f package-lock.json && echo built || { echo 'sh: vite: command not found' >&2; exit 127; }"];
    let spec = JobSpec::from_argv(argv.iter().map(|a| a.to_string()).collect()).unwrap();
    let archive = pack(proj.path(), &spec.files.exclude).unwrap();
    let notices = std::sync::Mutex::new(Vec::new());
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let outcome = bc_cli::run::run_on_mac(
        &client_key,
        net,
        agent.addr(),
        &spec,
        archive,
        &mut out,
        &mut err,
        std::future::pending(),
        Some(proj.path()),
        &bc_cli::route::RouteMemory::none(),
        |n: &str| notices.lock().unwrap().push(n.to_string()),
    )
    .await
    .unwrap();
    let notices = notices.into_inner().unwrap();
    assert_eq!(outcome.exit, Exit::Code(0), "{notices:?} {}", String::from_utf8_lossy(&err));
    assert!(String::from_utf8_lossy(&out).contains("built"));
    assert!(notices.iter().any(|n| n.contains("Installing dependencies") && n.contains("Running again")), "{notices:?}");
}

#[tokio::test]
async fn service_keeps_running_after_detach_until_stopped() {
    let (relay_map, _url, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let net = NetConfig { relays: Some(relay_map.clone()), relay_only: true, insecure_relay_tls: true };
    let agent = bind(SecretKey::generate(), net.clone()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), agent.online()).await.unwrap();
    let state = tempfile::tempdir().unwrap();
    let paired = PairedDevices::new(state.path().join("paired"));
    let client_key = SecretKey::generate();
    paired.add(client_key.public()).unwrap();
    let work_root = tempfile::tempdir().unwrap();
    // A 1 s default timeout, which a service must ignore.
    let config = AgentConfig::new(
        RunnerConfig { work_root: work_root.path().to_path_buf(), default_timeout: Duration::from_secs(1) },
        paired,
    );
    let accept = agent.clone();
    tokio::spawn(async move {
        while let Some(incoming) = accept.accept().await {
            let config = config.clone();
            tokio::spawn(async move {
                if let Ok(conn) = incoming.await {
                    let _ = serve_connection(&config, conn).await;
                }
            });
        }
    });
    let memory = bc_cli::route::RouteMemory::none();

    let proj = project();
    let mut spec = JobSpec::from_argv(
        ["sh", "-c", "echo up; while true; do sleep 0.1; done"].iter().map(|a| a.to_string()).collect(),
    )
    .unwrap();
    spec.job.service = true;
    let archive = pack(proj.path(), &spec.files.exclude).unwrap();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let outcome = bc_cli::run::run_on_mac(
        &client_key,
        net.clone(),
        agent.addr(),
        &spec,
        archive,
        &mut out,
        &mut err,
        tokio::time::sleep(Duration::from_secs(2)),
        None,
        &memory,
        |_: &str| {},
    )
    .await
    .unwrap();
    assert_eq!(outcome.exit, Exit::Detached, "{}", String::from_utf8_lossy(&err));
    assert_eq!(String::from_utf8_lossy(&out), "up\n");

    // Still running after the client left and past the timeout.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let list = bc_cli::run::list_services(&client_key, net.clone(), agent.addr(), &memory).await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].job_id, outcome.job_id);
    // Another paired device sees none of it and cannot stop it.
    let other = SecretKey::generate();
    let paired = PairedDevices::new(state.path().join("paired"));
    paired.add(other.public()).unwrap();
    assert!(bc_cli::run::list_services(&other, net.clone(), agent.addr(), &memory).await.unwrap().is_empty());
    assert!(bc_cli::run::stop_service(&other, net.clone(), agent.addr(), &memory, &outcome.job_id).await.is_err());

    // Attaching replays what was missed.
    let mut out = Vec::new();
    let attached = bc_cli::run::attach_service(
        &client_key,
        net.clone(),
        agent.addr(),
        &memory,
        &outcome.job_id,
        &mut out,
        &mut Vec::new(),
        tokio::time::sleep(Duration::from_secs(1)),
    )
    .await
    .unwrap();
    assert_eq!(attached.exit, Exit::Detached);
    assert_eq!(String::from_utf8_lossy(&out), "up\n");

    bc_cli::run::stop_service(&client_key, net.clone(), agent.addr(), &memory, &outcome.job_id).await.unwrap();
    for _ in 0..100 {
        let gone = bc_cli::run::list_services(&client_key, net.clone(), agent.addr(), &memory).await.unwrap().is_empty();
        if gone && std::fs::read_dir(work_root.path()).unwrap().next().is_none() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("service or its workspace is still there after stop");
}

#[tokio::test]
async fn forwards_a_service_port_to_localhost() {
    if std::process::Command::new("python3").arg("--version").output().is_err() {
        eprintln!("python3 not installed; skipping");
        return;
    }
    let (relay_map, _url, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let net = NetConfig { relays: Some(relay_map.clone()), relay_only: true, insecure_relay_tls: true };
    let agent = bind(SecretKey::generate(), net.clone()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), agent.online()).await.unwrap();
    let state = tempfile::tempdir().unwrap();
    let paired = PairedDevices::new(state.path().join("paired"));
    let client_key = SecretKey::generate();
    paired.add(client_key.public()).unwrap();
    let work_root = tempfile::tempdir().unwrap();
    let config = AgentConfig::new(
        RunnerConfig { work_root: work_root.path().to_path_buf(), default_timeout: Duration::from_secs(30) },
        paired,
    );
    let accept = agent.clone();
    tokio::spawn(async move {
        while let Some(incoming) = accept.accept().await {
            let config = config.clone();
            tokio::spawn(async move {
                if let Ok(conn) = incoming.await {
                    let _ = serve_connection(&config, conn).await;
                }
            });
        }
    });
    let memory = bc_cli::route::RouteMemory::none();

    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let proj = project();
    std::fs::write(proj.path().join("index.html"), "<p>served from the Mac</p>").unwrap();
    let mut spec = JobSpec::from_argv(
        ["python3", "-m", "http.server", &port.to_string(), "-b", "127.0.0.1"].iter().map(|a| a.to_string()).collect(),
    )
    .unwrap();
    spec.job.service = true;
    let archive = pack(proj.path(), &spec.files.exclude).unwrap();
    let outcome = bc_cli::run::run_on_mac(
        &client_key,
        net.clone(),
        agent.addr(),
        &spec,
        archive,
        &mut Vec::new(),
        &mut Vec::new(),
        tokio::time::sleep(Duration::from_secs(1)),
        None,
        &memory,
        |_: &str| {},
    )
    .await
    .unwrap();
    assert_eq!(outcome.exit, Exit::Detached);

    let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let forwarding = {
        let (key, net, mac, memory) = (client_key.clone(), net.clone(), agent.addr(), memory.clone());
        tokio::spawn(async move {
            // The service list after each poll is not what this test checks.
            let on_event = move |e| {
                if !matches!(e, bc_cli::forward::ForwardEvent::Services(_)) {
                    let _ = tx.send(e);
                }
            };
            bc_cli::forward::forward_services(&key, net, mac, &memory, on_event, async {
                let _ = stop_rx.await;
            })
            .await
        })
    };
    let opened = tokio::time::timeout(Duration::from_secs(20), events.recv()).await.unwrap().unwrap();
    let bc_cli::forward::ForwardEvent::Opened(f) = opened else { panic!("{opened:?}") };
    assert_eq!(f.remote, port);
    // Agent and client share this machine, so the same port is taken and another is used.
    assert_ne!(f.local, port);

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut tcp = tokio::net::TcpStream::connect(("127.0.0.1", f.local)).await.unwrap();
    tcp.write_all(b"GET /index.html HTTP/1.0\r\nHost: localhost\r\n\r\n").await.unwrap();
    let mut body = String::new();
    tcp.read_to_string(&mut body).await.unwrap();
    assert!(body.starts_with("HTTP/1.0 200"), "{body}");
    assert!(body.contains("served from the Mac"), "{body}");

    bc_cli::run::stop_service(&client_key, net.clone(), agent.addr(), &memory, &outcome.job_id).await.unwrap();
    let closed = tokio::time::timeout(Duration::from_secs(20), events.recv()).await.unwrap().unwrap();
    assert_eq!(closed, bc_cli::forward::ForwardEvent::Closed { job_id: outcome.job_id.clone() });
    let _ = stop_tx.send(());
    forwarding.await.unwrap().unwrap();
}
