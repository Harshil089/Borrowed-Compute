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
    let (exit, ..) = run(&s, &["sh", "-c", "exit 4"]).await;
    assert_eq!(exit, Exit::Code(4));
    let AgentEvent::JobStarted { job_id, client: c, argv } = rx.recv().await.unwrap() else {
        panic!("expected JobStarted");
    };
    assert_eq!(c, client);
    assert_eq!(argv, ["sh", "-c", "exit 4"]);
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
