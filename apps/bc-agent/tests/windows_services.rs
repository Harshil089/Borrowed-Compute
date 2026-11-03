//! Services on a Windows lender: the agent finds the ports a job's server listens on and
//! reaches it through a relay inside the job's AppContainer, which the PC's loopback
//! rules would otherwise keep it out of.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::Duration;

use bc_agent::runner::RunnerConfig;
use bc_agent::services::Services;
use bc_proto::wire::JobRequest;

#[tokio::test]
async fn services_are_reached_through_a_relay() {
    let node = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).map(|d| d.join("node.exe")).find(|n| n.is_file());
    let Some(node) = node else { return eprintln!("node is not installed; skipping") };
    bc_agent::sandbox_windows::set_relay_program(env!("CARGO_BIN_EXE_bc-agent").into());

    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let server = format!("require('net').createServer(s => s.on('data', d => s.end('pong ' + d))).listen({port}, '127.0.0.1')");
    let req = JobRequest {
        job_id: "web".into(),
        argv: vec![node.display().to_string(), "-e".into(), server],
        workdir: ".".into(),
        env: BTreeMap::new(),
        timeout_secs: None,
        service: true,
        archive_bytes: 0,
    };
    let mut archive = tar::Builder::new(Vec::new());
    archive.finish().unwrap();
    let root = tempfile::tempdir().unwrap();
    let runner = RunnerConfig { work_root: root.path().into(), default_timeout: Duration::from_secs(30), denied_ports: Vec::new(), guest: false };
    let services = Services::default();
    let owner = iroh::SecretKey::from_bytes(&[1; 32]).public();
    drop(services.start(runner, req, archive.into_inner().unwrap(), owner, None));

    let mut listening = false;
    for _ in 0..100 {
        if services.ports("web").await == vec![port] {
            listening = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(listening, "port {port} never showed up: {:?}", services.list(None));

    let job = services.job("web").unwrap().job;
    let mut relayed = tokio::task::spawn_blocking(move || job.relay(port)).await.unwrap().expect("relay connects");
    relayed.input.write_all(b"ping").unwrap();
    drop(relayed.input);
    let mut reply = String::new();
    relayed.output.read_to_string(&mut reply).unwrap();
    assert_eq!(reply, "pong ping");

    // A port nobody listens on is refused, not hung.
    let job = services.job("web").unwrap().job;
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    assert!(tokio::task::spawn_blocking(move || job.relay(closed)).await.unwrap().is_err());

    services.stop("web", None).unwrap();
}
