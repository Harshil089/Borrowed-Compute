//! Manual check that real toolchains work inside the job sandbox: runs a few small
//! projects through the runner exactly as jobs run. Needs network for dependencies.
//! `cargo run -p bc-agent --example sandbox_check`

use std::collections::BTreeMap;
use std::time::Duration;

use bc_agent::runner::{run_job, RunnerConfig};
use bc_proto::wire::{AgentMsg, JobRequest};

fn archive(files: &[(&str, &str)]) -> Vec<u8> {
    let mut b = tar::Builder::new(Vec::new());
    for (path, body) in files {
        let mut h = tar::Header::new_gnu();
        h.set_size(body.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        b.append_data(&mut h, path, body.as_bytes()).unwrap();
    }
    b.into_inner().unwrap()
}

async fn job(name: &str, argv: &[&str], files: &[(&str, &str)]) {
    let root = tempfile::tempdir().unwrap();
    let config = RunnerConfig { work_root: root.path().into(), default_timeout: Duration::from_secs(600) };
    let req = JobRequest {
        job_id: name.into(),
        argv: argv.iter().map(|s| s.to_string()).collect(),
        workdir: ".".into(),
        env: BTreeMap::new(),
        timeout_secs: None,
        archive_bytes: 0,
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(1024);
    let tar = archive(files);
    let runner = tokio::spawn(async move { run_job(&config, &req, tar, tx).await });
    let mut out = Vec::new();
    let mut end = String::new();
    while let Some(m) = rx.recv().await {
        match m {
            AgentMsg::Output { data, .. } => out.extend(data),
            AgentMsg::Exited { code, .. } => end = format!("exit {code:?}"),
            AgentMsg::Failed { reason } => end = format!("failed: {reason}"),
            _ => {}
        }
    }
    runner.await.unwrap();
    let text = String::from_utf8_lossy(&out);
    let tail: Vec<&str> = text.lines().rev().take(if name == "node-ts" { 30 } else { 3 }).collect();
    println!("== {name}: {end}\n   {}", tail.into_iter().rev().collect::<Vec<_>>().join("\n   "));
}

#[tokio::main]
async fn main() {
    if std::env::args().nth(1).as_deref() == Some("node") {
        job("node-ts", &["sh", "-c", "npm install --silent is-number@7 && npx --yes tsx main.ts"], &[
            ("package.json", "{\"name\":\"sb\",\"version\":\"1.0.0\"}"),
            ("main.ts", "import isNumber from 'is-number'; const n: number = 5; console.log('ts ok', isNumber(n));"),
        ]).await;
        return;
    }
    job("rust", &["cargo", "run", "-q"], &[
        ("Cargo.toml", "[package]\nname=\"sb\"\nversion=\"0.1.0\"\nedition=\"2021\"\n[dependencies]\nitoa=\"1\"\n"),
        ("src/main.rs", "fn main(){ println!(\"rust ok {}\", itoa::Buffer::new().format(7)); }"),
    ]).await;
    job("python-venv", &["sh", "-c", "python3 -m venv .venv && .venv/bin/pip install -q six && .venv/bin/python main.py"], &[
        ("main.py", "import six; print('python ok', six.__version__)"),
    ]).await;
    job("node-ts", &["sh", "-c", "npm install --silent is-number@7 && npx --yes tsx main.ts"], &[
        ("package.json", "{\"name\":\"sb\",\"version\":\"1.0.0\"}"),
        ("main.ts", "import isNumber from 'is-number'; const n: number = 5; console.log('ts ok', isNumber(n));"),
    ]).await;
    job("brew-tools", &["sh", "-c", "python3 -c 'print(\"brew python ok\")' && go version"], &[]).await;
    job("multiprocessing", &["python3", "mp.py"], &[
        ("mp.py", "import multiprocessing as m\ndef sq(x): return x*x\nif __name__=='__main__':\n    with m.Pool(4) as p: print('pool ok', sum(p.map(sq, range(100))))\n    q=m.Queue(); l=m.Lock(); q.put(1); print('queue ok', q.get())\n"),
    ]).await;
    job("blocked", &["sh", "-c", "cat ~/.zshrc >/dev/null 2>&1 && echo LEAK || echo 'home blocked ok'; ls ~/Documents >/dev/null 2>&1 && echo LEAK || echo 'documents blocked ok'"], &[]).await;
}
