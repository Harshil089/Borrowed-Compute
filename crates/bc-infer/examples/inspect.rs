//! Reads a model's header (one file or split) and prints its shape:
//! `cargo run -p bc-infer --example inspect -- <gguf or https address>`.
use bc_infer::{Config, Source};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let src = Source::parse_local(&std::env::args().nth(1).expect("a model"))?;
    let h = src.header().await?;
    let cfg = Config::from_header(&h)?;
    let total: u64 = h.tensors.iter().map(|t| t.range.end - t.range.start).sum();
    let files = h.tensors.iter().map(|t| t.file).max().unwrap_or(0) + 1;
    println!(
        "{} {}: {} layers, width {}, {} heads ({} kv), vocab {}, {} tensors in {files} file(s), {:.1} GB",
        cfg.arch,
        h.get("general.name").and_then(|v| v.as_str()).unwrap_or("?"),
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.vocab, h.tensors.len(), total as f64 / 1e9
    );
    Ok(())
}
