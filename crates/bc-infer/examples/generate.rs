//! Generates text from a GGUF model on this computer, whole and split in two, and checks
//! that both give the same tokens: `cargo run -p bc-infer --release --example generate --
//! <model.gguf or https URL> "<prompt>" [tokens] [split layer]`.

use std::collections::HashMap;
use std::time::Instant;

use bc_infer::{model, sample, Config, Part, Source, Tokenizer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let source = Source::parse_local(&args[1])?;
    let prompt = args.get(2).cloned().unwrap_or_else(|| "Once upon a time".into());
    let n: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(48);
    let header = source.header().await?;
    let cfg = Config::from_header(&header)?;
    let split: u32 = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(cfg.n_layer as u32 / 2);
    let tok = Tokenizer::from_header(&header)?;
    let all = 0..cfg.n_layer as u32;
    let tensors = source.tensors(&model::needed(&header, &all, true, true), |_, _| {}).await?;
    let whole = Part::new(cfg.clone(), all, true, true, &tensors)?;
    let a = Part::new(cfg.clone(), 0..split, true, false, &pick(&tensors, &header, 0..split, true, false))?;
    let b = Part::new(cfg.clone(), split..cfg.n_layer as u32, false, true, &pick(&tensors, &header, split..cfg.n_layer as u32, false, true))?;
    let text = if tok.template == bc_infer::tokenizer::Template::Plain { prompt.clone() } else { tok.chat_prompt(&[("user".into(), prompt.clone())]) };
    let ids = tok.encode(&text);
    println!("{} ({:?}, {} layers, split at {split}), prompt {} tokens", source.label(), cfg.arch, cfg.n_layer, ids.len());

    let (mut kv, mut ka, mut kb) = (whole.new_kv(), a.new_kv(), b.new_kv());
    let (mut out_whole, mut out_split) = (Vec::new(), Vec::new());
    let t = Instant::now();
    let (mut pos, mut next) = (0, ids.clone());
    for _ in 0..n {
        let mut x = whole.embed(&next)?;
        whole.forward(&mut x, next.len(), pos, &mut kv)?;
        let s = whole.logits(&x[(next.len() - 1) * cfg.n_embd..], 1)?;
        pos += next.len();
        let id = sample::argmax(&s);
        if tok.is_stop(id) {
            break;
        }
        out_whole.push(id);
        next = vec![id];
    }
    let secs = t.elapsed().as_secs_f64();
    let (mut pos, mut next) = (0, ids.clone());
    for _ in 0..out_whole.len() {
        let mut x = a.embed(&next)?;
        a.forward(&mut x, next.len(), pos, &mut ka)?;
        b.forward(&mut x, next.len(), pos, &mut kb)?;
        let s = b.logits(&x[(next.len() - 1) * cfg.n_embd..], 1)?;
        pos += next.len();
        let id = sample::argmax(&s);
        out_split.push(id);
        next = vec![id];
    }
    println!("---\n{}\n---", tok.decode(&out_whole));
    println!("{} tokens in {secs:.2}s ({:.1} tokens/s, prompt included)", out_whole.len(), out_whole.len() as f64 / secs);
    println!("split in two gives the same tokens: {}", out_whole == out_split);
    Ok(())
}

fn pick(all: &HashMap<String, std::sync::Arc<bc_infer::quant::QTensor>>, h: &bc_infer::gguf::Header, layers: std::ops::Range<u32>, first: bool, last: bool) -> HashMap<String, std::sync::Arc<bc_infer::quant::QTensor>> {
    model::needed(h, &layers, first, last).into_iter().map(|t| (t.name.clone(), all[&t.name].clone())).collect()
}
