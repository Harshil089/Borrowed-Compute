//! A small random model as a GGUF file, for tests here and in the crates that use this one.

use crate::gguf::write::{gguf, Meta};
use crate::quant::quantize_q8_0;

/// A small random Llama-shaped model as a GGUF file: `layers` layers, width 64.
pub fn tiny_model(layers: u32, neox: bool) -> Vec<u8> {
    let (d, ff, vocab, heads, kv_heads) = (64u64, 128u64, 96u64, 4u32, 2u32);
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = |n: u64, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed % 2001) as f32 / 1000.0 - 1.0) * scale
            })
            .collect()
    };
    let q8 = |v: Vec<f32>| quantize_q8_0(&v);
    let f32b = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![("token_embd.weight".into(), vec![d, vocab], 8, q8(rnd(d * vocab, 1.0)))];
    for l in 0..layers {
        let p = |s: &str| format!("blk.{l}.{s}");
        let kvd = d / heads as u64 * kv_heads as u64;
        tensors.push((p("attn_norm.weight"), vec![d], 0, f32b(rnd(d, 0.2).iter().map(|v| v + 1.0).collect())));
        tensors.push((p("attn_q.weight"), vec![d, d], 8, q8(rnd(d * d, 0.15))));
        tensors.push((p("attn_k.weight"), vec![d, kvd], 8, q8(rnd(d * kvd, 0.15))));
        tensors.push((p("attn_v.weight"), vec![d, kvd], 8, q8(rnd(d * kvd, 0.15))));
        tensors.push((p("attn_q.bias"), vec![d], 0, f32b(rnd(d, 0.1))));
        tensors.push((p("attn_output.weight"), vec![d, d], 8, q8(rnd(d * d, 0.15))));
        tensors.push((p("ffn_norm.weight"), vec![d], 0, f32b(rnd(d, 0.2).iter().map(|v| v + 1.0).collect())));
        tensors.push((p("ffn_gate.weight"), vec![d, ff], 8, q8(rnd(d * ff, 0.15))));
        tensors.push((p("ffn_up.weight"), vec![d, ff], 8, q8(rnd(d * ff, 0.15))));
        tensors.push((p("ffn_down.weight"), vec![ff, d], 8, q8(rnd(ff * d, 0.15))));
    }
    tensors.push(("output_norm.weight".into(), vec![d], 0, f32b(vec![1.0; d as usize])));
    tensors.push(("output.weight".into(), vec![d, vocab], 8, q8(rnd(d * vocab, 0.3))));
    let arch = if neox { "qwen2" } else { "llama" };
    // Tokens: specials, then printable characters (the test tokenizer is per character).
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    tokens.extend((0..(vocab as usize - 3)).map(|i| char::from(b' ' + (i % 95) as u8).to_string()));
    let meta = vec![
        ("general.architecture", Meta::Str(arch.into())),
        ("general.name", Meta::Str("tiny".into())),
        (if neox { "qwen2.embedding_length" } else { "llama.embedding_length" }, Meta::U32(d as u32)),
        (if neox { "qwen2.block_count" } else { "llama.block_count" }, Meta::U32(layers)),
        (if neox { "qwen2.attention.head_count" } else { "llama.attention.head_count" }, Meta::U32(heads)),
        (if neox { "qwen2.attention.head_count_kv" } else { "llama.attention.head_count_kv" }, Meta::U32(kv_heads)),
        (if neox { "qwen2.feed_forward_length" } else { "llama.feed_forward_length" }, Meta::U32(ff as u32)),
        (if neox { "qwen2.context_length" } else { "llama.context_length" }, Meta::U32(2048)),
        ("tokenizer.ggml.model", Meta::Str("llama".into())),
        ("tokenizer.ggml.tokens", Meta::Strs(tokens)),
        ("tokenizer.ggml.scores", Meta::F32s(vec![0.0; vocab as usize])),
        ("tokenizer.ggml.bos_token_id", Meta::U32(1)),
        ("tokenizer.ggml.eos_token_id", Meta::U32(2)),
    ];
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors.iter().map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone())).collect();
    gguf(&meta, &tensors)
}

/// A `rows`×`cols` tensor of `ty` with random weights and sane scales, for checking every
/// block format's kernels against each other.
pub fn random_tensor(ty: crate::quant::GgmlType, rows: usize, cols: usize, seed: u64) -> crate::quant::QTensor {
    use crate::quant::{f32_to_f16, GgmlType::*};
    let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let len = ty.bytes_for(rows * cols).expect("whole blocks");
    let mut data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
    let (n, size) = ty.block();
    let small = |next: &mut dyn FnMut() -> u64| f32_to_f16(((next() % 1000) as f32 + 1.0) / 40000.0).to_le_bytes();
    for blk in data.chunks_exact_mut(size) {
        match ty {
            F32 => blk.copy_from_slice(&(((next() % 2001) as f32 - 1000.0) / 1000.0).to_le_bytes()),
            F16 => blk.copy_from_slice(&f32_to_f16(((next() % 2001) as f32 - 1000.0) / 1000.0).to_le_bytes()),
            BF16 => blk.copy_from_slice(&(((((next() % 2001) as f32 - 1000.0) / 1000.0).to_bits() >> 16) as u16).to_le_bytes()),
            Q6K => blk[208..210].copy_from_slice(&small(&mut next)),
            _ => {
                blk[0..2].copy_from_slice(&small(&mut next));
                if matches!(ty, Q4_1 | Q5_1 | Q4K | Q5K) {
                    blk[2..4].copy_from_slice(&small(&mut next));
                }
            }
        }
    }
    debug_assert_eq!(len, rows * cols / n * size);
    crate::quant::QTensor::new(ty, rows, cols, data).expect("valid tensor")
}
