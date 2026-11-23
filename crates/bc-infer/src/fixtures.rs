//! A small random model as a GGUF file, for tests here and in the crates that use this one.

use crate::gguf::write::{gguf, Meta};
use crate::quant::quantize_q8_0;

/// A small random Llama-shaped model as a GGUF file: `layers` layers, width 64.
pub fn tiny_model(layers: u32, neox: bool) -> Vec<u8> {
    llama_like(layers, if neox { "qwen2" } else { "llama" })
}

/// [`tiny_model`] shaped like Qwen3: no query bias, each head's query and key normed.
pub fn tiny_qwen3(layers: u32) -> Vec<u8> {
    llama_like(layers, "qwen3")
}

fn llama_like(layers: u32, arch: &str) -> Vec<u8> {
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
        if arch == "qwen3" {
            let hd = d / heads as u64;
            tensors.push((p("attn_q_norm.weight"), vec![hd], 0, f32b(rnd(hd, 0.2).iter().map(|v| v + 1.0).collect())));
            tensors.push((p("attn_k_norm.weight"), vec![hd], 0, f32b(rnd(hd, 0.2).iter().map(|v| v + 1.0).collect())));
        } else {
            tensors.push((p("attn_q.bias"), vec![d], 0, f32b(rnd(d, 0.1))));
        }
        tensors.push((p("attn_output.weight"), vec![d, d], 8, q8(rnd(d * d, 0.15))));
        tensors.push((p("ffn_norm.weight"), vec![d], 0, f32b(rnd(d, 0.2).iter().map(|v| v + 1.0).collect())));
        tensors.push((p("ffn_gate.weight"), vec![d, ff], 8, q8(rnd(d * ff, 0.15))));
        tensors.push((p("ffn_up.weight"), vec![d, ff], 8, q8(rnd(d * ff, 0.15))));
        tensors.push((p("ffn_down.weight"), vec![ff, d], 8, q8(rnd(ff * d, 0.15))));
    }
    tensors.push(("output_norm.weight".into(), vec![d], 0, f32b(vec![1.0; d as usize])));
    tensors.push(("output.weight".into(), vec![d, vocab], 8, q8(rnd(d * vocab, 0.3))));
    // Tokens: specials, then printable characters (the test tokenizer is per character).
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    tokens.extend((0..(vocab as usize - 3)).map(|i| char::from(b' ' + (i % 95) as u8).to_string()));
    let meta = vec![
        ("general.architecture".to_string(), Meta::Str(arch.into())),
        ("general.name".to_string(), Meta::Str("tiny".into())),
        (format!("{arch}.embedding_length"), Meta::U32(d as u32)),
        (format!("{arch}.block_count"), Meta::U32(layers)),
        (format!("{arch}.attention.head_count"), Meta::U32(heads)),
        (format!("{arch}.attention.head_count_kv"), Meta::U32(kv_heads)),
        (format!("{arch}.feed_forward_length"), Meta::U32(ff as u32)),
        (format!("{arch}.context_length"), Meta::U32(2048)),
        ("tokenizer.ggml.model".to_string(), Meta::Str("llama".into())),
        ("tokenizer.ggml.tokens".to_string(), Meta::Strs(tokens)),
        ("tokenizer.ggml.scores".to_string(), Meta::F32s(vec![0.0; vocab as usize])),
        ("tokenizer.ggml.bos_token_id".to_string(), Meta::U32(1)),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(2)),
    ];
    let meta: Vec<(&str, Meta)> = meta.into_iter().map(|(k, v)| (&*Box::leak(k.into_boxed_str()), v)).collect();
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors.iter().map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone())).collect();
    gguf(&meta, &tensors)
}

/// A small random Qwen3.5-shaped model: `layers` layers, every second one gated attention,
/// the others gated DeltaNet. `fused` stores q, k, v and z as one matrix and β with α as
/// another, as newer conversions do; the weights are the same either way.
pub fn tiny_qwen35(layers: u32, fused: bool) -> Vec<u8> {
    qwen35_like(layers, fused, false)
}

/// [`tiny_qwen35`] with a mixture of 8 experts (2 a token, 32 wide) and a shared expert in
/// every layer (`qwen35moe`); `fused` also stacks each expert's gate and up together.
pub fn tiny_qwen35_moe(layers: u32, fused: bool) -> Vec<u8> {
    qwen35_like(layers, fused, true)
}

fn qwen35_like(layers: u32, fused: bool, moe: bool) -> Vec<u8> {
    let (experts, used, ef) = (8u64, 2u32, 32u64);
    // Width 64; attention: 4 heads of 16, 2 key/value heads, a quarter of each rotated;
    // DeltaNet: 2 key heads and 4 value heads of 16, a convolution of 4 taps.
    let (d, ff, vocab, heads, kv_heads, hd) = (64u64, 128u64, 96u64, 4u64, 2u64, 16u64);
    let (kh, vh, sh, taps) = (2u64, 4u64, 16u64, 4u64);
    let (key_dim, value_dim) = (kh * sh, vh * sh);
    let conv_dim = 2 * key_dim + value_dim;
    let mut seed = 0x2545f4914f6cdd1du64;
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
    let q8 = |v: &[f32]| quantize_q8_0(v);
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let ones = |v: Vec<f32>| v.iter().map(|x| x + 1.0).collect::<Vec<f32>>();
    let mut t: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![("token_embd.weight".into(), vec![d, vocab], 8, q8(&rnd(d * vocab, 1.0)))];
    for l in 0..layers {
        let p = |s: &str| format!("blk.{l}.{s}");
        t.push((p("attn_norm.weight"), vec![d], 0, f32b(&ones(rnd(d, 0.2)))));
        t.push((p("post_attention_norm.weight"), vec![d], 0, f32b(&ones(rnd(d, 0.2)))));
        if l % 2 == 0 {
            let (qkv, z) = (rnd(d * conv_dim, 0.15), rnd(d * value_dim, 0.15));
            let (beta, alpha) = (rnd(d * vh, 0.15), rnd(d * vh, 0.15));
            if fused {
                t.push((p("attn_qkv_gate.weight"), vec![d, conv_dim + value_dim], 8, q8(&[qkv, z].concat())));
                t.push((p("ssm_beta_alpha.weight"), vec![d, 2 * vh], 8, q8(&[beta, alpha].concat())));
            } else {
                t.push((p("attn_qkv.weight"), vec![d, conv_dim], 8, q8(&qkv)));
                t.push((p("attn_gate.weight"), vec![d, value_dim], 8, q8(&z)));
                t.push((p("ssm_beta.weight"), vec![d, vh], 8, q8(&beta)));
                t.push((p("ssm_alpha.weight"), vec![d, vh], 8, q8(&alpha)));
            }
            t.push((p("ssm_conv1d.weight"), vec![taps, conv_dim], 0, f32b(&rnd(taps * conv_dim, 0.5))));
            t.push((p("ssm_dt.bias"), vec![vh], 0, f32b(&rnd(vh, 1.0))));
            // −e^A for A around 0: decays between e^-1 and e^-0.1 a token.
            t.push((p("ssm_a"), vec![vh], 0, f32b(&rnd(vh, 0.5).iter().map(|a| -(a.exp())).collect::<Vec<_>>())));
            t.push((p("ssm_norm.weight"), vec![sh], 0, f32b(&ones(rnd(sh, 0.2)))));
            t.push((p("ssm_out.weight"), vec![value_dim, d], 8, q8(&rnd(value_dim * d, 0.15))));
        } else {
            t.push((p("attn_q.weight"), vec![d, 2 * heads * hd], 8, q8(&rnd(d * 2 * heads * hd, 0.15))));
            t.push((p("attn_k.weight"), vec![d, kv_heads * hd], 8, q8(&rnd(d * kv_heads * hd, 0.15))));
            t.push((p("attn_v.weight"), vec![d, kv_heads * hd], 8, q8(&rnd(d * kv_heads * hd, 0.15))));
            t.push((p("attn_output.weight"), vec![heads * hd, d], 8, q8(&rnd(heads * hd * d, 0.15))));
            t.push((p("attn_q_norm.weight"), vec![hd], 0, f32b(&ones(rnd(hd, 0.2)))));
            t.push((p("attn_k_norm.weight"), vec![hd], 0, f32b(&ones(rnd(hd, 0.2)))));
        }
        if moe {
            t.push((p("ffn_gate_inp.weight"), vec![d, experts], 8, q8(&rnd(d * experts, 0.5))));
            let (gate, up) = (rnd(d * ef * experts, 0.15), rnd(d * ef * experts, 0.15));
            if fused {
                // Each expert's gate rows, then its up rows.
                let per = (d * ef) as usize;
                let both: Vec<f32> = (0..experts as usize).flat_map(|x| gate[x * per..(x + 1) * per].iter().chain(&up[x * per..(x + 1) * per]).copied().collect::<Vec<_>>()).collect();
                t.push((p("ffn_gate_up_exps.weight"), vec![d, 2 * ef, experts], 8, q8(&both)));
            } else {
                t.push((p("ffn_gate_exps.weight"), vec![d, ef, experts], 8, q8(&gate)));
                t.push((p("ffn_up_exps.weight"), vec![d, ef, experts], 8, q8(&up)));
            }
            t.push((p("ffn_down_exps.weight"), vec![ef, d, experts], 8, q8(&rnd(ef * d * experts, 0.15))));
            t.push((p("ffn_gate_inp_shexp.weight"), vec![d], 0, f32b(&rnd(d, 0.3))));
            t.push((p("ffn_gate_shexp.weight"), vec![d, ef], 8, q8(&rnd(d * ef, 0.15))));
            t.push((p("ffn_up_shexp.weight"), vec![d, ef], 8, q8(&rnd(d * ef, 0.15))));
            t.push((p("ffn_down_shexp.weight"), vec![ef, d], 8, q8(&rnd(ef * d, 0.15))));
        } else {
            t.push((p("ffn_gate.weight"), vec![d, ff], 8, q8(&rnd(d * ff, 0.15))));
            t.push((p("ffn_up.weight"), vec![d, ff], 8, q8(&rnd(d * ff, 0.15))));
            t.push((p("ffn_down.weight"), vec![ff, d], 8, q8(&rnd(ff * d, 0.15))));
        }
    }
    t.push(("output_norm.weight".into(), vec![d], 0, f32b(&vec![1.0; d as usize])));
    t.push(("output.weight".into(), vec![d, vocab], 8, q8(&rnd(d * vocab, 0.3))));
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    tokens.extend((0..(vocab as usize - 3)).map(|i| char::from(b' ' + (i % 95) as u8).to_string()));
    let arch = if moe { "qwen35moe" } else { "qwen35" };
    let k = |s: &str| format!("{arch}.{s}");
    let mut meta = vec![
        ("general.architecture".to_string(), Meta::Str(arch.into())),
        (k("embedding_length"), Meta::U32(d as u32)),
        (k("block_count"), Meta::U32(layers)),
        (k("attention.head_count"), Meta::U32(heads as u32)),
        (k("attention.head_count_kv"), Meta::U32(kv_heads as u32)),
        (k("attention.key_length"), Meta::U32(hd as u32)),
        (k("rope.dimension_count"), Meta::U32(hd as u32 / 4)),
        (k("rope.freq_base"), Meta::F32(10_000_000.0)),
        (k("feed_forward_length"), Meta::U32(ff as u32)),
        (k("context_length"), Meta::U32(2048)),
        (k("full_attention_interval"), Meta::U32(2)),
        (k("ssm.conv_kernel"), Meta::U32(taps as u32)),
        (k("ssm.state_size"), Meta::U32(sh as u32)),
        (k("ssm.group_count"), Meta::U32(kh as u32)),
        (k("ssm.time_step_rank"), Meta::U32(vh as u32)),
        (k("ssm.inner_size"), Meta::U32(value_dim as u32)),
        ("tokenizer.ggml.model".into(), Meta::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), Meta::Strs(tokens)),
        ("tokenizer.ggml.scores".into(), Meta::F32s(vec![0.0; vocab as usize])),
        ("tokenizer.ggml.bos_token_id".into(), Meta::U32(1)),
        ("tokenizer.ggml.eos_token_id".into(), Meta::U32(2)),
    ];
    if moe {
        meta.push((k("expert_count"), Meta::U32(experts as u32)));
        meta.push((k("expert_used_count"), Meta::U32(used)));
        meta.push((k("expert_feed_forward_length"), Meta::U32(ef as u32)));
    }
    let meta: Vec<(&str, Meta)> = meta.into_iter().map(|(k, v)| (&*Box::leak(k.into_boxed_str()), v)).collect();
    let t: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = t.iter().map(|(n, d, ty, b)| (n.as_str(), d.clone(), *ty, b.clone())).collect();
    gguf(&meta, &t)
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
            Q3K => blk[108..110].copy_from_slice(&small(&mut next)),
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
