//! Text to token ids and back, from the vocabulary inside the GGUF header.
//!
//! Two families:
//! - **SentencePiece** (`tokenizer.ggml.model = llama`): spaces become `▁`, the text is
//!   split into characters, and the adjacent pair whose merge has the highest score is
//!   merged until none is in the vocabulary. Characters missing from it fall back to byte
//!   tokens `<0xNN>`.
//! - **Byte-level BPE** (`gpt2`, used by Qwen and Llama 3): bytes map to printable
//!   characters, the text is cut into words first (letters, single digits, punctuation
//!   runs, whitespace, after the GPT-4 pattern), and each word merges the pair with the
//!   lowest merge rank until none is left.
//!
//! Control tokens such as `<|im_start|>` written in the text are matched whole first.

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::gguf::{Header, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Spm,
    Bpe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>role\n…<|im_end|>` (Qwen).
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>` (Llama 3).
    Llama3,
    /// No chat format: the text as is.
    Plain,
}

pub struct Tokenizer {
    kind: Kind,
    tokens: Vec<String>,
    scores: Vec<f32>,
    index: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    specials: Vec<(String, u32)>,
    pub bos: Option<u32>,
    pub eos: u32,
    add_bos: bool,
    pub template: Template,
    stops: Vec<u32>,
    /// Digits per word: Llama 3 groups up to three, Qwen splits every digit.
    digits: usize,
}

fn byte_to_char() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_header(h: &Header) -> Result<Self> {
        let strs = |k: &str| -> Vec<String> {
            h.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default()
        };
        let tokens = strs("tokenizer.ggml.tokens");
        if tokens.is_empty() {
            anyhow::bail!("the model has no vocabulary");
        }
        let kind = match h.get("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("llama") {
            "gpt2" => Kind::Bpe,
            _ => Kind::Spm,
        };
        let scores: Vec<f32> = h
            .get("tokenizer.ggml.scores")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_else(|| vec![0.0; tokens.len()]);
        let types: Vec<u64> = h
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1)).collect())
            .unwrap_or_default();
        let ranks = strs("tokenizer.ggml.merges")
            .into_iter()
            .enumerate()
            .filter_map(|(i, m)| m.split_once(' ').map(|(a, b)| ((a.to_string(), b.to_string()), i)))
            .collect();
        let index: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        // Control (3) and user-defined (4) tokens are matched whole, longest first.
        let mut specials: Vec<(String, u32)> = types
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == 3 || t == 4)
            .map(|(i, _)| (tokens[i].clone(), i as u32))
            .filter(|(s, _)| !s.is_empty())
            .collect();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let id = |k: &str| h.get(k).and_then(Value::as_u64).map(|v| v as u32);
        let eos = id("tokenizer.ggml.eos_token_id").context("no end-of-text token")?;
        let bos = id("tokenizer.ggml.bos_token_id");
        let add_bos = h
            .get("tokenizer.ggml.add_bos_token")
            .map(|v| matches!(v, Value::Bool(true)))
            .unwrap_or(kind == Kind::Spm);
        let template = if index.contains_key("<|im_start|>") {
            Template::ChatMl
        } else if index.contains_key("<|start_header_id|>") {
            Template::Llama3
        } else {
            Template::Plain
        };
        let mut stops = vec![eos];
        stops.extend(["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>"].iter().filter_map(|s| index.get(*s).copied()));
        let digits = match h.get("tokenizer.ggml.pre").and_then(Value::as_str) {
            Some("llama-bpe" | "llama3" | "smaug-bpe") => 3,
            _ => 1,
        };
        Ok(Tokenizer { kind, tokens, scores, index, ranks, specials, bos, eos, add_bos, template, stops, digits })
    }

    pub fn vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stops.contains(&id)
    }

    /// A chat as the model was trained to read it, ending where the assistant speaks.
    pub fn chat_prompt(&self, messages: &[(String, String)]) -> String {
        let mut s = String::new();
        match self.template {
            Template::ChatMl => {
                if !messages.iter().any(|(r, _)| r == "system") {
                    s.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
                }
                for (role, text) in messages {
                    s.push_str(&format!("<|im_start|>{role}\n{text}<|im_end|>\n"));
                }
                s.push_str("<|im_start|>assistant\n");
            }
            Template::Llama3 => {
                for (role, text) in messages {
                    s.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"));
                }
                s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            Template::Plain => {
                for (_, text) in messages {
                    s.push_str(text);
                    s.push('\n');
                }
            }
        }
        s
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            out.extend(self.bos);
        }
        let mut rest = text;
        let mut first = true;
        while !rest.is_empty() {
            // The earliest special token in what is left, if any.
            let hit = self.specials.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id))).min_by_key(|(at, len, _)| (*at, std::cmp::Reverse(*len)));
            let (plain, special) = match hit {
                Some((at, len, id)) => (&rest[..at], Some((id, at + len))),
                None => (rest, None),
            };
            if !plain.is_empty() {
                match self.kind {
                    Kind::Spm => self.spm(plain, first, &mut out),
                    Kind::Bpe => self.bpe(plain, &mut out),
                }
            }
            first = false;
            match special {
                Some((id, end)) => {
                    out.push(id);
                    rest = &rest[end..];
                }
                None => break,
            }
        }
        out
    }

    fn spm(&self, text: &str, prefix_space: bool, out: &mut Vec<u32>) {
        let mut s = String::new();
        if prefix_space {
            s.push(' ');
        }
        s.push_str(text);
        let s = s.replace(' ', "▁");
        let mut syms: Vec<String> = s.chars().map(|c| c.to_string()).collect();
        loop {
            let best = (0..syms.len().saturating_sub(1))
                .filter_map(|i| {
                    let merged = format!("{}{}", syms[i], syms[i + 1]);
                    self.index.get(&merged).map(|&id| (i, self.scores.get(id as usize).copied().unwrap_or(0.0)))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
            let Some((i, _)) = best else { break };
            let next = syms.remove(i + 1);
            syms[i].push_str(&next);
        }
        for sym in syms {
            match self.index.get(&sym) {
                Some(&id) => out.push(id),
                None => {
                    for b in sym.bytes() {
                        out.push(self.index.get(&format!("<0x{b:02X}>")).copied().unwrap_or(0));
                    }
                }
            }
        }
    }

    fn bpe(&self, text: &str, out: &mut Vec<u32>) {
        let map = byte_to_char();
        for word in pretokenize(text, self.digits) {
            let mut parts: Vec<String> = word.bytes().map(|b| map[b as usize].to_string()).collect();
            loop {
                let best = (0..parts.len().saturating_sub(1))
                    .filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|&r| (r, i)))
                    .min();
                let Some((_, i)) = best else { break };
                let next = parts.remove(i + 1);
                parts[i].push_str(&next);
            }
            for p in parts {
                match self.index.get(&p) {
                    Some(&id) => out.push(id),
                    None => out.extend(p.chars().filter_map(|c| self.index.get(&c.to_string()).copied())),
                }
            }
        }
    }

    /// Bytes a token stands for (control tokens decode to nothing).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.specials.iter().any(|(_, s)| *s == id) || Some(id) == self.bos || id == self.eos {
            return Vec::new();
        }
        match self.kind {
            Kind::Spm => {
                if t.len() == 6 && t.starts_with("<0x") && t.ends_with('>') {
                    if let Ok(b) = u8::from_str_radix(&t[3..5], 16) {
                        return vec![b];
                    }
                }
                t.replace('▁', " ").into_bytes()
            }
            Kind::Bpe => {
                let map = byte_to_char();
                t.chars().filter_map(|c| map.iter().position(|&m| m == c).map(|b| b as u8)).collect()
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        let s = String::from_utf8_lossy(&bytes).into_owned();
        // SentencePiece adds a space before the first word; take it back off.
        match (self.kind, s.strip_prefix(' ')) {
            (Kind::Spm, Some(rest)) => rest.to_string(),
            _ => s,
        }
    }
}

/// Splits text into words the way GPT-4-style tokenizers do (contractions, letter runs
/// with one leading non-letter, single digits, punctuation runs, newlines, whitespace).
pub fn pretokenize(text: &str, digits: usize) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let is_l = |x: char| x.is_alphabetic();
    let is_n = |x: char| x.is_numeric();
    let is_nl = |x: char| x == '\r' || x == '\n';
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let start = i;
        // 's 't 're 've 'm 'll 'd
        if c[i] == '\'' {
            let rest: String = c[i + 1..c.len().min(i + 3)].iter().collect::<String>().to_lowercase();
            if let Some(n) = ["ll", "re", "ve", "s", "t", "m", "d"].iter().find(|s| rest.starts_with(**s)).map(|s| s.len()) {
                i += 1 + n;
                out.push(c[start..i].iter().collect());
                continue;
            }
        }
        if is_l(c[i]) || (!is_n(c[i]) && !is_nl(c[i]) && i + 1 < c.len() && is_l(c[i + 1])) {
            i += 1;
            while i < c.len() && is_l(c[i]) {
                i += 1;
            }
        } else if is_n(c[i]) {
            let end = (i + digits.max(1)).min(c.len());
            i += 1;
            while i < end && is_n(c[i]) {
                i += 1;
            }
        } else if !c[i].is_whitespace() || (c[i] == ' ' && i + 1 < c.len() && !c[i + 1].is_whitespace() && !is_l(c[i + 1]) && !is_n(c[i + 1])) {
            if c[i] == ' ' {
                i += 1;
            }
            while i < c.len() && !c[i].is_whitespace() && !is_l(c[i]) && !is_n(c[i]) {
                i += 1;
            }
            while i < c.len() && is_nl(c[i]) {
                i += 1;
            }
        } else {
            let mut end = i;
            while end < c.len() && c[end].is_whitespace() {
                end += 1;
            }
            let run = &c[i..end];
            if let Some(last_nl) = run.iter().rposition(|&x| is_nl(x)) {
                i += last_nl + 1;
            } else if end < c.len() && end - i > 1 {
                i = end - 1; // leave one space for the next word
            } else {
                i = end;
            }
        }
        out.push(c[start..i].iter().collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::write::{gguf, Meta};

    fn header(meta: Vec<(&str, Meta)>) -> Header {
        crate::gguf::parse(&gguf(&meta, &[])).unwrap()
    }

    #[test]
    fn pretokenizes_like_gpt4() {
        assert_eq!(pretokenize("Hello world, it's 42!\n\n  ok", 1), ["Hello", " world", ",", " it", "'s", " ", "4", "2", "!\n\n", " ", " ok"]);
        assert_eq!(pretokenize("12345", 3), ["123", "45"]);
    }

    #[test]
    fn spm_merges_by_score_and_falls_back_to_bytes() {
        let tokens = ["<unk>", "<s>", "</s>", "▁", "h", "i", "▁h", "▁hi", "<0xC3>", "<0xA9>"];
        let h = header(vec![
            ("tokenizer.ggml.model", Meta::Str("llama".into())),
            ("tokenizer.ggml.tokens", Meta::Strs(tokens.iter().map(|s| s.to_string()).collect())),
            ("tokenizer.ggml.scores", Meta::F32s(vec![0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -0.5, -0.1, 0.0, 0.0])),
            ("tokenizer.ggml.bos_token_id", Meta::U32(1)),
            ("tokenizer.ggml.eos_token_id", Meta::U32(2)),
        ]);
        let t = Tokenizer::from_header(&h).unwrap();
        let ids = t.encode("hi é");
        assert_eq!(ids, [1, 7, 3, 8, 9]);
        assert_eq!(t.decode(&ids), "hi é");
    }

    #[test]
    fn bpe_merges_by_rank_and_keeps_specials_whole() {
        let map = byte_to_char();
        let sp = map[b' ' as usize].to_string();
        let tokens: Vec<String> = ["h", "e", "y", &sp, "he", "hey", &format!("{sp}h"), &format!("{sp}hey"), "<|im_end|>", "!"].iter().map(|s| s.to_string()).collect();
        let h = header(vec![
            ("tokenizer.ggml.model", Meta::Str("gpt2".into())),
            ("tokenizer.ggml.tokens", Meta::Strs(tokens)),
            ("tokenizer.ggml.token_type", Meta::I32s(vec![1, 1, 1, 1, 1, 1, 1, 1, 3, 1])),
            ("tokenizer.ggml.merges", Meta::Strs(vec!["h e".into(), "he y".into(), format!("{sp} h"), format!("{sp}h e"), format!("{sp} hey")])),
            ("tokenizer.ggml.eos_token_id", Meta::U32(8)),
        ]);
        let t = Tokenizer::from_header(&h).unwrap();
        let ids = t.encode("hey hey!<|im_end|>");
        assert_eq!(ids, [5, 7, 9, 8]);
        assert_eq!(t.decode(&ids), "hey hey!");
        assert!(t.is_stop(8));
        assert_eq!(t.template, Template::Plain, "no <|im_start|> in this vocabulary");
    }
}
