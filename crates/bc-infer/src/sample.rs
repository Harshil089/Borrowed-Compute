//! Picking the next token from the model's scores.
//!
//! Temperature 0 takes the highest score (greedy), which is deterministic: the same
//! prompt gives the same text on any split of the model, and speculative decoding can
//! check a draft exactly. Otherwise p(i) = softmax(score/T), cut to the smallest set of
//! tokens whose probability reaches `top_p` (nucleus sampling), and drawn with a seeded
//! generator so a request can be replayed.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0.0, top_p: 1.0, seed: 0 }
    }
}

pub fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0;
    for (i, s) in scores.iter().enumerate() {
        if *s > scores[best] {
            best = i;
        }
    }
    best as u32
}

/// `step` varies the draw between positions of one request.
pub fn sample(scores: &[f32], s: &Sampling, step: u64) -> u32 {
    if s.temperature <= 0.0 {
        return argmax(scores);
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<(u32, f32)> = scores.iter().enumerate().map(|(i, v)| (i as u32, ((v - max) / s.temperature).exp())).collect();
    let total: f32 = p.iter().map(|x| x.1).sum();
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut keep = 0;
    let mut mass = 0.0;
    for (_, w) in &p {
        keep += 1;
        mass += w / total;
        if mass >= s.top_p {
            break;
        }
    }
    p.truncate(keep.max(1));
    let kept: f32 = p.iter().map(|x| x.1).sum();
    let mut r = uniform(s.seed ^ step.wrapping_mul(0x9e3779b97f4a7c15)) * kept;
    for (id, w) in &p {
        r -= w;
        if r <= 0.0 {
            return *id;
        }
    }
    p[0].0
}

/// A uniform number in [0, 1) from a seed (splitmix64).
fn uniform(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_nucleus() {
        let scores = [0.1, 3.0, 2.9, -5.0];
        assert_eq!(sample(&scores, &Sampling::default(), 0), 1);
        // A tight nucleus keeps only the top token.
        let s = Sampling { temperature: 1.0, top_p: 0.01, seed: 7 };
        assert!((0..20).all(|i| sample(&scores, &s, i) == 1));
        // A wide one draws both close candidates, never the far ones.
        let s = Sampling { temperature: 1.0, top_p: 0.95, seed: 7 };
        let picks: Vec<u32> = (0..200).map(|i| sample(&scores, &s, i)).collect();
        assert!(picks.contains(&1) && picks.contains(&2) && !picks.contains(&3));
        assert_eq!(picks, (0..200).map(|i| sample(&scores, &s, i)).collect::<Vec<_>>(), "seeded");
    }
}
