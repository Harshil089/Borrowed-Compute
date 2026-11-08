//! Growing a tree of guesses with a draft model (SpecInfer; growth by cumulative
//! probability as in EAGLE-2). Used by the coordinator for its own part of the tree and
//! by swarm drafters on other devices for theirs.

use anyhow::Result;
use bc_infer::{Kv, Part, ROOT};

/// Guesses per pass at most, how deep, how many children a guess may have at the root
/// and below, and how many guesses each level keeps.
pub const TREE_NODES: usize = 40;
pub const TREE_DEPTH: usize = 6;
pub const TREE_ROOT_K: usize = 4;
pub const TREE_CHILD_K: usize = 3;
pub const TREE_WIDTH: usize = 8;

/// The `k` likeliest tokens of `scores` with their log-probabilities, likeliest first.
pub fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + scores.iter().map(|s| (s - max).exp()).sum::<f32>().ln();
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (t, &s) in scores.iter().enumerate() {
        if best.len() < k || s > best[best.len() - 1].1 {
            let at = best.partition_point(|b| b.1 >= s);
            best.insert(at, (t as u32, s));
            best.truncate(k);
        }
    }
    best.into_iter().map(|(t, s)| (t, s - lse)).collect()
}

/// Grows a tree of guesses after the draft's cached text (`pos` tokens), starting from
/// `roots` (first tokens with their log-probabilities): level by level, each guess on the
/// frontier proposes its likeliest next tokens, and the level keeps those whose whole
/// path is likeliest. Each level is one tree pass of the draft over every guess so far.
/// Returns (token, parent) per guess, roots first; parents index earlier guesses, or
/// `ROOT` for the roots.
pub fn grow(part: &Part, kv: &mut Kv, pos: usize, roots: Vec<(u32, f32)>, budget: usize) -> Result<Vec<(u32, u32)>> {
    let vocab = part.cfg.vocab;
    let mut nodes: Vec<(u32, u32, f32)> = roots.into_iter().take(budget).map(|(t, lp)| (t, ROOT, lp)).collect();
    let mut frontier: Vec<usize> = (0..nodes.len()).collect();
    for _ in 1..TREE_DEPTH {
        if frontier.is_empty() || nodes.len() >= budget {
            break;
        }
        let tokens: Vec<u32> = nodes.iter().map(|n| n.0).collect();
        let parents: Vec<u32> = nodes.iter().map(|n| n.1).collect();
        let mut x = part.embed(&tokens)?;
        part.forward_tree(&mut x, tokens.len(), pos, Some(&parents), kv)?;
        // Scores only for the newest level: the output head over the whole vocabulary is
        // the dearest product of a small draft model.
        let width = part.cfg.n_embd;
        let rows: Vec<f32> = frontier.iter().flat_map(|&f| x[f * width..(f + 1) * width].iter().copied()).collect();
        let scores = part.logits(&rows, frontier.len())?;
        let mut next: Vec<(f32, u32, usize)> = Vec::new();
        for (i, &f) in frontier.iter().enumerate() {
            for (t, lp) in top_k(&scores[i * vocab..(i + 1) * vocab], TREE_CHILD_K) {
                next.push((nodes[f].2 + lp, t, f));
            }
        }
        next.sort_by(|a, b| b.0.total_cmp(&a.0));
        next.truncate(TREE_WIDTH.min(budget - nodes.len()));
        frontier.clear();
        for (lp, t, f) in next {
            frontier.push(nodes.len());
            nodes.push((t, f as u32, lp));
        }
    }
    Ok(nodes.into_iter().map(|(t, p, _)| (t, p)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_the_likeliest_in_order_with_log_probabilities() {
        let scores = [0.0, 3.0, 1.0, 3.0, -2.0];
        let best = top_k(&scores, 3);
        assert_eq!(best.iter().map(|b| b.0).collect::<Vec<_>>(), [1, 3, 2]);
        let total: f32 = scores.iter().map(|s| s.exp()).sum();
        assert!((best[0].1 - (3f32.exp() / total).ln()).abs() < 1e-5);
    }
}
