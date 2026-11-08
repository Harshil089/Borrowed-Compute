//! Growing a tree of guesses with a draft model (SpecInfer; growth by cumulative
//! probability as in EAGLE-2). Used by the coordinator for its own part of the tree and
//! by swarm drafters on other devices for theirs.

use anyhow::Result;
use bc_infer::{Kv, Part, ROOT};

/// Guesses per pass at most, how deep, how many children a guess may have at the root
/// and below, and how many guesses each level keeps.
// shortcut: 15 guesses (16 rows, one tile of the batched GPU product) is what pays on an
// M2, where a 41-row check costs 6 one-row passes; size it per GPU once CUDA's tensor
// cores are measured.
pub const TREE_NODES: usize = 15;
pub const TREE_DEPTH: usize = 6;
pub const TREE_ROOT_K: usize = 4;
pub const TREE_CHILD_K: usize = 3;
pub const TREE_WIDTH: usize = 8;

/// Tokens the draft model scores: the first ids of the vocabulary. Byte-pair vocabularies
/// number tokens by how early (so how often) they were merged, and the output head over
/// all ~150k is about a quarter of a small draft's work. The target still checks every
/// token, so a rarer one only goes unguessed.
// shortcut: id order as frequency, true of byte-pair vocabularies; count real use if a
// model's vocabulary is ordered otherwise.
pub const DRAFT_VOCAB: usize = 32768;
/// Guesses read from the text itself at most, and the longest repeat they key on.
pub const LOOKUP_TOKENS: usize = 8;
const LOOKUP_KEY: usize = 4;

/// Guesses that cost nothing (prompt lookup): where the text's last few tokens appeared
/// before, the tokens that followed them then. Code, edits and summaries repeat their
/// input a lot. Tries the longest key first (4 tokens, down to 2) and the latest place.
pub fn lookup(text: &[u32], max: usize) -> Vec<u32> {
    for n in (2..=LOOKUP_KEY).rev() {
        if text.len() <= n {
            continue;
        }
        let key = &text[text.len() - n..];
        if let Some(at) = text[..text.len() - 1].windows(n).rposition(|w| w == key) {
            let from = at + n;
            return text[from..(from + max).min(text.len())].to_vec();
        }
    }
    Vec::new()
}

/// Adds `chain` (guesses in order after the confirmed text) to a tree, sharing the
/// nodes it already has, until the tree holds `max` nodes.
pub fn add_chain(nodes: &mut Vec<(u32, u32)>, chain: &[u32], max: usize) {
    let mut parent = ROOT;
    for &t in chain {
        parent = match nodes.iter().position(|&n| n == (t, parent)) {
            Some(i) => i as u32,
            None if nodes.len() < max => {
                nodes.push((t, parent));
                nodes.len() as u32 - 1
            }
            None => return,
        };
    }
}

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
/// path is likeliest. Each level is one pass of the draft over that level's guesses
/// alone: the earlier levels are already in its cache. Returns (token, parent) per
/// guess, roots first; parents index earlier guesses, or `ROOT` for the roots.
pub fn grow(part: &Part, kv: &mut Kv, pos: usize, roots: Vec<(u32, f32)>, budget: usize) -> Result<Vec<(u32, u32)>> {
    let mut nodes: Vec<(u32, u32, f32)> = roots.into_iter().take(budget).map(|(t, lp)| (t, ROOT, lp)).collect();
    // Guesses already run through the draft; the rest are the frontier.
    let mut done = 0;
    for _ in 1..TREE_DEPTH {
        if done == nodes.len() || nodes.len() >= budget {
            break;
        }
        let tokens: Vec<u32> = nodes[done..].iter().map(|n| n.0).collect();
        let parents: Vec<u32> = nodes.iter().map(|n| n.1).collect();
        let mut x = part.embed(&tokens)?;
        part.forward_tree(&mut x, tokens.len(), pos + done, Some(&parents), kv)?;
        let scores = part.logits_upto(&x, tokens.len(), DRAFT_VOCAB)?;
        let vocab = scores.len() / tokens.len();
        let mut next: Vec<(f32, u32, usize)> = Vec::new();
        for (i, f) in (done..nodes.len()).enumerate() {
            for (t, lp) in top_k(&scores[i * vocab..(i + 1) * vocab], TREE_CHILD_K) {
                next.push((nodes[f].2 + lp, t, f));
            }
        }
        done = nodes.len();
        next.sort_by(|a, b| b.0.total_cmp(&a.0));
        next.truncate(TREE_WIDTH.min(budget - nodes.len()));
        nodes.extend(next.into_iter().map(|(lp, t, f)| (t, f as u32, lp)));
    }
    Ok(nodes.into_iter().map(|(t, p, _)| (t, p)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_continues_where_the_text_repeated() {
        // "1 2 3 4 5 ... 1 2 3" goes on with "4 5 9" as before.
        let text = [1, 2, 3, 4, 5, 9, 7, 1, 2, 3];
        assert_eq!(lookup(&text, 3), [4, 5, 9]);
        assert_eq!(lookup(&text, 20), [4, 5, 9, 7, 1, 2, 3]);
        // The longest key wins over a later, shorter match; nothing repeated, nothing.
        assert_eq!(lookup(&[8, 2, 3, 6, 9, 3, 2, 3], 2), [6, 9]);
        assert!(lookup(&[1, 2, 3, 4], 4).is_empty());
        assert!(lookup(&[], 4).is_empty());
    }

    #[test]
    fn a_chain_joins_the_tree_where_they_agree() {
        let mut nodes = vec![(10, ROOT), (11, 0), (12, ROOT)];
        add_chain(&mut nodes, &[10, 11, 13, 14], 5);
        assert_eq!(nodes, [(10, ROOT), (11, 0), (12, ROOT), (13, 1), (14, 3)]);
        add_chain(&mut nodes, &[12, 15], 5);
        assert_eq!(nodes.len(), 5, "the tree is full");
        let mut empty = Vec::new();
        add_chain(&mut empty, &[7, 7], 9);
        assert_eq!(empty, [(7, ROOT), (7, 0)]);
    }

    #[test]
    fn top_k_is_the_likeliest_in_order_with_log_probabilities() {
        let scores = [0.0, 3.0, 1.0, 3.0, -2.0];
        let best = top_k(&scores, 3);
        assert_eq!(best.iter().map(|b| b.0).collect::<Vec<_>>(), [1, 3, 2]);
        let total: f32 = scores.iter().map(|s| s.exp()).sum();
        assert!((best[0].1 - (3f32.exp() / total).ln()).abs() < 1e-5);
    }
}
