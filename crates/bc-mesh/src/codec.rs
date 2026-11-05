//! Activations on the wire.
//!
//! Between pipeline stages only the hidden state travels: one row of `hidden` floats per
//! token. Decoding sends one row per token, so bytes barely matter; a long prompt sends
//! thousands of rows at once, and then bytes are the whole cost, above all through a
//! relay. Int8 with a scale per group of [`GROUP`] values halves fp16's bytes. Groups are
//! small because LLM hidden states carry a few very large "outlier" channels: one scale
//! per row would flatten every other value in the row to zero.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Values that share one int8 scale.
pub const GROUP: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    /// Exact, 4 bytes a value. For tests and for stages that must match bit for bit.
    F32,
    /// About 1.06 bytes a value: an `f32` scale per group of [`GROUP`] int8 values.
    Int8,
}

/// Bytes `values` floats take on the wire.
pub fn encoded_len(values: usize, encoding: Encoding) -> usize {
    match encoding {
        Encoding::F32 => values * 4,
        Encoding::Int8 => values + values.div_ceil(GROUP) * 4,
    }
}

pub fn encode(values: &[f32], encoding: Encoding) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded_len(values.len(), encoding));
    match encoding {
        Encoding::F32 => values.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        Encoding::Int8 => {
            for group in values.chunks(GROUP) {
                let max = group.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = if max > 0.0 && max.is_finite() { max / 127.0 } else { 1.0 };
                out.extend_from_slice(&scale.to_le_bytes());
                out.extend(group.iter().map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8 as u8));
            }
        }
    }
    out
}

/// Decodes `values` floats. Refuses input whose length does not match, so a short or
/// tampered frame never turns into garbage activations.
pub fn decode(bytes: &[u8], values: usize, encoding: Encoding) -> Result<Vec<f32>> {
    if bytes.len() != encoded_len(values, encoding) {
        bail!("{} bytes do not hold {values} {encoding:?} values", bytes.len());
    }
    let mut out = Vec::with_capacity(values);
    match encoding {
        Encoding::F32 => {
            out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        }
        Encoding::Int8 => {
            let mut rest = bytes;
            while out.len() < values {
                let n = GROUP.min(values - out.len());
                let scale = f32::from_le_bytes(rest[..4].try_into().unwrap());
                out.extend(rest[4..4 + n].iter().map(|&q| q as i8 as f32 * scale));
                rest = &rest[4 + n..];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trips_exactly() {
        let v = vec![1.5, -0.0, f32::MAX, 3.25e-12];
        assert_eq!(decode(&encode(&v, Encoding::F32), 4, Encoding::F32).unwrap(), v);
    }

    #[test]
    fn int8_stays_within_half_a_step_of_each_groups_scale() {
        // Hidden-state-like: small values plus one outlier channel.
        let mut v: Vec<f32> = (0..1000).map(|i| ((i * 37 % 101) as f32 - 50.0) / 40.0).collect();
        v[3] = 900.0;
        let bytes = encode(&v, Encoding::Int8);
        assert_eq!(bytes.len(), encoded_len(1000, Encoding::Int8));
        assert!(bytes.len() < 1000 * 2 * 55 / 100, "well under fp16's size");
        let back = decode(&bytes, 1000, Encoding::Int8).unwrap();
        for (g, (orig, got)) in v.chunks(GROUP).zip(back.chunks(GROUP)).enumerate() {
            let step = orig.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
            for (a, b) in orig.iter().zip(got) {
                assert!((a - b).abs() <= step / 2.0 + 1e-6, "group {g}: {a} became {b}");
            }
        }
        // The outlier only coarsens its own group: the rest keep ~1% error.
        let tail_err = v[GROUP..].iter().zip(&back[GROUP..]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(tail_err < 0.01, "{tail_err}");
    }

    #[test]
    fn rejects_wrong_lengths() {
        let bytes = encode(&[1.0; 10], Encoding::Int8);
        assert!(decode(&bytes, 11, Encoding::Int8).is_err());
        assert!(decode(&bytes[1..], 10, Encoding::Int8).is_err());
        assert!(decode(&[0; 7], 2, Encoding::F32).is_err());
    }

    #[test]
    fn zeros_and_non_finite_groups_do_not_divide_by_zero() {
        let back = decode(&encode(&[0.0; 70], Encoding::Int8), 70, Encoding::Int8).unwrap();
        assert!(back.iter().all(|v| *v == 0.0));
    }
}
