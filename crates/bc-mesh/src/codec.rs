//! Activations on the wire: exact `f32`, four bytes a value.
//!
//! Earlier versions sent int8 with a scale per group of 64 values. That error, small per
//! hop, enters the residual stream, and after a few dozen tokens it can flip the choice
//! between two nearly tied tokens: the text drifts from what the model alone writes. The
//! mesh now sends exact values, so a split model writes bit for bit what the whole one
//! does. Lossless coding is not worth it either: measured on Llama 3.2 1B hidden states,
//! three of the four bytes of each float are close to random (7.8 to 8.0 bits of entropy),
//! so no lossless coder could save more than 17%.

use anyhow::{bail, Result};

pub fn encode(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Decodes `values` floats. Refuses input whose length does not match, so a short or
/// tampered frame never turns into garbage activations.
pub fn decode(bytes: &[u8], values: usize) -> Result<Vec<f32>> {
    if Some(bytes.len()) != values.checked_mul(4) {
        bail!("{} bytes do not hold {values} values", bytes.len());
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_exactly() {
        let v = vec![1.5, -0.0, f32::MAX, 3.25e-12, f32::MIN_POSITIVE / 2.0];
        let back = decode(&encode(&v), v.len()).unwrap();
        assert!(v.iter().zip(&back).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn rejects_wrong_lengths() {
        assert!(decode(&[0; 7], 2).is_err());
        assert!(decode(&encode(&[1.0; 3]), 4).is_err());
    }
}
