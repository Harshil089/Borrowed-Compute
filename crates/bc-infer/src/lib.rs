//! Borrowed Compute's own inference engine: runs a quantized transformer, or any
//! contiguous range of its layers, on the CPU. Weights stay compressed in memory and are
//! read from the model's published GGUF file by byte range, never stored on disk.

#[doc(hidden)]
pub mod fixtures;
pub mod gguf;
pub mod gpu;
pub mod model;
pub mod pin;
pub mod pool;
pub mod quant;
pub mod sample;
pub mod source;
pub mod tokenizer;

pub use model::{Config, Kv, Part, MAX_TREE, ROOT};
pub use sample::Sampling;
pub use source::Source;
pub use tokenizer::Tokenizer;

use anyhow::Result;

/// A whole model on this device: for the coordinator's draft model, and for checking a
/// split against an unsplit run.
pub struct Local {
    pub part: Part,
    pub kv: Kv,
    pub tokenizer: Tokenizer,
}

impl Local {
    pub async fn load(source: &Source) -> Result<Self> {
        Self::load_with(source, |_, _| {}).await
    }

    /// [`Local::load`], reporting bytes fetched and bytes in all as it goes.
    pub async fn load_with(source: &Source, progress: impl Fn(u64, u64) + Send + Sync) -> Result<Self> {
        let header = source.header().await?;
        let cfg = Config::from_header(&header)?;
        let layers = 0..cfg.n_layer as u32;
        let wanted = model::needed(&header, &layers, true, true);
        let tensors = source.tensors(&wanted, progress).await?;
        let part = Part::new(cfg, layers, true, true, &tensors)?;
        let kv = part.new_kv();
        Ok(Local { part, kv, tokenizer: Tokenizer::from_header(&header)? })
    }

    /// Scores after each of `tokens`, placed at `pos..` (rows × vocab).
    pub fn scores(&mut self, tokens: &[u32], pos: usize) -> Result<Vec<f32>> {
        let mut x = self.part.embed(tokens)?;
        self.part.forward(&mut x, tokens.len(), pos, &mut self.kv)?;
        self.part.logits(&x, tokens.len())
    }
}
