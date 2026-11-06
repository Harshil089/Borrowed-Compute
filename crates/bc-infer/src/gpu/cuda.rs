//! NVIDIA GPUs through CUDA. The driver is loaded when the program starts, not linked, so
//! the app runs on PCs without one. The kernels ship as PTX (`kernels.ptx`, built from
//! `cuda.cu` and `kernels.h` by `build-ptx.sh`), which the driver compiles for whatever
//! card is there: users need the NVIDIA driver, not the CUDA Toolkit. Weights are copied
//! into the card's memory.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::{Arg, Cmd, K};

const PTX: &str = include_str!("kernels.ptx");
use crate::quant::Bytes;

pub struct Buf(CudaSlice<u8>);

pub struct Backend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: HashMap<K, CudaFunction>,
}

fn err(e: impl std::fmt::Debug) -> anyhow::Error {
    anyhow!("{e:?}")
}

impl Backend {
    pub fn new() -> Result<Self> {
        // The loaders panic when the libraries are missing; that only means no CUDA here.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(0)).map_err(|_| anyhow!("no CUDA driver"))?.map_err(err)?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(err)?;
        let mut kernels = HashMap::new();
        for k in K::ALL {
            kernels.insert(k, module.load_function(k.name()).map_err(err)?);
        }
        let stream = ctx.default_stream();
        Ok(Backend { ctx, stream, kernels })
    }

    pub fn name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "NVIDIA GPU".into())
    }

    /// Free memory on the card now.
    pub fn memory(&self) -> u64 {
        self.ctx.mem_get_info().map(|(free, _)| free as u64).unwrap_or(0)
    }

    pub fn unified(&self) -> bool {
        false
    }

    pub fn weights(&self, data: &Bytes) -> Result<Buf> {
        Ok(Buf(self.stream.clone_htod(&data[..]).map_err(err)?))
    }

    pub fn upload(&self, data: &[f32]) -> Result<Buf> {
        let bytes: Vec<u8> = if data.is_empty() { vec![0; 4] } else { data.iter().flat_map(|v| v.to_le_bytes()).collect() };
        Ok(Buf(self.stream.clone_htod(&bytes).map_err(err)?))
    }

    pub fn zeros(&self, floats: usize) -> Result<Buf> {
        Ok(Buf(self.stream.alloc_zeros::<u8>(floats.max(1) * 4).map_err(err)?))
    }

    pub fn read(&self, b: &Buf, floats: usize) -> Result<Vec<f32>> {
        let bytes = self.stream.clone_dtoh(&b.0).map_err(err)?;
        if bytes.len() < floats * 4 {
            bail!("reading {floats} floats from a {}-byte buffer", bytes.len());
        }
        Ok(bytes[..floats * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn run(&self, cmds: &[Cmd]) -> Result<()> {
        for c in cmds {
            let mut l = self.stream.launch_builder(&self.kernels[&c.k]);
            for a in &c.args {
                match a {
                    Arg::B(b) => l.arg(&b.0),
                    Arg::U(v) => l.arg(v),
                    Arg::F(v) => l.arg(v),
                };
            }
            let cfg = LaunchConfig { grid_dim: (c.grid.0, c.grid.1, 1), block_dim: (c.threads, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: every kernel's arguments are listed in its entry point's order, and
            // the grid sizes keep every index inside the buffers it reads and writes.
            unsafe { l.launch(cfg) }.map_err(err)?;
        }
        self.stream.synchronize().map_err(err)
    }
}
