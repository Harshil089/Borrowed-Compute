//! Frames of the mesh protocol (`MESH_ALPN`), sent with `bc_net::send_msg`.

use std::ops::Range;

use anyhow::{bail, Result};
use iroh::EndpointAddr;
use serde::{Deserialize, Serialize};

use crate::codec::{self, Encoding};
use crate::ticket::Ticket;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshMsg {
    // Probing a link and the device behind it.
    Ping { nonce: u64 },
    Pong { nonce: u64 },
    Report,
    Device(DeviceReport),
    Bulk {
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    BulkDone,
    /// Bytes received and microseconds from the first to the last.
    BulkAck { bytes: u64, micros: u64 },
    // Running a pipeline.
    Setup(StageSetup),
    /// From the previous stage, on its own connection.
    Feed { ticket: Ticket },
    Ready,
    Batch(Batch),
    /// No more batches; passes down the pipeline once everything before it is done.
    End,
    Error { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceReport {
    pub name: String,
    pub os: String,
    pub cores: u32,
    pub mem_total: u64,
    pub mem_available: u64,
    /// Measured memory copy bandwidth, bytes a second.
    pub mem_bw: f64,
    /// True when this device can run pipeline stages (it has an inference engine).
    pub engine: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSetup {
    pub session: u64,
    pub index: u16,
    pub count: u16,
    pub model: String,
    pub layers: Range<u32>,
    /// The stage to feed, with the ticket that lets this device feed it.
    pub next: Option<(EndpointAddr, Ticket)>,
    pub encoding: Encoding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    /// Set by the coordinator; stages keep it, so results can be matched to requests.
    pub seq: u64,
    pub payload: Payload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Payload {
    Tokens(Vec<u32>),
    Hidden {
        rows: u32,
        cols: u32,
        encoding: Encoding,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
}

/// What a stage reads and writes: token ids at the ends of a model, hidden states
/// (one row per token) in between.
#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    Tokens(Vec<u32>),
    Hidden(Tensor),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Payload {
    pub fn encode(activation: &Activation, encoding: Encoding) -> Payload {
        match activation {
            Activation::Tokens(t) => Payload::Tokens(t.clone()),
            Activation::Hidden(t) => Payload::Hidden {
                rows: t.rows as u32,
                cols: t.cols as u32,
                encoding,
                data: codec::encode(&t.data, encoding),
            },
        }
    }

    pub fn decode(self) -> Result<Activation> {
        match self {
            Payload::Tokens(t) => Ok(Activation::Tokens(t)),
            Payload::Hidden { rows, cols, encoding, data } => {
                let Some(values) = (rows as usize).checked_mul(cols as usize) else {
                    bail!("{rows}x{cols} is too large");
                };
                let data = codec::decode(&data, values, encoding)?;
                Ok(Activation::Hidden(Tensor { rows: rows as usize, cols: cols as usize, data }))
            }
        }
    }
}
