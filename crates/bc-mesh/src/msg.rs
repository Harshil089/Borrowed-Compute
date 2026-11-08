//! Frames of the mesh protocol (`MESH_ALPN`), sent with `bc_net::send_msg`.

use std::ops::Range;

use anyhow::{bail, Result};
use iroh::EndpointAddr;
use serde::{Deserialize, Serialize};

use crate::codec;
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
    /// Coordinator to device A: measure your link to `addr` (device B).
    ProbePeer { addr: EndpointAddr, ticket: Ticket },
    /// Device A to device B: the coordinator that signed the ticket, which B must trust.
    PeerHello { coordinator: iroh::EndpointId, ticket: Ticket },
    /// Device A to the coordinator: the link it measured.
    PeerLink { rtt_ms: f64, up_bytes_s: f64, via_relay: bool },
    // Running a pipeline.
    Setup(StageSetup),
    /// From the previous stage, on its own connection.
    Feed { ticket: Ticket },
    /// A stage's weights so far while it loads, bytes fetched of all it needs: sent to
    /// the coordinator before `Ready`, so a long download shows as progress, not a hang.
    Loading { done: u64, total: u64 },
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
    /// The GPU its engine runs layers on, if it has one.
    #[serde(default)]
    pub gpu: Option<GpuReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuReport {
    pub name: String,
    /// Memory for weights: free memory on a discrete card. On unified memory (Apple) the
    /// GPU reads the same RAM the CPU does, so the device's free memory is what counts.
    pub mem_bytes: u64,
    pub unified: bool,
    /// Bytes of weights a second it reads in a decoding product, as measured.
    pub bw: f64,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    /// Set by the coordinator; stages keep it, so results can be matched to requests.
    pub seq: u64,
    pub payload: Payload,
    #[serde(default)]
    pub meta: Meta,
}

/// What a batch is part of. Stages keep a key/value cache per `stream` (one conversation)
/// and cut it back to `pos` before running the batch, so a rejected speculative guess or a
/// replayed request never leaves stale entries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub stream: u64,
    /// Position of the batch's first row in the stream.
    pub pos: u32,
    /// The last stage returns a token for every row (speculative checking) instead of
    /// only the last.
    pub all_rows: bool,
    pub sampling: bc_infer::Sampling,
    /// Rows (tokens) in the batch, as sent.
    pub rows: u32,
    /// Milliseconds each stage spent computing, appended stage by stage: what live
    /// balancing measures.
    pub stage_ms: Vec<f32>,
    /// Which run of speculative guesses the batch belongs to. A newer epoch for the same
    /// stream means the coordinator found a wrong guess, so every older batch still
    /// queued at a stage is worthless and is passed on without computing.
    #[serde(default)]
    pub epoch: u64,
    /// Passed on without computing (see `epoch`); the payload is empty.
    #[serde(default)]
    pub skipped: bool,
    /// The rows form a tree: each row's parent row, or `bc_infer::ROOT` (see
    /// `Part::forward_tree`). `None` is a line.
    #[serde(default)]
    pub tree: Option<Vec<u32>>,
    /// Before this batch: of the rows the last tree pass placed from position `.0`, keep
    /// only `.1` (the accepted path), in order (`Kv::keep`).
    #[serde(default)]
    pub keep: Option<(u32, Vec<u32>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Payload {
    Tokens(Vec<u32>),
    Hidden {
        rows: u32,
        cols: u32,
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
    pub fn encode(activation: &Activation) -> Payload {
        match activation {
            Activation::Tokens(t) => Payload::Tokens(t.clone()),
            Activation::Hidden(t) => Payload::Hidden {
                rows: t.rows as u32,
                cols: t.cols as u32,
                data: codec::encode(&t.data),
            },
        }
    }

    pub fn decode(self) -> Result<Activation> {
        match self {
            Payload::Tokens(t) => Ok(Activation::Tokens(t)),
            Payload::Hidden { rows, cols, data } => {
                let Some(values) = (rows as usize).checked_mul(cols as usize) else {
                    bail!("{rows}x{cols} is too large");
                };
                let data = codec::decode(&data, values)?;
                Ok(Activation::Hidden(Tensor { rows: rows as usize, cols: cols as usize, data }))
            }
        }
    }
}
