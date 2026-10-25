//! Messages between the Windows client and the Mac agent.
//!
//! Each message is CBOR, sent as a frame: a 4-byte big-endian length, then the body.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::spec::JobSpec;

/// Bumped on any incompatible change to these messages.
pub const PROTOCOL_VERSION: u32 = 1;

/// Largest frame either side accepts.
pub const MAX_FRAME: usize = 16 << 20;

#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("frame of {0} bytes is over the {MAX_FRAME} byte limit")]
    TooLarge(usize),
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
}

/// The job as the agent runs it: resolved from a [`JobSpec`] on the client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRequest {
    pub job_id: String,
    pub argv: Vec<String>,
    /// Relative to the uploaded project root, with `/` separators.
    pub workdir: String,
    pub env: BTreeMap<String, String>,
    pub timeout_secs: Option<u64>,
    /// Size of the project archive that follows in `Upload` messages.
    pub archive_bytes: u64,
}

impl JobRequest {
    pub fn from_spec(job_id: String, spec: &JobSpec, archive_bytes: u64) -> Self {
        JobRequest {
            job_id,
            argv: spec.job.argv.clone(),
            workdir: spec.job.workdir.replace('\\', "/"),
            env: spec.job.env.clone(),
            // `validate` already checked the timeout parses.
            timeout_secs: spec.timeout().ok().flatten().map(|d| d.as_secs()),
            archive_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    Hello {
        protocol: u32,
    },
    Submit(JobRequest),
    /// Part of the project archive, in order.
    Upload {
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    UploadDone,
    Cancel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentMsg {
    Hello {
        protocol: u32,
    },
    Accepted {
        job_id: String,
    },
    Rejected {
        reason: String,
    },
    Started,
    Output {
        stream: OutputStream,
        /// Increases by one for each `Output` message in a job.
        seq: u64,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    /// `code` is `None` when the process was killed by a signal or the timeout.
    Exited {
        code: Option<i32>,
        timed_out: bool,
    },
    Failed {
        reason: String,
    },
}

pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, WireError> {
    let mut body = Vec::new();
    ciborium::into_writer(msg, &mut body).map_err(|e| WireError::Encode(e.to_string()))?;
    if body.len() > MAX_FRAME {
        return Err(WireError::TooLarge(body.len()));
    }
    let mut frame = Vec::with_capacity(4 + body.len());
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}

pub fn decode_body<T: DeserializeOwned>(body: &[u8]) -> Result<T, WireError> {
    ciborium::from_reader(body).map_err(|e| WireError::Decode(e.to_string()))
}

/// Reads the length prefix and checks it against [`MAX_FRAME`].
pub fn frame_len(prefix: [u8; 4]) -> Result<usize, WireError> {
    let len = u32::from_be_bytes(prefix) as usize;
    if len > MAX_FRAME {
        return Err(WireError::TooLarge(len));
    }
    Ok(len)
}

pub fn write_frame<T: Serialize>(w: &mut impl Write, msg: &T) -> Result<(), WireError> {
    w.write_all(&encode(msg)?)?;
    Ok(())
}

pub fn read_frame<T: DeserializeOwned>(r: &mut impl Read) -> Result<T, WireError> {
    let mut prefix = [0u8; 4];
    r.read_exact(&mut prefix)?;
    let mut body = vec![0u8; frame_len(prefix)?];
    r.read_exact(&mut body)?;
    decode_body(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn round_trips_messages() {
        let spec = JobSpec::from_argv(vec!["cargo".into(), "test".into()]).unwrap();
        let client = [
            ClientMsg::Hello {
                protocol: PROTOCOL_VERSION,
            },
            ClientMsg::Submit(JobRequest::from_spec("j1".into(), &spec, 3)),
            ClientMsg::Upload {
                data: vec![1, 2, 3],
            },
            ClientMsg::UploadDone,
        ];
        let agent = [
            AgentMsg::Started,
            AgentMsg::Output {
                stream: OutputStream::Stderr,
                seq: 7,
                data: b"warning\n".to_vec(),
            },
            AgentMsg::Exited {
                code: Some(0),
                timed_out: false,
            },
        ];

        let mut buf = Vec::new();
        for m in &client {
            write_frame(&mut buf, m).unwrap();
        }
        for m in &agent {
            write_frame(&mut buf, m).unwrap();
        }

        let mut r = Cursor::new(buf);
        for m in &client {
            assert_eq!(&read_frame::<ClientMsg>(&mut r).unwrap(), m);
        }
        for m in &agent {
            assert_eq!(&read_frame::<AgentMsg>(&mut r).unwrap(), m);
        }
    }

    #[test]
    fn bytes_are_compact() {
        // serde_bytes keeps a 1 KiB chunk close to 1 KiB on the wire.
        let frame = encode(&ClientMsg::Upload {
            data: vec![0xAB; 1024],
        })
        .unwrap();
        assert!(frame.len() < 1024 + 32, "frame was {} bytes", frame.len());
    }

    #[test]
    fn rejects_oversized_frame() {
        let prefix = ((MAX_FRAME + 1) as u32).to_be_bytes();
        assert!(matches!(frame_len(prefix), Err(WireError::TooLarge(_))));
    }

    #[test]
    fn normalizes_windows_workdir() {
        let spec = JobSpec::from_toml(
            "version = 1\n[job]\nargv = [\"make\"]\nworkdir = \"sub\\\\dir\"\ntimeout = \"90s\"\n",
        )
        .unwrap();
        let req = JobRequest::from_spec("j".into(), &spec, 0);
        assert_eq!(req.workdir, "sub/dir");
        assert_eq!(req.timeout_secs, Some(90));
    }
}
