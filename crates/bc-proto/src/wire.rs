//! Messages between the Windows client and the Mac agent.
//!
//! Each message is CBOR, sent as a frame: a 4-byte big-endian length, then the body.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::spec::JobSpec;

/// Bumped on any incompatible change to these messages.
pub const PROTOCOL_VERSION: u32 = 2;

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
    /// Runs as a service: no timeout, and it keeps running when the client disconnects.
    #[serde(default)]
    pub service: bool,
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
            service: spec.job.service,
            archive_bytes,
        }
    }
}

/// What a job used on the Mac. CPU time and peak memory cover the job's process and
/// every process it waited for, such as the compilers a build tool starts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobUsage {
    pub wall_ms: u64,
    pub user_ms: u64,
    pub system_ms: u64,
    /// Largest resident memory of any single process in the job.
    pub max_rss_bytes: u64,
}

impl JobUsage {
    /// "CPU 12.4 s (3.1 cores) · peak memory 210 MB · ran 4.0 s", for people.
    pub fn summary(&self) -> String {
        let cpu_ms = self.user_ms + self.system_ms;
        // Short jobs are common; "0.0 s" would hide what they used.
        let secs = |ms: u64| if ms < 1000 { format!("{ms} ms") } else { format!("{:.1} s", ms as f64 / 1000.0) };
        let cores = if self.wall_ms > 0 {
            format!(" ({:.1} cores)", cpu_ms as f64 / self.wall_ms as f64)
        } else {
            String::new()
        };
        format!(
            "CPU {}{cores} · peak memory {:.0} MB · ran {}",
            secs(cpu_ms),
            self.max_rss_bytes as f64 / (1 << 20) as f64,
            secs(self.wall_ms),
        )
    }
}

/// A service running on the Mac, as a client or the Mac app sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub job_id: String,
    pub argv: Vec<String>,
    /// Unix seconds.
    pub started_at: u64,
    /// The device that started it; only that device can attach to it or stop it.
    pub owner: String,
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
    /// Stops the job. For a service, closing the connection instead leaves it running.
    Cancel,
    /// Streams a running service's recent and new output, like a fresh `Submit` would.
    Attach {
        job_id: String,
    },
    /// Stops a service; answered with `Accepted` or `Rejected`.
    Stop {
        job_id: String,
    },
    /// Answered with `Services`: the ones this device started.
    ListServices,
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
        /// Missing from agents that predate usage reporting, so it needs no protocol bump.
        #[serde(default)]
        usage: Option<JobUsage>,
    },
    Failed {
        reason: String,
    },
    Services(Vec<ServiceInfo>),
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
                usage: Some(JobUsage { wall_ms: 1, user_ms: 2, system_ms: 3, max_rss_bytes: 4 }),
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
    fn usage_is_compatible_both_ways() {
        // The message as agents and clients without usage reporting define it.
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        enum OldAgentMsg {
            Exited { code: Option<i32>, timed_out: bool },
        }
        let old = OldAgentMsg::Exited { code: Some(1), timed_out: false };
        let new: AgentMsg = decode_body(&encode(&old).unwrap()[4..]).unwrap();
        assert_eq!(new, AgentMsg::Exited { code: Some(1), timed_out: false, usage: None });

        let new = AgentMsg::Exited {
            code: Some(1),
            timed_out: false,
            usage: Some(JobUsage::default()),
        };
        let old: OldAgentMsg = decode_body(&encode(&new).unwrap()[4..]).unwrap();
        assert_eq!(old, OldAgentMsg::Exited { code: Some(1), timed_out: false });
    }

    #[test]
    fn summarizes_usage() {
        let u = JobUsage { wall_ms: 4000, user_ms: 11_000, system_ms: 1_400, max_rss_bytes: 210 << 20 };
        assert_eq!(u.summary(), "CPU 12.4 s (3.1 cores) · peak memory 210 MB · ran 4.0 s");
        let quick = JobUsage { wall_ms: 70, user_ms: 23, system_ms: 17, max_rss_bytes: 14 << 20 };
        assert_eq!(quick.summary(), "CPU 40 ms (0.6 cores) · peak memory 14 MB · ran 70 ms");
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
