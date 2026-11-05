//! Where weights come from: the model's published file, read by byte range straight into
//! memory. Nothing is written to disk, so when the stage holding them ends they are gone.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::gguf::{self, Header, NeedMore, TensorInfo};
use crate::quant::QTensor;

/// Ranges this far apart or closer are fetched as one request.
const MERGE_GAP: u64 = 1 << 20;
/// Requests are cut into pieces of this size and fetched in parallel.
const PIECE: u64 = 32 << 20;
const PARALLEL: usize = 6;

#[derive(Debug, Clone)]
pub enum Source {
    /// https anywhere; plain http only on this computer.
    Http(String),
    /// A local file, for the coordinator's own models and tests. Never accepted from
    /// another device.
    File(std::path::PathBuf),
}

impl Source {
    /// A model location given by a person on this device.
    pub fn parse_local(s: &str) -> Result<Self> {
        if s.starts_with("http://") || s.starts_with("https://") {
            return Source::parse_remote(s);
        }
        Ok(Source::File(s.into()))
    }

    /// A model location sent by another device: web addresses only, and plain http only
    /// to this computer, so a device cannot point a lender at files or its local network.
    pub fn parse_remote(s: &str) -> Result<Self> {
        let url = reqwest::Url::parse(s).with_context(|| format!("{s:?} is not a web address"))?;
        let host = url.host_str().unwrap_or_default().trim_start_matches('[').trim_end_matches(']');
        let loopback = host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
        match url.scheme() {
            "https" => Ok(Source::Http(s.into())),
            "http" if loopback => Ok(Source::Http(s.into())),
            _ => bail!("models must come from an https address"),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Source::Http(u) => u.clone(),
            Source::File(p) => p.display().to_string(),
        }
    }

    pub async fn read(&self, start: u64, end: u64) -> Result<Vec<u8>> {
        match self {
            Source::File(p) => {
                let p = p.clone();
                tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut f = std::fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?;
                    let len = f.metadata()?.len();
                    f.seek(SeekFrom::Start(start))?;
                    let mut buf = vec![0; (end.min(len) - start.min(len)) as usize];
                    f.read_exact(&mut buf)?;
                    Ok(buf)
                })
                .await?
            }
            Source::Http(url) => {
                let resp = client()
                    .get(url)
                    .header(reqwest::header::RANGE, format!("bytes={start}-{}", end - 1))
                    .send()
                    .await
                    .with_context(|| format!("fetching {url}"))?;
                let status = resp.status();
                if status.as_u16() == 416 {
                    return Ok(Vec::new());
                }
                if status.as_u16() != 206 {
                    bail!("{url} answered {status} to a range request; it must serve byte ranges");
                }
                Ok(resp.bytes().await?.to_vec())
            }
        }
    }

    /// The GGUF header, reading more of the file until it fits.
    pub async fn header(&self) -> Result<Header> {
        let mut len = 1u64 << 20;
        loop {
            let bytes = self.read(0, len).await?;
            match gguf::parse(&bytes) {
                Ok(h) => return Ok(h),
                Err(e) if e.downcast_ref::<NeedMore>().is_some() && (bytes.len() as u64) == len && len < 1 << 30 => len *= 4,
                Err(e) => return Err(e),
            }
        }
    }

    /// Fetches `wanted` tensors into memory: adjacent ranges merged, big ones split, a few
    /// requests at a time. `progress` gets bytes done and bytes in all.
    pub async fn tensors(&self, wanted: &[&TensorInfo], progress: impl Fn(u64, u64) + Send + Sync) -> Result<HashMap<String, Arc<QTensor>>> {
        let mut ranges: Vec<(u64, u64)> = wanted.iter().map(|t| (t.range.start, t.range.end)).collect();
        ranges.sort();
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for (s, e) in ranges {
            match spans.last_mut() {
                Some(last) if s <= last.1 + MERGE_GAP => last.1 = last.1.max(e),
                _ => spans.push((s, e)),
            }
        }
        let pieces: Vec<(u64, u64)> = spans
            .iter()
            .flat_map(|&(s, e)| (s..e).step_by(PIECE as usize).map(move |p| (p, (p + PIECE).min(e))))
            .collect();
        let total: u64 = pieces.iter().map(|(s, e)| e - s).sum();
        let done = std::sync::atomic::AtomicU64::new(0);
        let mut fetched: Vec<(u64, Vec<u8>)> = Vec::with_capacity(pieces.len());
        for batch in pieces.chunks(PARALLEL) {
            let results = futures_join(batch.iter().map(|&(s, e)| async move { self.read(s, e).await.map(|b| (s, e, b)) })).await;
            for r in results {
                let (s, e, b) = r?;
                if b.len() as u64 != e - s {
                    bail!("the model file ended early");
                }
                progress(done.fetch_add(e - s, std::sync::atomic::Ordering::Relaxed) + (e - s), total);
                fetched.push((s, b));
            }
        }
        fetched.sort_by_key(|(s, _)| *s);
        let slice = |start: u64, end: u64| -> Result<Vec<u8>> {
            let mut out = Vec::with_capacity((end - start) as usize);
            for (s, b) in &fetched {
                let e = s + b.len() as u64;
                if e <= start || *s >= end {
                    continue;
                }
                let from = start.max(*s) - s;
                let to = end.min(e) - s;
                out.extend_from_slice(&b[from as usize..to as usize]);
            }
            if out.len() as u64 != end - start {
                bail!("missing bytes {start}..{end}");
            }
            Ok(out)
        };
        let mut map = HashMap::new();
        for t in wanted {
            let (cols, rows) = t.shape2();
            let data = slice(t.range.start, t.range.end)?;
            map.insert(t.name.clone(), Arc::new(QTensor::new(t.ty, rows, cols, data).with_context(|| format!("tensor {}", t.name))?));
        }
        Ok(map)
    }
}

fn client() -> reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| reqwest::Client::builder().user_agent("borrowed-compute").build().expect("http client"))
        .clone()
}

/// Polls the futures together, so each one's request runs while the others wait.
async fn futures_join<F: std::future::Future>(futs: impl Iterator<Item = F>) -> Vec<F::Output> {
    let mut pinned: Vec<std::pin::Pin<Box<F>>> = futs.map(Box::pin).collect();
    let mut results: Vec<Option<F::Output>> = (0..pinned.len()).map(|_| None).collect();
    std::future::poll_fn(|cx| {
        let mut pending = false;
        for (i, f) in pinned.iter_mut().enumerate() {
            if results[i].is_none() {
                match f.as_mut().poll(cx) {
                    std::task::Poll::Ready(v) => results[i] = Some(v),
                    std::task::Poll::Pending => pending = true,
                }
            }
        }
        if pending {
            std::task::Poll::Pending
        } else {
            std::task::Poll::Ready(())
        }
    })
    .await;
    results.into_iter().map(|r| r.unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_sources_are_web_addresses_only() {
        assert!(Source::parse_remote("https://huggingface.co/x/y.gguf").is_ok());
        assert!(Source::parse_remote("http://127.0.0.1:9000/m.gguf").is_ok());
        assert!(Source::parse_remote("http://192.168.1.4/m.gguf").is_err());
        assert!(Source::parse_remote("/etc/passwd").is_err());
        assert!(Source::parse_remote("file:///etc/passwd").is_err());
    }

    #[tokio::test]
    async fn fetches_only_the_tensors_asked_for() {
        let file = crate::model::tests::tiny_model(3, false);
        let dir = std::env::temp_dir().join(format!("bc-infer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        std::fs::write(&path, &file).unwrap();
        let src = Source::File(path);
        let h = src.header().await.unwrap();
        let wanted = crate::model::needed(&h, &(1..2), false, false);
        let map = src.tensors(&wanted, |_, _| {}).await.unwrap();
        assert!(map.keys().all(|k| k.starts_with("blk.1.")) && map.len() == wanted.len());
        let t = h.tensor("blk.1.attn_q.weight").unwrap();
        assert_eq!(map["blk.1.attn_q.weight"].data, file[t.range.start as usize..t.range.end as usize]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
