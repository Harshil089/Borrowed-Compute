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
const PIECE: u64 = 8 << 20;
const PARALLEL: usize = 6;
/// Tries per piece before a download gives up.
const PIECE_TRIES: u32 = 5;
/// A piece still under MIN_BYTES_PER_SEC this many seconds in is dropped and tried again.
const SLOW_AFTER_SECS: f64 = 20.0;
const MIN_BYTES_PER_SEC: f64 = 128.0 * 1024.0;
/// Download progress is reported at least this many bytes apart.
const REPORT_EVERY: u64 = 4 << 20;

#[derive(Debug, Clone)]
pub enum Source {
    /// https anywhere; plain http only on this computer.
    Http(String),
    /// A local file, for the coordinator's own models and tests. Never accepted from
    /// another device.
    File(std::path::PathBuf),
    /// A model published as several files (`…-00001-of-00007.gguf`, and so on): the first
    /// holds the metadata, each holds some of the tensors.
    Split(Vec<Source>),
}

/// The other files of a split model, from the first one's name: `x-00001-of-00003.gguf`
/// gives `x-00001…`, `x-00002…` and `x-00003-of-00003.gguf`.
fn split_parts(s: &str) -> Option<Vec<String>> {
    let stem = s.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (base, first) = head.rsplit_once('-')?;
    if first.len() != 5 || count.len() != 5 || first != "00001" {
        return None;
    }
    let n: usize = count.parse().ok()?;
    (n > 1).then(|| (1..=n).map(|i| format!("{base}-{i:05}-of-{count}.gguf")).collect())
}

impl Source {
    /// A model location given by a person on this device.
    pub fn parse_local(s: &str) -> Result<Self> {
        if let Some(parts) = split_parts(s) {
            return Ok(Source::Split(parts.iter().map(|p| Source::one_local(p)).collect::<Result<_>>()?));
        }
        Source::one_local(s)
    }

    fn one_local(s: &str) -> Result<Self> {
        if s.starts_with("http://") || s.starts_with("https://") {
            return Source::one_remote(s);
        }
        Ok(Source::File(s.into()))
    }

    /// A model location sent by another device: web addresses only, and plain http only
    /// to this computer, so a device cannot point a lender at files or its local network.
    pub fn parse_remote(s: &str) -> Result<Self> {
        if let Some(parts) = split_parts(s) {
            return Ok(Source::Split(parts.iter().map(|p| Source::one_remote(p)).collect::<Result<_>>()?));
        }
        Source::one_remote(s)
    }

    fn one_remote(s: &str) -> Result<Self> {
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
            Source::Split(parts) => parts.first().map(|p| p.label()).unwrap_or_default(),
        }
    }

    pub async fn read(&self, start: u64, end: u64) -> Result<Vec<u8>> {
        self.read_counted(start, end, &|_| {}).await
    }

    /// One piece of a download, tried again on a fresh connection when it fails or
    /// crawls. Bytes of a failed try are taken back from the progress count.
    async fn read_piece(&self, start: u64, end: u64, arrived: &(dyn Fn(i64) + Send + Sync)) -> Result<Vec<u8>> {
        let mut tries = 0;
        loop {
            let got = std::sync::atomic::AtomicU64::new(0);
            let count = |n: u64| {
                got.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                arrived(n as i64);
            };
            match self.read_counted(start, end, &count).await {
                Ok(b) => return Ok(b),
                Err(_) if tries + 1 < PIECE_TRIES => {
                    tries += 1;
                    arrived(-(got.into_inner() as i64));
                }
                Err(e) => return Err(e.context(format!("bytes {start}..{end}, after {PIECE_TRIES} tries"))),
            }
        }
    }

    /// [`Source::read`], telling `arrived` about each piece of the body as it comes in, so
    /// a download of gigabytes shows progress by the megabyte.
    async fn read_counted(&self, start: u64, end: u64, arrived: &(dyn Fn(u64) + Send + Sync)) -> Result<Vec<u8>> {
        match self {
            Source::Split(_) => bail!("a split model is read file by file"),
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
                .inspect(|b| arrived(b.len() as u64))
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
                let mut resp = resp;
                let mut body = Vec::with_capacity((end - start) as usize);
                let started = std::time::Instant::now();
                while let Some(chunk) = resp.chunk().await? {
                    arrived(chunk.len() as u64);
                    body.extend_from_slice(&chunk);
                    // A connection to a slow server trickles but never times out; drop it
                    // and let a fresh one (often to a faster server) take over.
                    let secs = started.elapsed().as_secs_f64();
                    if secs > SLOW_AFTER_SECS && (body.len() as f64) < MIN_BYTES_PER_SEC * secs {
                        bail!("{url} is sending under {} KB/s", MIN_BYTES_PER_SEC as u64 >> 10);
                    }
                }
                Ok(body)
            }
        }
    }

    /// The GGUF header, reading more of the file until it fits. For a split model: the
    /// first file's metadata with every file's tensors, each marked with its file.
    pub async fn header(&self) -> Result<Header> {
        if let Source::Split(parts) = self {
            let headers = futures_join(parts.iter().map(|p| Box::pin(p.header()))).await;
            let mut merged: Option<Header> = None;
            for (i, h) in headers.into_iter().enumerate() {
                let mut h = h.with_context(|| format!("part {} of the model", i + 1))?;
                h.tensors.iter_mut().for_each(|t| t.file = i);
                match merged.as_mut() {
                    None => merged = Some(h),
                    Some(m) => m.tensors.extend(h.tensors),
                }
            }
            return merged.context("a split model with no parts");
        }
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
        if let Source::Split(parts) = self {
            let mut map = HashMap::new();
            for (i, part) in parts.iter().enumerate() {
                let mine: Vec<&TensorInfo> = wanted.iter().copied().filter(|t| t.file == i).collect();
                if !mine.is_empty() {
                    map.extend(part.file_tensors(&mine, &progress).await?);
                }
            }
            return Ok(map);
        }
        self.file_tensors(wanted, &progress).await
    }

    async fn file_tensors(&self, wanted: &[&TensorInfo], progress: &(dyn Fn(u64, u64) + Send + Sync)) -> Result<HashMap<String, Arc<QTensor>>> {
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
        // Bytes as they arrive, reported every REPORT_EVERY and at the end.
        // Negative when a failed try's bytes are taken back.
        let done = std::sync::atomic::AtomicI64::new(0);
        let reported = std::sync::atomic::AtomicU64::new(0);
        let arrived = |n: i64| {
            use std::sync::atomic::Ordering::Relaxed;
            let now = (done.fetch_add(n, Relaxed) + n).max(0) as u64;
            if now.abs_diff(reported.load(Relaxed)) >= REPORT_EVERY || now == total {
                reported.store(now, Relaxed);
                progress(now, total);
            }
        };
        let mut fetched: Vec<(u64, Vec<u8>)> = Vec::with_capacity(pieces.len());
        for batch in pieces.chunks(PARALLEL) {
            let arrived = &arrived;
            let results = futures_join(batch.iter().map(|&(s, e)| async move { self.read_piece(s, e, arrived).await.map(|b| (s, e, b)) })).await;
            for r in results {
                let (s, e, b) = r?;
                if b.len() as u64 != e - s {
                    bail!("the model file ended early");
                }
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
        .get_or_init(|| {
            reqwest::Client::builder()
                .user_agent("borrowed-compute")
                // No bytes for this long means a dead connection, not a slow one.
                .read_timeout(std::time::Duration::from_secs(20))
                .connect_timeout(std::time::Duration::from_secs(20))
                .build()
                .expect("http client")
        })
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
    fn split_names_expand() {
        assert_eq!(
            split_parts("https://h/m/x-Q4-00001-of-00003.gguf").unwrap(),
            ["https://h/m/x-Q4-00001-of-00003.gguf", "https://h/m/x-Q4-00002-of-00003.gguf", "https://h/m/x-Q4-00003-of-00003.gguf"]
        );
        assert!(split_parts("https://h/m/x-00002-of-00003.gguf").is_none(), "only from the first part");
        assert!(split_parts("https://h/m/x.gguf").is_none());
        assert!(matches!(Source::parse_remote("https://h/m/x-00001-of-00002.gguf").unwrap(), Source::Split(p) if p.len() == 2));
    }

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
        let last = std::sync::Mutex::new((0, 0));
        let map = src.tensors(&wanted, |done, total| *last.lock().unwrap() = (done, total)).await.unwrap();
        let (done, total) = last.into_inner().unwrap();
        assert!(total > 0 && done == total, "the last report says everything arrived: {done} of {total}");
        assert!(map.keys().all(|k| k.starts_with("blk.1.")) && map.len() == wanted.len());
        let t = h.tensor("blk.1.attn_q.weight").unwrap();
        assert_eq!(map["blk.1.attn_q.weight"].data[..], file[t.range.start as usize..t.range.end as usize]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
