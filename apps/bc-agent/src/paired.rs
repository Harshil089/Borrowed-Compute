//! Devices allowed to send jobs.
//!
//! A local file with one endpoint id per line, checked on every connection. When the
//! agent is signed in, it overwrites this file with the cloud's approved pairings; when
//! not, `bc-agent allow` edits it by hand for development.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bc_net::EndpointId;

#[derive(Debug, Clone)]
pub struct PairedDevices {
    path: PathBuf,
}

impl PairedDevices {
    pub fn new(path: PathBuf) -> Self {
        PairedDevices { path }
    }

    /// Read on every check, so `bc-agent allow` takes effect without a restart.
    pub fn load(&self) -> Result<BTreeSet<EndpointId>> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", self.path.display())),
        };
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.parse().with_context(|| format!("bad device id {l:?}")))
            .collect()
    }

    pub fn contains(&self, id: &EndpointId) -> bool {
        self.load().map(|set| set.contains(id)).unwrap_or(false)
    }

    pub fn add(&self, id: EndpointId) -> Result<()> {
        let mut set = self.load()?;
        set.insert(id);
        self.write(&set)
    }

    pub fn remove(&self, id: &EndpointId) -> Result<bool> {
        let mut set = self.load()?;
        let removed = set.remove(id);
        self.write(&set)?;
        Ok(removed)
    }

    /// Replaces the whole list, as when syncing from the cloud.
    pub fn replace(&self, set: &BTreeSet<EndpointId>) -> Result<()> {
        if self.load().ok().as_ref() == Some(set) {
            return Ok(());
        }
        self.write(set)
    }

    fn write(&self, set: &BTreeSet<EndpointId>) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let body: String = set.iter().map(|id| format!("{id}\n")).collect();
        std::fs::write(&self.path, body).with_context(|| format!("writing {}", self.path.display()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
