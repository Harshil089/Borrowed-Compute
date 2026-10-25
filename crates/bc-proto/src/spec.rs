//! The `bc.toml` job spec.
//!
//! Unknown keys are rejected so that typos fail before a job is sent.

use std::collections::BTreeMap;
use std::path::{Component, Path};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The only spec version this build understands.
pub const SPEC_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("invalid bc.toml: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("unsupported spec version {0}, expected {SPEC_VERSION}")]
    Version(u32),
    #[error("job.argv must not be empty")]
    EmptyArgv,
    #[error("job.argv[0] must not be empty")]
    EmptyProgram,
    #[error("job.workdir must be a relative path inside the project: {0}")]
    Workdir(String),
    #[error("invalid job.timeout {value:?}: {reason}")]
    Timeout { value: String, reason: String },
    #[error("invalid resources.memory {0:?}, expected a value like \"512MiB\" or \"16GiB\"")]
    Memory(String),
    #[error("resources.cpu must be at least 1")]
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub version: u32,
    pub job: Job,
    /// Phase two: toolchains the agent provisions. Kept as raw TOML until then.
    #[serde(default)]
    pub toolchain: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub files: Files,
    #[serde(default)]
    pub outputs: Outputs,
    #[serde(default)]
    pub resources: Resources,
    pub ml: Option<Ml>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    /// Program and arguments. Never a shell string.
    pub argv: Vec<String>,
    #[serde(default = "default_workdir")]
    pub workdir: String,
    pub timeout: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Files {
    /// Glob patterns to leave out, on top of `.gitignore`.
    #[serde(default)]
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Outputs {
    /// Paths to return to the client after the job.
    #[serde(default)]
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub cpu: Option<u32>,
    pub memory: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Ml {
    pub backend: MlBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MlBackend {
    Cpu,
    Mps,
    Mlx,
}

fn default_workdir() -> String {
    ".".to_string()
}

impl JobSpec {
    /// Parses and validates a `bc.toml` document.
    pub fn from_toml(text: &str) -> Result<Self, SpecError> {
        let spec: JobSpec = toml::from_str(text)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Builds a spec from a command line, as in `bc run -- cargo test`.
    pub fn from_argv(argv: Vec<String>) -> Result<Self, SpecError> {
        let spec = JobSpec {
            version: SPEC_VERSION,
            job: Job {
                argv,
                workdir: default_workdir(),
                timeout: None,
                env: BTreeMap::new(),
            },
            toolchain: BTreeMap::new(),
            files: Files::default(),
            outputs: Outputs::default(),
            resources: Resources::default(),
            ml: None,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        if self.version != SPEC_VERSION {
            return Err(SpecError::Version(self.version));
        }
        match self.job.argv.first() {
            None => return Err(SpecError::EmptyArgv),
            Some(program) if program.is_empty() => return Err(SpecError::EmptyProgram),
            Some(_) => {}
        }
        validate_relative(&self.job.workdir)?;
        self.timeout()?;
        self.memory_bytes()?;
        if self.resources.cpu == Some(0) {
            return Err(SpecError::Cpu);
        }
        Ok(())
    }

    pub fn timeout(&self) -> Result<Option<Duration>, SpecError> {
        self.job
            .timeout
            .as_deref()
            .map(|value| {
                humantime::parse_duration(value).map_err(|e| SpecError::Timeout {
                    value: value.to_string(),
                    reason: e.to_string(),
                })
            })
            .transpose()
    }

    pub fn memory_bytes(&self) -> Result<Option<u64>, SpecError> {
        self.resources
            .memory
            .as_deref()
            .map(|value| parse_bytes(value).ok_or_else(|| SpecError::Memory(value.to_string())))
            .transpose()
    }
}

fn validate_relative(path: &str) -> Result<(), SpecError> {
    // Windows-style separators are normalized so `sub\dir` and `C:\x` are checked too.
    let normalized = path.replace('\\', "/");
    let bad = normalized.contains(':')
        || Path::new(&normalized)
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir));
    if bad {
        return Err(SpecError::Workdir(path.to_string()));
    }
    Ok(())
}

/// Parses sizes like `512MiB`, `16GiB`, `1GB` or a plain byte count.
fn parse_bytes(value: &str) -> Option<u64> {
    let value = value.trim();
    let split = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split);
    let number: u64 = number.parse().ok()?;
    let factor: u64 = match unit.trim() {
        "" | "B" => 1,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        _ => return None,
    };
    number.checked_mul(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
version = 1

[job]
argv    = ["cargo", "test", "--release"]
workdir = "."
timeout = "30m"
env     = { RUST_BACKTRACE = "1" }

[toolchain]
rust = "1.82"

[files]
exclude = ["target/**", ".git/**"]

[outputs]
paths = ["target/release/myapp"]

[resources]
cpu    = 8
memory = "16GiB"

[ml]
backend = "mps"
"#;

    #[test]
    fn parses_full_spec() {
        let spec = JobSpec::from_toml(FULL).unwrap();
        assert_eq!(spec.job.argv, ["cargo", "test", "--release"]);
        assert_eq!(spec.timeout().unwrap(), Some(Duration::from_secs(30 * 60)));
        assert_eq!(spec.memory_bytes().unwrap(), Some(16 << 30));
        assert_eq!(spec.resources.cpu, Some(8));
        assert_eq!(spec.ml.unwrap().backend, MlBackend::Mps);
        assert_eq!(spec.job.env["RUST_BACKTRACE"], "1");
        assert!(spec.toolchain.contains_key("rust"));
    }

    #[test]
    fn minimal_spec_uses_defaults() {
        let spec = JobSpec::from_toml("version = 1\n[job]\nargv = [\"make\"]\n").unwrap();
        assert_eq!(spec.job.workdir, ".");
        assert!(spec.files.exclude.is_empty());
        assert!(spec.outputs.paths.is_empty());
        assert_eq!(spec.timeout().unwrap(), None);
    }

    #[test]
    fn rejects_unknown_keys() {
        let err = JobSpec::from_toml("version = 1\n[job]\nargv = [\"make\"]\nargs = []\n");
        assert!(matches!(err, Err(SpecError::Parse(_))));
    }

    #[test]
    fn rejects_wrong_version() {
        let err = JobSpec::from_toml("version = 2\n[job]\nargv = [\"make\"]\n");
        assert!(matches!(err, Err(SpecError::Version(2))));
    }

    #[test]
    fn rejects_empty_argv() {
        assert!(matches!(
            JobSpec::from_argv(vec![]),
            Err(SpecError::EmptyArgv)
        ));
        assert!(matches!(
            JobSpec::from_argv(vec![String::new()]),
            Err(SpecError::EmptyProgram)
        ));
    }

    #[test]
    fn rejects_workdir_outside_project() {
        for bad in ["..", "../x", "a/../../b", "/abs", "C:\\x", "\\\\server\\share"] {
            let spec = format!("version = 1\n[job]\nargv = [\"make\"]\nworkdir = {bad:?}\n");
            assert!(
                matches!(JobSpec::from_toml(&spec), Err(SpecError::Workdir(_))),
                "{bad} should be rejected"
            );
        }
        for good in [".", "sub", "sub/dir", "sub\\dir", "./sub"] {
            let spec = format!("version = 1\n[job]\nargv = [\"make\"]\nworkdir = {good:?}\n");
            assert!(JobSpec::from_toml(&spec).is_ok(), "{good} should be accepted");
        }
    }

    #[test]
    fn rejects_bad_timeout_and_memory() {
        let t = JobSpec::from_toml("version = 1\n[job]\nargv = [\"make\"]\ntimeout = \"soon\"\n");
        assert!(matches!(t, Err(SpecError::Timeout { .. })));
        let m = JobSpec::from_toml(
            "version = 1\n[job]\nargv = [\"make\"]\n[resources]\nmemory = \"lots\"\n",
        );
        assert!(matches!(m, Err(SpecError::Memory(_))));
    }

    #[test]
    fn parses_byte_sizes() {
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("512MiB"), Some(512 << 20));
        assert_eq!(parse_bytes("2GB"), Some(2_000_000_000));
        assert_eq!(parse_bytes("1 GiB"), Some(1 << 30));
        assert_eq!(parse_bytes("1TiB"), None);
        assert_eq!(parse_bytes("GiB"), None);
    }
}
