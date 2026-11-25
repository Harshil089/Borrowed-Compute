//! The run log: everything that happened in a conversation's turns, one JSON line per
//! event in `<home>/runs/<conversation>.jsonl`, appended as it happens. The conversation
//! file holds what the model reads; the log holds what was done, so a turn cut off by a
//! crash or a quit can say which calls may have run, and `bc agent log` can show a run
//! step by step. Large tool outputs are kept beside it (`<conversation>/<call>.txt`).

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEvent {
    /// The person's message starts a turn.
    Turn { at: u64, text: String },
    /// The harness chose what to do before the model (see `route`).
    Route { at: u64, route: String },
    /// One step's answer, as the model wrote it.
    Model { at: u64, text: String, prompt_tokens: usize, cached_tokens: usize, generated: usize },
    /// Written before a call runs: one with no `Finished` was cut off and may have run.
    Started { at: u64, call: String, name: String, arguments: Value, by_harness: bool },
    Finished { at: u64, call: String, ok: bool, text: String },
    /// Older messages were replaced by this summary to fit the context.
    Compacted { at: u64, messages: usize, summary: String },
    Ended { at: u64, reason: String },
    /// An answer failed a check (`check`); `again` when it was written again.
    Check { at: u64, problem: String, again: bool },
}

pub struct RunLog {
    file: PathBuf,
}

impl RunLog {
    pub fn at(home: &Path, conversation: &str) -> Self {
        RunLog { file: home.join("runs").join(format!("{conversation}.jsonl")) }
    }

    /// The calls of the latest turn that finished, by tool name, with their results: what
    /// an answer is checked against (`check`). From the log, not the messages, so trimming
    /// the context to fit changes nothing here.
    pub fn this_turn(&self) -> Vec<(String, String)> {
        let events = self.read();
        let start = events.iter().rposition(|e| matches!(e, RunEvent::Turn { .. })).map_or(0, |i| i + 1);
        let mut names = std::collections::HashMap::new();
        let mut out = Vec::new();
        for e in &events[start..] {
            match e {
                RunEvent::Started { call, name, .. } => {
                    names.insert(call.clone(), name.clone());
                }
                RunEvent::Finished { call, text, .. } => out.push((names.get(call).cloned().unwrap_or_default(), text.clone())),
                _ => {}
            }
        }
        out
    }

    /// Where this conversation's large outputs go.
    pub fn artifacts(&self) -> PathBuf {
        self.file.with_extension("")
    }

    /// Best effort: a full disk must not stop the conversation, only its record.
    pub fn append(&self, event: &RunEvent) {
        let line = match serde_json::to_string(event) {
            Ok(l) => l + "\n",
            Err(_) => return,
        };
        if let Some(dir) = self.file.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).read(true).append(true).open(&self.file) {
            // A line torn by a crash would swallow this one too: start on a fresh line.
            let torn = {
                use std::io::{Read, Seek, SeekFrom};
                let mut last = [0u8];
                f.seek(SeekFrom::End(-1)).is_ok() && f.read_exact(&mut last).is_ok() && last[0] != b'\n'
            };
            let _ = f.write_all(if torn { format!("\n{line}") } else { line }.as_bytes());
            if matches!(event, RunEvent::Ended { .. }) {
                let _ = f.sync_data();
            }
        }
    }

    /// Every event, oldest first. A line cut short by a crash is skipped.
    pub fn read(&self) -> Vec<RunEvent> {
        std::fs::read_to_string(&self.file).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
    }

    /// Calls of the last turn that started and never finished, when that turn never ended:
    /// the app stopped while they ran, so they may or may not have happened.
    pub fn interrupted(&self) -> Vec<(String, Value)> {
        let events = self.read();
        let last = events.iter().rposition(|e| matches!(e, RunEvent::Turn { .. }));
        let Some(start) = last else { return Vec::new() };
        let turn = &events[start..];
        if turn.iter().any(|e| matches!(e, RunEvent::Ended { .. })) {
            return Vec::new();
        }
        turn.iter()
            .filter_map(|e| match e {
                RunEvent::Started { call, name, arguments, .. } if !turn.iter().any(|f| matches!(f, RunEvent::Finished { call: c, .. } if c == call)) => Some((name.clone(), arguments.clone())),
                _ => None,
            })
            .collect()
    }

    /// Closes a turn cut off earlier, so it is reported once.
    pub fn close_interrupted(&self) {
        let events = self.read();
        if events.iter().rposition(|e| matches!(e, RunEvent::Turn { .. })).is_some_and(|s| !events[s..].iter().any(|e| matches!(e, RunEvent::Ended { .. }))) {
            self.append(&RunEvent::Ended { at: super::memory::now(), reason: "cut off".into() });
        }
    }

    pub fn delete(&self) {
        let _ = std::fs::remove_file(&self.file);
        let _ = std::fs::remove_dir_all(self.artifacts());
    }
}

/// A run as lines for a person to read: `bc agent log`.
pub fn describe(events: &[RunEvent]) -> String {
    let clip = |t: &str, n: usize| super::format::clip(&t.replace('\n', " "), n);
    events
        .iter()
        .map(|e| match e {
            RunEvent::Turn { at, text } => format!("\n[{}] person: {}", super::memory::day_of(*at), clip(text, 200)),
            RunEvent::Route { route, .. } => format!("  route: {route}"),
            RunEvent::Model { text, prompt_tokens, cached_tokens, generated, .. } => {
                format!("  model ({generated} tokens written, {cached_tokens} of {prompt_tokens} prompt tokens reused): {}", clip(text, 300))
            }
            RunEvent::Started { name, arguments, by_harness, .. } => format!("  {} {name} {}", if *by_harness { "harness calls" } else { "calls" }, clip(&arguments.to_string(), 200)),
            RunEvent::Finished { ok, text, .. } => format!("    {} {}", if *ok { "->" } else { "-> failed:" }, clip(text, 200)),
            RunEvent::Compacted { messages, summary, .. } => format!("  compacted {messages} older messages: {}", clip(summary, 300)),
            RunEvent::Ended { reason, .. } => format!("  ended: {reason}"),
            RunEvent::Check { problem, again, .. } => format!("  check: {problem}{}", if *again { ", answering again" } else { ", kept with a note" }),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_off_turn_names_its_unfinished_calls_once() {
        let dir = tempfile::tempdir().unwrap();
        let log = RunLog::at(dir.path(), "c1");
        log.append(&RunEvent::Turn { at: 1, text: "make a.txt".into() });
        log.append(&RunEvent::Started { at: 2, call: "x".into(), name: "list_dir".into(), arguments: Value::Null, by_harness: false });
        log.append(&RunEvent::Finished { at: 3, call: "x".into(), ok: true, text: "a".into() });
        log.append(&RunEvent::Started { at: 4, call: "y".into(), name: "write_file".into(), arguments: serde_json::json!({"path": "a.txt"}), by_harness: false });
        // A torn last line from a crash is skipped.
        std::fs::OpenOptions::new().append(true).open(dir.path().join("runs/c1.jsonl")).unwrap().write_all(b"{\"kind\":\"fini").unwrap();
        assert_eq!(log.read().len(), 4);
        log.append(&RunEvent::Route { at: 5, route: "answer".into() });
        assert_eq!(log.read().len(), 5);
        assert_eq!(log.interrupted(), [("write_file".to_string(), serde_json::json!({"path": "a.txt"}))]);
        log.close_interrupted();
        assert!(log.interrupted().is_empty());
        assert!(describe(&log.read()).contains("calls write_file {\"path\":\"a.txt\"}"));
        log.delete();
        assert!(log.read().is_empty());
    }
}
