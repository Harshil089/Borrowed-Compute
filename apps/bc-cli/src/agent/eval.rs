//! Measuring the agent: a list of messages with what should happen for each (the route
//! the harness takes, a tool that gets called, words in the answer), run against a
//! model, each in a fresh home so nothing carries over. `bc agent eval` runs a file of
//! them; `docs/agent-evals/cases.json` is the starting set. A case can send earlier
//! messages first (`before`) to test follow-ups, and a failed case is sorted by kind. Calls that change files or run
//! commands are denied, so a run has no side effects; reading the web is allowed.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::log::RunEvent;
use super::memory::{Conversation, Memory};
use super::tools::{Risk, Tools};
use super::{Agent, Brain, Chat, Decision, Event, Settings};

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub message: String,
    /// A folder for the file tools, relative to the case file.
    #[serde(default)]
    pub folder: Option<String>,
    /// `answer`, `web`, `files` or `past conversations`.
    #[serde(default)]
    pub expect_route: Option<String>,
    /// A tool called by the model or the harness.
    #[serde(default)]
    pub expect_tool: Option<String>,
    /// Words in the final answer, any case.
    #[serde(default)]
    pub expect_text: Option<String>,
    /// Words that must not be in the final answer, any case ("cannot access").
    #[serde(default)]
    pub expect_not: Option<String>,
    /// Most calls the turn may make, the harness's included.
    #[serde(default)]
    pub max_calls: Option<usize>,
    /// Messages sent first in the same conversation, for follow-ups ("yes, do it").
    #[serde(default)]
    pub before: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub message: String,
    pub pass: bool,
    pub route: String,
    pub tools: Vec<String>,
    pub answer: String,
    /// What did not match.
    pub misses: Vec<String>,
    /// The checks the answer failed on the way (`check`), whether fixed or not.
    pub checks: Vec<String>,
    /// For a failed case, what kind of failure: format (calls that could not be read or
    /// did not fit their tool), loop (stopped for repeating or limits), refusal (said it
    /// cannot), ungrounded (kept numbers no result has), wrong tool, wrong answer, error.
    pub kind: String,
}

/// Runs every case in turn; `base` is where case folders are found, `scratch` where
/// each case gets its own home.
pub async fn run(brain: Arc<dyn Brain>, cases: &[Case], base: &Path, scratch: &Path, settings: Settings) -> Result<Vec<Outcome>> {
    let mut out = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let home = scratch.join(format!("case-{i}"));
        let memory = Memory::at(&home);
        let folder = case.folder.as_ref().map(|f| base.join(f));
        let conv = Conversation::new(folder.as_ref().map(|f| f.display().to_string()));
        let tools = Tools::new(folder.as_deref(), memory.clone(), conv.id.clone(), None, Vec::new())?;
        let called: Arc<Mutex<Vec<String>>> = Arc::default();
        let seen = called.clone();
        let risks = tools.clone();
        let agent = Agent {
            brain: brain.clone(),
            fallback: None,
            tools,
            memory: memory.clone(),
            events: Arc::new(move |e| {
                if let Event::Reply { calls, .. } = e {
                    seen.lock().unwrap().extend(calls.into_iter().map(|c| c.name));
                }
            }),
            approve: Arc::new(move |a| {
                let d = if matches!(risks.risk(&a.name), Risk::Net | Risk::Read) { Decision::Allow } else { Decision::Deny };
                Box::pin(async move { d })
            }),
        };
        let mut chat = Chat::new(conv);
        for earlier in &case.before {
            agent.turn(&mut chat, earlier, settings, Arc::default()).await?;
        }
        called.lock().unwrap().clear();
        let ended = agent.turn(&mut chat, &case.message, settings, Arc::default()).await;
        // This turn's part of the run log.
        let events = memory.run_log(&chat.conv.id).read();
        let events = &events[events.iter().rposition(|e| matches!(e, RunEvent::Turn { .. })).unwrap_or(0)..];
        let route = events.iter().find_map(|e| if let RunEvent::Route { route, .. } = e { Some(route.clone()) } else { None }).unwrap_or_default();
        let checks: Vec<String> = events.iter().filter_map(|e| if let RunEvent::Check { problem, .. } = e { Some(problem.clone()) } else { None }).collect();
        let kept_problem = events.iter().rev().find_map(|e| if let RunEvent::Check { problem, again: false, .. } = e { Some(problem.clone()) } else { None });
        let ended_for = events.iter().rev().find_map(|e| if let RunEvent::Ended { reason, .. } = e { Some(reason.clone()) } else { None }).unwrap_or_default();
        let bad_calls = events.iter().any(|e| matches!(e, RunEvent::Model { text, .. } if text.contains("<tool_call>")) ) && events.iter().any(|e| matches!(e, RunEvent::Finished { ok: false, text, .. } if text.starts_with("Error:")));
        let answer = chat.conv.messages.iter().rev().find(|m| m.role == "assistant").map(|m| super::format::parse(&m.content).text).unwrap_or_default();
        let tools: Vec<String> = called.lock().unwrap().clone();
        let mut misses = Vec::new();
        if let Err(e) = ended {
            misses.push(format!("failed: {e:#}"));
        }
        // Scored only where a route was taken: without the decision step, most are not.
        if let Some(r) = case.expect_route.as_ref().filter(|_| !route.is_empty()) {
            if *r != route {
                misses.push(format!("route {route:?}, wanted {r:?}"));
            }
        }
        if let Some(t) = &case.expect_tool {
            if !tools.contains(t) {
                misses.push(format!("no call to {t}"));
            }
        }
        if let Some(t) = &case.expect_text {
            if !answer.to_lowercase().contains(&t.to_lowercase()) {
                misses.push(format!("the answer lacks {t:?}"));
            }
        }
        if let Some(t) = &case.expect_not {
            if answer.to_lowercase().contains(&t.to_lowercase()) {
                misses.push(format!("the answer says {t:?}"));
            }
        }
        if let Some(n) = case.max_calls {
            if tools.len() > n {
                misses.push(format!("{} calls, at most {n} wanted", tools.len()));
            }
        }
        let kind = if misses.is_empty() {
            String::new()
        } else if misses.iter().any(|m| m.starts_with("failed")) {
            "error".into()
        } else if matches!(ended_for.as_str(), "repeating" | "budget" | "step limit") {
            "loop".into()
        } else if kept_problem.as_deref() == Some("denies its tools") || case.expect_not.as_ref().is_some_and(|_| misses.iter().any(|m| m.starts_with("the answer says"))) {
            "refusal".into()
        } else if kept_problem.as_deref() == Some("numbers in no result") {
            "ungrounded".into()
        } else if bad_calls {
            "format".into()
        } else if misses.iter().any(|m| m.starts_with("no call") || m.starts_with("route")) {
            "wrong tool".into()
        } else {
            "wrong answer".into()
        };
        let _ = std::fs::remove_dir_all(&home);
        out.push(Outcome { message: case.message.clone(), pass: misses.is_empty(), route, tools, answer, misses, checks, kind });
    }
    Ok(out)
}
