//! Measuring the agent: a list of messages with what should happen for each (the route
//! the harness takes, a tool that gets called, words in the answer), run against a
//! model, each in a fresh home so nothing carries over. `bc agent eval` runs a file of
//! them; `docs/agent-evals.json` is the starting set. Calls that change files or run
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
        let ended = agent.turn(&mut chat, &case.message, settings, Arc::default()).await;
        let route = memory.run_log(&chat.conv.id).read().iter().find_map(|e| if let RunEvent::Route { route, .. } = e { Some(route.clone()) } else { None }).unwrap_or_default();
        let answer = chat.conv.messages.iter().rev().find(|m| m.role == "assistant").map(|m| super::format::parse(&m.content).text).unwrap_or_default();
        let tools: Vec<String> = called.lock().unwrap().clone();
        let mut misses = Vec::new();
        if let Err(e) = ended {
            misses.push(format!("failed: {e:#}"));
        }
        if let Some(r) = &case.expect_route {
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
        let _ = std::fs::remove_dir_all(&home);
        out.push(Outcome { message: case.message.clone(), pass: misses.is_empty(), route, tools, answer, misses });
    }
    Ok(out)
}
