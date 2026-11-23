//! The agent: a model on the mesh that works through a task with tools, step by step,
//! and remembers past conversations.
//!
//! Each step, the model writes an answer; if it calls tools, they run (after the person
//! allows the ones that change something, see [`tools::Risk`]) and their results go back
//! for the next step, until it answers without a call. A conversation is kept as the
//! token ids already fed, and its stream keeps them on the stages ([`Cached`]), so a step
//! sends only the tool results and the start of the next answer, not the history again.
//! Calls in one answer run at the same time.

pub mod format;
pub mod log;
pub mod memory;
pub mod remind;
pub mod tools;

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use bc_infer::Tokenizer;
use bc_mesh::llm::{Cached, Llm, Stats};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use format::{head_tail, parse, system_prompt, Turns};
use log::RunEvent;
use memory::{Conversation, Memory, Message};
use tools::{Risk, Tools};

/// Steps one message may take before the agent stops and says so.
pub const MAX_STEPS: usize = 25;
/// Most tokens the model writes in one step.
const STEP_TOKENS: usize = 2048;
/// Most tokens the model writes in one turn, all steps together: a runaway model stops
/// here instead of filling the context.
const TURN_TOKENS: usize = 16_384;
/// Room kept for the next answer when the conversation nears the model's context.
const ANSWER_ROOM: usize = 1024;

/// A model the agent thinks with: the mesh's [`Llm`], or a scripted one in tests.
pub trait Brain: Send + Sync {
    fn tokenizer(&self) -> &Tokenizer;
    fn context(&self) -> usize;
    fn complete<'a>(&'a self, cache: &'a mut Cached, ids: &'a [u32], max: usize, on_text: &'a mut (dyn FnMut(&str) + Send)) -> BoxFuture<'a, Result<(Vec<u32>, Stats)>>;
}

impl Brain for Llm {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context(&self) -> usize {
        self.cfg.context
    }

    fn complete<'a>(&'a self, cache: &'a mut Cached, ids: &'a [u32], max: usize, on_text: &'a mut (dyn FnMut(&str) + Send)) -> BoxFuture<'a, Result<(Vec<u32>, Stats)>> {
        // Greedy, so the draft model's guesses can be checked several at a time.
        Box::pin(self.generate_cached(cache, ids, max, bc_infer::Sampling::default(), on_text))
    }
}

/// The conversation a set of tools works for: the one whose run log records its calls.
fn chat_id_of(tools: &Tools) -> String {
    tools.conversation.clone()
}

/// Pages the harness opens from its own search, so the model reads articles, not only
/// their snippets.
const PREFETCH_PAGES: usize = 2;

fn call_key(c: &format::Call) -> String {
    format!("{} {}", c.name, c.arguments)
}

/// What to search the web for, when a message asks for something only the web has: news,
/// anything current, or a search outright. The question's own words, less the asking.
fn web_query(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| lower.split(|c: char| !c.is_alphanumeric()).any(|x| x == *w));
    let phrase = |phrases: &[&str]| phrases.iter().any(|p| lower.contains(p));
    let wants = has(&["news", "headlines", "latest", "weather", "forecast", "google", "internet", "online"])
        || phrase(&["on the web", "search the web", "web search", "look up", "who won", "score of", "price of", "right now", "this week", "today's", "todays", "yesterday's", "currently"]);
    if !wants {
        return None;
    }
    const ASKING: &[&str] = &[
        "can", "could", "would", "you", "please", "check", "search", "look", "up", "for", "find", "on", "the", "web", "internet", "online", "google", "me", "tell",
        "give", "show", "just", "a", "an", "of", "about", "what", "whats", "is", "are", "some", "read", "get",
        // News results are today's already; the word only finds pages about dates.
        "today", "today's", "todays", "now", "currently",
    ];
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| !ASKING.contains(&w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase().replace('’', "'").as_str()))
        .collect();
    let query = words.join(" ").trim_matches(|c: char| !c.is_alphanumeric()).to_string();
    Some(if query.is_empty() { text.trim().to_string() } else { query })
}

/// A reminder the message asks for, when it says the time: ("9pm", "call Ayush at 9pm").
fn reminder_request(text: &str) -> Option<(String, String)> {
    let lower = text.to_lowercase();
    let i = lower.find("remind me")?;
    remind::parse_when(text, chrono::Local::now())?;
    let what = text[i + "remind me".len()..].trim();
    let what = ["to ", "that ", "about ", "of "].iter().find_map(|p| what.strip_prefix(p)).unwrap_or(what).trim_end_matches(['.', '!']);
    Some((text.to_string(), what.to_string()))
}

/// What the person decided about a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Allow,
    /// Allow this tool for the rest of the conversation.
    AllowAlways,
    Deny,
}

/// A call waiting for the person.
#[derive(Debug, Clone, Serialize)]
pub struct Approval {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub risk: Risk,
}

pub type Approver = Arc<dyn Fn(Approval) -> BoxFuture<'static, Decision> + Send + Sync>;

#[derive(Debug, Clone, Serialize)]
pub struct CallView {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    /// Why the call could not be read.
    pub error: Option<String>,
}

/// What happens during a turn, for a window to show as it goes.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// The model's text as it writes it, tool calls and all.
    Text { text: String },
    /// One step's answer, read.
    Reply { text: String, thought: String, calls: Vec<CallView> },
    Approve(Approval),
    Result { id: String, name: String, ok: bool, text: String },
    Done { steps: usize, prompt_tokens: usize, cached_tokens: usize, generated: usize, tokens_per_s: f64, context_used: usize, context: usize },
    Error { message: String },
}

pub type Events = Arc<dyn Fn(Event) + Send + Sync>;

#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Settings {
    /// Let the model reason before it answers: better plans, more tokens a step.
    #[serde(default)]
    pub think: bool,
    /// Run every call without asking.
    #[serde(default)]
    pub auto_approve: bool,
}

/// One open conversation: what it says, and what the model has been fed of it.
pub struct Chat {
    pub conv: Conversation,
    ids: Vec<u32>,
    cache: Cached,
    allowed: Arc<Mutex<HashSet<String>>>,
    /// The model `ids` were made for: its tokenizer's address, which changes when another
    /// model (or the same one again) starts on the Mesh page.
    brain: usize,
}

impl Chat {
    pub fn new(conv: Conversation) -> Self {
        Chat { conv, ids: Vec::new(), cache: Cached::default(), allowed: Arc::default(), brain: 0 }
    }
}

pub struct Agent {
    pub brain: Arc<dyn Brain>,
    pub tools: Tools,
    pub memory: Memory,
    pub events: Events,
    pub approve: Approver,
}

const BASE_PROMPT: &str = "You are the assistant in Borrowed Compute, running on the person's own devices.\n\
- When a tool can answer or do what the person asks, call it at once. Do not describe what you could do, and do not ask which details they want when a sensible choice exists.\n\
- For anything current or that you do not know (news, results, prices, weather, docs), call web_search, then fetch_url on the best result when the snippets are not enough. When search results are already there, answer from them. Say where the answer came from.\n\
- Each message ends with the local time it was sent; use it for questions about the date and time.\n\
- When the person tells you something to keep (a fact, a preference, a plan) or asks you to remember it, call remember.\n\
- You remember the person's other conversations: the latest are listed below, and related lines may come with a message. They are background, which the person cannot see: never bring them up, quote or list them on your own. Use them only when the person asks about or refers to something from before; when it is not there, call recall.\n\
- Never claim to have done something you did not do with a tool. To remind the person of something later, call remind; say the time its result gives.\n\
- In a project: look with the tools before you change anything, make exact edits, and check your work by running it when you can. When done, say briefly what you did.\n\
- Ask only when you cannot go on without an answer.";

impl Agent {
    fn system(&self, conversation: &str) -> String {
        let mut base = BASE_PROMPT.to_string();
        if let Some(root) = &self.tools.root {
            // The top of the folder comes with the prompt, so "what is in this" needs no
            // tool call and the model knows from the start which files it can open.
            let top = self.tools.list_dir(".").unwrap_or_default();
            base.push_str(&format!(
                "\n\nThe person chose the project folder {}: \"this\", \"here\" and \"the project\" mean it, and file paths are relative to it. Its top level:\n{top}",
                root.display()
            ));
        }
        let recent = self.memory.recent(Some(conversation));
        if !recent.is_empty() {
            base.push_str(&format!("\n\nWhat the person said in their latest other conversations, newest first (use recall to search all of them):\n{recent}"));
        }
        let notes = self.memory.notes_for_prompt();
        if !notes.is_empty() {
            base.push_str(&format!("\n\nNotes from earlier conversations (use recall to search them in full):\n{notes}"));
        }
        system_prompt(&base, &self.tools.specs())
    }

    /// The tokens of a conversation as the model reads it, from its messages.
    fn render(&self, chat: &Chat, turns: &Turns) -> Vec<u32> {
        let tk = turns.tokenizer;
        let mut ids = tk.encode(&turns.open(&self.system(&chat.conv.id)));
        let mut results: Vec<(String, String)> = Vec::new();
        for m in &chat.conv.messages {
            if m.role == "tool" {
                results.push((m.name.clone().unwrap_or_default(), m.content.clone()));
                continue;
            }
            if !results.is_empty() {
                ids.extend(tk.encode_segment(&turns.results(&std::mem::take(&mut results))));
            }
            match m.role.as_str() {
                "user" => ids.extend(tk.encode_segment(&turns.user(&m.content))),
                _ => ids.extend(tk.encode_segment(&format!("{}{}", m.content, turns.close()))),
            }
        }
        if !results.is_empty() {
            ids.extend(tk.encode_segment(&turns.results(&results)));
        }
        ids
    }

    /// Answers `text` in `chat`, step by step, until the model answers without calling a
    /// tool, `stop` is set, or [`MAX_STEPS`] pass. The conversation is saved after each step.
    pub async fn turn(&self, chat: &mut Chat, text: &str, settings: Settings, stop: Arc<AtomicBool>) -> Result<()> {
        let result = self.steps(chat, text, settings, stop).await;
        let reason = match &result {
            Ok(reason) => reason.clone(),
            Err(e) => {
                // The stages' state is unknown after a failure: start the stream over next time.
                chat.cache.reset();
                (self.events)(Event::Error { message: format!("{e:#}") });
                format!("error: {e:#}")
            }
        };
        self.memory.run_log(&chat.conv.id).append(&RunEvent::Ended { at: memory::now(), reason });
        let _ = self.memory.save(&chat.conv);
        result.map(|_| ())
    }

    /// Gives why the turn ended.
    async fn steps(&self, chat: &mut Chat, text: &str, settings: Settings, stop: Arc<AtomicBool>) -> Result<String> {
        let brain = self.brain.clone();
        let tk = brain.tokenizer();
        let turns = Turns { tokenizer: tk, think: settings.think };
        // Another model's token ids mean nothing to this one, and its stages are gone.
        let brain_id = tk as *const Tokenizer as usize;
        if chat.brain != brain_id {
            chat.brain = brain_id;
            chat.ids.clear();
            chat.cache = Cached::default();
        }
        if chat.ids.is_empty() {
            chat.ids = self.render(chat, &turns);
        }
        if chat.conv.title.is_empty() {
            chat.conv.title = format::clip(text.lines().next().unwrap_or(text).trim(), 60);
        }
        let log = self.memory.run_log(&chat.conv.id);
        // A turn cut off by a crash or a quit: its unfinished calls may or may not have run.
        let cut: Vec<String> = log.interrupted().iter().map(|(name, args)| format!("{name} {args}")).collect();
        log.close_interrupted();
        if !cut.is_empty() {
            (self.events)(Event::Error { message: format!("The last answer was cut off while running {}; it may or may not have finished.", cut.join(", ")) });
        }
        log.append(&RunEvent::Turn { at: memory::now(), text: text.to_string() });
        chat.conv.messages.push(Message { role: "user".into(), content: text.to_string(), name: None });
        chat.conv.updated = memory::now();
        // shortcut: the time and related lines go to the model but not into the saved
        // message, so a re-render (reopening, trimming) drops them; store them if answers
        // need them later.
        let mut read = format!(
            "{text}\n\n(Sent {}; {} of the {} tokens of context used)",
            chrono::Local::now().format("%A %Y-%m-%d %H:%M, local time"),
            chat.ids.len(),
            brain.context()
        );
        if !cut.is_empty() {
            read.push_str(&format!("\n\n(The app stopped during your last answer while running: {}. It may or may not have finished; check before doing it again.)", cut.join("; ")));
        }
        if let Some(lines) = self.memory.related(text, Some(&chat.conv.id)) {
            read.push_str(&format!("\n\n<earlier_conversations>\n{lines}\n</earlier_conversations>"));
        }
        chat.ids.extend(tk.encode_segment(&turns.user(&read)));
        chat.cache.stop_with(stop.clone());
        // Calls made this turn, so the same one is not run twice.
        let mut made: HashSet<String> = HashSet::new();
        // The harness looks things up itself when the person asks for something on the web,
        // so a model that would not call web_search still answers from real results.
        // Likewise a reminder asked for with a time: set, whatever the model would do.
        if let Some((when, what)) = reminder_request(text) {
            self.harness_calls(chat, &turns, &mut made, vec![format::Call { name: "remind".into(), arguments: serde_json::json!({ "when": when, "text": what }) }]).await;
        }
        if let Some(query) = web_query(text).filter(|_| self.tools.specs().iter().any(|t| t.name == "web_search")) {
            let found = self.harness_calls(chat, &turns, &mut made, vec![format::Call { name: "web_search".into(), arguments: serde_json::json!({ "query": query }) }]).await;
            let pages: Vec<format::Call> = found
                .lines()
                .map(str::trim)
                .filter(|l| l.starts_with("https://"))
                .take(PREFETCH_PAGES)
                .map(|url| format::Call { name: "fetch_url".into(), arguments: serde_json::json!({ "url": url }) })
                .collect();
            if !pages.is_empty() && !stop.load(Ordering::Relaxed) {
                self.harness_calls(chat, &turns, &mut made, pages).await;
            }
        }
        let mut total = Stats::default();
        let mut steps = 0;
        let mut decode_ms = 0.0;
        let mut stuck = 0;
        let mut reason = "stopped".to_string();
        while steps < MAX_STEPS && !stop.load(Ordering::Relaxed) {
            steps += 1;
            self.fit(chat, &turns)?;
            let max = (brain.context() - chat.ids.len()).min(STEP_TOKENS);
            let events = self.events.clone();
            let mut on_text = move |t: &str| events(Event::Text { text: t.to_string() });
            let (out, stats) = brain.complete(&mut chat.cache, &chat.ids, max, &mut on_text).await?;
            total.prompt_tokens += stats.prompt_tokens;
            total.cached_tokens += stats.cached_tokens;
            total.generated += stats.generated;
            decode_ms += (stats.total_ms - stats.first_token_ms).max(0.0);
            let raw = tk.decode(&out);
            log.append(&RunEvent::Model { at: memory::now(), text: raw.clone(), prompt_tokens: stats.prompt_tokens, cached_tokens: stats.cached_tokens, generated: stats.generated });
            chat.ids.extend(&out);
            chat.ids.extend(tk.encode_segment(turns.close()));
            chat.conv.messages.push(Message { role: "assistant".into(), content: raw.clone(), name: None });
            chat.conv.updated = memory::now();
            let reply = parse(&raw);
            let calls: Vec<(CallView, Option<format::Call>)> = reply
                .calls
                .into_iter()
                .map(|c| {
                    let id = format!("{:016x}", rand::random::<u64>());
                    match c {
                        Ok(c) => (CallView { id, name: c.name.clone(), arguments: c.arguments.clone(), error: None }, Some(c)),
                        Err(e) => (CallView { id, name: String::new(), arguments: Value::Null, error: Some(e) }, None),
                    }
                })
                .collect();
            (self.events)(Event::Reply { text: reply.text, thought: reply.thought, calls: calls.iter().map(|c| c.0.clone()).collect() });
            if calls.is_empty() || stop.load(Ordering::Relaxed) {
                let _ = self.memory.save(&chat.conv);
                if calls.is_empty() {
                    reason = "answered".into();
                }
                break;
            }
            // A small model can call the same thing again and again; the second time it hears
            // so, and a second answer of nothing but repeats ends the turn.
            let repeated: Vec<bool> = calls.iter().map(|(_, c)| c.as_ref().is_some_and(|c| !made.insert(call_key(c)))).collect();
            let allowed = &chat.allowed;
            let results = futures_util::future::join_all(calls.iter().zip(&repeated).map(|((view, call), &again)| async move {
                if again {
                    let text = "You already made this exact call; its result is above. Answer from it, or try something different.".to_string();
                    (self.events)(Event::Result { id: view.id.clone(), name: view.name.clone(), ok: false, text: text.clone() });
                    text
                } else {
                    self.handle(view, call.as_ref(), settings, allowed).await
                }
            }))
            .await;
            if repeated.iter().all(|&r| r) {
                stuck += 1;
            }
            let mut pairs = Vec::new();
            for ((view, _), text) in calls.iter().zip(results) {
                chat.conv.messages.push(Message { role: "tool".into(), content: text.clone(), name: Some(view.name.clone()) });
                pairs.push((view.name.clone(), text));
            }
            chat.ids.extend(tk.encode_segment(&turns.results(&pairs)));
            let _ = self.memory.save(&chat.conv);
            if stuck >= 2 {
                (self.events)(Event::Error { message: "Stopped: the model kept repeating the same call. Send another message to go on.".into() });
                reason = "repeating".into();
                break;
            }
            if total.generated >= TURN_TOKENS {
                (self.events)(Event::Error { message: format!("Stopped: this answer reached its budget of {TURN_TOKENS} written tokens. Send another message to go on.") });
                reason = "budget".into();
                break;
            }
            if steps == MAX_STEPS {
                (self.events)(Event::Error { message: format!("Stopped after {MAX_STEPS} steps. Send another message to let it go on.") });
                reason = "step limit".into();
            }
        }
        let tokens_per_s = if decode_ms > 0.0 { total.generated as f64 / (decode_ms / 1000.0) } else { 0.0 };
        (self.events)(Event::Done {
            steps,
            prompt_tokens: total.prompt_tokens,
            cached_tokens: total.cached_tokens,
            generated: total.generated,
            tokens_per_s,
            context_used: chat.ids.len(),
            context: brain.context(),
        });
        Ok(reason)
    }

    /// Calls the harness makes for the model, written into the conversation as if the model
    /// had made them: the window shows them as calls, and the model reads the results as
    /// its own. They come from the person's own words, so they are not asked about. Gives
    /// the results, joined.
    async fn harness_calls(&self, chat: &mut Chat, turns: &Turns<'_>, made: &mut HashSet<String>, calls: Vec<format::Call>) -> String {
        let tk = turns.tokenizer;
        let raw: String = calls.iter().map(|c| format!("<tool_call>\n{}\n</tool_call>", serde_json::json!({ "name": c.name, "arguments": c.arguments }))).collect::<Vec<_>>().join("\n");
        chat.ids.extend(tk.encode_segment(&format!("{raw}{}", turns.close())));
        chat.conv.messages.push(Message { role: "assistant".into(), content: raw, name: None });
        let views: Vec<CallView> = calls.iter().map(|c| CallView { id: format!("{:016x}", rand::random::<u64>()), name: c.name.clone(), arguments: c.arguments.clone(), error: None }).collect();
        (self.events)(Event::Reply { text: String::new(), thought: String::new(), calls: views.clone() });
        let results = futures_util::future::join_all(calls.iter().zip(&views).map(|(c, view)| async move {
            let (ok, text) = self.run_call(&chat_id_of(&self.tools), &view.id, c, true, tools::OUTPUT_CHARS).await;
            (self.events)(Event::Result { id: view.id.clone(), name: c.name.clone(), ok, text: text.clone() });
            text
        }))
        .await;
        let mut pairs = Vec::new();
        for (c, text) in calls.iter().zip(&results) {
            made.insert(call_key(c));
            chat.conv.messages.push(Message { role: "tool".into(), content: text.clone(), name: Some(c.name.clone()) });
            pairs.push((c.name.clone(), text.clone()));
        }
        chat.ids.extend(tk.encode_segment(&turns.results(&pairs)));
        results.join("\n")
    }

    /// Runs one call, recorded in the run log before and after. Output longer than `keep`
    /// characters is kept whole as an artifact and the model gets its start and end, with
    /// the artifact's id to read the rest.
    async fn run_call(&self, conversation: &str, id: &str, call: &format::Call, by_harness: bool, keep: usize) -> (bool, String) {
        let log = self.memory.run_log(conversation);
        log.append(&RunEvent::Started { at: memory::now(), call: id.to_string(), name: call.name.clone(), arguments: call.arguments.clone(), by_harness });
        let (ok, text) = match self.tools.call(&call.name, &call.arguments).await {
            Ok(t) => (true, t),
            Err(e) => (false, format!("Error: {e:#}")),
        };
        let text = if text.chars().count() > keep {
            let dir = log.artifacts();
            let kept = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(dir.join(format!("{id}.txt")), &text)).is_ok();
            let note = if kept { format!("\n[The whole output is {} characters, kept as artifact {id}: read the rest with read_artifact.]", text.chars().count()) } else { String::new() };
            format!("{}{note}", head_tail(&text, keep))
        } else {
            text
        };
        log.append(&RunEvent::Finished { at: memory::now(), call: id.to_string(), ok, text: format::clip(&text, 2000) });
        (ok, text)
    }

    /// Makes room for the next answer: the oldest tool results are cut down first. The
    /// tokens then change from that point, so the next step sends them again.
    fn fit(&self, chat: &mut Chat, turns: &Turns) -> Result<()> {
        let limit = self.brain.context().saturating_sub(ANSWER_ROOM);
        if chat.ids.len() <= limit {
            return Ok(());
        }
        const TRIMMED: &str = "[left out to save room]";
        for i in 0..chat.conv.messages.len() {
            let m = &mut chat.conv.messages[i];
            if m.role == "tool" && m.content != TRIMMED {
                m.content = TRIMMED.into();
                chat.ids = self.render(chat, turns);
                if chat.ids.len() <= limit {
                    return Ok(());
                }
            }
        }
        bail!("this conversation no longer fits in the model's context of {} tokens; start a new one", self.brain.context())
    }

    async fn handle(&self, view: &CallView, call: Option<&format::Call>, settings: Settings, allowed: &Mutex<HashSet<String>>) -> String {
        let Some(call) = call else {
            let text = format!("Error: {}", view.error.as_deref().unwrap_or("unreadable call"));
            (self.events)(Event::Result { id: view.id.clone(), name: String::new(), ok: false, text: text.clone() });
            return text;
        };
        let risk = Tools::risk(&call.name);
        let asked = risk != Risk::Read && !settings.auto_approve && !allowed.lock().unwrap().contains(&call.name);
        if asked {
            let approval = Approval { id: view.id.clone(), name: call.name.clone(), arguments: call.arguments.clone(), risk };
            (self.events)(Event::Approve(approval.clone()));
            match (self.approve)(approval).await {
                Decision::Deny => {
                    let text = "The person did not allow this call.".to_string();
                    (self.events)(Event::Result { id: view.id.clone(), name: call.name.clone(), ok: false, text: text.clone() });
                    return text;
                }
                Decision::AllowAlways => {
                    allowed.lock().unwrap().insert(call.name.clone());
                }
                Decision::Allow => {}
            }
        }
        let (ok, text) = self.run_call(&chat_id_of(&self.tools), &view.id, call, false, tools::OUTPUT_CHARS * 2).await;
        (self.events)(Event::Result { id: view.id.clone(), name: call.name.clone(), ok, text: text.clone() });
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_infer::gguf::write::{gguf, Meta};

    /// A byte-level BPE vocabulary of ChatML control tokens and every byte (as GPT-2 maps
    /// bytes to characters), with no merges: any text encodes, one token a byte.
    fn tokenizer() -> Tokenizer {
        let mut tokens: Vec<String> = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"].map(String::from).to_vec();
        let mut types = vec![2, 3, 3, 3, 3];
        let mut extra = 0;
        for b in 0..256u32 {
            let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let c = if printable { char::from_u32(b).unwrap() } else { extra += 1; char::from_u32(255 + extra).unwrap() };
            tokens.push(c.to_string());
            types.push(1);
        }
        let n = tokens.len();
        let meta = vec![
            ("tokenizer.ggml.model", Meta::Str("gpt2".into())),
            ("tokenizer.ggml.tokens", Meta::Strs(tokens)),
            ("tokenizer.ggml.scores", Meta::F32s(vec![0.0; n])),
            ("tokenizer.ggml.token_type", Meta::I32s(types)),
            ("tokenizer.ggml.bos_token_id", Meta::U32(1)),
            ("tokenizer.ggml.eos_token_id", Meta::U32(2)),
        ];
        Tokenizer::from_header(&bc_infer::gguf::parse(&gguf(&meta, &[])).unwrap()).unwrap()
    }

    /// Answers from a script, one a step, and records what it was fed.
    struct Script {
        tk: Tokenizer,
        answers: Mutex<Vec<String>>,
        fed: Mutex<Vec<Vec<u32>>>,
    }

    impl Brain for Script {
        fn tokenizer(&self) -> &Tokenizer {
            &self.tk
        }
        fn context(&self) -> usize {
            1 << 16
        }
        fn complete<'a>(&'a self, _: &'a mut Cached, ids: &'a [u32], _: usize, on_text: &'a mut (dyn FnMut(&str) + Send)) -> BoxFuture<'a, Result<(Vec<u32>, Stats)>> {
            self.fed.lock().unwrap().push(ids.to_vec());
            let mut answers = self.answers.lock().unwrap();
            let a = if answers.len() > 1 { answers.remove(0) } else { answers[0].clone() };
            on_text(&a);
            let out = self.tk.encode_segment(&a);
            Box::pin(async move { Ok((out, Stats::default())) })
        }
    }

    struct Run {
        agent: Agent,
        brain: Arc<Script>,
        seen: Arc<Mutex<Vec<Event>>>,
        dir: tempfile::TempDir,
    }

    fn agent(answers: &[&str], decision: Decision) -> Run {
        let dir = tempfile::tempdir().unwrap();
        let brain = Arc::new(Script { tk: tokenizer(), answers: Mutex::new(answers.iter().map(|s| s.to_string()).collect()), fed: Mutex::default() });
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let memory = Memory::at(&dir.path().join("home"));
        std::fs::create_dir_all(dir.path().join("project")).unwrap();
        let tools = Tools::new(Some(&dir.path().join("project")), memory.clone(), "c".into(), None, Vec::new()).unwrap();
        let agent = Agent {
            brain: brain.clone(),
            tools,
            memory,
            events: Arc::new(move |e| s.lock().unwrap().push(e)),
            approve: Arc::new(move |_| Box::pin(async move { decision })),
        };
        Run { agent, brain, seen, dir }
    }

    const WRITE: &str = "Writing it.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n</tool_call>";

    #[tokio::test]
    async fn a_call_runs_once_allowed_and_its_result_goes_back() {
        let r = agent(&[WRITE, "Done: a.txt says hi."], Decision::Allow);
        let mut chat = Chat::new(Conversation::new(None));
        r.agent.turn(&mut chat, "make a.txt", Settings::default(), Arc::default()).await.unwrap();
        assert_eq!(std::fs::read_to_string(r.dir.path().join("project/a.txt")).unwrap(), "hi");
        let kinds: Vec<String> = r.seen.lock().unwrap().iter().map(|e| serde_json::to_value(e).unwrap()["kind"].as_str().unwrap().to_string()).collect();
        assert_eq!(kinds, ["text", "reply", "approve", "result", "text", "reply", "done"]);
        let roles: Vec<&str> = chat.conv.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
        assert_eq!(chat.conv.title, "make a.txt");
        // Each step's prompt continues the last one exactly: nothing fed is ever rewritten.
        let fed = r.brain.fed.lock().unwrap();
        assert!(fed[1].starts_with(&fed[0]) && fed[1].len() > fed[0].len());
        let tail = r.brain.tk.decode(&fed[1][fed[0].len()..]);
        assert!(tail.starts_with(WRITE) && tail.contains("<tool_response>\nWrote 2 bytes to a.txt.\n</tool_response>"), "{tail}");
        // Saved, and the same tokens come back from the saved messages. Only the system
        // prompt differs (it lists the folder as it is now, a.txt included), and the
        // message's sending time, which is not saved.
        let saved = r.agent.memory.load(&chat.conv.id).unwrap();
        assert_eq!(saved.messages, chat.conv.messages);
        let mut reopened = Chat::new(saved);
        let turns = Turns { tokenizer: &r.brain.tk, think: false };
        reopened.ids = r.agent.render(&reopened, &turns);
        let im_end = r.brain.tk.token_id("<|im_end|>").unwrap();
        let after_system = |ids: &[u32]| {
            let text = r.brain.tk.decode(&ids[ids.iter().position(|&t| t == im_end).unwrap()..]);
            match (text.find("\n\n(Sent "), text.find("context used)")) {
                (Some(a), Some(b)) => format!("{}{}", &text[..a], &text[b + "context used)".len()..]),
                _ => text,
            }
        };
        assert_eq!(after_system(&reopened.ids), after_system(&chat.ids));
        assert!(r.brain.tk.decode(&reopened.ids).contains("a.txt (2 bytes)"));
    }

    #[tokio::test]
    async fn a_denied_call_changes_nothing_and_the_model_hears_why() {
        let r = agent(&[WRITE, "Fine, I will not."], Decision::Deny);
        let mut chat = Chat::new(Conversation::new(None));
        r.agent.turn(&mut chat, "make a.txt", Settings::default(), Arc::default()).await.unwrap();
        assert!(!r.dir.path().join("project/a.txt").exists());
        assert_eq!(chat.conv.messages[2].content, "The person did not allow this call.");
        // Auto-approve asks nobody.
        let r = agent(&[WRITE, "Done."], Decision::Deny);
        let mut chat = Chat::new(Conversation::new(None));
        r.agent.turn(&mut chat, "make a.txt", Settings { auto_approve: true, think: false }, Arc::default()).await.unwrap();
        assert!(r.dir.path().join("project/a.txt").exists());
        assert!(!r.seen.lock().unwrap().iter().any(|e| matches!(e, Event::Approve(_))));
    }

    #[tokio::test]
    async fn long_output_is_kept_whole_and_every_step_is_logged() {
        let search = "<tool_call>{\"name\": \"search\", \"arguments\": {\"text\": \"needle\"}}</tool_call>";
        let r = agent(&[search, "Found them."], Decision::Allow);
        let line = format!("needle {}\n", "x".repeat(180));
        std::fs::write(r.dir.path().join("project/big.txt"), line.repeat(60)).unwrap();
        let mut chat = Chat::new(Conversation::new(None));
        chat.conv.id = "c".into();
        // The last turn was cut off while writing a file.
        let log = r.agent.memory.run_log("c");
        log.append(&RunEvent::Turn { at: 1, text: "write it".into() });
        log.append(&RunEvent::Started { at: 2, call: "0".into(), name: "write_file".into(), arguments: serde_json::json!({"path": "a.txt"}), by_harness: false });
        r.agent.turn(&mut chat, "find the needles", Settings::default(), Arc::default()).await.unwrap();
        assert!(r.seen.lock().unwrap().iter().any(|e| matches!(e, Event::Error { message } if message.contains("cut off while running write_file"))));
        assert!(r.brain.tk.decode(&r.brain.fed.lock().unwrap()[0]).contains("may or may not have finished"));
        let result = &chat.conv.messages[2].content;
        let id = result.split("kept as artifact ").nth(1).and_then(|t| t.split(':').next()).expect(result).to_string();
        assert!(r.agent.tools.call("read_artifact", &serde_json::json!({"id": id, "offset": 50})).await.unwrap().starts_with("   50  "));
        assert!(r.agent.tools.call("read_artifact", &serde_json::json!({"id": "../c"})).await.is_err());
        let text = log::describe(&log.read());
        assert!(text.contains("ended: cut off") && text.contains("calls search") && text.contains("ended: answered"), "{text}");
        r.agent.memory.delete("c").ok();
        assert!(log.read().is_empty());
    }

    #[tokio::test]
    async fn the_agent_stops_a_repeating_model_or_when_told() {
        // The same call again is not run; the model hears so, and a second answer of only
        // repeats ends the turn.
        let looping = "<tool_call>{\"name\": \"list_dir\", \"arguments\": {}}</tool_call>";
        let r = agent(&[looping], Decision::Allow);
        let mut chat = Chat::new(Conversation::new(None));
        r.agent.turn(&mut chat, "go", Settings::default(), Arc::default()).await.unwrap();
        assert_eq!(r.brain.fed.lock().unwrap().len(), 3);
        assert!(chat.conv.messages[4].content.starts_with("You already made this exact call"));
        assert!(r.seen.lock().unwrap().iter().any(|e| matches!(e, Event::Error { message } if message.contains("kept repeating"))));
        // Told to stop while the first call waits for approval: no second step.
        let stop = Arc::new(AtomicBool::new(false));
        let r = agent(&[WRITE], Decision::Allow);
        let flag = stop.clone();
        let agent = Agent { approve: Arc::new(move |_| { flag.store(true, Ordering::Relaxed); Box::pin(async { Decision::Allow }) }), ..r.agent };
        let mut chat = Chat::new(Conversation::new(None));
        agent.turn(&mut chat, "go", Settings::default(), stop).await.unwrap();
        assert_eq!(r.brain.fed.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_reminder_with_a_time_is_read_from_the_message() {
        assert_eq!(reminder_request("remind me to call ayush at 9pm"), Some(("remind me to call ayush at 9pm".into(), "call ayush at 9pm".into())));
        assert_eq!(reminder_request("Can you remind me in 20 minutes to stretch."), Some(("Can you remind me in 20 minutes to stretch.".into(), "in 20 minutes to stretch".into())));
        assert_eq!(reminder_request("remind me to call ayush"), None);
        assert_eq!(reminder_request("call ayush at 9pm"), None);
    }

    #[test]
    fn asking_for_news_or_the_web_gives_a_search_of_the_question() {
        assert_eq!(web_query("can you check for today’s cybersecurity news on the web").as_deref(), Some("cybersecurity news"));
        assert_eq!(web_query("just give me the latest F1 grand prix results").as_deref(), Some("latest F1 grand prix results"));
        assert_eq!(web_query("what's the weather in Pune?").as_deref(), Some("what's weather in Pune"));
        assert_eq!(web_query("read the README and fix the build"), None);
        assert_eq!(web_query("search the project for TODO"), None);
    }

    #[tokio::test]
    async fn an_unreadable_call_is_answered_with_its_error() {
        let r = agent(&["<tool_call>{\"name\": \"list_dir\", \"arguments\": [1]}</tool_call>", "Sorry."], Decision::Allow);
        let mut chat = Chat::new(Conversation::new(None));
        r.agent.turn(&mut chat, "go", Settings::default(), Arc::default()).await.unwrap();
        assert!(chat.conv.messages[2].content.starts_with("Error: the arguments of list_dir are not an object"));
    }

    #[test]
    fn the_system_prompt_carries_the_notes_and_the_tools() {
        let r = agent(&["x"], Decision::Allow);
        r.agent.memory.remember("the owner likes short answers").unwrap();
        let s = r.agent.system("c");
        assert!(s.contains("Notes from earlier conversations") && s.contains("- the owner likes short answers"));
        assert!(s.contains("\"name\":\"edit_file\"") && s.contains("chose the project folder"));
    }
}
