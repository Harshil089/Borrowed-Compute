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
pub mod memory;
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
use memory::{Conversation, Memory, Message};
use tools::{Risk, Tools};

/// Steps one message may take before the agent stops and says so.
pub const MAX_STEPS: usize = 25;
/// Most tokens the model writes in one step.
const STEP_TOKENS: usize = 2048;
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
    Done { steps: usize, prompt_tokens: usize, cached_tokens: usize, generated: usize, tokens_per_s: f64 },
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
- For anything current or that you do not know (news, results, prices, weather, docs), call web_search, then fetch_url on the best result when the snippets are not enough. Say where the answer came from.\n\
- When the person tells you something to keep (a fact, a preference, a plan) or asks you to remember it, call remember.\n\
- Lines from earlier conversations may come with a message, marked as such. When the person refers to something from before that is not there, call recall.\n\
- Never claim to have done something you did not do with a tool. You cannot act later on your own: you cannot set reminders, alarms or timers. Say so, and keep a note with remember instead.\n\
- In a project: look with the tools before you change anything, make exact edits, and check your work by running it when you can. When done, say briefly what you did.\n\
- Ask only when you cannot go on without an answer.";

impl Agent {
    fn system(&self) -> String {
        let mut base = BASE_PROMPT.to_string();
        base.push_str(&format!("\n\nToday is {}.", memory::day_of(memory::now())));
        if let Some(root) = &self.tools.root {
            base.push_str(&format!(" The project folder is {}; file paths are relative to it.", root.display()));
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
        let mut ids = tk.encode(&turns.open(&self.system()));
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
        if let Err(e) = &result {
            // The stages' state is unknown after a failure: start the stream over next time.
            chat.cache.reset();
            (self.events)(Event::Error { message: format!("{e:#}") });
        }
        let _ = self.memory.save(&chat.conv);
        result
    }

    async fn steps(&self, chat: &mut Chat, text: &str, settings: Settings, stop: Arc<AtomicBool>) -> Result<()> {
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
        chat.conv.messages.push(Message { role: "user".into(), content: text.to_string(), name: None });
        chat.conv.updated = memory::now();
        // shortcut: the related lines go to the model but not into the saved message, so a
        // re-render (reopening, trimming) drops them; store them if answers need them later.
        let read = match self.memory.related(text, Some(&chat.conv.id)) {
            Some(lines) => format!("{text}\n\n<earlier_conversations>\n{lines}\n</earlier_conversations>"),
            None => text.to_string(),
        };
        chat.ids.extend(tk.encode_segment(&turns.user(&read)));
        chat.cache.stop_with(stop.clone());
        let mut total = Stats::default();
        let mut steps = 0;
        let mut decode_ms = 0.0;
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
                break;
            }
            let results = futures_util::future::join_all(calls.iter().map(|(view, call)| self.handle(view, call.as_ref(), settings, &chat.allowed))).await;
            let mut pairs = Vec::new();
            for ((view, _), text) in calls.iter().zip(results) {
                chat.conv.messages.push(Message { role: "tool".into(), content: text.clone(), name: Some(view.name.clone()) });
                pairs.push((view.name.clone(), text));
            }
            chat.ids.extend(tk.encode_segment(&turns.results(&pairs)));
            let _ = self.memory.save(&chat.conv);
            if steps == MAX_STEPS {
                (self.events)(Event::Error { message: format!("Stopped after {MAX_STEPS} steps. Send another message to let it go on.") });
            }
        }
        let tokens_per_s = if decode_ms > 0.0 { total.generated as f64 / (decode_ms / 1000.0) } else { 0.0 };
        (self.events)(Event::Done { steps, prompt_tokens: total.prompt_tokens, cached_tokens: total.cached_tokens, generated: total.generated, tokens_per_s });
        Ok(())
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
        let (ok, text) = match self.tools.call(&call.name, &call.arguments).await {
            Ok(t) => (true, t),
            Err(e) => (false, format!("Error: {e:#}")),
        };
        // Results go into the model's context, so they are kept short.
        let text = head_tail(&text, tools::OUTPUT_CHARS * 2);
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
        // Saved, and the same tokens come back from the saved messages.
        let saved = r.agent.memory.load(&chat.conv.id).unwrap();
        assert_eq!(saved.messages, chat.conv.messages);
        let mut reopened = Chat::new(saved);
        let turns = Turns { tokenizer: &r.brain.tk, think: false };
        reopened.ids = r.agent.render(&reopened, &turns);
        assert_eq!(reopened.ids, chat.ids);
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
    async fn the_agent_stops_after_the_step_limit_or_when_told() {
        let looping = "<tool_call>{\"name\": \"list_dir\", \"arguments\": {}}</tool_call>";
        let r = agent(&[looping], Decision::Allow);
        let mut chat = Chat::new(Conversation::new(None));
        r.agent.turn(&mut chat, "go", Settings::default(), Arc::default()).await.unwrap();
        assert_eq!(r.brain.fed.lock().unwrap().len(), MAX_STEPS);
        assert!(r.seen.lock().unwrap().iter().any(|e| matches!(e, Event::Error { message } if message.contains("Stopped after"))));
        // Told to stop while the first call waits for approval: no second step.
        let stop = Arc::new(AtomicBool::new(false));
        let r = agent(&[WRITE], Decision::Allow);
        let flag = stop.clone();
        let agent = Agent { approve: Arc::new(move |_| { flag.store(true, Ordering::Relaxed); Box::pin(async { Decision::Allow }) }), ..r.agent };
        let mut chat = Chat::new(Conversation::new(None));
        agent.turn(&mut chat, "go", Settings::default(), stop).await.unwrap();
        assert_eq!(r.brain.fed.lock().unwrap().len(), 1);
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
        let s = r.agent.system();
        assert!(s.contains("Notes from earlier conversations") && s.contains("- the owner likes short answers"));
        assert!(s.contains("\"name\":\"edit_file\"") && s.contains("The project folder is"));
    }
}
