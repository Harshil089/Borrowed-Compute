//! What the agent remembers: past conversations, searchable with `recall`, and notes it
//! keeps with `remember`, which go into every conversation's system prompt.
//!
//! Everything lives on the device that runs the conversation, in its home folder
//! (`conversations/<id>.json`, `memory.md`). Lenders never see it except as the prompt
//! activations they already compute on.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Notes go into every system prompt, so they are kept short: the newest this many
/// characters.
pub const NOTES_IN_PROMPT: usize = 2000;
/// Past conversations searched by `recall`, newest first.
const RECALL_CONVERSATIONS: usize = 500;
const RECALL_HITS: usize = 5;
const RELATED_HITS: usize = 3;
/// Other conversations every new one starts with, the last few things said in each.
const RECENT_CONVERSATIONS: usize = 8;
const RECENT_LINES: usize = 4;
const RECENT_CHARS: usize = 160;
/// Related lines come with every message, so they are kept short.
const RELATED_CHARS: usize = 300;
const CHUNK_CHARS: usize = 600;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// `user`, `assistant` (its raw text, tool calls included) or `tool`.
    pub role: String,
    pub content: String,
    /// For a tool result: the tool's name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    /// Unix seconds.
    pub created: u64,
    pub updated: u64,
    /// The project folder the file tools work in, if any.
    #[serde(default)]
    pub folder: Option<String>,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub id: String,
    pub title: String,
    pub updated: u64,
    pub folder: Option<String>,
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

impl Conversation {
    pub fn new(folder: Option<String>) -> Self {
        let t = now();
        Conversation { id: format!("{t:x}-{:08x}", rand::random::<u32>()), title: String::new(), created: t, updated: t, folder, messages: Vec::new() }
    }
}

/// The memory of one device: its conversations and notes.
#[derive(Clone)]
pub struct Memory {
    root: PathBuf,
}

impl Memory {
    pub fn at(home: &Path) -> Self {
        Memory { root: home.to_path_buf() }
    }

    /// What happened in a conversation's turns, and its large tool outputs.
    pub fn run_log(&self, conversation: &str) -> super::log::RunLog {
        super::log::RunLog::at(&self.root, conversation)
    }

    /// The reminders kept beside the conversations.
    pub fn reminders(&self) -> super::remind::Reminders {
        super::remind::Reminders::at(&self.root)
    }

    fn dir(&self) -> PathBuf {
        self.root.join("conversations")
    }

    fn path(&self, id: &str) -> Result<PathBuf> {
        // Ids come from the window: only the ones this module makes.
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            anyhow::bail!("not a conversation id: {id}");
        }
        Ok(self.dir().join(format!("{id}.json")))
    }

    pub fn save(&self, c: &Conversation) -> Result<()> {
        std::fs::create_dir_all(self.dir())?;
        let path = self.path(&c.id)?;
        // Written aside, then renamed, so a crash never leaves half a conversation.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(c)?)?;
        std::fs::rename(&tmp, &path).with_context(|| format!("saving {}", path.display()))
    }

    pub fn load(&self, id: &str) -> Result<Conversation> {
        let path = self.path(id)?;
        let bytes = std::fs::read(&path).with_context(|| format!("no conversation {id}"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.path(id)?;
        self.run_log(id).delete();
        std::fs::remove_file(path).context("deleting the conversation")
    }

    pub fn delete_all(&self) -> Result<()> {
        let _ = std::fs::remove_dir_all(self.root.join("runs"));
        match std::fs::remove_dir_all(self.dir()) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }

    /// Every conversation, newest first. Unreadable files are skipped.
    pub fn all(&self) -> Vec<Conversation> {
        let Ok(dir) = std::fs::read_dir(self.dir()) else { return Vec::new() };
        let mut all: Vec<Conversation> = dir
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok())
            .collect();
        all.sort_by_key(|c| std::cmp::Reverse(c.updated));
        all
    }

    pub fn list(&self) -> Vec<Summary> {
        self.all().into_iter().map(|c| Summary { id: c.id, title: c.title, updated: c.updated, folder: c.folder }).collect()
    }

    /// What the person said in the latest other conversations, newest first: every new
    /// conversation starts knowing it, with nothing for the person to copy into notes.
    pub fn recent(&self, except: Option<&str>) -> String {
        self.all()
            .into_iter()
            .filter(|c| Some(c.id.as_str()) != except && c.messages.iter().any(|m| m.role == "user"))
            .take(RECENT_CONVERSATIONS)
            .map(|c| {
                let said: Vec<&Message> = c.messages.iter().filter(|m| m.role == "user").collect();
                let lines: Vec<String> = said[said.len().saturating_sub(RECENT_LINES)..]
                    .iter()
                    .map(|m| format!("- {}", super::format::clip(&m.content.trim().replace('\n', " "), RECENT_CHARS)))
                    .collect();
                format!("[{} · {}]\n{}", c.title, day_of(c.updated), lines.join("\n"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn notes(&self) -> String {
        std::fs::read_to_string(self.root.join("memory.md")).unwrap_or_default()
    }

    pub fn set_notes(&self, text: &str) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::write(self.root.join("memory.md"), text).context("saving the notes")
    }

    pub fn remember(&self, note: &str) -> Result<()> {
        let note = note.trim().replace('\n', " ");
        if note.is_empty() {
            anyhow::bail!("the note is empty");
        }
        let mut notes = self.notes();
        if !notes.is_empty() && !notes.ends_with('\n') {
            notes.push('\n');
        }
        notes.push_str(&format!("- {note}\n"));
        self.set_notes(&notes)
    }

    /// The notes for a system prompt: the newest lines that fit.
    pub fn notes_for_prompt(&self) -> String {
        let notes = self.notes();
        let mut kept: Vec<&str> = Vec::new();
        let mut size = 0;
        for line in notes.lines().rev().filter(|l| !l.trim().is_empty()) {
            size += line.len() + 1;
            if size > NOTES_IN_PROMPT {
                break;
            }
            kept.push(line);
        }
        kept.reverse();
        kept.join("\n")
    }

    /// The passages of past conversations (all but `except`) that best match `query`.
    pub fn recall(&self, query: &str, except: Option<&str>) -> String {
        self.hits(query, except, RECALL_HITS, CHUNK_CHARS).unwrap_or_else(|| "Nothing in past conversations matches.".to_string())
    }

    /// Past conversations' lines that share a word with `message`, for the agent to read
    /// with it, so it recalls them even when the model would not think to search.
    pub fn related(&self, message: &str, except: Option<&str>) -> Option<String> {
        self.hits(message, except, RELATED_HITS, RELATED_CHARS)
    }

    fn hits(&self, query: &str, except: Option<&str>, k: usize, chars: usize) -> Option<String> {
        let all: Vec<Conversation> = self.all().into_iter().filter(|c| Some(c.id.as_str()) != except).take(RECALL_CONVERSATIONS).collect();
        let mut chunks: Vec<(&Conversation, &str, String)> = Vec::new();
        for c in &all {
            for m in &c.messages {
                let text = if m.role == "assistant" { super::format::parse(&m.content).text } else { m.content.clone() };
                for piece in split(&text, CHUNK_CHARS) {
                    chunks.push((c, m.role.as_str(), piece));
                }
            }
        }
        let docs: Vec<Vec<String>> = chunks.iter().map(|(_, _, t)| words(t)).collect();
        let hits = bm25(&words(query), &docs, k);
        if hits.is_empty() {
            return None;
        }
        let out = hits
            .into_iter()
            .map(|i| {
                let (c, role, text) = &chunks[i];
                format!("[{} · {}] {role}: {}", c.title, day_of(c.updated), super::format::clip(text.trim(), chars))
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        Some(out)
    }
}

/// Pieces of about `max` characters, cut between words.
fn split(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in text.split_inclusive(char::is_whitespace) {
        if cur.len() + w.len() > max && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push_str(w);
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Words that say nothing about what a message is about; matching on them would recall
/// every conversation.
const STOPWORDS: &[&str] = &[
    "am", "an", "as", "at", "be", "by", "do", "if", "in", "is", "it", "no", "or", "so", "up", "us", "about", "after", "again", "all", "also", "and", "any", "are", "but", "can", "could", "did", "does", "for", "from", "had", "has", "have", "her",
    "him", "his", "how", "its", "just", "like", "me", "more", "my", "not", "now", "of", "off", "on", "one", "our", "out", "please", "she", "should",
    "some", "than", "that", "the", "them", "then", "there", "they", "this", "to", "too", "was", "way", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
];

/// Lowercase words of two characters or more, leaving out [`STOPWORDS`].
pub(super) fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 2).map(str::to_lowercase).filter(|w| !STOPWORDS.contains(&w.as_str())).collect()
}

/// The `k` best documents for `query` by BM25 (k1 = 1.2, b = 0.75), best first; only
/// documents sharing a word with the query.
pub(super) fn bm25(query: &[String], docs: &[Vec<String>], k: usize) -> Vec<usize> {
    if docs.is_empty() || query.is_empty() {
        return Vec::new();
    }
    let n = docs.len() as f64;
    let avg = docs.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in docs {
        let mut seen: Vec<&str> = d.iter().map(String::as_str).collect();
        seen.sort_unstable();
        seen.dedup();
        for w in seen {
            *df.entry(w).or_default() += 1;
        }
    }
    let mut terms: Vec<&str> = query.iter().map(String::as_str).collect();
    terms.sort_unstable();
    terms.dedup();
    let mut scored: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            let len = d.len() as f64;
            let score: f64 = terms
                .iter()
                .filter_map(|t| {
                    let f = d.iter().filter(|w| w == t).count() as f64;
                    let df = *df.get(t)? as f64;
                    (f > 0.0).then(|| ((n - df + 0.5) / (df + 0.5) + 1.0).ln() * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len / avg)))
                })
                .sum();
            (score > 0.0).then_some((i, score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// A Unix time as a date (UTC).
pub fn day_of(secs: u64) -> String {
    // Days to a civil date (Howard Hinnant's algorithm).
    let z = (secs / 86400) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(title: &str, updated: u64, texts: &[(&str, &str)]) -> Conversation {
        Conversation {
            id: format!("{updated:x}-00000000"),
            title: title.into(),
            created: updated,
            updated,
            folder: None,
            messages: texts.iter().map(|(r, t)| Message { role: r.to_string(), content: t.to_string(), name: None }).collect(),
        }
    }

    #[test]
    fn recall_finds_the_conversation_that_talked_about_it() {
        let dir = tempfile::tempdir().unwrap();
        let m = Memory::at(dir.path());
        m.save(&conv("Deploy", 1_700_000_000, &[("user", "how do I deploy the website to vercel"), ("assistant", "Run vercel deploy --prod inside website/.")])).unwrap();
        m.save(&conv("Pasta", 1_700_100_000, &[("user", "a recipe for pasta with garlic"), ("assistant", "Boil the pasta, fry the garlic.")])).unwrap();
        let hit = m.recall("vercel deploy", None);
        assert!(hit.starts_with("[Deploy · 2023-11-14]"), "{hit}");
        assert!(!hit.contains("Pasta"), "{hit}");
        assert_eq!(m.recall("quantum chromodynamics", None), "Nothing in past conversations matches.");
        // The conversation asking is left out.
        assert!(!m.recall("vercel", Some("6553f100-00000000")).contains("Deploy"));
        assert_eq!(m.list().iter().map(|s| s.title.as_str()).collect::<Vec<_>>(), ["Pasta", "Deploy"]);
        // A message comes with what it refers to, but not with lines that only share words
        // like "what" or "you".
        m.save(&conv("Calls", 1_700_200_000, &[("user", "remind me to call Ayush at 9pm"), ("assistant", "I cannot set reminders; noted it.")])).unwrap();
        assert!(m.related("what did i remind you of?", None).is_some_and(|r| r.contains("Ayush")));
        assert_eq!(m.related("what would you do", None), None);
        // A new conversation starts with what was said in the others, newest first.
        let recent = m.recent(Some("6553f100-00000000"));
        assert!(recent.starts_with("[Calls · 2023-11-17]\n- remind me to call Ayush at 9pm\n[Pasta"), "{recent}");
        assert!(!recent.contains("Deploy"), "{recent}");
    }

    #[test]
    fn notes_keep_the_newest_lines_that_fit() {
        let dir = tempfile::tempdir().unwrap();
        let m = Memory::at(dir.path());
        m.remember("the owner likes short answers").unwrap();
        m.remember("line\ntwo").unwrap();
        assert_eq!(m.notes(), "- the owner likes short answers\n- line two\n");
        m.set_notes(&(0..500).map(|i| format!("- note {i}\n")).collect::<String>()).unwrap();
        let p = m.notes_for_prompt();
        assert!(p.len() <= NOTES_IN_PROMPT && p.ends_with("- note 499") && !p.contains("note 0\n"));
        assert!(m.remember("  ").is_err());
    }

    #[test]
    fn conversations_round_trip_and_ids_stay_in_their_folder() {
        let dir = tempfile::tempdir().unwrap();
        let m = Memory::at(dir.path());
        let c = Conversation::new(Some("/tmp/x".into()));
        m.save(&c).unwrap();
        assert_eq!(m.load(&c.id).unwrap().folder.as_deref(), Some("/tmp/x"));
        assert!(m.load("../memory").is_err() && m.delete("a/b").is_err());
        m.delete(&c.id).unwrap();
        assert!(m.list().is_empty());
        m.delete_all().unwrap();
    }

    #[test]
    fn bm25_prefers_rare_words_and_short_documents() {
        let docs: Vec<Vec<String>> = ["the cat sat", "the dog sat on the mat with the cat", "the the the"].iter().map(|d| words(d)).collect();
        assert_eq!(bm25(&words("cat"), &docs, 5), [0, 1]);
        assert!(bm25(&words("zebra"), &docs, 5).is_empty());
        assert_eq!(day_of(0), "1970-01-01");
    }
}
