//! What the agent can do. Each tool has a risk level, which decides whether the person is
//! asked first (see `Policy` in the parent module).
//!
//! File tools work only inside the conversation's folder: every path is resolved against
//! it, and anything that leads out (`..`, an absolute path, a symlink) is refused. That is
//! a trust boundary: the model's arguments are untrusted input.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use futures_util::future::BoxFuture;
use serde::Serialize;
use serde_json::{json, Value};

use super::format::{head_tail, ToolSpec};
use super::memory::Memory;

/// Most characters of a command's output or a page the model sees.
pub const OUTPUT_CHARS: usize = 4000;
/// Most characters `read_file` returns at once; the model pages on with `offset`.
const READ_CHARS: usize = 8000;
const READ_LINES: usize = 200;
const LIST_ENTRIES: usize = 200;
const SEARCH_HITS: usize = 50;
const NEWS_RESULTS: usize = 5;
const WEB_RESULTS: usize = 5;
const FETCH_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Risk {
    /// Looks only: never asks.
    Read,
    /// Changes files in the folder, or the notes.
    Write,
    /// Runs a command on a device.
    Run,
    /// Reaches the internet.
    Net,
}

/// A command for a device to run on a copy of the folder.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub folder: PathBuf,
    pub argv: Vec<String>,
    /// A device name from [`Tools::devices`]; `None` is the first (this computer, when it
    /// can run jobs).
    pub device: Option<String>,
}

/// Runs a command and returns its exit code and output. The app provides it: on this
/// computer in its job sandbox, or on a paired device as a job.
pub type Runner = Arc<dyn Fn(RunRequest) -> BoxFuture<'static, Result<(i32, String)>> + Send + Sync>;

/// The tools one conversation has.
#[derive(Clone)]
pub struct Tools {
    /// The folder file tools work in, canonical; `None` leaves them out.
    pub root: Option<PathBuf>,
    pub memory: Memory,
    /// The conversation using them, left out of `recall`.
    pub conversation: String,
    pub runner: Option<Runner>,
    /// Devices `run` can use, the default first.
    pub devices: Vec<String>,
}

fn spec(name: &str, description: &str, properties: Value, required: &[&str]) -> ToolSpec {
    ToolSpec { name: name.into(), description: description.into(), parameters: json!({ "type": "object", "properties": properties, "required": required }) }
}

impl Tools {
    pub fn new(root: Option<&Path>, memory: Memory, conversation: String, runner: Option<Runner>, devices: Vec<String>) -> Result<Self> {
        let root = match root {
            Some(r) => Some(r.canonicalize().with_context(|| format!("the folder {} is not there", r.display()))?),
            None => None,
        };
        Ok(Tools { root, memory, conversation, runner, devices })
    }

    /// The tools this conversation offers the model.
    pub fn specs(&self) -> Vec<ToolSpec> {
        let p = |d: &str| json!({ "type": "string", "description": d });
        let mut out = Vec::new();
        if self.root.is_some() {
            out.push(spec("list_dir", "List the files and folders in a folder of the project.", json!({ "path": p("Folder, relative to the project. Default: the project itself.") }), &[]));
            out.push(spec(
                "read_file",
                "Read a text file of the project, with line numbers. Long files come a part at a time: ask again with a later offset.",
                json!({ "path": p("File, relative to the project."), "offset": { "type": "integer", "description": "First line to read, from 1." }, "limit": { "type": "integer", "description": "Lines to read, at most 200." } }),
                &["path"],
            ));
            out.push(spec("search", "Find lines containing some text (any case) in the project's files.", json!({ "text": p("The text to find."), "path": p("Folder or file to search, relative to the project. Default: all of it.") }), &["text"]));
            out.push(spec("write_file", "Create or replace a file of the project with this content.", json!({ "path": p("File, relative to the project."), "content": p("The whole new content.") }), &["path", "content"]));
            out.push(spec(
                "edit_file",
                "Replace one exact piece of text in a file of the project. The old text must appear exactly once: include enough of the lines around it.",
                json!({ "path": p("File, relative to the project."), "old": p("The exact text to replace."), "new": p("The text to put there.") }),
                &["path", "old", "new"],
            ));
            if self.runner.is_some() && !self.devices.is_empty() {
                out.push(spec(
                    "run",
                    "Run a command (a build, tests, a script) on a copy of the project and get its output. Changes the command makes to files are not kept; change files with write_file or edit_file.",
                    json!({ "command": p("The command, such as `cargo test` or `npm run build`."), "device": { "type": "string", "enum": self.devices, "description": format!("Where to run it. Default: {}.", self.devices[0]) } }),
                    &["command"],
                ));
            }
        }
        out.push(spec("web_search", "Search the web. Gives titles, addresses and snippets of the top results; read one in full with fetch_url.", json!({ "query": p("What to search for.") }), &["query"]));
        out.push(spec("fetch_url", "Read a web page (https) as text.", json!({ "url": p("The https address.") }), &["url"]));
        out.push(spec("recall", "Search the past conversations for something said before.", json!({ "query": p("Words to look for.") }), &["query"]));
        out.push(spec("remember", "Keep a short note for every future conversation: a fact about the person, a preference, a decision.", json!({ "note": p("The note, one line.") }), &["note"]));
        out
    }

    pub fn risk(name: &str) -> Risk {
        match name {
            "write_file" | "edit_file" | "remember" => Risk::Write,
            "run" => Risk::Run,
            "fetch_url" | "web_search" => Risk::Net,
            _ => Risk::Read,
        }
    }

    /// Runs one call; the error is what the model is told.
    pub async fn call(&self, name: &str, args: &Value) -> Result<String> {
        let s = |k: &str| args.get(k).and_then(|v| v.as_str().map(String::from).or_else(|| (!v.is_null()).then(|| v.to_string())));
        let need = |k: &str| s(k).with_context(|| format!("{name} needs \"{k}\""));
        let num = |k: &str| s(k).and_then(|v| v.trim().parse::<usize>().ok());
        match name {
            "list_dir" => self.list_dir(&s("path").unwrap_or_else(|| ".".into())),
            "read_file" => self.read_file(&need("path")?, num("offset").unwrap_or(1), num("limit").unwrap_or(READ_LINES)),
            "search" => self.search(&need("text")?, &s("path").unwrap_or_else(|| ".".into())),
            "write_file" => {
                let path = self.resolve(&need("path")?)?;
                let content = need("content")?;
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                std::fs::write(&path, &content)?;
                Ok(format!("Wrote {} bytes to {}.", content.len(), self.shown(&path)))
            }
            "edit_file" => {
                let path = self.resolve(&need("path")?)?;
                let (old, new) = (need("old")?, need("new")?);
                let text = std::fs::read_to_string(&path).with_context(|| format!("cannot read {}", self.shown(&path)))?;
                match text.matches(old.as_str()).count() {
                    0 => bail!("the old text is not in {}; read the file and copy it exactly", self.shown(&path)),
                    1 => {
                        std::fs::write(&path, text.replacen(old.as_str(), &new, 1))?;
                        Ok(format!("Edited {}.", self.shown(&path)))
                    }
                    n => bail!("the old text is in {} {n} times; include more of the lines around it", self.shown(&path)),
                }
            }
            "run" => {
                let runner = self.runner.clone().context("running commands is not available here")?;
                let folder = self.root.clone().context("choose a folder first")?;
                let argv = crate::flows::split_command(&need("command")?)?;
                let device = s("device").filter(|d| !d.is_empty());
                if let Some(d) = &device {
                    if !self.devices.contains(d) {
                        bail!("no device called {d}; use one of: {}", self.devices.join(", "));
                    }
                }
                let (code, out) = runner(RunRequest { folder, argv, device }).await?;
                Ok(format!("Exit code {code}.\n{}", head_tail(&out, OUTPUT_CHARS)))
            }
            "fetch_url" => fetch(&need("url")?).await,
            "web_search" => web_search(&need("query")?).await,
            "recall" => Ok(self.memory.recall(&need("query")?, Some(&self.conversation))),
            "remember" => {
                self.memory.remember(&need("note")?)?;
                Ok("Noted.".to_string())
            }
            _ => bail!("there is no tool called {name}"),
        }
    }

    /// `path` inside the folder, or an error if it leads out.
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        let root = self.root.as_ref().context("no folder is open for file tools")?;
        let p = Path::new(path.trim());
        let rel = if p.is_absolute() { p.strip_prefix(root).map_err(|_| anyhow::anyhow!("{path} is outside the project"))? } else { p };
        // Lexically first: `..` may not climb above the folder.
        let mut out = root.clone();
        for c in rel.components() {
            match c {
                Component::Normal(x) => out.push(x),
                Component::CurDir => {}
                Component::ParentDir => {
                    if out == *root {
                        bail!("{path} is outside the project");
                    }
                    out.pop();
                }
                _ => bail!("{path} is outside the project"),
            }
        }
        // Then on disk: the deepest part that exists, symlinks followed, must still be inside.
        let mut existing = out.as_path();
        while !existing.exists() {
            existing = existing.parent().context("no such place")?;
        }
        if !existing.canonicalize()?.starts_with(root) {
            bail!("{path} leads outside the project");
        }
        Ok(out)
    }

    fn shown(&self, path: &Path) -> String {
        let root = self.root.as_deref().unwrap_or(Path::new(""));
        let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/");
        if rel.is_empty() { ".".into() } else { rel }
    }

    pub fn list_dir(&self, path: &str) -> Result<String> {
        let dir = self.resolve(path)?;
        let mut entries: Vec<(bool, String, u64)> = std::fs::read_dir(&dir)
            .with_context(|| format!("{} is not a folder", self.shown(&dir)))?
            .flatten()
            .filter(|e| e.file_name() != ".git")
            .map(|e| {
                let meta = e.metadata().ok();
                (meta.as_ref().is_some_and(|m| m.is_dir()), e.file_name().to_string_lossy().into_owned(), meta.map_or(0, |m| m.len()))
            })
            .collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let total = entries.len();
        let mut out: Vec<String> = entries.into_iter().take(LIST_ENTRIES).map(|(d, n, len)| if d { format!("{n}/") } else { format!("{n} ({len} bytes)") }).collect();
        if total > LIST_ENTRIES {
            out.push(format!("… and {} more", total - LIST_ENTRIES));
        }
        Ok(if out.is_empty() { "The folder is empty.".into() } else { out.join("\n") })
    }

    fn read_file(&self, path: &str, offset: usize, limit: usize) -> Result<String> {
        let file = self.resolve(path)?;
        let bytes = std::fs::read(&file).with_context(|| format!("cannot read {}", self.shown(&file)))?;
        if bytes.contains(&0) {
            bail!("{} is not a text file", self.shown(&file));
        }
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let first = offset.max(1);
        let mut out = String::new();
        let mut last = first - 1;
        for (i, line) in lines.iter().enumerate().skip(first - 1).take(limit.clamp(1, READ_LINES)) {
            let l = format!("{:>5}  {line}\n", i + 1);
            if out.len() + l.len() > READ_CHARS {
                break;
            }
            out.push_str(&l);
            last = i + 1;
        }
        if lines.is_empty() {
            return Ok(format!("{} is empty.", self.shown(&file)));
        }
        if last < lines.len() {
            out.push_str(&format!("[lines {first}–{last} of {}; read on with offset {}]", lines.len(), last + 1));
        }
        Ok(out)
    }

    fn search(&self, text: &str, path: &str) -> Result<String> {
        let start = self.resolve(path)?;
        let needle = text.to_lowercase();
        if needle.is_empty() {
            bail!("search needs some text");
        }
        let mut hits = Vec::new();
        for entry in ignore::WalkBuilder::new(&start).build().flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) || entry.metadata().map_or(true, |m| m.len() > FETCH_BYTES as u64) {
                continue;
            }
            let Ok(bytes) = std::fs::read(entry.path()) else { continue };
            if bytes.contains(&0) {
                continue;
            }
            for (i, line) in String::from_utf8_lossy(&bytes).lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    hits.push(format!("{}:{}: {}", self.shown(entry.path()), i + 1, super::format::clip(line.trim(), 200)));
                    if hits.len() >= SEARCH_HITS {
                        hits.push(format!("[stopped at {SEARCH_HITS} matches; search a smaller folder]"));
                        return Ok(hits.join("\n"));
                    }
                }
            }
        }
        Ok(if hits.is_empty() { format!("No line contains \"{text}\".") } else { hits.join("\n") })
    }
}

/// A web page as text: https only, at most 1 MB read.
async fn fetch(url: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(url.trim()).context("not a web address")?;
    if parsed.scheme() != "https" {
        bail!("only https addresses");
    }
    Ok(head_tail(&download(parsed).await?, OUTPUT_CHARS))
}

/// Bing's RSS results, news and web at once: no account or key, and plain XML. Web
/// results alone answer "today's news" with encyclopedia pages, and a model then searches
/// again and again. DuckDuckGo's keyless page puts automated clients behind a bot check,
/// which the agent must not try to pass.
async fn web_search(query: &str) -> Result<String> {
    let rss = |path: &str| {
        let mut url = reqwest::Url::parse("https://www.bing.com/").expect("a fixed address").join(path).expect("a fixed path");
        url.query_pairs_mut().append_pair("q", query.trim()).append_pair("format", "rss");
        download(url)
    };
    let (news, web) = tokio::join!(rss("news/search"), rss("search"));
    if let (Err(e), Err(_)) = (&news, &web) {
        bail!("the search did not answer: {e:#}");
    }
    let news = news.map(|x| search_results(&x, NEWS_RESULTS, true)).unwrap_or_default();
    let web = web.map(|x| search_results(&x, WEB_RESULTS, false)).unwrap_or_default();
    let mut out = Vec::new();
    for (head, items) in [("News", news), ("Web", web)] {
        if !items.is_empty() {
            out.push(format!("{head}:\n{}", items.join("\n")));
        }
    }
    Ok(if out.is_empty() { "No results.".to_string() } else { out.join("\n\n") })
}

/// The items of an RSS search page: title (with its date, for news), address, snippet.
fn search_results(xml: &str, max: usize, dated: bool) -> Vec<String> {
    let tag = |item: &str, name: &str| -> String {
        let open = format!("<{name}>");
        let start = item.find(&open).map(|i| i + open.len());
        let end = item.find(&format!("</{name}>"));
        match (start, end) {
            (Some(a), Some(b)) if a <= b => html_text(&html_text(&item[a..b])).replace('\n', " "),
            _ => String::new(),
        }
    };
    xml.split("<item>")
        .skip(1)
        .take(max)
        .map(|item| {
            let link = tag(item, "link");
            // News links go through a Bing redirect over http; the article's own https
            // address is in its `url` parameter, and fetch_url takes only https.
            let link = reqwest::Url::parse(&link)
                .ok()
                .and_then(|u| u.query_pairs().find(|(k, _)| k == "url").map(|(_, v)| v.into_owned()))
                .unwrap_or(link);
            // "Mon, 23 Nov 2026 12:34:00 GMT" to "Mon, 23 Nov 2026".
            let date = tag(item, "pubDate");
            let date = if dated && date.len() >= 16 { format!(" ({})", &date[..16]) } else { String::new() };
            format!("- {}{date}\n  {link}\n  {}", tag(item, "title"), tag(item, "description"))
        })
        .collect()
}

/// A page's text, HTML reduced to its words, at most [`FETCH_BYTES`] of it.
async fn download(parsed: reqwest::Url) -> Result<String> {
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(20)).build()?;
    let mut resp = client.get(parsed).header("user-agent", "BorrowedCompute-agent").send().await?;
    let status = resp.status();
    let html = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.contains("html"));
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        body.extend_from_slice(&chunk);
        if body.len() >= FETCH_BYTES {
            body.truncate(FETCH_BYTES);
            break;
        }
    }
    let text = String::from_utf8_lossy(&body);
    let text = if html { html_text(main_part(&text)) } else { text.into_owned() };
    if !status.is_success() {
        bail!("the page answered {status}: {}", super::format::clip(&text, 300));
    }
    Ok(text)
}

/// `&#8211;` and `&#x2014;` as the characters they stand for.
fn numeric_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("&#") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 2..];
        let end = tail.find(';').filter(|&e| e <= 8);
        let code = end.and_then(|e| match tail[..e].strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => tail[..e].parse().ok(),
        });
        match (end, code.and_then(char::from_u32)) {
            (Some(e), Some(c)) => {
                out.push(c);
                rest = &tail[e + 1..];
            }
            _ => {
                out.push_str("&#");
                rest = tail;
            }
        }
    }
    out.push_str(rest);
    out
}

/// A page's article (or main part) when it marks one, without the menus around it, which
/// would otherwise fill the model's room before the story starts.
fn main_part(html: &str) -> &str {
    let lower = html.to_ascii_lowercase();
    for (open, close) in [("<article", "</article>"), ("<main", "</main>")] {
        if let (Some(a), Some(b)) = (lower.find(open), lower.rfind(close)) {
            if a < b {
                return &html[a..b + close.len()];
            }
        }
    }
    html
}

/// The readable text of a page: no scripts, styles or tags, whitespace collapsed.
fn html_text(html: &str) -> String {
    // ASCII lowercase keeps every byte where it is, so indices into both agree.
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0;
    while i < html.len() {
        if lower[i..].starts_with("<script") || lower[i..].starts_with("<style") {
            let close = if lower[i..].starts_with("<script") { "</script>" } else { "</style>" };
            i = lower[i..].find(close).map_or(html.len(), |e| i + e + close.len());
            continue;
        }
        if html[i..].starts_with('<') {
            let end = html[i..].find('>').map_or(html.len(), |e| i + e + 1);
            // Block tags end a line.
            let tag = &lower[i..end];
            if ["<p", "<br", "<div", "<li", "<h1", "<h2", "<h3", "<tr", "</p", "</div", "</h"].iter().any(|t| tag.starts_with(t)) {
                out.push('\n');
            }
            i = end;
            continue;
        }
        let next = html[i..].find('<').map_or(html.len(), |e| i + e);
        out.push_str(&html[i..next]);
        i = next;
    }
    let out = numeric_entities(&out.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")).replace("&amp;", "&");
    let mut lines: Vec<String> = out.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|l| !l.is_empty()).collect();
    lines.dedup();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(dir: &Path) -> Tools {
        Tools::new(Some(dir), Memory::at(&dir.join(".mem")), "c".into(), None, Vec::new()).unwrap()
    }

    #[test]
    fn paths_stay_inside_the_folder() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("project");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(outer.path().join("secret.txt"), "no").unwrap();
        let t = tools(&root);
        let root = root.canonicalize().unwrap();
        assert_eq!(t.resolve("src/../a.txt").unwrap(), root.join("a.txt"));
        assert_eq!(t.resolve("./new/dir/file").unwrap(), root.join("new/dir/file"));
        assert_eq!(t.resolve(root.join("src").to_str().unwrap()).unwrap(), root.join("src"));
        for bad in ["../secret.txt", "src/../../secret.txt", "/etc/passwd"] {
            assert!(t.resolve(bad).is_err(), "{bad} must be refused");
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outer.path(), root.join("out")).unwrap();
            assert!(t.resolve("out/secret.txt").is_err(), "a symlink out is refused");
            assert!(t.resolve("out/new.txt").is_err(), "and so is a new file through it");
        }
    }

    #[tokio::test]
    async fn file_tools_read_write_edit_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let t = tools(dir.path());
        let call = |n: &'static str, a: Value| {
            let t = t.clone();
            async move { t.call(n, &a).await }
        };
        assert_eq!(call("write_file", json!({"path": "src/a.rs", "content": "fn main() {\n    println!(\"hi\");\n}\n"})).await.unwrap(), "Wrote 34 bytes to src/a.rs.");
        assert_eq!(call("read_file", json!({"path": "src/a.rs", "offset": "2", "limit": 1})).await.unwrap(), "    2      println!(\"hi\");\n[lines 2–2 of 3; read on with offset 3]");
        assert!(call("edit_file", json!({"path": "src/a.rs", "old": "nope", "new": "x"})).await.unwrap_err().to_string().contains("not in src/a.rs"));
        call("write_file", json!({"path": "b.txt", "content": "x x"})).await.unwrap();
        assert!(call("edit_file", json!({"path": "b.txt", "old": "x", "new": "y"})).await.unwrap_err().to_string().contains("2 times"));
        call("edit_file", json!({"path": "src/a.rs", "old": "\"hi\"", "new": "\"hello\""})).await.unwrap();
        assert_eq!(call("search", json!({"text": "HELLO"})).await.unwrap(), "src/a.rs:2: println!(\"hello\");");
        assert_eq!(call("list_dir", json!({})).await.unwrap(), "src/\nb.txt (3 bytes)");
        assert!(call("read_file", json!({})).await.unwrap_err().to_string().contains("needs \"path\""));
        assert!(call("nope", json!({})).await.is_err());
        // Without a runner there is no `run`, and without a folder no file tools.
        assert!(!t.specs().iter().any(|s| s.name == "run"));
        let bare = Tools::new(None, Memory::at(dir.path()), "c".into(), None, Vec::new()).unwrap();
        assert_eq!(bare.specs().iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["web_search", "fetch_url", "recall", "remember"]);
        assert!(bare.call("read_file", &json!({"path": "b.txt"})).await.is_err());
    }

    #[tokio::test]
    async fn run_goes_to_the_runner_and_fetch_wants_https() {
        let dir = tempfile::tempdir().unwrap();
        let runner: Runner = Arc::new(|r: RunRequest| Box::pin(async move { Ok((3, format!("{:?} on {:?}", r.argv, r.device))) }));
        let t = Tools::new(Some(dir.path()), Memory::at(dir.path()), "c".into(), Some(runner), vec!["This computer".into(), "mini".into()]).unwrap();
        assert_eq!(t.call("run", &json!({"command": "cargo test --lib", "device": "mini"})).await.unwrap(), "Exit code 3.\n[\"cargo\", \"test\", \"--lib\"] on Some(\"mini\")");
        assert!(t.call("run", &json!({"command": "ls", "device": "elsewhere"})).await.is_err());
        assert!(fetch("http://example.com").await.unwrap_err().to_string().contains("only https"));
        assert_eq!(Tools::risk("run"), Risk::Run);
        assert_eq!(Tools::risk("read_file"), Risk::Read);
    }

    #[test]
    fn search_results_read_rss_items() {
        let xml = "<rss><channel><title>Bing</title><item><title>Race &amp; result</title><link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;url=https%3a%2f%2fexample.com%2fa&amp;c=1</link><description>Winner &lt;b&gt;took&lt;/b&gt;\nit</description><pubDate>Mon, 23 Nov 2026 12:34:00 GMT</pubDate></item><item><title>Two</title><link>https://example.com/b</link></item></channel></rss>";
        assert_eq!(search_results(xml, 5, true), ["- Race & result (Mon, 23 Nov 2026)\n  https://example.com/a\n  Winner took it", "- Two\n  https://example.com/b\n  "]);
        assert_eq!(search_results(xml, 1, false), ["- Race & result\n  https://example.com/a\n  Winner took it"]);
        assert!(search_results("<rss></rss>", 5, false).is_empty());
    }

    #[test]
    fn pages_become_text() {
        assert_eq!(html_text("A &#8211; B &#x2014; C &#oops; &amp;#38;"), "A – B — C &#oops; &#38;");
        assert_eq!(main_part("<nav>menu</nav><article><p>Story</p></article><footer>x</footer>"), "<article><p>Story</p></article>");
        let t = html_text("<html><head><style>p{}</style><script>var x = '<p>';</script></head><body><h1>Title</h1><p>One &amp; two</p><div>three</div></body></html>");
        assert_eq!(t, "Title\nOne & two\nthree");
    }
}
