//! Stage 1 of a message, when no shortcut in `route` matched: which tool, if any. Every
//! tool (MCP servers' too) is ranked against the message by BM25 over its name,
//! description and a few example phrasings, the way TinyAgent's ToolRAG (arXiv 2409.00608)
//! shows a small model only the tools that matter ("Less is More", arXiv 2411.15399: fewer
//! tools, better choices). The model then picks one of the top few or "answer directly"
//! with decoding held to the digit tokens, so even a 0.6B model gives a valid choice. A new
//! tool needs no routing code: its description is what finds it.

use bc_mesh::llm::Cached;

use super::format::{clip, ToolSpec, Turns};
use super::{memory, Brain};

/// Tools offered to the model besides answering directly.
pub const CANDIDATES: usize = 3;

/// Ways people ask for a built-in tool, so ranking finds it from everyday words.
fn examples(name: &str) -> &'static str {
    match name {
        "web_search" => "search look up google online internet news latest current today price weather score who won release",
        "fetch_url" => "open read website page link url site article",
        "list_dir" => "list files folder directory project what is in here",
        "read_file" => "read open show file contents summarize explain this code readme",
        "search" => "find grep where is used defined occurrences in the project code",
        "write_file" => "create write new file save",
        "edit_file" => "change edit fix rename replace update modify code line",
        "run" => "run build test compile execute command script install start",
        "recall" => "remember earlier before previous conversation last time we talked you said",
        "remind" => "remind me reminder alarm notify later at tomorrow",
        "remember" => "remember note keep in mind my name i prefer from now on",
        "clock" => "time clock timezone what time is it now in",
        "calculate" => "calculate compute math sum multiply divide percent how much is",
        "use_skill" => "skill instructions how to",
        "find_tools" => "tool integration app connect",
        _ => "",
    }
}

/// Tools the harness offers on its own terms, never as a first move.
const NEVER_FIRST: &[&str] = &["read_artifact", "task"];

/// The tools that best match `text`, best first: only those sharing a word with it.
pub fn rank<'a>(specs: &'a [ToolSpec], text: &str) -> Vec<&'a ToolSpec> {
    let offered: Vec<&ToolSpec> = specs.iter().filter(|t| !NEVER_FIRST.contains(&t.name.as_str())).collect();
    let docs: Vec<Vec<String>> = offered
        .iter()
        .map(|t| memory::words(&format!("{} {} {}", t.name.replace("__", " ").replace('_', " "), t.description, examples(&t.name))))
        .collect();
    memory::bm25(&memory::words(text), &docs, CANDIDATES).into_iter().map(|i| offered[i]).collect()
}

/// Asks the model to pick one of `candidates` or answer directly. `None` is an answer.
pub async fn pick(brain: &dyn Brain, cache: &mut Cached, text: &str, candidates: &[&ToolSpec]) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    let mut options = vec!["Answer directly: greetings, chat, writing, explanations, anything you already know.".to_string()];
    options.extend(candidates.iter().map(|t| format!("{}: {}", t.name, first_sentence(&t.description))));
    let system = format!(
        "Pick what to do first for the person's message. Reply with the number only.\n{}\nPick a tool when the message needs something you cannot know or do yourself: current facts, the person's files, the time, exact arithmetic, an action.",
        options.iter().enumerate().map(|(i, o)| format!("{}. {o}", i + 1)).collect::<Vec<_>>().join("\n")
    );
    let i = choose(brain, cache, &system, text, options.len()).await?;
    (i > 0).then(|| candidates[i - 1].name.clone())
}

fn first_sentence(text: &str) -> String {
    clip(text.split(". ").next().unwrap_or(text).trim_end_matches('.'), 140)
}

/// The model's pick of `n` numbered options for `text` (from 0), by the one digit it may
/// write; `None` when the vocabulary has no run of digit tokens to hold it to, or a device
/// from before `Sampling::allow` wrote something else.
pub async fn choose(brain: &dyn Brain, cache: &mut Cached, system: &str, text: &str, n: usize) -> Option<usize> {
    let tk = brain.tokenizer();
    let range = digits(tk, n)?;
    let turns = Turns { tokenizer: tk, think: false };
    let mut ids = tk.encode(&turns.open(system));
    ids.extend(tk.encode_segment(&turns.user(&clip(text, 2000))));
    match brain.choose(cache, &ids, range).await {
        Ok(id) if (range[0]..range[1]).contains(&id) => Some((id - range[0]) as usize),
        _ => None,
    }
}

/// The token ids of "1" to "n", when they run consecutively, as byte-level vocabularies
/// (Qwen, Llama 3) have them.
pub fn digits(tk: &bc_infer::Tokenizer, n: usize) -> Option<[u32; 2]> {
    if n == 0 || n > 9 {
        return None;
    }
    let first = tk.token_id("1")?;
    (1..n).all(|i| tk.token_id(&(i + 1).to_string()) == Some(first + i as u32)).then_some([first, first + n as u32])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec { name: name.into(), description: description.into(), parameters: json!({}) }
    }

    #[test]
    fn ranking_finds_the_tool_by_everyday_words() {
        let specs = vec![
            spec("read_file", "Read a text file of the project, with line numbers."),
            spec("web_search", "Search the web."),
            spec("recall", "Search the past conversations for something said before."),
            spec("read_artifact", "Read more of a long tool output."),
            spec("linear__create_issue", "Create an issue in Linear with a title and description."),
        ];
        let first = |t: &str| rank(&specs, t).first().map(|t| t.name.clone());
        assert_eq!(first("any news on the latest iphone price").as_deref(), Some("web_search"));
        assert_eq!(first("summarize the readme file").as_deref(), Some("read_file"));
        assert_eq!(first("what did we talk about last time").as_deref(), Some("recall"));
        assert_eq!(first("open an issue in linear for the login bug").as_deref(), Some("linear__create_issue"));
        assert!(rank(&specs, "hello there").is_empty());
        assert!(!rank(&specs, "read more of the long output").iter().any(|t| t.name == "read_artifact"));
    }
}
