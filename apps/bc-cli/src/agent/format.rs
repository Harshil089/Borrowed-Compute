//! How a conversation with tools is written for the model, and how its tool calls are
//! read back.
//!
//! Tools are listed in the system prompt as JSON schemas inside `<tools>…</tools>`, and the
//! model calls one by writing `<tool_call>{"name": …, "arguments": {…}}</tool_call>`: the
//! Hermes format Qwen2.5, Qwen3 and Qwen3.5 were trained on. Qwen3.5 sometimes writes the
//! XML form instead (`<function=name><parameter=key>value</parameter></function>`), and
//! Llama 3.1 a bare JSON object; both are read too. Results go back as
//! `<tool_response>…</tool_response>` in a user turn (ChatML) or as the `ipython` role
//! (Llama 3).
//!
//! A conversation is fed to the model as segments of text, each appended to the tokens
//! already fed, so a conversation's earlier tokens never change and the stages keep them.

use bc_infer::tokenizer::Template;
use bc_infer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A tool as the model sees it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON schema of the arguments.
    pub parameters: Value,
}

/// One call the model wrote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: Value,
}

/// An assistant turn, read: the text for the person, the model's reasoning if it thought
/// first, and its calls (each one, or why it could not be read).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reply {
    pub text: String,
    pub thought: String,
    pub calls: Vec<Result<Call, String>>,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// The system prompt: the base instructions, then the tools.
pub fn system_prompt(base: &str, tools: &[ToolSpec]) -> String {
    if tools.is_empty() {
        return base.to_string();
    }
    let mut s = format!("{base}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for t in tools {
        s.push('\n');
        s.push_str(&json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } }).to_string());
    }
    s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    s
}

/// The writer of a conversation's segments for one model's chat format.
pub struct Turns<'a> {
    pub tokenizer: &'a Tokenizer,
    /// Let the model reason before it answers (models with `<think>`).
    pub think: bool,
}

impl Turns<'_> {
    fn template(&self) -> Template {
        self.tokenizer.template
    }

    /// The opening: the system prompt.
    pub fn open(&self, system: &str) -> String {
        match self.template() {
            Template::ChatMl => format!("<|im_start|>system\n{system}<|im_end|>\n"),
            Template::Llama3 => format!("<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"),
            Template::Plain => format!("{system}\n"),
        }
    }

    /// The person's message, then the start of the model's answer.
    pub fn user(&self, text: &str) -> String {
        match self.template() {
            Template::ChatMl => format!("<|im_start|>user\n{text}<|im_end|>\n{}", self.answer()),
            Template::Llama3 => format!("<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>{}", self.answer()),
            Template::Plain => format!("{text}\n"),
        }
    }

    /// Tool results (tool name, result), then the start of the model's next answer.
    pub fn results(&self, results: &[(String, String)]) -> String {
        match self.template() {
            Template::ChatMl => {
                let body: Vec<String> = results.iter().map(|(_, r)| format!("<tool_response>\n{r}\n</tool_response>")).collect();
                format!("<|im_start|>user\n{}<|im_end|>\n{}", body.join("\n"), self.answer())
            }
            Template::Llama3 => {
                let body: String = results.iter().map(|(_, r)| format!("<|start_header_id|>ipython<|end_header_id|>\n\n{r}<|eot_id|>")).collect();
                format!("{body}{}", self.answer())
            }
            Template::Plain => results.iter().map(|(n, r)| format!("{n}: {r}\n")).collect(),
        }
    }

    /// What ends the model's answer, after the tokens it wrote (it stops before writing it).
    pub fn close(&self) -> &'static str {
        match self.template() {
            Template::ChatMl => "<|im_end|>\n",
            Template::Llama3 => "<|eot_id|>",
            Template::Plain => "\n",
        }
    }

    /// The start of an answer. Without thinking, a model that has `<think>` gets an empty
    /// one, which is how Qwen3 and Qwen3.5 are told to answer straight away: far fewer
    /// tokens a step.
    fn answer(&self) -> String {
        let head = match self.template() {
            Template::ChatMl => "<|im_start|>assistant\n",
            Template::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            Template::Plain => "",
        };
        if !self.think && self.tokenizer.token_id(THINK_OPEN).is_some() {
            format!("{head}{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")
        } else {
            head.to_string()
        }
    }
}

/// Reads an assistant turn.
pub fn parse(raw: &str) -> Reply {
    let mut reply = Reply::default();
    let mut rest = raw;
    // Reasoning comes first; with thinking off the block is empty.
    if let Some(end) = rest.find(THINK_CLOSE) {
        let inner = rest[..end].trim_start().strip_prefix(THINK_OPEN).unwrap_or(&rest[..end]);
        reply.thought = inner.trim().to_string();
        rest = &rest[end + THINK_CLOSE.len()..];
    }
    let mut text = String::new();
    while let Some(at) = rest.find(CALL_OPEN) {
        text.push_str(&rest[..at]);
        let body = &rest[at + CALL_OPEN.len()..];
        let (inner, after) = match body.find(CALL_CLOSE) {
            Some(end) => (&body[..end], &body[end + CALL_CLOSE.len()..]),
            // The answer ended inside the call: read what is there.
            None => (body, ""),
        };
        reply.calls.push(read_call(inner.trim()));
        rest = after;
    }
    text.push_str(rest);
    let text = text.trim();
    // Llama 3.1 calls a tool by answering with only a JSON object.
    if reply.calls.is_empty() && text.starts_with('{') && text.ends_with('}') {
        if let Ok(call) = read_json(text) {
            reply.calls.push(Ok(call));
            return reply;
        }
    }
    reply.text = text.to_string();
    reply
}

fn read_call(inner: &str) -> Result<Call, String> {
    if inner.starts_with('{') {
        return read_json(inner);
    }
    if inner.contains("<function=") {
        return read_xml(inner);
    }
    Err(format!("not a tool call: {}", clip(inner, 200)))
}

fn read_json(inner: &str) -> Result<Call, String> {
    let v: Value = serde_json::from_str(inner).map_err(|e| format!("the call is not valid JSON ({e}): {}", clip(inner, 200)))?;
    let name = v.get("name").and_then(Value::as_str).ok_or("the call has no \"name\"")?.to_string();
    let mut arguments = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or_else(|| json!({}));
    // Some models write the arguments as a JSON string.
    if let Value::String(s) = &arguments {
        arguments = serde_json::from_str(s).map_err(|e| format!("the arguments of {name} are not valid JSON ({e})"))?;
    }
    if !arguments.is_object() {
        return Err(format!("the arguments of {name} are not an object"));
    }
    Ok(Call { name, arguments })
}

fn read_xml(inner: &str) -> Result<Call, String> {
    let start = inner.find("<function=").unwrap() + "<function=".len();
    let end = inner[start..].find('>').ok_or("an unfinished <function=")? + start;
    let name = inner[start..end].trim().to_string();
    let mut arguments = serde_json::Map::new();
    let mut rest = &inner[end + 1..];
    while let Some(at) = rest.find("<parameter=") {
        let s = at + "<parameter=".len();
        let e = rest[s..].find('>').ok_or("an unfinished <parameter=")? + s;
        let key = rest[s..e].trim().to_string();
        let body = &rest[e + 1..];
        let close = body.find("</parameter>").unwrap_or(body.len());
        // One newline on each side is the format's, not the value's.
        let value = body[..close].strip_prefix('\n').unwrap_or(&body[..close]);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(key, Value::String(value.to_string()));
        rest = &body[(close + "</parameter>".len()).min(body.len())..];
    }
    Ok(Call { name, arguments: Value::Object(arguments) })
}

/// At most `max` characters of `s`, cut at a character boundary.
pub fn clip(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

/// `s` cut to about `max` characters, keeping its start and end (where a command's
/// errors and summary usually are).
pub fn head_tail(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let half = max / 2;
    let head: String = s.chars().take(half).collect();
    let tail: String = s.chars().skip(n - half).collect();
    format!("{head}\n… [{} characters left out] …\n{tail}", n - 2 * half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_hermes_calls_and_the_text_around_them() {
        let r = parse("<think>\nlook first\n</think>\n\nLet me check.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"list_dir\", \"arguments\": \"{}\"}\n</tool_call>");
        assert_eq!(r.thought, "look first");
        assert_eq!(r.text, "Let me check.");
        assert_eq!(r.calls, vec![Ok(Call { name: "read_file".into(), arguments: json!({"path": "a.txt"}) }), Ok(Call { name: "list_dir".into(), arguments: json!({}) })]);
    }

    #[test]
    fn reads_the_xml_form_and_bare_json() {
        let r = parse("<tool_call>\n<function=write_file>\n<parameter=path>\nb.txt\n</parameter>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(r.calls, vec![Ok(Call { name: "write_file".into(), arguments: json!({"path": "b.txt", "content": "line one\nline two"}) })]);
        let r = parse("{\"name\": \"search\", \"parameters\": {\"text\": \"fn main\"}}");
        assert_eq!(r.calls, vec![Ok(Call { name: "search".into(), arguments: json!({"text": "fn main"}) })]);
        assert_eq!(r.text, "");
    }

    #[test]
    fn a_broken_call_says_why_and_plain_text_has_none() {
        let r = parse("<tool_call>{\"name\": \"run\", \"arguments\": {\"command\": }</tool_call>");
        assert!(matches!(&r.calls[..], [Err(e)] if e.contains("not valid JSON")), "{r:?}");
        let r = parse("<tool_call>{\"arguments\": {}}</tool_call>");
        assert!(matches!(&r.calls[..], [Err(e)] if e.contains("no \"name\"")));
        // An answer cut off inside a call still reads what is there.
        let r = parse("<tool_call>{\"name\": \"list_dir\", \"arguments\": {}}");
        assert!(r.calls[0].is_ok());
        let r = parse("Done: {braces} in text are fine.");
        assert!(r.calls.is_empty());
        assert_eq!(r.text, "Done: {braces} in text are fine.");
    }

    #[test]
    fn long_output_keeps_both_ends() {
        let s: String = (0..1000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        let cut = head_tail(&s, 100);
        assert!(cut.starts_with(&s[..50]) && cut.ends_with(&s[950..]) && cut.contains("900 characters left out"));
        assert_eq!(head_tail("short", 100), "short");
        assert_eq!(clip("héllo", 2), "hé…");
    }

    #[test]
    fn the_system_prompt_lists_each_tool() {
        let s = system_prompt("Be brief.", &[ToolSpec { name: "list_dir".into(), description: "List a folder.".into(), parameters: json!({"type": "object", "properties": {}}) }]);
        assert!(s.starts_with("Be brief.\n\n# Tools"));
        assert!(s.contains("{\"function\":{\"description\":\"List a folder.\",\"name\":\"list_dir\""), "{s}");
        assert_eq!(system_prompt("Be brief.", &[]), "Be brief.");
    }
}
