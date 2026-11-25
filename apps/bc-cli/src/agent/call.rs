//! Stage 2: the call for a tool stage 1 picked. Where the message itself gives the
//! arguments (search words, a named site, the place for a clock), code fills them in and
//! the model writes nothing. Otherwise the harness writes the start of the call for the
//! model, `<tool_call>{"name": "...", "arguments": {`, so the tool and the format are
//! fixed and only the arguments are the model's. Every call, the model's own included, is
//! checked against its tool's schema before it runs, and a wrong one comes back naming the
//! exact field: constrained output fixes the form of a call, not its sense (arXiv
//! 2609.23742), so the check is what tells the model what to correct.

use serde_json::{json, Value};

use super::format::ToolSpec;
use super::{facts, route};

/// Arguments code can read from the question for `tool`, so the model writes nothing.
pub fn known_args(tool: &str, question: &str) -> Option<Value> {
    match tool {
        "web_search" => Some(json!({ "query": route::search_words(question) })),
        "fetch_url" => route::domain_in(question).map(|d| json!({ "url": format!("https://{d}/"), "focus": route::search_words(question) })),
        "recall" => Some(json!({ "query": question })),
        "clock" => Some(json!({ "place": facts::place_in(question).map(|(name, _)| name).unwrap_or_default() })),
        "calculate" => facts::expression(question).map(|e| json!({ "expression": e })),
        "list_dir" => Some(json!({ "path": "." })),
        _ => None,
    }
}

/// The start of a call to `tool` that the model completes.
pub fn prefill(tool: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"{tool}\", \"arguments\": {{")
}

/// Whether `args` fit `spec`'s schema: an object, every required field there and not
/// empty, each field of its type and, for a list of choices, one of them, no unknown
/// fields. The error names what to fix.
pub fn check_args(spec: &ToolSpec, args: &Value) -> Result<(), String> {
    let Some(given) = args.as_object() else {
        return Err(format!("{}'s arguments must be a JSON object", spec.name));
    };
    let props = spec.parameters.get("properties").and_then(Value::as_object);
    let known: Vec<&str> = props.map(|p| p.keys().map(String::as_str).collect()).unwrap_or_default();
    for need in spec.parameters.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
        let missing = match given.get(need) {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => s.trim().is_empty(),
            _ => false,
        };
        if missing {
            return Err(format!("{} needs \"{need}\" ({})", spec.name, describe(props.and_then(|p| p.get(need)))));
        }
    }
    for (key, value) in given {
        let Some(schema) = props.and_then(|p| p.get(key)) else {
            if props.is_some() {
                return Err(format!("{} has no argument \"{key}\"; its arguments are: {}", spec.name, known.join(", ")));
            }
            continue;
        };
        let fits = match schema.get("type").and_then(Value::as_str) {
            // Numbers written as text ("3") are read as numbers by the tools.
            Some("integer") => value.is_i64() || value.is_u64() || value.as_str().is_some_and(|s| s.trim().parse::<i64>().is_ok()),
            Some("number") => value.is_number() || value.as_str().is_some_and(|s| s.trim().parse::<f64>().is_ok()),
            Some("boolean") => value.is_boolean(),
            Some("string") => value.is_string(),
            Some("array") => value.is_array(),
            Some("object") => value.is_object(),
            _ => true,
        };
        if !fits {
            return Err(format!("{}'s \"{key}\" must be {}", spec.name, describe(Some(schema))));
        }
        if let Some(choices) = schema.get("enum").and_then(Value::as_array) {
            if !choices.contains(value) {
                return Err(format!("{}'s \"{key}\" must be one of {}", spec.name, Value::Array(choices.clone())));
            }
        }
    }
    Ok(())
}

fn describe(schema: Option<&Value>) -> String {
    let Some(s) = schema else { return "a value".into() };
    let kind = s.get("type").and_then(Value::as_str).unwrap_or("a value");
    match s.get("description").and_then(Value::as_str) {
        Some(d) => format!("{kind}: {d}"),
        None => kind.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ToolSpec {
        ToolSpec {
            name: "run".into(),
            description: String::new(),
            parameters: json!({ "type": "object", "properties": { "command": { "type": "string", "description": "The command." }, "device": { "type": "string", "enum": ["Mac", "PC"] }, "limit": { "type": "integer" } }, "required": ["command"] }),
        }
    }

    #[test]
    fn arguments_are_checked_against_the_schema() {
        assert_eq!(check_args(&spec(), &json!({ "command": "ls" })), Ok(()));
        assert_eq!(check_args(&spec(), &json!({ "command": "ls", "limit": "3" })), Ok(()));
        assert_eq!(check_args(&spec(), &json!({})).unwrap_err(), "run needs \"command\" (string: The command.)");
        assert_eq!(check_args(&spec(), &json!({ "command": " " })).unwrap_err(), "run needs \"command\" (string: The command.)");
        assert!(check_args(&spec(), &json!({ "command": "ls", "device": "Linux" })).unwrap_err().contains("one of [\"Mac\",\"PC\"]"));
        assert!(check_args(&spec(), &json!({ "command": "ls", "cwd": "/" })).unwrap_err().contains("no argument \"cwd\""));
        assert!(check_args(&spec(), &json!({ "command": 3 })).unwrap_err().contains("must be string"));
        assert!(check_args(&spec(), &json!("ls")).is_err());
    }

    #[test]
    fn the_message_fills_what_it_can() {
        assert_eq!(known_args("web_search", "can you search the web about huggingface").unwrap()["query"], "huggingface");
        assert_eq!(known_args("fetch_url", "what is tryzenith.cloud").unwrap()["url"], "https://tryzenith.cloud/");
        assert_eq!(known_args("clock", "time in tokyo please").unwrap()["place"], "tokyo");
        assert!(known_args("edit_file", "fix the bug").is_none());
        assert!(prefill("edit_file").ends_with("\"arguments\": {"));
    }
}
