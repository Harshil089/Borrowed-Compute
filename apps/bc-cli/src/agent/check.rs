//! Stage 4: the answer is checked before the person sees it, by code against what this
//! turn's tools actually returned (the "tool receipts" of arXiv 2603.10060), never by the
//! model judging itself. A small model's usual misses each have a test here: denying
//! abilities it has, promising a lookup without making it, claiming a search or a read
//! that did not happen, numbers that appear in no result, and repeating its last answer.
//! The loop answers each once: the harness does what was promised, or the model hears the
//! exact problem and answers again (Reflexion's retry, with a typed reason instead of the
//! model's own critique). A second miss keeps the answer with a line saying what could not
//! be confirmed, so nothing is hidden.

/// What is wrong with an answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Problem {
    Empty,
    /// "I cannot browse the web" with web tools at hand.
    Denies,
    /// "Let me search…" with no call.
    Promise,
    /// "I searched" or "the file says" with no such call this turn.
    Unbacked(&'static str),
    /// Numbers given as found that no result of this turn contains.
    Ungrounded(Vec<String>),
    /// "No results" when this turn's search found some.
    FalseNothing,
    /// Nearly the answer before it, word for word.
    Echo,
}

/// What the answer is checked against.
pub struct Receipts<'a> {
    pub question: &'a str,
    /// The tools the model has.
    pub tools: Vec<String>,
    /// This turn's calls, by tool name, with their results.
    pub calls: Vec<(String, String)>,
    pub last_answer: &'a str,
}

impl Receipts<'_> {
    fn called(&self, tool: &str) -> bool {
        self.calls.iter().any(|(n, _)| n == tool)
    }
}

impl Problem {
    pub fn name(&self) -> &'static str {
        match self {
            Problem::Empty => "empty",
            Problem::Denies => "denies its tools",
            Problem::Promise => "promised without a call",
            Problem::Unbacked(_) => "claims a call that was not made",
            Problem::Ungrounded(_) => "numbers in no result",
            Problem::FalseNothing => "says nothing was found",
            Problem::Echo => "repeats the last answer",
        }
    }

    /// What the model is told before it answers again.
    pub fn note(&self) -> String {
        match self {
            Problem::Empty => "Your answer was empty. Answer the person's message.".into(),
            Problem::Denies => "You do have tools for this: web_search and fetch_url reach the web. Use them or the results above; do not say you cannot.".into(),
            Problem::Promise => "You said you would look it up but made no call. Make the call now, in this answer.".into(),
            Problem::Unbacked(tool) => format!("You made no {tool} call in this answer's turn, so do not say you did. Call it, or answer without claiming it."),
            Problem::Ungrounded(n) => format!(
                "These numbers are in none of the results: {}. Give only numbers the results contain, or say the results do not give them.",
                n.join(", ")
            ),
            Problem::FalseNothing => "The search did return results (above). Answer from them, or say what in them does not fit.".into(),
            Problem::Echo => "That repeats your previous answer. Reply to the person's latest message itself.".into(),
        }
    }

    /// The line kept under an answer that still has the problem.
    pub fn caveat(&self) -> String {
        match self {
            Problem::Ungrounded(n) => format!("(Check: {} appear in none of the results; treat them as unconfirmed.)", n.join(", ")),
            Problem::Unbacked(tool) => format!("(Check: no {tool} was made for this answer.)"),
            Problem::FalseNothing => "(Check: the search did return results; see the calls above.)".into(),
            _ => String::new(),
        }
    }
}

const DENIES: &[&str] = &[
    "cannot browse", "can't browse", "unable to browse", "not able to browse", "cannot access the internet", "can't access the internet", "unable to access the internet",
    "don't have access to the internet", "do not have access to the internet", "cannot access the web", "can't access the web", "no internet access",
    "cannot access external", "can't access external", "unable to access external", "don't have real-time", "do not have real-time", "cannot access real-time",
    "can't access real-time", "cannot access the live", "can't access the live", "cannot access current", "i don't have the ability to search",
];
const PROMISES: &[&str] = &[
    "let me search", "let me look", "let me check", "let me find", "let me fetch", "i'll search", "i will search", "i'll look", "i will look", "i'll check", "i will check",
    "i'll find", "i will find", "i need to search", "i would need to search", "i would need to call", "i would need to look", "i'll fetch", "i will fetch",
];
const CLAIMS_SEARCH: &[&str] = &["i searched", "i have searched", "i've searched", "i looked up", "i've looked up", "according to my search", "my search shows", "my search found", "the search results show"];
const CLAIMS_READ: &[&str] = &["i read the file", "i've read the file", "i have read the file", "after reading the file"];
const NOTHING: &[&str] = &["no results", "found no results", "search found nothing", "returned nothing", "couldn't find any results", "could not find any results"];

/// The first problem with `answer`, if any.
pub fn check(answer: &str, r: &Receipts) -> Option<Problem> {
    let text = answer.trim();
    if text.is_empty() {
        return Some(Problem::Empty);
    }
    let lower = text.to_lowercase().replace('’', "'");
    let has = |list: &[&str]| list.iter().any(|p| lower.contains(p));
    let web = r.tools.iter().any(|t| t == "web_search");
    if web && has(DENIES) {
        return Some(Problem::Denies);
    }
    if has(PROMISES) {
        return Some(Problem::Promise);
    }
    if has(CLAIMS_SEARCH) && !r.called("web_search") {
        return Some(Problem::Unbacked("web_search"));
    }
    if has(CLAIMS_READ) && !r.called("read_file") {
        return Some(Problem::Unbacked("read_file"));
    }
    let found = r.calls.iter().any(|(n, out)| n == "web_search" && (out.starts_with("News:") || out.starts_with("Web:")));
    if found && has(NOTHING) {
        return Some(Problem::FalseNothing);
    }
    if !r.calls.is_empty() {
        let unseen = ungrounded(text, r);
        if !unseen.is_empty() {
            return Some(Problem::Ungrounded(unseen));
        }
    }
    if echoes(text, r.last_answer) {
        return Some(Problem::Echo);
    }
    None
}

/// Numbers that look like facts (four digits and more, or with a decimal point or digit
/// grouping) in the answer that neither the results nor the question contain. Years are
/// left out: an answer may say the date.
fn ungrounded(answer: &str, r: &Receipts) -> Vec<String> {
    let norm = |s: &str| s.replace(',', "");
    let evidence: String = r.calls.iter().map(|(_, out)| norm(out)).chain([norm(r.question)]).collect::<Vec<_>>().join("\n");
    let mut out: Vec<String> = Vec::new();
    for raw in answer.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == ',')) {
        let n = raw.trim_matches(|c: char| c == '.' || c == ',');
        let digits = n.chars().filter(char::is_ascii_digit).count();
        if digits == 0 {
            continue;
        }
        let plain = norm(n);
        let year = digits == 4 && !n.contains(['.', ',']) && plain.parse::<u32>().is_ok_and(|y| (1900..=2100).contains(&y));
        let factual = digits >= 4 || n.contains('.') || n.contains(',');
        if factual && !year && !evidence.contains(&plain) && !out.contains(&n.to_string()) {
            out.push(n.to_string());
        }
    }
    out
}

/// Whether two answers share nearly all their words (and the earlier one says something).
fn echoes(answer: &str, last: &str) -> bool {
    let words = |t: &str| t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect::<std::collections::HashSet<_>>();
    let (a, b) = (words(answer), words(last));
    if b.len() < 8 {
        return false;
    }
    let shared = a.intersection(&b).count();
    shared * 10 >= a.union(&b).count() * 9
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipts<'a>(calls: &[(&str, &str)], last: &'a str) -> Receipts<'a> {
        Receipts { question: "q", tools: vec!["web_search".into(), "fetch_url".into()], calls: calls.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(), last_answer: last }
    }

    #[test]
    fn the_answers_seen_in_this_project_are_caught() {
        let none = receipts(&[], "");
        // From real runs on qwen3.5-2b and qwen3-0.6b.
        assert_eq!(check("I cannot determine the current time because I cannot access the web.", &none), Some(Problem::Denies));
        assert_eq!(check("I do not have any information about it because I cannot browse the live web.", &none), Some(Problem::Denies));
        assert_eq!(check("To give you accurate information, I would need to call a web search.", &none), Some(Problem::Promise));
        assert_eq!(check("Let me search for that.", &none), Some(Problem::Promise));
        assert_eq!(check("I searched and Bitcoin is at $60,200.", &none), Some(Problem::Unbacked("web_search")));
        assert_eq!(check("  ", &none), Some(Problem::Empty));
        let last = "I'm Qwen3.5, a large language model developed by Tongyi Lab. I understand over 100 languages.";
        assert_eq!(check("I'm Qwen3.5, a large language model developed by Tongyi Lab. I understand over 100 languages.", &receipts(&[], last)), Some(Problem::Echo));
        assert_eq!(check("Hello! How can I help you today?", &none), None);
    }

    #[test]
    fn numbers_must_come_from_the_results() {
        let r = receipts(&[("web_search", "Web:\n- Bitcoin price\n  https://x\n  BTC trades at $60,212.50 today, up 1.2%.")], "");
        assert_eq!(check("Bitcoin is $60,212.50, up 1.2% (as of 2026).", &r), None);
        assert_eq!(check("Bitcoin is $61,900 today.", &r), Some(Problem::Ungrounded(vec!["61,900".into()])));
        assert_eq!(check("The search returned no results for that.", &r), Some(Problem::FalseNothing));
        // A turn with no results is not held to them.
        assert_eq!(check("17 * 23 is 391, and 2^12 is 4096.", &receipts(&[], "")), None);
        assert!(Problem::Ungrounded(vec!["61,900".into()]).caveat().contains("61,900"));
    }
}
