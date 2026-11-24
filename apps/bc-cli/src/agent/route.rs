//! What the harness does before the model answers, so a model too small to decide on a
//! tool still answers from real results. Two fast paths read the message itself (a
//! reminder with a time; news or anything on the web). Otherwise a decision step asks
//! the model to pick one numbered action, with decoding held to those digits
//! (`Sampling::allow`): even a model that never writes a tool call picks a valid one.

use bc_infer::Tokenizer;
use bc_mesh::llm::Cached;

use super::format::Turns;
use super::Brain;

/// What to do first for a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Answer,
    Web,
    Files,
    Past,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Route::Answer => "answer",
            Route::Web => "web",
            Route::Files => "files",
            Route::Past => "past conversations",
        }
    }

    fn option(self) -> &'static str {
        match self {
            Route::Answer => "Answer directly: greetings, chat, writing, maths, code questions, explanations, anything you already know.",
            Route::Web => "Search the web: news, current events, prices, weather, sports, people, anything recent or that you may not know.",
            Route::Files => "Look in the chosen project folder: its files or code.",
            Route::Past => "Search past conversations: the person refers to something said before that is not in this conversation.",
        }
    }
}

/// Asks the model which of `options` fits `text` best, by the one digit it may write.
/// Gives `Answer` when the vocabulary has no run of digit tokens to hold it to, or when
/// a device from before `Sampling::allow` wrote something else.
pub async fn decide(brain: &dyn Brain, cache: &mut Cached, text: &str, options: &[Route]) -> Route {
    let tk = brain.tokenizer();
    let Some(range) = digits(tk, options.len()) else { return Route::Answer };
    let list: String = options.iter().enumerate().map(|(i, r)| format!("{}. {}\n", i + 1, r.option())).collect();
    // Worked examples keep a small model from answering 1 to everything.
    let examples: String = EXAMPLES
        .iter()
        .filter_map(|(message, route)| options.iter().position(|o| o == route).map(|i| format!("\"{message}\" -> {}\n", i + 1)))
        .collect();
    let system = format!("Pick what to do first for the person's message. Reply with the number only.\n{list}\nExamples:\n{examples}");
    let turns = Turns { tokenizer: tk, think: false };
    let mut ids = tk.encode(&turns.open(&system));
    ids.extend(tk.encode_segment(&turns.user(&super::format::clip(text, 2000))));
    match brain.choose(cache, &ids, range).await {
        Ok(id) if (range[0]..range[1]).contains(&id) => options[(id - range[0]) as usize],
        _ => Route::Answer,
    }
}

const EXAMPLES: &[(&str, Route)] = &[
    ("hello", Route::Answer),
    ("who is the prime minister of Japan now?", Route::Web),
    ("what does the main function do?", Route::Files),
    ("write a short poem about the sea", Route::Answer),
    ("what was the name of the hotel I told you about?", Route::Past),
    ("how much does an iPhone 17 cost?", Route::Web),
    ("summarize the readme", Route::Files),
    ("what is 12 squared?", Route::Answer),
];

/// The token ids of "1" to "n", when they run consecutively, as byte-level vocabularies
/// (Qwen, Llama 3) have them.
fn digits(tk: &Tokenizer, n: usize) -> Option<[u32; 2]> {
    let first = tk.token_id("1")?;
    (1..n).all(|i| tk.token_id(&(i + 1).to_string()) == Some(first + i as u32)).then_some([first, first + n as u32])
}

/// Whether the message points back at something said in another conversation.
pub fn refers_to_past(text: &str) -> bool {
    let lower = text.to_lowercase().replace('’', "'");
    [
        "i mentioned", "i told you", "told you about", "we discussed", "we talked", "we spoke", "did i ask", "did i say", "did i tell", "did i mention", "you said",
        "last time", "previous conversation", "earlier conversation", "other conversation", "remember what", "remember when", "remember the", "what was that",
        "did i ", "have i ", "i asked you", "i said", "remind you", "do you remember", "yesterday", "last week", "earlier today",
    ]
    .iter()
    .any(|p| lower.contains(p))
}

/// Whether the message is about the chosen folder by its words alone.
pub fn refers_to_folder(text: &str) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).collect();
    ["folder", "folders", "file", "files", "directory", "project", "repo", "codebase", "readme"].iter().any(|w| words.contains(w))
        || ["in this", "in here", "this code", "these files"].iter().any(|p| lower.contains(p))
}

/// What to search the web for, when a message asks for something only the web has: news,
/// anything current, or a search outright.
pub fn web_query(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| lower.split(|c: char| !c.is_alphanumeric()).any(|x| x == *w));
    let phrase = |phrases: &[&str]| phrases.iter().any(|p| lower.contains(p));
    let wants = has(&["news", "headlines", "latest", "weather", "forecast", "google", "internet", "online"])
        || phrase(&["on the web", "search the web", "web search", "look up", "who won", "score of", "price of", "right now", "this week", "today's", "todays", "yesterday's", "currently"]);
    wants.then(|| search_words(text))
}

/// The question's own words, less the asking: what goes to the search. Bing's keyless
/// results match words loosely, and "how", "latest" or "news" alone pull in news portals'
/// front pages, so those go too, unless nothing else is left.
pub fn search_words(text: &str) -> String {
    const ASKING: &[&str] = &[
        "can", "could", "would", "you", "please", "check", "search", "look", "up", "for", "find", "on", "the", "web", "internet", "online", "google", "me", "tell",
        "give", "show", "just", "a", "an", "of", "about", "what", "whats", "what's", "is", "are", "some", "read", "get", "how", "hows", "how's", "i", "do", "does",
        "want", "know", "any", "there", "and", "right",
        // News results are today's already; the word only finds pages about dates.
        "today", "today's", "todays", "now", "currently",
    ];
    let word = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase().replace('’', "'");
    let mut words: Vec<&str> = text.split_whitespace().filter(|w| !ASKING.contains(&word(w).as_str())).collect();
    let topic: Vec<&str> = words.iter().copied().filter(|w| !RECENT.contains(&word(w).as_str())).collect();
    if !topic.is_empty() {
        words = topic;
    }
    let query = words.join(" ").trim_matches(|c: char| !c.is_alphanumeric()).to_string();
    if query.is_empty() {
        text.trim().to_string()
    } else {
        query
    }
}

/// Whether a message is about the date or time, so the model needs the clock.
pub fn asks_time(text: &str) -> bool {
    const TIME: &[&str] = &[
        "time", "date", "day", "today", "tonight", "tomorrow", "yesterday", "now", "week", "weekend", "month", "year", "clock", "when", "morning", "evening",
        "afternoon", "hour", "hours", "minutes", "schedule", "remind", "deadline", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    ];
    text.to_lowercase().split(|c: char| !c.is_alphanumeric()).any(|w| TIME.contains(&w))
}

/// Words that ask for something recent rather than name it.
pub const RECENT: &[&str] = &["latest", "recent", "news", "headlines", "update", "updates", "new"];

/// A reminder the message asks for, when it says the time: (the message, what to remind).
pub fn reminder_request(text: &str) -> Option<(String, String)> {
    let lower = text.to_lowercase();
    let i = lower.find("remind me")?;
    super::remind::parse_when(text, chrono::Local::now())?;
    let what = text[i + "remind me".len()..].trim();
    let what = ["to ", "that ", "about ", "of "].iter().find_map(|p| what.strip_prefix(p)).unwrap_or(what).trim_end_matches(['.', '!']);
    Some((text.to_string(), what.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reminder_with_a_time_is_read_from_the_message() {
        assert_eq!(reminder_request("remind me to call ayush at 9pm"), Some(("remind me to call ayush at 9pm".into(), "call ayush at 9pm".into())));
        assert_eq!(reminder_request("Can you remind me in 20 minutes to stretch."), Some(("Can you remind me in 20 minutes to stretch.".into(), "in 20 minutes to stretch".into())));
        assert_eq!(reminder_request("remind me to call ayush"), None);
        assert_eq!(reminder_request("call ayush at 9pm"), None);
    }

    #[test]
    fn some_messages_route_by_their_words_alone() {
        assert!(refers_to_past("what was that restaurant I mentioned before?"));
        assert!(refers_to_past("remember what we discussed about the deployment last week?"));
        assert!(!refers_to_past("what is a restaurant?"));
        assert!(refers_to_past("what did i remind you of?"));
        assert!(!refers_to_past("Hola!") && !refers_to_past("how are you?"));
        assert!(refers_to_folder("what do you see in this folder?"));
        assert!(refers_to_folder("summarize the README"));
        assert!(!refers_to_folder("write a haiku about rain"));
    }

    #[test]
    fn asking_for_news_or_the_web_gives_a_search_of_the_question() {
        assert_eq!(web_query("can you check for today’s cybersecurity news on the web").as_deref(), Some("cybersecurity"));
        assert_eq!(web_query("just give me the latest F1 grand prix results").as_deref(), Some("F1 grand prix results"));
        assert_eq!(web_query("how about the latest colima pipeline").as_deref(), Some("colima pipeline"));
        assert_eq!(web_query("latest news?").as_deref(), Some("latest news"));
        assert_eq!(web_query("what's the weather in Pune?").as_deref(), Some("weather in Pune"));
        assert_eq!(web_query("read the README and fix the build"), None);
        assert!(asks_time("what day is it?") && asks_time("remind me at 9") && !asks_time("hmmm") && !asks_time("can you code?"));
        assert_eq!(web_query("search the project for TODO"), None);
    }
}
