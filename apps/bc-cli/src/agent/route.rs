//! Shortcuts: what the harness does before the model answers when the message's own words
//! say it (a reminder with a time, the time somewhere, arithmetic, the web or a named
//! site, "yes, search it", past conversations, the folder), so a model too small to
//! choose a tool still answers from real results. Messages no shortcut matches go to
//! `pick`, which lets the model choose among the tools that fit.

/// What to do first for a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Web,
    Files,
    Past,
    /// The time somewhere or arithmetic, worked out by code (`facts`).
    Fact,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Route::Web => "web",
            Route::Files => "files",
            Route::Past => "past conversations",
            Route::Fact => "fact",
        }
    }
}

/// The place whose time the message asks for ("" for here), when it asks for one.
pub fn clock_request(text: &str) -> Option<String> {
    let lower = text.to_lowercase().replace('’', "'");
    let asks = ["time in", "time is it", "what time", "what's the time", "whats the time", "the time there", "time now", "current time", "local time", "time at", "time right now"]
        .iter()
        .any(|p| lower.contains(p));
    if !asks {
        return None;
    }
    Some(super::facts::place_in(text).map(|(name, _)| name).unwrap_or_default())
}

/// The arithmetic a message asks to work out, when it does.
pub fn sum_request(text: &str) -> Option<String> {
    let expr = super::facts::expression(text)?;
    let lower = text.to_lowercase();
    // A bare "9-10" is a range or a date as often as a sum.
    let asked = expr.contains(['*', '/', '^', '+']) || ["calculate", "what is", "what's", "whats", "how much is", "="].iter().any(|p| lower.contains(p));
    (asked && super::facts::calculate(&expr).is_ok()).then_some(expr)
}

/// A web address named in the message ("tryzenith.cloud"), opened directly: a search
/// engine knows little about a new site, but the site knows itself.
pub fn domain_in(text: &str) -> Option<String> {
    const ENDINGS: &[&str] = &[
        "com", "org", "net", "io", "ai", "dev", "app", "cloud", "co", "in", "uk", "xyz", "me", "so", "sh", "tech", "site", "info", "gg", "tv", "us", "de", "fr", "jp",
        "edu", "gov", "eu", "ca", "au", "page", "run", "tools", "studio", "online", "store", "blog",
    ];
    text.split_whitespace().find_map(|w| {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        let w = w.strip_prefix("https://").or(w.strip_prefix("http://")).unwrap_or(&w).trim_start_matches("www.").split('/').next()?.to_string();
        let (name, end) = w.rsplit_once('.')?;
        let ok = ENDINGS.contains(&end) && !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.') && !name.starts_with(['-', '.']);
        ok.then_some(w)
    })
}

/// Whether a short reply says yes to what the model offered last, and the offer was to
/// look something up: "yea so do it" after "I would need to search the web".
pub fn accepts_search(text: &str, last_answer: &str) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let yes = words.len() <= 6
        && words.iter().any(|w| ["yes", "yea", "yeah", "yep", "sure", "ok", "okay", "go", "do", "please", "search", "proceed"].contains(w));
    let offered = ["search", "look it up", "look up", "web", "browse", "find out"].iter().any(|p| last_answer.to_lowercase().contains(p));
    yes && offered
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
        assert_eq!(clock_request("whats the time in england").as_deref(), Some("england"));
        assert_eq!(clock_request("what time is it?").as_deref(), Some(""));
        assert_eq!(clock_request("how much time does rice take in india"), None);
        assert_eq!(sum_request("what is 17 times 23?").as_deref(), Some("17 * 23"));
        assert_eq!(sum_request("call me at 9-10"), None);
        assert_eq!(sum_request("the 2026 world cup"), None);
        assert_eq!(domain_in("what do you know about tryzenith.cloud").as_deref(), Some("tryzenith.cloud"));
        assert_eq!(domain_in("open https://www.example.com/a?b"), Some("example.com".into()));
        assert_eq!(domain_in("summarize notes.txt"), None);
        assert!(accepts_search("Yea so do it", "To give you accurate information, I would need to call a web search."));
        assert!(!accepts_search("yes", "Hello! How can I help?"));
        assert!(!accepts_search("do you think search engines are good for privacy and the world", "I can search the web."));
        assert!(asks_time("what day is it?") && asks_time("remind me at 9") && !asks_time("hmmm") && !asks_time("can you code?"));
        assert_eq!(web_query("search the project for TODO"), None);
    }
}
