//! Answers code gives exactly, so a small model only has to say them: the time anywhere
//! (from the timezone database, not a web page) and arithmetic. The harness calls these
//! itself when a message asks (`route::clock_request`, `route::sum_request`), the way
//! deterministic steps take control flow away from the model in harness designs such as
//! ReWOO's planner/worker split; the model can call them too.

use anyhow::{bail, Result};
use chrono::{DateTime, Offset, Utc};
use chrono_tz::Tz;

/// Names people use for a place that the timezone database spells differently, or for a
/// whole country. Several zones for a country that spans them.
const PLACES: &[(&str, &[&str])] = &[
    ("england", &["Europe/London"]),
    ("uk", &["Europe/London"]),
    ("united kingdom", &["Europe/London"]),
    ("britain", &["Europe/London"]),
    ("great britain", &["Europe/London"]),
    ("scotland", &["Europe/London"]),
    ("wales", &["Europe/London"]),
    ("ireland", &["Europe/Dublin"]),
    ("india", &["Asia/Kolkata"]),
    ("mumbai", &["Asia/Kolkata"]),
    ("delhi", &["Asia/Kolkata"]),
    ("new delhi", &["Asia/Kolkata"]),
    ("bangalore", &["Asia/Kolkata"]),
    ("bengaluru", &["Asia/Kolkata"]),
    ("pune", &["Asia/Kolkata"]),
    ("chennai", &["Asia/Kolkata"]),
    ("hyderabad", &["Asia/Kolkata"]),
    ("calcutta", &["Asia/Kolkata"]),
    ("ahmedabad", &["Asia/Kolkata"]),
    ("usa", &["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]),
    ("us", &["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]),
    ("america", &["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]),
    ("united states", &["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]),
    ("california", &["America/Los_Angeles"]),
    ("san francisco", &["America/Los_Angeles"]),
    ("seattle", &["America/Los_Angeles"]),
    ("texas", &["America/Chicago"]),
    ("dallas", &["America/Chicago"]),
    ("houston", &["America/Chicago"]),
    ("boston", &["America/New_York"]),
    ("washington", &["America/New_York"]),
    ("miami", &["America/New_York"]),
    ("canada", &["America/Toronto", "America/Vancouver"]),
    ("australia", &["Australia/Sydney", "Australia/Perth"]),
    ("new zealand", &["Pacific/Auckland"]),
    ("japan", &["Asia/Tokyo"]),
    ("china", &["Asia/Shanghai"]),
    ("beijing", &["Asia/Shanghai"]),
    ("korea", &["Asia/Seoul"]),
    ("south korea", &["Asia/Seoul"]),
    ("germany", &["Europe/Berlin"]),
    ("france", &["Europe/Paris"]),
    ("spain", &["Europe/Madrid"]),
    ("italy", &["Europe/Rome"]),
    ("netherlands", &["Europe/Amsterdam"]),
    ("switzerland", &["Europe/Zurich"]),
    ("russia", &["Europe/Moscow"]),
    ("brazil", &["America/Sao_Paulo"]),
    ("mexico", &["America/Mexico_City"]),
    ("uae", &["Asia/Dubai"]),
    ("emirates", &["Asia/Dubai"]),
    ("abu dhabi", &["Asia/Dubai"]),
    ("saudi arabia", &["Asia/Riyadh"]),
    ("pakistan", &["Asia/Karachi"]),
    ("bangladesh", &["Asia/Dhaka"]),
    ("nepal", &["Asia/Kathmandu"]),
    ("sri lanka", &["Asia/Colombo"]),
    ("indonesia", &["Asia/Jakarta"]),
    ("philippines", &["Asia/Manila"]),
    ("vietnam", &["Asia/Ho_Chi_Minh"]),
    ("thailand", &["Asia/Bangkok"]),
    ("malaysia", &["Asia/Kuala_Lumpur"]),
    ("egypt", &["Africa/Cairo"]),
    ("south africa", &["Africa/Johannesburg"]),
    ("nigeria", &["Africa/Lagos"]),
    ("kenya", &["Africa/Nairobi"]),
    ("turkey", &["Europe/Istanbul"]),
    ("israel", &["Asia/Jerusalem"]),
    ("argentina", &["America/Argentina/Buenos_Aires"]),
    ("utc", &["UTC"]),
    ("gmt", &["Etc/GMT"]),
];

/// Lowercase words joined by single spaces, padded, so a place matches whole words only.
fn padded(text: &str) -> String {
    let words: Vec<String> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
    format!(" {} ", words.join(" "))
}

/// The place a message names, and its zones: the longest name that matches, from the
/// table above or any city in the timezone database ("tokyo", "new york", "sao paulo").
pub fn place_in(text: &str) -> Option<(String, Vec<Tz>)> {
    let text = padded(text);
    let mut best: Option<(String, Vec<Tz>)> = None;
    let mut consider = |name: String, zones: Vec<Tz>| {
        if text.contains(&format!(" {name} ")) && best.as_ref().is_none_or(|(b, _)| name.len() > b.len()) {
            best = Some((name, zones));
        }
    };
    for (name, zones) in PLACES {
        consider(name.to_string(), zones.iter().filter_map(|z| z.parse().ok()).collect());
    }
    for tz in chrono_tz::TZ_VARIANTS {
        let city = tz.name().rsplit('/').next().unwrap_or("").replace('_', " ").to_lowercase();
        // Three letters and fewer are abbreviations ("EST", "MST") that read as words.
        if city.len() > 3 && !tz.name().starts_with("Etc/") {
            consider(city, vec![tz]);
        }
    }
    best
}

/// The time now in `place` (a city or country), or here when it is empty.
pub fn clock(place: &str, now: DateTime<Utc>) -> Result<String> {
    let local = |t: DateTime<chrono::FixedOffset>| t.format("%H:%M (%-I:%M %p) on %A, %-d %B %Y").to_string();
    if place.trim().is_empty() || padded(place).contains(" here ") {
        return Ok(format!("It is {} here (this computer's time).", local(now.with_timezone(&chrono::Local).fixed_offset())));
    }
    let Some((name, zones)) = place_in(place) else {
        bail!("no timezone known for \"{place}\"; give a city or a country");
    };
    let lines: Vec<String> = zones
        .iter()
        .map(|tz| {
            let t = now.with_timezone(tz);
            let offset = t.offset().fix().local_minus_utc();
            let utc = if offset == 0 { "UTC".to_string() } else { format!("UTC{}{}:{:02}", if offset < 0 { '-' } else { '+' }, offset.abs() / 3600, offset.abs() % 3600 / 60) };
            format!("{}: {} ({}, {utc})", tz.name(), local(t.fixed_offset()), t.format("%Z"))
        })
        .collect();
    Ok(format!("The time in {name} now (from the timezone database):\n{}", lines.join("\n")))
}

/// Arithmetic in words or symbols: "17 times 23", "15% of 240", "(3 + 4) * 2", "2^10".
pub fn calculate(text: &str) -> Result<String> {
    let expr = expression(text).ok_or_else(|| anyhow::anyhow!("no arithmetic in \"{text}\""))?;
    let mut p = Parser { s: expr.as_bytes(), i: 0 };
    let v = p.sum()?;
    p.space();
    if p.i != p.s.len() {
        bail!("cannot read \"{expr}\" as arithmetic");
    }
    if !v.is_finite() {
        bail!("{expr} has no finite value");
    }
    let shown = if v.fract() == 0.0 && v.abs() < 1e15 { format!("{}", v as i64) } else { format!("{v:.6}").trim_end_matches('0').trim_end_matches('.').to_string() };
    Ok(format!("{expr} = {shown}"))
}

/// The arithmetic part of a message, in symbols, when there is one with an operator.
pub fn expression(text: &str) -> Option<String> {
    let mut t = format!(" {} ", text.to_lowercase().replace(',', ""));
    for (word, sym) in [
        (" multiplied by ", " * "),
        (" divided by ", " / "),
        (" times ", " * "),
        (" x ", " * "),
        (" plus ", " + "),
        (" minus ", " - "),
        (" over ", " / "),
        ("% of ", " / 100 * "),
        (" percent of ", " / 100 * "),
        (" to the power of ", " ^ "),
        (" squared", " ^ 2"),
        (" cubed", " ^ 3"),
        ("×", "*"),
        ("÷", "/"),
    ] {
        t = t.replace(word, sym);
    }
    // The longest run of arithmetic characters that has a number, an operator, a number.
    let runs = t.split(|c: char| !(c.is_ascii_digit() || " .+-*/^()".contains(c)));
    let best = runs.map(str::trim).filter(|r| has_operation(r)).max_by_key(|r| r.len())?;
    Some(best.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn has_operation(r: &str) -> bool {
    let b = r.as_bytes();
    (1..b.len()).any(|i| b"+-*/^".contains(&b[i]) && b[..i].iter().any(u8::is_ascii_digit) && b[i + 1..].iter().any(u8::is_ascii_digit))
}

/// Recursive descent: sum of products of powers of signed atoms.
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self.s.get(self.i) == Some(&b' ') {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.space();
        self.s.get(self.i).copied()
    }
    fn sum(&mut self) -> Result<f64> {
        let mut v = self.product()?;
        while let Some(op @ (b'+' | b'-')) = self.peek() {
            self.i += 1;
            let r = self.product()?;
            v = if op == b'+' { v + r } else { v - r };
        }
        Ok(v)
    }
    fn product(&mut self) -> Result<f64> {
        let mut v = self.power()?;
        while let Some(op @ (b'*' | b'/')) = self.peek() {
            self.i += 1;
            let r = self.power()?;
            if op == b'/' && r == 0.0 {
                bail!("division by zero");
            }
            v = if op == b'*' { v * r } else { v / r };
        }
        Ok(v)
    }
    fn power(&mut self) -> Result<f64> {
        let base = self.atom()?;
        if self.peek() == Some(b'^') {
            self.i += 1;
            return Ok(base.powf(self.power()?));
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<f64> {
        match self.peek() {
            Some(b'-') => {
                self.i += 1;
                Ok(-self.atom()?)
            }
            Some(b'(') => {
                self.i += 1;
                let v = self.sum()?;
                if self.peek() != Some(b')') {
                    bail!("a bracket is not closed");
                }
                self.i += 1;
                Ok(v)
            }
            _ => {
                let start = self.i;
                while self.s.get(self.i).is_some_and(|c| c.is_ascii_digit() || *c == b'.') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i]).ok().and_then(|n| n.parse().ok()).ok_or_else(|| anyhow::anyhow!("expected a number"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn arithmetic_in_words_and_symbols() {
        let v = |t: &str| calculate(t).unwrap().rsplit(" = ").next().unwrap().to_string();
        assert_eq!(v("what is 17 times 23?"), "391");
        assert_eq!(v("15% of 240"), "36");
        assert_eq!(v("(3 + 4) * 2 - 1"), "13");
        assert_eq!(v("2^10"), "1024");
        assert_eq!(v("10 divided by 4"), "2.5");
        assert_eq!(v("1,200 x 3"), "3600");
        assert!(calculate("1/0").is_err());
        assert_eq!(expression("what time is it in england"), None);
        assert_eq!(expression("the 2026 world cup"), None);
        assert_eq!(expression("call me at 9-10"), Some("9-10".into()));
    }

    #[test]
    fn places_find_their_zones() {
        let names = |t: &str| place_in(t).map(|(_, z)| z.iter().map(|z| z.name().to_string()).collect::<Vec<_>>());
        assert_eq!(names("whats the time in england").unwrap(), ["Europe/London"]);
        assert_eq!(names("time in New York right now?").unwrap(), ["America/New_York"]);
        assert_eq!(names("what time is it in tokyo").unwrap(), ["Asia/Tokyo"]);
        assert_eq!(names("time in sao paulo").unwrap(), ["America/Sao_Paulo"]);
        assert_eq!(names("time in the usa").unwrap().len(), 4);
        assert_eq!(names("what time is it"), None);
        // 2026-07-01 12:00 UTC is 13:00 in London (summer time).
        let noon = Utc.with_ymd_and_hms(2026, 7, 1, 12, 0, 0).unwrap();
        let out = clock("england", noon).unwrap();
        assert!(out.contains("Europe/London: 13:00") && out.contains("UTC+1:00"), "{out}");
        assert!(clock("atlantis", noon).is_err());
    }
}
