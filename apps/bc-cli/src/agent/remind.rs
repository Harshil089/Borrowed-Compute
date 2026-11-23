//! Reminders the agent sets: a time and a line of text, kept in `<home>/reminders.json`
//! until the app shows them. The app checks for due ones while it runs; one that came due
//! while it was closed shows at the next start.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Local, NaiveDate, NaiveTime, TimeZone};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reminder {
    /// Unix seconds.
    pub at: i64,
    pub text: String,
}

pub struct Reminders {
    file: PathBuf,
}

impl Reminders {
    pub fn at(home: &Path) -> Self {
        Reminders { file: home.join("reminders.json") }
    }

    pub fn all(&self) -> Vec<Reminder> {
        std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn save(&self, all: &[Reminder]) -> Result<()> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.file.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(all)?)?;
        std::fs::rename(&tmp, &self.file).context("saving the reminders")
    }

    /// Sets a reminder for `when` (see [`parse_when`]); says when it will show.
    pub fn add(&self, when: &str, text: &str) -> Result<String> {
        let at = parse_when(when, Local::now()).with_context(|| format!("cannot read \"{when}\" as a time; use one like 21:00, 9pm, in 20 minutes, tomorrow 9am or 2026-11-24 09:00"))?;
        let text = text.trim();
        if text.is_empty() {
            bail!("the reminder needs some text");
        }
        let mut all = self.all();
        all.push(Reminder { at: at.timestamp(), text: text.to_string() });
        self.save(&all)?;
        Ok(format!("Reminder set for {}: {text}. The app shows it then while it is open, or as soon as it opens.", at.format("%A %Y-%m-%d %H:%M")))
    }

    /// Takes the reminders due by `now` out of the file.
    pub fn take_due(&self, now: i64) -> Vec<Reminder> {
        let (due, later): (Vec<Reminder>, Vec<Reminder>) = self.all().into_iter().partition(|r| r.at <= now);
        if !due.is_empty() && self.save(&later).is_err() {
            // Shown again next time rather than lost.
            return Vec::new();
        }
        due
    }
}

/// A time as people and models write it, from `now`: "21:00", "9pm", "9:30 am", "in 20
/// minutes", "in 2 hours", "tomorrow 9am", "2026-11-24 09:00". A bare hour goes to its next
/// occurrence: "9" at 20:30 is 21:00, and a time already past today is tomorrow.
pub fn parse_when(text: &str, now: DateTime<Local>) -> Option<DateTime<Local>> {
    let lower = text.trim().to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
    // "in 20 minutes", "in 2 hours", "in 1 hr"
    if let Some(i) = words.iter().position(|w| *w == "in") {
        if let (Some(n), Some(unit)) = (words.get(i + 1).and_then(|n| n.parse::<i64>().ok()), words.get(i + 2)) {
            let d = match unit.trim_end_matches('s') {
                "minute" | "min" | "mins" => Duration::minutes(n),
                "hour" | "hr" | "hrs" => Duration::hours(n),
                "day" => Duration::days(n),
                _ => return None,
            };
            return Some(now + d);
        }
    }
    // "2026-11-24 09:00"
    let date = words.iter().find_map(|w| NaiveDate::parse_from_str(w, "%Y-%m-%d").ok());
    let tomorrow = words.contains(&"tomorrow");
    // The time: "21:00", "9pm", "9:30", "9" followed by "pm"
    let mut time = None;
    for (i, w) in words.iter().enumerate() {
        let next = words.get(i + 1).copied().unwrap_or("");
        let (body, mut half) = match w.strip_suffix("am").or(w.strip_suffix("a.m.")) {
            Some(b) => (b, Some(false)),
            None => match w.strip_suffix("pm").or(w.strip_suffix("p.m.")) {
                Some(b) => (b, Some(true)),
                None => (*w, None),
            },
        };
        if half.is_none() {
            half = match next {
                "am" | "a.m." => Some(false),
                "pm" | "p.m." => Some(true),
                _ => None,
            };
        }
        let (h, m) = match body.split_once(':') {
            Some((h, m)) => (h.parse::<u32>().ok(), m.parse::<u32>().ok()),
            None if half.is_some() || words.get(i.wrapping_sub(1)) == Some(&"at") => (body.parse::<u32>().ok(), Some(0)),
            None => (None, None),
        };
        let (Some(mut h), Some(m)) = (h, m) else { continue };
        if m > 59 || h > 23 {
            continue;
        }
        match half {
            Some(pm) if h <= 12 => h = h % 12 + if pm { 12 } else { 0 },
            Some(_) => continue,
            None => {}
        }
        // Without am or pm, a morning hour may mean the evening.
        let bare = half.is_none() && h < 12;
        time = Some((NaiveTime::from_hms_opt(h, m, 0)?, bare));
        break;
    }
    let (time, bare) = time?;
    let at = |d: NaiveDate, t: NaiveTime| Local.from_local_datetime(&d.and_time(t)).earliest();
    if let Some(d) = date {
        return at(d, time);
    }
    let today = now.date_naive();
    if tomorrow {
        return at(today.succ_opt()?, time);
    }
    // The next time it reads as, today or tomorrow; a bare hour may also be the afternoon.
    let mut options = vec![at(today, time)?, at(today.succ_opt()?, time)?];
    if bare {
        options.push(at(today, time + Duration::hours(12))?);
    }
    options.into_iter().filter(|t| *t > now).min()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 11, 23, 20, 30, 0).unwrap()
    }

    fn when(text: &str) -> Option<String> {
        parse_when(text, now()).map(|t| t.format("%Y-%m-%d %H:%M").to_string())
    }

    #[test]
    fn times_read_as_people_write_them() {
        assert_eq!(when("9pm").as_deref(), Some("2026-11-23 21:00"));
        assert_eq!(when("at 9").as_deref(), Some("2026-11-23 21:00"));
        assert_eq!(when("9:30 am").as_deref(), Some("2026-11-24 09:30"));
        assert_eq!(when("21:00").as_deref(), Some("2026-11-23 21:00"));
        assert_eq!(when("8pm").as_deref(), Some("2026-11-24 20:00"));
        assert_eq!(when("in 20 minutes").as_deref(), Some("2026-11-23 20:50"));
        assert_eq!(when("in 2 hours").as_deref(), Some("2026-11-23 22:30"));
        assert_eq!(when("tomorrow 9am").as_deref(), Some("2026-11-24 09:00"));
        assert_eq!(when("2026-12-01 07:15").as_deref(), Some("2026-12-01 07:15"));
        assert_eq!(when("call ayush at 9pm").as_deref(), Some("2026-11-23 21:00"));
        assert_eq!(when("sometime"), None);
        assert_eq!(when("13pm"), None);
    }

    #[test]
    fn reminders_are_kept_until_due() {
        let dir = tempfile::tempdir().unwrap();
        let r = Reminders::at(dir.path());
        assert!(r.add("in 5 minutes", "call Ayush").unwrap().starts_with("Reminder set for"));
        assert!(r.add("whenever", "x").is_err());
        let at = r.all()[0].at;
        assert!(r.take_due(at - 1).is_empty());
        assert_eq!(r.take_due(at).iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["call Ayush"]);
        assert!(r.all().is_empty());
    }
}
