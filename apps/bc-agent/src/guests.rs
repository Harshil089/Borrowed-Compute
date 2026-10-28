//! Devices on other accounts that the owner let use this Mac.
//!
//! The owner turns sharing on and creates an invite for one person. That person's device
//! presents the invite over [`bc_net::GUEST_ALPN`] and gets a code, which the owner types
//! here to approve, the same way as pairing through the cloud. Approved devices are kept in
//! `guests.json` next to the paired list and are checked on every connection, alongside it.
//! Turning sharing off refuses every guest without forgetting them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bc_net::EndpointId;
use bc_proto::guest::{Invite, TOKEN_ALPHABET, TOKEN_LEN};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

/// An invite works once, within this time.
pub const INVITE_LIFETIME: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// A request waits this long for the owner before the guest is told to try again.
pub const REQUEST_LIFETIME: Duration = Duration::from_secs(10 * 60);
/// Requests waiting at once. Each needs a valid invite, so this only bounds a burst.
pub const MAX_REQUESTS: usize = 8;
/// Wrong codes before a request is dropped, as with cloud pairing.
const MAX_CODE_TRIES: u32 = 3;
/// Characters kept of a name typed by the owner or sent by a guest's device.
const MAX_NAME: usize = 64;
const CODE_ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";

#[derive(Debug, Default, Serialize, Deserialize)]
struct Saved {
    sharing: bool,
    invites: Vec<SavedInvite>,
    guests: Vec<Guest>,
}

/// Only the token's hash is kept, so reading the file does not give out invites.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedInvite {
    id: String,
    name: String,
    token_sha256: String,
    /// Unix seconds.
    expires: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guest {
    pub device: String,
    /// The name the owner gave the invite, such as the person's.
    pub name: String,
    /// The name the guest's device sent.
    pub device_name: String,
    /// Unix seconds.
    pub added: u64,
}

struct Request {
    invite_name: String,
    device: EndpointId,
    device_name: String,
    code: String,
    attempts: u32,
    created: Instant,
    answer: oneshot::Sender<Result<(), String>>,
}

#[derive(Default)]
struct Inner {
    saved: Saved,
    requests: HashMap<u64, Request>,
    next_request: u64,
}

/// What the Mac app shows.
#[derive(Debug, Clone, Serialize)]
pub struct GuestsView {
    pub sharing: bool,
    pub invites: Vec<InviteView>,
    pub guests: Vec<Guest>,
    pub requests: Vec<RequestView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InviteView {
    pub id: String,
    pub name: String,
    pub expires: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestView {
    pub id: u64,
    pub name: String,
    pub device: String,
    pub device_name: String,
}

/// A request that is waiting for the owner: show `code` to the guest and wait on `answer`.
pub struct Pending {
    pub id: u64,
    pub code: String,
    pub answer: oneshot::Receiver<Result<(), String>>,
}

#[derive(Clone, Default)]
pub struct Guests {
    /// `None` keeps everything in memory, for tests.
    path: Option<PathBuf>,
    inner: Arc<Mutex<Inner>>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn sha256_hex(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

fn random_text(alphabet: &[u8], len: usize) -> String {
    (0..len).map(|_| alphabet[rand::random::<usize>() % alphabet.len()] as char).collect()
}

/// Compares codes the way people type them: any case, with or without the dash.
fn normalize(code: &str) -> String {
    code.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).collect()
}

fn clean_name(name: &str) -> String {
    name.chars().filter(|c| !c.is_control()).take(MAX_NAME).collect::<String>().trim().to_string()
}

impl std::fmt::Debug for Guests {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Guests").field("path", &self.path).finish_non_exhaustive()
    }
}

impl Guests {
    /// Reads `path`; a missing or unreadable file starts with sharing off and nobody in it.
    pub fn load(path: PathBuf) -> Self {
        let saved = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                eprintln!("ignoring {}: {e}", path.display());
                Saved::default()
            }),
            Err(_) => Saved::default(),
        };
        Guests { path: Some(path), inner: Arc::new(Mutex::new(Inner { saved, ..Inner::default() })) }
    }

    fn save(&self, saved: &Saved) {
        let Some(path) = &self.path else { return };
        let result = (|| {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_vec_pretty(saved)?)?;
            std::fs::rename(&tmp, path)
        })();
        if let Err(e) = result {
            eprintln!("could not save {}: {e}", path.display());
        }
    }

    /// True if `device` may run jobs as a guest right now.
    pub fn allows(&self, device: &EndpointId) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.saved.sharing && self.is_guest_in(&inner, device)
    }

    pub fn is_guest(&self, device: &EndpointId) -> bool {
        self.is_guest_in(&self.inner.lock().unwrap(), device)
    }

    fn is_guest_in(&self, inner: &Inner, device: &EndpointId) -> bool {
        let device = device.to_string();
        inner.saved.guests.iter().any(|g| g.device == device)
    }

    pub fn sharing(&self) -> bool {
        self.inner.lock().unwrap().saved.sharing
    }

    /// Turning sharing off also turns away every waiting request.
    pub fn set_sharing(&self, on: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.saved.sharing = on;
        if !on {
            for (_, r) in inner.requests.drain() {
                let _ = r.answer.send(Err("the Mac's owner stopped sharing it".into()));
            }
        }
        self.save(&inner.saved);
    }

    /// A new invite for `name`, reaching the Mac whose code is `mac`. The text is shown to
    /// the owner once and never kept.
    pub fn create_invite(&self, name: &str, mac: &str) -> Result<String, String> {
        let name = clean_name(name);
        if name.is_empty() {
            return Err("give the invite a name, such as the person's".into());
        }
        let token = random_text(TOKEN_ALPHABET, TOKEN_LEN);
        let mut inner = self.inner.lock().unwrap();
        let now = now();
        inner.saved.invites.retain(|i| i.expires > now);
        inner.saved.invites.push(SavedInvite {
            id: random_text(CODE_ALPHABET, 10),
            name,
            token_sha256: sha256_hex(&token),
            expires: now + INVITE_LIFETIME.as_secs(),
        });
        self.save(&inner.saved);
        Ok(Invite { mac: mac.to_string(), token }.to_text())
    }

    pub fn delete_invite(&self, id: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.saved.invites.retain(|i| i.id != id);
        self.save(&inner.saved);
    }

    /// Forgets a guest. Returns false if `device` was not one.
    pub fn remove_guest(&self, device: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let before = inner.saved.guests.len();
        inner.saved.guests.retain(|g| g.device != device);
        let removed = inner.saved.guests.len() != before;
        self.save(&inner.saved);
        removed
    }

    /// Uses up the invite whose secret is `token` and starts a request for the owner to
    /// approve. Errors are shown to the guest as they are.
    pub fn request(&self, token: &str, device: EndpointId, device_name: &str) -> Result<Pending, String> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.saved.sharing {
            return Err("this Mac is not shared right now".into());
        }
        inner.requests.retain(|_, r| r.created.elapsed() < REQUEST_LIFETIME);
        if inner.requests.len() >= MAX_REQUESTS {
            return Err("this Mac has too many requests waiting; try again later".into());
        }
        let now = now();
        let hash = sha256_hex(token);
        let Some(at) = inner.saved.invites.iter().position(|i| i.token_sha256 == hash && i.expires > now) else {
            return Err("this invite is not valid: it was used, expired or deleted. Ask for a new one".into());
        };
        let invite = inner.saved.invites.remove(at);
        self.save(&inner.saved);
        let code = random_text(CODE_ALPHABET, 8);
        let code = format!("{}-{}", &code[..4], &code[4..]);
        let (answer, waiting) = oneshot::channel();
        inner.next_request += 1;
        let id = inner.next_request;
        let device_name = match clean_name(device_name) {
            n if n.is_empty() => "Unnamed device".to_string(),
            n => n,
        };
        inner.requests.insert(
            id,
            Request { invite_name: invite.name, device, device_name, code: code.clone(), attempts: 0, created: Instant::now(), answer },
        );
        Ok(Pending { id, code, answer: waiting })
    }

    /// Approves a request if `code` matches the one its device shows.
    pub fn approve(&self, id: u64, code: &str) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();
        let Some(request) = inner.requests.get_mut(&id) else {
            return Err("this request was cancelled".into());
        };
        if normalize(code) != normalize(&request.code) {
            request.attempts += 1;
            if request.attempts >= MAX_CODE_TRIES {
                let request = inner.requests.remove(&id).unwrap();
                let _ = request.answer.send(Err("the code was typed wrong three times".into()));
                return Err("wrong code three times; the request was cancelled".into());
            }
            return Err("that code does not match the one on the other device".into());
        }
        let request = inner.requests.remove(&id).unwrap();
        let device = request.device.to_string();
        inner.saved.guests.retain(|g| g.device != device);
        inner.saved.guests.push(Guest { device, name: request.invite_name, device_name: request.device_name, added: now() });
        self.save(&inner.saved);
        let _ = request.answer.send(Ok(()));
        Ok(())
    }

    pub fn deny(&self, id: u64) {
        if let Some(r) = self.inner.lock().unwrap().requests.remove(&id) {
            let _ = r.answer.send(Err("the Mac's owner declined".into()));
        }
    }

    /// Drops a request whose device went away or waited too long.
    pub fn cancel(&self, id: u64) {
        self.inner.lock().unwrap().requests.remove(&id);
    }

    pub fn view(&self) -> GuestsView {
        let mut inner = self.inner.lock().unwrap();
        let now = now();
        inner.saved.invites.retain(|i| i.expires > now);
        inner.requests.retain(|_, r| r.created.elapsed() < REQUEST_LIFETIME);
        let mut requests: Vec<RequestView> = inner
            .requests
            .iter()
            .map(|(id, r)| RequestView {
                id: *id,
                name: r.invite_name.clone(),
                device: r.device.to_string(),
                device_name: r.device_name.clone(),
            })
            .collect();
        requests.sort_by_key(|r| r.id);
        GuestsView {
            sharing: inner.saved.sharing,
            invites: inner
                .saved
                .invites
                .iter()
                .map(|i| InviteView { id: i.id.clone(), name: i.name.clone(), expires: i.expires })
                .collect(),
            guests: inner.saved.guests.clone(),
            requests,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(n: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[n; 32]).public()
    }

    fn token(text: &str) -> String {
        Invite::parse(text).unwrap().token
    }

    #[test]
    fn an_invite_works_once_and_needs_the_right_code() {
        let guests = Guests::default();
        let mac = "a".repeat(64);
        assert!(guests.create_invite("Alex", &mac).is_ok());
        let invite = guests.create_invite("Sam", &mac).unwrap();
        assert!(guests.request(&token(&invite), device(1), "pc").is_err(), "sharing is off");

        guests.set_sharing(true);
        let mut pending = guests.request(&token(&invite), device(1), "Sam's PC\n").unwrap();
        assert!(guests.request(&token(&invite), device(2), "pc").is_err(), "the invite is used up");
        assert_eq!(guests.view().invites.len(), 1, "only Alex's invite is left");
        assert_eq!(guests.view().requests[0].device_name, "Sam's PC");

        assert!(guests.approve(pending.id, "XXXX-XXXX").is_err());
        assert!(!guests.allows(&device(1)));
        guests.approve(pending.id, &pending.code.to_lowercase().replace('-', "")).unwrap();
        assert_eq!(pending.answer.try_recv().unwrap(), Ok(()));
        assert!(guests.allows(&device(1)));
        assert_eq!(guests.view().guests[0].name, "Sam");

        guests.set_sharing(false);
        assert!(!guests.allows(&device(1)), "sharing off refuses guests");
        assert!(guests.is_guest(&device(1)), "but remembers them");
        guests.set_sharing(true);
        assert!(guests.remove_guest(&device(1).to_string()));
        assert!(!guests.allows(&device(1)));
    }

    #[test]
    fn three_wrong_codes_cancel_a_request() {
        let guests = Guests::default();
        guests.set_sharing(true);
        let invite = guests.create_invite("Alex", &"a".repeat(64)).unwrap();
        let mut pending = guests.request(&token(&invite), device(1), "pc").unwrap();
        for _ in 0..3 {
            assert!(guests.approve(pending.id, "WRONG").is_err());
        }
        assert!(pending.answer.try_recv().unwrap().is_err());
        assert!(guests.approve(pending.id, &pending.code).is_err(), "the request is gone");
        assert!(!guests.allows(&device(1)));
    }

    #[test]
    fn keeps_guests_and_only_invite_hashes_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("guests.json");
        let guests = Guests::load(path.clone());
        guests.set_sharing(true);
        let invite = guests.create_invite("Alex", &"a".repeat(64)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains(&token(&invite)), "the secret itself is never saved");
        let pending = guests.request(&token(&invite), device(1), "pc").unwrap();
        guests.approve(pending.id, &pending.code).unwrap();

        let again = Guests::load(path);
        assert!(again.sharing());
        assert!(again.allows(&device(1)));
    }
}
