//! Sharing a Mac with people on other accounts.
//!
//! The Mac's owner creates an [`Invite`] and sends it to one person. That person's device
//! opens a connection on the guest protocol, presents the invite with [`GuestMsg::Request`]
//! and gets a code back, which the owner types on the Mac to approve. The cloud is not
//! involved: the guest needs no account on the owner's cloud, and the owner's cloud keeps
//! its single owner.

use serde::{Deserialize, Serialize};

/// Written in front of every invite so a pasted invite is recognized as one.
pub const INVITE_PREFIX: &str = "bc-invite:";

/// What a guest pastes: which Mac to reach and the secret that proves they were invited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invite {
    /// The Mac's endpoint ID (its Mac code).
    pub mac: String,
    pub token: String,
}

impl Invite {
    /// `bc-invite:<Mac code>.<token>`.
    pub fn to_text(&self) -> String {
        format!("{INVITE_PREFIX}{}.{}", self.mac, self.token)
    }

    /// Accepts the invite as sent, with or without the prefix and surrounding spaces.
    pub fn parse(text: &str) -> Option<Invite> {
        let text = text.trim();
        let text = text.strip_prefix(INVITE_PREFIX).unwrap_or(text);
        let (mac, token) = text.split_once('.')?;
        let ok = |s: &str, len: usize| s.len() == len && s.chars().all(|c| c.is_ascii_alphanumeric());
        // An endpoint ID is 64 hex characters.
        if !ok(mac, 64) || !ok(token, TOKEN_LEN) {
            return None;
        }
        Some(Invite { mac: mac.to_ascii_lowercase(), token: token.to_string() })
    }
}

/// Invite secrets are this many characters from [`TOKEN_ALPHABET`]: about 143 bits.
pub const TOKEN_LEN: usize = 24;
pub const TOKEN_ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";

/// From the guest's device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuestMsg {
    /// Asks to use the Mac. `device_name` is shown to the owner next to the invite's name.
    Request { token: String, device_name: String },
}

/// From the Mac. A request gets `Pending` and then, on the same stream, `Approved` or
/// `Rejected`; a bad invite gets `Rejected` at once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuestReply {
    /// Show `code` to the guest, who reads it to the Mac's owner.
    Pending { code: String },
    Approved,
    Rejected { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invites_round_trip_and_reject_junk() {
        let invite = Invite { mac: "a".repeat(64), token: "B".repeat(TOKEN_LEN) };
        assert_eq!(Invite::parse(&invite.to_text()), Some(invite.clone()));
        assert_eq!(Invite::parse(&format!("  {}.{}\n", "A".repeat(64), "B".repeat(TOKEN_LEN))), Some(invite));
        assert_eq!(Invite::parse("bc-invite:abc.def"), None);
        assert_eq!(Invite::parse(&format!("{}.{}", "a".repeat(64), "B".repeat(TOKEN_LEN - 1))), None);
        assert_eq!(Invite::parse(&format!("{}.{}!", "a".repeat(64), "B".repeat(TOKEN_LEN - 1))), None);
    }
}
