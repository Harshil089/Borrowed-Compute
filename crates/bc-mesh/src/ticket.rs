//! Stage tickets: how two lenders that never paired with each other agree to talk.
//!
//! Every device in a pipeline is paired with the coordinator, the device that started
//! it, but not necessarily with its neighbours. Activations must still flow device to
//! device, never back through the coordinator, or every token would cross the slowest
//! link twice. So the coordinator signs a ticket per hop: "in session S, `from` may feed
//! `to`". The receiving stage accepts a feed only with a ticket signed by the coordinator
//! that set its session up, naming the connecting device and itself, unexpired.

use anyhow::{bail, Result};
use iroh::{EndpointId, SecretKey};
use serde::{Deserialize, Serialize};

/// Long enough to set a pipeline up; a running pipeline never needs a new ticket.
pub const TICKET_SECS: u64 = 300;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ticket {
    pub session: u64,
    pub from: EndpointId,
    pub to: EndpointId,
    /// Unix seconds.
    pub expires: u64,
    pub signature: iroh::Signature,
}

fn message(session: u64, from: &EndpointId, to: &EndpointId, expires: u64) -> Vec<u8> {
    let mut m = b"borrowed-compute/mesh-ticket/1".to_vec();
    m.extend_from_slice(&session.to_be_bytes());
    m.extend_from_slice(from.as_bytes());
    m.extend_from_slice(to.as_bytes());
    m.extend_from_slice(&expires.to_be_bytes());
    m
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

impl Ticket {
    pub fn sign(coordinator: &SecretKey, session: u64, from: EndpointId, to: EndpointId) -> Self {
        let expires = now() + TICKET_SECS;
        let signature = coordinator.sign(&message(session, &from, &to, expires));
        Ticket { session, from, to, expires, signature }
    }

    /// Accepts the ticket only for this session, coordinator, sender and receiver.
    pub fn check(&self, session: u64, coordinator: &EndpointId, from: &EndpointId, me: &EndpointId) -> Result<()> {
        if self.session != session || &self.from != from || &self.to != me {
            bail!("the ticket is for another hop");
        }
        if self.expires < now() {
            bail!("the ticket has expired");
        }
        coordinator
            .verify(&message(self.session, &self.from, &self.to, self.expires), &self.signature)
            .map_err(|_| anyhow::anyhow!("the ticket is not signed by the pipeline's coordinator"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_signer_hop_and_session() {
        let coord = SecretKey::generate();
        let (a, b, c) = (SecretKey::generate().public(), SecretKey::generate().public(), SecretKey::generate().public());
        let t = Ticket::sign(&coord, 7, a, b);
        assert!(t.check(7, &coord.public(), &a, &b).is_ok());
        assert!(t.check(8, &coord.public(), &a, &b).is_err(), "other session");
        assert!(t.check(7, &coord.public(), &c, &b).is_err(), "another device presents it");
        assert!(t.check(7, &coord.public(), &a, &c).is_err(), "sent to another stage");
        assert!(t.check(7, &SecretKey::generate().public(), &a, &b).is_err(), "other coordinator");
        let mut forged = t.clone();
        forged.expires += 1;
        assert!(forged.check(7, &coord.public(), &a, &b).is_err(), "edited after signing");
        let mut old = Ticket::sign(&coord, 7, a, b);
        old.expires = 1;
        old.signature = coord.sign(&message(7, &a, &b, 1));
        assert!(old.check(7, &coord.public(), &a, &b).is_err(), "expired");
    }
}
