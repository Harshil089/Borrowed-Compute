//! Shared types for Borrowed Compute.
//!
//! - [`spec`]: the `bc.toml` job spec that users write.
//! - [`wire`]: the messages the client and agent exchange over the network.
//! - [`cloud`]: JSON bodies for the cloud API (sign-in, devices, pairing).
//! - [`guest`]: invites and the messages that share a Mac with another account.

pub mod cloud;
pub mod guest;
pub mod spec;
pub mod wire;

pub use spec::{JobSpec, SpecError};
pub use wire::{AgentMsg, ClientMsg, JobRequest, WireError, PROTOCOL_VERSION};
