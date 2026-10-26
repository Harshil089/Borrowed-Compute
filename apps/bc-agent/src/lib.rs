//! Mac agent library: the job runner, the per-connection protocol, and the running
//! service shared by `bc-agent serve` and the menu-bar app.

pub mod cloud_sync;
pub mod events;
pub mod paired;
pub mod runner;
pub mod sandbox;
pub mod server;
pub mod service;
