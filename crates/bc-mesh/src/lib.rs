//! Several devices working as one.
//!
//! A coordinator (any device) measures its links to the devices it is paired with
//! ([`probe`]), decides which device runs which part of a model ([`plan`]), and streams
//! work through them as a pipeline ([`pipeline`] on the coordinator, [`stage`] on each
//! device). Activations move device to device on their own connections, authorized by
//! tickets the coordinator signs ([`ticket`]), as exact values ([`codec`]). See `docs/mesh.md` for the design and the reasoning behind it.

pub mod codec;
pub mod engine;
pub mod llm;
pub mod msg;
pub mod pipeline;
pub mod plan;
pub mod probe;
pub mod stage;
pub mod ticket;
pub mod tree;

pub use bc_net::MESH_ALPN;
