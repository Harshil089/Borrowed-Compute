//! What the agent reports to a watching UI, such as the menu-bar app.

use bc_proto::wire::JobUsage;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub type EventSender = mpsc::UnboundedSender<AgentEvent>;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    JobStarted { job_id: String, client: String, argv: Vec<String> },
    JobFinished { job_id: String, end: JobEnd },
    JobRejected { client: String, reason: String },
    /// Result of each cloud sync: heartbeat plus paired-list refresh.
    CloudSync { ok: bool, error: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobEnd {
    Exited { code: Option<i32>, timed_out: bool, usage: Option<JobUsage> },
    Failed { reason: String },
    Cancelled,
    /// The client went away before the job reported an end.
    Disconnected,
}
