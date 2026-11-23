# Harness: the agent runtime around the mesh model

The Agent page runs a model on the person's own devices and lets it work with tools. The
harness is everything around the model: what it is told, which tools it gets, what runs
without it, what is remembered, what stops it, and what is recorded. A small model on a
laptop mesh is weak at planning and at deciding when to call a tool, so the harness must
carry as much of the work as it can: **the quality of an answer should depend on the
harness first and the model second.**

This document maps a full "universal platform harness" brief onto Borrowed Compute. Much
of that brief assumes a hosted multi-tenant service (Kubernetes, Postgres, S3, RBAC,
SOC 2 audit). Borrowed Compute is one person's devices, local-first, with zero data on
lenders (owner decision 2026-10-25) and weights in RAM only (2026-11-06). Those parts are
listed under "Not doing" with the reason; everything else is planned here in phases,
each one a PR with tests.

## 1. Architecture

```
┌────────────────────────── Desktop app (Tauri, one process) ──────────────────────────┐
│ Agent page (ui/agent.js)        Mesh page (ui/mesh.js)       OpenAI endpoint (serve.rs)│
│      │ agent_send / approve / stop     │ start model               │ /v1/chat/completions│
│      ▼ agent-event (typed, ordered)    ▼                           ▼                     │
│ ┌──────────────────────────── crates/bc-harness ────────────────────────────────────┐ │
│ │ Run loop ── Policy (approval, budgets, repeat guard, step limit)                    │ │
│ │   │  ├─ Router: harness-side actions before the model (search, remind, recall)     │ │
│ │   │  ├─ Context manager: prompt layout for KV reuse, budgets, offload, compaction  │ │
│ │   │  └─ Run log: append-only events per run (resume, replay, evals)                │ │
│ │   ▼                                                                                 │ │
│ │ Model gateway (Brain) ── mesh Llm │ OpenAI-compatible HTTP (Ollama, LM Studio, …)  │ │
│ │   │      constrained decoding (JSON schema to grammar), fallback, token budget     │ │
│ │   ▼                                                                                 │ │
│ │ Tool registry ── native tools │ MCP servers (stdio, HTTP) │ sub-agents             │ │
│ │   │      middleware: validate, risk, approve, cache, clip, offload, record         │ │
│ │   ▼                                                                                 │ │
│ │ Memory ── conversations │ BM25 recall │ notes │ reminders │ skills │ artifacts     │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│      │ run                         │ stages (activations only)                           │
└──────┼─────────────────────────────┼────────────────────────────────────────────────────┘
       ▼                             ▼
  Job sandbox (Seatbelt /       Paired devices: LLM stages (bc-mesh), jobs (bc-agent)
  AppContainer), here or on a
  paired device
```

Control plane: the run loop, policy and router, all in the app's process. Data plane:
model text, tool results and fetched pages, always treated as untrusted and kept in
`<tool_response>` turns, never in the system prompt.

## 2. Where it stands (release 0.3.37 and the open PRs)

| Brief section | Built | Where |
|---|---|---|
| Agent loop, approvals, stop | Steps until no call, 25-step cap, Allow / Always / Deny, Stop at the next token | `apps/bc-cli/src/agent/mod.rs` |
| Prompt and KV cache reuse | Conversations kept as token ids, append-only; stages keep the stream; a step sends only new tokens (98–99% reuse measured in the app) | `Cached` in `bc-mesh/src/llm.rs` |
| Tools | Folder-confined files, `run` in the job sandbox here or on a paired device, `web_search`, `fetch_url`, `recall`, `remember`, `remind` | `agent/tools.rs`, `agent/remind.rs` |
| Harness-side actions | Search plus the top 2 pages when the message asks for the web; reminder when the message says "remind me" with a time; related past lines with every message | `web_query`, `reminder_request`, `harness_calls`, `Memory::related` |
| Memory | Saved conversations, BM25 recall, the latest 8 conversations in each new prompt, notes | `agent/memory.rs` |
| Resilience | Repeat guard, model switch resets token ids, context trimming of old tool results | `agent/mod.rs` |
| Streaming API | OpenAI-compatible endpoint with tool calls and prompt reuse | `apps/bc-cli/src/serve.rs` |
| Sandbox | Per-job Seatbelt or AppContainer, workspace deleted after each job | `apps/bc-agent` |

Gaps, by what they cost the person today:

1. **Small models fail to decide.** They answer "I can search for that" instead of
   calling. The router covers news and reminders; everything else still depends on the
   model. Fix: a constrained decision step (Phase 1).
2. **Malformed calls.** Free-text JSON from a 1–4B model breaks often. Fix: grammar-
   constrained decoding for tool calls (Phase 1).
3. ~~**`fetch_url` could reach private addresses**~~ (`https://192.168.1.1`,
   `https://localhost`) if a page told the model to: an SSRF path into the home network.
   Fixed in Phase 0: `check_url` and the `PublicOnly` resolver.
4. **Nothing survives a crash mid-turn**, and nothing can be replayed. Fix: the run log
   (Phase 1).
5. **Context fills on long conversations** and only old tool results get trimmed. Fix:
   offload and compaction (Phase 2).
6. **Only built-in tools.** Fix: MCP client (Phase 2).
7. **One model backend.** Fix: an OpenAI-compatible HTTP brain and fallback (Phase 2).

## 3. Core data models

These are the types the phases add, in Rust, next to the existing ones (`ToolSpec`,
`Call`, `Risk`, `Event`, `Conversation`).

```rust
/// One line of `<home>/runs/<conversation>.jsonl`, append-only. Every non-deterministic
/// input is here, so a turn can be resumed after a crash or replayed with no model.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEvent {
    Turn { seq: u64, at: i64, text: String, settings: Settings, model: String },
    Model { seq: u64, at: i64, step: u32, tokens: Vec<u32>, stats: Stats },
    Decision { seq: u64, at: i64, call: String, decision: Decision },
    /// Written before the tool runs: on resume, a started call with no `Finished` is
    /// not repeated blindly if it can change something; the person is asked.
    Started { seq: u64, at: i64, call: String, name: String, arguments: Value, risk: Risk, by: Caller },
    Finished { seq: u64, at: i64, call: String, ok: bool, text: String, artifact: Option<String> },
    Compacted { seq: u64, at: i64, upto: u64, summary: String },
    Ended { seq: u64, at: i64, reason: EndReason },
}

pub enum Caller { Model, Harness }
pub enum EndReason { Answered, Stopped, StepLimit, Repeating, Budget, Error(String) }

/// What a run may spend and do. Defaults come from Settings; sub-agents get a share.
pub struct Policy {
    pub max_steps: u32,
    pub max_generated: usize,            // tokens written in one turn
    pub allowed: HashSet<String>,        // tools allowed for the conversation
    pub auto_approve: bool,
    pub tools: Option<HashSet<String>>,  // a sub-agent's scoped tool set
}

/// A tool's large output, kept on this device instead of in the context.
pub struct Artifact { pub id: String, pub call: String, pub bytes: u64, pub mime: String }

/// A tool from any source, as the registry sees it.
pub struct ToolDef {
    pub spec: ToolSpec,                  // name, description, JSON schema
    pub risk: Risk,
    pub source: ToolSource,              // Native, Mcp { server }, SubAgent
    pub idempotent: bool,                // safe to run again on resume
}
```

No database: one JSONL file per conversation, beside the conversation file, written
with `O_APPEND` and fsync at turn end. A torn last line is dropped on read.

## 4. Repository layout

```
crates/bc-harness/            (moved out of apps/bc-cli/src/agent, which re-exports it)
  src/lib.rs                  Agent, Brain, Settings, Event
  src/run.rs                  the loop, policy, repeat guard, resume
  src/log.rs                  RunEvent, append, read, replay
  src/router.rs               web_query, reminder_request, the constrained decision step
  src/context.rs              prompt layout, budgets, offload, compaction
  src/format.rs               Hermes format, parser, grammar from JSON schema
  src/gateway/{mesh,http}.rs  Brain for the mesh Llm and for OpenAI-compatible servers
  src/tools/{native,mcp,subagent}.rs, src/tools/middleware.rs
  src/memory/{conversations,recall,notes,reminders,skills}.rs
  src/eval.rs                 replay a recorded run against a Script brain or a model
  tests/                      scripted-brain trajectories, MCP stand-in server
crates/bc-infer/src/grammar.rs   constrained sampling (token masks from a grammar)
apps/bc-desktop/src/agent.rs     commands, runner, reminders, MCP server settings
apps/bc-cli/src/main.rs          `bc agent eval <run>`, `bc agent replay <run>`
```

## 5. Components

### 5.1 Run loop: durable, resumable, replayable

- Every `RunEvent` is appended before its effect is visible: `Started` before a tool
  runs, `Model` before its calls are parsed. Resume reads the log, rebuilds the token
  ids, and continues from the last event. A `Started` without `Finished` runs again only
  if the tool is idempotent (read, search, fetch); otherwise the person is asked.
- Approval waits are already suspendable in memory; with the log they survive a restart:
  the pending call shows again when the conversation opens.
- Replay feeds recorded `Model` and `Finished` events instead of calling the model and
  tools, so any run becomes a deterministic test (`bc agent replay`), and a recorded run
  with a new model or prompt becomes an eval (`bc agent eval`): same tool results,
  compare the calls and the answer.

### 5.2 Router: the harness acts before the model

Today: keyword rules for the web and reminders. Next, a **decision step** that works with
any model because the output is constrained: before the free answer, the model fills

```json
{"action": "answer" | "web_search" | "read_files" | "run" | "recall" | "remind", "argument": "..."}
```

with decoding masked to that grammar, so even a 0.6B model can only produce a valid
choice. The harness then makes the call itself (as `harness_calls` does now) and the
model writes the answer from the results. It costs one short constrained generation per
message, on a prefix the stages already hold. The keyword rules stay as a fast path.

### 5.3 Model gateway

- `Brain` stays the interface. Add an OpenAI-compatible HTTP brain (Ollama, LM Studio,
  llama.cpp server, vLLM, or a hosted API with its key in the OS keychain through
  `bc-secrets`, never in a prompt or log).
- Fallback: an ordered list of brains; a brain that fails or answers slower than its
  measured speed by a margin is skipped for a minute (circuit breaker), and the turn
  continues from the run log on the next one. Mesh first, since it keeps data on the
  person's devices.
- Constrained decoding in `bc-infer`: compile a JSON schema (the tool's parameters) to a
  token mask per step; used for tool-call arguments and the decision step. Speculative
  decoding still works: the target's mask filters the draft's guesses.
- Budgets: `Policy.max_generated` per turn, counted from `Stats`.

### 5.4 Tool registry and middleware

Every call goes through one pipeline, in order: parse, validate against the schema
(repair message back to the model on failure), repeat guard, risk and approval, rate
limit, run with a timeout, clip, offload above 8 KB to an artifact, record. Native tools
and MCP tools share it.

- **MCP client**: stdio servers started by the app (configured on the Agent page), and
  Streamable HTTP servers. Tools are named `server__tool`; their risk is Run unless the
  person marks a server read-only. With more than 30 tools, the prompt lists only a
  `find_tools(query)` tool (BM25 over names and descriptions) and loads schemas on
  demand, so the prompt stays small and cached.
- **Sub-agents**: a `task(goal, tools)` tool runs a child `Agent` with its own context, a
  subset of tools and a share of the budget; only its final answer returns to the parent.
- **Browser**: `fetch_url` reads static pages. Pages that need scripts get a hidden
  Tauri web view later (Phase 3), never the person's own browser profile.

### 5.5 Context manager

- Layout for reuse: system prompt and tool list first and stable for a conversation,
  then turns, append-only (as now). Anything that changes per message (time, related
  lines, context use) goes at the end of the user turn, never into the prefix.
- Offload: a tool result over 8 KB is saved as an artifact under
  `<home>/runs/<conversation>/` and the model gets its head and tail plus
  `read_artifact(id, offset)`.
- Compaction: near the limit, old turns are summarized by the model into one
  `Compacted` event, and the ids are re-rendered once (one full prefill), instead of
  refusing with "start a new one".

### 5.6 Memory

- Episodic: conversations and run logs, BM25 recall, the latest 8 in each prompt,
  related lines with each message (built). Add recency weighting to recall.
- Semantic: index the chosen folder's files with BM25 at conversation start (code
  chunked by blank lines and definitions), so `search` ranks instead of greps. Embedding
  search waits for a small embedding model on the mesh.
- Procedural: skills in `<home>/skills/<name>/SKILL.md` (instructions plus scripts),
  listed by name and one line in the prompt, loaded in full with `use_skill(name)`.

## 6. Threat model

| Threat | Path | Mitigation |
|---|---|---|
| Prompt injection from a page or file | Fetched text tells the model to run, write or send something | Anything but reading asks the person (Allow / Always / Deny); results stay in `<tool_response>` turns; harness-side calls come only from the person's own words, never from tool output |
| SSRF into the home network | `fetch_url("https://192.168.1.1/…")` from injected text | Literal addresses checked before the request; names resolved only to public addresses (`PublicOnly`, so no gap between check and connect); every redirect checked again; https only (built, Phase 0) |
| Data leaving through search | Injected text makes the model search with private file contents | `web_search` is Net risk when the model calls it, so it asks; the harness's own searches use only the person's message |
| Sandbox escape | `run` with a hostile command | Seatbelt or AppContainer per job, workspace deleted after, the real folder never mounted (built) |
| Folder escape | `..`, absolute paths, symlinks | `Tools::resolve` canonicalizes and refuses (built, tested) |
| Token and step exhaustion | A model looping or writing forever | Step limit, repeat guard (built), per-turn token budget (Phase 1) |
| Prompt exposure on the mesh | Activations reveal the prompt to every stage | Mesh only on paired devices, guests only when the owner allows (built) |
| Secrets in context | API keys for hosted brains or MCP servers | Kept in the OS keychain, passed at the tool or HTTP boundary, never rendered into prompts or run logs (Phase 2) |
| Malicious MCP server | A server's tool descriptions inject instructions | Servers are added by the person only; their tools are Run risk by default; descriptions are clipped |

## 7. Phases

| Phase | Scope | Done when |
|---|---|---|
| 0 | SSRF guard on `fetch_url` and `web_search`; reminders and context use (built) | Tests refuse loopback, private, link-local, metadata and IPv6-wrapped addresses, decimal forms like `https://2130706433`, and `localhost` by name; redirects go through the same check, not yet covered by a test |
| 1 | Move to `crates/bc-harness`; run log with resume and replay; constrained decoding in `bc-infer`; the decision step; token budget | A turn killed mid-call resumes without repeating a write; `bc agent replay` reproduces a recorded run exactly; qwen3-0.6b picks the right action on a 30-prompt set at least 90% of the time (measure the rate without the step first, as the baseline) |
| 2 | OpenAI-compatible brain and fallback; MCP client (stdio and HTTP) with `find_tools`; artifact offload; compaction | An MCP filesystem server's tools work end to end; a 200-message conversation keeps going; killing the mesh mid-turn falls over to a local brain |
| 3 | Sub-agents; skills; folder index; hidden-web-view browsing; eval sets in CI with the Script brain | Eval suite runs on every PR in under a minute with no model |

Verification for each phase: unit tests with the scripted brain, a recorded-run replay
test, and a real-model check on qwen3-0.6b and qwen3-8b (the two catalog models with
tool training), reported with its numbers.

## Not doing

- **Multi-tenant RBAC, ABAC, OAuth for the harness, per-tenant budgets**: one person's
  devices; pairing and the cloud's account checks already decide who may use what.
- **Postgres, Redis, S3, Kubernetes, Helm**: conversations and run logs are files on the
  device that runs them, by the owner's zero-data decision; there is no server to scale.
- **Firecracker or gVisor**: the job sandbox (Seatbelt, AppContainer) already isolates
  `run`, on Mac and Windows where those do not exist.
- **Tamper-evident (hash-chained) audit logs for compliance**: the run log is for resume,
  replay and evals; no compliance regime applies. Adding a hash chain later is cheap.
- **TypeScript and Python SDKs, YAML workflows**: the OpenAI-compatible endpoint is the
  integration surface; agent tools such as opencode and Aider already speak it.
- **Anti-bot and proxy management for browsing**: the agent never tries to pass bot
  checks (the reason `web_search` uses Bing's RSS and not DuckDuckGo's page).
