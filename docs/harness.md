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
│ ┌──────────────────────── apps/bc-cli/src/agent ─────────────────────────────────────┐ │
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

## 2. What is built

| Part | Built | Where |
|---|---|---|
| Loop, approvals, stop | Steps until no call, 25-step cap, a 16,384-token budget per turn, repeat guard, Allow / Always / Deny, Stop at the next token | `agent/mod.rs` |
| Prompt and KV reuse | Conversations kept as token ids, append-only; the stages keep the stream; a step sends only new tokens (98–99% reuse measured in the app); per-message notes (time, context use) go at the end of the user turn, never into the prefix | `Cached` in `bc-mesh/src/llm.rs`, `agent/mod.rs` |
| Router | Shortcuts that read the message: a reminder with a time; news or the web; past-conversation phrases ("I mentioned", "we discussed"); folder words or a named file; and on a conversation's first message, a small folder (at most 4 files of 16 KB) is read whatever the words. The harness then makes the calls itself: a search and its top 2 pages, the files, or recall. A **decision step** (the model picks one numbered action, decoding held to those digit tokens by `Sampling::allow`) is built but off by default: see the measurements below | `agent/route.rs`, `harness_calls`, `files_for` |
| Run log | `<home>/runs/<conversation>.jsonl`: message, route, each answer with token counts, each call before and after, compaction, end reason. A turn cut off by a crash is reported, with the calls that may have run, to the person and the model. `bc agent list`, `bc agent log <id>` | `agent/log.rs` |
| Context | Tool output past its room is kept whole as an artifact (`read_artifact`); old tool results trimmed; then everything before the current turn replaced by the model's summary | `run_call`, `fit`, `summarize` |
| Model gateway | The mesh `Llm`, or any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, vLLM) through `/v1/completions` with a byte-level vocabulary in the model's chat format, used when no mesh model runs | `agent/http.rs`, `Tokenizer::byte_level` |
| Tools | Folder-confined files, `run` in the job sandbox here or on a paired device, `web_search` (news and web), `fetch_url` (public https only), `recall`, `remember`, `remind`, `read_artifact`, `use_skill`, `task` | `agent/tools.rs` |
| MCP | stdio and Streamable HTTP servers from `<home>/mcp.json`; tools as `<server>__<tool>`; `find_tools` (BM25) past 30 tools; each call asks unless the server is `readOnly` | `agent/mcp.rs` |
| Sub-agents | `task(goal)`: a helper with the same model and tools less `task`, a fresh context and the same approvals; only its answer returns | `subtask` |
| Memory | Saved conversations, BM25 recall (the latest conversations when no words match) pulled in when a message refers to the past, notes, reminders (notifications), skills (`<home>/skills/<name>/SKILL.md`) | `agent/memory.rs`, `agent/remind.rs` |
| Evals | `bc agent eval <cases> --url --model`: cases with an expected route, tool and answer words, each in a fresh home, side effects denied; 20 starting cases | `agent/eval.rs`, `docs/agent-evals/` |
| Streaming API | OpenAI-compatible endpoint with tool calls and prompt reuse | `apps/bc-cli/src/serve.rs` |
| Sandbox | Per-job Seatbelt or AppContainer, workspace deleted after each job | `apps/bc-agent` |

Measured with `bc agent eval docs/agent-evals/cases.json --mesh qwen3-0.6b` on an M-series
MacBook Air, the model on this computer alone (2026-11-23):

| Harness | Passed | Time |
|---|---|---|
| Before the shortcuts below, decision step on | 11 of 20 | |
| Before the shortcuts, decision step off | 6 of 20 | |
| Past-conversation and folder shortcuts, worked examples in the decision prompt | 17 of 20 (on), 18 of 20 (off) | |
| Plus reading a small folder on the first message: decision step off | **20 of 20** | 393 s |
| The same, decision step on | 18 of 20 | 614 s |
| Sampled answers (temperature 0.7, top-p 0.8), the time only on messages about time, search results filtered to the topic | 18 of 20 twice (17 × 23 answered wrong once, a file answer missed once; greedy decoding repeated the last answer to "hmmm") | 143 s |

The decision step picked "answer" for every message before it had worked examples, and
afterwards added time without a pass the shortcuts missed, so it is off by default
(`Settings::decision`; `bc agent eval --decision` measures it for other models). One run
with it on stopped with "the GPU stopped with status Error" on a long web-results prompt,
an engine bug to chase. qwen3-8b is not measured yet.

Not built yet, in the order they would pay off:

1. **A smaller, faster key/value cache.** The GPU error above was the cache running out
   of memory: f32 keys and values (229 KB a token for qwen3-0.6b) doubled from 8,192 to
   16,384 tokens while the old cache was still held. The cache now grows by a quarter,
   and a session's context is capped at what its stages hold (`context_room`: memory
   left after the weights, over the cache a token takes, 2.25 times for growing; at least
   8,192 tokens, where qwen3-0.6b ran on this M2), so the
   agent trims and summarizes before the GPU runs out. Storing the cache as f16 would
   halve it, and an attention kernel that shares each key and value read across query
   rows would end the slowdown on long prompts (9 s a 256-token chunk at 8k tokens,
   from re-reading the whole cache per row and head).
2. **Grammar-constrained tool arguments.** The decision step holds one token to a range;
   holding a whole JSON object to a schema needs a token mask per step sent to the last
   stage, a wire change.
3. **Deterministic replay through the loop.** The log records every answer and result;
   feeding them back instead of calling the model and tools would turn any run into a
   test.
4. **A mid-turn fallback.** The model server is used when no mesh model runs; a mesh
   failure mid-turn still ends the turn.
5. **Browsing pages that need scripts** (a hidden web view), a folder index for ranked
   search, and embedding recall.

## 3. Data

No database: one JSONL run log per conversation beside the conversation file, appended
as events happen and synced at the end of a turn, a torn last line skipped on read; large
outputs as files beside it; reminders in `<home>/reminders.json`; MCP servers in
`<home>/mcp.json`; the model server in `<home>/model-server.json`; skills in
`<home>/skills/`. All of it stays on the device that runs the agent.

```rust
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEvent {
    Turn { at: u64, text: String },
    Route { at: u64, route: String },
    Model { at: u64, text: String, prompt_tokens: usize, cached_tokens: usize, generated: usize },
    Started { at: u64, call: String, name: String, arguments: Value, by_harness: bool },
    Finished { at: u64, call: String, ok: bool, text: String },
    Compacted { at: u64, messages: usize, summary: String },
    Ended { at: u64, reason: String },
}
```

## 4. Layout

The agent stays in `apps/bc-cli/src/agent` (moving it to a crate of its own would only
move files):

```
mod.rs     Agent, Brain, Chat, the loop, harness calls, run_call, fit and summarize, subtask
route.rs   fast paths and the decision step
log.rs     the run log
format.rs  the Hermes tool-call format and its parser
tools.rs   built-in tools, MCP routing, risk, find_tools
mcp.rs     the MCP client
http.rs    the model-server brain
memory.rs  conversations, recall, notes, skills
remind.rs  reminders and the time parser
eval.rs    evals
```

## 5. Components

The design each part follows; section 2 says what of it is built.

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

- Episodic: conversations and run logs, BM25 recall when a message refers to the past
  (built). Memory sent with every message was dropped: qwen3-0.6b repeated it back in
  place of an answer. Add recency weighting to recall.
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

| Phase | Scope | Status |
|---|---|---|
| 0 | Public-pages-only `fetch_url` and `web_search`; reminders; context use | Built |
| 1 | Run log, cut-off turns, turn budget, the constrained decision step | Built; measured on qwen3-0.6b (20 of 20 with the shortcuts; the decision step off by default) |
| 2 | Artifacts and compaction, the model-server brain, MCP with `find_tools` | Built; fallback only when no mesh model runs |
| 3 | Sub-agents, skills, evals | Built; replay, a folder index and script-rendered pages remain |

Each part has tests with the scripted brain (and a stand-in model server and MCP
server), and the eval set has run on real qwen3-0.6b weights (section 2); qwen3-8b is
next.

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
