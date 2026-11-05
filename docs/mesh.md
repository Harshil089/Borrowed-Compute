# Mesh: several devices working as one

Goal: one device uses several of the owner's devices at once, on any network, and runs work across them that none could run alone — first of all a large language model, with no cloud. A coordinator in the middle decides where work goes and moves it as devices get busier or freer.

Other tools that split a model over several computers (exo, llama.cpp's RPC backend, Petals-style swarms of strangers aside) assume the computers share a local network. Borrowed Compute already connects paired devices across the internet, through NATs, with a relay as the fallback. The mesh builds on that. The hard part is the one the owner named: **won't the constant movement of data make it slow?** This document answers that with numbers, then describes what is built and what comes next.

## 1. Where the bytes are

Take Llama 3.3 70B at 4 bits: 42.5 GB of weights, 80 layers of about 0.52 GB, a hidden state of 8,192 values.

| What moves | Size | When |
|---|---|---|
| Weights | 42.5 GB | Never over the mesh (see 3.4) |
| Hidden state per token per hop (int8) | 8.7 KB | Every generated token |
| Hidden state per token per hop (fp16) | 16 KB | (not used) |
| A 4,000-token prompt per hop (int8) | 35 MB | Once per prompt |
| KV cache per token, all layers | 320 KB | Never moves; stays on the stage that made it |

Three conclusions:

1. **Generating tokens is not bandwidth-bound.** 8.7 KB a token is 87 KB/s at 10 tokens a second, which any link carries, relay included. What costs time is **latency**: every token crosses every hop in turn.
2. **Prompts are bandwidth-bound.** 35 MB through a relay at 2 MB/s is 17 seconds. This is the real throughput problem, and it shrinks with chunking, compression, direct paths and, above all, never resending what a stage already has (3.6).
3. **Weights must never travel at run time.** 42.5 GB at 50 Mbit/s is two hours. Each device gets its layers once, from the source, not from another device.

## 2. Why pipeline parallelism, and only that, crosses devices

There are two ways to split one model:

- **Tensor parallelism** splits every layer across devices. Each layer then needs two all-reduces: 160 network round trips per token for 80 layers. At 40 ms a round trip that is 6.4 seconds a token. It works inside one computer or a rack, never over the internet.
- **Pipeline parallelism** gives each device a contiguous range of layers. One hidden state crosses each boundary once per token. Three devices make two internal hops.

So: pipeline parallelism between devices, and whatever a device does internally (several GPUs, Metal, CPU threads) stays its own business.

## 3. Design

### 3.1 A ring, not a star

The coordinator sends token ids to the first stage. Each stage runs its layers and sends the hidden state **straight to the next device** on its own connection. The last stage sends the result back. A star, where every stage returns its output to the coordinator, would cross the slowest link twice per hop and put the coordinator's upload on the critical path of every layer boundary.

Stages that never paired with each other still have to talk. The coordinator signs a **ticket** per hop: "in session S, device A may feed device B", valid five minutes, ed25519 with the coordinator's device key (`ticket.rs`). B accepts A's connection only with a ticket signed by the coordinator that set B's stage up, naming A as sender and B as receiver. Every stage still checks that the coordinator itself is on its paired list. Guests are never part of a pipeline: activations reveal what the prompt says.

### 3.2 Latency or throughput: the planner's two goals

Measured inputs per device: free memory, memory bandwidth. Per link: round-trip time and upload rate. The probe measures all of them (`probe.rs`), for every paired device at once.

Decoding one token reads every weight once, so a layer's time on a device is its bytes divided by the device's memory bandwidth (with 70% efficiency). That gives two different optimal placements (`plan.rs`):

- **One conversation (latency).** The token visits every stage in turn, so time per token = the sum of all stages' work + all hops. Adding devices never makes one conversation faster; it only adds hops. The planner fills the fastest devices first and uses only as many as memory requires. **Many devices exist to fit bigger models, not to speed one up.**
- **Many requests (throughput).** With a request in flight on every stage, the slowest stage sets the pace. The planner balances layers so each stage takes equal time (assigning each layer where the stage total stays lowest, which is optimal for the slowest stage). When devices can hold the whole model, or a small group can, it makes several independent pipelines (replicas) instead of one long one, choosing the pipeline with the most tokens a second per device used. Two 2-device replicas serve as much as one 4-device pipeline with half the hops.

The planner also chooses the ring order, trying every subset and order of up to eight devices, so a far-away device sits in the middle of the ring instead of next to the coordinator.

### 3.3 Balancing live

The owner's devices are also used by their owners. When one gets busy, its measured speed drops and the slowest stage slows the whole pipeline. `replan` takes fresh measurements and moves layers only when the new placement is at least 15% better (hysteresis), so a short burst of load does not reshuffle the model. Changes apply between requests, because each stage's KV cache covers its own layers.

To make a shift cheap, each device keeps a margin of layers on either side of its range on local disk (planned, M4). Moving a boundary within the margin is instant: no download, only a new routing table.

### 3.4 Weights come from the source, by the byte range

Each device downloads only its layers, directly from where the model is published (Hugging Face), using HTTP range requests against the GGUF or safetensors index. A device holding 27 of 80 layers downloads 14 GB once. Nothing goes through another device or a relay.

**Owner decision needed.** The zero-data rule (2026-10-25) says the lending device keeps no project data. Model weights are not project data, but they are tens of gigabytes on someone's disk. The proposal: weights live only for a mesh session and are deleted when it stops, by default. An opt-in "keep this model" setting would skip the download next time. Until decided, M2 will build only the session-scoped default.

### 3.5 Hiding the latency: speculative decoding

Per-token latency through a 3-device ring over the internet is about two round trips, 80 ms at 40 ms per round trip, on top of compute. Speculative decoding cuts that:

- A small draft model on the coordinator (Llama 3.2 1B, for example) guesses the next several tokens locally.
- One pass through the ring checks all of them at once. Decoding is memory-bound, so checking 6 tokens costs about the same as computing 1.
- Typical chat acceptance is 2.5–3.5 tokens per pass, so network time per token drops by that factor.

The planner already takes this into account (`--speculate`). This is the single biggest answer to "won't the relay make it slow": the relay's added round trip gets shared across several tokens.

### 3.6 Prompts: chunked, compressed, sent once

- **Int8 on the wire, with a scale per group of 64 values** (`codec.rs`). This uses half the bytes of fp16. LLM hidden states have a few huge outlier channels, so one scale per row would flatten every other value to zero; per-group scales confine an outlier to its own group. Tests show under 1% error outside the outlier's group.
- **Chunked prefill.** A long prompt goes in chunks of a few hundred tokens. Stage 1 computes chunk 2 while chunk 1 crosses to stage 2, so transfer overlaps compute instead of adding to it.
- **Nothing is sent twice.** Each stage keeps its KV cache for the conversation, in memory only. In a chat, each turn sends only the new message's tokens, not the whole history again.
- **Direct paths where it matters.** The probe reports relay or direct per link. The planner puts boundaries on the fastest links. iroh keeps trying to punch through, and moves the connection off the relay mid-session when it can.

### 3.7 Inside a stage

Each stage runs three tasks joined by short queues: reading, computing and writing (`stage.rs`). It receives batch n+1 and sends batch n−1 while computing batch n. With several requests in flight, every device works at once. The tests check it: 8 batches through 3 stages that each take 150 ms must finish well under the 3.6 s they would take one after another (ideal: about 1.5 s).

## 4. What is built (M1)

`crates/bc-mesh`:

| Module | Does |
|---|---|
| `probe.rs` | Measures a device and its link over the mesh protocol: median of 5 pings, upload rate (2 MB), free memory, cores and memory bandwidth (measured once per process). |
| `plan.rs` | The planner: layer ranges, ring order, replicas, latency and throughput estimates, speculative decoding, and `replan` with hysteresis. Includes a catalog of models at 4 bits. |
| `ticket.rs` | Signed hop tickets. |
| `codec.rs` | Int8 activations with a scale per group of 64 values; f32 when results must match exactly. |
| `stage.rs` | The device side: answers probes; runs a stage (the `Stage` trait) between the previous device and the next. |
| `pipeline.rs` | The coordinator side: sets stages up last to first, then streams batches in at one end and results out at the other. |

**Protocol.** Mesh connections use their own protocol name, `MESH_ALPN` (`borrowed-compute/mesh/1`), so the job protocol is unchanged.

**Agent.** The agent serves mesh connections from paired devices while lending is on. It answers probes but turns stages away: there is no inference engine yet.

**CLI.**
- `bc mesh probe` measures every paired device at once.
- `bc mesh plan <model> [--throughput] [--speculate N]` shows where each layer range would run, and how fast.
- The same steps are in `bc_cli::mesh`, ready for the desktop app.

**Tests.** `crates/bc-mesh/tests/pipeline.rs` runs, through a local relay with direct paths off:
- a stand-in model split over three devices, matching a single-device run bit for bit with f32, and closely with int8;
- the stage overlap;
- three devices probed at once;
- refusal by a device without an engine;
- refusal of strangers' probes and forged feeds.

The planner has unit tests for every rule above.

## 5. Roadmap

- **M2 — the engine.** A `Stage` that runs real layers: candle (Rust, Metal and CUDA, reads GGUF) with a quantized Llama/Qwen forward pass limited to a layer range. Token ids in at the first stage, sampled tokens out at the last; a KV cache per stage; chunked prefill; weights fetched by byte range, kept only for the session. Why not llama.cpp's RPC backend? It is a star: the main host drives every device's graph and pushes weights over the network at load. That is the opposite of 3.1 and 3.4.
- **M3 — serving.** `bc mesh serve <model>` exposes an OpenAI-compatible endpoint on the coordinator's localhost. Requests are routed across replicas, with continuous batching on each stage.
- **M4 — live balancing.**
  - Stages report time per batch and queue depth.
  - `replan` runs on a timer and applies between requests.
  - Layer margins (3.3) make shifts instant.
  - Devices probe each other with tickets, so the planner uses real device-to-device links instead of the upper bound it uses now.
  - When a stage drops out, the pipeline is replanned and the requests in flight restart from their tokens.
- **M5 — speculative decoding**, with a draft model on the coordinator.
- **M6 — the app.** A Mesh page: devices and links, the plan, live load per stage, and "start serving".

## 6. Limits to know

- One conversation gets no faster from more devices (3.2). The mesh buys bigger models and more parallel requests.
- Through a relay, prompts of thousands of tokens take seconds to cross each hop. Multi-turn chat sends only the new turn, so this is paid once per conversation, not per message.
- Activations reveal the prompt, so only the owner's own paired devices take part. Guests never do.
