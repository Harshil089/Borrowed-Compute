# Mesh: several devices working as one

Goal: one device uses several of the owner's devices at once, on any network, and runs work across them that none could run alone — first of all a large language model, with no cloud. A coordinator in the middle decides where work goes and moves it as devices get busier or freer.

Other tools that split a model over several computers (exo, llama.cpp's RPC backend, Petals-style swarms of strangers aside) assume the computers share a local network. Borrowed Compute already connects paired devices across the internet, through NATs, with a relay as the fallback. The mesh builds on that. The hard part is the one the owner named: **won't the constant movement of data make it slow?** This document answers that with numbers, then describes what is built and what comes next.

## 1. Where the bytes are

Take Llama 3.3 70B at 4 bits: 42.5 GB of weights, 80 layers of about 0.52 GB, a hidden state of 8,192 values.

| What moves | Size | When |
|---|---|---|
| Weights | 42.5 GB | Never over the mesh (see 3.4) |
| Hidden state per token per hop (exact f32) | 32 KB | Every generated token |
| A 4,000-token prompt per hop (exact f32) | 131 MB | Once per prompt |
| KV cache per token, all layers | 320 KB | Never moves; stays on the stage that made it |

Three conclusions:

1. **Generating tokens is not bandwidth-bound.** 32 KB a token is 320 KB/s at 10 tokens a second, which any link carries, relay included. What costs time is **latency**: every token crosses every hop in turn.
2. **Prompts are bandwidth-bound.** 131 MB through a relay at 2 MB/s is a minute; through a direct 100 Mbit/s path, 10 seconds, about what the stages need to compute it anyway. It shrinks with chunking, direct paths and, above all, never resending what a stage already has (3.6).
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

### 3.1a This computer is part of its own mesh

The coordinator is a device too. It measures its own memory and GPU (`bc_cli::mesh::this_computer`) and the planner may give it layers, always the first ones: tokens start on this computer, so its stage costs no hop, and its links are the coordinator's own (`plan::estimate` treats them as free). It runs the same engine and fetches its layers the same way as a lending device, into memory only. When it holds the whole model, nothing crosses the network at all. Two computers with 5 GB free each now run a model neither could hold alone.

### 3.1b Shared computers

A computer someone shared with you (as a guest, through an invite) can be part of your Mesh too, but only if its owner turns on "Let guests use this device in their Mesh" on its Sharing page. It is off by default: hidden states passing through a stage reveal the conversation, so the owner's computer sees it, which the guest accepts by using it (the app marks it as someone else's) and the owner by allowing it. Without the switch the shared computer turns the probe away as unpaired, and the Mesh page says the owner has not allowed it.

### 3.2 Latency or throughput: the planner's two goals

Measured inputs per device: free memory, memory bandwidth, and its GPU if it has one (name, memory for weights, and the speed a real decoding product reaches on it). Per link: round-trip time and upload rate. The probe measures all of them (`probe.rs`), for every paired device at once.

Decoding one token reads every weight once, so a layer's time on a device is its bytes divided by the speed it reads them at. A device has up to two tiers (`stage_secs`): the layers its GPU holds run at the GPU's measured speed, the rest at 70% of its RAM's copy bandwidth. On Apple's unified memory the GPU reaches every layer in RAM; a discrete card holds what fits in 85% of its free memory, and the engine puts a stage's first layers there and the rest on the CPU. Each layer's cost only grows as a stage grows, so filling layer by layer where the next one costs least is optimal: GPU memory anywhere first, then the fastest RAM. That gives two different optimal placements (`plan.rs`):

- **One conversation (latency).** Without a draft model, the token visits every stage in turn, so time per token = the sum of all stages' work + all hops, and the planner fills the fastest devices (GPUs first) and uses only as many as memory requires. With a draft, pipelined speculation (3.5) lets more stages work at once, and the planner's estimate weighs that against the extra hops.
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

**Pipelined speculation: more devices make one conversation faster.** With one batch at a time, a 4-stage pipeline has three stages idle at any moment. So the coordinator keeps a batch per stage (plus one) in flight while a conversation is the pipeline's only one, each continuing the draft's guesses where the previous batch's left off (the idea of PipeInfer, Butler et al. 2024). While guesses hold, tokens arrive at the pace of the slowest stage, not the sum of all stages and hops. A wrong guess makes every later batch worthless: they are dropped by an epoch number as they come back, and the next batch starts at the corrected position, where every stage cuts its cache back (each batch carries its position). The text stays exactly what the target alone writes.

How much it gains depends on the draft. With per-token agreement α, a run of α/(1 − α) right guesses costs one full pass plus a batch interval per k guesses (`speculative_token_ms`):

| Draft agreement α | Tokens a pass, one batch at a time | 4 equal stages: time a token, one at a time | Pipelined |
|---|---|---|---|
| 0.5 | 1.9 | 0.52 passes | 0.52 passes (no gain) |
| 0.8 | 3.4 | 0.30 passes | 0.25 passes |
| 0.9 | 4.1 | 0.24 passes | 0.16 passes (1.5×) |
| 0.95 | 4.5 | 0.22 passes | 0.10 passes (2.2×) |

Measured with three stages taking 60 ms each and a draft that is always right: 40 tokens in 0.88 s, against 1.8 s one batch at a time. Code, structured output and long quotations from the context are where drafts agree most.

Two more things keep guessing from ever costing time:

- **Early cancellation.** Each batch carries its epoch. A stage reads ahead of what it computes, and when it has seen a newer epoch for a conversation, older batches still queued there are passed on empty, uncomputed. Replies are handled before any more drafting, so after a miss the corrected batch goes out at once.
- **Measured, not assumed** (`llm::Speed`). Guessing pays only when a stage checks several tokens for about the price of one and the draft agrees often. On CPU stages a 5-row check costs about one token (the integer products unpack each weight once for every row); on the M2's GPU it costs about three. So each pipeline keeps its measured milliseconds a token with and without guessing, decodes the faster way, and tries the other every eighth request in case the devices or the conversation changed.

### 3.6 Prompts: chunked, exact, sent once

- **Exact values on the wire** (`codec.rs`). Version 0.3.10 sent int8 with a scale per group of 64 values: a quarter of the bytes, but its small error entered the residual stream, and after a few dozen tokens it could flip a near tie, so the text drifted from what the model alone writes. Hidden states now cross as exact `f32`, and a split model writes bit for bit what the whole one does (checked on Llama 3.2 1B over three relayed stages: all 96 tokens identical). Lossless coding was measured and rejected: three of the four bytes of each float carry 7.8 to 8.0 bits of entropy, so no lossless coder could save more than 17%. The mesh protocol is `borrowed-compute/mesh/2` since this change.
- **Chunked prefill.** A long prompt goes in chunks of a few hundred tokens. Stage 1 computes chunk 2 while chunk 1 crosses to stage 2, so transfer overlaps compute instead of adding to it.
- **Nothing is sent twice.** Each stage keeps its KV cache for the conversation, in memory only. In a chat, each turn sends only the new message's tokens, not the whole history again.
- **Direct paths where it matters.** The probe reports relay or direct per link. The planner puts boundaries on the fastest links. iroh keeps trying to punch through, and moves the connection off the relay mid-session when it can.

### 3.7 Inside a stage

Each stage runs three tasks joined by short queues: reading, computing and writing (`stage.rs`). It receives batch n+1 and sends batch n−1 while computing batch n. With several requests in flight, every device works at once. The tests check it: 8 batches through 3 stages that each take 150 ms must finish well under the 3.6 s they would take one after another (ideal: about 1.5 s).

## 4. The engine (`crates/bc-infer`)

Borrowed Compute runs models with its own engine, written for this design: any contiguous range of layers runs on its own, and weights are read by byte range straight into memory.

**Weights stay compressed in memory.** GGUF stores each layer as blocks of 32 or 256 weights sharing one or two f16 scales. The engine supports these block formats, which together cover every common 4-, 5-, 6- and 8-bit file:

| Format | Formula |
|---|---|
| Q4_0 | x = d·(q − 8) |
| Q4_1 | x = d·q + m |
| Q5_0 | x = d·(q − 16), the fifth bit packed separately |
| Q5_1 | x = d·q + m, the fifth bit packed separately |
| Q8_0 | x = d·q |
| Q4_K, Q5_K | 8 groups of 32 per 256 weights, each with a 6-bit scale and min: x = d·sc·q − dmin·m |
| Q6_K | 16 groups of 16 with i8 scales: x = d·sc·(q − 32) |

F32, F16 and BF16 are read too.

**Products are integer.** Each input row is quantized to int8 per block of 32 (scale = max|x|/127). Each weight group, unpacked once to small integers with a scale s and an offset o, then meets the input:

> Σ wᵢxᵢ ≈ d_x · (s · Σ qᵢpᵢ + o · Σ pᵢ)

The inner sum is in i32 and compiles to the CPU's integer multiply-add. This is llama.cpp's method. A block is unpacked once and used for every input row while it is in cache, so a 300-token prompt chunk or a 5-row speculative check costs little more than one token.

**Work-stealing threads.** Products are cut into small pieces handed out from a shared counter. When the system pauses a thread, for instance because a lender's owner is using the machine, it delays one small piece rather than a fixed share. `BC_INFER_THREADS` caps the cores the engine takes.

**The transformer** follows the Llama and Qwen2 families. Each layer computes:

> h = x + W_o · Attn(RoPE(W_q·n(x)), RoPE(W_k·n(x)), W_v·n(x))
> x′ = h + W_down · (SiLU(W_gate·n(h)) ⊙ W_up·n(h))

- n is RMSNorm.
- Attention is grouped-query, with causal softmax(q·kᵀ/√d)·v.
- RoPE rotates each dimension pair by p·θᵢ, θᵢ = base^(−2i/d), divided by Llama 3.1's per-frequency factors when the file has them. Llama rotates neighbouring pairs; Qwen2 rotates i with i + d/2.
- Qwen's query/key/value biases and tied embeddings are handled.

**Text.**
- **SentencePiece:** score-ordered merges with byte fallback.
- **Byte-level BPE:** rank-ordered merges after GPT-4-style word splitting. Llama 3 groups digits in threes; Qwen splits every digit.
- **Control tokens** such as `<|im_start|>` are matched whole.
- **Chat templates:** ChatML and Llama 3.

**Checked against real models.** Each was run whole and split in two; both runs give identical tokens, with the split at layer n/2.

| Model | Tested with |
|---|---|
| stories15M (Q8_0, Q4_0) | A coherent story |
| Qwen2.5-0.5B-Instruct (Q4_K_M) | "The capital of France is Paris." |
| Llama-3.2-1B-Instruct (Q4_K_M) | "391" for 17×23, "Tokyo" for Japan's capital |

### 4.1 On a GPU (`gpu/`)

Decoding is bandwidth-bound, and a GPU reads memory several times faster than a CPU's cores can, so a part runs on the device's GPU whenever there is one, with no setting: **Metal** on a Mac (Apple's GPUs, what PyTorch calls MPS), **CUDA** on an NVIDIA card elsewhere. `BC_INFER_GPU=0` turns it off; `BC_INFER_GPU=debug` says why a GPU was not used.

- **A whole layer per trip.** RMSNorm, the seven products, biases, RoPE, the key/value cache append, attention and SwiGLU all run on the GPU, every layer of a pass in one command buffer: one trip to the GPU per pass, not one per product. The key/value cache lives on the card and grows by doubling.
- **One kernel source, two backends.** `kernels.h` is plain C both compilers take. Metal compiles it when the app starts. CUDA gets it ahead of time as PTX (`kernels.ptx`, built by `build-ptx.sh` with clang alone), which the NVIDIA driver compiles for whatever card it finds: users need only the driver, not the CUDA Toolkit. A test fails if `kernels.ptx` is older than its source.
- **Weights stay in their block format** (every format the CPU reads). The common ones (Q4_0, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K) have wide-load paths: each lane of a 32-wide SIMD group takes a whole group of 32 weights, reads two or four weight bytes and four inputs per load, and the lanes sum with a SIMD reduction. Attention uses an online softmax over tiles of 1,024 positions (Milakov and Gimelshein; FlashAttention), so any context length works with fixed on-chip memory.
- **No copy on a Mac.** Weights live in page-aligned memory (`quant::Bytes`), and each Metal buffer wraps those very pages (`newBufferWithBytesNoCopy`), so the GPU gets the whole model at no extra RAM. On a discrete card weights are copied into its memory; layers that do not fit run on the CPU after the ones that do.
- **Checked before use.** At startup the GPU's product is compared with the CPU's on every weight format; on any mismatch the engine stays on the CPU and says why. On the GPU itself results are exact and repeatable: no atomics, fixed reduction order, so a batch scores each row exactly as a one-row pass does (speculative checking relies on it) and a split model matches the whole one bit for bit. CPU and GPU round differently (the CPU quantizes inputs to int8 per block, the GPU multiplies in f32), as any two kinds of hardware do; a mixed pipeline is still exact across its own hops.
- **Measured.** On this M2 (with the screen and other apps using the GPU): plain copies reach 72 GB/s; a Q4_0 product 26 to 30 GB/s, against about 5 GB/s on its CPU cores. Llama 3.2 1B writes 23.8 tokens a second on the GPU against 9.9 on the CPU, the same text, and a split still matches the whole model.

## 5. The mesh, end to end

**Lending device** (`bc_mesh::engine`, served by the agent to paired devices while lending):
- A stage setup names the model's https address and a layer range.
- The device reads the GGUF header, then only its own tensors, by byte range, straight into RAM. Nothing is written to disk.
- Weights stay in memory while a stage uses them, plus two minutes, so a rebalanced pipeline that keeps most layers on this device does not fetch them again. Then they are dropped. Quitting the app drops everything.
- A stage refuses ranges larger than 90% of free memory.
- A stage keeps a KV cache per conversation (up to 16) and cuts it back to the batch's position before running. That makes speculative rollbacks and replays exact.
- The last stage samples: greedy, or temperature with top-p and a seed. It returns token ids, never the 150k-wide score vector.
- **Owner decision, 2026-11-06:** weights may stay in memory on a lending device while the model runs, never on disk.

**Coordinator** (`bc_mesh::llm`). The draft model is shared by every request; each request keeps its own cache, and drafting runs off the async threads.
- A `Client` multiplexes any number of requests over one pipeline. Stages answer in the order batches arrive, so requests go out back to back and every stage stays busy.
- `Llm::generate` sends a prompt in chunks of 256 tokens without waiting between them (chunked prefill, as in Sarathi). It then decodes, speculatively when a draft model is loaded and the request is greedy:
  - the draft guesses 4 tokens;
  - one pass returns the target's own choice at every row;
  - the longest agreeing prefix is kept, plus the target's next token.

  The text is exactly what the target alone writes (Leviathan et al., 2023, greedy case).
- Each stage reports its compute time per batch, which feeds live balancing.

**Serving** (`bc_cli::mesh::Session`, `bc_cli::serve`):
- **Starting:** `Session::start` probes every paired device at once and keeps those with an engine. It reads the model's real sizes from its header, plans, and opens one pipeline per replica.
- **Live balancing:** every 60 seconds `rebalance` turns the measured stage times back into an effective speed per device. A device whose owner got busy looks slower and gets fewer layers. Layers move only when the new plan is at least 15% better.
- **Measured links:** before planning, every pair of devices measures the link between them, at once. Device A connects to device B with a ticket the coordinator signed, and B accepts it only from a coordinator it is paired with. The planner then orders the ring by real device-to-device round trips instead of an upper bound.
- **Recovery:** when a pipeline fails, `recover` probes again, leaves out devices that do not answer, plans again and retries the request once.
- **Endpoint:** `bc mesh serve` exposes `POST /v1/chat/completions` (streamed or not) and `GET /v1/models` on localhost, so any OpenAI client uses the mesh.

**Commands and app:**

| Command | Does |
|---|---|
| `bc mesh probe` | Measures every paired device at once |
| `bc mesh plan [model] [--throughput] [--speculate N]` | Shows where each part would run and how fast |
| `bc mesh run <model> "<prompt>" [--draft <model>|none]` | Answers one prompt |
| `bc mesh serve <model> [--port 8484] [--throughput]` | Serves the model until Ctrl+C |

`<model>` is a catalog name (Qwen2.5 0.5B to 72B, Llama 3.1 8B, 3.2 1B, 3.3 70B, Hermes 3 405B) or a GGUF https address. Split GGUF files (`…-00001-of-00007.gguf`) are read as one model: the header merges every file's tensors, and each device fetches its layers from whichever files hold them.

The desktop app has a **Mesh** page under Use. It picks the model and the goal, measures the devices, shows where each part runs (with the measured time per stage), starts and stops serving, shows the endpoint address, and has a box to try the model.

**Tests:**

| Test | Checks |
|---|---|
| Unit | Every block format against its definition; integer kernels against dequantization; tokenizers, sampling, the planner, tickets, the codec |
| `crates/bc-mesh/tests/llm.rs` | Through a local relay with direct paths off: a model split over three devices that each fetched only their own layers writes exactly the single-computer text; two conversations and a 300-token chunked prompt run at once; speculative decoding is exact with a perfect and an imperfect draft, and takes far fewer passes with the perfect one |
| `apps/bc-cli/tests/mesh_serve.rs` | An OpenAI client chats with a model served by two devices, streamed and not, with the same answer both ways |
| `real_model_over_three_devices` (ignored; `BC_TEST_MODEL=…`) | Qwen2.5-0.5B over three relayed stages on one Mac: 33 tokens/s (stages at 7.6, 7.7 and 11.4 ms per token) |

## 6. References

| Source | What it gave |
|---|---|
| Huang et al., *GPipe* (2019) | Pipeline parallelism with micro-batches |
| Shoeybi et al., *Megatron-LM* (2019) | Tensor parallelism: two all-reduces per layer, unfit for wide-area links |
| Borzunov et al., *Petals* (2023) | Serving large models over the internet in layer blocks |
| Leviathan, Kalman and Matias, *Fast Inference from Transformers via Speculative Decoding* (2023); Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023) | Speculative decoding |
| Agrawal et al., *Sarathi* (2023) | Chunked prefill |
| Yu et al., *Orca* (2022) | Iteration-level scheduling |
| Butler et al., *PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation* (SC 2024) | Batches of guesses in flight on every stage at once, dropped on a miss |
| Dettmers et al., *LLM.int8()* (2022) | Outlier channels, hence per-group activation scales |
| Su et al., *RoFormer* (2021) | RoPE |
| Ainslie et al., *GQA* (2023) | Grouped-query attention |
| ggml / llama.cpp | The GGUF format and the K-quant layouts |

The planner's throughput split assigns each layer where the stage total stays smallest. For contiguous counts with per-stage costs nᵢ·tᵢ, this minimizes the largest stage time: an exchange argument shows that any other assignment can be improved toward it.

## 7. Limits and next steps

- **CUDA is untested on a card.** The CUDA backend compiles, its kernels are the ones Metal runs, and its PTX is generated and checked for staleness, but no NVIDIA GPU has run it yet; the startup check keeps a broken one on the CPU. AMD and Intel GPUs run on the CPU.
- **GPU products reach about 40% of the M2's bandwidth, and checking several guesses costs about 3× one token there.** The kernels are instruction-bound, not memory-bound. llama.cpp's hand-tuned kernels do better; simdgroup matrix instructions (and tensor cores on NVIDIA) for several rows are the next step. Until then the coordinator simply measures and does not guess where guessing is slower.
- **Pipelined speculation needs separate hardware.** On one computer every stage shares the same cores or GPU, so batches in flight wait for each other; the gain shows only across devices (measured here with simulated stage times).
- **Nothing drifts across devices.** The wire is exact, so any split writes what the whole model writes on one device. The engine's own products quantize each input row to int8 per block of 32, as llama.cpp does; that is part of the model's numerics on every device alike, not something the network adds.
- **Single conversation speed.** One conversation gets faster from a device with a GPU (several times), from more stages working at once with a good draft (pipelined speculation, up to about the number of stages), and from long prompts split across stages. Without a draft, or with one that rarely agrees, it is still fastest on the fewest, fastest devices. Tensor parallelism, the other way to speed one token up, needs a round trip per layer and is not worth it outside one machine (section 2).
- **Speculation is greedy only.** Sampling with a temperature runs one token a pass; speculative sampling (Leviathan et al.'s rejection rule) would extend it.
