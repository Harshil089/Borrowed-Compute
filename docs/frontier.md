# Frontier: more tokens per byte of weights

Decoding one token reads every weight once, so a single conversation runs at memory
bandwidth: a 4-bit 70B model is 35 GB a token, and Cerebras' 2,000 tokens a second is
70 TB/s of weight reads. Consumer devices have 0.1 to 1 TB/s each, and splitting a model
across them (pipeline parallelism) does not add their bandwidth for one conversation.

The only way past that wall is to get many tokens out of each pass over the weights.
Then decoding becomes compute-bound, and compute *does* add up across devices when each
does a lot of work between network trips. Three steps, built in this order:

| Phase | Idea | Tokens per weight pass | Status |
|---|---|---|---|
| 1. Swarm tree speculation | Many devices draft a tree of continuations; the target checks the whole tree in one pass | 3 to 8 (3.9 measured) | Built; pays with a big target and a small draft |
| 2. Experts across the internet | Mixture-of-experts models: each token touches a few experts; experts live on different devices | (bytes per token ÷ 10 to 20) | Planned |
| 3. Distributed diffusion LMs | Text written a block at a time, refined over a few steps, compute-bound and split across devices | 10 to 50 | Planned |

## Phase 1: swarm tree speculation

Today one draft model on the coordinator proposes one chain of 4 guesses; the target
keeps the agreeing prefix (2 to 4 tokens a pass). A tree checks many chains at once for
nearly the same cost, because decoding is memory-bound: reading the weights once for 64
rows costs little more than for 1. Published results (SpecInfer, Medusa, EAGLE-2) accept
3 to 6 tokens a pass with trees of 30 to 60 nodes.

What is new here: the drafting is spread over the mesh. Idle devices (and lenders with
spare cycles) each run a draft model and grow a different part of the tree, so the tree
is wider than one computer could draft in time, and the pipelined batches from
`llm.rs` keep every stage busy while drafts arrive.

1. **Tree verification in the engine.** A batch carries a tree: each row's parent row.
   Row i sits at position `pos + depth(i)` for RoPE and attends to the cache before
   `pos` plus its own ancestors (a tree attention mask). Its keys and values go to cache
   slot `pos + i`. The next batch names the accepted rows, and every stage moves their
   keys and values to consecutive slots before it continues. CPU first, then Metal and
   CUDA. Exit: a tree pass scores every node exactly as a one-token pass on its own path
   would, on every backend.
2. **Tree drafting and acceptance on the coordinator.** Grow the tree from the draft
   model's top choices (top-k at each depth, kept to a node budget by cumulative
   probability, as in EAGLE-2), check it in one pass, walk the target's greedy choice
   down the tree. Exit: text identical to plain greedy decoding, fewer passes per token.
3. **Swarm drafters.** Devices in the mesh that hold no stage, or have spare time, run a
   draft model behind a new `Draft` request: given the accepted text, return a subtree
   rooted at a given prefix. The coordinator merges subtrees within a time budget and
   sends whatever arrived. Exit: measured tokens per pass rising with drafter count.
4. **Planner and the measured switch.** The pipeline's measured speed (`llm::Speed`)
   already picks plain or speculative decoding; it learns tree width too.

### Where phase 1 stands

Built: tree passes on CPU, Metal and CUDA, cache keeping, tree drafting (a level at a
time, earlier levels kept in the draft's cache), acceptance, swarm drafters on idle
devices (`Meta.grow`, `tree.rs`), the measured choice between plain, chain and tree, and
batched products on Apple's matrix units (`k_matmul_mm`, `simdgroup_matrix`).

Batched products, 8192 x 8192 Q4_0 on an M2, GB/s of weights (higher is faster):

| Rows | 1 | 2 | 8 | 16 | 41 | 64 |
|---|---|---|---|---|---|---|
| `k_matmul` (before) | 38 | | 6.2 | 3.4 | 1.3 | 0.9 |
| `k_matmul_mm` | 35 to 38 (one row stays on `k_matmul`) | 12 | 12 | 11 to 14 | 4.5 to 5.5 | 3.5 to 4.4 |

The matrix units sum in their own order, so a batch row is now within float rounding of
a one-row pass (1e-7), not bit for bit: tree and chain write plain decoding's text
unless two tokens tie within rounding. Half-precision tiles are 30 to 50% faster again
but miss the engine's accuracy check, so they are not used.

Qwen2.5 1.5B with the 0.5B as draft, 43 tokens, one M2, same text every way:

| Way | Tokens a pass | Tokens a second (before the kernel) | Now |
|---|---|---|---|
| Plain | 1.0 | 10 to 17 | 16 to 17 |
| Chain (4 guesses) | 1.7 | 6 to 8 | 7.2 |
| Tree, 40 guesses | 3.9 | 2.4 | 4.8 |
| Tree, 15 guesses | 2.9 | | 7.8 |

**Guesses that cost nothing.** Every tree also takes up to 8 guesses read from the text
itself (prompt lookup: where the last 2 to 4 tokens appeared before, what followed), and
a fourth way of decoding, lookup, checks a tree of those alone with no draft model. The
draft scores only the first 32k tokens of the vocabulary (`DRAFT_VOCAB`), and only its
last row when it catches up. Same model and M2, asking for a small code edit (which
repeats its input):

| Way | Tokens a pass | Tokens a second |
|---|---|---|
| Plain | 1.0 | 15.1 to 15.5 |
| Chain | 2.5 | 10.5 |
| Tree (draft and lookup) | 4.3 | 13.8 |
| Lookup alone | 2.4 | **23.7** |

The first way faster than plain on this computer: half again, with no draft model at
all, for any model. On text that does not repeat (the sky question) lookup finds nothing
and the measured choice keeps plain.

A 15-guess round is now about 360 ms: the check (16 rows) 155 ms, 2.6 one-row passes,
and drafting 200 ms. What still holds it back on this pair, and next:

1. **The draft is a third of the target.** The 0.5B costs a third of the 1.5B a pass
   (its 151k-word output head is a quarter of it). Speculation pays with a big target
   and a small draft: a 7B to 32B target with the 0.5B, where a target pass is 5 to 20
   times a draft pass. Measure that next.
2. **The M2's matrix units are the ceiling**, about 1 TFLOPS achieved in fp32: a 16-row
   check can never be much under 2 one-row passes there. NVIDIA tensor cores have 30 to
   100 times the arithmetic per byte read, so trees of 40 to 60 should pay there; size the
   tree per GPU once measured (CUDA still runs batches on `k_matmul`).
3. **Draft while the target checks.** The draft waits for each check today; growing
   the next tree from the draft's likeliest path while the check runs would hide most
   of the drafting.
4. Measure the swarm with real devices: drafters on idle computers take drafting off
   the coordinator, which is now half of each round.

Until then the measured choice keeps plain decoding where it is faster, so nobody is
slowed down; it tries plain first on a new pipeline.

## Phase 2: experts across the internet

Models such as Qwen3-30B-A3B, gpt-oss and DeepSeek-V3 route each token to a few of
many experts per layer (3B of 30B, or 37B of 671B, active). The engine learns the
mixture-of-experts layer (router, top-k experts, shared expert); experts are placed on
devices by size and how often they are used together; each layer sends a token's hidden
state only to the devices holding its chosen experts and adds their answers. The router
of the next layer is cheap enough to run early, so its experts can be warmed while the
current layer computes. Exit: a model no single device here can hold, running across them.

## Phase 3: distributed diffusion language models

Diffusion LMs (LLaDA, Dream; Inception's Mercury reports ~1,000 tokens a second on one
H100) start from a block of masked tokens and fill it in over a few denoising steps, each
step a full bidirectional pass over the whole block. That is compute-bound: dozens of
tokens per weight read. Split across devices it needs one hidden-state exchange per
stage per step for a whole block, not per token. The engine learns bidirectional
attention, the mask token and the unmasking schedule; the mesh runs denoising steps
pipelined block after block. Exit: a diffusion model writing faster across the mesh than
on its fastest single device.

## How we measure

Every phase reports, on the same prompts: tokens a second for one conversation, tokens
per weight pass, and whether the text matches the plain decoder (phases 1 and 2 must
match exactly; phase 3 is a different kind of model, measured on quality benchmarks).
