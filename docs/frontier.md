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
| 1. Swarm tree speculation | Many devices draft a tree of continuations; the target checks the whole tree in one pass | 3 to 8 (3.9 measured) | Algorithm done; batched GPU kernel next |
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

Built: tree passes on CPU, Metal and CUDA (exact: a tree row scores bit for bit what its
path alone scores), cache keeping, tree drafting, acceptance, swarm drafters on idle
devices (`Meta.grow`, `tree.rs`), and the measured choice between plain, chain and tree.

Measured on one M2, Qwen2.5 1.5B with the 0.5B as draft, 43 tokens, same text every way:

| Way | Passes | Tokens a pass | Tokens a second |
|---|---|---|---|
| Plain | 43 | 1.0 | 10 to 17 |
| Chain | 25 | 1.7 | 6 to 8 |
| Tree | 11 | 3.9 | 2.4 |

Tree confirms 3.9 tokens a pass, but a 41-row pass costs about 25 times a one-row pass
on this GPU (1.5 s against 60 ms), and drafting the tree about 1 s. A batched kernel would
check 41 rows in roughly 40 to 60 ms. So the next steps are:

1. **Batched products on the GPU's matrix units** (Metal `simdgroup_matrix`, CUDA tensor
   cores via `mma`): rows 2 to 64 for little more than one. This is what turns 3.9 tokens
   a pass into 3 to 4 times the speed.
2. **Incremental drafting**: keep earlier tree levels in the draft's cache instead of
   running the whole tree again at each level (cuts drafting about 3 times).
3. Measure the swarm with real devices (drafters on idle computers).

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
