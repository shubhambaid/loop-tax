# Inkling 2 Mini: what is looped, and how it is stored

Sources (read 2026-11-26): the model card at `thinkingmachines/Inkling-2-Mini` (revision `c9d0283`), its `config.json`,
the Hugging Face blog post (huggingface.co/blog/inkling2mini), the Thinking Machines post (thinkingmachines.ai/news/inkling-2-mini),
the `TML-d24-varloop-research` model card, the ggml-org GGUF model cards, and the llama.cpp implementation
`src/models/inkling2-loop.cpp` at commit `07ba74b`. Machine-checked facts about the GGUF files are in `results/layout/*.json`
(produced by `python -m looptax.layout_check`).

## Layer layout

| Physical blocks | Role | Runs per token | Layer visits |
|---|---|---|---|
| 0-5 | prelude | 1 | 6 |
| 6-17 | loop core (shared weights) | 3 | 36 |
| 18-23 | coda | 1 | 6 |
| **24 stored** | | | **48** |

- Hidden size 2304, 24 query heads, 8 KV heads, head dim 128. Each group of six layers has five sliding-window layers
  (window 1024) and one global layer (blocks 5, 11, 17, 23).
- Every block has an MoE FFN: 128 routed experts, 8 active, expert width 1152, no shared expert, router scale 8.
- Parameters (from the GGUF tensor table): 26.07B in the text model, of which routed experts are 24.46B (93.8%),
  attention 0.65B (2.5%), token embedding and output head 0.46B each, loop controller tensors 0.028B.
  The loop core (blocks 6-17) holds 12.59B parameters and the prelude plus coda hold 12.56B, so the looped half of the
  weights is used for 36 of the 48 visits.

## Loop mechanics (llama.cpp `inkling2-loop.cpp`, matches the TML and HF descriptions)

1. Blocks 0-5 run once. Their output is the **prelude output**.
2. **Every pass starts from the prelude output** (shared input). Pass 1 runs blocks 6-17 as a plain stack.
3. Passes 2 and 3 run blocks 6-17 again with the same weights, plus a **message** from the previous pass: each core block
   mixes the previous pass's residual-block outputs with its own learned query (`loop_msg_query`, attention-residual style).
   A rank-64 **controller** per block (`loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`) looks at the current input and the
   message and decides how much of the input to forget and how much of the message to add, separately for the query, key,
   value and MoE inputs.
4. Only the last pass's output continues into blocks 18-23.
5. **Each visit keeps its own KV cache and short-convolution state** (llama.cpp allocates 24 + 2 x 12 = 48 cache slots
   that share the 24 physical weight sets).

Consequence for this study: the compounding path is not a simple `h -> f(h) -> f(f(h))` iteration. A weight error in
block k reaches the output once through pass 3 directly, and again through the messages that passes 1 and 2 hand on.
The toy simulation (Part 0) models the simpler iteration, so its numbers are only an intuition pump.

## Source precision (important for interpreting every result)

- Thinking Machines trained the routed experts with **NVFP4 forward passes from pretraining through RL**; the last four
  layers' experts ran in **MXFP8**. The released checkpoint (19.5 GB safetensors) stores experts of blocks 0-19 in NVFP4 and
  blocks 20-23 in MXFP8. Sources disagree on whether the non-expert weights are BF16.
- So the 4-bit expert grid is native to the model, not a lossy compression of it. The ggml-org `BF16` GGUF is those
  NVFP4/MXFP8 values upcast, and the ggml-org `NVFP4` GGUF keeps the original NVFP4 experts bit for bit (with their
  per-expert scales) and stores all other matrices at Q8_0. ggml-org report KL vs BF16 on wikitext-2 (40 x 512 tokens):
  Q8_0 0.0030, NVFP4 0.0029, IQ4_NL 0.0095 (these are their numbers, not ours).
- Quantizing an expert to Q4_K or IQ4_NL therefore re-projects values that already sit on an FP4 grid onto a different
  4-bit grid. Below 4 bits (Q3_K, Q2_K) we are genuinely removing information.

## How the GGUF represents the loop (the critical check)

Checked from the tensor tables of the actual files (`results/layout/`):

| File | arch | block_count | physical blocks | loop ctrl tensors in | duplicated per visit |
|---|---|---|---|---|---|
| ggml-org NVFP4 | `inkling2_loop` | 24 | 24 | blocks 6-17 | no |
| ggml-org BF16 | `inkling2_loop` | 24 | 24 | blocks 6-17 | no |
| ggml-org varloop Q8_0 | `inkling2_loop` (`loop.varying = true`) | 24 | 24 | blocks 6-17 | no |
| unsloth UD-Q4_K_M | `inkling2_loop` | 24 | 24 | blocks 6-17 | no |
| bartowski Q4_K_M | `inkling2_loop` | 24 | 24 | blocks 6-17 | no |
| lmstudio-community Q4_K_M | `inkling2_loop` | 24 | 24 | blocks 6-17 | no |
| AtomicChat Q4_K_M | `inkling2` (pre-merge naming) | **48** | 24 | blocks 6-17 | no |

The "48 blocks" files declare `block_count = 48` (the number of visits) under an older `inkling2` architecture name with
different loop tensor names, but still store each weight once. They do not load in mainline llama.cpp at our pinned
commit, and we do not use them. All our work uses `inkling2_loop` files where quantizing `blk.6` through `blk.17`
quantizes the one shared copy that all three passes use.

## Quantization rules in llama.cpp for this architecture

`src/llama-quant.cpp` never quantizes: short-conv kernels, clip scales, residual-mix queries, `attn_rel_bias` and all
`.loop_` controller tensors. They stay F32 in every variant, so "loop tensors" in our recipes means the shared block
weights (attention and experts of blocks 6-17), not the controllers.
