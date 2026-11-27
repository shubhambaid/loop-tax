# LoopTax lab notebook

Dated log of what ran, where, and what we learned. Newest entries at the bottom.

## 2026-11-26: Phase 0, recon and setup

**Repo.** Imported `loop-tax.bundle` (commit `1dba1e7`, toy simulation). Added `looptax/` package, `scripts/`,
`configs/`, `docs/ARCHITECTURE.md`, `Makefile`.

**Remote box (goosemini).** `results/env/remote_inventory.json`. Apple M4 Pro (12 CPU cores, 16 GPU cores, Metal 4),
24 GB unified memory, macOS 27.2, about 107 GB free disk (varies with purgeable space). No CUDA. Homebrew llama.cpp 1.0.0
(commit `b1b770d`) predates Inkling support, so we built llama.cpp from source at `07ba74b` with Metal
(`scripts/build_llama.sh`). `uv`, `tmux`, `cmake` present. MLX and PyTorch are not installed for the system Python.

**Network.** Measured on 2026-11-26: about 4.5 MB/s from goosemini to Hugging Face (500 MB range request in 110 s) and
about 4.5 MB/s laptop to goosemini over Tailscale (1 GB in 221 s, direct LAN path). The laptop gets 10-17 MB/s from HF on
short samples, but forwarding to goosemini is capped by the same link, so all downloads go straight to goosemini.
Time cost: about 65 min per 17.6 GB file, 1.7 h for a Q8_0, 3.2 h for a BF16.

**Memory constraint.** 24 GB unified memory cannot hold the BF16 (52 GB) or Q8_0 (27.8 GB) GGUF on the GPU. The
largest variant that fits is about 17-18 GB, and only if the Metal wired-memory limit allows it (default for 24 GB
machines is lower; to be measured at the smoke test).

**Model facts.** See `docs/ARCHITECTURE.md`. Key points: 24 physical blocks, blocks 6-17 run 3 times, every pass
restarts from the prelude output and receives a gated message from the previous pass, each visit has its own KV.
Experts are 93.8% of parameters and were trained with NVFP4 forward passes, so the NVFP4 expert grid is native.

**Critical check: loop representation.** Parsed the tensor tables of 7 GGUF files from 32-48 MB HTTP range requests
(`looptax/layout_check.py`, outputs in `results/layout/`). All ggml-org, unsloth, bartowski and lmstudio-community files
use `inkling2_loop` with 24 physical blocks and loop controller tensors only in blocks 6-17. One community file
(AtomicChat) declares 48 blocks under an older `inkling2` architecture but still stores 24 physical blocks; it is not used.

**Reference choice (proposed, pending Shubham's OK).** ggml-org `Inkling-2-Mini-NVFP4.gguf` (17.6 GB) as the KL
reference: it is the only near-lossless file that fits in memory, and its experts in blocks 0-19 are bit-identical to
the released checkpoint. ggml-org measured it at KL 0.0029 vs BF16, the same as Q8_0 (0.0030).

**Started.** Download of the NVFP4 reference and ggml-org `imatrix.gguf` on goosemini (tmux `dl_ref`).

**Open.** Metal memory limit, eval set, whether to also pull BF16 as the quantization source, varloop reference.

## 2026-11-26: Phase 0 wrap-up and Phase 1, uniform baselines

**Decisions (Shubham, 2026-11-26).** Reference is the ggml-org NVFP4 GGUF. Metal wired limit raised to 21504 MB on
goosemini. Download BF16 too, as a second quantization source. Varloop: convert from the original safetensors.
Eval set as proposed. No GSM8K. ggml-org imatrix for all variants. GitHub repo `shubhambaid/loop-tax`, private for now.

**Reference checks.** `Inkling-2-Mini-NVFP4.gguf` sha256 `d4fcaf47...bd37` matches the Hugging Face LFS etag.
Loads fully on Metal (10.8 GB resident; the rest is mmapped). 259 tok/s prefill (pp512), 29 tok/s decode (tg128).

**Eval set.** `configs/evalset_manifest.json`. Context 2048, llama-perplexity scores the second half of each chunk.
wikitext-2 test: 8 chunks (8192 scored tokens). Code: CPython v3.13.0 sources, 4 chunks; at this length only the first
Python files are reached, so this domain is effectively Python stdlib code. Multilingual: FLORES-200 devtest from the
ungated `mteb/flores` copy (FLORES+ is gated and the token had not accepted its terms), 8 languages in rotating
paragraphs, 4 chunks. Reference logits: 6.5 GB on goosemini (`kld/ref-nvfp4/`).

**Pipeline control.** `identity` variant (re-quantize the reference with every tensor pinned to its own type) gives
KL = 0.000000 and 100% top-1 agreement on all three domains. llama-quantize copies a tensor byte for byte when the
target type equals the current one, so scan variants differ from the reference only where we ask.

**Phase 1 results** (`results/phase1-baselines/summary.csv`, chart `assets/phase1_kl_vs_size.png`). Source = NVFP4 GGUF,
requantized, ggml-org imatrix. Mean KL averaged over the three domains:
Q3_K_M 13.7 GB 0.0233, IQ4_XS 14.4 GB 0.0150, IQ4_NL 14.9 GB 0.0148, Q4_K_M 17.5 GB 0.0117 (0.0129 without imatrix),
Q5_K_M 19.9 GB 0.0061, Q6_K 23.5 GB 0.0037, Q8_0 27.8 GB 0.0021.

**Surprise 1: the down-projection experts cannot use K-quants.** `ffn_down_exps` has 1152 columns (the expert width),
not a multiple of 256, so llama.cpp falls back: Q2_K/Q3_K to Q4_0, Q4_K to Q5_0, Q5_K to Q5_1, Q6_K to Q8_0,
IQ4_XS to IQ4_NL. A third of all weights therefore never goes below about 4.5 bits per weight with stock types, and
the "K" recipes are larger than their names suggest (Q4_K_M is 5.36 bpw and 17.5 GB, same size as the NVFP4 file,
because its "more bits" layers push some down experts to Q6_K, which falls back to Q8_0). IQ4_XS and IQ4_NL end up
nearly identical for the same reason.

**Surprise 2 (tentative): Q4_K_M looks worse than the native NVFP4 file at the same size.** At 17.5 GB, Q4_K_M sits at
KL 0.0117 from the NVFP4 reference, while ggml-org measured the NVFP4 file at KL 0.0029 from BF16 (their number, ctx
512, different text). Comparing those two numbers is not valid: our KL is measured against NVFP4 itself, so NVFP4 scores
0 here by construction. To check it properly, re-measure NVFP4 and Q4_K_M against a BF16 reference on our eval set
once BF16 is downloaded. If it holds, re-quantizing weights trained on an FP4 grid onto a different 4-bit grid costs
real accuracy, and every loop-aware recipe must also be compared against the NVFP4 file.

**Note on llama.cpp's "more bits" heuristic.** Q4_K_M/Q5_K_M upgrade `ffn_down` for layers chosen by physical index
(`i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2`). It does not know that blocks 6-17 run three times.

**Speed (llama-bench, Metal, all layers on GPU).** pp512 / tg128 tok/s: NVFP4 259 / 29.0, IQ4_NL 260 / 30.8,
IQ4_XS 257 / 30.7, Q4_K_M 254 / 30.0, Q3_K_M 245 / 27.0, Q5_K_M 243 / 25.6. Q6_K and Q8_0 do not fit in GPU memory;
they were evaluated with partial CPU offload and not benchmarked.

## 2026-11-26: Design review (Shubham) and changes

**Position confound.** All 12 looped blocks sit in the middle of the network, so a per-block scan cannot tell "loops"
from "middle layers matter more". Fix: the headline experiment now holds position and weights fixed and changes only
the reuse count. On TML-d24-varloop-research, one block is quantized and the model is run with 1-5 loop passes; each
run is compared with the unquantized reference at the same pass count (`configs/phase4_passes.yaml`, `looptax/run_passes.py`).
Perturbed blocks: 6, 9, 12, 17 (loop) at Q2_K, block 9 also at Q4_0. Controls: blocks 3 (prelude) and 20 (coda),
which run once whatever the pass count. If their KL also rises with passes, the rise is not caused by reuse.
Prediction to test: if the error adds up coherently across passes, its amplitude grows like N and KL (quadratic in
small logit changes) like N^2; if it adds up like independent noise, amplitude grows like sqrt(N) and KL like N.

**Varloop reference.** Every routed expert in all 24 blocks at NVFP4 (requantized from our Q8_0 conversion with the
ggml-org varloop imatrix), everything else Q8_0 (`scripts/make_varloop_ref.sh`). So all blocks start from the same
precision and the reference fits in GPU memory. Its own distance from the Q8_0 conversion is measured at 3 passes
(`configs/varloop_refcheck.yaml`) before the Q8_0 is deleted for disk space.

**Blocks start from different precisions on Inkling.** In the Inkling NVFP4 reference, experts of blocks 0-19 are
NVFP4 (about 4.5 bpw) and blocks 20-23 are Q8_0, so "drop block 20 to Q2_K" removes far more information than "drop
block 9" (0.51 GB vs 0.18 GB smaller file), yet scores lower KL. Raw KL per block is not a fair comparison. Fix:
`looptax/weight_error.py` measures, for every scan variant, the bits removed and the relative squared weight error per
tensor family, by dequantizing both files. Sensitivity will be reported as KL per bit removed and per unit of weight
error as well as raw KL. First check, block 9 at Q2_K: attention rel. error 7.7% (0.020 GB removed), gate_up 7.7%
(0.159 GB), down experts (Q4_0 fallback) 0.7% with no bits removed.

**"Q2_K block" is really attention + gate_up.** Because of the 1152-column fallback, the down experts land at Q4_0 in
every scan run, about the same storage as NVFP4. The scan therefore mostly measures attention plus the gate_up
experts. The per-family runs (experts only, attention only) on 8 blocks are in the scan, and the headline experiment
adds a Q4_0 perturbation, a block-32 type that hits all three families with no fallback.

**Sample size.** 16 chunks (about 16k scored tokens) per run is screening only. Built a disjoint held-out slice
(`configs/evalset_holdout_manifest.json`: wikitext from character 300000, five other CPython files, FLORES devtest
sentences 240-1011). Candidates are rerun on it before any claim goes on the site.

## 2026-11-26: Second review (Shubham), headline design fixes

**Block 3 is not a no-reuse control.** Every pass restarts from the prelude output, so an error in a prelude block is
injected into all N passes. Headline blocks are now reported in three categories: loop (6, 9, 12, 17), prelude,
"injected into every pass" (3), and coda (20, 22), the clean control that runs once, after the loop. Added block 22
(Q2_K) and block 20 (Q4_0) so the control category has two blocks and both perturbation types.

**Fit and uncertainty.** `looptax/alpha_fit.py` fits KL(N) = c N^alpha per perturbed block by least squares on
log KL vs log N, with a paired bootstrap over evaluation chunks (same resampled chunks at every N, 5000 resamples,
percentile 95% interval). Per-chunk values come from a one-line logging patch to llama-perplexity
(`scripts/perplexity-chunk-sums.patch`, exact running sums per chunk) built at the pinned commit into a separate
`build-lt/` directory; the KL computation is unchanged and results record which build produced them. Checked on
synthetic data with known exponents: recovers alpha = 2.00 [1.98, 2.02] and -0.01 [-0.03, 0.02].

**Trained pass counts.** Model card (thinkingmachines/TML-d24-varloop-research@91a57fe): midtraining with 1-5 passes
sampled 1:2:6:2:1, post-training with 3 passes only, "may behave unpredictably at pass counts other than 3". Fits use
N = 1-5. N = 6 (llama.cpp allows up to 6) is run for the headline blocks, marked as extrapolation in the result files,
and excluded from the fit.

**Disk.** With Shubham's OK, deleted the Hugging Face hub and datasets caches on goosemini (117 GB of unrelated
models and datasets; token files kept). 154 GB free afterwards.

**BF16 check (priority).** `configs/bf16check.yaml`: against the BF16 GGUF on the same eval set, measure the ggml-org
NVFP4 file, ggml-org IQ4_NL, our Q4_K_M and IQ4_NL made from BF16 and from the NVFP4 file, and Q8_0 from BF16. The
GPU queue (`scripts/gpu_queue.sh`) runs it as soon as the BF16 download has finished, ahead of the headline variants.

## 2026-11-26: Varloop conversion check

`scripts/convert_varloop.sh` converted thinkingmachines/TML-d24-varloop-research@91a57fe to Q8_0 (27,820,098,944 bytes,
sha256 `7735ed54...a3b6`). Compared with ggml-org's own `TML-d24-varloop-research-Q8_0.gguf` (27,820,099,264 bytes) via
the header: all 704 tensors have the same names, types and shapes; every tokenizer array (200,058 tokens, 424,413
merges, token types, special ids, pre-tokenizer `inkling2`) is identical. Metadata differs only in `general.name`,
`general.basename`, default sampling keys and the chat template (320 bytes). The converter's transformers warning
about the tokenizer regex therefore did not change the GGUF tokenizer. The NVFP4R reference built from it:
15,589,508,704 bytes, sha256 `09e8d57f...d891d`. The safetensors were deleted after conversion.

Queue bug, fixed: `scripts/gpu_queue.sh` assigned a shell variable named `TMUX`, which overrode tmux's own socket
variable, so its "wait for the scan" checks failed and at 16:14 it tried to start the BF16 check on a partial
download and the varloop steps before their files existed. Every attempt failed at model load and wrote no result
files; partial logit directories were deleted. The queue now also requires the BF16 file to match the HF size and
sha256 before use.

## 2026-11-26: Phase 2 scan, screening results

All 48 single-block runs (Q2_K and Q3_K) plus per-family runs on blocks 2, 5, 8, 11, 14, 17, 20, 23 finished at 19:35
(`results/phase2-scan/`, weight error in `results/phase2-scan-weight-error/`, chart `assets/phase2_scan_normalised.png`).
Screening only: 16 chunks per run.

**Same-sized shock everywhere.** The relative squared weight error of the perturbed block is nearly the same for every
block: 0.042-0.043 at Q2_K and 0.0136-0.0147 at Q3_K, including blocks 20-23 that start from Q8_0. Their bits removed
differ a lot (0.18 GB vs 0.69 GB at Q2_K), so per-bit normalisation and per-weight-error normalisation disagree for
those blocks; weight error is the fairer yardstick for sensitivity, bits removed is what matters for recipes.

**KL per unit of weight error.** Mean over blocks, loop (6-17) vs the 12 run-once blocks: Q2_K 0.222 vs 0.122 (1.82x),
Q3_K 0.344 vs 0.210 (1.64x). But the run-once group mixes two very different populations. Prelude blocks 0-5 sit at
0.12-0.15 (Q2_K) and 0.30-0.34 (Q3_K); coda blocks 19-22 at 0.08-0.10 at both widths. At Q3_K, loop blocks 13-15
(0.25-0.27) are below the prelude. So "loop vs not loop" is mostly "loop and prelude vs coda", which is exactly what the
position confound predicts, and the prelude also feeds every pass. Only the headline experiment can separate these.

**Where the loop excess sits.** Peaks at blocks 11, 12, 16, 17 at both widths. 11 and 17 are global-attention layers
(the full-attention layer of each six-block group is 5, 11, 17, 23). Per-family runs at Q2_K: attention-only KL is
0.0051-0.0054 in prelude blocks 2 and 5, 0.0060-0.0063 in loop blocks 8 and 14, but 0.0118 and 0.0104 in loop blocks
11 and 17; coda 20 and 23: 0.0014 and 0.0035. Experts-only KL is similar in prelude and loop blocks (0.0036-0.0048) and
lower in the coda (0.0023-0.0025). Attention and experts runs remove very different numbers of bits (0.16 vs 1.27 GB),
so compare within a family, not across.

**Next.** Extended the per-family runs (experts only, attention only) to all 24 blocks (32 more runs, about 2 h, queued
after the headline): the Phase 3 sensitivity-guided recipe varies only gate/up experts, so it should be guided by
experts-only sensitivity, not by whole-block KL that is partly attention.

## 2026-11-26 night: BF16 check failure, early headline numbers

**BF16 reference failed silently, fixed.** With llama.cpp's automatic placement (`--fit`), and with `-ngl 4`, the BF16
GGUF (52 GB) ran Metal out of memory ("Insufficient Memory", "failed to decode") and llama-perplexity still exited 0.
The runner recorded six empty results. Fixes: the runner now rejects any run whose output lacks the expected metrics
(`require()` in `looptax/run.py`), skips an existing valid reference, and sizes its disk guard from the weight count.
The empty BF16 results and stub logit files were deleted, and all earlier results were re-checked (every Phase 1,
Phase 2, refcheck and Phase 4 file has its metrics; Phase 4 files have per-chunk sums). The BF16 reference now runs
CPU only (`--device none`): one 2048-token chunk takes about 9.5 minutes (PPL 5.0328 on the first wikitext chunk vs
5.0385 for the NVFP4 file), so 16 chunks take about 2.5 hours. Partial offload (`-ngl 8 --no-op-offload`) was slower.
The overnight queue runs the headline and grid first (GPU), then the BF16 check, then the Phase 2 extension.

**Varloop reference distance.** NVFP4R vs our Q8_0 conversion at 3 passes: KL 0.0116 (wikitext), 0.0105 (code), top-1
agreement 95.4% / 97.3%. About as far as a Q4_K_M. Every headline variant is compared with NVFP4R itself, so this does
not enter the headline numbers, but it is a limitation of "full precision" in the headline wording.

**Reference perplexity vs passes (NVFP4R, wikitext, 8 chunks):** 1: 6.018, 2: 5.786, 3: 5.606, 4: 5.597, 5: 5.592,
6: 5.606 (extrapolated). Gains stop after 3 passes.

**Early headline (screening, 4 of 10 blocks).** KL vs the reference at the same N, fit on N = 1-5:
block 6 (loop) alpha -0.06 [-0.13, 0.01]; block 9 (loop) 0.19 [0.12, 0.26]; block 3 (prelude) 0.00 [-0.08, 0.10];
block 20 (coda) 0.04 [-0.03, 0.12]. KL hardly changes with the number of passes, for loop blocks as for the controls;
all blocks show a bump at N = 2. Block 9 is already more sensitive than blocks 3 and 20 at N = 1, where it runs once.
Checked that the pass override reaches the quantized runs: their perplexity tracks the reference at each N
(block 9: 6.04 at N = 1, 5.62 at N = 3). If this holds, the loop tax hypothesis (error compounding with reuse) is
not supported for this model; the extra sensitivity of middle blocks comes from position, not reuse. Not a finding
until the remaining blocks are in and the held-out rerun (`configs/phase4_holdout.yaml`: blocks 9, 17, 20, 3 at
N = 1, 3, 5, 24 chunks of disjoint text) agrees.

## 2026-11-27 morning: overnight results (headline, grid, BF16 check, Phase 2 extension)

The overnight queue finished at 06:11. Every result file was re-validated (metrics present, per-chunk sums present for
Phase 4). Charts: `assets/headline_passes.png`, `assets/grid_bits_passes.png`, `assets/bf16_check.png`.

**Headline (screening): quantization error of a reused block does not compound with passes.** One block perturbed,
TML-d24-varloop-research run at N = 1-6 passes, KL vs the NVFP4R reference at the same N, wikitext 8 + code 4 chunks,
fit KL = c N^alpha on N = 1-5 (`results/phase4-passes/alpha_fit.json`, 5000 paired bootstrap resamples over chunks).
N = 6 was never trained and is shown but not fitted.

| block | category | type | alpha N=1-5 [95% CI] | alpha N=2-5 [95% CI] | KL N=1 | KL N=3 | KL N=5 | KL N=6* |
|---|---|---|---|---|---|---|---|---|
| 3 | prelude | Q2_K | 0.00 [-0.08, 0.10] | -0.21 [-0.32, -0.10] | 0.00594 | 0.00662 | 0.00612 | 0.00623 |
| 6 | loop | Q2_K | -0.06 [-0.13, 0.01] | -0.28 [-0.40, -0.16] | 0.00850 | 0.00873 | 0.00813 | 0.00827 |
| 9 | loop | Q2_K | 0.19 [0.12, 0.26] | 0.04 [-0.08, 0.15] | 0.00835 | 0.01113 | 0.01165 | 0.01207 |
| 12 | loop | Q2_K | 0.07 [-0.02, 0.14] | -0.12 [-0.23, -0.00] | 0.01185 | 0.01407 | 0.01374 | 0.01430 |
| 17 | loop | Q2_K | 0.28 [0.22, 0.33] | 0.14 [0.07, 0.21] | 0.00788 | 0.01149 | 0.01253 | 0.01326 |
| 20 | coda | Q2_K | 0.04 [-0.03, 0.12] | -0.04 [-0.10, 0.02] | 0.00594 | 0.00639 | 0.00635 | 0.00644 |
| 22 | coda | Q2_K | -0.05 [-0.10, 0.02] | -0.13 [-0.18, -0.08] | 0.00534 | 0.00520 | 0.00505 | 0.00509 |
| 3 | prelude | Q4_0 | 0.29 [0.19, 0.38] | 0.03 [-0.06, 0.10] | 0.00240 | 0.00375 | 0.00392 | 0.00436 |
| 9 | loop | Q4_0 | 0.47 [0.38, 0.54] | 0.22 [0.13, 0.30] | 0.00196 | 0.00367 | 0.00434 | 0.00489 |
| 20 | coda | Q4_0 | 0.01 [-0.04, 0.05] | -0.09 [-0.18, -0.01] | 0.00091 | 0.00094 | 0.00093 | 0.00094 |

Reading: coherent compounding predicts alpha near 2, independent addition near 1. All loop blocks are far below 1:
-0.06 to 0.28 at Q2_K, 0.47 for block 9 at Q4_0. Coda controls are flat (-0.05 to 0.04). Every curve, controls
included, has a step from N = 1 to N = 2; pass 1 runs the loop with no cross-pass message and no controllers, so that
step is a change of computation, not just of reuse count. Fitting N = 2-5 only: loop blocks -0.28 to 0.22, prelude
-0.21 to 0.03, coda -0.13 to -0.04. Relative to the controls, loop blocks 9 and 17 grow slightly (about +0.2 to +0.3 in
alpha), blocks 6 and 12 not at all. The extra sensitivity of loop blocks is mostly present already at N = 1, when the
block runs once (block 12 at Q2_K: 0.0119 vs 0.0053-0.0059 for coda blocks): it comes with the position, not with
reuse. The prelude, whose error is injected into every pass, behaves like the loop blocks at Q4_0 (alpha 0.29) and
like the coda at Q2_K (0.00). Pending: held-out confirmation (`configs/phase4_holdout.yaml`, running).

**Bits x passes grid** (gate/up experts at the stated type in all blocks, down experts NVFP4, attention Q8_0; KL vs
the reference at N = 3, i.e. distance from the model as shipped; decode tok/s from llama-completion, tg128):
NVFP4R 1p 0.0799 (61.0 tok/s), 2p 0.0355 (38.5), 4p 0.0109 (22.4), 5p 0.0165 (18.5); IQ4_XS 3p 0.0074 (27.7);
Q3_K 2p 0.0481 (35.6), 3p 0.0166 (26.2); Q2_K 3p 0.0500 (28.8). Answer to "Q3 with 3 passes or more bits with 2?":
Q3_K at 3 passes (0.0166) is closer to the shipped model than full-precision NVFP4R at 2 passes (0.0355). More than 3
passes moves the model away from its 3-pass behaviour (perplexity flat: 5.606, 5.597, 5.592). Quantization cost at a
given N (KL vs the reference at the same N) is flat from N = 2 to 5 for every bit width (IQ4_XS 0.0072-0.0074,
Q3_K 0.0158-0.0166, Q2_K 0.0495-0.0513), with a lower value at N = 1. Quantizing all blocks shows the same picture as
one block: no compounding.

**BF16 check (CPU-only BF16 reference, 3 domains, 16 chunks; 2.8 h).** Mean KL vs BF16:
ggml-org NVFP4 17.6 GB 0.00238; Q8_0 from BF16 27.8 GB 0.00261; ggml-org IQ4_NL 17.6 GB 0.00692;
Q4_K_M 17.5 GB 0.01180 (from BF16) / 0.01166 (from NVFP4); IQ4_NL 14.9 GB 0.01474 / 0.01475. So the tentative Phase 1
observation holds against BF16: at the same size, the NVFP4 file is about 5x closer to the released weights than
Q4_K_M, and even closer than Q8_0. The quantization source (BF16 vs the NVFP4 file) makes no measurable difference, so
the NVFP4-sourced Phases 1-3 stand. The NVFP4 file vs BF16 (0.0024) also bounds how much the NVFP4 reference itself
differs from the release.

**Phase 2 extension** (experts only and attention only, all 24 blocks, Q2_K). Mean KL x 1e-3, prelude / loop / coda:
experts 4.22 / 4.19 / 2.27; attention 5.02 / 7.74 / 2.38 (peaks 11.8 at block 11, 10.4 at 17, 10.2 at 12, 8.6 at 16).
Expert sensitivity does not distinguish loop from prelude; attention does, mostly in two global-attention layers and
their neighbours. Weight-error normalisation for these 32 runs is running.

**Community.** ggml-org IQ4_NL (experts of blocks 0-19 IQ4_NL, rest Q8_0, 17.6 GB): KL 0.0068 vs our NVFP4 reference;
unsloth UD-Q4_K_M (keeps the NVFP4 experts, other tensors lower): 16.6 GB, KL 0.0020 vs our NVFP4 reference, 257 / 28
tok/s. It is 1 GB smaller than the NVFP4 file at very small cost: the strongest same-size baseline so far, and the
one any Phase 3 recipe has to beat.

**Phase 3 queued** after the held-out run (`configs/phase3_recipes.yaml`, 19 recipes): uniform gate/up types, loop-hi
and its mirror edge-hi at matched sizes, and an experts-guided greedy recipe ranked by the experts-only scan (it
upgrades prelude and early-loop blocks and 16-17 first, coda last).

## 2026-11-27: Held-out confirmation of the headline

`configs/phase4_holdout.yaml`: blocks 9, 17 (loop), 3 (prelude), 20 (coda) at Q2_K, N = 1, 3, 5, on the disjoint
held-out slice (wikitext 16 chunks from character 300000, code 8 chunks from five other CPython files; 24 chunks,
twice the screening). Fit KL = c N^alpha on N = 1, 3, 5 (`results/phase4-holdout/alpha_fit.json`, 5000 paired
bootstrap resamples over chunks). Screening alpha refit on the same three N for comparison.

| block | category | screening alpha (N = 1, 3, 5) | held-out alpha [95% CI] | held-out KL N = 1 / 3 / 5 |
|---|---|---|---|---|
| 9 | loop | 0.22 | 0.23 [0.19, 0.28] | 0.00761 / 0.01038 / 0.01090 |
| 17 | loop | 0.30 | 0.30 [0.27, 0.34] | 0.00759 / 0.01092 / 0.01228 |
| 3 | prelude | 0.03 | 0.10 [0.04, 0.16] | 0.00591 / 0.00714 / 0.00678 |
| 20 | coda | 0.04 | -0.04 [-0.07, -0.01] | 0.00509 / 0.00475 / 0.00478 |

The held-out run reproduces the screening exponents for the loop blocks almost exactly and keeps the coda control
flat. **Finding:** quantization error in a reused block does grow with the number of passes, but weakly and
sublinearly (alpha about 0.2-0.3 for the blocks that grow at all; 6 and 12 do not), far from the alpha = 1 of
independent addition or the alpha = 2 of coherent compounding that the toy simulation suggested. Most of the extra
sensitivity of middle blocks is already present at N = 1, when the block runs once. For this model the "loop tax" is
a small effect on top of a position effect, not a compounding one.

Housekeeping: the holdout session was meant to delete the BF16 GGUF once the BF16 check was complete, but macOS
`wc -l` pads its output with spaces and the integer test failed, so the file stayed; deleted by hand at 06:23 after
checking the 8 result files (108 GB free afterwards).

## 2026-11-27: Results review (Shubham) and changes

1. **Reframe around the negative headline.** README and site now lead with "reuse does not compound quantization error"
   using the N = 2-5 fit (screening: loop blocks -0.28 to 0.22, coda controls -0.13 to -0.04), the N = 1 position
   effect, and why the toy intuition fails (each pass restarts from the prelude output; only a gated message carries
   over). The toy section is retitled "the intuition we tested". The held-out confirmation run covered N = 1, 3, 5 only,
   so the N = 2-5 statistic is not yet held-out confirmed: the held-out run is extended to N = 2 and 4
   (`configs/phase4_holdout.yaml`), and the site says "confirmed" automatically only once the held-out N = 2-5 fit exists.
2. **BF16 framing.** The BF16 GGUF is the released NVFP4/MXFP8 values upcast; the NVFP4 GGUF is the near-lossless
   representation of the release, and any K-quant re-grids FP4-trained weights. Worded as a statement about FP4-trained
   weights, not "NVFP4 beats Q8_0".
3. **Phase 3 rescoped.** (a) >= 16 GB frontier (`configs/phase3_frontier.yaml`, `looptax/make_frontier_config.py`):
   native NVFP4 experts kept, coda experts and attention tuned, BF16 attention in the most sensitive blocks (11, 12, 16,
   17, then 6, 10 from the attention-only scan). Attention above Q8_0 needs BF16 values, which the NVFP4 GGUF does not
   have, so the source is a hybrid GGUF (`looptax/make_hybrid.py`): NVFP4 tensors and their scales copied bit for bit
   from the ggml-org file, everything else BF16 from our conversion of the release safetensors. `unsloth-like` rebuilds
   unsloth's layout from that source as a control (estimated 16.599 GB vs 16.611 GB for unsloth's file). Target: beat
   unsloth UD-Q4_K_M. (b) < 16 GB frontier (`configs/phase3_sub16.yaml`): uniform gate/up types, experts-guided
   recipes at the uniform sizes, and one matched-size loop-hi vs edge-hi pair (Q2_K, 13.67 GB) as the direct hypothesis
   test; the other loop-hi variants are dropped. The first Phase 3 run was stopped before any recipe finished.
4. **BF16 for final numbers.** Every Phase 3 recipe is also measured against the stored BF16 reference logits
   (`extra_refs` in the runner, results under `domains_vs.ref-bf16`). Unsloth's file is re-downloaded to measure it
   against BF16 too (`configs/community_vs_bf16.yaml`).

Disk: deleted the Phase 4 screening reference logits (29 GB); their per-chunk sums are in the result files.

## 2026-11-27: Held-out N = 2-5 confirmation

Held-out run extended to N = 2 and 4 (`results/phase4-holdout/`, 24 chunks of disjoint text, 5000 paired bootstrap
resamples). alpha over N = 2-5, held-out vs screening:

| block | category | held-out alpha N = 2-5 [95% CI] | screening alpha N = 2-5 | held-out alpha N = 1-5 [95% CI] |
|---|---|---|---|---|
| 9 | loop | 0.06 [0.00, 0.11] | 0.04 | 0.21 [0.16, 0.25] |
| 17 | loop | 0.19 [0.15, 0.24] | 0.14 | 0.29 [0.25, 0.33] |
| 3 | prelude | -0.12 [-0.20, -0.04] | -0.21 | 0.07 [0.02, 0.13] |
| 20 | coda | -0.09 [-0.14, -0.05] | -0.04 | -0.05 [-0.08, -0.02] |

The negative headline holds on held-out text: once the cross-pass message is active (N >= 2), the error of a reused
block grows at most like N^0.19 (block 17), N^0.06 for block 9, against N^-0.09 for the coda control. The held-out
logits were deleted after the run (29 + 20 GB); the per-chunk sums are in the result files.

## 2026-11-27: Phase 3, under 16 GB

`configs/phase3_sub16.yaml`, source = NVFP4 GGUF, ggml-org imatrix, down experts NVFP4, attention Q8_0; only the gate/up
experts vary. Every recipe measured against the NVFP4 file and against the BF16 logits (`domains_vs.ref-bf16`).
Chart: `assets/phase3_pareto.png` (KL vs BF16, other GGUFs from the BF16 check for context). Screening, 16 chunks.

| recipe | family | size (GB) | KL vs NVFP4 file | KL vs BF16 [std err] | top-1 vs BF16 (wikitext) | decode tok/s |
|---|---|---|---|---|---|---|
| uniform-q2_k | uniform | 11.77 | 0.0406 | 0.0406 [0.0008] | 91.09 | 29.5 |
| guided-12.6gb | guided | 12.56 | 0.0323 | 0.0323 [0.0007] | 92.11 | 28.7 |
| uniform-iq3_xxs | uniform | 12.66 | 0.0192 | 0.0191 [0.0004] | 94.15 | 27.5 |
| guided-13.4gb | guided | 13.36 | 0.0280 | 0.0279 [0.0006] | 92.96 | 28.6 |
| uniform-q3_k | uniform | 13.42 | 0.0134 | 0.0135 [0.0003] | 95.17 | 26.2 |
| edge-hi-loop-q2_k | edge-hi | 13.68 | 0.0248 | 0.0249 [0.0005] | 93.05 | 28.8 |
| guided-13.7gb | guided | 13.68 | 0.0247 | 0.0245 [0.0006] | 93.65 | 28.8 |
| loop-hi-edges-q2_k | loop-hi | 13.68 | 0.0215 | 0.0215 [0.0005] | 94.07 | 28.7 |
| guided-15.1gb | guided | 14.95 | 0.0094 | 0.0115 [0.0003] | 95.91 | 28.3 |
| uniform-iq4_xs | uniform | 15.08 | 0.0065 | 0.0067 [0.0001] | 96.75 | 27.7 |
| uniform-nvfp4 | uniform | 15.59 | 0.0019 | 0.0041 [0.0001] | 97.75 | 28.2 |
| uniform-q4_k | uniform | 15.59 | 0.0064 | 0.0064 [0.0001] | 96.69 | 28.2 |

**Uniform beats mixing below 15 GB.** Every recipe that leaves some blocks at Q2_K and gives others NVFP4 is beaten by a
uniform type of the same or smaller size: uniform IQ3_XXS (12.66 GB, 0.0191 vs BF16) beats all three mixed recipes at
13.68 GB (0.0215-0.0249) and the guided 13.36 GB one (0.0279). Q2_K is lossy enough here that spreading bits evenly wins.

**The direct hypothesis test.** At matched size (13.68 GB), loop blocks at NVFP4 with prelude and coda at Q2_K score
0.0215 [0.0005] vs BF16; the mirror image, 0.0249 [0.0005]. Spending the bits on the loop is better by about 14%,
a difference of about five standard errors. This is consistent with the headline: the loop holds the middle of the
network, which is the sensitive part with or without reuse; it does not need a reuse effect to explain it.

**Experts-guided recipes lose.** The greedy recipe ranked blocks by their single-block experts-only KL, which rates
prelude and loop experts alike, so it upgraded prelude blocks early. At 13.68 GB it lands at 0.0245, worse than loop-hi
(0.0215), and at every size it is worse than uniform. Single-block sensitivities do not add up to the joint effect of
quantizing many blocks at once; a guide for multi-block recipes needs to be measured on multi-block perturbations.

**15.6 GB.** All experts NVFP4 (blocks 20-23 requantized from Q8_0) is the best recipe under 16 GB: 0.0041 vs BF16,
against 0.0064 for Q4_K experts of the same size and 0.0067 for IQ4_XS at 15.08 GB. Against BF16 it is worse than
against the NVFP4 file (0.0019) because blocks 20-23 were released at MXFP8 and lose precision at NVFP4.
