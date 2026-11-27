"""Stage results/ as a Hugging Face dataset (shubhambaid/looptax-results).

Copies every file under results/ with the same privacy scrub the site applies (the remote box's hostname becomes
`remote-m4pro`, absolute home-directory paths become relative to the work root), and writes a dataset card whose tables
are generated from the results files themselves. The layout mirrors results/, so a download into results/ of a clone
of the repo is enough to re-run the analysis scripts.

    python -m looptax.hf_dataset --out /tmp/looptax-results
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATASET = "shubhambaid/looptax-results"
GITHUB = "https://github.com/shubhambaid/loop-tax"
SITE = "https://looptax.alethic-labs.com"
FORBIDDEN = ("/Users/", "/home/", "goosemini", "—")


def scrubber():
    host = json.loads((RESULTS / "env" / "remote_inventory.json").read_text()).get("hostname")

    def scrub(text: str) -> str:
        t = re.sub(r"/Users/[^/\"'\s]+/looptax/", "", text)
        t = re.sub(r"/(?:Users|home)/[^/\"'\s]+", "~", t)
        return t.replace(host, "remote-m4pro") if host else t
    return scrub


def rows(exp: str) -> dict[str, dict]:
    with open(RESULTS / exp / "summary.csv") as f:
        return {r["variant"]: r for r in csv.DictReader(f)}


def fmt(x, nd=4) -> str:
    return f"{float(x):.{nd}f}"


def card(experiments: list[str], n_runs: int, n_chunked: int, commit: str) -> str:
    hold = json.loads((RESULTS / "phase4-holdout" / "alpha_fit.json").read_text())
    alpha_tab = "\n".join(
        f"| {r['block']} | {r['category']} | {r['qtype']} | {r['alpha_2to5']:.2f} "
        f"[{r['alpha_2to5_ci95'][0]:.2f}, {r['alpha_2to5_ci95'][1]:.2f}] | {r['chunks']} |"
        for r in sorted(hold, key=lambda r: (r["qtype"], r["block"])))
    bf, fd, fc = rows("bf16check"), rows("floor-device"), rows("floor-cpu-vs-bf16")
    bf_tab = "\n".join(f"| `{v}` | {fmt(r['file_gb'], 2)} | {fmt(r['kld_mean_avg'])} |"
                       for v, r in sorted(bf.items(), key=lambda kv: float(kv[1]["kld_mean_avg"] or 0))
                       if r["role"] == "variant")
    floor = fd["nvfp4-file-gpu"]["kld_mean_avg"]
    p3 = {**rows("phase3-sub16"), **rows("phase3-frontier"), **rows("community-vs-bf16")}
    p3_tab = "\n".join(f"| `{v}` | {fmt(r['file_gb'], 2)} | {fmt(r['ref-bf16_kld_mean_avg'])} |"
                       for v, r in sorted(p3.items(), key=lambda kv: float(kv[1]["file_gb"])))
    configs = "\n".join(f"  - config_name: {e}\n    data_files: {e}/summary.csv" + ("\n    default: true" if e == "phase4-holdout" else "")
                        for e in experiments)
    return f"""---
license: apache-2.0
pretty_name: LoopTax results
tags:
  - benchmark-results
  - quantization
  - llama.cpp
  - gguf
  - looped-transformer
  - kl-divergence
size_categories:
  - n<1K
configs:
{configs}
---

# LoopTax: raw results

Every number behind [LoopTax]({GITHUB}) ([write-up]({SITE})), a study of whether quantization error compounds in a
looped transformer. The model is Inkling 2 Mini from Thinking Machines Lab: 24 stored decoder layers, blocks 6-17
looped three times, 48 layer visits per token. Model weights in the study are temporary by design (quantized,
evaluated, recorded, deleted); these files are what survives.

**Short answer: reuse does not compound the error.** Quantizing one block and running the loop 2 to 5 times, KL
against the full-precision model at the same pass count scales like N^alpha with alpha near zero, not 1 to 2.

## Held-out confirmation

From `phase4-holdout/alpha_fit.json`: KL = c N^alpha fitted over N = 2-5 passes on the variable-loop sibling
checkpoint (TML-d24-varloop-research), text disjoint from the screening set, 95% paired bootstrap CI over chunks
(5000 resamples). Prelude blocks run once but feed every pass; coda blocks run once after the loop (the control).

| block | category | type | alpha (N=2-5) [95% CI] | chunks |
|---|---|---|---|---|
{alpha_tab}

## Measurement floor

The same NVFP4 file run on CPU (reference) and on GPU differs by mean KL {fmt(floor)} (`floor-device`). The BF16
reference logits were computed on CPU and every quant was run on GPU, so any KL against BF16 below roughly that value
is not distinguishable from the release. The NVFP4 file run on CPU against the BF16 logits gives
{fmt(fc['nvfp4-file-cpu']['kld_mean_avg'])} (`floor-cpu-vs-bf16`). Comparisons where reference and quant both run on
GPU (every KL against the NVFP4 or varloop reference, Phases 1-4) have no such floor: an identity control gives KL
exactly 0.

### Against the BF16 release (`bf16check`)

| variant | GB | mean KL vs BF16 |
|---|---|---|
{bf_tab}

### Phase 3 recipes against BF16 (`phase3-sub16`, `phase3-frontier`, `community-vs-bf16`)

| variant | GB | mean KL vs BF16 |
|---|---|---|
{p3_tab}

Under 16 GB the differences are well above the floor. At 16 GB and up, unsloth's UD-Q4_K_M, our rebuild of its
layout and every recipe except the smallest (`s6-bf16-rest-q6k`) are within it, so none can be shown to beat the
others. No recipe here is offered as a better quant.

## Layout

One directory per experiment, mirroring `results/` in the repo ({n_runs} run records):

| path | what it holds |
|---|---|
| `<experiment>/<variant>.json` | one run: spec, quantize command, per-domain KL / PPL / top-token agreement, file size, speed, provenance |
| `<experiment>/summary.csv` | one row per run (also browsable in the dataset viewer, one config per experiment) |
| `phase4-*/alpha_fit.json` | the N^alpha fits with bootstrap CIs |
| `phase2-scan-weight-error/` | relative weight error and bits removed per quantized tensor family |
| `layout/` | tensor layout and types of the GGUF files used |
| `env/remote_inventory.json` | hardware and software of the compute box (hostname removed) |
| `manifest.json` | index of every result file with commit, machine and timestamp |

Experiments: {", ".join(f"`{e}`" for e in experiments)}.

### Per-chunk sums

{n_chunked} run records (Phase 4, the BF16 check, the floor and reference checks) were made with a patched
`llama-perplexity` and carry `domains.<domain>.chunk_sums`: one row per evaluation chunk with cumulative
`[chunk, sum_kld, sum_kld2, sum_nll, sum_nll_base, count, n_same_top]`. Differencing consecutive rows gives per-chunk
means, which is what the paired bootstraps resample. Other runs carry per-domain means and standard errors only.

## Re-running the statistics

No GPU needed; the fits only read these files.

```bash
git clone {GITHUB} && cd loop-tax
pip install -r requirements.txt
hf download {DATASET} --repo-type dataset --local-dir results
python -m looptax.alpha_fit --exp phase4-holdout   # held-out fits (rewrites results/phase4-holdout/alpha_fit.json)
python -m looptax.alpha_fit                        # screening fits, phase4-passes
python -m looptax.summarize                        # rebuild summary.csv and manifest.json
```

## Setup and caveats

- llama.cpp at commit 07ba74babd586c6b5f558d196e3cf7cc1eea26f4 (Metal), plus a small patch that prints per-chunk sums.
- Single machine: Apple M4 Pro, 24 GB. Context 2048, second half of each chunk scored, so one chunk is 1023 tokens.
- Domains: wikitext, code, multilingual. Screening and held-out slices are disjoint (manifests in `configs/` in the repo).
- Mean KL in the tables is the unweighted average over domains.
- Each record names the repo commit it was produced at; this upload was staged at commit `{commit[:12]}`.
- Paths are relative to the work root and the machine's hostname is replaced by `remote-m4pro`.

## Citation

```bibtex
@misc{{baid2026looptax,
  author       = {{Baid, Shubham}},
  title        = {{LoopTax: does quantization error compound in looped transformers?}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{DATASET}}}}},
  note         = {{Code: {GITHUB}}}
}}
```
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    scrub = scrubber()
    n_runs = n_chunked = 0
    for src in sorted(p for p in RESULTS.rglob("*") if p.is_file()):
        dst = out / src.relative_to(RESULTS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = scrub(src.read_text())
        if src.suffix == ".json":
            rec = json.loads(text)
            if src.parent.name == "env":
                rec.pop("hostname", None)
                text = json.dumps(rec, indent=1) + "\n"
            if isinstance(rec, dict) and "domains" in rec:
                n_runs += 1
                n_chunked += any("chunk_sums" in m for m in rec["domains"].values())
        dst.write_text(text)
    experiments = sorted(p.parent.name for p in out.glob("*/summary.csv"))
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    (out / "README.md").write_text(card(experiments, n_runs, n_chunked, commit))
    bad = [(str(f.relative_to(out)), s) for f in out.rglob("*") if f.is_file()
           for s in FORBIDDEN if s in f.read_text()]
    if bad:
        raise SystemExit(f"scrub incomplete: {bad[:10]}")
    print(f"staged {sum(1 for f in out.rglob('*') if f.is_file())} files, {n_runs} runs ({n_chunked} with chunk sums) in {out}")


if __name__ == "__main__":
    main()
