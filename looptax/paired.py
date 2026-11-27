"""Paired comparison of variants on the same evaluation chunks.

For each variant with per-chunk sums (runs made with the build-lt llama-perplexity), per-chunk mean KL is recovered for
every domain and concatenated in a fixed order. Differences against a baseline variant are taken chunk by chunk (same
text, same reference) and bootstrapped over chunks (B resamples, percentile 95% interval), for the mean KL difference
and the ratio of mean KLs. Paired differences are far less noisy than comparing two independent means.

    python -m looptax.paired results/phase3-holdout --baseline unsloth-like
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .alpha_fit import chunk_kl


def load(d: pathlib.Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("role") != "variant":
            continue
        doms = sorted(rec["domains"])
        try:
            out[rec["variant"]] = {"kl": np.concatenate([chunk_kl(rec["domains"][x]) for x in doms]),
                                   "gb": rec.get("weights", {}).get("file_gb"), "domains": doms}
        except ValueError:
            continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args(argv)
    d = pathlib.Path(a.dir)
    v = load(d)
    base = v[a.baseline]["kl"]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(base), size=(a.boot, len(base)))
    res = []
    for vid, x in v.items():
        diff = x["kl"] - base
        bd = diff[idx].mean(axis=1)
        br = x["kl"][idx].mean(axis=1) / base[idx].mean(axis=1)
        res.append({"variant": vid, "file_gb": x["gb"], "kl_mean": float(x["kl"].mean()), "chunks": int(len(diff)),
                    "diff_vs_baseline": float(diff.mean()),
                    "diff_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                    "ratio_vs_baseline": float(x["kl"].mean() / base.mean()),
                    "ratio_ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]})
    res.sort(key=lambda r: r["kl_mean"])
    (d / f"paired_vs_{a.baseline}.json").write_text(json.dumps({"baseline": a.baseline, "bootstrap": a.boot, "results": res}, indent=1) + "\n")
    for r in res:
        print(f"{r['variant']:26s} {r['file_gb'] or 0:7.3f} GB  KL {r['kl_mean']:.5f}  diff {r['diff_vs_baseline']:+.5f} "
              f"[{r['diff_ci95'][0]:+.5f}, {r['diff_ci95'][1]:+.5f}]  ratio {r['ratio_vs_baseline']:.3f} "
              f"[{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
