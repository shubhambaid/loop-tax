"""Headline analysis: how does the KL caused by quantizing ONE block scale with the number of loop passes N?

For each perturbed block (results/phase4-passes/<id>-p<N>.json, KL vs the reference at the same N) fit
    KL(N) = c * N^alpha        on the trained pass counts only (N = 1..5; N = 6 is extrapolation, reported apart)
by least squares on log KL vs log N. Uncertainty: paired bootstrap over evaluation chunks (each chunk = 1023 scored
tokens; the same resampled chunk indices are used at every N, because the text is the same), B resamples, percentile
95% interval.

Reading guide (small-perturbation regime, KL ~ quadratic in the logit change):
  alpha ~ 0   the effect does not depend on N              expected for coda blocks (run once, after the loop)
  alpha ~ 1   errors from the N applications add like independent noise
  alpha ~ 2   errors add coherently (the "loop tax")
  alpha < 1   later passes damp the error (gates / restart from the prelude output)
Prelude blocks run once, but their error is injected into every pass, so they are reported as their own category.

    python -m looptax.alpha_fit [--boot 5000]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
EXP = REPO / "results" / "phase4-passes"


def chunk_kl(domain_rec: dict) -> np.ndarray:
    """Per-chunk mean KL from cumulative sums printed by the patched llama-perplexity."""
    rows = domain_rec.get("chunk_sums")
    if not rows:
        raise ValueError("no chunk_sums (was the run made with build-lt?)")
    s = np.array([r[1] for r in rows])
    n = np.array([r[5] for r in rows], dtype=float)
    ds, dn = np.diff(np.concatenate([[0.0], s])), np.diff(np.concatenate([[0.0], n]))
    return ds / dn


def load() -> dict:
    """{variant: {"meta": spec, "N": {n: per-chunk KL array (domains concatenated in a fixed order)}}}"""
    out: dict = {}
    for f in sorted(EXP.glob("*-p*.json")):
        rec = json.loads(f.read_text())
        spec = rec.get("spec", {})
        if rec.get("role") != "variant" or spec.get("kind") != "headline":
            continue
        doms = sorted(rec["domains"])
        kl = np.concatenate([chunk_kl(rec["domains"][d]) for d in doms])
        v = out.setdefault(rec["variant"], {"meta": spec, "N": {}, "domains": doms})
        v["N"][rec["passes"]] = kl
    return out


def fit_alpha(ns: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([np.ones_like(ns), np.log(ns)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return float(coef[1]), float(np.exp(coef[0]))


def _fit_with_ci(v: dict, ns: list[int], idx: np.ndarray) -> tuple[float, list[float]]:
    K = np.stack([v["N"][n] for n in ns])
    alpha, _ = fit_alpha(np.array(ns, float), K.mean(axis=1))
    bm = K[:, idx].mean(axis=2)
    ba = np.array([fit_alpha(np.array(ns, float), np.clip(bm[:, b], 1e-12, None))[0] for b in range(idx.shape[0])])
    return alpha, [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]


def analyse(boot: int, trained: list[int], seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    res = []
    for vid, v in load().items():
        ns = [n for n in trained if n in v["N"]]
        if len(ns) < 3:
            continue
        K = np.stack([v["N"][n] for n in ns])          # [len(ns), chunks]
        n_chunks = K.shape[1]
        means = K.mean(axis=1)
        alpha, c = fit_alpha(np.array(ns, float), means)
        idx = rng.integers(0, n_chunks, size=(boot, n_chunks))
        bmeans = K[:, idx].mean(axis=2)                # [len(ns), boot]
        balpha = np.array([fit_alpha(np.array(ns, float), np.clip(bmeans[:, b], 1e-12, None))[0] for b in range(boot)])
        ratio = {n: float(v["N"][n].mean() / v["N"][1].mean()) for n in sorted(v["N"]) if 1 in v["N"]}
        # secondary fit without N = 1: pass 1 runs the loop with no cross-pass message, so N = 1 -> 2 changes the
        # computation itself, not only the reuse count
        ns2 = [n for n in ns if n >= 2]
        a2, ci2 = _fit_with_ci(v, ns2, idx) if len(ns2) >= 3 else (None, None)
        res.append({
            "variant": vid, "block": v["meta"].get("block"), "category": v["meta"].get("category"),
            "qtype": v["meta"].get("qtype"), "domains": v["domains"], "chunks": int(n_chunks), "fit_passes": ns,
            "alpha": alpha, "alpha_ci95": [float(np.percentile(balpha, 2.5)), float(np.percentile(balpha, 97.5))],
            "c": c, "alpha_2to5": a2, "alpha_2to5_ci95": ci2, "kl_mean_by_passes": {str(n): float(v["N"][n].mean()) for n in sorted(v["N"])},
            "kl_ratio_to_1_pass": {str(n): r for n, r in ratio.items()},
            "extrapolated_passes": sorted(n for n in v["N"] if n not in trained),
            "bootstrap": {"resamples": boot, "unit": "evaluation chunk (paired across N)", "seed": seed},
        })
    return sorted(res, key=lambda r: (r["qtype"], {"prelude": 0, "loop": 1, "coda": 2}.get(r["category"], 3), r["block"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--trained", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args(argv)
    res = analyse(a.boot, a.trained)
    (EXP / "alpha_fit.json").write_text(json.dumps(res, indent=1) + "\n")
    for r in res:
        lo, hi = r["alpha_ci95"]
        lo2, hi2 = r["alpha_2to5_ci95"]
        print(f"{r['variant']:16s} {r['category']:8s} alpha {r['alpha']:5.2f} [{lo:5.2f}, {hi:5.2f}]  "
              f"alpha(2-5) {r['alpha_2to5']:5.2f} [{lo2:5.2f}, {hi2:5.2f}]  "
              + " ".join(f"N{n}:{k:.5f}" for n, k in r["kl_mean_by_passes"].items()))


if __name__ == "__main__":
    main()
