"""Build results/manifest.json and one summary CSV per experiment from the per-run JSON files.

    python -m looptax.summarize            # all experiments under results/
"""

from __future__ import annotations

import csv
import json
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
SKIP_DIRS = {"env", "layout"}
DOMAINS = ("wikitext", "code", "multilingual")


def row(rec: dict) -> dict:
    w = rec.get("weights", {})
    r = {"experiment": rec["experiment"], "variant": rec["variant"], "role": rec.get("role"),
         "file_gb": w.get("file_gb"), "bpw": w.get("bpw")}
    spec = rec.get("spec", {})
    r["passes"] = rec.get("passes")
    for k in ("ftype", "block", "family", "qtype", "gate_up", "est_gb"):
        r[k] = spec.get(k)
    r["imatrix"] = bool(spec.get("imatrix", True)) if spec else None
    kl, kl_err, n = 0.0, 0.0, 0
    for d in DOMAINS:
        m = rec.get("domains", {}).get(d, {})
        for k in ("kld_mean", "kld_mean_err", "kld_p99", "same_top_p_pct", "ppl_q", "ppl_ratio", "ppl"):
            if k in m:
                r[f"{d}_{k}"] = m[k]
        if "kld_mean" in m:
            kl += m["kld_mean"]
            kl_err += m.get("kld_mean_err", 0) ** 2
            n += 1
    for d in DOMAINS:
        m = rec.get("domains_vs3", {}).get(d, {})
        for k in ("kld_mean", "kld_mean_err", "same_top_p_pct", "ppl_q"):
            if k in m:
                r[f"{d}_vs3_{k}"] = m[k]
    for ref_id, doms in (rec.get("domains_vs") or {}).items():
        vals = [m["kld_mean"] for m in doms.values() if "kld_mean" in m]
        errs = [m.get("kld_mean_err", 0) for m in doms.values() if "kld_mean" in m]
        if vals:
            r[f"{ref_id}_kld_mean_avg"] = sum(vals) / len(vals)
            r[f"{ref_id}_kld_mean_avg_err"] = sum(e * e for e in errs) ** 0.5 / len(vals)
        if "wikitext" in doms:
            r[f"{ref_id}_wikitext_same_top_p_pct"] = doms["wikitext"].get("same_top_p_pct")
    sp = rec.get("speed") or {}
    for k in ("pp_tps", "tg_tps"):
        if k in sp:
            r[k] = sp[k]
    if n:
        r["kld_mean_avg"] = kl / n          # unweighted average over domains
        r["kld_mean_avg_err"] = kl_err ** 0.5 / n
    for b in rec.get("bench") or []:
        if b.get("n_prompt") and not b.get("n_gen"):
            r["pp_tps"] = b["avg_ts"]
        if b.get("n_gen") and not b.get("n_prompt"):
            r["tg_tps"] = b["avg_ts"]
    r["git_commit"] = rec.get("git_commit")
    r["llama_cpp_commit"] = rec.get("llama_cpp_commit")
    r["timestamp"] = rec.get("timestamp")
    return r


def main():
    manifest = []
    for exp_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        rows = []
        for f in sorted(exp_dir.glob("*.json")):
            rec = json.loads(f.read_text())
            if "experiment" not in rec:
                continue
            rows.append(row(rec))
            manifest.append({"file": str(f.relative_to(RESULTS)), "experiment": rec["experiment"], "variant": rec["variant"],
                             "git_commit": rec.get("git_commit"), "llama_cpp_commit": rec.get("llama_cpp_commit"),
                             "machine": rec.get("machine"), "timestamp": rec.get("timestamp")})
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(exp_dir / "summary.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
    for extra in ("env", "layout"):
        for f in sorted((RESULTS / extra).glob("*.json")):
            manifest.append({"file": str(f.relative_to(RESULTS)), "experiment": extra})
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} result files indexed")


if __name__ == "__main__":
    main()
