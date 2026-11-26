"""Experiment runner: build each variant, measure KL / PPL / top-1 vs the reference, benchmark speed, record everything.

Runs on the remote box. Resumable: a variant whose result file exists is skipped.

    python -m looptax.run configs/phase1.yaml --make-base     # reference logits + reference metrics (once)
    python -m looptax.run configs/phase1.yaml                 # all variants
    python -m looptax.run configs/phase1.yaml --only q4_k_m   # one variant
    python -m looptax.run configs/phase1.yaml --dry-run       # print commands only

Paths in the config are relative to $LOOPTAX_ROOT (default ~/looptax) unless absolute.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

import yaml

from .gguf_header import weight_stats

ROOT = pathlib.Path(os.environ.get("LOOPTAX_ROOT", pathlib.Path.home() / "looptax"))
REPO = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / "llama.cpp" / "build" / "bin"


def rp(p: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else ROOT / p


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git(*args, cwd=REPO) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


# the remote copy of the repo is an rsync without .git; remote.sh writes the commit into REVISION.
# Read once at start-up so a later sync cannot relabel results produced by the code already running.
_REV_FILE = REPO / "REVISION"
GIT_COMMIT = _REV_FILE.read_text().strip() if _REV_FILE.exists() else git("rev-parse", "HEAD")


def provenance() -> dict:
    return {
        "git_commit": GIT_COMMIT,
        "llama_cpp_commit": git("rev-parse", "HEAD", cwd=ROOT / "llama.cpp"),
        "machine": socket.gethostname(),
        "timestamp": now(),
    }


def sh(cmd: list[str], log: pathlib.Path, dry: bool) -> str:
    line = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"$ {line}", flush=True)
    if dry:
        return ""
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    out = r.stdout + r.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n### {now()} $ {line}\n{out}\n### exit {r.returncode} after {time.time() - t0:.0f}s\n")
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}), see {log}:\n{out[-3000:]}")
    return out


_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
KL_PATTERNS = {
    "ppl_q": rf"Mean PPL\(Q\)\s*:\s*{_NUM}\s*±\s*{_NUM}",
    "ppl_base": rf"Mean PPL\(base\)\s*:\s*{_NUM}\s*±\s*{_NUM}",
    "ln_ppl_ratio": rf"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*{_NUM}\s*±\s*{_NUM}",
    "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*{_NUM}\s*±\s*{_NUM}",
    "kld_mean": rf"Mean\s+KLD:\s*{_NUM}\s*±\s*{_NUM}",
    "kld_max": rf"Maximum KLD:\s*{_NUM}",
    "kld_p99_9": rf"99\.9%\s+KLD:\s*{_NUM}",
    "kld_p99": rf"99\.0%\s+KLD:\s*{_NUM}",
    "kld_p95": rf"95\.0%\s+KLD:\s*{_NUM}",
    "kld_p90": rf"90\.0%\s+KLD:\s*{_NUM}",
    "kld_median": rf"Median\s+KLD:\s*{_NUM}",
    "same_top_p_pct": rf"Same top p:\s*{_NUM}\s*±\s*{_NUM}",
    "rms_dp_pct": rf"RMS Δp\s*:\s*{_NUM}\s*±\s*{_NUM}",
}


def parse_kl(out: str) -> dict:
    res = {}
    for k, pat in KL_PATTERNS.items():
        m = re.search(pat, out)
        if not m:
            continue
        res[k] = float(m.group(1))
        if m.lastindex and m.lastindex >= 2:
            res[k + "_err"] = float(m.group(2))
    m = re.search(r"Final estimate: PPL = " + _NUM + r" \+/- " + _NUM, out)
    if m:
        res["ppl"], res["ppl_err"] = float(m.group(1)), float(m.group(2))
    chunks = re.findall(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", out)
    if chunks:
        res["chunks"], res["n_ctx"] = int(chunks[-1][0]), int(chunks[-1][1])
    return res


def gpu_args(cfg: dict, v: dict) -> list[str]:
    llama = {**cfg.get("llama", {}), **v.get("llama", {})}
    # ngl: null leaves layer placement to llama.cpp's --fit (for variants larger than GPU memory)
    args = [] if llama.get("ngl", 99) is None else ["-ngl", str(llama.get("ngl", 99))]
    if llama.get("fit") is not None:
        args += ["-fit", llama["fit"]]
    for kv in llama.get("override_kv", []):
        args += ["--override-kv", kv]
    return args + [str(x) for x in llama.get("extra", [])]


def base_path(cfg: dict, domain: str) -> pathlib.Path:
    ref = cfg["reference"]
    return rp(cfg.get("kld_dir", "kld")) / ref["id"] / f"{domain}-c{cfg['eval']['ctx']}.kld"


def make_base(cfg: dict, dry: bool):
    ref = cfg["reference"]
    out_dir = REPO / "results" / cfg["experiment"]
    log = rp("logs") / f"{cfg['experiment']}-reference.log"
    rec = {"experiment": cfg["experiment"], "variant": ref["id"], "role": "reference", "config": cfg,
           "model_file": str(rp(ref["model"]).name), **provenance(), "domains": {}}
    for domain, chunks in cfg["eval"]["domains"].items():
        bp = base_path(cfg, domain)
        bp.parent.mkdir(parents=True, exist_ok=True)
        cmd = [BIN / "llama-perplexity", "-m", rp(ref["model"]), "-f", rp(cfg["eval"]["dir"]) / f"{domain}.txt",
               "-c", str(cfg["eval"]["ctx"]), "--chunks", str(chunks), "--kl-divergence-base", bp, *gpu_args(cfg, ref)]
        t0 = time.time()
        out = sh(cmd, log, dry)
        rec["domains"][domain] = {**parse_kl(out), "seconds": round(time.time() - t0, 1),
                                  "base_file_bytes": bp.stat().st_size if bp.exists() else None}
    if cfg.get("bench") and not dry:
        rec["bench"] = bench(cfg, ref, rp(ref["model"]), log, dry)
    if not dry:
        rec["weights"] = summarize_weights(rp(ref["model"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ref['id']}.json").write_text(json.dumps(rec, indent=1) + "\n")


def bench(cfg: dict, v: dict, model: pathlib.Path, log: pathlib.Path, dry: bool) -> list[dict] | None:
    b = cfg["bench"]
    llama = {**cfg.get("llama", {}), **v.get("llama", {})}
    # llama-bench takes -ngl but not --override-kv or -fit; variants that need those set bench: false
    clean = [BIN / "llama-bench", "-m", model, "-p", str(b.get("pp", 512)), "-n", str(b.get("tg", 128)),
             "-r", str(b.get("reps", 3)), "-o", "json", "-ngl", str(llama.get("ngl", 99))]
    out = sh(clean, log, dry)
    if dry:
        return None
    m = re.search(r"\[\s*\{.*\}\s*\]", out, re.S)
    rows = json.loads(m.group(0)) if m else []
    keep = ("n_prompt", "n_gen", "avg_ts", "stddev_ts", "n_gpu_layers", "flash_attn", "n_threads", "backends", "gpu_info", "build_commit")
    return [{k: r.get(k) for k in keep} for r in rows]


def summarize_weights(path: pathlib.Path) -> dict:
    st = weight_stats(str(path))
    by_type: dict[str, int] = {}
    for t in st["tensors"].values():
        by_type[t["type"]] = by_type.get(t["type"], 0) + t["nbytes"]
    return {"file_bytes": st["file_bytes"], "file_gb": round(st["file_bytes"] / 1e9, 3), "bpw": round(st["bpw"], 4),
            "n_weights": st["n_weights"], "bytes_by_type": by_type,
            "tensor_types": {n: t["type"] for n, t in st["tensors"].items()}}


def build_variant(cfg: dict, v: dict, log: pathlib.Path, dry: bool) -> tuple[pathlib.Path, bool, list[str] | None]:
    if v.get("file"):
        return rp(v["file"]), False, None
    out = rp(cfg.get("work_dir", "work")) / f"{cfg['experiment']}-{v['id']}.gguf"
    cmd = [BIN / "llama-quantize", "--allow-requantize"]
    im = v.get("imatrix", cfg.get("imatrix_default", False))
    if im:
        cmd += ["--imatrix", rp(cfg["imatrix"] if im is True else im)]
    if v.get("pure"):
        cmd += ["--pure"]
    for o in v.get("overrides", []):
        cmd += ["--tensor-type", o]
    cmd += [rp(v.get("source", cfg["source"])), out, v["ftype"], str(cfg.get("quant_threads", os.cpu_count()))]
    if not out.exists() or dry:
        out.parent.mkdir(parents=True, exist_ok=True)
        # output size relative to the NVFP4 source: Q8_0 everywhere is ~1.6x, Q6_K ~1.25x, others and scans <= 1.05x
        ratio = 1.05 if v.get("pure") else {"Q8_0": 1.65, "Q6_K": 1.3, "Q5_K_M": 1.15}.get(v["ftype"], 1.05)
        need = rp(v.get("source", cfg["source"])).stat().st_size * ratio + (5 << 30) if not dry else 0
        free = shutil.disk_usage(out.parent).free
        if free < need:
            raise RuntimeError(f"not enough disk for {out.name}: {free / 1e9:.1f} GB free, want {need / 1e9:.1f} GB")
        tmp_out = out.with_suffix(".partial.gguf")
        cmd_tmp = [*cmd[:-3], tmp_out, *cmd[-2:]]
        sh(cmd_tmp, log, dry)
        if not dry:
            tmp_out.rename(out)
    return out, True, [str(c) for c in cmd]


def run_variant(cfg: dict, v: dict, dry: bool):
    out_dir = REPO / "results" / cfg["experiment"]
    res_path = out_dir / f"{v['id']}.json"
    if res_path.exists():
        print(f"skip {v['id']} (result exists)")
        return
    log = rp("logs") / f"{cfg['experiment']}-{v['id']}.log"
    t0 = time.time()
    model, made, qcmd = build_variant(cfg, v, log, dry)
    t_quant = time.time() - t0
    rec = {"experiment": cfg["experiment"], "variant": v["id"], "role": "variant", "spec": v,
           "reference": cfg["reference"]["id"], "eval": cfg["eval"], "model_file": model.name,
           "quantize_command": qcmd, "quantize_seconds": round(t_quant, 1) if made else None,
           **provenance(), "domains": {}}
    for domain in cfg["eval"]["domains"]:
        cmd = [BIN / "llama-perplexity", "-m", model, "--kl-divergence-base", base_path(cfg, domain), "--kl-divergence",
               "-c", str(cfg["eval"]["ctx"]), *gpu_args(cfg, v)]
        t1 = time.time()
        out = sh(cmd, log, dry)
        rec["domains"][domain] = {**parse_kl(out), "seconds": round(time.time() - t1, 1)}
    if dry:
        return
    rec["weights"] = summarize_weights(model)
    if cfg.get("bench") and v.get("bench", True):
        rec["bench"] = bench(cfg, v, model, log, dry)
    rec["total_seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"done {v['id']}: " + ", ".join(f"{d} KLD {m.get('kld_mean')}" for d, m in rec["domains"].items()), flush=True)
    if made and not v.get("keep", cfg.get("keep_variants", False)):
        model.unlink()
    elif not made and v.get("delete_after") and model.resolve().is_relative_to(rp("models").resolve()):
        # downloaded comparison files (community GGUFs) are removed once measured, to free disk
        model.unlink()
        print(f"deleted {model.name} after measuring it", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--make-base", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(REPO / a.config if not os.path.isabs(a.config) else a.config))
    if a.make_base:
        make_base(cfg, a.dry_run)
        return
    failed = []
    for v in cfg["variants"]:
        if a.only and v["id"] not in a.only:
            continue
        try:
            run_variant(cfg, v, a.dry_run)
        except Exception as e:  # keep going; failures are logged and reported at the end
            print(f"FAILED {v['id']}: {e}", file=sys.stderr, flush=True)
            failed.append(v["id"])
    if failed:
        print("failed variants: " + " ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
