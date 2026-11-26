"""Build the fixed, versioned eval texts used for every KL / perplexity run.

Three domains, each a plain-text file fed to llama-perplexity:
  wikitext      wikitext-2-raw-v1 test split (Salesforce/wikitext, pinned revision)
  code          Python and C sources from CPython at a pinned tag
  multilingual  FLORES-200 devtest sentences (ungated mteb/flores mirror), 8 languages interleaved in paragraphs

Writes data/eval/<domain>.txt and data/eval/manifest.json (sources, revisions, sha256, sizes).
The texts are not committed (licenses vary); the manifest is, so any copy can be checked.

    python -m looptax.build_evalset --out data/eval
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import urllib.request

WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
CPYTHON_TAG = "v3.13.0"
CPYTHON_FILES = [
    "Lib/heapq.py", "Lib/bisect.py", "Lib/textwrap.py", "Lib/json/decoder.py", "Lib/json/encoder.py",
    "Lib/functools.py", "Lib/dataclasses.py", "Lib/graphlib.py", "Lib/statistics.py", "Lib/fractions.py",
    "Objects/listsort.txt", "Objects/listobject.c", "Modules/_heapqmodule.c",
]
# openlanguagedata/flores_plus is gated; mteb/flores is an ungated parquet copy of FLORES-200 (CC-BY-SA-4.0)
FLORES = ("mteb/flores", "69edd3336d102dd184dd531b57defac6c9c60c8f", "devtest.parquet")
FLORES_LANGS = ["deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl", "zho_Hans", "jpn_Jpan", "hin_Deva", "arb_Arab"]
FLORES_SENTENCES = 240          # first N devtest sentences per language
FLORES_PARAGRAPH = 12           # sentences per paragraph before switching language


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "looptax-evalset"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def hf_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def build_wikitext() -> tuple[str, dict]:
    import pyarrow.parquet as pq
    repo, rev, path = WIKITEXT
    table = pq.read_table(io.BytesIO(_get(hf_url(repo, rev, path))))
    # same convention as llama.cpp's scripts/get-wikitext-2.sh: the raw lines joined as-is
    text = "".join(table.column("text").to_pylist())
    return text, {"source": f"hf://datasets/{repo}", "revision": rev, "file": path}


def build_code() -> tuple[str, dict]:
    parts = []
    for f in CPYTHON_FILES:
        src = _get(f"https://raw.githubusercontent.com/python/cpython/{CPYTHON_TAG}/{f}").decode("utf-8")
        parts.append(f"# file: {f}\n{src}\n")
    return "\n".join(parts), {"source": "https://github.com/python/cpython", "revision": CPYTHON_TAG, "files": CPYTHON_FILES}


def build_multilingual() -> tuple[str, dict]:
    import pyarrow.parquet as pq
    repo, rev, path = FLORES
    table = pq.read_table(io.BytesIO(_get(hf_url(repo, rev, path))))
    by_lang = {lang: table.column(lang).to_pylist()[:FLORES_SENTENCES] for lang in FLORES_LANGS}
    paras = []
    for start in range(0, FLORES_SENTENCES, FLORES_PARAGRAPH):
        for lang in FLORES_LANGS:
            paras.append(" ".join(by_lang[lang][start:start + FLORES_PARAGRAPH]))
    return "\n\n".join(paras) + "\n", {
        "source": f"hf://datasets/{repo}", "revision": rev, "file": path, "split": "devtest", "languages": FLORES_LANGS,
        "sentences_per_language": FLORES_SENTENCES, "sentences_per_paragraph": FLORES_PARAGRAPH,
        "layout": "paragraphs cycle through the languages in order"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fn in (("wikitext", build_wikitext), ("code", build_code), ("multilingual", build_multilingual)):
        text, meta = fn()
        data = text.encode("utf-8")
        (out / f"{name}.txt").write_bytes(data)
        meta.update({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest[name] = meta
        print(name, meta["bytes"], meta["sha256"][:16])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
