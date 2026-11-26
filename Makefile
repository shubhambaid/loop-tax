# LoopTax. Heavy targets run on the remote box via scripts/remote.sh (tmux, logged, resumable).
PY ?= python3
R := scripts/remote.sh

.PHONY: help setup remote-sync remote-inventory remote-build remote-download remote-ls sync-results toy charts site

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## local venv with analysis deps
	uv venv .venv && uv pip install -p .venv -r requirements.txt

remote-sync: ## copy code to the remote box
	$(R) sync

remote-inventory: ## record remote hardware/software into results/env
	$(R) sync && $(R) sh bash scripts/inventory.sh > results/env/remote_inventory.json

remote-build: ## build pinned llama.cpp (Metal) on the remote
	$(R) run build -- bash scripts/build_llama.sh

remote-download: ## download reference GGUF + imatrix, then varloop checkpoint and BF16 source
	$(R) run dl_ref -- bash scripts/download_models.sh inkling-imatrix inkling-nvfp4
	$(R) run dl_more -- bash scripts/download_models.sh after:dl_ref varloop-hf inkling-bf16

remote-ls: ## list remote tmux sessions
	$(R) ls

sync-results: ## pull small result files from the remote into results/
	bash scripts/sync_results.sh

toy: ## rerun the toy simulation (Part 0, random weights)
	cd toy && $(PY) sim.py

charts: ## regenerate toy charts
	cd toy && $(PY) charts.py arch depth budget
