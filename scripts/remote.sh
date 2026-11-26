#!/usr/bin/env bash
# Run a command on the remote compute box inside a named tmux session, logging to ~/looptax/logs.
#   scripts/remote.sh sync                      copy repo code (not results/models) to the remote
#   scripts/remote.sh run NAME -- CMD...        sync, then run CMD in tmux session NAME (cwd = remote repo)
#   scripts/remote.sh sh CMD...                 run CMD directly over ssh (cwd = remote repo)
#   scripts/remote.sh tail NAME                 show the last lines of NAME's log
#   scripts/remote.sh ls                        list tmux sessions
set -euo pipefail
REMOTE="${LOOPTAX_REMOTE:-goosemini@100.104.128.73}"
ROOT="${LOOPTAX_REMOTE_ROOT:-looptax}"           # relative to remote $HOME
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV='export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH; export LOOPTAX_ROOT=$HOME/'"$ROOT"';'

sync_code() {
  rsync -az --delete \
    --exclude .git --exclude results/ --exclude site/node_modules --exclude site/dist \
    --exclude '*.gguf' --exclude .venv --exclude __pycache__ --exclude loop-tax.bundle \
    "$HERE/" "$REMOTE:$ROOT/repo/"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  sync) sync_code ;;
  run)
    name="$1"; shift; [[ "${1:-}" == "--" ]] && shift
    sync_code
    q=$(printf '%q ' "$@")
    ssh "$REMOTE" "$ENV mkdir -p \$LOOPTAX_ROOT/logs; cd \$LOOPTAX_ROOT/repo && \
      tmux new-session -d -s $name \"$ENV cd \$LOOPTAX_ROOT/repo && { $q; echo EXIT=\\\$?; } >> \$LOOPTAX_ROOT/logs/$name.log 2>&1\" && \
      echo started tmux session $name, log: ~/$ROOT/logs/$name.log"
    ;;
  sh) ssh "$REMOTE" "$ENV cd \$LOOPTAX_ROOT/repo 2>/dev/null || cd \$LOOPTAX_ROOT; $*" ;;
  tail) ssh "$REMOTE" "tail -n ${2:-30} $ROOT/logs/$1.log" ;;
  ls) ssh "$REMOTE" "$ENV tmux ls" ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
