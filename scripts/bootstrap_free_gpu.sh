#!/usr/bin/env bash
# Free-GPU bootstrap for Kaggle / Colab: installs deps and Ollama, pulls the models named in a
# config, benchmarks them, runs the generation, and zips everything for download.
# No paid tiers, no API keys. Untested on real Kaggle/Colab as of writing: expect to adjust.
#
#   CONFIG=configs/my_run.yaml MAX_HOURS=4 bash scripts/bootstrap_free_gpu.sh
#   DRY_RUN=1 bash scripts/bootstrap_free_gpu.sh      # print the steps, change nothing
#
# Env: CONFIG (needs verified Ollama tags), MAX_HOURS (empty = full grid), OUT (artifact dir),
#      SKIP_SYNC=1 (deps already installed), SKIP_OLLAMA=1 (no Ollama, e.g. scripted test configs).
# WARNING: responses may contain harmful-compliance text. Keep Kaggle/Colab outputs private.
set -euo pipefail

CONFIG="${CONFIG:-configs/local_small.example.yaml}"
MAX_HOURS="${MAX_HOURS:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_OLLAMA="${SKIP_OLLAMA:-0}"
UV="uv run"
[ "$SKIP_SYNC" = 1 ] && UV="uv run --no-sync"

if [ -n "${OUT:-}" ]; then :
elif [ -d /kaggle/working ]; then OUT=/kaggle/working/artifacts
elif [ -d /content ]; then OUT=/content/artifacts
else OUT="$PWD/artifacts"; fi

run() { echo "+ $*"; if [ "$DRY_RUN" != 1 ]; then "$@"; fi; }
step() { echo; echo "== $*"; }

cd "$(dirname "$0")/.."
step "platform: out=$OUT config=$CONFIG max_hours=${MAX_HOURS:-full grid} dry_run=$DRY_RUN"
run mkdir -p "$OUT"

step "python deps"
command -v uv >/dev/null 2>&1 || run python3 -m pip install -q uv
[ "$SKIP_SYNC" = 1 ] || run uv sync --extra local
run $UV indicsafe doctor

step "ollama"
if [ "$SKIP_OLLAMA" = 1 ]; then
  echo "skipped (SKIP_OLLAMA=1)"
elif ! command -v ollama >/dev/null 2>&1; then
  command -v zstd >/dev/null 2>&1 || run bash -c 'apt-get update -qq && apt-get install -y -qq zstd'
  run bash -c 'curl -fsSL https://ollama.com/install.sh | sh'
fi
if [ "$SKIP_OLLAMA" = 1 ]; then :
elif [ "$DRY_RUN" = 1 ]; then
  echo "+ nohup ollama serve > $OUT/ollama.log 2>&1 &   (then wait up to 60 s for :11434)"
  echo "+ for each tag in: indicsafe models --config $CONFIG  ->  ollama pull <tag>"
else
  if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    nohup ollama serve > "$OUT/ollama.log" 2>&1 &
    for _ in $(seq 1 60); do
      curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
      sleep 1
    done
    curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || { echo "ollama did not start; see $OUT/ollama.log"; exit 1; }
  fi
  $UV indicsafe models --config "$CONFIG" | while read -r tag; do
    if [ -n "$tag" ]; then echo "+ ollama pull $tag"; ollama pull "$tag"; fi
  done
fi

step "bench + run (re-run this script to resume after a dead session)"
run $UV indicsafe bench --config "$CONFIG" --bench-file "$OUT/bench.json"
if [ -n "$MAX_HOURS" ]; then
  run $UV indicsafe run --config "$CONFIG" --bench-file "$OUT/bench.json" --out "$OUT/runs" --max-hours "$MAX_HOURS"
else
  run $UV indicsafe run --config "$CONFIG" --bench-file "$OUT/bench.json" --out "$OUT/runs"
fi

step "package artifacts"
if [ "$DRY_RUN" = 1 ]; then
  echo "+ (cd $OUT && python3 -m zipfile -c indicsafe_artifacts.zip runs bench.json)"
else
  (cd "$OUT" && rm -f indicsafe_artifacts.zip && python3 -m zipfile -c indicsafe_artifacts.zip runs bench.json)
fi
echo; echo "done: $OUT/indicsafe_artifacts.zip"
