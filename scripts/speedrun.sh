#!/usr/bin/env bash
# The reference local run. One command, from examples to a measured System.
#
# nanochat keeps runs/speedrun.sh as "the reference way" to train its model, and
# that convention is worth copying: a project whose headline number has one
# canonical reproduction path is a project others can check. Your numbers will not
# match ours exactly — different hardware, different model server — and that is the
# point, because you can run it and see for yourself.
#
#   ./scripts/speedrun.sh                      # the restaurant-call job
#   MODEL=qwen2.5:14b ./scripts/speedrun.sh    # a different local model
#
# Needs: a local OpenAI-compatible model server on 127.0.0.1:11434 (Ollama by
# default). No cloud. No account. No key.

set -euo pipefail

EXAMPLES="${EXAMPLES:-examples/bucko-restaurant/generated.jsonl}"
MODEL="${MODEL:-qwen2.5:7b}"
TRIALS="${TRIALS:-6}"

echo "== job: $EXAMPLES"
echo "== model: $MODEL   candidates: $TRIALS"

echo
echo "== is the model server up?"
curl -sS -m 5 -o /dev/null "http://127.0.0.1:11434/health" \
  || { echo "no model server on 127.0.0.1:11434; start Ollama first"; exit 1; }

echo
echo "== audit the labels before trusting any score"
uv run python -m mekoy.cli check-eval "$EXAMPLES"

echo
echo "== approve the eval, then compile"
uv run python -m mekoy.cli eval --approve "$EXAMPLES"
uv run python -m mekoy.cli compile "$EXAMPLES" --trials "$TRIALS" --model "$MODEL" --track

echo
echo "== the card"
uv run python -m mekoy.cli report "$EXAMPLES"

echo
echo "== done. reproduce this at any time with the same command."
