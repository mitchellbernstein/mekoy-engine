# Mekoy

Local compiler for owned AI systems. The plan is [`PLAN.md`](PLAN.md). Agent rules: [`AGENTS.md`](AGENTS.md).

## Run locally with Ollama

Extract-and-verify receipts from **text**, not PDFs. Open-weight models via Ollama. No closed-model distillation. No hosting.

Install Ollama, pull `qwen2.5:7b`, and keep it on `http://127.0.0.1:11434`. Then:

```bash
uv sync
uv run mekoy eval examples/cord-receipt/examples.jsonl --approve
uv run mekoy compile examples/cord-receipt/examples.jsonl --quick
uv run mekoy invoke path/to/receipt.txt --examples examples/cord-receipt/examples.jsonl --k-shot 4
```

`mekoy eval --approve` writes `.eval-approved` next to the examples file. Compile refuses to run without that stamp.

`--quick` searches one arm (`k=0`, `retries=1`) so a 7B demo finishes in minutes. Omit `--quick` to search k-shot × retry on the holdout.

`uv run pytest` does not need a model. Compile and invoke talk to `http://127.0.0.1:11434/v1`.

`mekoy serve` is not in this local MVP. Use `mekoy invoke`.
