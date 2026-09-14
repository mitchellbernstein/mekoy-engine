# mekoy-engine

**Compile a specialist.** You describe a job and hand over examples. The engine tries
many harness configurations against a held-out set, keeps the winner, and hands back
a reproducible System with its measured score — running entirely on your machine.

The model is not the product. The **harness** is: the brief, the examples, the format
rules, the deterministic checks, the retries. In our own measurements, changing only
the harness moved accuracy from **0.905 to 0.950 on the same model** — a bigger gain
than swapping to a model twice the size, which actually scored *worse* while running
twice as slow.

## What it does

```
describe a job  →  hand over examples  →  the engine searches the harness
                                        →  held-out test scored once
                                        →  a System: spec + score + receipts
```

A System is versioned and reproducible from its spec, the data, and the search log.
It can be run locally, self-hosted, or shipped as a bundle with a compose file.

## Quickstart

Needs a local OpenAI-compatible model server (Ollama by default). No cloud, no
account, no API key.

```bash
ollama pull qwen2.5:7b           # or any local model
uv sync
./scripts/speedrun.sh            # the reference run, end to end
```

Or step by step:

```bash
uv run python -m mekoy.cli check-eval examples/bucko-restaurant/generated.jsonl
uv run python -m mekoy.cli eval --approve examples/bucko-restaurant/generated.jsonl
uv run python -m mekoy.cli compile examples/bucko-restaurant/generated.jsonl --trials 6
uv run python -m mekoy.cli report examples/bucko-restaurant/generated.jsonl
```

`compile` refuses to run without the approval stamp. The search reads the dev slice
only; the test slice is scored once, after selection.

## What the search actually varies

Every axis is searched, none is assumed:

| axis | what it tries |
|---|---|
| examples shown | 0, 2, 4, 8 |
| retries | 0 or one repair pass |
| format enforcement | free text, valid JSON, exact schema |
| instruction | several written variants, plus reflected ones |
| which examples | first *n*, or bootstrapped by success |
| model | any local model you register |
| self-consistency | one sample, or majority vote over several |

Runs are **staged**: candidates are priced on a minibatch, pruned, and only survivors
get a full pass. Selection is **Pareto** over quality, cost, and latency — never one
number.

## Tasks included

Three task classes, four real corpora. Add your own by writing a task definition.

| task | corpus | rows | source |
|---|---|---|---|
| restaurant calls | `examples/bucko-restaurant/` | 271 | hand-labeled + generated |
| receipts | `examples/cord-receipt/cord.jsonl` | 80 | CORD (real OCR receipts) |
| receipts | `examples/cord-receipt/sroie.jsonl` | 105 | SROIE (real scanned receipts) |
| intent classification | `examples/banking77/` | 616 | BANKING77, all 77 intents |

## Measured results, and where they lose

On a 54-row held-out slice, compiled locally on a consumer Mac:

| | compiled (local) | frontier |
|---|---|---|
| accuracy | **0.940** | 0.918 – 0.958 |
| cost per document | **$0.00000** | $0.01700 |
| speed per document | 8.0s | 1.4 – 1.7s |

Beats five of six frontier configurations. **Not faster than a frontier API.** Loses
slightly to GPT-5.5 on accuracy, by about three fields out of 378.

Every failed approach is written up too, including a metric bug that flattered our
own numbers twice. See [`WRITEUP.md`](WRITEUP.md).

## What it does not do

- **No cloud.** Nothing leaves your machine. There is a local HTTP API you can
  self-host; we do not operate it for you.
- **No closed-model training data.** Outputs from GPT, Claude, or Gemini never enter
  training or compile data. A closed API may only be a *score baseline*.
- **No OCR.** Text in, structured data out.
- **Training is optional and off by default.** The report says `training: skipped`
  when the harness alone clears the bar, which is the usual outcome.

## Verification

```bash
uv run pytest -q                       # the full suite
uv run python -m mekoy.cli check-eval <examples.jsonl>   # audit labels against their own text
```

`check-eval` is the part we would ask you to try first. It reports rows whose
transcript does not support its own label — the failure that makes a benchmark say
anything at all.

## Layout

```
src/mekoy/
  search.py     the search: candidates, staged pruning, Pareto, SLO stop
  reflect.py    reflective optimizer: read the failures, rewrite the brief
  verify.py     deterministic checks — the gate a candidate must clear
  score.py      field scoring
  tasks.py      task definitions: schema, prompt, gate, scorer
  compile.py    compile a System and produce its report
  train.py      optional LoRA, optional from-scratch training
  api/          the local HTTP control plane
  mcp_server.py the MCP connector
```

## Licence and split

This repository is the engine. A separate private repository holds our portal,
cloud runner, and marketplace; the engine never imports it and does not need it.

Engine builds are self-sufficient: clone, `uv sync`, run.
