# Working on the engine

Rules for anyone changing this code, human or agent. They are constraints that came out
of measurements, not preferences.

## The non-negotiables

- **No closed-model distillation.** Never put GPT / Claude / Gemini or other closed-API
  outputs into SFT, DPO, RL, or LoRA data. Contract risk, and providers fingerprint it.
  Allowed compile data: your own gold labels, your own production traces, synthetic data
  from **open** models whose licences permit it, and deterministic checks. A closed API
  may only ever be an **opt-in score baseline** — never a source of training data.

- **No search without an approved eval.** `compile` refuses to run until the examples are
  approved and the labels have been audited. A score computed against unverified labels
  is worse than no score, because it looks like evidence.

- **No mock-as-done.** Verify against real APIs and real eval sets. A branch that has
  only ever run against a stub is untested, not finished.

- **Keep the engine self-sufficient.** This repository must never import the portal, a
  cloud runner, or billing. If a feature only works because a hosted service exists, it
  does not belong here.

## Rules the measurements bought

- **The harness is model plus deterministic policy** — verify, retries, call limits. A
  prompt never owns a total or an approval decision. Changing only the harness moved
  accuracy 0.905 → 0.950 on the same model, which is why the harness is the product and
  the model is a knob.

- **Always A/B unconstrained.** Constrained decoding buys structural validity and can
  cost field accuracy; it did on llama.cpp. Never assume the grammar helped.

- **Free text is judged, never gated on exact match.** Compare meaning, not characters.

- **The search never sees the examples it is judged on.** Selection reads `dev`; `test`
  is scored once, after a winner exists.

- **Report what was not run.** `training: skipped` is a first-class outcome. A report
  that hides a skipped stage is half a report.

- **Say when the machine is the bottleneck.** A slow compile is usually one request at a
  time or a context reserved far larger than the work needs. Measure it, do not quote a
  number from someone else's hardware.

## Layout

```text
src/mekoy/
  search.py     candidates, staged pruning, Pareto, SLO stop
  reflect.py    reflective optimiser: read the failures, rewrite the brief
  verify.py     deterministic checks — the gate a candidate must clear
  score.py      field scoring
  tasks.py      task definitions: schema, prompt, gate, scorer
  compile.py    compile a System and produce its report
  doctor.py     check the local setup and price it against the hardware
  train.py      optional LoRA, optional from-scratch training
  api/          the local HTTP control plane
  mcp_server.py the MCP connector
examples/       labelled corpora, one directory per task
connectors/     per-assistant connection config
plugins/        packaged skill for agent harnesses
```

## Before you commit

```bash
uv run ruff check .          # lint
uv run ruff format --check src tests
uv run pytest -q             # the full suite
```

The suite must collect and pass **without** the optional extras installed. `dspy` and
`mlflow` are optional; import them lazily or guard the import, never at module level in
a way that breaks collection.
