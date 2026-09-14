# AGENTS.md

This repo is an **AI System Compiler**. You compile owned **Systems** for one job. You do not build a generic coding agent, a document-AI company, or a GPU cloud.

**Precedence:** [`PLAN.md`](PLAN.md) §0. If this file fights §0, §0 wins.

## Vocabulary

| Word | Meaning |
|---|---|
| **Compiler** | Us. Hidden search over model + harness + eval + runtime. |
| **System** | The artifact we emit, host, sell, and route. |
| **Researcher** | Internal search policy (ASHA/Hyperband + GEPA). Not a chat persona. Not a paper-reader in Phase I. |
| **Lab model** | Large **open-weight** model used at compile time (evals, reflection, optional open-to-open distill). |
| **Orchestrator** | Phase III router across **listed Systems** (jobs). Not “which chat model is smartest.” |

Never call the artifact a compiler, a worker, or a model card.

## What you are building (Phase I)

A user (usually inside Claude / Claude Code / Codex / ChatGPT / Grok) describes a job and gives examples. You:

1. Propose an eval. **Do not search until they approve the checks.**
2. Search: open models, prompts, constrained decode, verify/retry, optional LoRA.
3. Emit a System they can **host on our API, self-host, or download**.

First proof job: closed-schema extract-and-verify from **already-text** (receipt / simple invoice). Not PDFs-from-pixels.

Ship together: Researcher (hidden), hosted invoke, download bundle, CLI + API, **one** agent connector.

## What you are not building now

Marketplace, payouts, catalog orchestrator, extra connectors after the first, OCR/layout, FX or Pi as the production runtime, GRPO/RL, a 50-technique registry UI, an agent that browses arXiv, owning GPUs, Rails, a generic coding-agent product.

## Hard bans

- **No closed-model distillation.** Do not put GPT / Claude / Gemini / other closed-API outputs in SFT, DPO, RL, or LoRA data. Contract risk; providers fingerprint it. Allowed compile data: customer gold, customer prod traces, synthetic from **open** models whose licenses allow it, deterministic checks. Closed APIs only as an **opt-in score baseline** or optional 2% runtime tail.
- **No FX / Pi production harness.** Steal ideas (checkpoints, tool-schema budgets). Production is a schema → decode → validate → retry/verify loop.
- **No Rails.** Compiler and control plane are Python.
- **No search without an approved eval.**
- **No publish default-on.** Catalog is Phase III; twice-confirmed opt-in; our eval badges, not self-report.

## Stack

| Layer | Use |
|---|---|
| Compiler / API | Python 3.12+, uv, ruff, pytest, FastAPI |
| Inner opt | DSPy + GEPA (MIPROv2 fallback). DSPy is IR, not the product. |
| Serving | vLLM + structured outputs (xgrammar). Ollama/MLX for local/dev. |
| Train (if needed) | Fireworks LoRA SFT; Together backup. One method in v1. |
| Eval | Promptfoo + deterministic schema/numeric checks |
| Experiments | MLflow |
| Web (optional) | Next.js App Router + Tailwind + shadcn |
| Control plane host | Fly.io |
| GPU jobs | Fireworks / Modal / Together — rented |

Layout when code exists:

```text
compiler/   spec, eval, program, search, techniques, runtime, providers, cli, api, connectors
web/        optional Next.js
examples/   first fixture: cord-receipt (text only)
```

## How to work

- Eval-first. Generate checks from the schema + examples; user says yes; then compile.
- Harness is model + deterministic policy (verify, retries, call limits). Prompt does not own totals or eval approval. See `.planning/research/harness-langchain-2026-06-03.md`.
- All options on the table **inside the Researcher**. The user never names LoRA or GEPA.
- Training is first-class and skippable. Report when it was not run.
- Privacy slider is real: default compile and runtime stay on open weights we (or they) run.
- Verify against real APIs and real eval sets. No mock-as-done.
- Apple signing, if it ever appears: Studio Yeehaw LLC, team `YQJ7BM8326`, prefer `com.studioyeehaw.*`.

## Connector tools (user-facing)

`inspect_task`, `propose_eval`, `compile_system`, `get_compile_status`, `get_report`, `deploy_system` (`hosted` | `self_host` | `download`), `invoke_system`, `list_systems`.

Never expose `train_model`, provider keys, LoRA rank, or GEPA budgets.

Phase III (not now): `publish_system`, `run` with `system` / `set` / `allowlist` / `denylist`.
