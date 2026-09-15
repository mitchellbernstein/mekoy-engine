"""Local CLI: eval gate, compile, invoke. No hosted control plane."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Annotated, assert_never

import typer
import uvicorn
from rich.console import Console

from mekoy.api.auth import API_KEY_ENV, settings_from_env
from mekoy.bakeoff import LaneSpec, format_bakeoff, run_bakeoff
from mekoy.banking77 import load_banking77
from mekoy.bootstrap import bootstrap_demos
from mekoy.compare import compare_cards, format_comparison
from mekoy.compile import Budget, SearchSpace, compile_system, format_report
from mekoy.cord import load_cord
from mekoy.dataset import load_examples, load_task_examples, split_examples
from mekoy.doctor import check, probe, render
from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.evalgen import audit, expand_scenarios
from mekoy.harness import extract
from mekoy.openai_runtime import ASTRA_MODEL, OpenAICompleter, api_key_from_env
from mekoy.outcome import RestaurantOutcome
from mekoy.paraphrase import generate, label_problems, write_corpus
from mekoy.receipt import load_receipt_examples, numeric_problems
from mekoy.runtime import Completer, OllamaCompleter
from mekoy.sroie import load_sroie
from mekoy.tasks import task_for_path
from mekoy.tracking import DEFAULT_LOG_NAME, record, run_from_report
from mekoy.verify import VerifyFail, VerifyOk, explain

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

_DEFAULT_URL = "http://127.0.0.1:11434/v1"
_DEFAULT_MODEL = "qwen2.5:7b"
#: Fewer than this and there is no train/dev/test to speak of.
_MIN_EVAL_ROWS = 3


def _boundary[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except (CompileError, ModelUnreachableError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    return wrapped


@app.command("eval")
@_boundary
def evaluate(
    examples: Annotated[Path, typer.Argument(help="examples.jsonl")],
    approve: Annotated[
        bool,
        typer.Option("--approve", help="Record that these examples are the eval."),
    ] = False,
) -> None:
    """Load examples and require an explicit eval yes before compile."""
    task = task_for_path(examples)
    rows: tuple[object, ...] = (
        load_task_examples(examples, task)
        if task.name != "restaurant"
        else load_examples(examples)
    )
    split = split_examples(rows)  # type: ignore[arg-type]
    summary = (
        f"task: {task.name}\n"
        f"{len(rows)} examples -> {len(split.train)} train / "
        f"{len(split.dev)} dev / {len(split.test)} test"
    )
    console.print(summary)
    console.print(
        "compile selects on dev; the test number is scored once",
    )
    if task.name != "restaurant":
        console.print(f"scored fields: {' '.join(task.scored_fields)}")
    if not approve:
        console.print("Eval is not approved. Re-run with --approve to unlock compile.")
        raise typer.Exit(code=2)
    stamp = examples.parent / ".eval-approved"
    _ = stamp.write_text("approved\n", encoding="utf-8")
    console.print(f"Eval approved: {stamp}")


@app.command("compile")
@_boundary
def compile_cmd(
    examples: Annotated[Path, typer.Argument(help="examples.jsonl")],
    model: Annotated[str, typer.Option(help="Open-weight model id")] = _DEFAULT_MODEL,
    base_url: Annotated[
        str, typer.Option(help="OpenAI-compatible base URL")
    ] = _DEFAULT_URL,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="One candidate: k=0, retries=1."),
    ] = False,
    trials: Annotated[int, typer.Option(help="Max candidates to try on dev.")] = 5,
    bootstrap: Annotated[
        bool,
        typer.Option(
            "--bootstrap",
            help="Add a candidate whose shots are chosen by BootstrapFewShot.",
        ),
    ] = False,
    track: Annotated[
        bool,
        typer.Option(
            "--track",
            help="Record this compile for cross-run comparison.",
        ),
    ] = False,
    also_model: Annotated[
        list[str] | None,
        typer.Option(
            "--also-model",
            help="Additional base model to search over (repeatable).",
        ),
    ] = None,
) -> None:
    """Search the harness space on dev, then measure the winner on test."""
    _require_eval(examples)
    task = task_for_path(examples)
    rows: tuple[object, ...] = (
        load_task_examples(examples, task)
        if task.name != "restaurant"
        else load_examples(examples)
    )
    split = split_examples(rows)  # type: ignore[arg-type]
    space = (
        SearchSpace.single()
        if quick
        else SearchSpace.for_task(task, train_n=len(split.train))
    )
    console.print(f"task: {task.name}")
    extra = tuple(also_model or ())
    if extra:
        # The search samples over several open models; it cannot pick one
        # it has no completer for, so each is registered here.
        space = replace(space, models=("", *extra))
        console.print(f"models searched: {', '.join(('default', *extra))}")
    if bootstrap:
        space = replace(space, bootstrap=(False, True))
        stage0 = bootstrap_demos(
            task,
            trainset=_pairs(split.train),
            model_id=model,
            base_url=base_url,
        )
        console.print(f"Stage 0 bootstrap kept {len(stage0)} demonstration(s)")
        bootstrapped = stage0
    else:
        bootstrapped = ()
    completer = OllamaCompleter(base_url=base_url, model=model)
    models = {name: OllamaCompleter(base_url=base_url, model=name) for name in extra}
    report = compile_system(
        completer,
        split,
        space,
        Budget(trials=trials),
        task=task,
        bootstrapped=bootstrapped,
        models=models,
    )
    text = format_report(report)
    console.print(text)
    out = examples.parent / "compile-report.txt"
    _ = out.write_text(text + "\n", encoding="utf-8")
    console.print(f"wrote {out}")
    if track:
        tracked, artifact = run_from_report(report, task=task.name, artifact_text=text)
        where = record(
            tracked,
            artifact_text=artifact,
            log_path=examples.parent / DEFAULT_LOG_NAME,
        )
        console.print(f"tracked: {where}")


@app.command("invoke")
@_boundary
def invoke_cmd(
    text_file: Annotated[Path, typer.Argument(help="Call transcript text file")],
    examples: Annotated[
        Path | None,
        typer.Option(help="examples.jsonl for few-shot"),
    ] = None,
    model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
    base_url: Annotated[str, typer.Option()] = _DEFAULT_URL,
    k_shot: Annotated[int, typer.Option()] = 0,
    retries: Annotated[int, typer.Option()] = 1,
) -> None:
    """Extract one document through the local harness."""
    text = text_file.read_text(encoding="utf-8")
    shots: tuple[tuple[str, RestaurantOutcome], ...] = ()
    if examples is not None and k_shot > 0:
        rows = load_examples(examples)
        shots = tuple((row.text, row.outcome) for row in rows[:k_shot])
    completer = OllamaCompleter(base_url=base_url, model=model)
    result = extract(completer, text=text, shots=shots, retries=retries)
    match result:
        case VerifyOk(outcome=outcome):
            # Plain stdout, not the rich console: rich wraps long lines, which
            # makes the output unparseable when piped.
            sys.stdout.write(outcome.model_dump_json(indent=2) + "\n")
        case VerifyFail():
            console.print(f"verify failed: {explain(result)}")
            raise typer.Exit(code=1)
        case _ as unreachable:
            assert_never(unreachable)


@app.command("bakeoff")
@_boundary
def bakeoff_cmd(
    examples: Annotated[Path, typer.Argument(help="examples.jsonl")],
    model: Annotated[str, typer.Option(help="Open-weight challenger")] = _DEFAULT_MODEL,
    base_url: Annotated[str, typer.Option()] = _DEFAULT_URL,
    baseline: Annotated[str, typer.Option(help="OpenAI model id")] = ASTRA_MODEL,
    baseline_model: Annotated[
        str | None,
        typer.Option(
            "--baseline-model",
            help="Run the baseline locally instead of against a closed API.",
        ),
    ] = None,
    quick: Annotated[bool, typer.Option("--quick")] = False,
    trials: Annotated[int, typer.Option(help="Max candidates on dev.")] = 5,
) -> None:
    """Same test split: a strong baseline one-shot vs the compiled System."""
    _require_eval(examples)
    split = split_examples(load_examples(examples))
    if baseline_model is not None:
        # A local baseline keeps the whole bake-off runnable with no key and no
        # data leaving the machine.
        base_completer: Completer = OllamaCompleter(
            base_url=base_url, model=baseline_model
        )
        baseline_name = baseline_model
    else:
        base_completer = OpenAICompleter(api_key=api_key_from_env(), model=baseline)
        baseline_name = baseline
    local = OllamaCompleter(base_url=base_url, model=model)
    space = (
        SearchSpace.single() if quick else SearchSpace.local(train_n=len(split.train))
    )
    result = run_bakeoff(
        split=split,
        baseline=LaneSpec(name=baseline_name, completer=base_completer),
        challenger=LaneSpec(name=model, completer=local),
        space=space,
        budget=Budget(trials=trials),
    )
    text = format_bakeoff(result)
    console.print(text)
    out = examples.parent / "bakeoff-report.txt"
    _ = out.write_text(text + "\n", encoding="utf-8")
    console.print(f"wrote {out}")


@app.command("ingest")
@_boundary
def ingest_cmd(
    texts: Annotated[Path, typer.Argument(help="JSONL with a 'text' field")],
    model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
    base_url: Annotated[str, typer.Option()] = _DEFAULT_URL,
    k_shot: Annotated[int, typer.Option()] = 2,
    shots_from: Annotated[
        Path | None, typer.Option(help="approved examples.jsonl for shots")
    ] = None,
) -> None:
    """Draft labels for raw transcripts. A human still approves them."""
    rows = _read_texts(texts)
    shots: tuple[tuple[str, RestaurantOutcome], ...] = ()
    if shots_from is not None and k_shot > 0:
        approved = load_examples(shots_from)
        shots = tuple((r.text, r.outcome) for r in approved[:k_shot])
    completer = OllamaCompleter(base_url=base_url, model=model)
    out_rows: list[str] = []
    rejected = 0
    for text in rows:
        result = extract(completer, text=text, shots=shots, retries=1)
        match result:
            case VerifyOk(outcome=outcome):
                out_rows.append(
                    json.dumps(
                        {"text": text, "outcome": outcome.model_dump()},
                        ensure_ascii=False,
                    )
                )
            case VerifyFail():
                rejected += 1
            case _ as unreachable:
                assert_never(unreachable)
    out = texts.with_name(texts.stem + ".draft.jsonl")
    _ = out.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    console.print(f"wrote {out} ({len(out_rows)} drafts, {rejected} rejected)")
    console.print(
        "[yellow]These are model drafts, not gold. Read every row before \
using it as an eval.[/yellow]"
    )


def _pairs(rows: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    """(text, gold) pairs from any task's rows, for the Stage 0 bootstrap."""
    return tuple((str(r.text), r.outcome) for r in rows)  # type: ignore[attr-defined]


def _read_texts(path: Path) -> tuple[str, ...]:
    """Read raw transcripts from JSONL ('text' key) or a plain text file."""
    raw = path.read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            out.append(line.strip())
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
            out.append(parsed["text"])
    if not out:
        msg = f"no transcripts found in {path}"
        raise CompileError(message=msg)
    return tuple(out)


@app.command("compare")
@_boundary
def compare_cmd(
    left: Annotated[Path, typer.Argument(help="first compile-report.txt")],
    right: Annotated[Path, typer.Argument(help="second compile-report.txt")],
) -> None:
    """Compare two compile reports on their measured test numbers."""
    cmp = compare_cards(
        left.stem,
        left.read_text(encoding="utf-8"),
        right.stem,
        right.read_text(encoding="utf-8"),
    )
    console.print(format_comparison(cmp))


@app.command("gen-eval")
@_boundary
def gen_eval_cmd(
    out: Annotated[Path, typer.Argument(help="where to write examples.jsonl")],
    target: Annotated[int, typer.Option(help="How many scenarios to render.")] = 240,
    model: Annotated[str, typer.Option()] = _DEFAULT_MODEL,
    base_url: Annotated[str, typer.Option()] = _DEFAULT_URL,
    attempts: Annotated[int, typer.Option(help="Render attempts per case.")] = 2,
) -> None:
    """Render evaluated edge cases. Gold comes from the scenario, not the model."""
    scenarios = expand_scenarios(target=target)
    problems = audit(scenarios)
    if problems:
        msg = "scenario table violates the gate:\n" + "\n".join(problems)
        raise CompileError(message=msg)
    completer = OllamaCompleter(base_url=base_url, model=model)
    accepted, rejected = generate(completer, scenarios, attempts=attempts)
    text = write_corpus(out, accepted)
    console.print(
        f"wrote {text} ({len(accepted)} accepted, {len(rejected)} rejected "
        f"of {len(scenarios)})"
    )
    if rejected:
        console.print("[yellow]rejected on fact checks:[/yellow]")
        for scenario, reasons in rejected[:5]:
            console.print(f"  {scenario.restaurant}: {'; '.join(reasons)}")
    if len(accepted) < _MIN_EVAL_ROWS:
        msg = "too few accepted rows to form an eval"
        raise CompileError(message=msg)


@app.command("load-cord")
@_boundary
def load_cord_cmd(
    out: Annotated[Path, typer.Argument(help="where to write the corpus")],
    split: Annotated[str, typer.Option(help="CORD split to fetch")] = "test",
    limit: Annotated[int, typer.Option(help="How many rows to fetch.")] = 100,
    keep_unsound: Annotated[
        bool,
        typer.Option(
            "--keep-unsound",
            help="Keep gold rows that break their own arithmetic.",
        ),
    ] = False,
) -> None:
    """Fetch CORD receipts as text-only examples. Real data, not a fixture."""
    path, kept, dropped = load_cord(
        out, split=split, limit=limit, sound_only=not keep_unsound
    )
    console.print(f"wrote {path} ({kept} rows)")
    if dropped:
        console.print(
            f"[yellow]dropped {dropped} CORD rows whose gold breaks a "
            f"deterministic arithmetic identity[/yellow]"
        )


@app.command("load-sroie")
@_boundary
def load_sroie_cmd(
    out: Annotated[Path, typer.Argument(help="where to write the corpus")],
    limit: Annotated[int, typer.Option(help="How many receipts to fetch.")] = 120,
    workers: Annotated[int, typer.Option(help="Concurrent fetches.")] = 8,
) -> None:
    """Fetch SROIE receipts as text-only examples. Four annotated fields each."""
    path, kept, dropped, unreachable = load_sroie(out, limit=limit, workers=workers)
    console.print(f"wrote {path} ({kept} rows)")
    if dropped:
        console.print(
            f"[yellow]dropped {dropped} rows whose gold breaks the "
            f"arithmetic gate[/yellow]"
        )
    if unreachable:
        console.print(f"[yellow]{unreachable} receipts could not be fetched[/yellow]")


@app.command("load-banking77")
@_boundary
def load_banking77_cmd(
    out: Annotated[Path, typer.Argument(help="where to write the corpus")],
    limit: Annotated[int, typer.Option(help="How many utterances to fetch.")] = 600,
) -> None:
    """Fetch BANKING77 utterances. Classification, not extraction."""
    path, kept, skipped = load_banking77(out, limit=limit)
    console.print(f"wrote {path} ({kept} rows, {skipped} dropped)")


@app.command("check-eval")
@_boundary
def check_eval_cmd(
    examples: Annotated[Path, typer.Argument(help="examples.jsonl")],
    limit: Annotated[int, typer.Option(help="How many failures to print.")] = 5,
) -> None:
    """Audit whether each transcript supports its own label.

    Catches the eval defect that costs the most and shows the least: a row whose
    text never says the thing its label claims. Scoring a System against those is
    unfair, and the unfair rows are invisible without a check like this.
    """
    if _is_receipt_fixture(examples):
        _check_receipts(examples, limit)
        return
    rows = load_examples(examples)
    problems = [(row, label_problems(row.outcome, row.text)) for row in rows]
    bad = [(row, why) for row, why in problems if why]
    console.print(
        f"{len(rows)} rows: {len(rows) - len(bad)} support their label, "
        f"{len(bad)} do not"
    )
    if bad:
        console.print(
            f"[yellow]unsupportable rows (showing {min(limit, len(bad))}):[/yellow]"
        )
        for row, why in bad[:limit]:
            console.print(
                f"  {row.outcome.restaurant} [{row.outcome.intent}]: {'; '.join(why)}"
            )
            console.print(f"    {row.text[:110]}")


def _is_receipt_fixture(path: Path) -> bool:
    """Receipts carry a `receipt` key; restaurant rows carry `outcome`."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(row, dict) and "receipt" in row
    return False


def _check_receipts(examples: Path, limit: int) -> None:
    """Audit a receipt fixture against its own arithmetic."""
    rows = load_receipt_examples(examples)
    checked = [(row, numeric_problems(row.receipt)) for row in rows]
    bad = [(row, why) for row, why in checked if why]
    console.print(
        f"{len(rows)} receipts: {len(rows) - len(bad)} arithmetically sound, "
        f"{len(bad)} not"
    )
    if bad:
        console.print(
            f"[yellow]unsound receipts (showing {min(limit, len(bad))}):[/yellow]"
        )
        for row, why in bad[:limit]:
            console.print(f"  {row.receipt.merchant}: {'; '.join(why)}")


@app.command("report")
@_boundary
def report_cmd(
    examples: Annotated[Path, typer.Argument(help="examples.jsonl")],
    markdown: Annotated[
        bool,
        typer.Option("--markdown", help="Show the markdown card instead."),
    ] = False,
) -> None:
    """Print the compile report for a corpus.

    The API exposes GET /v1/systems/:id/report and the CLI has a `report`
    CLI command. This reads the card the compile wrote next to the corpus, so a
    report is readable without the API running.
    """
    name = "compile-report.md" if markdown else "compile-report.txt"
    path = examples.parent / name
    if not path.is_file():
        msg = f"no report at {path}; run `mekoy compile {examples}` first"
        raise CompileError(message=msg)
    console.print(path.read_text(encoding="utf-8").rstrip())


@app.command("serve")
@_boundary
def serve_cmd(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8787,
    reload: Annotated[bool, typer.Option("--reload", help="Dev auto-reload.")] = False,
) -> None:
    """Run the HTTP control plane.

    An OpenAI-compatible invoke server is the goal, reached through `serve`. This
    used to print "not in local MVP" while the app it should serve already existed
    at `mekoy.api.main` — it does not host anything for a customer, which is a
    different thing from not existing.
    """
    console.print(
        f"serving http://{host}:{port}  (OpenAPI at http://{host}:{port}/docs)"
    )
    if not settings_from_env().enabled:
        console.print(
            f"[yellow]/v1 and /mcp are open; set {API_KEY_ENV} to require a "
            f"bearer key[/yellow]"
        )
    uvicorn.run("mekoy.api.main:app", host=host, port=port, reload=reload)


def _require_eval(examples: Path) -> None:
    stamp = examples.parent / ".eval-approved"
    if not stamp.is_file():
        msg = "eval is not approved; run: mekoy eval --approve"
        raise CompileError(message=msg)


@app.command()
def doctor(
    measure: bool = False,
) -> None:
    """Check the local setup and say what is slowing it down."""
    typer.echo(render(check()))
    if measure:
        typer.echo("\nmeasuring concurrency on this machine (a few dozen short calls):")
        result = probe()
        typer.echo(
            f"  {result.explain()}"
            if result
            else "  could not measure; is the server up?"
        )


def main() -> None:
    """CLI entry. CompileError is mapped at each command, then Click exits."""
    app()


if __name__ == "__main__":
    main()
