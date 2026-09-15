"""Downloadable System directory: spec, brief, examples, checks, report, and README.

A bundle is what a recipient keeps when they take a System away from here, so it has to
be complete. It previously carried a spec, a report, a compose file, and a README - no
brief, no labeled examples, and no checks - which made it a pointer to a model rather
than a System someone could run, understand, or check. The harness is the System, and a
harness without its examples and its checks is not one.
"""

from __future__ import annotations

import json
from pathlib import Path

from mekoy.compile import CompileReport
from mekoy.dataset import TaskExample
from mekoy.errors import CompileError
from mekoy.outcome import RestaurantOutcome
from mekoy.spec import SPEC_VERSION, OwnershipFlags, Slos, SystemSpec, write_spec

#: A downloaded System runs locally and can leave our hosts. PLAN §31.
_OWNED = OwnershipFlags(runtime_owned=True, downloadable=True)

#: Where the harness is installed from. A bundle tells a recipient to install something,
#: so the instruction has to point somewhere they can actually reach.
_REPO_URL = "https://github.com/mitchellbernstein/mekoy-engine"


def _outcome_json(outcome: object) -> object:
    """An example's label, as JSON, however it happens to be represented."""
    dump = getattr(outcome, "model_dump", None)
    return dump(mode="json") if callable(dump) else outcome


def _examples_text(examples: tuple[TaskExample, ...]) -> str:
    """The labeled rows, one JSON object per line.

    JSON Lines rather than something bespoke, because a recipient should be able to read
    it with any tool, and it is the shape the engine already loads.
    """
    return (
        "\n".join(
            json.dumps({"text": row.text, "outcome": _outcome_json(row.outcome)})
            for row in examples
        )
        + "\n"
    )


def _field_names(spec: SystemSpec) -> list[str]:
    """The fields an answer has to contain, from the schema that travels with it."""
    properties = spec.json_schema.get("properties")
    return list(properties.keys()) if isinstance(properties, dict) else []


def _checks_text(spec: SystemSpec) -> str:
    """What the engine checks before it accepts an answer, in plain words.

    The checks are the part of a System that says what "correct" means. A bundle that
    omits them leaves the recipient unable to tell a good answer from a bad one, and
    that is the difference between owning a System and owning a black box.
    """
    fields = _field_names(spec)
    listed = ", ".join(f"`{name}`" for name in fields) or "see the schema"
    return (
        f"# What this System checks\n\n"
        f"Quality gate: **{spec.slos.quality:.3f}** field accuracy, measured on "
        f"held-out\n"
        f"documents the search never read.\n\n"
        f"An answer has to satisfy all of these before it is accepted:\n\n"
        f"1. It parses as JSON matching `schema` in `spec.json`.\n"
        f"2. Every field the schema requires is present: {listed}.\n"
        f"3. Where the task has arithmetic, the numbers agree with each other.\n"
        f"4. Claims the answer makes are supported by the document it read.\n\n"
        f"A candidate that fails these is rejected rather than reported, and the\n"
        f"compile card in `report.txt` records the reasons it saw.\n"
    )


def _brief(spec: SystemSpec) -> str:
    """The job, in words, plus the honest limits of what travelled with it."""
    fields = _field_names(spec)
    return (
        f"# Job\n\n{spec.task}\n\n"
        f"## What it returns\n\n"
        f"One record per document, with these fields:\n\n"
        + "".join(f"- `{name}`\n" for name in fields)
        + f"\n## How it was built\n\n"
        f"The engine searched harness configurations and kept the winner. The chosen\n"
        f"harness shows **{spec.k_shot}** example(s) and makes **{spec.retries}**\n"
        f"repair attempt(s) per document, calling **{spec.model_id}**.\n\n"
        f"## What you own\n\n"
        f"You own the specialisation: this brief, the labeled examples, the checks,\n"
        f"and the measured score. The model underneath is not yours, and the checks\n"
        f"began as ours. Re-pointing the harness at another OpenAI-compatible\n"
        f"server is supported and worth testing on your own hardware, because the\n"
        f"measured quality came from the harness, not from a set of weights.\n"
    )


def spec_for(
    report: CompileReport,
    *,
    task: str,
    model_id: str,
) -> SystemSpec:
    """Build the portable spec from a finished compile.

    The whole winner's harness travels with the System, not a selection of it. This
    function used to hand-pick `k_shot` and `retries` and silently drop the other five
    axes, so the restaurant winner `k=4 r=0 schema strict` was exported as `k=4 r=0` and
    a recipient running the bundle got the **default** brief instead of the strict one.
    The score they were quoted was measured on a harness they did not receive.

    `spec_version` is raised here, and not merely carried, because that version is what
    tells a reader these axes are present rather than defaulted.
    """
    winner = report.winner
    config = winner.config
    slos = report.slos or Slos(
        quality=winner.quality,
        cost_per_doc=report.test.cost_usd,
        latency_ms=report.test.latency_ms,
    )
    return SystemSpec(
        spec_version=SPEC_VERSION,
        task=task or "restaurant call extraction",
        json_schema=RestaurantOutcome.model_json_schema(),
        slos=slos,
        model_id=model_id,
        k_shot=config.k_shot,
        retries=config.retries,
        constrained=config.constrained,
        prompt=config.prompt,
        schema_constrained=config.schema,
        consistency=config.consistency,
        bootstrap=config.bootstrap,
        ownership=_OWNED,
    )


def write_bundle(  # noqa: PLR0913 - a bundle names each artefact it writes
    directory: Path,
    spec: SystemSpec,
    report_text: str,
    *,
    examples: tuple[TaskExample, ...] = (),
    shots: tuple[TaskExample, ...] = (),
    holdout: tuple[TaskExample, ...] = (),
    run_log: str = "",
    environment: dict[str, object] | None = None,
) -> Path:
    """Write everything a recipient needs to run and check the System.

    A harness that shows examples needs those examples to be the same harness, and it
    needs them to be the same *rows*. `examples` is everything the compile was given;
    `shots` is the subset the harness actually shows, which is not the same thing. The
    compile scores its test split with `shots=split.train`, so the head of the TRAINING
    split is what the model saw, and the split is hash-dealt, so that is not the head of
    the file. Shipping the corpus and letting a reader take its first `k_shot` rows made
    a bundle whose own verification scored a different System: one row in eight in
    common on the restaurant corpus.
    """
    if spec.k_shot > 0 and not examples:
        msg = (
            f"this System shows k_shot={spec.k_shot} examples, so they are part of the "
            "harness, but none were supplied. A bundle without them would run a "
            "different System and report it as this one."
        )
        raise CompileError(message=msg)
    if directory.exists() and not directory.is_dir():
        msg = f"not a directory: {directory}"
        raise CompileError(message=msg)
    directory.mkdir(parents=True, exist_ok=True)
    _ = write_spec(directory / "spec.json", spec)
    report = report_text if report_text.endswith("\n") else f"{report_text}\n"
    _ = (directory / "report.txt").write_text(report, encoding="utf-8")
    _ = (directory / "brief.md").write_text(_brief(spec), encoding="utf-8")
    _ = (directory / "checks.md").write_text(_checks_text(spec), encoding="utf-8")
    if examples:
        _ = (directory / "examples.jsonl").write_text(
            _examples_text(examples), encoding="utf-8"
        )
    if shots:
        # Exactly the rows the harness shows, not the pool they were drawn from. A file
        # holding the whole training split would make a reader slice it themselves, and
        # slicing it wrongly is how this bundle verified as a different System.
        _ = (directory / "shots.jsonl").write_text(
            _examples_text(shots[: spec.k_shot]), encoding="utf-8"
        )
    # The held-out rows are what make the advertised score checkable rather than
    # asserted. Without them a recipient has our number and no way to test it, which is
    # the difference between evidence and a claim.
    if holdout:
        _ = (directory / "holdout.jsonl").write_text(
            _examples_text(holdout), encoding="utf-8"
        )
    if run_log:
        _ = (directory / "run-log.txt").write_text(run_log, encoding="utf-8")
    if environment:
        _ = (directory / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _ = (directory / "docker-compose.yml").write_text(_compose(), encoding="utf-8")
    _ = (directory / "README.md").write_text(
        _readme(spec, has_examples=bool(examples)), encoding="utf-8"
    )
    return directory


def read_shots(directory: Path) -> tuple[TaskExample, ...]:
    """The rows this System's harness shows, as recorded at export.

    Returns empty when the bundle predates the file, which a reader must treat as
    "cannot verify" rather than "shows nothing": the harness showed something, and not
    knowing what is not the same as knowing it showed nothing.
    """
    return _read_rows(directory / "shots.jsonl")


def _read_rows(path: Path) -> tuple[TaskExample, ...]:
    """Read a JSON Lines fixture into examples."""
    if not path.is_file():
        return ()
    rows: list[TaskExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = payload.get("text")
        if isinstance(text, str):
            rows.append(TaskExample(text=text, outcome=payload.get("outcome")))
    return tuple(rows)


def _compose() -> str:
    """The model server.

    The pull cannot live here - a compose file cannot pull for you - so the README gives
    it as a real step rather than leaving a recipient to notice a commented line. The
    image is pinned so a bundle keeps working.
    """
    return (
        "# Model server for this System.\n"
        "# The harness runs in the mekoy CLI; see README.md.\n"
        "services:\n"
        "  ollama:\n"
        "    image: ollama/ollama:latest\n"
        "    ports:\n"
        '      - "11434:11434"\n'
        "    volumes:\n"
        "      - ollama:/root/.ollama\n"
        "    environment:\n"
        "      OLLAMA_KEEP_ALIVE: 30m\n"
        "    # On a GPU box, uncomment to reserve the device. Left off so the same\n"
        "    # file runs on CPU: a hard nvidia reservation fails to start anywhere\n"
        "    # without the nvidia driver.\n"
        "    # deploy:\n"
        "    #   resources:\n"
        "    #     reservations:\n"
        "    #       devices:\n"
        "    #         - driver: nvidia\n"
        "    #           count: all\n"
        "    #           capabilities: [gpu]\n"
        "volumes:\n"
        "  ollama:\n"
    )


def _readme(spec: SystemSpec, *, has_examples: bool) -> str:
    """The recipient's instructions, in the order they will run them."""
    invoke = (
        f"mekoy invoke document.txt --model {spec.model_id} "
        f"--k-shot {spec.k_shot} --retries {spec.retries}"
    )
    examples_step = (
        "The labeled examples the harness shows are in `examples.jsonl`, and the\n"
        "configuration that was measured is in `spec.json`.\n"
        if has_examples
        else (
            "**This bundle carries no examples.** The compile used the ones the\n"
            "author supplied. Pass `--examples` pointing at your own file if the\n"
            "harness shows any, or the extraction will run without them.\n"
        )
    )
    return (
        f"# System\n\n"
        f"{spec.task}\n\n"
        f"## Run it\n\n"
        f"1. Start the model server and pull the model:\n\n"
        f"```\n"
        f"docker compose up -d\n"
        f"docker compose exec ollama ollama pull {spec.model_id}\n"
        f"```\n\n"
        f"2. Install the harness. The System is the harness, not the weights, so\n"
        f"   this is the part that has to be here:\n\n"
        f"```\n"
        f"git clone {_REPO_URL}\n"
        f"cd mekoy-engine && uv sync\n"
        f"```\n\n"
        f"3. Put a document in a file and extract:\n\n"
        f"```\n"
        f"{invoke}\n"
        f"```\n\n"
        f"{examples_step}\n"
        f"## What is in here\n\n"
        f"- `brief.md` — the job, the fields, and what was measured.\n"
        f"- `checks.md` — what an answer has to satisfy before it is accepted.\n"
        f"- `examples.jsonl` — every labeled row the compile was given.\n"
        f"- `shots.jsonl` — the rows the harness shows, which is what the score was\n"
        f"  measured using. A verifier needs these, not a guess at them.\n"
        f"- `spec.json` — schema, SLOs, ownership flags, the harness knobs, and\n"
        f"  `spec_version`.\n"
        f"- `report.txt` — the compile card: what was tried and what it scored.\n"
        f"- `holdout.jsonl` — the held-out rows the advertised score was measured\n"
        f"  on.\n"
        f"- `run-log.txt` — what the search tried, in order.\n"
        f"- `environment.json` — the model, server, and seed behind the number.\n"
        f"- `docker-compose.yml` — the model server.\n"
        f"\n"
        f"## Check the number yourself\n\n"
        f"The advertised score is not ours to assert. Recompute it from this\n"
        f"directory:\n\n"
        f"```\n"
        f"mekoy verify-bundle .\n"
        f"```\n\n"
        f"That re-runs the held-out rows through the same harness and scorer and\n"
        f"reports whether it agrees. It needs the model, so start the server first.\n"
        f"If it\n"
        f"disagrees, the number was not reproducible and you should not trust it.\n"
    )
