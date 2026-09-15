"""Downloadable System directory: spec, report, compose, and invoke README."""

from __future__ import annotations

from pathlib import Path

from mekoy.compile import CompileReport
from mekoy.errors import CompileError
from mekoy.outcome import RestaurantOutcome
from mekoy.spec import OwnershipFlags, Slos, SystemSpec, write_spec

#: A downloaded System runs locally and can leave the machine that built it.
_OWNED = OwnershipFlags(runtime_owned=True, downloadable=True)


def spec_for(
    report: CompileReport,
    *,
    task: str,
    model_id: str,
) -> SystemSpec:
    """Build the portable spec from a finished compile.

    The winner's k-shot, retries, and SLOs travel with the System, so a download
    reproduces the harness that was measured rather than a default.
    """
    winner = report.winner
    slos = report.slos or Slos(
        quality=winner.quality,
        cost_per_doc=report.test.cost_usd,
        latency_ms=report.test.latency_ms,
    )
    return SystemSpec(
        task=task or "restaurant call extraction",
        json_schema=RestaurantOutcome.model_json_schema(),
        slos=slos,
        model_id=model_id,
        k_shot=winner.config.k_shot,
        retries=winner.config.retries,
        ownership=_OWNED,
    )


def write_bundle(directory: Path, spec: SystemSpec, report_text: str) -> Path:
    """Write spec.json, report.txt, docker-compose.yml, and README.md."""
    if directory.exists() and not directory.is_dir():
        msg = f"not a directory: {directory}"
        raise CompileError(message=msg)
    directory.mkdir(parents=True, exist_ok=True)
    _ = write_spec(directory / "spec.json", spec)
    report = report_text if report_text.endswith("\n") else f"{report_text}\n"
    _ = (directory / "report.txt").write_text(report, encoding="utf-8")
    _ = (directory / "docker-compose.yml").write_text(_compose(spec), encoding="utf-8")
    _ = (directory / "README.md").write_text(_readme(spec), encoding="utf-8")
    return directory


def _compose(spec: SystemSpec) -> str:
    """Model server only.

    The self-host bar is a compose file that works on a GPU box. The
    System is the harness, not the weights, so the bundle ships the server the
    harness calls and the spec that says which model it wants.
    """
    return (
        "# Model server for this System.\n"
        "# The harness itself runs in the mekoy CLI, not in this container.\n"
        "services:\n"
        "  ollama:\n"
        "    image: ollama/ollama\n"
        "    ports:\n"
        '      - "11434:11434"\n'
        "    volumes:\n"
        "      - ollama:/root/.ollama\n"
        f"    # Pull the compiled model on first start:\n"
        f"    #   docker compose exec ollama ollama pull {spec.model_id}\n"
        "    #\n"
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


def _readme(spec: SystemSpec) -> str:
    invoke = (
        f"mekoy invoke document.txt --model {spec.model_id} "
        f"--k-shot {spec.k_shot} --retries {spec.retries}"
    )
    return (
        f"# System\n\n"
        f"{spec.task}\n\n"
        f"## Run it\n\n"
        f"1. Start the model server:\n\n"
        f"```\n"
        f"docker compose up -d\n"
        f"docker compose exec ollama ollama pull {spec.model_id}\n"
        f"```\n\n"
        f"2. Install the harness (the compiled System is the harness, not the\n"
        f"   weights):\n\n"
        f"```\n"
        f"uv tool install mekoy   # or: pip install -e <repo>\n"
        f"```\n\n"
        f"3. Put a document in a file and extract:\n\n"
        f"```\n"
        f"{invoke}\n"
        f"```\n\n"
        f"To build the all-in-one image instead of using compose, use the\n"
        f"repository Dockerfile, which serves Ollama and the CLI together.\n\n"
        f"## What is in here\n\n"
        f"- `spec.json` — schema, SLOs, ownership flags, and the harness knobs the\n"
        f"  compile selected.\n"
        f"- `report.txt` — the compile card: what was tried and what it scored.\n"
        f"- `docker-compose.yml` — the model server.\n"
    )
