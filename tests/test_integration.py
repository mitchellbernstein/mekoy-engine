"""Golden path: approve eval -> compile -> invoke, through the MCP server."""

import json
from pathlib import Path

from typer.testing import CliRunner

from mekoy.cli import app
from mekoy.dataset import ExampleRecord, load_examples
from mekoy.mcp_server import Server

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")


class _GoldEcho:
    local: bool = True

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text = {r.text: r.outcome.model_dump_json() for r in rows}
        self.prompts: list[str] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        del system, constrained, temperature, schema
        self.prompts.append(user)
        tail = user.rsplit("Text:\n", 1)[-1]
        return self._by_text.get(tail.rsplit("\nJSON:", 1)[0], "{}")


def _call(server: Server, name: str, arguments: dict[str, object]) -> str:
    reply = server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
    )
    assert reply is not None
    return reply


def _fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "examples.jsonl"
    _ = dest.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def test_golden_path_writes_a_report_and_replays_the_winner(tmp_path: Path) -> None:
    examples = _fixture(tmp_path)
    rows = load_examples(examples)
    completer = _GoldEcho(rows)
    server = Server(completer)

    inspected = _call(
        server, "inspect_task", {"job": "restaurant call", "examples": str(examples)}
    )
    assert "what counts as unacceptable" in inspected
    assert "dev" in inspected
    assert "test" in inspected

    proposed = _call(
        server, "propose_eval", {"examples": str(examples), "approve": True}
    )
    assert "Eval approved" in proposed
    assert "selection: dev" in proposed
    assert (tmp_path / ".eval-approved").is_file()

    compiled = _call(server, "compile_system", {"examples": str(examples)})
    assert "winner" in compiled
    assert "training: skipped" in compiled
    assert (tmp_path / "compile-report.txt").is_file()
    md = tmp_path / "compile-report.md"
    assert md.is_file()
    body = md.read_text()
    assert "## Pareto front" in body
    assert "## Every arm measured" in body

    listed = _call(server, "list_systems", {})
    assert "approved=True" in listed

    prompts_before = len(completer.prompts)
    invoked = _call(
        server,
        "invoke_system",
        {"text": rows[0].text, "examples": str(examples)},
    )
    assert "booked" in invoked, invoked
    # Invoke must run the same harness the compile selected, not defaults.
    assert len(completer.prompts) > prompts_before


def test_report_includes_test_and_dev_separately(tmp_path: Path) -> None:
    examples = _fixture(tmp_path)
    server = Server(_GoldEcho(load_examples(examples)))
    _ = _call(server, "propose_eval", {"examples": str(examples), "approve": True})
    report = _call(server, "compile_system", {"examples": str(examples)})
    assert "dev     quality=" in report
    assert "test    quality=" in report
    assert "scored once" not in report  # that belongs on the bakeoff card


def test_download_bundle_is_written_after_a_compile(tmp_path: Path) -> None:
    examples = _fixture(tmp_path)
    server = Server(_GoldEcho(load_examples(examples)))
    _ = _call(server, "propose_eval", {"examples": str(examples), "approve": True})
    _ = _call(server, "compile_system", {"examples": str(examples)})
    reply = _call(
        server, "deploy_system", {"examples": str(examples), "mode": "download"}
    )
    assert "bundle written" in reply
    bundle = tmp_path / "bundle"
    for name in ("spec.json", "report.txt", "README.md", "docker-compose.yml"):
        assert (bundle / name).is_file(), name
    compose = (bundle / "docker-compose.yml").read_text()
    assert "ollama/ollama" in compose
    assert "11434:11434" in compose
    # The README must not tell the reader to run a build the bundle cannot do.
    readme = (bundle / "README.md").read_text()
    assert "docker compose up -d" in readme
    assert "docker build -t system" not in readme
    spec = json.loads((bundle / "spec.json").read_text())
    assert spec["ownership"]["downloadable"] is True
    assert spec["ownership"]["runtime_owned"] is True
    assert "restaurant" in spec["task"]
    assert "properties" in spec["schema"]


def test_download_before_a_compile_is_refused(tmp_path: Path) -> None:
    examples = _fixture(tmp_path)
    server = Server(_GoldEcho(load_examples(examples)))
    _ = _call(server, "propose_eval", {"examples": str(examples), "approve": True})
    reply = _call(
        server, "deploy_system", {"examples": str(examples), "mode": "download"}
    )
    assert "no compile yet" in reply


def test_cli_compile_and_eval_run_without_a_model(tmp_path: Path) -> None:
    """The CLI must wire Budget correctly; the library tests cannot see this."""
    examples = _fixture(tmp_path)
    runner = CliRunner()
    approved = runner.invoke(app, ["eval", str(examples), "--approve"])
    assert approved.exit_code == 0, approved.output
    assert (tmp_path / ".eval-approved").is_file()
    assert "dev" in approved.output
    assert "test" in approved.output


def test_cli_compare_and_gen_eval_are_wired() -> None:
    runner = CliRunner()
    for command in ("gen-eval", "compare", "ingest", "bakeoff", "compile", "invoke"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{command}: {result.output}"
