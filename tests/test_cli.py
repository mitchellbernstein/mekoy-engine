import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mekoy import cli
from mekoy.cli import app, main

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")
_RUNNER = CliRunner()


def _copy_examples(tmp_path: Path) -> Path:
    dest = tmp_path / "examples.jsonl"
    _ = dest.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def test_help_lists_commands() -> None:
    result = _RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("eval", "compile", "invoke", "serve"):
        assert name in result.stdout


def test_eval_without_approve_exits_2(tmp_path: Path) -> None:
    examples = _copy_examples(tmp_path)
    result = _RUNNER.invoke(app, ["eval", str(examples)])
    assert result.exit_code == 2
    assert not (tmp_path / ".eval-approved").is_file()


def test_eval_approve_writes_stamp(tmp_path: Path) -> None:
    examples = _copy_examples(tmp_path)
    result = _RUNNER.invoke(app, ["eval", str(examples), "--approve"])
    assert result.exit_code == 0
    assert (tmp_path / ".eval-approved").is_file()


def test_compile_without_eval_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = _copy_examples(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mekoy", "compile", str(examples)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_serve_runs_the_control_plane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`serve` exists and the invoke server behind it exists. It used to exit 2."""
    called: dict[str, object] = {}

    def _fake_run(app_path: str, **kwargs: object) -> None:
        called["app"] = app_path
        called.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", _fake_run)
    result = _RUNNER.invoke(app, ["serve", "--port", "9123"])
    assert result.exit_code == 0
    assert called["app"] == "mekoy.api.main:app"
    assert called["port"] == 9123


def test_serve_warns_when_the_api_is_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEKOY_API_KEY", raising=False)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)
    result = _RUNNER.invoke(app, ["serve"])
    assert "open" in result.stdout


def test_report_prints_the_compile_card(tmp_path: Path) -> None:
    """`report` exists; it was missing entirely."""
    examples = _copy_examples(tmp_path)
    (tmp_path / "compile-report.txt").write_text("winner  k=0 r=1\n")
    result = _RUNNER.invoke(app, ["report", str(examples)])
    assert result.exit_code == 0
    assert "winner  k=0 r=1" in result.stdout


def test_report_without_a_compile_says_so(tmp_path: Path) -> None:
    examples = _copy_examples(tmp_path)
    result = _RUNNER.invoke(app, ["report", str(examples)])
    assert result.exit_code == 1
    assert "no report at" in result.stdout


def test_main_eval_without_approve_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = _copy_examples(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mekoy", "eval", str(examples)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
