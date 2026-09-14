"""Experiment tracking. The MLflow path is skipped when the extra is absent."""

import json
from pathlib import Path

import mlflow
import pytest

from mekoy.compile import Budget, CompileReport, SearchSpace, compile_system
from mekoy.dataset import ExampleRecord, load_examples, split_examples
from mekoy.tracking import (
    TrackedRun,
    available,
    record,
    run_from_report,
    to_jsonl,
    to_mlflow,
)

_FIXTURE = Path("examples/bucko-restaurant/examples.jsonl")
_ROW = (
    '{"restaurant":"Uchi","intent":"availability","status":"confirmed",'
    '"party_size":2,"when":"Friday","under_name":null,'
    '"evidence":"A table for two is open Friday.","booked":false}'
)


class _Echo:
    local: bool = True

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
        tail = user.rsplit("Text:\n", 1)[-1]
        return _ROW if tail else "{}"


def _report() -> CompileReport:
    rows = load_examples(_FIXTURE)
    split = split_examples(rows)
    return compile_system(_EchoFor(rows), split, SearchSpace.single(), Budget(trials=1))


class _EchoFor:
    local: bool = True

    def __init__(self, rows: tuple[ExampleRecord, ...]) -> None:
        self._by_text = {r.text: r.outcome.model_dump_json() for r in rows}

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
        tail = user.rsplit("Text:\n", 1)[-1].rsplit("\nJSON:", 1)[0]
        return self._by_text.get(tail, "{}")


def _run() -> TrackedRun:
    return TrackedRun(
        task="receipt",
        config="k=8 r=3 free default",
        k_shot=8,
        retries=3,
        constrained=False,
        prompt="default",
        trials=4,
        pruned=2,
        stopped_early=False,
        dev_quality=0.916,
        test_quality=0.738,
        strict_quality=0.729,
        schema_rate=0.938,
        cost_usd=0.0,
        latency_ms=29422.0,
        dev_n=16,
        test_n=16,
    )


def test_a_record_separates_params_from_metrics() -> None:
    """What was chosen versus what was measured. A ranker needs the split."""
    run = _run()
    assert set(run.params) == {
        "task",
        "config",
        "k_shot",
        "retries",
        "constrained",
        "prompt",
        "training",
    }
    assert "test_quality" in run.metrics
    assert "test_quality" not in run.params


def test_a_record_serialises_to_one_jsonl_line(tmp_path: Path) -> None:
    path = to_jsonl(tmp_path / "runs.jsonl", _run())
    line = path.read_text().strip()
    assert "\n" not in line
    row = json.loads(line)
    assert row["task"] == "receipt"
    assert row["test_quality"] == pytest.approx(0.738)
    assert row["stopped_early"] is False


def test_appending_keeps_earlier_runs(tmp_path: Path) -> None:
    """The point of tracking is comparison over time."""
    path = tmp_path / "runs.jsonl"
    _ = to_jsonl(path, _run())
    _ = to_jsonl(path, _run())
    assert len(path.read_text().strip().splitlines()) == 2


def test_jsonl_backend_needs_no_dependency(tmp_path: Path) -> None:
    where = record(_run(), log_path=tmp_path / "runs.jsonl", backend="jsonl")
    assert where.endswith("runs.jsonl")
    assert Path(where).is_file()


def test_auto_always_writes_the_local_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSONL line is durable; MLflow is the interface over it."""
    monkeypatch.setattr("mekoy.tracking.available", lambda: False)
    where = record(_run(), log_path=tmp_path / "runs.jsonl", backend="auto")
    assert where.endswith("runs.jsonl")
    assert (tmp_path / "runs.jsonl").is_file()


def test_auto_keeps_the_local_line_when_mlflow_is_present(
    tmp_path: Path,
) -> None:
    """The bug this exists for: auto wrote only MLflow and dropped `log_path`."""
    if not available():
        pytest.skip("mlflow extra not installed")
    log = tmp_path / "runs.jsonl"
    where = record(
        _run(),
        log_path=log,
        backend="auto",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    assert log.is_file()
    assert "mlflow:" in where


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown tracking backend"):
        record(_run(), backend="wandb")


def test_a_report_becomes_a_record() -> None:
    report = _report()
    run, artifact = run_from_report(report, task="restaurant", artifact_text="card")
    assert run.task == "restaurant"
    assert run.trials == report.tried
    assert run.test_n == len(report.test.scores)
    assert artifact == "card"


def test_availability_is_reported_not_assumed() -> None:
    assert isinstance(available(), bool)


@pytest.mark.skipif(not available(), reason="mlflow extra not installed")
def test_mlflow_records_params_and_metrics(tmp_path: Path) -> None:
    """Verified against a real store, not mocked: an unverified branch is worse
    than no branch."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = to_mlflow(_run(), artifact_text="winner card", tracking_uri=uri)
    assert run_id
    mlflow.set_tracking_uri(uri)
    frame = mlflow.search_runs(experiment_names=["mekoy-compiles"])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["params.task"] == "receipt"
    assert str(row["params.k_shot"]) == "8"
    assert float(row["metrics.test_quality"]) == pytest.approx(0.738)
