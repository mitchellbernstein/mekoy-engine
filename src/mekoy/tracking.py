"""Experiment tracking: one record per compile, comparable across runs.

MLflow is the experiments layer. What
it buys is the thing a single report card cannot: comparing compiles over time and
across task classes, which is what a catalog ranker would eventually read.

Two decisions:

- **Optional dependency.** MLflow pulls 51 packages. It lives in the `mlflow`
  extra, imported lazily, so a core install pays nothing for it.
- **A dependency-free fallback that is not a stub.** When MLflow is absent the same
  record is appended to a JSONL file, so tracking works with no install. The two
  backends write the same fields; MLflow adds a UI, not information.

The record is deliberately flat and fully typed. A tracking layer that logs a
blob is a log file; this one can be compared column-wise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mekoy.compile import CompileReport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

__all__ = ["TrackedRun", "available", "record", "to_jsonl", "to_mlflow"]

#: Default log when no path is given. Beside the corpus it describes.
DEFAULT_LOG_NAME = "compile-runs.jsonl"
#: MLflow experiment name. One experiment keeps every task class comparable.
EXPERIMENT = "mekoy-compiles"


def available() -> bool:
    """Whether the optional MLflow dependency is installed."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class TrackedRun:
    """Everything worth comparing about one compile."""

    task: str
    config: str
    k_shot: int
    retries: int
    constrained: bool
    prompt: str
    trials: int
    pruned: int
    stopped_early: bool
    dev_quality: float
    test_quality: float
    strict_quality: float
    schema_rate: float
    cost_usd: float
    latency_ms: float
    dev_n: int
    test_n: int
    training: str = "skipped"
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def params(self) -> dict[str, object]:
        """What was chosen. MLflow treats these as run parameters."""
        return {
            "task": self.task,
            "config": self.config,
            "k_shot": self.k_shot,
            "retries": self.retries,
            "constrained": self.constrained,
            "prompt": self.prompt,
            "training": self.training,
        }

    @property
    def metrics(self) -> dict[str, float]:
        """What was measured. MLflow treats these as run metrics."""
        return {
            "dev_quality": self.dev_quality,
            "test_quality": self.test_quality,
            "strict_quality": self.strict_quality,
            "schema_rate": self.schema_rate,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "trials": float(self.trials),
            "pruned": float(self.pruned),
            "dev_n": float(self.dev_n),
            "test_n": float(self.test_n),
        }

    def as_line(self) -> str:
        """One JSONL row, params and metrics flattened for column-wise reading."""
        payload: dict[str, Any] = {
            "recorded_at": self.recorded_at,
            **self.params,
            **self.metrics,
            "stopped_early": self.stopped_early,
        }
        return json.dumps(payload, ensure_ascii=False)


def run_from_report(
    report: CompileReport, *, task: str, artifact_text: str = ""
) -> tuple[TrackedRun, str]:
    """Build a record from a compile, plus the report text to keep as an artifact."""
    winner, test = report.winner, report.test
    return (
        TrackedRun(
            task=task,
            config=winner.config.label,
            k_shot=winner.config.k_shot,
            retries=winner.config.retries,
            constrained=winner.config.constrained,
            prompt=winner.config.prompt,
            trials=report.tried,
            pruned=report.pruned,
            stopped_early=report.stopped_early,
            dev_quality=winner.quality,
            test_quality=test.quality,
            strict_quality=test.strict_quality,
            schema_rate=test.schema_rate,
            cost_usd=test.cost_usd,
            latency_ms=test.latency_ms,
            dev_n=len(winner.scores),
            test_n=len(test.scores),
        ),
        artifact_text,
    )


def to_jsonl(path: Path, run: TrackedRun) -> Path:
    """Append one run to a JSONL log, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        _ = handle.write(run.as_line() + "\n")
    return path


def to_mlflow(
    run: TrackedRun,
    *,
    artifact_text: str = "",
    tracking_uri: str | None = None,
    experiment: str = EXPERIMENT,
) -> str:
    """Log one run to MLflow and return its run id.

    `tracking_uri` defaults to a SQLite store under `.mlflow/` because MLflow 3
    requires a database backend; a bare directory is no longer a valid file store.
    """
    mlflow = _require_mlflow()
    uri = tracking_uri or f"sqlite:///{Path('.mlflow/mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{run.task}:{run.config}") as active:
        mlflow.log_params({k: str(v) for k, v in run.params.items()})
        mlflow.log_metrics(run.metrics)
        mlflow.set_tags({"training": run.training, "recorded_at": run.recorded_at})
        if artifact_text:
            mlflow.log_text(artifact_text, "compile-report.txt")
        return str(active.info.run_id)


def record(
    run: TrackedRun,
    *,
    artifact_text: str = "",
    log_path: Path | None = None,
    tracking_uri: str | None = None,
    backend: str = "auto",
) -> str:
    """Record one run. Returns every place it landed, joined by " + ".

    `backend` is "jsonl", "mlflow", or "auto". Auto writes **both**: the JSONL
    line is the durable local record that sits next to the corpus, and MLflow is
    the interface on top of it when the extra is installed. Writing only MLflow
    would silently discard the `log_path` a caller supplied, which is what the
    first version of this did.
    """
    if backend not in {"auto", "jsonl", "mlflow"}:
        msg = f"unknown tracking backend: {backend!r}"
        raise ValueError(msg)
    places: list[str] = []
    if backend in {"auto", "jsonl"}:
        places.append(str(to_jsonl(log_path or Path(DEFAULT_LOG_NAME), run)))
    if backend == "mlflow" or (backend == "auto" and available()):
        run_id = to_mlflow(run, artifact_text=artifact_text, tracking_uri=tracking_uri)
        places.append(f"mlflow:{run_id}")
    return " + ".join(places)


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        msg = (
            "MLflow tracking needs the optional dependency: install with "
            "`uv sync --extra mlflow` or `pip install 'mekoy[mlflow]'`"
        )
        raise ImportError(msg) from exc
    return mlflow


def fields_of(run: TrackedRun) -> Mapping[str, object]:
    """Flat view of a record, useful for tests and for a future ranker."""
    return asdict(run)
