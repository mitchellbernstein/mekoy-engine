"""In-memory Systems and Runs. Phase is the record type."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from mekoy.compile import CompileReport, Trial, format_report
from mekoy.dataset import TaskExample
from mekoy.errors import CompileError
from mekoy.score import ExampleScore
from mekoy.search import HarnessConfig
from mekoy.spec import SystemSpec, load_spec
from mekoy.tasks import task_by_name


class Phase(StrEnum):
    """Public lifecycle. Derived from the System record type."""

    DRAFT = "draft"
    EVAL_APPROVED = "eval_approved"
    COMPILED = "compiled"


class RunStatus(StrEnum):
    """Compile run status."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class NotFoundError(CompileError):
    """Unknown system or run id."""


class PhaseError(CompileError):
    """Request is illegal in the current System phase."""


@dataclass(frozen=True, slots=True)
class DraftSystem:
    """System whose eval has not been approved."""

    id: str
    task: str
    examples: tuple[TaskExample, ...]
    #: Task class name (`restaurant`, `receipt`, `banking77`). The job text is
    #: `task`; this is which compiler path to run.
    task_name: str = "restaurant"


@dataclass(frozen=True, slots=True)
class ApprovedSystem:
    """System whose eval is approved and compile may run."""

    id: str
    task: str
    examples: tuple[TaskExample, ...]
    task_name: str = "restaurant"


@dataclass(frozen=True, slots=True)
class CompiledSystem:
    """System with a searched winner. Invoke is allowed."""

    id: str
    task: str
    examples: tuple[TaskExample, ...]
    winner: Trial
    report: str
    task_name: str = "restaurant"


type SystemRecord = DraftSystem | ApprovedSystem | CompiledSystem


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One compile attempt against a System."""

    id: str
    system_id: str
    status: RunStatus
    quick: bool
    #: The base model this run searched against. Invoke must replay it: a System
    #: is the harness *plus* the model, and calling a different one is not the
    #: System that was measured.
    model: str = ""
    report: str | None = None
    winner: Trial | None = None
    error: str | None = None


def phase_of(record: SystemRecord) -> Phase:
    """Map the record type to the public phase."""
    match record:
        case DraftSystem():
            return Phase.DRAFT
        case ApprovedSystem():
            return Phase.EVAL_APPROVED
        case CompiledSystem():
            return Phase.COMPILED
        case _ as unreachable:
            assert_never(unreachable)


#: Tables. The index lives here; the per-System artifact directory stays the source of
#: truth for the System itself, so a user can copy the folder and leave.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS systems (
    id        TEXT PRIMARY KEY,
    task      TEXT NOT NULL,
    task_name TEXT NOT NULL,
    phase     TEXT NOT NULL,
    examples  TEXT NOT NULL,
    winner    TEXT,
    report    TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id         TEXT PRIMARY KEY,
    system_id  TEXT NOT NULL,
    status     TEXT NOT NULL,
    quick      INTEGER NOT NULL,
    model      TEXT NOT NULL,
    report     TEXT,
    winner     TEXT,
    error      TEXT,
    seq        INTEGER NOT NULL
);
"""


def _outcome_out(outcome: object) -> object:
    """An example's label, as JSON. Pydantic models become plain objects."""
    dump = getattr(outcome, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return outcome


def _outcome_in(task_name: str, data: object) -> object:
    """Rebuild an example's label using the task's own model.

    The label has to come back as the type the task's gate and scorer expect, and the
    task name stored beside it is what says which type that is. Without it a reloaded
    System would score its own examples differently from the run that produced it.
    """
    model = getattr(task_by_name(task_name), "model", None)
    validate = getattr(model, "model_validate", None)
    if callable(validate) and isinstance(data, dict):
        return validate(data)
    return data


def _trial_out(trial: Trial) -> dict[str, object]:
    """A measured candidate as JSON."""
    return {
        "config": asdict(trial.config),
        "scores": [asdict(s) if is_dataclass(s) else s for s in trial.scores],
        "latency_ms": trial.latency_ms,
        "cost_usd": trial.cost_usd,
        "reasons": list(trial.reasons),
    }


def _trial_in(data: dict[str, object]) -> Trial:
    """Rebuild a measured candidate from JSON."""
    known = {f.name for f in fields(HarnessConfig)}
    config = {
        str(k): v
        for k, v in dict(data["config"]).items()
        if str(k) in known  # type: ignore[arg-type]
    }
    scores = tuple(
        ExampleScore(**s)  # type: ignore[misc]
        for s in data.get("scores", [])
        if isinstance(s, dict)
    )
    return Trial(
        config=HarnessConfig(**config),  # type: ignore[arg-type]
        scores=scores,
        latency_ms=float(data.get("latency_ms") or 0.0),  # type: ignore[arg-type]
        cost_usd=float(data.get("cost_usd") or 0.0),  # type: ignore[arg-type]
        reasons=tuple(str(r) for r in data.get("reasons", [])),  # type: ignore[union-attr]
    )


#: The newest spec format this code understands. A bundle from the future is skipped
#: rather than half-read, because a field whose meaning changed cannot be guessed.
_SPEC_VERSION_KNOWN = 1


def _examples_from(path: Path) -> tuple[TaskExample, ...]:
    """Read a bundle's labeled rows back, if it carries any."""
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


def _task_name_for(spec: SystemSpec) -> str:
    """Which task class a recovered System was compiled with.

    The spec does not name the class, so it is inferred from the job text the way the
    loader infers it from a row. Getting this wrong would score a recovered System with
    the wrong gate.
    """
    text = spec.task.lower()
    if "receipt" in text:
        return "receipt"
    if "intent" in text or "categor" in text or "classif" in text:
        return "banking77"
    return "restaurant"


class Store:
    """Systems and runs, optionally surviving the process that made them.

    In-memory by default, which is what the tests want. Given a `db_path` it writes
    through to SQLite on every change and reads back on construction, so a restart no
    longer loses the index. SQLite is in the standard library, needs no service, and
    runs on one laptop, which is what a self-hoster has.
    """

    def __init__(self, *, db_path: Path | None = None) -> None:
        """Start empty, or from disk when a database is given."""
        self._systems: dict[str, SystemRecord] = {}
        self._runs: dict[str, RunRecord] = {}
        self._seq = 0
        self._db: sqlite3.Connection | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.executescript(_SCHEMA)
            self._read_back()

    def recover(self, artifacts_root: Path) -> tuple[int, int]:
        """Fix up what a previous process left behind, at startup.

        Two things go wrong when a process dies, and both are silent.

        A run that was in flight is still recorded as `running`, and nothing will ever
        move it: the caller cannot tell a slow compile from a dead one. Any run still
        `running` when a fresh process starts is orphaned by definition, because this
        process has run nothing yet.

        And the index can be lost while the Systems it described are still on disk. A
        compiled System is a directory; the database is only how it is found. When the
        index is empty the directories are read back, which is what makes the artifacts
        the source of truth rather than a cache of the database.

        Returns (runs_failed, systems_recovered).
        """
        if self._db is None:
            return (0, 0)
        failed = 0
        for (run_id,) in list(
            self._db.execute(
                "SELECT id FROM runs WHERE status = ?", (RunStatus.RUNNING,)
            )
        ):
            _ = self.fail_run(
                run_id,
                "interrupted: the process running this compile is gone",
            )
            failed += 1
        recovered = self._rebuild_from(artifacts_root) if not self._systems else 0
        return (failed, recovered)

    def _rebuild_from(self, artifacts_root: Path) -> int:
        """Read Systems back off disk when the index is empty.

        A bundle written before `spec_version` and `examples.jsonl` existed
        restores as a compiled System without its examples, which is a real
        limitation and the reason those fields were added. Newer ones restore whole.
        compiled System without its examples, which is a real limitation and the reason
        those fields were added. Newer bundles restore whole.
        """
        if not artifacts_root.is_dir():
            return 0
        found = 0
        for directory in sorted(artifacts_root.glob("sys_*")):
            spec_path = directory / "spec.json"
            if not spec_path.is_file():
                continue
            try:
                spec = load_spec(spec_path)
            except CompileError:
                continue
            if spec.spec_version > _SPEC_VERSION_KNOWN:
                continue
            examples = _examples_from(directory / "examples.jsonl")
            report_path = directory / "report.txt"
            report = (
                report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
            )
            winner = Trial(
                config=HarnessConfig(
                    k_shot=spec.k_shot,
                    retries=spec.retries,
                    constrained=True,
                    model=spec.model_id,
                ),
                scores=(),
            )
            record = CompiledSystem(
                id=directory.name,
                task=spec.task,
                task_name=_task_name_for(spec),
                examples=examples,
                winner=winner,
                report=report,
            )
            self._systems[record.id] = record
            self._write_system(record)
            found += 1
        return found

    # --- persistence -----------------------------------------------------

    def _read_back(self) -> None:
        """Load every record. Called once, at construction."""
        if self._db is None:
            return
        for row in self._db.execute(
            "SELECT id, task, task_name, phase, examples, winner, report FROM systems"
        ):
            sid, task, task_name, phase, examples, winner, report = row
            restored = tuple(
                TaskExample(
                    text=str(e["text"]), outcome=_outcome_in(task_name, e["outcome"])
                )
                for e in json.loads(examples)
            )
            parsed = _trial_in(json.loads(winner)) if winner else None
            if phase == Phase.COMPILED and parsed is not None:
                self._systems[sid] = CompiledSystem(
                    id=sid,
                    task=task,
                    task_name=task_name,
                    examples=restored,
                    winner=parsed,
                    report=report or "",
                )
            elif phase == Phase.EVAL_APPROVED:
                self._systems[sid] = ApprovedSystem(
                    id=sid, task=task, task_name=task_name, examples=restored
                )
            else:
                self._systems[sid] = DraftSystem(
                    id=sid, task=task, task_name=task_name, examples=restored
                )
        for row in self._db.execute(
            "SELECT id, system_id, status, quick, model, report, winner, error, seq "
            "FROM runs ORDER BY seq"
        ):
            rid, sid, status, quick, model, report, winner, error, seq = row
            self._runs[rid] = RunRecord(
                id=rid,
                system_id=sid,
                status=RunStatus(status),
                quick=bool(quick),
                model=model or "",
                report=report,
                winner=_trial_in(json.loads(winner)) if winner else None,
                error=error,
            )
            self._seq = max(self._seq, int(seq) + 1)

    def _write_system(self, record: SystemRecord) -> None:
        """Write one System through to the database."""
        if self._db is None:
            return
        winner = (
            json.dumps(_trial_out(record.winner))
            if isinstance(record, CompiledSystem)
            else None
        )
        report = record.report if isinstance(record, CompiledSystem) else None
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO systems "
                "(id, task, task_name, phase, examples, winner, report) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.task,
                    record.task_name,
                    str(phase_of(record)),
                    json.dumps(
                        [
                            {"text": e.text, "outcome": _outcome_out(e.outcome)}
                            for e in record.examples
                        ]
                    ),
                    winner,
                    report,
                ),
            )

    def _write_run(self, run: RunRecord) -> None:
        """Write one run through to the database."""
        if self._db is None:
            return
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO runs "
                "(id, system_id, status, quick, model, report, winner, error, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.system_id,
                    str(run.status),
                    int(run.quick),
                    run.model,
                    run.report,
                    json.dumps(_trial_out(run.winner)) if run.winner else None,
                    run.error,
                    self._seq,
                ),
            )
        self._seq += 1

    # --- the interface ----------------------------------------------------

    def create(
        self,
        *,
        task: str,
        examples: tuple[TaskExample, ...],
        task_name: str = "restaurant",
    ) -> DraftSystem:
        """Insert a draft System."""
        record = DraftSystem(
            id=f"sys_{uuid.uuid4().hex}",
            task=task,
            task_name=task_name,
            examples=examples,
        )
        self._systems[record.id] = record
        self._write_system(record)
        return record

    def get_system(self, system_id: str) -> SystemRecord:
        """Return a System or raise NotFoundError."""
        record = self._systems.get(system_id)
        if record is None:
            msg = f"system not found: {system_id}"
            raise NotFoundError(message=msg)
        return record

    def list_systems(self) -> tuple[SystemRecord, ...]:
        """Every System, newest last.

        The catalog cannot be built on an API that cannot enumerate its own Systems, so
        this is the floor under ranked listings, and it is also how a user finds what
        they built yesterday. Insertion order is preserved by the dict and by the run
        sequence, so the order is creation order.
        """
        return tuple(self._systems.values())

    def get_run(self, run_id: str) -> RunRecord:
        """Return a run or raise NotFoundError."""
        record = self._runs.get(run_id)
        if record is None:
            msg = f"run not found: {run_id}"
            raise NotFoundError(message=msg)
        return record

    def approve(self, system_id: str) -> SystemRecord:
        """Record eval approval. Idempotent."""
        current = self.get_system(system_id)
        match current:
            case DraftSystem(id=sid, task=task, task_name=task_name, examples=examples):
                approved = ApprovedSystem(
                    id=sid, task=task, task_name=task_name, examples=examples
                )
                self._systems[sid] = approved
                self._write_system(approved)
                return approved
            case ApprovedSystem() | CompiledSystem():
                return current
            case _ as unreachable:
                assert_never(unreachable)

    def require_approved(self, system_id: str) -> ApprovedSystem | CompiledSystem:
        """Gate compile on eval approval."""
        record = self.get_system(system_id)
        if isinstance(record, DraftSystem):
            msg = "eval is not approved; POST /v1/systems/{id}/evals with approve=true"
            raise PhaseError(message=msg)
        return record

    def require_compiled(self, system_id: str) -> CompiledSystem:
        """Gate invoke on a finished compile."""
        record = self.get_system(system_id)
        if isinstance(record, CompiledSystem):
            return record
        msg = "system is not compiled; POST /v1/systems/{id}/compile first"
        raise PhaseError(message=msg)

    def latest_run(self, system_id: str) -> RunRecord | None:
        """The most recent run for a System, or None if it has never compiled.

        Services order by insertion, which is creation order, so the last match is
        the newest.
        """
        _ = self.get_system(system_id)
        runs = [r for r in self._runs.values() if r.system_id == system_id]
        return runs[-1] if runs else None

    def create_run(self, system_id: str, *, quick: bool, model: str = "") -> RunRecord:
        """Open a compile run. Eval must already be approved."""
        _ = self.require_approved(system_id)
        run = RunRecord(
            id=f"run_{uuid.uuid4().hex}",
            system_id=system_id,
            status=RunStatus.RUNNING,
            quick=quick,
            model=model,
        )
        self._runs[run.id] = run
        self._write_run(run)
        return run

    def succeed_run(self, run_id: str, report: CompileReport) -> RunRecord:
        """Attach the winner to the run and the System."""
        run = self.get_run(run_id)
        text = format_report(report)
        done = replace(
            run,
            status=RunStatus.SUCCEEDED,
            report=text,
            winner=report.winner,
            error=None,
        )
        self._runs[run_id] = done
        current = self.get_system(run.system_id)
        compiled = CompiledSystem(
            id=current.id,
            task=current.task,
            task_name=current.task_name,
            examples=current.examples,
            winner=report.winner,
            report=text,
        )
        self._systems[run.system_id] = compiled
        # Written on compile, not on deploy. A System that was compiled and never
        # deployed used to exist only in RAM, so closing the laptop lost it with no
        # restart involved; the deploy route was the only writer.
        self._write_system(compiled)
        self._write_run(done)
        return done

    def fail_run(self, run_id: str, error: str) -> RunRecord:
        """Mark the run failed. System phase is unchanged."""
        run = self.get_run(run_id)
        failed = replace(run, status=RunStatus.FAILED, error=error)
        self._runs[run_id] = failed
        self._write_run(failed)
        return failed
