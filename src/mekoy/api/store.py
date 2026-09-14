"""In-memory Systems and Runs. Phase is the record type."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import assert_never

from mekoy.compile import CompileReport, Trial, format_report
from mekoy.dataset import TaskExample
from mekoy.errors import CompileError


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


class Store:
    """Process-local maps. Each create_app() owns one instance."""

    def __init__(self) -> None:
        """Start empty."""
        self._systems: dict[str, SystemRecord] = {}
        self._runs: dict[str, RunRecord] = {}

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
        return record

    def get_system(self, system_id: str) -> SystemRecord:
        """Return a System or raise NotFoundError."""
        record = self._systems.get(system_id)
        if record is None:
            msg = f"system not found: {system_id}"
            raise NotFoundError(message=msg)
        return record

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
        self._systems[run.system_id] = CompiledSystem(
            id=current.id,
            task=current.task,
            task_name=current.task_name,
            examples=current.examples,
            winner=report.winner,
            report=text,
        )
        return done

    def fail_run(self, run_id: str, error: str) -> RunRecord:
        """Mark the run failed. System phase is unchanged."""
        run = self.get_run(run_id)
        failed = replace(run, status=RunStatus.FAILED, error=error)
        self._runs[run_id] = failed
        return failed
