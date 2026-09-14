"""JSONL examples and the three-way split: train shots, dev select, test report.

PLAN §16.5: user examples in the holdout are never used to pick a winner. The
search sees `train` and `dev` only. `test` is scored once, after selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from mekoy.errors import CompileError
from mekoy.outcome import RestaurantOutcome
from mekoy.tasks import Task

_MIN_ROWS = 3
#: An eval slice needs both intents, or it cannot score the
#: availability-versus-reservation call at all.
_MIN_INTENTS = 2
_TEST_FRACTION = 0.2
_DEV_FRACTION = 0.2


class ExampleRecord(BaseModel):
    """One labeled document."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    text: str
    outcome: RestaurantOutcome


@dataclass(frozen=True, slots=True)
class Split:
    """Train shots, dev for selection, test for the number you publish."""

    train: tuple[ExampleRecord, ...]
    dev: tuple[ExampleRecord, ...]
    test: tuple[ExampleRecord, ...]

    @property
    def holdout(self) -> tuple[ExampleRecord, ...]:
        """Test set, for callers that predate the dev/test split."""
        return self.test


def coerce_label(model: type[BaseModel], value: object) -> dict[str, object]:
    """Wrap a bare label value into the shape a task's schema expects.

    A classification row carries its label as a scalar (`"card_arrival"`), while an
    extraction row carries an object. The schema says which field owns it, so this
    is derivable rather than a second special case per caller.
    """
    if isinstance(value, dict):
        return value
    fields = list(model.model_fields)
    msg = f"label is not an object and {model.__name__} has {len(fields)} fields"
    if len(fields) != 1:
        raise CompileError(message=msg)
    return {fields[0]: value}


@dataclass(frozen=True, slots=True)
class TaskExample:
    """One labeled document for any task."""

    text: str
    outcome: object


def load_task_examples(path: Path, task: Task) -> tuple[TaskExample, ...]:
    """Load a fixture whose label key is `outcome`, `receipt`, or `label`.

    One loader for every task: the row layout is the same, only the schema under
    the label differs, and the task already knows its schema.
    """
    model = task.model
    if not path.is_file():
        msg = f"examples not found: {path}"
        raise CompileError(message=msg)
    rows: list[TaskExample] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            # `label` is the classification key; the others are extraction.
            label = raw.get("outcome", raw.get("receipt", raw.get("label")))
            # A classification row carries its label as a bare value, and the
            # schema that owns it says which field it fills.
            rows.append(
                TaskExample(
                    text=str(raw["text"]),
                    outcome=model.model_validate(coerce_label(model, label)),
                )
            )
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            msg = f"{path}:{line_no}: {exc}"
            raise CompileError(message=msg) from exc
    if len(rows) < _MIN_ROWS:
        msg = f"need at least {_MIN_ROWS} examples (one train, one dev, one test)"
        raise CompileError(message=msg)
    return tuple(rows)


def load_examples(path: Path) -> tuple[ExampleRecord, ...]:
    """Load examples.jsonl."""
    if not path.is_file():
        msg = f"examples not found: {path}"
        raise CompileError(message=msg)
    rows: list[ExampleRecord] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(ExampleRecord.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            msg = f"{path}:{line_no}: {exc}"
            raise CompileError(message=msg) from exc
    if len(rows) < _MIN_ROWS:
        msg = f"need at least {_MIN_ROWS} examples (one train, one dev, one test)"
        raise CompileError(message=msg)
    return tuple(rows)


def _rank(row: ExampleRecord) -> str:
    """Stable per-row order, independent of how the file happens to be sorted."""
    return hashlib.sha256(row.text.encode("utf-8")).hexdigest()


def split_examples(rows: tuple[ExampleRecord, ...]) -> Split:
    """~60/20/20 train/dev/test over rows sorted by their own hash.

    Sorting by a hash of the row text first means the split does not depend on
    how the file happens to be ordered, so a fixture written family-by-family
    (all the booked calls first) still spreads across all three slices.

    Selection happens on dev. Nothing in the search reads `test`, so the test
    score measures the compiled System rather than the search.
    """
    n = len(rows)
    if n < _MIN_ROWS:
        msg = f"need at least {_MIN_ROWS} examples (one train, one dev, one test)"
        raise CompileError(message=msg)
    test_n = max(1, round(n * _TEST_FRACTION))
    dev_n = max(1, round(n * _DEV_FRACTION))
    if test_n + dev_n >= n:
        test_n = dev_n = 1
    train_n = n - dev_n - test_n
    ordered = sorted(rows, key=_rank)
    return Split(
        train=ordered[:train_n],
        dev=ordered[train_n : train_n + dev_n],
        test=ordered[train_n + dev_n :],
    )


def audit_coverage(split: Split) -> tuple[str, ...]:
    """Complain when a slice cannot exercise the decision the eval exists for.

    A test slice with no booked reservation cannot tell a System that books
    things from one that never does.
    """
    problems: list[str] = []
    for label, part in (("test", split.test), ("dev", split.dev)):
        if not hasattr(part[0].outcome, "booked"):
            continue  # This check speaks only to the restaurant decision.
        booked = [r for r in part if r.outcome.booked]
        not_booked = [r for r in part if not r.outcome.booked]
        intents = {r.outcome.intent for r in part}
        if not booked:
            problems.append(f"{label}: no booked reservation")
        if not not_booked:
            problems.append(f"{label}: no unbooked call")
        if len(intents) < _MIN_INTENTS:
            problems.append(f"{label}: only one intent ({sorted(intents)})")
    return tuple(problems)
