"""A job defined in data, for people who are not going to edit this repository.

The engine shipped three tasks as Python objects, and adding a fourth meant writing
code: a `gate` and a `score` were callables, and `ALL_TASKS` was a tuple literal.
Everything the compiler needs from a task is data - what fields exist, which are free
text, which are times, and what makes an answer unacceptable - so a job can be
described rather than programmed.

Three kinds of check cover what the shipped tasks do, and they are the whole vocabulary:

- `required`: the field must be present and not empty.
- `arithmetic`: numeric fields must satisfy a sum relationship, which is how a receipt
  is rejected when its line items do not add up.
- `supported`: a claim must be backed by the document it came from, which is how a
  restaurant call gets rejected when it claims a booking the transcript never
  supports.

Anything a domain needs beyond that is a new check verb, and adding one is a
deliberate act rather than something a user is handed by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mekoy.errors import CompileError

#: Two numbers within this are the same amount of money. Half a cent, because a label
#: carrying `128.57` and a computed `128.565` are the same figure at two decimal places.
_CENT = 0.005

#: The check verbs a definition may use.
_CHECKS = frozenset({"required", "arithmetic", "supported"})


class FieldSpec(BaseModel):
    """One field of a job's answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    type: str = Field(default="string")
    #: Free text: scored on meaning rather than exact string.
    phrase: bool = False
    #: A time: scored without formatting differences.
    time: bool = False


class CheckSpec(BaseModel):
    """A deterministic rule an answer has to satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: required | arithmetic | supported
    kind: str
    #: Fields this rule reads.
    fields: tuple[str, ...] = ()
    #: For arithmetic: `a + b = c` becomes ["a", "b"], "c".
    parts: tuple[str, ...] = ()
    total: str = ""
    #: For supported: the field holding the document text.
    source: str = "text"
    #: For supported: the field holding the claim.
    claim: str = ""
    #: Shown to a reader when the rule rejects an answer.
    message: str = ""


class JobDefinition(BaseModel):
    """A whole job, as data: what to return, and what makes an answer wrong."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    #: The job in words, used as the instruction when no prompt is given.
    task: str = Field(min_length=1)
    prompt: str = ""
    fields: tuple[FieldSpec, ...] = Field(min_length=1)
    checks: tuple[CheckSpec, ...] = ()
    #: Retry counts worth trying. An arithmetic gate often needs a repair pass.
    retry_ladder: tuple[int, ...] = (0, 1)

    def names(self) -> tuple[str, ...]:
        """The field names, in order."""
        return tuple(spec.name for spec in self.fields)


def load_job(path: Path) -> JobDefinition:
    """Read a job definition from a JSON file."""
    if not path.is_file():
        msg = f"job definition not found: {path}"
        raise CompileError(message=msg)
    try:
        return JobDefinition.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        msg = f"{path}: {exc}"
        raise CompileError(message=msg) from exc


def job_from_examples(directory: Path, *, name: str, task: str) -> JobDefinition | None:
    """Infer a job definition from a labeled examples file, or None if unreadable.

    A user hands over examples and a description; asking them to also hand-write a
    schema would be asking them to do the compiler's job. The first row's label gives
    the fields and their types, and the checks are inferred from the values: numbers
    get an arithmetic check only when the labels actually satisfy a sum, so no rule is
    invented that the user's own data would fail.
    """
    rows = _rows(directory)
    if not rows:
        return None
    first = rows[0]
    if not isinstance(first, dict):
        return None
    specs = tuple(
        FieldSpec(name=str(key), type=_type_name(value)) for key, value in first.items()
    )
    return JobDefinition(
        name=name,
        task=task,
        fields=specs,
        checks=(
            # Every field is required. A user's own labels are the statement of what a
            # complete answer looks like, so a field they filled in on every row is one
            # they expect back. Without a required check the gate has nothing to reject
            # and every candidate passes, which is not a gate.
            CheckSpec(
                kind="required",
                fields=tuple(spec.name for spec in specs),
                message="a required field was missing from the answer",
            ),
            *_infer_checks(rows),
        ),
    )


def _rows(directory: Path) -> list[dict[str, object]]:
    """Every labeled row in a JSON Lines file, ignoring anything unreadable."""
    if not directory.is_file():
        return []
    out: list[dict[str, object]] = []
    for line in directory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = payload.get("outcome") or payload.get("receipt") or payload.get("label")
        if isinstance(label, dict):
            out.append(label)
    return out


def _type_name(value: object) -> str:
    """The JSON type of a label value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _infer_checks(rows: list[dict[str, object]]) -> tuple[CheckSpec, ...]:
    """Rules the user's own labels already satisfy, and no others.

    Inferring a check the data fails would make every candidate look broken, which is a
    worse failure than not checking at all.
    """
    if not rows:
        return ()
    numeric = [
        key
        for key in rows[0]
        if all(
            isinstance(row.get(key), int | float) and not isinstance(row.get(key), bool)
            for row in rows
        )
    ]
    checks: list[CheckSpec] = []
    for total in numeric:
        others = [key for key in numeric if key != total]
        for pair in _pairs(others):
            left, right = pair
            if all(
                _close(row.get(left, 0) + row.get(right, 0), row.get(total))
                for row in rows
            ):
                checks.append(
                    CheckSpec(
                        kind="arithmetic",
                        fields=(left, right, total),
                        parts=(left, right),
                        total=total,
                        message=f"{left} + {right} != {total}",
                    )
                )
                break
    return tuple(checks)


def _pairs(values: list[str]) -> list[tuple[str, str]]:
    """Every unordered pair, in a stable order."""
    return [
        (values[i], values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]


def _num(value: object) -> float:
    """A number from a label, or zero, so a sum can be computed without a cast dance."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _close(left: object, right: object) -> bool:
    """Whether two numbers are equal to within a cent."""
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return False
    return abs(float(left) - float(right)) < _CENT


@dataclass(frozen=True, slots=True)
class JobProblem:
    """One reason a definition cannot be compiled."""

    field: str
    message: str = field(default="")

    def __str__(self) -> str:
        """Readable in a validation message."""
        return f"{self.field}: {self.message}" if self.message else self.field


def _check_arithmetic(check: CheckSpec, known: set[str]) -> list[JobProblem]:
    """Problems with an arithmetic rule."""
    problems: list[JobProblem] = []
    if not check.parts or not check.total:
        problems.append(
            JobProblem("checks", "an arithmetic check needs parts and a total")
        )
    problems.extend(
        JobProblem(name, "arithmetic check names an unknown field")
        for name in (*check.parts, check.total)
        if name not in known
    )
    return problems


def _check_names(check: CheckSpec, known: set[str]) -> list[JobProblem]:
    """Problems with a rule that reads fields by name."""
    wanted = check.fields if check.kind == "required" else (check.claim,)
    return [
        JobProblem(name, f"{check.kind} check names an unknown field")
        for name in wanted
        if name and name not in known
    ]


def validate_job(definition: JobDefinition) -> tuple[JobProblem, ...]:
    """Every reason this definition cannot become a task.

    Checked before compiling, so a typo in a field name is a message rather than a run
    that fails halfway through with no explanation.
    """
    known = set(definition.names())
    problems: list[JobProblem] = []
    if not known:
        problems.append(JobProblem("fields", "a job needs at least one field"))
    for check in definition.checks:
        if check.kind not in _CHECKS:
            problems.extend(
                [
                    JobProblem(
                        "checks",
                        f"unknown check {check.kind!r}; known: {sorted(_CHECKS)}",
                    )
                ]
            )
            continue
        problems.extend(
            _check_arithmetic(check, known)
            if check.kind == "arithmetic"
            else _check_names(check, known)
        )
    if not any(check.kind == "required" for check in definition.checks):
        problems.append(
            JobProblem(
                "checks",
                "a job needs at least one required check, or nothing is rejected",
            )
        )
    return tuple(problems)
