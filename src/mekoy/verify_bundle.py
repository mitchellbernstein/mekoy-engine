"""Recompute an advertised score from a bundle, and say whether it holds.

The engine's number is ours no matter how carefully it was measured: our scorer, our
corpus, our split. A skeptic is right to call it self-reported, and the only answer
that is not "trust us" is to hand them the rows, the harness, and the scorer so they
can recompute it. That is what this command does, and why the bundle carries
`holdout.jsonl` and `environment.json` at all.

It reports a match, a mismatch, or that it could not run. It never rounds a mismatch
into a pass, and a missing model is a failure to verify rather than a pass by default:
the point is that the answer is checkable, so an unverifiable bundle must say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mekoy.dataset import TaskExample
from mekoy.errors import CompileError
from mekoy.harness import PROMPTS, Decode, extract
from mekoy.runtime import Completer
from mekoy.score import ExampleScore
from mekoy.spec import SystemSpec, load_spec
from mekoy.tasks import task_by_name
from mekoy.verify import VerifyFail, VerifyOk

#: How far the recomputed number may sit from the advertised one. Field accuracy is a
#: ratio over a countable number of fields, so a genuine reproduction lands exactly;
#: the tolerance exists for the last-bit float difference of a different summation
#: order, not to absorb real disagreement.
_TOLERANCE = 0.005

#: The file the advertised number is read from, and the line shape it uses.
_REPORT_LINE = "test"


@dataclass(frozen=True)
class Verification:
    """What re-running a bundle's held-out rows actually produced."""

    advertised: float
    recomputed: float
    rows: int
    failed_rows: int
    #: Set when the bundle cannot be checked at all, with the reason. Distinct from a
    #: mismatch: a mismatch says the number is wrong, this says the number is unknown.
    unverifiable: str | None = None

    @property
    def agrees(self) -> bool:
        """True when the recomputed score reproduces the advertised one."""
        if self.unverifiable is not None:
            return False
        return abs(self.recomputed - self.advertised) <= _TOLERANCE

    def explain(self) -> str:
        """The verdict, with both numbers so a reader can judge for themselves."""
        if self.unverifiable is not None:
            return f"not verified: {self.unverifiable}"
        if self.rows == 0:
            return "not verified: the bundle carries no held-out rows to score"
        verdict = "REPRODUCED" if self.agrees else "DID NOT REPRODUCE"
        return (
            f"{verdict}: advertised {self.advertised:.3f}, recomputed "
            f"{self.recomputed:.3f} on {self.rows} held-out rows"
            + (
                f" ({self.failed_rows} rejected by the gate)"
                if self.failed_rows
                else ""
            )
        )


def advertised_quality(directory: Path) -> float:
    """The number the bundle claims, read from its own report."""
    report = directory / "report.txt"
    if not report.is_file():
        msg = f"no report.txt in {directory}, so there is no number to check"
        raise CompileError(message=msg)
    for line in report.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(_REPORT_LINE):
            for word in stripped.split():
                if word.startswith("quality="):
                    try:
                        return float(word.split("=", 1)[1])
                    except ValueError:
                        break
    msg = f"no test quality line in {report}, so there is no number to check"
    raise CompileError(message=msg)


def read_holdout(directory: Path, task_name: str) -> tuple[TaskExample, ...]:
    """The held-out rows a bundle was scored on."""
    path = directory / "holdout.jsonl"
    if not path.is_file():
        return ()
    task = task_by_name(task_name)
    rows: list[TaskExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        label = payload.get("outcome") or payload.get("receipt")
        model = getattr(task, "model", None)
        validate = getattr(model, "model_validate", None)
        if callable(validate) and isinstance(label, dict):
            label = validate(label)
        rows.append(TaskExample(text=str(payload.get("text", "")), outcome=label))
    return tuple(rows)


def read_environment(directory: Path) -> dict[str, object]:
    """The model and settings the advertised number was produced with."""
    path = directory / "environment.json"
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def verify(
    directory: Path,
    *,
    completer: Completer,
    spec: SystemSpec | None = None,
) -> Verification:
    """Re-run the held-out rows and report whether the number holds.

    The harness is the winner's: every axis that was measured. Running the rows through
    anything else tests a different System and proves nothing about this one - which is
    exactly what this function used to do, by hardcoding the task's default brief, the
    constraint, and the schema while reading only the shot count and the retries. It
    reported a mismatch against a bundle whose number was correct.

    A bundle written before the spec recorded the harness cannot be verified at all, so
    it says so. Scoring its silent defaults and reporting the difference as a failed
    reproduction would blame the bundle for something the reader did.
    """
    resolved = spec or load_spec(directory)
    if not resolved.carries_harness:
        # Checked before the holdout, because this is the reason that cannot be fixed by
        # adding a file later: nobody recorded which harness produced the number.
        return Verification(
            advertised=advertised_quality(directory),
            recomputed=0.0,
            rows=0,
            failed_rows=0,
            unverifiable=(
                f"this bundle records spec_version {resolved.spec_version}, which does "
                "not carry the harness (prompt, schema constraint, constrained, "
                "consistency, bootstrap), so its score cannot be recomputed. A System "
                "exported before those axes were recorded has to be recompiled, not "
                "re-verified"
            ),
        )
    task = task_by_name(_task_name(resolved))
    rows = read_holdout(directory, task.name)
    if not rows:
        return Verification(
            advertised=advertised_quality(directory),
            recomputed=0.0,
            rows=0,
            failed_rows=0,
            unverifiable=(
                "this bundle carries no held-out rows to score, so the advertised "
                "number cannot be recomputed from it"
            ),
        )

    harness = resolved.harness()
    examples = _examples_from(directory)
    shots = tuple((row.text, row.outcome) for row in examples[: harness.k_shot])
    # Read from the spec, not assumed. A reused brief or a decode that was not pinned to
    # the schema is part of what was measured, so it is part of what must be re-run.
    system = PROMPTS.get(harness.prompt, task.prompt)
    decode = Decode(
        system=system,
        constrained=harness.constrained,
        schema=task.model.model_json_schema() if harness.schema else None,
    )

    scores: list[ExampleScore] = []
    rejected = 0
    for record in rows:
        result = extract(
            completer,
            text=record.text,
            shots=shots,
            retries=harness.retries,
            decode=decode,
            task=task,
        )
        if isinstance(result, VerifyFail):
            rejected += 1
            scores.append(_zero())
            continue
        if isinstance(result, VerifyOk):
            scores.append(task.score_pair(gold=record.outcome, pred=result.outcome))
            continue
        scores.append(_zero())

    total = sum(s.quality for s in scores) / len(scores)
    return Verification(
        advertised=advertised_quality(directory),
        recomputed=total,
        rows=len(scores),
        failed_rows=rejected,
    )


def _zero() -> ExampleScore:
    """A candidate the gate rejected scores nothing, which is what the compile did."""
    return ExampleScore(
        schema_ok=False, field_hits=0, field_total=1, line_f1=0.0, strict_hits=0
    )


def _task_name(spec: SystemSpec) -> str:
    """Which task class a bundle was compiled with, inferred from its job text."""
    text = spec.task.lower()
    if "receipt" in text:
        return "receipt"
    if "intent" in text or "categor" in text or "classif" in text:
        return "banking77"
    return "restaurant"


def _examples_from(directory: Path) -> tuple[TaskExample, ...]:
    """The in-context rows the harness shows, if the bundle carries them."""
    path = directory / "examples.jsonl"
    if not path.is_file():
        return ()
    rows: list[TaskExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            TaskExample(
                text=str(payload.get("text", "")), outcome=payload.get("outcome")
            )
        )
    return tuple(rows)
