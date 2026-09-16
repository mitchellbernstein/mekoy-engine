"""Compare models on the same held-out rows through one fixed harness.

A user is shown one number for their System and nothing to compare it to. The honest
comparison is not "this model is better" - it is the same harness, the same held-out
rows, one axis changed. That is what a re-point measures, and it is why the card names
every axis it held fixed: a harness that was not the one measured proves nothing about
the model swap.

The measurement lives here rather than in the experiment script because the connector
and the portal both need it, and a second copy of a measurement is a second answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from mekoy.harness import Decode, extract
from mekoy.runtime import Completer
from mekoy.score import ExampleScore
from mekoy.spec import SystemSpec
from mekoy.tasks import Task
from mekoy.verify import VerifyFail, VerifyOk

__all__ = ["Measurement", "compare_models", "format_model_comparison", "measure"]

#: Two scores closer than this are the same score on a small slice, and the card says
#: so rather than crowning a winner the rows cannot support.
_NOISE = 0.01


@dataclass(frozen=True, slots=True)
class Measurement:
    """One model's score on the held-out rows through one fixed harness."""

    model: str
    quality: float
    strict_quality: float
    schema_rate: float
    rows: int
    rejected: int
    latency_ms: float

    def line(self) -> str:
        """One report row, in the shape the compile cards use."""
        return (
            f"{self.model:16} quality={self.quality:.3f} "
            f"strict={self.strict_quality:.3f} schema={self.schema_rate:.3f} "
            f"rejected={self.rejected} {self.latency_ms:.0f}ms"
        )

    def as_json(self) -> dict[str, object]:
        """The same numbers as a dict, for a caller that reads them instead of prose."""
        return asdict(self)


def _shots(
    spec: SystemSpec, train: tuple[object, ...]
) -> tuple[tuple[str, object], ...]:
    """The in-context rows the harness shows, exactly `k_shot` of them.

    Taken from the train slice in the same order the search uses, because the harness
    includes *which* examples it shows, not only how many.
    """
    return tuple((row.text, row.outcome) for row in train[: spec.k_shot])  # type: ignore[attr-defined]


def _zero(task: Task) -> ExampleScore:
    """A rejected candidate scores nothing, which is what the compile did with it."""
    return ExampleScore(
        schema_ok=False,
        field_hits=0,
        field_total=len(task.scored_fields),
        line_f1=0.0,
        strict_hits=0,
    )


def measure(  # noqa: PLR0913 - a measurement names the axes it holds fixed
    spec: SystemSpec,
    *,
    completer: Completer,
    model: str,
    rows: tuple[object, ...],
    train: tuple[object, ...],
    task: Task,
) -> Measurement:
    """Run the spec's harness over `rows` through `completer`.

    Every axis comes off the spec: prompt variant, shot count, retries, whether the
    decode was constrained and whether it was pinned to the schema. A re-point that
    substituted a default here would measure a different System and report the
    difference as a model effect, which is the exact error this exists to avoid.
    """
    harness = spec.harness()
    shots = _shots(spec, train)
    system = task.prompt_variants.get(harness.prompt, task.prompt)
    decode = Decode(
        system=system,
        constrained=harness.constrained,
        schema=task.model.model_json_schema() if harness.schema else None,
    )
    scores: list[ExampleScore] = []
    rejected = 0
    elapsed = 0.0
    for row in rows:
        start = perf_counter()
        result = extract(
            completer,
            text=row.text,  # type: ignore[attr-defined]
            shots=shots,
            retries=harness.retries,
            decode=decode,
            task=task,
        )
        elapsed += perf_counter() - start
        if isinstance(result, VerifyFail):
            rejected += 1
            scores.append(_zero(task))
        elif isinstance(result, VerifyOk):
            scores.append(
                task.score_pair(gold=row.outcome, pred=result.outcome)  # type: ignore[attr-defined]
            )
        else:
            scores.append(_zero(task))
    n = max(1, len(rows))
    return Measurement(
        model=model,
        quality=sum(s.quality for s in scores) / n,
        strict_quality=sum(s.strict_quality for s in scores) / n,
        schema_rate=sum(1 for s in scores if s.schema_ok) / n,
        rows=len(scores),
        rejected=rejected,
        latency_ms=elapsed / n * 1000.0,
    )


def compare_models(
    spec: SystemSpec,
    *,
    source: Measurement,
    target: Measurement,
) -> dict[str, object]:
    """The comparison as numbers, with the axes held fixed spelled out."""
    return {
        "harness": {
            "k_shot": spec.k_shot,
            "retries": spec.retries,
            "constrained": spec.constrained,
            "schema_constrained": spec.schema_constrained,
            "prompt": spec.prompt,
            "consistency": spec.consistency,
            "bootstrap": spec.bootstrap,
        },
        "held_out_rows": source.rows,
        "source": source.as_json(),
        "target": target.as_json(),
        "delta_quality": target.quality - source.quality,
        "quality_winner": _winner(source.quality, target.quality),
        # The axis a user assumes away: speed is not size. The larger model here was
        # also the faster one, and hiding that would make the card look like a mistake.
        "faster": _winner(source.latency_ms, target.latency_ms, lower_is_better=True),
    }


def _winner(left: float, right: float, *, lower_is_better: bool = False) -> str:
    """'source', 'target', or 'tie', with the noise band applied to quality only."""
    if lower_is_better:
        if left == right:
            return "tie"
        return "source" if left < right else "target"
    if abs(left - right) < _NOISE:
        return "tie"
    return "source" if left > right else "target"


def format_model_comparison(
    *,
    spec: SystemSpec,
    source: Measurement,
    target: Measurement,
    source_label: str,
    target_label: str,
) -> str:
    """The side-by-side card, and the plain reading of it."""
    delta = target.quality - source.quality
    if abs(delta) < _NOISE:
        reading = (
            f"READING: the two models land within {abs(delta):.3f} of each other. "
            "The measured value sits in the harness: swapping the weights barely "
            "moves the score, so what a customer owns is the harness, not the model."
        )
    elif delta < 0:
        reading = (
            f"READING: re-pointing cost {abs(delta):.3f} quality. The harness carries "
            "most of the value but the model is not interchangeable: the smaller "
            "weights are measurably worse on this job."
        )
    else:
        reading = (
            f"READING: re-pointing GAINED {delta:.3f}. The larger model is better on "
            "this job, so the harness alone does not explain the score; the weights "
            "carry part of it."
        )
    faster = "source" if source.latency_ms < target.latency_ms else "target"
    if source.latency_ms == target.latency_ms:
        speed = "the two models took the same time per document"
    else:
        fast, slow = (source, target) if faster == "source" else (target, source)
        which = (
            "being the smaller model" if fast is source else "being the larger model"
        )
        speed = (
            f"{fast.model} was faster ({fast.latency_ms:.0f}ms vs "
            f"{slow.latency_ms:.0f}ms per document) despite {which}; "
            "speed and size are not the same axis"
        )
    return "\n".join(
        [
            f"held-out rows: {source.rows} (test, scored once)",
            (
                f"harness  k={spec.k_shot} r={spec.retries} "
                f"{'grammar' if spec.constrained else 'free'}"
                f"{' schema-pinned' if spec.schema_constrained else ''} "
                f"prompt={spec.prompt} consistency={spec.consistency} "
                f"bootstrap={spec.bootstrap}"
            ),
            f"only the model changes: {source_label} -> {target_label}",
            f"source model  {source.model:16} {source.line()}",
            f"target model  {target.model:16} {target.line()}",
            f"delta quality {delta:+.3f}",
            speed,
            reading,
        ]
    )
