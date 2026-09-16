"""GEPA-light: reflective prompt evolution, bounded by a metric-call cap.

This stage runs after the candidate pool: survivors get a light reflective pass,
with a metric-call cap of 50-150 and a large open model as the reflection LM. This
is that stage, built on DSPy 3.3.

Three decisions worth stating:

- **Optional dependency.** The risk is DSPy lock-in: "System.json
  must be loadable without DSPy for invoke." DSPy lives in the `gepa` extra and is
  imported lazily, so the core compile path never touches it.
- **GEPA is judged by our metric.** The metric parses a reply through the task's
  schema and gate, scores it with the task's scorer, and returns the gate's own
  reasons as feedback. The optimizer is therefore pushed toward a System that
  satisfies the checks we enforce, not toward a generic objective.
- **The reflection LM is local.** The ban on closed-model compile data covers
  reflection too: an instruction written by a closed API is closed-model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from mekoy.errors import CompileError
from mekoy.tasks import Task
from mekoy.verify import VerifyFail, VerifyOk, parse_and_gate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_AUTO",
    "DEFAULT_METRIC_CALLS",
    "GepaOutcome",
    "available",
    "feedback_for",
    "field_feedback",
    "optimize",
]

#: GEPA is capped at 50-150 metric calls. The floor of that band keeps a
#: laptop run bounded while still giving reflection something to work with.
#:
#: Both a cap and `auto` are wanted, but dspy rejects them together: "Exactly one of
#: max_metric_calls, max_full_evals, auto must be set." The cap is the part that
#: matters, because it is what bounds a compile, so that is the default and `auto`
#: is opt-in instead.
DEFAULT_METRIC_CALLS = 60
#: `auto` level, used only when `max_metric_calls` is None.
DEFAULT_AUTO = "light"


def available() -> bool:
    """Whether the optional DSPy dependency is installed."""
    try:
        import dspy  # noqa: F401
    except ImportError:
        return False
    return True


def _ollama_base(url: str) -> str:
    """Base URL as litellm's ollama_chat provider wants it.

    That provider appends its own path, so the OpenAI-compatible `/v1` suffix our
    runtime uses makes it request `/v1/api/chat` and get a 404.
    """
    return url.rstrip("/").removesuffix("/v1")


def _require_dspy() -> Any:
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        msg = (
            "GEPA needs the optional dependency: install with "
            "`uv sync --extra gepa` or `pip install 'mekoy[gepa]'`"
        )
        raise CompileError(message=msg) from exc
    return dspy


@dataclass(frozen=True, slots=True)
class GepaOutcome:
    """What the reflection stage produced."""

    instructions: str
    metric_calls: int
    baseline: float
    best: float

    @property
    def improved(self) -> bool:
        """Whether reflection found something better than where it started."""
        return self.best > self.baseline


def field_feedback(task: Task, gold: object, pred: object) -> str:
    """Name the scored fields that disagree, as text a reflection model can use.

    Deliberately a rough comparison, not the scorer: this is prompt material, so
    naming every disagreement is more useful than reproducing the exact metric.
    The number that decides anything comes from the task's scorer.
    """
    wrong = [
        f"{name}: expected {getattr(gold, name)!r}, produced {getattr(pred, name)!r}"
        for name in task.scored_fields
        if str(getattr(gold, name)) != str(getattr(pred, name))
    ]
    if not wrong:
        return "All scored fields agreed."
    return "Fields that disagreed: " + "; ".join(wrong)


def feedback_for(task: Task, gold: object, raw: str) -> tuple[float, str]:
    """Score one reply and explain it, in the shape GEPA wants.

    Returns (score, feedback). The score is the task's own quality; the feedback
    names the gate reasons or the fields that disagreed, because a bare "0.71"
    tells a reflection model nothing it can act on.
    """
    result = parse_and_gate(raw, model=task.model, gate=task.gate)
    if isinstance(result, VerifyFail):
        return 0.0, "Rejected by the deterministic gate: " + "; ".join(result.reasons)
    # VerifyFail returned above, so this is the verified branch. `cast` rather
    # than `assert`, because asserts vanish under -O.
    outcome = cast("VerifyOk", result).outcome
    return task.score_pair(gold, outcome).quality, field_feedback(task, gold, outcome)


def optimize(  # noqa: PLR0913 - the stage names its own knobs
    task: Task,
    *,
    trainset: Sequence[tuple[str, object]],
    valset: Sequence[tuple[str, object]],
    model_id: str,
    base_url: str,
    max_metric_calls: int | None = DEFAULT_METRIC_CALLS,
    auto: str | None = None,
    seed: int = 0,
) -> GepaOutcome:
    """Evolve the task's instructions against its own gate and scorer.

    `trainset` and `valset` are (text, gold) pairs.
    """
    dspy = _require_dspy()
    if not trainset or not valset:
        msg = "GEPA needs a non-empty trainset and valset"
        raise CompileError(message=msg)
    if (max_metric_calls is None) == (auto is None):
        msg = "set exactly one of max_metric_calls or auto; dspy rejects both"
        raise CompileError(message=msg)
    budget: dict[str, object] = (
        {"max_metric_calls": max_metric_calls}
        if max_metric_calls is not None
        else {"auto": auto}
    )

    class Extract(dspy.Signature):
        """Replaced below with the task's current instructions."""

        document: str = dspy.InputField()
        result: str = dspy.OutputField(desc="JSON only, no commentary")

    # dspy reads a Signature's instructions from `__doc__`, so this is how the
    # task's prompt becomes the starting point GEPA evolves from.
    Extract.__doc__ = task.prompt
    program = dspy.Predict(Extract)

    def metric(gold: Any, pred: Any, *_: Any) -> Any:
        score, feedback = feedback_for(task, gold.gold, getattr(pred, "result", ""))
        return dspy.Prediction(score=score, feedback=feedback)

    def to_example(pair: tuple[str, object]) -> Any:
        text, gold = pair
        return dspy.Example(document=text, gold=gold).with_inputs("document")

    train_examples = [to_example(p) for p in trainset]
    val_examples = [to_example(p) for p in valset]

    api_base = _ollama_base(base_url)
    task_lm = dspy.LM(
        f"ollama_chat/{model_id}",
        api_base=api_base,
        api_key="ollama",
        max_tokens=700,
        temperature=0.0,
    )
    optimizer = dspy.GEPA(
        metric=metric,
        **budget,
        reflection_lm=dspy.LM(
            f"ollama_chat/{model_id}",
            api_base=api_base,
            api_key="ollama",
            max_tokens=900,
            temperature=0.7,
        ),
        # Pareto over complementary traces, the discipline that keeps a search from
        # collapsing onto a single behaviour.
        candidate_selection_strategy="pareto",
        # A local server serialises generation; threads only add contention and
        # non-determinism here.
        num_threads=1,
        seed=seed,
    )
    # Baseline, optimisation, and final measurement all need the LM in scope.
    with dspy.context(lm=task_lm):
        baseline = _mean_score(metric, program, val_examples)
        compiled = optimizer.compile(
            program, trainset=train_examples, valset=val_examples
        )
        best = _mean_score(metric, compiled, val_examples)
    return GepaOutcome(
        instructions=_instructions_of(compiled, fallback=task.prompt),
        metric_calls=int(max_metric_calls or 0),
        baseline=baseline,
        best=best,
    )


def _instructions_of(program: Any, *, fallback: str) -> str:
    """Pull the evolved instruction text back out of the compiled program."""
    predict = getattr(program, "predict", None) or program
    signature = getattr(predict, "signature", None)
    text = getattr(signature, "instructions", None)
    return text if isinstance(text, str) and text.strip() else fallback


def _mean_score(metric: Any, program: Any, examples: list[Any]) -> float:
    """Evaluate a program over examples. Gives the before/after number."""
    total = 0.0
    for example in examples:
        prediction = program(document=example.document)
        total += float(getattr(metric(example, prediction), "score", 0.0))
    return total / max(1, len(examples))
