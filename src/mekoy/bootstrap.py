"""BootstrapFewShot: pick the demonstrations that survive the gate.

PLAN 15.4 makes this Stage 0 of the optimisation engine, before any search over
k-shot and decode. What ships without it is "take the first k training rows", which
is a sample, not a selection: it has no idea whether a given row teaches anything
or whether the model already handles it.

DSPy's BootstrapFewShot traces the program over the training set and keeps the
demonstrations whose traces passed the metric. Run against our metric, "passed"
means *cleared the task's deterministic gate*, which is the only notion of good
this project trusts.

The demonstration's label is taken from gold rather than from the trace. The
selection is the contribution; teaching a System from a possibly-wrong model output
is not something this pipeline should do.

Optional dependency, like `gepa`: PLAN 35 keeps DSPy out of the core install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mekoy.errors import CompileError
from mekoy.gepa import _ollama_base, _require_dspy, feedback_for
from mekoy.tasks import Task

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = ["DEFAULT_MAX_DEMOS", "bootstrap_demos"]

#: Demonstrations to keep. PLAN 15.5 brackets shots at 0 and 4, so four is the
#: ceiling a compile would ever ask for.
DEFAULT_MAX_DEMOS = 4
#: Candidate demonstrations to consider before selecting.
DEFAULT_MAX_LABELED = 12


def bootstrap_demos(  # noqa: PLR0913 - the stage names its own knobs
    task: Task,
    *,
    trainset: Sequence[tuple[str, object]],
    model_id: str,
    base_url: str,
    max_demos: int = DEFAULT_MAX_DEMOS,
    max_labeled: int = DEFAULT_MAX_LABELED,
) -> tuple[tuple[str, object], ...]:
    """Return the training pairs worth using as shots.

    An empty tuple means bootstrapping found nothing usable, which is a signal in
    itself: it says the model cannot clear the gate on any training row yet, so
    few-shot selection is not the problem to solve next.
    """
    dspy = _require_dspy()
    if not trainset:
        msg = "bootstrap needs a non-empty trainset"
        raise CompileError(message=msg)

    class Extract(dspy.Signature):
        """Replaced below with the task's current instructions."""

        document: str = dspy.InputField()
        result: str = dspy.OutputField(desc="JSON only, no commentary")

    Extract.__doc__ = task.prompt
    program = dspy.Predict(Extract)

    def metric(gold: Any, pred: Any, *_: Any) -> Any:
        score, feedback = feedback_for(task, gold.gold, getattr(pred, "result", ""))
        return dspy.Prediction(score=score, feedback=feedback)

    def to_example(pair: tuple[str, object]) -> Any:
        text, gold = pair
        return dspy.Example(document=text, gold=gold).with_inputs("document")

    train_examples = [to_example(p) for p in trainset]
    # Gold by text, so a kept demonstration is labelled with the truth rather than
    # with whatever the trace produced.
    gold_by_text = dict(trainset)

    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_labeled,
        max_rounds=1,
    )
    lm = dspy.LM(
        f"ollama_chat/{model_id}",
        api_base=_ollama_base(base_url),
        api_key="ollama",
        max_tokens=700,
        temperature=0.0,
    )
    with dspy.context(lm=lm):
        compiled = optimizer.compile(program, trainset=train_examples)

    return _kept_pairs(compiled, program, gold_by_text, limit=max_demos)


def _kept_pairs(
    compiled: Any,
    program: Any,
    gold_by_text: dict[str, object],
    *,
    limit: int,
) -> tuple[tuple[str, object], ...]:
    """Map DSPy's kept demonstrations back onto (text, gold) pairs."""
    demos = list(getattr(compiled, "demos", []) or [])
    demos.extend(getattr(program, "demos", []) or [])
    out: list[tuple[str, object]] = []
    seen: set[str] = set()
    for demo in demos:
        text = getattr(demo, "document", None)
        if not isinstance(text, str) or text in seen:
            continue
        gold = gold_by_text.get(text)
        if gold is None:
            continue
        seen.add(text)
        out.append((text, gold))
        if len(out) >= limit:
            break
    return tuple(out)
