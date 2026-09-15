"""Compile: search on dev, publish the test number.

The winner is chosen from `dev` only. `test` is scored once, after
selection, so the published number measures the System rather than the search.
Training is not run on this path; that is a first-class outcome in the report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

from mekoy.dataset import Split
from mekoy.privacy import assert_local
from mekoy.search import (
    Budget,
    HarnessConfig,
    SearchSpace,
    Slos,
    Trial,
    evaluate,
    search,
)
from mekoy.tasks import RESTAURANT, Task

__all__ = [
    "Budget",
    "CompileReport",
    "HarnessConfig",
    "SearchSpace",
    "Slos",
    "Trial",
    "compile_system",
    "format_report",
]

_TRAINING = "skipped"


@dataclass(frozen=True, slots=True)
class CompileReport:
    """Winner, every candidate tried, and the untouched test score."""

    winner: Trial
    trials: tuple[Trial, ...]
    test: Trial
    stopped_early: bool
    slos: Slos | None = None
    #: What it cost to produce this System, which is the number a buyer weighs
    #: against the per-document price. nanochat's headline metric is "time to
    #: GPT-2" rather than GPT-2's score, and the same framing applies here: the
    #: work of compiling is a cost, and a report that hides it is half a report.
    wall_seconds: float = 0.0
    model_calls: int = 0

    @property
    def tried(self) -> int:
        """Distinct candidates measured. Disclose this with the score."""
        return len({id(trial.config) for trial in self.trials})

    @property
    def pruned(self) -> int:
        """Candidates dropped by the minibatch rung before a full dev pass."""
        deepest: dict[int, int] = {}
        for trial in self.trials:
            key = id(trial.config)
            deepest[key] = max(deepest.get(key, 0), len(trial.scores))
        full = max(deepest.values(), default=0)
        return sum(1 for depth in deepest.values() if depth < full)

    def cost_to_produce(self, *, gpu_usd_per_hour: float = 0.0) -> str:
        """One line for the work of compiling, in time and money.

        `gpu_usd_per_hour` is zero for local hardware, which is the honest default:
        a local compile costs electricity and time, not dollars.
        """
        minutes = self.wall_seconds / 60.0
        dollars = self.wall_seconds / 3600.0 * gpu_usd_per_hour
        return (
            f"compile cost: {minutes:.1f} min, {self.model_calls} model calls, "
            f"{self.tried} candidate(s) tried, ${dollars:.2f}"
        )

    def describe_selection(self) -> str:
        """One line stating what the score was selected on."""
        return (
            f"selected on dev (n={len(self.winner.scores)}) "
            f"from {self.tried} candidate(s); "
            f"test (n={len(self.test.scores)}) scored once"
        )


def compile_system(  # noqa: PLR0913, PLR0917 - the compile entry point names its knobs
    completer: object,
    split: Split,
    space: SearchSpace,
    budget: Budget | None = None,
    allow_closed: bool = False,
    task: Task = RESTAURANT,
    bootstrapped: tuple[tuple[str, object], ...] = (),
    models: Mapping[str, object] | None = None,
) -> CompileReport:
    """Search the harness space on dev, then measure the winner on test.

    Refuses a closed-API completer unless `allow_closed` is set, so no closed
    model output can quietly become compile data.
    """
    assert_local(completer, allow_closed=allow_closed)
    spend = budget or Budget()
    started = perf_counter()
    winner, tried, stopped = search(
        completer,  # type: ignore[arg-type]
        shots=split.train,
        dev=split.dev,
        space=space,
        budget=spend,
        task=task,
        bootstrapped=bootstrapped,
        models=models,  # type: ignore[arg-type]
    )
    test = evaluate(
        completer,  # type: ignore[arg-type]
        split.test,
        winner.config,
        shots=split.train,
        task=task,
        bootstrapped=bootstrapped,
        models=models,  # type: ignore[arg-type]
    )
    return CompileReport(
        winner=winner,
        trials=tried,
        test=test,
        stopped_early=stopped,
        slos=spend.slos,
        wall_seconds=perf_counter() - started,
        model_calls=_calls_in(tried) + len(test.scores),
    )


def _calls_in(trials: tuple[Trial, ...]) -> int:
    """Model calls spent across every candidate measured."""
    return sum(len(trial.scores) for trial in trials)


def format_report(report: CompileReport) -> str:
    """Human compile card. No technique jargon required to read the winner line."""
    w, t = report.winner, report.test
    lines = [
        f"winner  {w.config.label}",
        (
            f"dev     quality={w.quality:.3f} schema={w.schema_rate:.3f} "
            f"n={len(w.scores)}"
        ),
        (
            f"test    quality={t.quality:.3f} strict={t.strict_quality:.3f} "
            f"schema={t.schema_rate:.3f} n={len(t.scores)}"
        ),
        f"cost    ${t.cost_usd:.5f}/doc  latency={t.latency_ms:.0f}ms/doc",
        (
            f"trials  {report.tried} candidate(s), {report.pruned} pruned at "
            f"the rung, stopped_on_slo={report.stopped_early}"
        ),
        report.cost_to_produce(),
        f"training: {_TRAINING}",
    ]
    if t.schema_rate == 0.0 and w.reasons:
        # A search where every candidate was rejected has no winner to report.
        # Saying so, with the reason, is more useful than a 0.000 row.
        unique = list(dict.fromkeys(w.reasons))
        lines.append(f"blocked: every candidate failed the gate, e.g. {unique[0]}")
    lines.append("arms:")
    lines.extend(
        f"  {trial.config.label:28} dev={trial.quality:.3f} "
        f"schema={trial.schema_rate:.3f} ${trial.cost_usd:.5f} "
        f"{trial.latency_ms:.0f}ms"
        for trial in report.trials
    )
    return "\n".join(lines)
