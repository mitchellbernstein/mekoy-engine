"""Same-split bake-off: a strong baseline vs the compiled open System.

Both lanes are scored on `test`, which neither selection nor the baseline ever
saw. The verdict is a two-part rule: match or beat quality, and cost
less. Either half alone is not a win.
"""

from __future__ import annotations

from dataclasses import dataclass

from mekoy.compile import CompileReport, SearchSpace, compile_system
from mekoy.dataset import Split
from mekoy.runtime import Completer
from mekoy.search import Budget, HarnessConfig, Trial, evaluate

__all__ = ["Bakeoff", "LaneSpec", "format_bakeoff", "run_bakeoff"]

#: Typical production call: one JSON completion, no shots, no verifier retry.
_BASELINE_CONFIG = HarnessConfig(k_shot=0, retries=0, constrained=True)


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """Who runs a lane and what to call it."""

    name: str
    completer: Completer


@dataclass(frozen=True, slots=True)
class Bakeoff:
    """Baseline vs compiled challenger, both on the untouched test split."""

    baseline: Trial
    challenger: Trial
    baseline_name: str
    challenger_name: str
    report: CompileReport

    @property
    def quality_win(self) -> bool:
        """Challenger matches or beats the baseline on quality."""
        return self.challenger.quality >= self.baseline.quality

    @property
    def cheaper(self) -> bool:
        """Challenger costs less per document. A free baseline breaks the tie."""
        if self.baseline.cost_usd <= 0:
            return self.challenger.cost_usd <= self.baseline.cost_usd
        return self.challenger.cost_usd < self.baseline.cost_usd

    @property
    def verdict(self) -> str:
        """The honest one-liner, including the partial wins."""
        if self.quality_win and self.cheaper:
            return "CHALLENGER WINS (quality >= baseline, cheaper)"
        if self.quality_win:
            return "QUALITY TIE OR WIN, COST NOT LOWER"
        if self.cheaper:
            return "CHEAPER BUT BEHIND ON QUALITY"
        return "BASELINE STILL AHEAD"


def run_bakeoff(
    *,
    split: Split,
    baseline: LaneSpec,
    challenger: LaneSpec,
    space: SearchSpace,
    budget: Budget | None = None,
) -> Bakeoff:
    """Compile the challenger on dev, then score both lanes once on test."""
    report = compile_system(
        challenger.completer,
        split,
        space,
        budget,
    )
    base = evaluate(baseline.completer, split.test, _BASELINE_CONFIG)
    return Bakeoff(
        baseline=base,
        challenger=report.test,
        baseline_name=baseline.name,
        challenger_name=challenger.name,
        report=report,
    )


def format_bakeoff(result: Bakeoff) -> str:
    """Human card. Quality first, then dollars, then latency."""
    b, c = result.baseline, result.challenger
    lines = [
        f"holdout n={len(b.scores)} (test, scored once)",
        (
            f"baseline   {result.baseline_name:22} quality={b.quality:.3f} "
            f"strict={b.strict_quality:.3f} schema={b.schema_rate:.3f} "
            f"${b.cost_usd:.5f}/doc {b.latency_ms:.0f}ms"
        ),
        (
            f"challenger {result.challenger_name:22} quality={c.quality:.3f} "
            f"strict={c.strict_quality:.3f} schema={c.schema_rate:.3f} "
            f"${c.cost_usd:.5f}/doc {c.latency_ms:.0f}ms"
        ),
        (
            f"speed      challenger "
            f"{'faster' if c.latency_ms < b.latency_ms else 'slower'}"
        ),
        f"compile    {result.report.winner.config.label}",
        f"selection  {result.report.describe_selection()}",
        "training: skipped",
        f"verdict: {result.verdict}",
    ]
    return "\n".join(lines)
