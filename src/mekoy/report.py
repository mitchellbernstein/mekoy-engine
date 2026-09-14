"""Markdown compile report: winner, Pareto front, and every failed arm.

PLAN §41.11. The plain-text card is for a terminal; this is the artifact that
travels with a downloaded System, so it has to name the tradeoff rather than just
the winner.
"""

from __future__ import annotations

from mekoy.compile import CompileReport
from mekoy.search import Trial, pareto_front

__all__ = ["render_markdown"]


def render_markdown(report: CompileReport, *, task: str = "") -> str:
    """Full report: selection provenance, Pareto front, and all arms."""
    front = pareto_front(tuple(_fully_evaluated(report)))
    lines = [
        "# Compile report",
        "",
        task or "Restaurant call extraction (frozen result, no booking).",
        "",
        "## Winner",
        "",
        f"- harness: `{report.winner.config.label}`",
        f"- dev quality: {report.winner.quality:.3f} (n={len(report.winner.scores)})",
        f"- test quality: {report.test.quality:.3f} (n={len(report.test.scores)})",
        f"- strict test quality: {report.test.strict_quality:.3f}",
        f"- gate failures on test: {_gate_failures(report)}",
        f"- cost: ${report.test.cost_usd:.5f}/doc",
        f"- latency: {report.test.latency_ms:.0f} ms/doc",
        "- training: skipped",
        "",
        "## Selection provenance",
        "",
        f"- {report.describe_selection()}",
        f"- candidates measured: {report.tried}",
        f"- pruned at the minibatch rung: {report.pruned}",
        f"- stopped early on an SLO: {report.stopped_early}",
        "",
        "## Pareto front (quality, cost, latency)",
        "",
        "| harness | dev quality | $/doc | ms/doc |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{t.config.label}` | {t.quality:.3f} | ${t.cost_usd:.5f} | "
        f"{t.latency_ms:.0f} |"
        for t in front
    )
    lines.extend(
        [
            "",
            "## Every arm measured",
            "",
            "| harness | rows | dev quality | schema | $/doc | ms/doc |",
            "|---|---|---|---|---|---|",
        ]
    )
    lines.extend(
        f"| `{t.config.label}` | {len(t.scores)} | {t.quality:.3f} | "
        f"{t.schema_rate:.3f} | ${t.cost_usd:.5f} | {t.latency_ms:.0f} |"
        for t in report.trials
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `quality` normalizes free-text phrasing (PLAN §16.3); `strict` is raw",
            "  string equality and is reported so the difference stays visible.",
            "- Arms with fewer rows than the winner were priced on a minibatch and",
            "  pruned before a full dev pass; they are excluded from the Pareto front.",
        ]
    )
    return "\n".join(lines) + "\n"


def _gate_failures(report: CompileReport) -> int:
    return sum(1 for s in report.test.scores if not s.schema_ok)


def _fully_evaluated(report: CompileReport) -> list[Trial]:
    """Only trials that saw the whole dev slice: the fair comparison set."""
    deepest: dict[int, Trial] = {}
    for trial in report.trials:
        key = id(trial.config)
        if key not in deepest or len(trial.scores) > len(deepest[key].scores):
            deepest[key] = trial
    full = max((len(t.scores) for t in deepest.values()), default=0)
    return [t for t in deepest.values() if len(t.scores) == full]
