"""Compare two compiled Systems, or two bake-offs, on the same test split.

PLAN §41.25. Comparison is only meaningful when both sides were measured on the
same examples, so the card states the split it was given and refuses to guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mekoy.bakeoff import Bakeoff
from mekoy.compile import CompileReport
from mekoy.search import Trial

__all__ = [
    "Comparison",
    "compare_bakeoffs",
    "compare_cards",
    "compare_reports",
    "format_comparison",
]

#: A quality gap smaller than this is inside the noise of a small test slice.
_MATERIAL = 0.01


@dataclass(frozen=True, slots=True)
class Comparison:
    """Two candidates on one test split, and the honest verdict."""

    left: str
    right: str
    left_quality: float
    right_quality: float
    left_cost: float
    right_cost: float
    left_latency_ms: float
    right_latency_ms: float
    n_test: int

    @property
    def quality_winner(self) -> str:
        """'left', 'right', or 'tie'."""
        delta = self.left_quality - self.right_quality
        if abs(delta) < _MATERIAL:
            return "tie"
        return "left" if delta > 0 else "right"

    @property
    def verdict(self) -> str:
        """Plain sentence naming the axis that decided it, or reporting a tie."""
        winner = self.quality_winner
        if winner == "tie":
            return f"quality tie within {_MATERIAL:.2f} on n={self.n_test}"
        better = self.left if winner == "left" else self.right
        cheaper = self.left_cost < self.right_cost
        lead = better if cheaper else f"{better} (not cheaper)"
        return f"{lead} leads on quality"

    @property
    def faster(self) -> str:
        """'left', 'right', or 'tie' on measured latency per document."""
        if not (self.left_latency_ms and self.right_latency_ms):
            return "tie"
        if self.left_latency_ms == self.right_latency_ms:
            return "tie"
        return "left" if self.left_latency_ms < self.right_latency_ms else "right"


def compare_reports(
    left_name: str,
    left: CompileReport,
    right_name: str,
    right: CompileReport,
) -> Comparison:
    """Compare two compiles. Both must have been scored on the same test rows."""
    _same_split(left, right)
    return _build(left_name, left.test, right_name, right.test)


def compare_bakeoffs(
    left_name: str, left: Bakeoff, right_name: str, right: Bakeoff
) -> Comparison:
    """Compare the challenger lanes of two bake-offs on the same test rows."""
    if len(left.challenger.scores) != len(right.challenger.scores):
        msg = (
            f"different test sizes: {len(left.challenger.scores)} vs "
            f"{len(right.challenger.scores)}. Scores are not comparable."
        )
        raise ValueError(msg)
    return _build(left_name, left.challenger, right_name, right.challenger)


_COST = re.compile(r"\$([\d.]+)/doc")
_LAT = re.compile(r"([\d.]+)ms")
_N = re.compile(r"n=(\d+)")
_QUALITY = re.compile(r"quality=([\d.]+)")


def compare_cards(
    left_name: str, left_text: str, right_name: str, right_text: str
) -> Comparison:
    """Compare two renderered compile cards by parsing their measured lines.

    Reports are the durable artifact, so comparison reads them instead of
    requiring both compiles to still be in memory.
    """
    left = _parse_card(left_text)
    right = _parse_card(right_text)
    if left["n"] != right["n"]:
        msg = (
            f"different test sizes: {left['n']} vs {right['n']}. "
            "Scores are not comparable."
        )
        raise ValueError(msg)
    return Comparison(
        left=left_name,
        right=right_name,
        left_quality=left["quality"],
        right_quality=right["quality"],
        left_cost=left["cost"],
        right_cost=right["cost"],
        left_latency_ms=left["latency"],
        right_latency_ms=right["latency"],
        n_test=left["n"],
    )


def _parse_card(text: str) -> dict[str, float]:
    """Pull test quality, cost, latency, and n out of a compile card."""
    if "test" not in text:
        msg = "card has no test line; nothing to compare"
        raise ValueError(msg)
    tail = text.split("test", 1)[1]
    quality = _QUALITY.search(tail)
    cost = _COST.search(tail)
    latency = _LAT.search(tail)
    n = _N.search(tail)
    if not (quality and cost and latency and n):
        msg = "could not parse the card; expected quality, $/doc, ms, and n"
        raise ValueError(msg)
    return {
        "quality": float(quality.group(1)),
        "cost": float(cost.group(1)),
        "latency": float(latency.group(1)),
        "n": float(n.group(1)),
    }


def _same_split(left: CompileReport, right: CompileReport) -> None:
    if len(left.test.scores) != len(right.test.scores):
        msg = (
            f"different test sizes: {len(left.test.scores)} vs "
            f"{len(right.test.scores)}. Scores are not comparable."
        )
        raise ValueError(msg)


def _build(left_name: str, left: Trial, right_name: str, right: Trial) -> Comparison:
    return Comparison(
        left=left_name,
        right=right_name,
        left_quality=left.quality,
        right_quality=right.quality,
        left_cost=left.cost_usd,
        right_cost=right.cost_usd,
        left_latency_ms=left.latency_ms,
        right_latency_ms=right.latency_ms,
        n_test=len(left.scores),
    )


def format_comparison(cmp: Comparison) -> str:
    """Card: quality, cost, speed, then the verdict."""
    return "\n".join(
        (
            f"test n={cmp.n_test}",
            (
                f"A {cmp.left:24} quality={cmp.left_quality:.3f} "
                f"${cmp.left_cost:.5f}/doc {cmp.left_latency_ms:.0f}ms"
            ),
            (
                f"B {cmp.right:24} quality={cmp.right_quality:.3f} "
                f"${cmp.right_cost:.5f}/doc {cmp.right_latency_ms:.0f}ms"
            ),
            f"quality: {cmp.quality_winner}",
            f"faster:  {cmp.faster}",
            f"verdict: {cmp.verdict}",
        )
    )
