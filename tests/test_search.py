"""What the search keeps, and what it must never let a stopwatch decide.

The engine's central claim is that the same spec and the same data compile to the same
System. These tests hold the parts of that claim that are easy to break by accident.
"""

from __future__ import annotations

from mekoy.score import ExampleScore
from mekoy.search import HarnessConfig, Trial, pareto_front, pick


def _scored(quality: float) -> ExampleScore:
    """One scored row. `quality` is the share of fields that matched."""
    hits = round(quality * 4)
    return ExampleScore(
        schema_ok=True,
        field_hits=hits,
        field_total=4,
        line_f1=quality,
        strict_hits=hits,
    )


def _trial(
    k_shot: int, *, latency_ms: float, quality: float = 1.0, retries: int = 0
) -> Trial:
    """A trial whose quality and cost are fixed, so only the tie-break can differ."""
    return Trial(
        config=HarnessConfig(k_shot=k_shot, retries=retries, constrained=True),
        scores=tuple(_scored(quality) for _ in range(4)),
        latency_ms=latency_ms,
        cost_usd=0.0,
    )


def test_a_timing_cannot_decide_the_winner() -> None:
    """The same spec must compile to the same System, whatever the clock said.

    Two candidates that tie on quality and cost is the common case, not a rare one: a
    job the model already handles scores the same at `k=0` and at `k=2`, and both cost
    nothing locally. When a measured latency broke that tie, the winner depended on how
    busy the machine was, and scoring rows concurrently made it worse because concurrent
    requests queue behind each other. The tie-break is now a property of the harness.
    queue behind each other. The tie-break is now a property of the harness.
    """
    fast_second = pick((_trial(0, latency_ms=50.0), _trial(2, latency_ms=900.0)))
    fast_first = pick((_trial(0, latency_ms=900.0), _trial(2, latency_ms=50.0)))
    assert fast_second.config.k_shot == 0
    assert fast_first.config.k_shot == 0


def test_effort_counts_the_calls_a_document_costs() -> None:
    """Effort is what the tie-break reads, so it must be exact and not a timing."""
    plain = HarnessConfig(k_shot=0, retries=0, constrained=True)
    one_retry = HarnessConfig(k_shot=0, retries=1, constrained=True)
    voted = HarnessConfig(k_shot=0, retries=0, constrained=True, consistency=3)
    assert Trial(config=plain, scores=()).effort == 1
    assert Trial(config=one_retry, scores=()).effort == 2
    assert Trial(config=voted, scores=()).effort == 3


def test_effort_multiplies_retries_and_votes_together() -> None:
    """A retry on top of a vote costs both, and the search should see that."""
    both = HarnessConfig(k_shot=0, retries=1, constrained=True, consistency=3)
    assert Trial(config=both, scores=()).effort == 6


def test_pareto_keeps_a_single_trial() -> None:
    """Dropping latency from the tie-break must not collapse the front."""
    only = _trial(0, latency_ms=900.0)
    assert pareto_front((only,)) == (only,)


def test_pareto_keeps_a_cheaper_arm_that_scores_less() -> None:
    """An arm that is worse on quality but does less work is still worth reporting.

    This is the trade the front exists to show: more shots and a repair pass score
    higher, fewer shots cost less. Neither dominates, so the reader sees both.
    """
    accurate = _trial(4, latency_ms=10.0, quality=1.0, retries=1)
    cheap = _trial(0, latency_ms=10.0, quality=0.5)
    front = pareto_front((accurate, cheap))
    assert {t.config.k_shot for t in front} == {0, 4}


def test_pareto_drops_an_arm_that_loses_on_everything() -> None:
    """Nothing about a dominated candidate is worth reporting."""
    better = _trial(0, latency_ms=10.0, quality=1.0)
    worse = Trial(
        config=HarnessConfig(k_shot=4, retries=1, constrained=True),
        scores=tuple(_scored(0.5) for _ in range(4)),
        latency_ms=99.0,
        cost_usd=1.0,
    )
    assert pareto_front((better, worse)) == (better,)
